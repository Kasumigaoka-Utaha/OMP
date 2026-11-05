"""OMP-JVP: one-step MeanFlow with directional alignment (paper Eqs. 12-13)."""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from omp.common.pytorch_util import dict_apply
from omp.model.common.normalizer import LinearNormalizer
from omp.model.mean.conditional_unet1d_meanflow_dis import ConditionalUnet1D
from omp.model.vision.pointnet_extractor import OMPEncoder
from omp.policy.base_policy import BasePolicy


class OMPJVPPolicy(BasePolicy):
    """Learn interval mean velocity using the analytical total derivative.

    ``guidance_scale=1`` follows the straight interpolation path in the paper.
    A value of 2 retains the velocity extrapolation used in earlier experiments.
    Both derivative implementations use the same chosen path and objective.
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=1,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="mlp",
        pointcloud_encoder_cfg=None,
        guidance_scale=1.0,
        directional_weight=1.0,
        dispersive_weight=0.5,
        epsilon_dir=1e-6,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("OMP conditions on observation features via obs_as_global_cond=True.")
        if num_inference_steps not in (None, 1):
            raise ValueError("OMP generates actions in one network evaluation.")
        if not 1 <= n_obs_steps <= horizon:
            raise ValueError("n_obs_steps must be between 1 and horizon.")
        if not 1 <= n_action_steps <= horizon - n_obs_steps + 1:
            raise ValueError("The execution horizon must fit after the observation history.")
        if epsilon_dir <= 0 or directional_weight < 0 or dispersive_weight < 0:
            raise ValueError("Loss weights must be nonnegative and epsilon_dir must be positive.")

        self.action_shape = shape_meta["action"]["shape"]
        if len(self.action_shape) not in (1, 2):
            raise NotImplementedError(f"Unsupported action shape {self.action_shape}")
        self.action_dim = int(np.prod(self.action_shape))
        obs_shapes = {key: meta["shape"] for key, meta in shape_meta["obs"].items()}
        self.obs_encoder = OMPEncoder(
            observation_space=obs_shapes,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        self.obs_feature_dim = self.obs_encoder.output_shape()
        global_cond_dim = self.obs_feature_dim
        if "cross_attention" not in condition_type:
            global_cond_dim *= n_obs_steps
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.condition_type = condition_type
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.obs_as_global_cond = obs_as_global_cond
        self.num_inference_steps = 1
        self.guidance_scale = guidance_scale
        self.directional_weight = directional_weight
        self.dispersive_weight = dispersive_weight
        self.epsilon_dir = epsilon_dir
        self.flow_ratio = 0.5
        self.time_dist = ("lognorm", -0.4, 1.0)
        self.kwargs = kwargs

    def _encode_observations(self, observations):
        if not self.use_pc_color:
            observations = dict(observations)
            observations["point_cloud"] = observations["point_cloud"][..., :3]
        batch_size = next(iter(observations.values())).shape[0]
        encoder_inputs = dict_apply(
            observations,
            lambda value: value[:, :self.n_obs_steps].reshape(-1, *value.shape[2:]),
        )
        features = self.obs_encoder(encoder_inputs)
        if "cross_attention" in self.condition_type:
            return features.reshape(batch_size, self.n_obs_steps, -1)
        return features.reshape(batch_size, -1)

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate an action chunk in one evaluation, z_0 = z_1 - u(z_1,0,1|c)."""
        global_cond = self._encode_observations(self.normalizer.normalize(obs_dict))
        batch_size = global_cond.shape[0]
        noise = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            device=global_cond.device,
            dtype=global_cond.dtype,
        )
        t = noise.new_ones(batch_size)
        r = noise.new_zeros(batch_size)
        velocity = self.model(
            sample=noise, timestep=t, r=r, global_cond=global_cond, training=False,
        )
        action_pred = self.normalizer["action"].unnormalize(noise - velocity)
        start = self.n_obs_steps - 1
        return {
            "action": action_pred[:, start:start + self.n_action_steps],
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def mean_velocity_and_derivative(self, z, t, r, velocity, global_cond):
        """Return (mean velocity, features) and du/dt with r held fixed.

        The observation context stays in the closure so reconstruction and feature
        losses retain gradients to the point-cloud encoder.
        """
        prediction, derivative = torch.autograd.functional.jvp(
            lambda sample, timestep, start: self.model(
                sample=sample, timestep=timestep, r=start, global_cond=global_cond,
            ),
            (z, t, r),
            (velocity, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True,
        )
        return prediction, derivative[0]

    def compute_loss(self, batch):
        global_cond = self._encode_observations(self.normalizer.normalize(batch["obs"]))
        actions = self.normalizer["action"].normalize(batch["action"])
        t, r = self.sample_t_r(actions.shape[0], actions.device)
        t_view, r_view = t[:, None, None], r[:, None, None]
        noise = torch.randn_like(actions)
        z = (1 - t_view) * actions + t_view * noise
        # v_0 is the true noise-to-action displacement used by Eq. 12.
        v_0 = noise - actions
        velocity = v_0
        if self.guidance_scale != 1:
            with torch.no_grad():
                instantaneous = self.model(
                    sample=z, timestep=t, r=t, global_cond=global_cond, training=False,
                )
            velocity = self.guidance_scale * v_0 + (1 - self.guidance_scale) * instantaneous

        prediction, derivative = self.mean_velocity_and_derivative(z, t, r, velocity, global_cond)
        mean_velocity, features = prediction
        target = (velocity - (t_view - r_view) * derivative).detach()
        error = mean_velocity - target
        meanflow_loss = adaptive_l2_loss(error)
        directional_loss = self.directional_alignment_loss(v_0, mean_velocity)
        dispersive_loss = mean_velocity.new_zeros(())
        if self.dispersive_weight:
            for layer_features in features:
                dispersive_loss = dispersive_loss + self.dispersive_loss(layer_features)
        loss = (
            meanflow_loss
            + self.dispersive_weight * dispersive_loss
            + self.directional_weight * directional_loss
        )
        return loss, {
            "bc_loss": loss.item(),
            "mse_val": error.detach().square().mean().item(),
            "meanflow_loss": meanflow_loss.item(),
            "directional_loss": directional_loss.item(),
            "dis_loss": dispersive_loss.item(),
        }

    def directional_alignment_loss(self, target, prediction):
        """Paper Eq. 12, with finite values for zero and antiparallel vectors."""
        target = target.flatten(1)
        prediction = prediction.flatten(1)
        cosine = F.cosine_similarity(target, prediction, dim=1, eps=self.epsilon_dir)
        probability = ((cosine.clamp(-1, 1) + 1) / 2).clamp_min(self.epsilon_dir)
        return -probability.log().mean()

    def dispersive_loss(self, features, tau=1.0):
        distances = torch.cdist(features, features, p=2).square()
        distances = distances / distances.max().clamp_min(self.epsilon_dir)
        return torch.exp(-distances / tau).mean().clamp_min(self.epsilon_dir).log()

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == "uniform":
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == "lognorm":
            mu, sigma = self.time_dist[1:]
            logits = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-logits))
        else:
            raise ValueError(f"Unsupported time distribution: {self.time_dist[0]}")
        t = np.maximum(samples[:, 0], samples[:, 1])
        r = np.minimum(samples[:, 0], samples[:, 1])
        selected = np.random.permutation(batch_size)[:int(self.flow_ratio * batch_size)]
        r[selected] = t[selected]
        return torch.as_tensor(t, device=device), torch.as_tensor(r, device=device)


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """MeanFlow's adaptive L2 reconstruction term with stopped sample weights."""
    squared_error = error.square().mean(dim=tuple(range(1, error.ndim)))
    weights = (squared_error + c).pow(gamma - 1).detach()
    return (weights * squared_error).mean()
