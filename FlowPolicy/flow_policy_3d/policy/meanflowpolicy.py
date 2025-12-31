import sys

sys.path.append('FlowPolicy/flow_policy_3d')
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import copy
import time
import numpy as np
from flow_policy_3d.sde_lib import ConsistencyFM
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.policy.base_policy import BasePolicy
from flow_policy_3d.model.flow.meanflow_conditional_unet1d import MeanflowConditionalUnet1D
from flow_policy_3d.model.flow.mask_generator import LowdimMaskGenerator
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.model_util import print_params
from flow_policy_3d.model.vision.pointnet_extractor import FlowPolicyEncoder
import warnings
from einops import rearrange
from functools import partial

warnings.filterwarnings("ignore")
def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlowPolicy(BasePolicy):
    def __init__(self,
                 shape_meta: dict,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
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
                 Conditional_ConsistencyFM=None,
                 eta=0.01,
                 **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.jvp_api = 'autograd'
        self.w = 0.5
        self.cfg_ratio = 0.1
        self.flow_ratio = 0.5

        # parse shape_meta
        self.time_dist =['lognorm', -2.0, 2.0]
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # point cloud encoder
        obs_encoder = FlowPolicyEncoder(observation_space=obs_dict,
                                        img_crop_shape=crop_shape,
                                        out_channel=encoder_output_dim,
                                        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                        use_pc_color=use_pc_color,
                                        pointnet_type=pointnet_type,
                                        )

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        # obs_as_global_cond=true
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[FlowUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[FlowUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        model = MeanflowConditionalUnet1D(
            input_dim=input_dim,
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

        self.obs_encoder = obs_encoder
        self.model = model

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if Conditional_ConsistencyFM is None:
            Conditional_ConsistencyFM = {
                'eps': 1e-2,
                'num_segments': 2,
                'boundary': 1,
                'delta': 1e-2,
                'alpha': 1e-5,
                'num_inference_step': 1
            }
        self.eta = eta
        self.eps = Conditional_ConsistencyFM['eps']
        self.num_segments = Conditional_ConsistencyFM['num_segments']
        self.boundary = Conditional_ConsistencyFM['boundary']
        self.delta = Conditional_ConsistencyFM['delta']
        self.alpha = Conditional_ConsistencyFM['alpha']
        self.num_inference_step = Conditional_ConsistencyFM['num_inference_step']
        self.jvp_fn = torch.autograd.functional.jvp
        self.create_graph = True

        print_params(self)

    # ========= inference  ============

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # print(f'1 : {nobs_features.shape}')#2,128
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        e = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device,
            generator=None)


        t = torch.ones((B,), device=device, dtype=dtype)  # t = 1
        begin = torch.zeros((B,), device=device, dtype=dtype)  # t = 0

        # import ipdb;
        # ipdb.set_trace()

        z = e - self.model(e, begin, t, local_cond=local_cond, global_cond=global_cond)

        z[cond_mask] = cond_data[cond_mask]  # a1
        # unnormalize prediction
        naction_pred = z[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        # print('evaluation...')
        return result

    # ========= training  ============
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta = self.delta
        alpha = self.alpha
        reduce_op = torch.mean
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size, -1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        # gt & noise


        target = target
        a0 = torch.randn(trajectory.shape, device=trajectory.device)

        t, r = self.sample_t_r(batch_size, trajectory.device)

        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")

        e = torch.randn_like(target)

        z = (1 - t_) * target + t_ * e
        v = e - target


        if global_cond is not None:
            # assert self.cfg_ratio is not None
            # cfg_mask = condition_mask

            if self.w is not None:
                with torch.no_grad():
                    u_t = self.model(z, r, t, local_cond = local_cond, global_cond = global_cond)
                v_hat = self.w * v + (1 - self.w) * u_t
                # if self.cfg_uncond == 'v':
                #     # In the unconditional case, v = w * v + (1 - w) * u,
                #     # so if we're choosing to use 'v' for uncond settings, we can just keep v.
                #     # Apply this only to the unconditional samples indicated by cfg_mask.
                #     cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                #     v_hat = torch.where(cfg_mask, v, v_hat)
            else:
                v_hat = v

            # forward pass
            # u = model(z, t, r, y=c)
        model_partial = partial(self.model, global_cond=global_cond)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        loss_dict = {'bc_loss':
                         loss.item(), }
        # print('training...')
        return loss, loss_dict


