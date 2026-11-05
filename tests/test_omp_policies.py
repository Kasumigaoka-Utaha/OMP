"""Numerical and gradient checks for both released OMP variants.

Run with ``python -m unittest discover -s tests -v`` after installing the
training requirements. On machines without PyTorch these checks are skipped.
"""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "OMP"))
HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    from omegaconf import OmegaConf
    from omp.model.common.normalizer import LinearNormalizer
    from omp.policy.omp_dde_policy import OMPDDEPolicy
    from omp.policy.omp_jvp_policy import OMPJVPPolicy

    class AffineMeanVelocity(torch.nn.Module):
        """A known derivative: du/dt = 2 * velocity + 3, with fixed r."""

        def forward(self, sample, timestep, r, global_cond, training=True):
            value = (
                2 * sample + 3 * timestep[:, None, None]
                + 5 * r[:, None, None] + global_cond[:, :1, None]
            )
            if training:
                return value, value.flatten(1).unsqueeze(0)
            return value


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class OMPPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def bare_policy(self, policy_type):
        policy = policy_type.__new__(policy_type)
        torch.nn.Module.__init__(policy)
        policy.model = AffineMeanVelocity()
        policy.epsilon_dir = 1e-6
        policy.dde_epsilon = 0.005
        return policy

    def make_small_policy(self, policy_type, **overrides):
        options = dict(
            shape_meta={
                "action": {"shape": [2]},
                "obs": {
                    "point_cloud": {"shape": [8, 3]},
                    "agent_pos": {"shape": [4]},
                },
            },
            horizon=4,
            n_obs_steps=2,
            n_action_steps=3,
            diffusion_step_embed_dim=8,
            down_dims=(8, 16, 32),
            kernel_size=3,
            n_groups=4,
            encoder_output_dim=8,
            pointnet_type="mlp",
            pointcloud_encoder_cfg=OmegaConf.create({
                "in_channels": 3,
                "out_channels": 8,
                "use_layernorm": True,
                "final_norm": "layernorm",
                "normal_channel": False,
            }),
        )
        options.update(overrides)
        policy = policy_type(**options)
        batch = {
            "obs": {
                "point_cloud": torch.randn(3, 4, 8, 3),
                "agent_pos": torch.randn(3, 4, 4),
            },
            "action": torch.randn(3, 4, 2),
        }
        normalizer = LinearNormalizer()
        normalizer.fit({**batch["obs"], "action": batch["action"]})
        policy.set_normalizer(normalizer)
        return policy, batch

    def test_jvp_and_dde_match_known_total_derivative(self):
        torch.manual_seed(0)
        z = torch.randn(3, 4, 2, dtype=torch.float64)
        velocity = torch.randn_like(z)
        # Endpoints also check that the central difference holds r fixed.
        t = torch.tensor([0., 0.5, 1.], dtype=torch.float64)
        r = torch.tensor([0., 0.2, 0.7], dtype=torch.float64)
        context = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        expected = 2 * velocity + 3
        for policy_type in (OMPJVPPolicy, OMPDDEPolicy):
            with self.subTest(variant=policy_type.__name__):
                policy = self.bare_policy(policy_type)
                prediction, derivative = policy.mean_velocity_and_derivative(z, t, r, velocity, context)
                torch.testing.assert_close(derivative, expected)
                self.assertTrue(prediction[0].requires_grad)
                if policy_type is OMPDDEPolicy:
                    self.assertFalse(derivative.requires_grad)

    def test_derivative_prediction_preserves_context_gradient(self):
        for policy_type in (OMPJVPPolicy, OMPDDEPolicy):
            with self.subTest(variant=policy_type.__name__):
                policy = self.bare_policy(policy_type)
                z = torch.zeros(2, 4, 2)
                t = torch.ones(2)
                r = torch.zeros(2)
                context = torch.ones(2, 1, requires_grad=True)
                prediction, _ = policy.mean_velocity_and_derivative(z, t, r, z, context)
                prediction[0].sum().backward()
                torch.testing.assert_close(context.grad, torch.full_like(context, 8))

    def test_losses_are_finite_for_zero_and_antiparallel_vectors(self):
        policy = self.bare_policy(OMPJVPPolicy)
        for target, prediction in (
            (torch.zeros(2, 4, 2), torch.zeros(2, 4, 2)),
            (torch.ones(2, 4, 2), -torch.ones(2, 4, 2)),
        ):
            prediction.requires_grad_()
            loss = policy.directional_alignment_loss(target, prediction)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
        features = torch.zeros(1, 8, requires_grad=True)
        loss = policy.dispersive_loss(features)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_small_policies_backpropagate_and_predict_once(self):
        for policy_type in (OMPJVPPolicy, OMPDDEPolicy):
            with self.subTest(variant=policy_type.__name__):
                torch.manual_seed(0)
                policy, batch = self.make_small_policy(policy_type)
                loss, metrics = policy.compute_loss(batch)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(metrics["directional_loss"], 0)
                loss.backward()
                for module in (policy.model, policy.obs_encoder.extractor):
                    gradients = [param.grad for param in module.parameters() if param.grad is not None]
                    self.assertTrue(gradients, "Missing model/point-encoder gradients")
                    self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                    self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0)
                policy.train()
                with patch.object(policy.model, "forward", wraps=policy.model.forward) as forward:
                    result = policy.predict_action(batch["obs"])
                self.assertEqual(forward.call_count, 1)
                self.assertEqual(result["action"].shape, (3, 3, 2))
                self.assertEqual(result["action_pred"].shape, (3, 4, 2))
                self.assertTrue(torch.isfinite(result["action"]).all())
                self.assertFalse(result["action"].requires_grad)
                self.assertTrue(policy.model.training)

    def test_invalid_dde_step_and_execution_horizon_are_rejected(self):
        with self.assertRaises(ValueError):
            OMPDDEPolicy(dde_epsilon=0)
        with self.assertRaises(ValueError):
            self.make_small_policy(OMPJVPPolicy, n_action_steps=4)


if __name__ == "__main__":
    unittest.main()
