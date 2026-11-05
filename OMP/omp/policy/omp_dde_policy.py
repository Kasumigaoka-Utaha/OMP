"""OMP-DDE: the OMP objective with a central finite-difference derivative."""

import torch

from omp.policy.omp_jvp_policy import OMPJVPPolicy


class OMPDDEPolicy(OMPJVPPolicy):
    """Replace JVP by the Differential Derivation Equation (paper Eq. 14).

    Both perturbed evaluations run without autograd. The unperturbed prediction
    retains the ordinary training graph, including the observation encoder.
    """

    def __init__(self, *args, dde_epsilon=0.005, **kwargs):
        if dde_epsilon <= 0:
            raise ValueError("dde_epsilon must be positive.")
        super().__init__(*args, **kwargs)
        self.dde_epsilon = dde_epsilon

    def mean_velocity_and_derivative(self, z, t, r, velocity, global_cond):
        prediction = self.model(sample=z, timestep=t, r=r, global_cond=global_cond)
        epsilon = self.dde_epsilon
        with torch.no_grad():
            plus = self.model(
                sample=z + epsilon * velocity,
                timestep=t + epsilon,
                r=r,
                global_cond=global_cond,
                training=False,
            )
            minus = self.model(
                sample=z - epsilon * velocity,
                timestep=t - epsilon,
                r=r,
                global_cond=global_cond,
                training=False,
            )
            derivative = (plus - minus) / (2 * epsilon)
        return prediction, derivative
