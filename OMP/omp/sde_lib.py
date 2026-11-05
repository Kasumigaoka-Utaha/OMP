"""Sampling parameters shared by the FlowPolicy inference loop."""


class ConsistencyFM:
    """Keep the original FlowPolicy noise schedule and sampling parameters.

    FlowPolicy integrates its velocity model directly in ``predict_action``;
    no separate ODE solver or score-model registry is required.
    """

    def __init__(
        self,
        init_type="gaussian",
        noise_scale=1.0,
        use_ode_sampler="rk45",
        sigma_var=0.0,
        ode_tol=1e-5,
        sample_N=None,
    ):
        if sample_N is not None:
            self.sample_N = sample_N
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.use_ode_sampler = use_ode_sampler
        self.ode_tol = ode_tol
        self.sigma_t = lambda t: (1.0 - t) * sigma_var
