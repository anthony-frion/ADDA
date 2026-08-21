import torch
from NLADV.time_stepping import TimeIntegrator


class NonlinAdv1D(torch.nn.Module):
    """1D nonlinear advection solver using spectral methods and a user-specified time integrator.

    Source Materials:
    https://github.com/Ceyron/exponax
    https://doi.org/10.1006/jcph.2002.6995
    """

    def __init__(
        self,
        domain_size,
        num_spat,
        dt,
        linear_coeffs,
        advect_coeff,
        integrator: TimeIntegrator,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.domain_size = domain_size
        self.num_spat = num_spat
        self.dt = dt
        self.dx = domain_size / num_spat
        self.coeffs = linear_coeffs
        self.advect_coeff = advect_coeff
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        wavenum = torch.fft.rfftfreq(num_spat, d=domain_size / num_spat, device=self.device, dtype=dtype)
        alias_mask = wavenum < 2 / 3 * torch.max(wavenum)
        spat_der = 1j * 2 * torch.pi * wavenum
        lin_op = sum(c * (spat_der**k) for k, c in enumerate(linear_coeffs))
        self.register_buffer("wavenum", wavenum)
        self.register_buffer("alias_mask", alias_mask)
        self.register_buffer("spat_der", spat_der)
        self.register_buffer("lin_op", lin_op)
        self.integrator = integrator(self.dt, lin_op, self.num_spat)

    def nonlinear_hat(self, u=None, u_hat=None):
        """Compute the nonlinear term in Real space and transform it to Fourier space."""
        if u is None:
            u = torch.fft.irfft(u_hat, n=self.num_spat)
        u2 = -(self.advect_coeff / 2.0) * u**2
        u2_hat = torch.fft.rfft(u2)
        u2_hat *= self.alias_mask
        return self.spat_der * u2_hat

    def forward(self, u):
        """Perform a single time step using the specified time integrator."""
        u_hat = torch.fft.rfft(u)
        u_next_hat = self.integrator.step(u, u_hat, self.nonlinear_hat)
        u_next = torch.fft.irfft(u_next_hat, n=self.num_spat)
        return u_next

    def integrate(self, time, state):
        """Necessary to fit the convention of adda.

        Integrate the state forward in time and return the full trajectory.
        """
        B, S = state.shape
        rollout = []
        for _ in range(len(time)):
            state = self.forward(state)
            rollout.append(state)
        rollout = torch.stack(rollout, dim=1)
        return rollout
