from abc import ABC, abstractmethod

import torch


class TimeIntegrator(torch.nn.Module, ABC):
    """Abstract base class for time integrators."""

    def __init__(self, dt):
        super().__init__()
        self.dt = dt

    @abstractmethod
    def _precompute(self):
        pass

    @abstractmethod
    def step(self, *args):
        pass


class ETD(TimeIntegrator):
    """Exponential Time Differencing (ETD) integrator."""

    def __init__(self, dt, lin_op):
        super().__init__(dt)
        self.register_buffer("lin_op", lin_op)
        self._precompute()

    def _precompute(self):
        """Precompute the ETD coefficients."""
        exp_term = torch.exp(self.dt * self.lin_op)
        c = torch.where(
            self.lin_op == 0,
            torch.tensor(self.dt, device=self.lin_op.device, dtype=self.lin_op.dtype),
            (exp_term - 1) / self.lin_op,
        )

        self.register_buffer("exp_term", exp_term)
        self.register_buffer("c", c)

    def step(self, u_hat, u_nl_hat):
        """Perform a single ETD time step."""
        return self.exp_term * u_hat + self.c * u_nl_hat


class ETDRK2(TimeIntegrator):
    """Exponential Time Differencing Runge-Kutta 2nd order (ETDRK2) integrator."""

    def __init__(self, dt, lin_op, num_spat):
        super().__init__(dt)
        self.num_spat = num_spat
        self.register_buffer("lin_op", lin_op)
        self._precompute()

    def _precompute(self):
        """Precompute the ETDRK2 coefficients."""
        exp_term = torch.exp(self.dt * self.lin_op)

        c1 = torch.where(
            self.lin_op == 0,
            torch.tensor(self.dt, device=self.lin_op.device, dtype=self.lin_op.dtype),
            (exp_term - 1) / self.lin_op,
        )

        c2 = torch.where(
            self.lin_op == 0,
            torch.tensor(
                self.dt / 2,
                device=self.lin_op.device,
                dtype=self.lin_op.dtype,
            ),
            (exp_term - 1 - self.dt * self.lin_op) / (self.lin_op**2 * self.dt),
        )

        self.register_buffer("exp_term", exp_term)
        self.register_buffer("c1", c1)
        self.register_buffer("c2", c2)

    def step(self, u, u_hat, nl_hat):
        """Perform a single ETDRK2 time step."""
        u_nl_hat = nl_hat(u=u)
        a_n = self.exp_term * u_hat + self.c1 * u_nl_hat
        u_nl_hat_1 = nl_hat(u_hat=a_n)
        return a_n + (u_nl_hat_1 - u_nl_hat) * self.c2


class CrankNicolson(TimeIntegrator):
    """Crank-Nicolson time integrator."""

    def __init__(self, dt, lin_op, num_spat):
        super().__init__(dt)
        self.num_spat = num_spat
        self.register_buffer("lin_op", lin_op)
        self._precompute()

    def _precompute(self):
        """Precompute the Crank-Nicolson coefficients."""
        self.register_buffer("num_lin", 1.0 + 0.5 * self.dt * self.lin_op)
        self.register_buffer("den_lin", 1.0 - 0.5 * self.dt * self.lin_op)

    def step(self, u, u_hat, nl_hat):
        """Perform a single Crank-Nicolson time step."""
        N_hat = nl_hat(u=u)
        return (self.num_lin * u_hat + self.dt * N_hat) / self.den_lin


class ImplicitEuler(TimeIntegrator):
    """Implicit Euler time integrator."""

    def __init__(self, dt, lin_op):
        super().__init__(dt)
        self.register_buffer("lin_op", lin_op)
        self._precompute()

    def _precompute(
        self,
    ):
        """Precompute the Implicit Euler coefficients."""
        self.register_buffer("num_lin", self.dt)
        self.register_buffer("den_lin", 1.0 - self.dt * self.lin_op)

    def step(self, u, u_hat, nl_hat):
        """Perform a single Implicit Euler time step."""
        N_hat = nl_hat(u=u)
        u_next_hat = (self.num_lin * u_hat + self.dt * N_hat) / self.den_lin

        return u_next_hat


class AB2AM2(TimeIntegrator):
    """Adams-Bashforth 2nd order / Adams-Moulton 2nd order (AB2AM2) integrator."""

    def __init__(self, dt, lin_op):
        super().__init__(dt)
        self.register_buffer("lin_op", lin_op)
        self._precompute()

    def _precompute(self, *args):
        """Precompute the AB2AM2 coefficients."""
        self.register_buffer("num_lin", 1.0 + 0.5 * self.dt * self.lin_op)
        self.register_buffer("den_lin", 1.0 - 0.5 * self.dt * self.lin_op)

    def step(self, u, u_hat, u_hat_prev, nl_hat):
        """Perform a single AB2AM2 time step."""
        N_hat = nl_hat(u=u)
        N_hat_prev = nl_hat(u_hat=u_hat_prev)
        u_next_hat = (
            self.num_lin * u_hat + (3.0 / 2.0) * self.dt * N_hat - (1.0 / 2.0) * self.dt * N_hat_prev
        ) / self.den_lin
        return u_next_hat
