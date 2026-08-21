from functools import partial
from unittest import TestCase

import torch
from adda.observation.operators import random_sparse_noisy_obs
from adda.probability.distributions import DiagonalGaussian
from adda.system.state import State
from adda.util.initialization import naive_initialization
from adda.variational.strong_constraint_4dvar import sc4dvar_single_window
from adda.variational.weak_constraint_4dvar import wc4dvar_single_window

# import from the diffops/tridiag package
from diffops.fabm.passive_model import PassiveModel
from diffops.gotm.gotm import TracerSimulationOperator
from tensordict import TensorDict


def synthetic_diffusivity(t, z):
    """Returns an approximate diffusivity field nu(z, t) [m^2/s]-like, resembling the pattern shown in the plot.

    Args:
        t (torch.Tensor): 1D tensor of time values [s]
        z (torch.Tensor): 1D tensor of depth values [m] (negative downward)
    Returns:
        torch.Tensor: 2D tensor of shape (len(z), len(t)) with diffusivities
    """
    T, Z = (t, z)

    # Background diffusivity (low)
    nu0 = 1e-6

    def gaussian_blob(t0, z0, sigma_t, sigma_z, amplitude, rho=0.0):
        dt = T - t0
        dz = Z - z0
        exponent = (
            (dt**2) / (sigma_t**2) - 2 * rho * dt * dz / (sigma_t * sigma_z) + (dz**2) / (sigma_z**2)
        ) / 2
        return amplitude * torch.exp(-exponent)

    blob1 = gaussian_blob(t0=5.0e3, z0=-1.2, sigma_t=2.5e3, sigma_z=1.0, amplitude=1e-3)
    blob2 = gaussian_blob(t0=10.0e3, z0=-11.2, sigma_t=3.0e3, sigma_z=1.8, amplitude=8e-3, rho=0.5)

    return nu0 + blob1 + blob2  # shape (len(t), len(z))


def setup_task(
    N=100,
    n_layers=10,
    dt=200.0,
    period_1=44714.0,  # period of 1st harmonic (eg. M2-tide) [s]
    amp_1=0.2,  # amplitude of 1st harmonic [m]
    phase_1=11178.5,  # phase of 1st harmonic [s]
    period_2=43200.0,  # period of 2nd harmonic (eg. S2-tide) [s]
    amp_2=0.04,  # amplitude of 2nd harmonic [m]
    phase_2=10800.0,  # phase of 2nd harmonic [s]
    ddl=torch.tensor(1.0, dtype=torch.float64),
    ddu=torch.tensor(1.0, dtype=torch.float64),
    dtype=torch.float64,
):
    t = torch.linspace(0.0, (N - 1) * dt, N, dtype=dtype)  # time [seconds]

    # the sea surface elevation:
    zeta = amp_1 * torch.sin(2 * torch.pi * (t - phase_1) / period_1) + amp_2 * torch.sin(
        2 * torch.pi * (t - phase_2) / period_2
    )

    # parameters for the parameterization of our coordinates (sigma-coordinates):
    sigma_k = torch.arange(0, n_layers + 1, dtype=dtype) / n_layers - 1
    beta_k = (torch.tanh((ddl + ddu) * (1 + sigma_k) - ddl) + torch.tanh(ddl)) / (torch.tanh(ddl) + torch.tanh(ddu)) - 1

    # the grid coordinates and layer heights:
    zi = zeta[:, None] + beta_k * (15 + zeta[:, None])
    h = torch.diff(zi, axis=1).unsqueeze_(0).unsqueeze_(2)

    # the initial concentrations:
    init_state_1 = torch.full((n_layers,), 1.0, dtype=dtype)

    nu = synthetic_diffusivity(t[:, None].expand((-1, zi.shape[1])), zi).unsqueeze_(0).unsqueeze_(2)

    initial_state = State(TensorDict({"pelagic": init_state_1[None, None, None, :]}, batch_size=(1, 1)))

    model = PassiveModel(torch.tensor(-5.0 / (60**2 * 24), dtype=dtype))
    forward_operator = TracerSimulationOperator(
        bc_up_type_dif="neumann",
        bc_down_type_dif="neumann",
        bc_up_value_dif=0.0,
        bc_down_value_dif=0.0,
        bc_up_type_adv="flux",
        bc_down_type_adv="flux",
        bc_up_value_adv=0.0,
        bc_down_value_adv=0.0,
        scheme_adv="p2_pdm",
        conserve=False,
        implicitness=1.0,
        allow_negative=False,
        model=model,
    )

    ts = forward_operator.integrate(initial_state.fields[:, 0], dt, h, nu)
    ts = State(ts, time_axis=t)

    noise_amplitude = 0.2  # observation noise s.d.
    p_obs = 0.25  # 75% of the variables are masked

    torch.random.manual_seed(42)
    groundtruth, obs_op, observations = random_sparse_noisy_obs(ts, noise_amplitude, p_obs)

    # ensure concentrations are always non-negative
    torch.clamp_min_(observations.state.fields["pelagic"], 0.0)

    # we define the dynamic_inputs State object that is needed for the simulation
    dynamic_inputs = State(TensorDict({"h": h, "nu": nu}, batch_size=h.shape[:2]))

    initialization = naive_initialization(observations)  # state estimate for all time points
    initialization_sc = initialization[:, :1]  # first time point only

    m_dyn = partial(next_step_function, forward_operator)

    return groundtruth, initial_state, obs_op, observations, dynamic_inputs, m_dyn, initialization, initialization_sc


def next_step_function(forward_operator, x: State, dt: float, dynamic_inputs: State, static_inputs: State):
    "this time stepping function expects the input State x to have a single field called 'pelagic'"
    B, T = x.fields.batch_size[:2]
    x_tensor = x.fields.reshape(-1)["pelagic"]  # combine batch and time dimensions, leave others intact
    h_tensor = dynamic_inputs.fields.reshape(-1)["h"]
    nu_tensor = dynamic_inputs.fields.reshape(-1)["nu"]
    if h_tensor.shape[0] > x_tensor.shape[0]:
        h_tensor = h_tensor[1:]
    if nu_tensor.shape[0] > x_tensor.shape[0]:
        nu_tensor = nu_tensor[:-1]
    integrated = forward_operator(x_tensor, dt, h_tensor, nu_tensor)
    integrated = integrated.reshape(B, T, *integrated.shape[1:])  # re-add batch dimension
    new_fields = TensorDict({"pelagic": integrated}, batch_size=(B, T))
    return State(new_fields, time_axis=x.time_axis + dt)  # return State object, advancing time


class Test4DVarGotm(TestCase):
    def test_singlewindow_sc4dvar_gotm(self, verbose=False):
        _, initial_state, obs_op, observations, dynamic_inputs, m_dyn, _, initialization_sc = setup_task()

        assimilated_ic = sc4dvar_single_window(
            m_dyn,
            observations,
            obs_op,
            initialization_sc,
            dynamic_inputs=dynamic_inputs,
            optimizer_pars={"lr": 1.0},
            n_steps=1,  # 5,
            verbose=verbose,
        )

        MSE0 = torch.mean((initialization_sc.fields["pelagic"].squeeze() - initial_state.fields["pelagic"]) ** 2)
        MSE = torch.mean((assimilated_ic.fields["pelagic"].squeeze() - initial_state.fields["pelagic"]) ** 2)

        if verbose:
            print("MSE  =", MSE)
            print("MSE0 =", MSE0)

        self.assertGreater(MSE0, 1.4 * MSE)

    def test_singlewindow_wc4dvar_gotm(self, verbose=False):
        groundtruth, initial_state, obs_op, observations, dynamic_inputs, m_dyn, initialization, _ = setup_task()

        n_steps = groundtruth.time_axis.nelement()
        N = groundtruth.fields["pelagic"].shape[-1]

        model_error_distribs = DiagonalGaussian(
            TensorDict({"pelagic": torch.zeros(1, n_steps - 1, 1, N)}),
            TensorDict({"pelagic": torch.ones(1, n_steps - 1, 1, N)}),
        )

        assimilated_states = wc4dvar_single_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs,
            initialization,
            dynamic_inputs=dynamic_inputs,
            optimizer_pars={"lr": 1.0},
            alpha=1e5,
            n_steps=10,  # 5,
            verbose=verbose,
        )

        MSE0 = torch.mean((initialization.fields["pelagic"].squeeze() - groundtruth.fields["pelagic"]) ** 2)
        MSE = torch.mean((assimilated_states.fields["pelagic"].squeeze() - groundtruth.fields["pelagic"]) ** 2)

        if verbose:
            print("MSE  =", MSE)
            print("MSE0 =", MSE0)

        self.assertGreater(MSE0, 2 * MSE)
