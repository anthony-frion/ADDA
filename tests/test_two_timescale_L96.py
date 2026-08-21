from functools import partial
from unittest import TestCase

import torch
from adda.observation.operators import random_sparse_noisy_obs
from adda.probability.distributions import DiagonalGaussian
from adda.system.state import State
from adda.util.initialization import interp_initialization
from adda.variational.weak_constraint_4dvar import wc4dvar_single_window
from mdml_sim.lorenz96 import L96Simulator
from tensordict import TensorDict


def setup_task(
    f=10,
    K=36,
    J=10,
    time_step=0.01,
    nb_steps=1000,
    nb_steps_burnt=400,
    noise_amplitude={"x": 1, "y": 0.1},
    p_obs={"x": 0.25, "y": 0.2},
):
    groundtruth = generate_groundtruth(f, K, J, time_step, nb_steps, nb_steps_burnt=nb_steps_burnt)
    groundtruth, obs_op, observations = random_sparse_noisy_obs(groundtruth, noise_amplitude, p_obs)
    T = groundtruth.time_axis.nelement()

    model_error_shape_x = (1, T - 1, K)
    model_error_shape_y = (1, T - 1, K, J)
    model_error_distribs = DiagonalGaussian(
        TensorDict(x=torch.zeros(*model_error_shape_x), y=torch.zeros(*model_error_shape_y)),
        TensorDict(x=torch.ones(*model_error_shape_x), y=torch.ones(*model_error_shape_y) * 5),
    )

    m_dyn = partial(next_step_function, L96Simulator("two_level", forcing=f, b=10, c=10, h=1))

    initialization = interp_initialization(observations, kind="nearest")

    return groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization


def generate_groundtruth(f, K, J, time_step, nb_steps, nb_steps_burnt=400):
    forward_operator = L96Simulator("two_level", forcing=f, b=10, c=10, h=1)
    initial_state_slow = (
        f
        * (
            0.5
            + torch.randn(
                (
                    1,
                    1,
                    K,
                )
            )
        )
        / max(J, 50)
    )
    initial_state_fast = f * (0.5 + torch.randn((1, 1, K, J))) / max(J, 50)
    initial_state = State(TensorDict(x=initial_state_slow, y=initial_state_fast))
    nb_steps_total = nb_steps + nb_steps_burnt
    forecast_steps = torch.arange(0, nb_steps_total) * torch.tensor(time_step, dtype=torch.float64)
    time_series = forward_operator.integrate(
        time=forecast_steps, state=(initial_state.fields["x"].flatten(0, 1), initial_state.fields["y"].flatten(0, 1))
    )
    time_series_td = TensorDict(x=time_series[0][:, nb_steps_burnt:], y=time_series[1][:, nb_steps_burnt:])
    return State(time_series_td, time_axis=forecast_steps[:nb_steps])


def next_step_function(forward_operator, x: State, dt: float, dynamic_inputs: State, static_inputs: State):
    B, T = x.fields.batch_size[:2]
    x_tuple = (
        x.fields.reshape(-1)["x"],
        x.fields.reshape(-1)["y"],
    )  # combine batch and time dimensions, leave others intact
    integrated = forward_operator.integrate(torch.arange(2) * dt, x_tuple)
    new_fields = TensorDict(x=integrated[0][:, 1].unsqueeze(0), y=integrated[1][:, 1].unsqueeze(0), batch_size=(B, T))
    return State(new_fields, time_axis=x.time_axis + dt)  # return State object, advancing time


class TestVariational(TestCase):
    def test_singlewindow(self):
        groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization = setup_task()
        assimilated_series = wc4dvar_single_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs,
            x_init=initialization,
            optimizer_pars={"lr": 5e-2},
            alpha=1e4,
            n_steps=50,
            verbose=False,
        )

        # sum MSE over space and time:
        MSE_x = ((assimilated_series - groundtruth).fields["x"] ** 2).mean()
        MSE_y = ((assimilated_series - groundtruth).fields["y"] ** 2).mean()

        print(f"MSE of assimilated trajectory for slow variables: {MSE_x}")
        print(f"MSE of assimilated trajectory for fast variables: {MSE_y}")
        self.assertTrue(
            MSE_x < 0.05, msg="The performance is worse than usual observed results for the slow variables."
        )
        self.assertTrue(
            MSE_y < 0.05, msg="The performance is worse than usual observed results for the fast variables."
        )
