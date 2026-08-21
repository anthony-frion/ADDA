from functools import partial
from unittest import TestCase

import numpy as np
import torch
from adda.observation.operators import MaskedIidGaussianTimeInterpolatedObsOp, random_sparse_noisy_obs
from adda.probability.distributions import DiagonalGaussian
from adda.system.state import State
from adda.util.initialization import naive_initialization
from adda.variational.base import assemble_analysis
from adda.variational.weak_constraint_4dvar import wc4dvar_single_window, wc4dvar_sliding_window
from mdml_sim.lorenz96 import L96Simulator
from tensordict import TensorDict


def setup_task(
    n_variables=40,
    time_step=0.01,
    nb_steps=800,
    nb_steps_burnt=400,
    noise_amplitude=1.0,
    p_obs=0.25,
    forcing=8.0,
):
    groundtruth = generate_groundtruth(
        n_variables, time_step, nb_steps, nb_steps_burnt=nb_steps_burnt, forward_operator=L96Simulator(forcing=forcing)
    )
    groundtruth, obs_op, observations = random_sparse_noisy_obs(groundtruth, noise_amplitude, p_obs)
    T = groundtruth.time_axis.nelement()

    model_error_shape = (1, T - 1, n_variables)
    model_error_distribs = DiagonalGaussian(
        torch.zeros(*model_error_shape),
        torch.ones(*model_error_shape),
    )

    m_dyn = partial(next_step_function, L96Simulator(forcing=forcing))

    initialization = naive_initialization(observations)

    return groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization


def generate_groundtruth(
    n_variables, time_step, nb_steps, nb_steps_burnt=400, forward_operator=L96Simulator(forcing=8)
):
    initial_state = torch.randn(1, n_variables)
    nb_steps_total = nb_steps + nb_steps_burnt
    forecast_steps = torch.arange(0, nb_steps_total) * torch.tensor(time_step, dtype=torch.float64)
    time_series = forward_operator.integrate(time=forecast_steps, state=initial_state).squeeze()
    time_series = time_series[nb_steps_burnt:]
    return State(time_series.reshape(1, *time_series.shape), time_axis=forecast_steps[:nb_steps])


def next_step_function(forward_operator, x: State, dt: float, dynamic_inputs: State, static_inputs: State):
    B, T = x.fields.batch_size[:2]
    x_tensor = x.fields.reshape(-1)["x"]  # combine batch and time dimensions, leave others intact
    integrated = forward_operator.integrate(torch.arange(2) * dt, x_tensor)[:, 1]
    integrated = integrated.unsqueeze(0)  # re-add batch dimension
    new_fields = TensorDict(x=integrated, batch_size=(1, T))
    return State(new_fields, time_axis=x.time_axis + dt)


class TestVariational(TestCase):
    def test_singlewindow(self):
        groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization = setup_task()
        assimilated_series = wc4dvar_single_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs,
            x_init=initialization,
            optimizer_pars={"lr": 1},
            alpha=1e4,
            n_steps=50,
            verbose=False,
        )

        # sum MSE over space and time:
        MSE = ((assimilated_series - groundtruth).fields["x"] ** 2).mean()

        print(f"MSE of assimilated trajectory: {MSE}")
        self.assertTrue(MSE < 0.02, msg="The performance is worse than usual observed results.")

    def test_slidingwindow(
        self, shift_fraction=0.4, discard_partial=False, window_duration_steps=200, nb_steps=800, nb_steps_burnt=400
    ):
        groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization = setup_task(
            nb_steps=nb_steps, nb_steps_burnt=nb_steps_burnt
        )

        window_duration = window_duration_steps * (groundtruth.time_axis[1] - groundtruth.time_axis[0])

        x_eachwin, windows, idx_ranges = wc4dvar_sliding_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs=model_error_distribs,
            x_init=initialization,
            window_duration=window_duration,
            window_shift=shift_fraction,
            discard_partial=discard_partial,
            background_prior=None,
            optimizer_pars={"lr": 1},
            alpha=1e4,
            n_steps=1,  # 50
            verbose=False,
        )

        is_weak = True

        analysis = assemble_analysis(
            x_eachwin,
            idx_ranges,
            obs_op.time_axis,
            is_weak,
            m_dyn=m_dyn,
        )

        # MSE = ((analysis - groundtruth).fields["x"] ** 2).mean()
        # print(f"Sliding-window 4D-Var first test: MSE of assimilated trajectory={MSE}")

    def test_slidingwindow_integer_duration_and_shift(
        self, window_shift=100, discard_partial=False, window_duration_steps=200, nb_steps=800, nb_steps_burnt=400
    ):
        assert window_duration_steps <= nb_steps, "incorrect arguments for this test"
        assert window_shift <= nb_steps, "incorrect arguments for this test"
        groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization = setup_task(
            nb_steps=nb_steps, nb_steps_burnt=nb_steps_burnt
        )

        x_eachwin, windows, idx_ranges = wc4dvar_sliding_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs=model_error_distribs,
            x_init=initialization,
            window_duration=window_duration_steps,
            window_shift=window_shift,
            discard_partial=discard_partial,
            background_prior=None,
            optimizer_pars={"lr": 1},
            alpha=1e4,
            n_steps=1,
            verbose=False,
        )
        assert int(idx_ranges[0][1] - idx_ranges[0][0]) == window_duration_steps, "incorrect size for first window"
        assert int(idx_ranges[1][0] - idx_ranges[0][0]) == window_shift, "incorrect shift between first 2 windows"

        is_weak = True

        analysis = assemble_analysis(
            x_eachwin,
            idx_ranges,
            obs_op.time_axis,
            is_weak,
            m_dyn=m_dyn,
        )

        ((analysis - groundtruth).fields["x"] ** 2).mean()

    def test_slidingwindow_integer_duration_float_shift(
        self, window_shift=0.5, discard_partial=False, window_duration_steps=200, nb_steps=800, nb_steps_burnt=400
    ):
        assert window_duration_steps <= nb_steps, "incorrect arguments for this test"
        groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization = setup_task(
            nb_steps=nb_steps, nb_steps_burnt=nb_steps_burnt
        )

        x_eachwin, windows, idx_ranges = wc4dvar_sliding_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs=model_error_distribs,
            x_init=initialization,
            window_duration=window_duration_steps,
            window_shift=window_shift,
            discard_partial=discard_partial,
            background_prior=None,
            optimizer_pars={"lr": 1},
            alpha=1e4,
            n_steps=1,
            verbose=False,
        )
        assert int(idx_ranges[0][1] - idx_ranges[0][0]) == window_duration_steps, "incorrect size for first window"
        assert int(idx_ranges[1][0] - idx_ranges[0][0]) == int(
            window_shift * window_duration_steps
        ), "incorrect shift between first 2 windows"

        is_weak = True

        analysis = assemble_analysis(
            x_eachwin,
            idx_ranges,
            obs_op.time_axis,
            is_weak,
            m_dyn=m_dyn,
        )

        ((analysis - groundtruth).fields["x"] ** 2).mean()

    def test_slidingwindow_irregular_obs_time(
        self,
        shift_fraction=0.4,
        discard_partial=False,
        window_duration_steps=200,
        nb_steps=800,
        nb_steps_burnt=400,
        n_variables=40,
        dt=0.01,
        noise_amplitude=0.1,
        p_obs=0.25,
        forcing=8.0,
    ):
        nb_steps_total = nb_steps + nb_steps_burnt
        initial_state = State(torch.randn(1, 1, n_variables))
        forecast_steps = torch.rand(nb_steps_total) * nb_steps_total * dt
        forecast_steps = torch.sort(forecast_steps).values
        forecast_steps = torch.unique_consecutive(forecast_steps)  # remove potential duplicated values
        forward_operator = L96Simulator(forcing=8.0)
        m_dyn = partial(next_step_function, forward_operator)

        ts = forward_operator.integrate(time=forecast_steps, state=initial_state.fields["x"].reshape(1, -1)).squeeze()
        true_ts = ts[nb_steps_burnt:]
        true_ts = State(
            true_ts.reshape(1, *true_ts.shape),
            time_axis=forecast_steps[nb_steps_burnt:] - forecast_steps[nb_steps_burnt],
        )

        groundtruth, obs_op, observations = random_sparse_noisy_obs(true_ts, noise_amplitude, p_obs)
        initialization = naive_initialization(observations)
        obs_op = MaskedIidGaussianTimeInterpolatedObsOp(obs_op.mask, obs_op.sigma, obs_op.sample_points)

        T = groundtruth.time_axis.nelement()
        model_error_shape = (1, T - 1, n_variables)
        model_error_distribs = DiagonalGaussian(
            torch.zeros(*model_error_shape),
            torch.ones(*model_error_shape),
        )

        window_duration = window_duration_steps * dt

        state_time_axis = torch.arange(0, (nb_steps - 1 + 0.01) * dt, dt, dtype=torch.float64)  # regular time sampling
        interpolated_init_tensor = torch.zeros((1, len(state_time_axis), n_variables))
        for i in range(n_variables):
            interpolated_init_tensor[0, :, i] = torch.Tensor(
                np.interp(state_time_axis, initialization.time_axis, initialization.fields["x"][0, :, i])
            )
        initialization = State(
            TensorDict(x=interpolated_init_tensor, batch_size=[1, len(state_time_axis)]), time_axis=state_time_axis
        )  # interpolated init on a regular time axis

        x_eachwin, windows, idx_ranges = wc4dvar_sliding_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs=model_error_distribs,
            x_init=initialization,
            state_time_axis=state_time_axis,
            window_duration=window_duration,
            window_shift=shift_fraction,
            discard_partial=discard_partial,
            background_prior=None,
            optimizer_pars={"lr": 1},
            alpha=1e4,
            n_steps=1,
            verbose=False,
        )

        is_weak = True

        analysis = assemble_analysis(
            x_eachwin,
            idx_ranges,
            state_time_axis,
            is_weak,
            m_dyn=m_dyn,
        )
