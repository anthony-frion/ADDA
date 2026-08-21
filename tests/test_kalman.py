from unittest import TestCase

import numpy as np
import pykalman
import torch
from adda.kalman.filter import kalman_filter
from adda.kalman.smoother import fixed_lag_kalman_smoother, kalman_smoother
from adda.observation.operators import LinearGaussianObsOp
from adda.probability.distributions import DiagonalGaussian
from adda.system.dynamics import LinearDynamics
from adda.system.state import State
from adda.util.state_space import rollout
from tensordict import TensorDict


def generate_testdata(
    T=100, obs_noise_sd=0.5, noise_init_sd=0.2, use_tensordict=False, model_error_covariance=None, diag_obs_error=True
):
    """Generate test data for Kalman algorithms."""
    ndims = 3
    mu_init, sigma_init = torch.ones(1, 1, ndims), torch.ones(1, 1, ndims) * noise_init_sd
    if use_tensordict:
        mu_init, sigma_init = (
            TensorDict(x=mu_init, batch_size=(1, 1)),
            TensorDict(x=sigma_init, batch_size=(1, 1)),
        )
    prior = DiagonalGaussian(mu_init, sigma_init)
    if not use_tensordict:
        prior.mean.batch_size, prior.sigma.batch_size = (1, 1), (1, 1)

    x0_gt = State(prior.sample())  # ground truth IC
    # create an almost antisymmetric matrix
    generator_mat = torch.Tensor([[0, 2, 1], [-2, 0, 2], [-1, -2, 0]]) * 0.1 + torch.randn(3, 3) * 0.01
    # our discrete linear system corresponds to the time integration of the almost antisymmetric matrix
    m_dyn = LinearDynamics(torch.matrix_exp(generator_mat), model_error_covariance)
    dt = 0.01

    x_all_gt = rollout(m_dyn, torch.arange(0, T * dt, dt), x0_gt)  # ground truth rollout

    H = State(torch.Tensor([[[[1, 1, 1], [1, -1, 1]]]]))
    if diag_obs_error:
        P_chol = State(torch.eye(2).reshape(1, 1, 2, 2) / obs_noise_sd)
    else:
        P_chol = State(torch.Tensor([[1, 1], [1, 2]]).reshape(1, 1, 2, 2) / obs_noise_sd)
    obs_op = LinearGaussianObsOp(H, P_chol)
    observations = obs_op.sample(x_all_gt)

    mu_init = State(mu_init)
    sigma_init = State(torch.eye(ndims).reshape(1, 1, ndims, ndims) * noise_init_sd**2)

    return m_dyn, x_all_gt, observations, obs_op, mu_init, sigma_init


def solve_with_pykalman(m_dyn, observations, obs_op, mu_init, sigma_init, model_error_covariance, smooth=False):
    transition_matrices = m_dyn.operator.numpy()
    observation_matrices = obs_op.H.fields["x"].numpy().squeeze()
    measurements = observations.state.fields["x"].numpy().squeeze()
    observation_covariance = torch.linalg.inv(
        obs_op.P_chol.fields["x"][0, 0] @ obs_op.P_chol.fields["x"][0, 0].T
    ).numpy()
    initial_state_mean = mu_init.fields["x"].squeeze().numpy()
    initial_state_covariance = sigma_init.fields["x"].squeeze().numpy()
    transition_covariance = model_error_covariance

    pykalman_filter = pykalman.KalmanFilter(
        transition_matrices,
        observation_matrices,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        transition_covariance=transition_covariance,
    )

    return pykalman_filter.smooth(measurements) if smooth else pykalman_filter.filter(measurements)


class TestKalman(TestCase):
    def test_kalman_filter(self):
        for use_tensordict in [False, True]:
            for model_error_covariance in [
                torch.zeros(3, 3),
                torch.eye(3) * 1e-3,
                torch.Tensor([[1, 0, 0], [0, 1, 1], [0, 1, 2]]) * 1e-3,
            ]:
                for diag_obs_error in [True, False]:
                    m_dyn, x_all_gt, observations, obs_op, mu_init, sigma_init = generate_testdata(
                        use_tensordict=use_tensordict,
                        model_error_covariance=model_error_covariance,
                        diag_obs_error=diag_obs_error,
                    )

                    assimilated = kalman_filter(
                        m_dyn,
                        observations,
                        obs_op,
                        mu_init,
                        sigma_init,
                        model_error_covariance=model_error_covariance.unsqueeze(0),
                    )[0]
                    pykalman_assimilated = solve_with_pykalman(
                        m_dyn, observations, obs_op, mu_init, sigma_init, model_error_covariance
                    )[0]
                    diff_with_pykalman = assimilated.fields["x"][0].numpy() - pykalman_assimilated
                    assert (
                        np.max(np.abs(diff_with_pykalman)) < 1e-4
                    ), "Our Kalman filter is inconsistent with the pykalman implementation"

    def test_fixed_lag_kalman_smoother(self):
        for use_tensordict in [False, True]:
            for model_error_covariance in [
                torch.zeros(3, 3),
                torch.eye(3) * 1e-3,
                torch.Tensor([[1, 0, 0], [0, 1, 1], [0, 1, 2]]) * 1e-3,
            ]:
                for diag_obs_error in [True, False]:
                    m_dyn, x_all_gt, observations, obs_op, mu_init, sigma_init = generate_testdata(
                        use_tensordict=use_tensordict,
                        model_error_covariance=model_error_covariance,
                        diag_obs_error=diag_obs_error,
                    )

                    assimilated = fixed_lag_kalman_smoother(
                        m_dyn,
                        observations,
                        obs_op,
                        mu_init,
                        sigma_init,
                        model_error_covariance=model_error_covariance.unsqueeze(0),
                        lag=5,
                    )[0]
                    ks_error = (x_all_gt - assimilated).fields["x"][0].clone()
                    RMSE = torch.mean(torch.sqrt(torch.mean(ks_error**2, dim=1)))
                    # print(f"RMSE of assimilated trajectory with fixed-lag KS: {RMSE}")
                    thr = 0.1 if model_error_covariance is None else 0.2
                    assert RMSE < thr, "The fixed-lag Kalman smoother performance is worse than usual results."

    def test_RTS_kalman_smoother(self):
        for use_tensordict in [False, True]:
            for model_error_covariance in [
                torch.zeros(3, 3),
                torch.eye(3) * 1e-3,
                torch.Tensor([[1, 0, 0], [0, 1, 1], [0, 1, 2]]) * 1e-3,
            ]:
                for diag_obs_error in [True, False]:
                    m_dyn, x_all_gt, observations, obs_op, mu_init, sigma_init = generate_testdata(
                        use_tensordict=use_tensordict,
                        model_error_covariance=model_error_covariance,
                        diag_obs_error=diag_obs_error,
                    )

                    assimilated = kalman_smoother(
                        m_dyn,
                        observations,
                        obs_op,
                        mu_init,
                        sigma_init,
                        model_error_covariance=model_error_covariance.unsqueeze(0),
                    )[0]
                    pykalman_assimilated = solve_with_pykalman(
                        m_dyn, observations, obs_op, mu_init, sigma_init, model_error_covariance, True
                    )[0]
                    diff_with_pykalman = assimilated.fields["x"][0].numpy() - pykalman_assimilated
                    assert (
                        np.max(np.abs(diff_with_pykalman)) < 1e-4
                    ), "Our Kalman smoother is inconsistent with the pykalman implementation"
