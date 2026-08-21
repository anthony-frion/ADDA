from unittest import TestCase

import torch
from adda.observation.operators import random_sparse_noisy_obs
from adda.probability.distributions import DiagonalGaussian
from adda.system.state import State
from adda.util.state_space import allclose, rollout
from adda.variational.base import assemble_analysis
from adda.variational.strong_constraint_4dvar import sc4dvar_single_window, sc4dvar_sliding_window
from tensordict import TensorDict


def m_id(x: State, dt, dynamic_inputs, static_inputs):
    return State(x.fields, x.time_axis + dt)


optimizer_class = torch.optim.LBFGS


optimizer_pars = dict(
    lr=1e0,
    max_iter=1000,
    max_eval=-1,
    tolerance_grad=1e-12,
    tolerance_change=1e-12,
    history_size=10,
)


n_steps = 10


def generate_testdata_sc_id(ndims=3, T=5, obs_noise_sd=1.0, p_obs=1.0, use_tensordict=False, regular_time=True):
    """Generate test data for variational unit algorithms."""
    mu_init, sigma_init = torch.zeros(1, 1, ndims), torch.ones(1, 1, ndims)
    if use_tensordict:
        mu_init, sigma_init = (
            TensorDict(x=mu_init, batch_size=(1, 1)),
            TensorDict(x=sigma_init, batch_size=(1, 1)),
        )
    prior = DiagonalGaussian(mu_init, sigma_init)
    if not use_tensordict:
        prior.mean.batch_size, prior.sigma.batch_size = (1, 1), (1, 1)

    x0_gt = State(prior.sample())  # ground truth IC
    if regular_time:
        rollout_times = torch.arange(T, dtype=mu_init.dtype)
    else:
        rollout_times = torch.rand(2 * T) * T  # double the number of timepoints to compensate for irregular sampling
        rollout_times = torch.sort(rollout_times).values
    x_all_gt = rollout(m_id, rollout_times, x0_gt)  # ground truth rollout

    p_obs = dict(x=p_obs)

    x_all_gt, obs_op, y_all = random_sparse_noisy_obs(x_all_gt, obs_noise_sd, p_obs)
    return x_all_gt, y_all, obs_op, prior, obs_noise_sd


def true_posterior_sc_id(obs_op, y_all, prior, obs_noise_sd=1.0):
    """Get the true posterior for a single-window strong-constraint 4D-Var test problem with identity dynamics.

    Args:
        obs_ops: observation operatros
        y_all: observations
        prior: prior distribution on initial state
        obs_noise_sd (float): obseration noise s.d.

    Returns:
        true_posterior: distribution object for the true posterior
    """
    n_obs = obs_op.mask.fields.sum(dim=1).unsqueeze(1)
    sum_obs = (y_all.state.fields * y_all.mask.fields).sum(dim=1).unsqueeze(1)
    Vo = obs_noise_sd**2
    Vp = prior.sigma**2

    # all TensorDicts:
    posterior_var = 1.0 / (1.0 / Vp + n_obs / Vo)
    posterior_sd = posterior_var.sqrt()
    posterior_mean = posterior_var * (prior.mu / Vp + sum_obs / Vo)

    gt_posterior = DiagonalGaussian(posterior_mean, posterior_sd)
    return gt_posterior


class TestVariational(TestCase):
    def test_sc4dvar_singlewindow(self):
        for use_tensordict in [False, True]:
            noise_sd = torch.rand(1).item() * 0.2 + 0.9

            x_all, y_all, obs_op, prior, noise_sd = generate_testdata_sc_id(
                T=5, obs_noise_sd=noise_sd, use_tensordict=use_tensordict
            )

            x0 = sc4dvar_single_window(
                m_id,
                y_all,
                obs_op,
                background_prior=prior,
                optimizer_class=optimizer_class,
                optimizer_pars=optimizer_pars,
                n_steps=n_steps,
            )

            gt_posterior = true_posterior_sc_id(obs_op, y_all, prior, obs_noise_sd=noise_sd)

            self.assertTrue(allclose(x0.fields, gt_posterior.mean), msg="incorrect posterior")

    def test_sc4dvar_singlewindow_with_normalization(self):
        noise_sd = torch.rand(1).item() * 0.2 + 0.9

        x_all, y_all, obs_op, prior, noise_sd = generate_testdata_sc_id(T=5, obs_noise_sd=noise_sd, use_tensordict=True)

        x0 = sc4dvar_single_window(
            m_id,
            y_all,
            obs_op,
            background_prior=prior,
            optimizer_class=optimizer_class,
            optimizer_pars=optimizer_pars,
            n_steps=n_steps,
            normalize=True,
        )

        gt_posterior = true_posterior_sc_id(obs_op, y_all, prior, obs_noise_sd=noise_sd)

        self.assertTrue(allclose(x0.fields, gt_posterior.mean), msg="incorrect posterior")

    def test_sc4dvar_singlewindow_with_irregular_time(self):
        noise_sd = torch.rand(1).item() * 0.2 + 0.9

        x_all, y_all, obs_op, prior, noise_sd = generate_testdata_sc_id(
            T=5, obs_noise_sd=noise_sd, use_tensordict=True, regular_time=False
        )

        x0 = sc4dvar_single_window(
            m_id,
            y_all,
            obs_op,
            background_prior=prior,
            optimizer_class=optimizer_class,
            optimizer_pars=optimizer_pars,
            n_steps=n_steps,
        )

        gt_posterior = true_posterior_sc_id(obs_op, y_all, prior, obs_noise_sd=noise_sd)
        self.assertTrue(allclose(x0.fields, gt_posterior.mean, atol=1e-3), msg="incorrect posterior")

    def test_sc4dvar_slidingwindow(self):
        window_duration = 5
        shift_fraction = 0.4

        for use_tensordict in [False, True]:
            for discard_partial in [False, True]:
                obs_noise_sd = torch.rand(1).item() * 0.2 + 0.9

                x_all, y_all, obs_op, prior, obs_noise_sd = generate_testdata_sc_id(
                    T=10, obs_noise_sd=obs_noise_sd, use_tensordict=use_tensordict
                )

                x0_eachwin, windows, idx_ranges = sc4dvar_sliding_window(
                    m_id,
                    y_all,
                    obs_op,
                    window_duration=window_duration,
                    window_shift=shift_fraction,
                    discard_partial=discard_partial,
                    background_prior=prior,
                    optimizer_class=optimizer_class,
                    optimizer_pars=optimizer_pars,
                    n_steps=n_steps,
                )
                dt = (obs_op.time_axis[-1] - obs_op.time_axis[0]) / len(obs_op.time_axis)
                for i, ((s, e), (t_s, t_e)) in enumerate(zip(idx_ranges, windows)):
                    gt_posterior = true_posterior_sc_id(
                        obs_op.restrict_time_domain(t_s, t_e + 1e-3 * dt),
                        y_all.restrict_time_domain(t_s, t_e + 1e-3 * dt),
                        prior,
                        obs_noise_sd=obs_noise_sd,
                    )

                    if i + 1 < len(idx_ranges):  # prepare for next window
                        prior = DiagonalGaussian(
                            gt_posterior.mu.detach().clone(), prior.sigma.detach().clone()
                        )  # new mean, old (co)variance
                        self.assertTrue(
                            allclose(x0_eachwin[i].fields, gt_posterior.mean, atol=1e-3),
                            msg=f"incorrect posterior on window {i}: {x0_eachwin[i].fields['x']} is different from {gt_posterior.mean['x']}, the difference is {gt_posterior.mean['x'] - x0_eachwin[i].fields['x']}",
                        )

                is_weak = False
                analysis = assemble_analysis(
                    x0_eachwin,
                    idx_ranges,
                    obs_op.time_axis,
                    is_weak,
                    m_dyn=m_id,
                )

    def test_sc4dvar_slidingwindow_with_irregular_time(self):
        window_duration = 5
        shift_fraction = 0.4

        for use_tensordict in [False, True]:
            for discard_partial in [False, True]:
                obs_noise_sd = torch.rand(1).item() * 0.2 + 0.9

                x_all, y_all, obs_op, prior, obs_noise_sd = generate_testdata_sc_id(
                    T=10, obs_noise_sd=obs_noise_sd, use_tensordict=use_tensordict, regular_time=False
                )

                x0_eachwin, windows, idx_ranges = sc4dvar_sliding_window(
                    m_id,
                    y_all,
                    obs_op,
                    window_duration=window_duration,
                    window_shift=shift_fraction,
                    discard_partial=discard_partial,
                    background_prior=prior,
                    optimizer_class=optimizer_class,
                    optimizer_pars=optimizer_pars,
                    n_steps=n_steps,
                )
                dt = (obs_op.time_axis[-1] - obs_op.time_axis[0]) / len(obs_op.time_axis)
                for i, ((s, e), (t_s, t_e)) in enumerate(zip(idx_ranges, windows)):
                    gt_posterior = true_posterior_sc_id(
                        obs_op.restrict_time_domain(t_s, t_e + 1e-3 * dt),
                        y_all.restrict_time_domain(t_s, t_e + 1e-3 * dt),
                        prior,
                        obs_noise_sd=obs_noise_sd,
                    )

                    if i + 1 < len(idx_ranges):  # prepare for next window
                        prior = DiagonalGaussian(
                            gt_posterior.mu.detach().clone(), prior.sigma.detach().clone()
                        )  # new mean, old (co)variance
                        self.assertTrue(
                            allclose(x0_eachwin[i].fields, gt_posterior.mean, atol=1e-3),
                            msg=f"incorrect posterior on window {i}: {x0_eachwin[i].fields['x']} is different from {gt_posterior.mean['x']}, the difference is {gt_posterior.mean['x'] - x0_eachwin[i].fields['x']}",
                        )

                is_weak = False
                analysis = assemble_analysis(
                    x0_eachwin,
                    idx_ranges,
                    obs_op.time_axis,
                    is_weak,
                    m_dyn=m_id,
                )
