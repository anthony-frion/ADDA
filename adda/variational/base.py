import math
import warnings
from typing import Callable, List, Tuple, Type, Union

import torch
from adda.observation.data import ObservationSet
from adda.observation.operators import ObservationOperator
from adda.probability.distributions import DiagonalGaussian, Distribution, impose_batch_sizes
from adda.util.initialization import naive_initialization
from adda.util.state_space import rollout, same_shape
from adda.util.typing import ListOfStates, State
from tensordict import TensorDict
from torch import full_like, searchsorted, Tensor
from torch.optim.optimizer import Optimizer


def sliding_windows(
    state_time_axis: Tensor,
    window_duration: Union[float, int],
    window_shift: Union[float, int] = None,
    discard_partial: bool = False,
) -> Tuple:
    """Get time windows for sliding window data assimilation.

    Args:
        state_time_axis (Tensor): 1D tensor containing times for the state
        window_duration (Union[float, int]): length of assimilation window in time units (float) or time steps (int)
        window_shift (Union[float, int], optional): shift between consecutive windows,
            as a fraction of window size (float) or as a number of time steps (int). Defaults to 0.25.
        discard_partial (bool, optional): Discard windows extending beyond final observation, defaults False.
    """
    assert isinstance(state_time_axis, Tensor), f"Invalid type for state_time_axis: {type(state_time_axis)}"
    assert len(state_time_axis.shape) == 1, f"Invalid shape for state_time_axis: {state_time_axis.shape}"
    assert type(window_duration) in [float, int], f"Invalid type for window_duration: {type(window_duration)}"
    assert type(window_shift) in [float, int], f"Invalid type for window_shift: {type(window_shift)}"
    t0 = state_time_axis[0]
    trange = state_time_axis[-1] - t0
    dt = state_time_axis.diff()
    mean_dt = dt.mean()

    fixed_dt = torch.allclose(dt, mean_dt, atol=1e-6)
    if state_time_axis.dtype is torch.float32:
        warnings.warn(
            "using float32 for state time axis can lead to precision issues when checking for fixed dt in long "
            "sequences. HINT: try generating and providing time tensor using double precision if shape mismatch "
            "assertion presents."
        )
    # try to do everything with integers if the shift and duration are multiples of a fixed dt
    if fixed_dt:
        r = (window_duration / mean_dt).item() if isinstance(window_duration, float) else window_duration
        s = r * window_shift if isinstance(window_shift, float) else window_shift
        if abs(r - round(r)) < 1e-6 and abs(s - round(s)) < 1e-6:
            r, s = round(r), round(s)
            assert r > 0 and s > 0, "window duration and shift must remain positive after rounding"
            if discard_partial:
                n_windows = (len(state_time_axis) - r) // s
            else:
                n_windows = math.ceil(len(state_time_axis) / s)
            idx_ranges = [(n * s, min(n * s + r, len(state_time_axis))) for n in range(n_windows)]
            windows = [
                (state_time_axis[s] - mean_dt / 2.0, min(state_time_axis[e - 1] + mean_dt / 2.0, state_time_axis[-1]))
                for (s, e) in idx_ranges
            ]
            return torch.tensor(windows), idx_ranges

    # time shift between consecutive windows
    assert window_shift > 0 and window_duration > 0, "shift and window duration must be strictly positive"

    if isinstance(window_shift, float):
        assert window_shift <= 1, "shifts of more than one full window are not supported"
        if isinstance(window_duration, float):
            shift = window_duration * window_shift if window_shift < 1.0 else window_duration
            if isinstance(window_duration, int) and shift % 1 == 0 and torch.is_floating_point(state_time_axis):
                shift = int(shift)
            if discard_partial:
                n_windows = int((trange - window_duration) // shift)  # round down
            else:
                n_windows = math.ceil(trange / shift)  # this doesn't do a window starting on the final obs. correct?

            windows = [
                (t0 + n * shift, min(t0 + n * shift + window_duration, state_time_axis[-1])) for n in range(n_windows)
            ]

        elif isinstance(window_duration, int):
            shift = int(window_duration * window_shift) if window_shift < 1.0 else window_duration  # integer shift
            assert shift > 0, f"With windows of {window_duration} steps and shift fraction of {window_shift}, "
            "the shift is strictly less than one time step."
            if discard_partial:
                n_windows = len(state_time_axis) // shift
            else:
                n_windows = math.ceil(len(state_time_axis) / shift)

            windows = [
                (
                    state_time_axis[n * shift],
                    state_time_axis[min(n * shift + window_duration, len(state_time_axis) - 1)],
                )
                for n in range(n_windows)
            ]

    elif isinstance(window_shift, int):  # window_shift and window_duration are both int
        shift = window_shift  # integer shift
        if discard_partial:
            n_windows = len(state_time_axis) // shift
        else:
            n_windows = math.ceil(len(state_time_axis) / shift)
        if isinstance(window_duration, int):
            assert window_shift <= window_duration, "shifts of more than one full window are not supported"
            windows = [
                (state_time_axis[n * shift], state_time_axis[n * shift + window_duration]) for n in range(n_windows)
            ]
        elif isinstance(window_duration, float):
            windows = [
                (state_time_axis[n * shift], min(state_time_axis[n * shift] + window_duration, state_time_axis[-1]))
                for n in range(n_windows)
            ]
    assert shift > 0, "shift must be strictly positive"

    idx_ranges = []  # index ranges into observation_times for each window
    s = 0
    for t_start, t_end in windows:
        s += searchsorted(state_time_axis[s:], t_start, right=False).item()
        e = (
            s + searchsorted(state_time_axis[s:], t_end, right=True).item()
        )  # index of first observation after the current window
        idx_ranges.append((s, e))
    return torch.tensor(windows), idx_ranges


def process_extra_inputs(dynamic_inputs: State, static_inputs: State, time_axis: Tensor):
    """Check and process dynamic/static inputs for 4D-Var.

    Args:
        dynamic_inputs (State): dynamic inputs to check
        static_inputs (State): static inputs to check
        time_axis (Tensor): timeaxis

    Returns:
        dynamic_inputs, static_inputs: checked inputs as State objects
    """
    T = time_axis.nelement()
    if dynamic_inputs is None:
        dynamic_inputs = State(TensorDict(batch_size=(1, T)), time_axis=time_axis)  # State with no fields
    else:
        assert isinstance(dynamic_inputs, State), "extra_inputs must be of type State"
        assert dynamic_inputs.fields.batch_size == (
            1,
            T,
        ), f"shape mismatch between {dynamic_inputs.fields.batch_size} and {(1, T)}"
    if static_inputs is None:
        static_inputs = State(TensorDict(batch_size=(1, 1)))  # State with no fields
    return dynamic_inputs, static_inputs


def process_4dvar_inputs(
    obs_op: ObservationOperator,
    m_dyn: Callable,
    dynamic_inputs: State,
    static_inputs: State,
    x_init: State,
    background_prior: Distribution,
    observations: ObservationSet,
    state_time_axis: Tensor = None,
    model_error_distribs: Distribution = None,
    window_duration: Union[float, int] = None,
    window_shift: Union[float, int] = None,
    discard_partial: bool = False,
    rollout_init: bool = True,
    c_proj: Callable = None,
) -> Tuple[Tensor, Union[List, Tensor], State, Distribution]:
    """Check/update inputs for 4D-Var routines.

    Args:
        obs_op (ObservationOperator): mapping from system state sequence to observations
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        x_init (State): first guess for initial state (strong-constraint) or all states (weak-constraint).
          Uses prior mean if None. should have a leading batch dimension, followed by time dimension for weak
            constraint.
        background_prior (Distribution): prior distribution in initial state
        observations (ObservationSet): observations data to be assimilated
        model_error_distribs (Distribution): distributions of model errors at
          each time step. Must have length 1 on batch axis.
        state_time_axis (Tensor, optional): The time axis on which the assimilated state should be defined.
            If unspecified, uses the time axis of x_init if possible, otherwise the time axis of obs_op.
        window_duration (Union[float, int]): length of assimilation window in time units (float) or time steps (int)
        window_shift (Union[float, int], optional): shift between consecutive windows,
            as a fraction of the window size (float) or as a number of time steps (int). Defaults to 0.25.
        discard_partial (bool, optional): Discard windows extending beyond final observation, defaults False.
        rollout_init (bool): indicates whether to copy or rollout the input state if it is only one time step,
            for weak-constraint 4D-Var
        c_proj (Callable, optional): optional projection from control-space fields to model-space fields.
            It must take and return a TensorDict with batch size (1, 1).
            For strong-constraint 4D-Var, the background prior remains in control space.
            Currently, this is only supported for single window strong-constraint 4D-Var.
    """
    is_weak = model_error_distribs is not None
    if state_time_axis is not None:
        if x_init is not None and is_weak:
            assert x_init.time_axis.dtype == state_time_axis.dtype and torch.allclose(
                x_init.time_axis, state_time_axis
            ), "x_init and state_time_axis are both defined and contradict each other"
    elif x_init is not None and is_weak:
        state_time_axis = x_init.time_axis
    else:
        assert hasattr(obs_op, "time_axis"), "if the observation operator has no time axis, "
        "then the state time axis should be specified using arguments state_time_axis or x_init"
        state_time_axis = obs_op.time_axis
    if is_weak:
        state_time_axis_dt = state_time_axis.diff()
        assert torch.allclose(
            state_time_axis_dt, state_time_axis_dt[0], atol=1e-6, rtol=1e-3
        ), "For weak-constraint 4D-Var, the state time axis must be regular."
    T = state_time_axis.nelement()

    assert isinstance(observations, ObservationSet), "Observations must be of type ObservationSet"
    if hasattr(observations, "time_axis"):
        assert (obs_op.time_axis == observations.time_axis).all(), "time axis mismatch"
    if hasattr(observations, "valid_interval"):
        assert (
            observations.valid_time_interval[0] >= obs_op.time_axis[0]
            and observations.valid_time_interval[1] <= obs_op.time_axis[-1]
        ), "observations outside operator time range"
    if isinstance(window_duration, Tensor):
        assert window_duration.nelement() == 1, " window_duration should have a single value or be a float or int"
        window_duration = window_duration.item()

    if window_duration is not None:
        windows, idx_ranges = sliding_windows(
            state_time_axis, window_duration, window_shift=window_shift, discard_partial=discard_partial
        )
        if c_proj is not None:
            raise ValueError(
                "projection of the initial state to the model space is"
                " currently not supported for sliding-window 4D-Var"
            )
    else:
        windows, idx_ranges = None, None

    if background_prior is not None:
        background_prior = impose_batch_sizes(background_prior, (1, 1), ["mu", "sigma"])

    if x_init is not None:
        assert x_init.fields.shape[0] == 1, "batching is not supported for classical 4D-Var"
        if background_prior is not None:
            assert same_shape(x_init.fields[:1, :1], background_prior.mean), "size mismsatch"
        if not is_weak:
            assert x_init.fields.shape[1] == 1, "the initialization of strong-constraint 4D-Var "
            "should have a time dimension of 1"
    elif not is_weak:
        x_init = State(background_prior.mean, time_axis=state_time_axis[:1])

    if is_weak:
        model_error_distribs = impose_batch_sizes(model_error_distribs, (1, T - 1), ["mu", "sigma"])
        if c_proj is not None:
            raise ValueError(
                "projection of the initial state to the model space is"
                " currently not supported for weak-constraint 4D-Var"
            )

        x_init = process_wc4dvar_init(
            x_init, m_dyn, background_prior, observations, state_time_axis=state_time_axis, rollout_init=rollout_init
        )
        if hasattr(model_error_distribs, "mu"):
            assert same_shape(
                x_init.fields[:, 1:], model_error_distribs.mu
            ), "dimension mismatch between state sequence and model error prior"
    x_init = x_init.detach().clone()  # might need to change this if we're differentiating through the 4D-Var operations
    x_init.fields.requires_grad_(True)

    dynamic_inputs, static_inputs = process_extra_inputs(dynamic_inputs, static_inputs, state_time_axis)

    return dynamic_inputs, static_inputs, x_init, state_time_axis, windows, idx_ranges, model_error_distribs


def process_wc4dvar_init(
    x_init: State,
    m_dyn: Callable,
    background_prior: Distribution,
    observations: ObservationSet,
    state_time_axis: Tensor = None,
    rollout_init: bool = True,
    dynamic_inputs: State = None,
    static_inputs: State = None,
):
    """Returns the time series corresponding to the initial state of the weak-constraint 4D-Var optimization.

    Args:
        x_init (State): a State passed as input to wc4dvar, containing either one time step or the whole time range
        m_dyn (Callable): time stepping operator, taking inputs x, dt, dynamic_inputs, static_inputs
        background_prior (Distribution): prior distribution of the initial state
        observations (ObservationSet): list of observations at each time point
        state_time_axis (Tensor, optional): The time axis on which the assimilated state should be defined.
        rollout_init (bool): if x_init is just the state at one time steps, decides whether to copy it or roll it out

    Returns:
        x_init (State): the processed initial state with the right format
    """
    if x_init is None and background_prior is not None:
        x_init = State(background_prior.mean)
    if isinstance(x_init, State):
        if x_init.fields.batch_size[1] == state_time_axis.nelement():
            x_init.time_axis = state_time_axis
        elif x_init.fields.batch_size[1] == 1:
            if rollout_init:
                x_init = rollout(m_dyn, state_time_axis, x_init, dynamic_inputs, static_inputs)
            else:
                x_init = x_init.expand_time(state_time_axis)
    else:
        assert x_init is None, "at this point of the processing x_init has to be a State or None"
        x_init = naive_initialization(observations)
        assert not torch.isnan(x_init).any(), (
            "naive initialization failed: some fields/locations have no data. To "
            "solve this problem, provide an x_init or background_prior"
        )
    return x_init


def sliding_window_4dvar(
    da_function: Callable,
    m_dyn: Callable,
    observations: ObservationSet,  # array or less
    obs_op: ObservationOperator,
    window_duration: Union[float, int],
    window_shift: Union[float, int] = 0.25,
    model_error_distribs: Distribution = None,
    discard_partial: bool = False,
    x_init: State = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    background_prior: Distribution = None,
    covariance_factors: Union[float, int, list] = 1,
    state_time_axis: Tensor = None,
    optimizer_class: Type[Optimizer] = torch.optim.LBFGS,
    optimizer_pars: dict = None,
    n_steps: int = 10,
    verbose=0,
    c_proj: Callable = None,
    **kwargs,
) -> ListOfStates:
    """Weak- or strong- constraint 4D-Var for overlapping windows of observations.

    Args:
        da_function (Callable): data assimiliation function to be used on each window
        m_dyn (Callable): time stepping operator, taking inputs x, t, dt, extra_inputs
        observations (ObservationSet): set of observations to be assimilated.
        obs_op (ObservationOperator): observation operator used to map between system states and observations
        window_duration (Union[float, int]): window length in time units
        window_shift (Union[float, int], optional): shift between consecutive windows,
            as a fraction of window size (float) or as a number of time steps (int). Defaults to 0.25.
        model_error_distribs (Distribution): distributions of model errors at each time step (only for weak-constraint)
        state_time_axis (Tensor, optional): The time axis on which the assimilated state should be defined.
            If unspecified, uses the time axis of x_init if possible, otherwise the time axis of obs_op.
        discard_partial (bool, optional): Drop windows extending beyond last obs, default False.
        x_init (State): first guess of initial state (strong-constraint) or state trajectory (weak-constraint).
          Uses prior mean if None. Should have leading batch, then time dimension (of length 1 for strong-constraint).
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
        background_prior (Distribution, optional): background prior for first window. Defaults to None.
        covariance_factors (Union[float, int, list]): factors of the covariance matrix
          for all prior terms after the first window
        optimizer_class: class of a pytorch optimizer, default is LBFGS
        optimizer_pars (dict): parameters for optimizer (learning rate etc.)
        n_steps (int): the number of optimization steps
        verbose (bool): if True, print loss at each iteration. default is False
        c_proj (Callable, optional): kept for API compatibility and forwarded to input checks.
            Sliding-window mode currently does not support c_proj and raises an assertion if provided.

    Returns:
        ListOfStates: initial state for each assimilation window
    """
    is_weak = model_error_distribs is not None
    (
        dynamic_inputs,
        static_inputs,
        x_init,
        state_time_axis,
        windows,
        idx_ranges,
        model_error_distribs,
    ) = process_4dvar_inputs(
        obs_op,
        m_dyn,
        dynamic_inputs,
        static_inputs,
        x_init,
        background_prior,
        observations,
        model_error_distribs=model_error_distribs,
        state_time_axis=state_time_axis,
        window_duration=window_duration,
        window_shift=window_shift,
        discard_partial=discard_partial,
        c_proj=c_proj,
    )
    device = x_init.device
    dt = (state_time_axis[-1] - state_time_axis[0]) / len(state_time_axis)  # mean time step
    if background_prior is not None:
        assert isinstance(background_prior, DiagonalGaussian), "background prior must be diagonal Gaussian."
        first_background_prior = DiagonalGaussian(background_prior.mu, background_prior.sigma)  # copy for later
    if type(covariance_factors) in [int, float]:
        covariance_factors = [covariance_factors for i in range(len(idx_ranges) - 1)]
    assert isinstance(covariance_factors, list), "covariance_factors must be int, float or list"

    x_eachwin = []  # IC (hc) or state trajectory (wc) for each window
    if is_weak:
        assert torch.allclose(
            x_init.time_axis, state_time_axis[: len(x_init.time_axis)]
        ), f"x_init.time_axis {x_init.time_axis} is different from state_time_axis {state_time_axis}"

    first_ind = 0

    for i, ((s, e), (t_s, t_e)) in enumerate(zip(idx_ranges, windows)):
        if verbose:
            print(f"Time steps {s} to {e}")
        if is_weak and i == 0:
            x_init = x_init.restrict_time_domain(t_s, t_e + 1e-3 * dt).detach().clone().requires_grad_()
        # positional arguments for this window
        obs_bounds = (x_init.time_axis[0], x_init.time_axis[-1] + 1e-3 * dt) if is_weak else (t_s, t_e + 1e-3 * dt)
        da_args = [
            m_dyn,
            observations.restrict_time_domain(obs_bounds[0], obs_bounds[1]).to(device),
            obs_op.restrict_time_domain(obs_bounds[0], obs_bounds[1]).to(device),
        ]
        if is_weak:
            first_ind = torch.searchsorted(state_time_axis, x_init.time_axis[0])
            da_args.append(model_error_distribs[:, first_ind : first_ind + len(x_init.time_axis) - 1])
        state_time_axis_arg = state_time_axis[s:e]
        # keyword arguments for this window
        da_kwargs = dict(
            x_init=x_init,
            state_time_axis=state_time_axis_arg,
            dynamic_inputs=dynamic_inputs[:, s : e - 1],
            static_inputs=static_inputs,
            background_prior=background_prior,
            optimizer_class=optimizer_class,
            optimizer_pars=optimizer_pars,
            n_steps=n_steps,
            verbose=verbose,
            check_inputs=False,  # already checked once, don't need to check again for each window
        )

        x = da_function(*da_args, **da_kwargs, **kwargs)
        x_eachwin.append(x)

        if i + 1 < len(idx_ranges):  # prepare for next window
            next_s, next_e = idx_ranges[i + 1]
            next_t_s, next_t_e = windows[i + 1]
            with torch.no_grad():
                if is_weak:
                    # to initialize the next window's trajectory, we use some of this window's trajectory and extend
                    # with a rollout as needed
                    x_init_fromprev = x[:, next_s - s :]
                    if (
                        e == next_e
                    ):  # this window has reached the end of the observations, no need for a forecast rollout
                        x_init = x_init_fromprev
                    else:  # generate a forecast to initialize unknown system states for next window
                        first_ind = torch.searchsorted(state_time_axis, x_init_fromprev.time_axis[-1])
                        t_rollout = state_time_axis[first_ind:next_e]
                        dynamic_inputs_win = dynamic_inputs[:, e:next_e]
                        x_init_fromroll = rollout(
                            m_dyn,
                            t_rollout,
                            x[:, -1:],
                            dynamic_inputs=dynamic_inputs_win,  # include dynamic_inputs for IC
                            static_inputs=static_inputs,
                        )[
                            :, 1:
                        ]  # discard IC

                        x_init = State(
                            torch.cat([x_init_fromprev.fields, x_init_fromroll.fields], dim=1),
                            time_axis=torch.cat([x_init_fromprev.time_axis, x_init_fromroll.time_axis], dim=0),
                        )

                else:  # for strong-constraint, we roll from start of this window to start of the next
                    t_rollout = obs_op.time_axis[s : next_s + 1]
                    x_init = rollout(m_dyn, t_rollout, x, dynamic_inputs[:, s:next_s], static_inputs)[
                        :, -1:
                    ]  # only final state

            x_init = x_init.detach().clone()
            x_init.fields.requires_grad_(True)
            if background_prior is not None:  # shift mean, scale covariance of background prior
                mu = x_init.fields[:, :1] if is_weak else x_init.fields
                mu = mu.detach().clone().reshape(background_prior.mu.shape)  # remove batch dimension
                background_prior = DiagonalGaussian(mu, first_background_prior.sigma * covariance_factors[i])

    return x_eachwin, windows, idx_ranges


def assemble_analysis(
    x_eachwin: ListOfStates,
    idx_ranges,
    time_axis: Tensor,
    is_weak: bool,
    m_dyn: Callable = None,
    dynamic_inputs: State = None,
    static_inputs: State = None,
) -> State:
    """Combines the results of a windowed DA method into a single analysis for all time steps.

    Args:
        x_eachwin (ListOfStates): assimilated state(s) for each analysis window
        idx_ranges: range of observations for each window
        time_axis (Tensor): global list of observation times for all windows
        is_weak (bool): was this a weak_constraint method
        m_dyn (Callable): time stepping operator, taking inputs x, t, dt, extra_inputs
        dynamic_inputs (State): extra inputs defined for each input state (e.g. TOA incoming solar irradiance)
        static_inputs (State): extra inputs that don't vary over time (e.g. bathymetry)
    """
    dynamic_inputs, static_inputs = process_extra_inputs(dynamic_inputs, static_inputs, time_axis)
    T = time_axis.nelement()
    analysis_fields = (
        full_like(x_eachwin[0].fields[:1, :1].clone().detach(), torch.nan).expand(1, T).clone()
    )  # clone() to actually copy the data
    # analysis is now a TensorDict with a singleton batch and time dimensions

    for i, (s, e) in enumerate(idx_ranges):
        # don't use the part of this analyis window that overlaps later windows
        e = idx_ranges[i + 1][0] if i < len(idx_ranges) - 1 else e
        n_thiswindow = e - s

        if is_weak:
            abs_reldiff = torch.abs(
                (time_axis[s:e] - x_eachwin[i].time_axis[:n_thiswindow]) * (e - s) / (time_axis[e - 1] - time_axis[s])
            )
            assert (
                abs_reldiff <= 1e-3
            ).all(), f"time axis mismatch: {time_axis[s:e]} is different from {x_eachwin[i].time_axis[:n_thiswindow]}"
            analysis_fields[:, s:e] = x_eachwin[i].fields[:, :n_thiswindow]
        else:
            x0 = x_eachwin[i]
            assert x0.time_axis[0] == time_axis[s], "time axis mismatch"
            # rollout to get state trajectory and remove batch dimension
            analysis_fields[:, s:e] = rollout(
                m_dyn, time_axis[s:e], x0, dynamic_inputs[:, s : e - 1], static_inputs
            ).fields

    return State(analysis_fields, time_axis=time_axis)
