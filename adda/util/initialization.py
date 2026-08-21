import torch
from adda.observation.data import FullStateObservationSet, MaskedStateObservationSet, ObservationSet
from adda.system.state import State
from scipy import interpolate
from torch import full_like


def naive_initialization(observations: ObservationSet, default_value=torch.nan) -> State:
    """A shortcut to interp_initialization with selection of next available observation."""
    return interp_initialization(observations, kind="next", default_value=default_value)


def interp_initialization(observations: ObservationSet, kind="linear", default_value=torch.nan, clone=False) -> State:
    """Return an interpolation of the observations, separately for each variable. This method acts sequentially on each
    variable, so it might be slow for states with many variables.

    Args:
        observations: a set of possibly noisy observations.
        kind: the kind of interpolation to perform. Refer to scipy.interpolate.interp1d for details.
        default_value: the value to attribute to variables that are not observed.
        clone: a boolean indicating whether to clone before outputting the initialization.
    """
    if isinstance(ObservationSet, FullStateObservationSet):
        return observations.state
    assert isinstance(
        observations, MaskedStateObservationSet
    ), f"nearest initialization not supported for observations of type {type(observations)}"
    init = full_like(observations.state.fields, torch.nan)
    B, T = observations.state.fields.batch_size
    assert B == 1, f"batching is not supported by interp_initialization but batch size is {B, T}"
    time_axis = observations.state.time_axis
    with torch.no_grad():
        for key, mask in observations.mask.fields.items():
            init_vals, mask_vals, obs_vals = (  # remove batch axis from each tensor
                init[key].squeeze(0),
                mask.squeeze(0),
                observations.state.fields[key].squeeze(0),
            )
            orig_shape = init_vals.shape[1:]
            init_vals_flat = init_vals.flatten(1, -1)
            mask_vals_flat = mask_vals.flatten(1, -1)
            obs_vals_flat = obs_vals.flatten(1, -1)
            for var in range(init_vals_flat.shape[1]):
                indexes = torch.where(mask_vals_flat[:, var] == 1)[0].to("cpu")  # maybe allow GPU later
                x = time_axis[indexes]
                if len(x) == 0:
                    init_vals_flat[:, var] = default_value
                elif len(x) == 1:
                    init_vals_flat[:, var] = obs_vals_flat[indexes[0], var]
                else:
                    y = obs_vals_flat[indexes, var]
                    interp = interpolate.interp1d(x.detach().cpu(), y.detach().cpu(), kind=kind)
                    min_ind, max_ind = int(indexes[0]), int(indexes[-1])
                    init_vals_flat[min_ind:max_ind, var] = torch.Tensor(interp(time_axis[min_ind:max_ind]))
                    init_vals_flat[:min_ind, var] = y[0]
                    init_vals_flat[max_ind:, var] = y[-1]
            init_vals = init_vals_flat.reshape(1, T, *orig_shape)
            init[key] = init_vals
    if clone:
        init = init.detach().clone()  # prevent sharing memory with observations
    return State(init, time_axis)


def fast_initialization(observations: ObservationSet, default_value=torch.nan, clone=False) -> State:
    """A basic initialization where each variable takes the value of the closest observation in the future. If no future
    observations are available, the most recent past observation will be used.

    Args:
        observations: a set of possibly noisy observations.
        default_value: the value to attribute to variables that are not observed.
        clone: a boolean indicating whether to clone before outputting the initialization.
    """
    if isinstance(ObservationSet, FullStateObservationSet):
        return observations.state
    assert isinstance(
        observations, MaskedStateObservationSet
    ), f"naive initialization not supported for observations of type {type(observations)}"
    init = full_like(observations.state.fields, torch.nan)
    B, T = observations.state.fields.batch_size
    assert B == 1, "batching is not supported by naive_initialization"
    with torch.no_grad():
        for key, mask in observations.mask.fields.items():
            init_vals, mask_vals, obs_vals = (  # remove batch axis from each tensor
                init[key].squeeze(0),
                mask.squeeze(0),
                observations.state.fields[key].squeeze(0),
            )

            last_element = torch.full(
                (1, *init_vals.shape[1:]), default_value, dtype=obs_vals.dtype, device=obs_vals.device
            )
            for t in range(T - 1, -1, -1):  # reverse order
                mask_t = mask_vals[t, ...]
                y = obs_vals[t : t + 1, mask_t]
                init_vals[: t + 1, mask_t] = y
                last_element[:, mask_t] = y

            missing_data = torch.isnan(init_vals)
            init_vals = torch.where(missing_data, last_element.expand(init_vals.shape), init_vals)

            init[key] = init_vals.unsqueeze(0)  # re-add batch dimension
    if clone:
        init = init.detach().clone()  # prevent sharing memory with observations
    return State(init, observations.state.time_axis)


def fast_initialization_strong_constraint(observations: ObservationSet, default_value=torch.nan, clone=False) -> State:
    """A basic initialization on the first time step of the series only, selecting the first available observation for
    each variable.

    Args:
        observations: a set of possibly noisy observations.
        kind: the kind of interpolation to perform. Refer to scipy.interpolate.interp1d for details.
        default_value: the value to attribute to variables that are not observed.
        clone: a boolean indicating whether to clone before outputting the initialization.
    """
    if isinstance(ObservationSet, FullStateObservationSet):
        return observations.state
    assert isinstance(
        observations, MaskedStateObservationSet
    ), f"naive initialization not supported for observations of type {type(observations)}"
    init = full_like(observations.state.fields[:, 0:1], default_value)  # only first time step
    B, T = observations.state.fields.batch_size
    assert B == 1, "batching is not support by naive_initialization"
    with torch.no_grad():
        for key, mask in observations.mask.fields.items():
            init_vals, mask_vals, obs_vals = (  # remove batch axis from each tensor
                init[key].squeeze(0),
                mask.squeeze(0),
                observations.state.fields[key].squeeze(0),
            )
            for t in range(T - 1, -1, -1):  # reverse order
                mask_t = mask_vals[t, ...]
                y = obs_vals[t : t + 1, mask_t]
                init_vals[:, mask_t] = y

            init[key] = init_vals.unsqueeze(0)  # re-add batch dimension
    if clone:
        init = init.detach().clone()  # prevent sharing memory with observation
    return State(init, observations.state.time_axis[0:1])
