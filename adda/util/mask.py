import warnings
from typing import Dict, List, Tuple, Union

import torch
from adda.system.state import State
from tensordict import TensorDict
from torch import Tensor


def mask_state(x: State, mask: State) -> TensorDict:
    """Apply mask to the system state variable(s)

    Args:
        x (State): system state variable(s)
        mask (State): mask with boolean. If x has a batch axis and mask does not, the mask will be broadcast.
    """
    x_masked = TensorDict(batch_size=())  # no batch dimensions after masking
    for key, v in x.fields.items():
        if v.shape[0] > 1:
            x_masked[key] = v[mask.fields[key].expand(v.shape[0], *mask.fields[key].shape[1:])]
        else:
            x_masked[key] = v[mask.fields[key]]
    return x_masked


def time_interpolation_of_regular_state(x: State, time_axis: Tensor) -> TensorDict:
    """Linearly interpolates a state x with a regular time axis to an arbitrary time axis.

    Args:
        x (State): system state variable(s). Should be regularly sampled in time.
        time_axis (State): an arbitrary time axis that x should be interpolated to.
    """
    dt = x.time_axis.diff()
    assert torch.allclose(dt, dt[0], atol=1e-6, rtol=1e-3), f"input x should be regularly sampled in time. {dt}"
    assert (
        x.time_axis[0] <= time_axis[0] and x.time_axis[-1] + 1e-3 * dt[0] >= time_axis[-1]
    ), f"window {(x.time_axis[0], x.time_axis[-1])} does not contain window {(time_axis[0], time_axis[-1])}"
    if x.time_axis[-1] < time_axis[-1]:
        warnings.warn(
            "A marginal part of time_axis is outside the bounds of state.time_axis, probably due to rounding errors."
        )
    x_left_indexes = ((time_axis - time_axis[0]) // dt[0]).to(torch.int)
    x_right_indexes = x_left_indexes + 1
    x_right_weights = (time_axis % dt[0]) / dt[0]
    x_left_weights = 1 - x_right_weights
    x_left, x_right = x[:, x_left_indexes], x[:, x_right_indexes]
    x_left.time_axis, x_right.time_axis = time_axis, time_axis
    x_interpolated = x_left_weights * x_left + x_right_weights * x_right
    return x_interpolated


def mask_like(
    x: State,
    p_obs: Dict,
    constant_obs_count_per_step: bool = True,
    constant_obs_count: bool = True,
) -> State:
    """Generate mask like State object.

    Args:
        x (State): the state from which the size and batch axis are used as reference for the output
        p_obs (Dict): a dictionary containing instructions on how to build the mask for each field of x
        constant_obs_count_per_step (bool, optional): If True, ensures that every time step has
            the same number of masked variables.
        constant_obs_count (bool, optional): If True, ensures that the global number of masked variables is fixed.

    Returns:
        State: a state with binary tensor fields, representing a mask with the same fields and time axis as the input x
    """
    m = TensorDict(batch_size=x.fields.batch_size)
    for key, val in x.fields.items():
        m[key] = generate_mask(
            val.shape,
            p_obs[key],
            constant_obs_count_per_step=constant_obs_count_per_step,
            constant_obs_count=constant_obs_count,
        )
    return State(m, x.time_axis)


def mask_like_from_tensor(
    x: State,
    td_mask: Dict,
) -> State:
    """Generate mask like State object using the observation proportions in td_mask.

    Args:
        x (State): the state from which the size and batch axis are used as reference for the output
        td_mask (Dict): a dictionary (or TensorDict) containing observation probabilities for each field of x

    Returns:
        State: a state with binary tensor fields, representing a mask with the same fields and time axis as the input x
    """
    m = TensorDict(batch_size=x.fields.batch_size)
    for key, val in x.fields.items():
        m[key] = generate_mask_from_tensor(
            val.shape,
            td_mask[key],
        )
    return State(m, x.time_axis)


def generate_mask(
    shape: Union[torch.Size, List, Tuple],
    p_obs: float,
    constant_obs_count_per_step: bool = True,
    constant_obs_count: bool = True,
) -> Union[TensorDict, Tensor]:
    """Generates a mask for a State object with batch and time axes shared across all fields.

    Args:
        shape: the shape of the mask
        p_obs: the proportion of observed variables across the trajectory.
        If p_obs is a boolean tensor than it is directly used as the mask.
        contant_obs_count_per_step: if this is true, enforces that the number of observed variables remains the same
          for each time step. Assumes first axis is time.
        constant_obs_count: if this is true, the total number of observed variables is calculated deterministically.
          It can still vary for each time step. Applied only if constant_obs_per_step is False.
    """
    shape = torch.Size(shape)
    if constant_obs_count_per_step:
        if shape.numel() > 1e9:
            warnings.warn(
                "Attempting to generate a high-dimensional mask, memory issues may occur. "
                "If so, try setting constant_obs_count_per_step to False for a less memory-hungry computation, "
                "or reducing the number of time steps."
            )
        field_dim = shape[1:].numel()
        n_obs_per_step = int(field_dim * p_obs)
        shape_flat = (shape[0], field_dim)
        sample_indexes = torch.rand(shape_flat, dtype=torch.float16).topk(n_obs_per_step, dim=-1).indices
        mask = torch.zeros(shape_flat, dtype=bool).scatter_(dim=-1, index=sample_indexes, value=True)
        mask = mask.reshape(shape)
    elif constant_obs_count:
        field_dim = shape.numel()
        n_obs = int(p_obs * field_dim)
        mask = torch.zeros(field_dim, dtype=bool)
        mask[:n_obs] = True
        mask = mask[torch.randperm(field_dim)].reshape(shape)
    else:
        mask = torch.rand(shape, dtype=torch.float16) < p_obs
    return mask


def generate_mask_from_tensor(shape: Union[torch.Size, List, Tuple], t_mask: Tensor) -> Union[TensorDict, Tensor]:
    """Generates a mask for a State object with batch and time axes shared across all fields. If t_mask is a boolean
    Tensor then it deterministically builds the mask, otherwise its elements are interpreted as probablities of
    observation.

    Args:
        shape: the shape of the mask
        t_mask: a binary tensor used to build the mask
    """
    shape = torch.Size(shape)
    assert isinstance(t_mask, Tensor), "argument 't_mask' must be a Tensor"
    if t_mask.dtype == torch.bool:
        mask = t_mask.reshape(shape)
    else:
        assert (
            torch.min(t_mask) >= 0 and torch.max(t_mask) <= 1
        ), f"elements of t_mask must be between 0 and 1 but are between {torch.min(t_mask)} and {torch.max(t_mask)}"
        uniform_samples = torch.rand_like(t_mask)
        mask = (t_mask > uniform_samples).reshape(shape)
    return mask
