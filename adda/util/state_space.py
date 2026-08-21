from typing import Callable, Union

import torch
from adda.system.state import State
from tensordict import TensorDict
from torch import Tensor


def rollout(
    m_dyn: Callable,
    output_times: Tensor,
    x0: State,
    dynamic_inputs: State = None,
    static_inputs: State = None,
    store_every=None,
) -> State:
    """Rollout a time-stepping function over a set of times.

    Args:
        m_dyn (Callable): time stepping function with inputs x, dt, dynamic_inputs, static_inputs
        x0 (State): initial state, a tensor or tensordict, with a batch dimension
        output_times (torch.Tensor): times relative to initialization to produce output. must be increasing.
                                    optionally includes 0.
        dynamic_inputs (State): extra inputs (e.g. BCs, forcing) for each instance of calling the time
                                    stepping function.
        static_inputs (State): extra inputs (e.g. bathymetry) for each instance of calling the time
                                    stepping function.
        store_every (int): if specified, indicates at which frequency the states should be stored during the rollout

    Returns:
        x (State): states at each output time. First dimension is batch, second is time.
    """
    assert x0.fields.batch_size[1] == 1, "rollout supports batching, but requires a singleton time dimension"
    t0 = x0.time_axis[0]
    output_ic = output_times[0] == t0
    if not output_ic:
        output_times = torch.cat((torch.full((1,), t0), output_times), dim=0)

    dt_all = torch.diff(output_times)

    x = x0
    x_all = [x0] if output_ic else []
    for i, dt in enumerate(dt_all):
        di = dynamic_inputs[:, i : i + 1] if dynamic_inputs is not None else None
        x = m_dyn(x, dt, di, static_inputs)
        if store_every is None or (i + 1) % store_every == 0:
            x_all.append(x)

    time_axis = output_times if output_ic else output_times[1:]
    if store_every is not None:
        time_axis = time_axis[store_every - 1 :: store_every]
    return State(torch.cat([x.fields for x in x_all], dim=1), time_axis=time_axis)


def allclose(x: Union[Tensor, TensorDict], y: Union[Tensor, TensorDict], **kwargs) -> bool:
    """Wrapper on torch.allclose() to support TensorDict.

    Args:
        x (State): the first data structure
        y (State): the second data structure

    Returns:
        bool: result
    """
    if isinstance(x, TensorDict):
        assert x.keys() == y.keys(), "key mismatch"
        for k, v in x.items():
            if not torch.allclose(v, y[k], **kwargs):
                return False
        return True
    return torch.allclose(x, y, **kwargs)


def weighted_sse(x: TensorDict, mu: TensorDict, sigma: TensorDict = None) -> Tensor:
    """Compute weighted sum of squared errors (x - mu) / sigma ** 2.

    Args:
        x (TensorDict): variable(s) for which to calculate the weighted sse
        mu (TensorDict): target for x
        sigma (TensorDict, optional): variance (or inverse weight) for each residual
    """
    wsse = TensorDict(batch_size=x.batch_size)
    nbatchdims = wsse.ndim
    z = x - mu if sigma is None else (x - mu) / sigma
    zsq = z * z
    wsse = None
    for k, v in z.items():
        dims = tuple(range(nbatchdims, v.ndim))
        if wsse is None:
            wsse = zsq[k].sum(dim=dims) if dims else zsq[k]
        else:
            wsse += zsq[k].sum(dim=dims) if dims else zsq[k]
    return wsse


def same_shape(x1: TensorDict, x2: TensorDict):
    """Check whether two TensorDicts have the same dimensions.

    Args:
        x1: first TD
        x2: second TD

    Returns:
        is_same (bool): True if the TDs the same batch/spatial dimensions (for all keys as needed).
    """
    if x1.shape != x2.shape or x1.keys() != x2.keys():
        return False
    for k in x1.keys():
        if x1[k].shape != x2[k].shape:
            return False
    return True
