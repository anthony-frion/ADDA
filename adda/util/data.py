from typing import Union

import torch
from tensordict import TensorDict
from torch import Tensor


def as_TensorDict(x: Union[Tensor, TensorDict]):
    """Cast a Tensor to a TensorDict as needed, creating a field 'x'."""
    if isinstance(x, Tensor):
        return TensorDict(x=x)
    else:
        assert isinstance(x, TensorDict), "x must be Tensor or TensorDict"
        return x


def isnumber(x):
    """Check if input is a number.

    Args:
        x: input to be tested

    Returns:
        bool: True if x is a float, int, or Tensor with no dimensions and one element.
    """
    return isinstance(x, float) or isinstance(x, int) or (isinstance(x, Tensor) and x.ndim == 0 and x.nelement() == 1)


def impose_batch_size(x: TensorDict, size: torch.Size) -> TensorDict:
    """Assign a batch_size to a TensorDict. Throw an error if the TensorDict has more dimensions than the desired
    batch_size.

    Args:
        x (TensorDict): input
        size (torch.Size): desired batch_size

    Returns:
        TensorDict: updated input
    """
    size = torch.Size(size)
    n = len(size)
    assert x.ndim <= n, "too many input batch dimensions"
    if x.ndim == 0:
        x.batch_size = size
    assert x.batch_size == size, "size mismatch"
    return x
