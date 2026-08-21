import warnings
from abc import ABC
from textwrap import indent
from typing import Dict, Union

import tensordict
import torch
from adda.util.data import isnumber
from adda.util.index import sorted_index_range
from tensordict import TensorDict
from torch import Tensor


class State(ABC):
    """Object representing a system state (trajectory).

    Its fields are a Tensordict with two shared (batch) dimensions: the batch dimension, followed by the time dimension.
    It also contains a time axis.
    """

    def __init__(self, x: Union[Tensor, TensorDict], time_axis: Tensor = None):
        """Initialize system state from Tensor or TensorDict input.

        Args:
            x (Union[Tensor, TensorDict]): Input data describing fields at each time point. If x is a tensor, it will be
             converted to a TensorDict with a single field "x"
            time_axis (Tensor, optional): Time values for each system state. Should match the second shared dimension of
             x. If ommited, will be populated with 64 bit signed integers starting at 0.
        """
        if isinstance(x, Tensor):
            assert x.ndim >= 2, "State tensor must have batch and time dimensions"
            x = TensorDict(x=x, batch_size=x.shape[:2])
        else:
            assert isinstance(x, TensorDict), "x must be Tensor or TensorDict"
            for v in x.values():
                assert isinstance(v, Tensor), "nested TensorDicts are not supported"
                assert v.ndim >= 2, "each field must have batch and time dimensions"
            if x.ndim < 2:
                x.batch_size = v.shape[:2]
            else:
                assert x.ndim == 2, f"invalid shape {x.ndim}, with batch size {x.batch_size}"

        assert isinstance(x, TensorDict), "x must be Tensor or TensorDict"
        self.time_axis = check_time_axis(time_axis, x.shape[1])
        self.fields = x

        self.shape = dict()
        for k, v in self.fields.items():
            self.shape[k] = v.shape

    def n_fields(self):
        """Returns the number of fields in the 'fields' attribute of this State object."""
        return len(self.fields.keys())

    def restrict_time_domain(self, tmin, tmax):
        """Returns a new State with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        if len(self.time_axis) == 1 and self.time_axis[0] >= tmin and self.time_axis[0] <= tmax:
            return self
        else:
            s, e = sorted_index_range(self.time_axis, tmin, tmax)
            new_fields = TensorDict()
            for key in self.fields.keys():
                new_fields[key] = self.fields[key][:, s:e, ...]
            return State(new_fields, self.time_axis[s:e])

    def restrict_fields(self, fields):
        """Returns a new State with a subset of the original fields.

        Args:
            fields(iterable): a subset of the keys corresponding to the State's fields
        """
        new_fields = TensorDict()
        for key in fields:
            new_fields[key] = self.fields[key]
        return State(new_fields, self.time_axis)

    def expand_time(self, time_axis):
        """Returns a new state with an expanded batch size and time axis."""
        assert self.fields.batch_size[1] == 1, "only a State with a single time step can be expanded in time"
        B, T = self.fields.batch_size[0], time_axis.nelement()
        new_state = self.clone()
        new_state.fields = new_state.fields.expand(B, T)
        new_state.time_axis = check_time_axis(time_axis, T)
        return new_state.clone()

    def mean_on_batch_axis(self):
        "Returns a new State with a batch size of 1, corresponding to the average of all batch elements of the input"
        new_fields = TensorDict()
        for k, v in self.fields.items():
            new_fields[k] = v.mean(0, True)
        return State(new_fields, self.time_axis).clone()

    def detach(self):
        """Detach from computation graph. See torch.Tensor.detach().

        Returns:
            State: detached State.
        """
        return State(self.fields.detach(), self.time_axis.detach())

    def clone(self):
        """Clone State. See torch.Tensor.clone().

        Returns:
            State: cloned State.
        """
        return State(self.fields.clone(), self.time_axis.clone())

    def to(self, *args, **kwargs):
        """Returns a copy of the State with its tensordict set as specified by the arguments."""
        return State(self.fields.to(*args, **kwargs), self.time_axis)

    def requires_grad_(self, requires_grad=True):
        """Returns a copy of the State object in which the chosen grad required is applied to all attributes."""
        return State(self.fields.requires_grad_(requires_grad), self.time_axis.requires_grad_(requires_grad))

    def fill_(self, val: Union[float, Dict]):
        """Fill values of state trajectory. See fill_ methods of Tensor and TensorDict.

        Args:
            val (Union[float, Dict]): Value to fill in. Can be a dict with an entry for each field, or a single float.

        Returns:
            State: State with filled in values.
        """
        if isinstance(val, float):
            for k in self.fields.keys():
                self.fields[k][...] = val
        elif isinstance(val, dict):
            for k in val:
                self.fields[k][...] = val[k]
        return self

    def __getitem__(self, key):
        """Indexing operator for State.

        Args:
            key: indexing key

        Returns:
            State: indexed key
        """
        if isinstance(key, tuple):
            if len(key) == 1:  # batch dimension
                return State(self.fields[key], self.time_axis)
            elif len(key) == 2:  # batch and time dimensions
                return State(self.fields[key], self.time_axis[key[1]])
            else:
                raise KeyError("invalid key")
        elif isinstance(key, int):  # batch dimension
            raise TypeError(
                "a State requires a batch dimension and cannot be indexed with a single int, use a range instead"
            )
        elif isinstance(key, slice) or key is Ellipsis:
            return State(self.fields[key], self.time_axis)
        elif isinstance(key, Tensor) and key.dtype is torch.bool:  # binary mask on batch dimension
            assert key.ndim == 1, "binary masking along batch dimension only"  # allow time also?
            assert key.nelement() == self.fields.batch_size[0], "size mismatch"
            return State(self.fields[key], self.time_axis)
        else:
            raise KeyError(f"invalid key: {key}")

    def __setitem__(self, key, value):
        """Indexed assignement operator for State.

        Args:
            key: indexing key
            value: values to be assigned
        """
        if isinstance(key, tuple):
            if len(key) == 0 or len(key) > 2:
                raise KeyError("tuple key must have 1 or 2 entries")
        elif isinstance(key, Tensor) and key.dtype is torch.bool:
            if key.ndim == 0 or key.ndim > 2:
                raise KeyError("only 1 or 2 dims for logical index")
        elif not (isinstance(key, int) or isinstance(key, slice) or (key is Ellipsis)):
            raise KeyError("invalid key")
        self.fields[key] = value

    def __mul__(self, other):
        """Multiplication operator."""
        if isinstance(other, State):
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            return State(self.fields * other.fields, self.time_axis)
        elif isnumber(other):
            return State(self.fields * other, self.time_axis)
        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """Right multiplication."""
        if isnumber(other) or isinstance(other, Tensor):
            return State(self.fields * other, self.time_axis)
        else:
            raise NotImplementedError()

    def __matmul__(self, other):
        """Matrix multiplication of the fields of the State.

        If other is a State, then it must have the same field names and same time axis as self. The result will be a new
        State with the same time axis and from which all fields are matrix multiplications of the corresponding fields
        of self and other.

        If other is a matrix, then all fields in self are right-multiplied by other.

        In all cases, the batch and time dimensions of the fields tensordict are batch dimensions in the multiplication,
        so they should remain unchanged.
        """
        for key in self.fields.keys():
            assert (
                len(self.fields[key].shape) >= 4
            ), "Each field must have at least 2 dimensions so that batch and time can"
            " be considered as batch dimensions in the multiplication."

        new_fields = TensorDict()
        if isinstance(other, State):
            assert (
                self.n_fields() == other.n_fields()
            ), f"The two States have respectively {self.n_fields()} and {other.n_fields()} fields"
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            for k in self.fields.keys():
                new_fields[k] = self.fields[k] @ other.fields[k]
        elif isinstance(other, Tensor):
            for k in self.fields.keys():
                new_fields[k] = self.fields[k] @ other
        else:
            raise TypeError("a State can only be mat-multiplied by a State or a Tensor")
        new_state = State(new_fields, self.time_axis)
        assert (
            new_state.fields.batch_size == self.fields.batch_size
        ), f"Batch dimensions have changed from {self.fields.batch_size} to {new_state.fields.batch_size}"
        return new_state

    def __div__(self, other):
        """Division operator."""
        if isinstance(other, State):
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            return State(self.fields / other.fields, self.time_axis)
        elif isnumber(other):
            return State(self.fields / other, self.time_axis)
        else:
            raise NotImplementedError()

    def __rdiv__(self, other):
        """Right-to-left division."""
        if isnumber(other):
            return State(other / self.fields, self.time_axis)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        """Addition operator."""
        if isinstance(other, State):
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            return State(self.fields + other.fields, self.time_axis)
        elif isnumber(other):
            return State(self.fields + other, self.time_axis)
        elif isinstance(other, Tensor):
            assert self.n_fields() == 1, "only States with one field can be summed to tensors"
            key = list(self.fields.keys())[0]
            return State(self.fields + TensorDict({key: other}), self.time_axis)
        else:
            raise NotImplementedError()

    def __radd__(self, other):
        "Right addition"
        if isnumber(other):
            return State(self.fields + other, self.time_axis)
        else:
            raise NotImplementedError()

    def __sub__(self, other):
        """Subtraction operator."""
        if isinstance(other, State):
            assert (self.time_axis == other.time_axis).all(), "time axes must match"
            return State(self.fields - other.fields, self.time_axis)
        if isnumber(other):
            return State(self.fields - other, self.time_axis)

    def __rsub__(self, other):
        """Right subtraction."""
        if isnumber(other):
            return State(other - self.fields, self.time_axis)

    def __pow__(self, other):
        """Exponentiation operator."""
        if isnumber(other):
            return State(self.fields**other, self.time_axis)
        else:
            raise NotImplementedError()

    def cat(self, other, dim):
        """Concatenate this State with another State, either in the batch dim (0) or the time dim (1)"""
        assert isinstance(other, State), "Only a State can be concatenated to a State"
        assert dim in [0, 1], "State objects can only be concatenated over dimension 0 or 1"
        if dim == 0:
            return State(tensordict.cat([self.fields, other.fields], dim), self.time_axis)
        else:
            return State(tensordict.cat([self.fields, other.fields], dim), torch.cat([self.time_axis, other.time_axis]))

    def __repr__(self) -> str:
        """Defines how to print a State object."""
        string_fields = indent(f"Fields: {self.fields}", 4 * " ")
        string_time_axis = indent(f"Time axis: {self.time_axis}", 4 * " ")
        return f"{type(self).__name__}(\n{string_fields}\n{string_time_axis})"

    @property
    def device(self):
        """Device of the fields."""
        return self.fields.device


def check_time_axis(time_axis, T):
    """Initialize time_axis if none, otherwise check validity."""
    if time_axis is None:
        return torch.arange(T)
    else:
        assert isinstance(time_axis, Tensor), f"{time_axis} is not a tensor"
        assert time_axis.ndim == 1, f"time axis must have one dimension, got shape {time_axis.shape}"
        assert time_axis.nelement() == T, f"time_axis has {time_axis.nelement()} elements, should have T={T}"
        if T == 0:
            warnings.warn("Instantiating an empty State")
        return time_axis
