from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from adda.system.state import State
from adda.util.index import sorted_index_range
from adda.util.mask import mask_state
from adda.util.state_space import same_shape
from tensordict import TensorDict
from torch import Tensor


class ObservationSet(ABC):
    """Base class for all observation set."""

    @abstractmethod
    def restrict_time_domain(self, tmin: float, tmax: float):
        """Returns a new ObservationSet with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def values(self) -> TensorDict:
        """Returns all observed values for all fields, in the form of a TensorDict with no batch dimensions.

        Returns:
            TensorDict: observed values
        """
        raise NotImplementedError


class FullStateObservationSet(ObservationSet):
    """ObservationSet in which all variables are observed at all time points."""

    def __init__(self, state: State):
        """
        Args:
            state (State): State object containing observed values
        """
        self.state = state

    def restrict_time_domain(self, tmin, tmax):
        """Returns a new FullStateObservationSet with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        return FullStateObservationSet(self.state.restrict_time_domain(tmin, tmax))

    @property
    def values(self):
        """Returns all observed values for all fields, in the form of a TensorDict with no batch dimensions.

        Returns:
            TensorDict: observed values
        """
        return self.state.fields


class MaskedStateObservationSet(ObservationSet):
    """ObservationSet in which only some variables are observed."""

    def __init__(self, state: State, mask: State, masked_value=torch.nan):
        """Args:
        state (State): State object containing observed values
        mask (State): State object containing boolean masks for each field in state
        masked_value: The value to give to variables that are masked out. Defaults to torch.nan
        """
        assert same_shape(state.fields, mask.fields), "state/mask mismatch"
        if masked_value is not None:
            for key in state.fields.keys():
                if torch.is_floating_point(state.fields[key]):
                    state.fields[key][torch.logical_not(mask.fields[key])] = masked_value
                else:
                    state.fields[key][torch.logical_not(mask.fields[key])] = 0
        self.masked_value = masked_value
        self.state, self.mask = state, mask

    def restrict_time_domain(self, tmin, tmax):
        """Returns a new MaskedStateObservationSet with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        return MaskedStateObservationSet(
            self.state.restrict_time_domain(tmin, tmax), self.mask.restrict_time_domain(tmin, tmax)
        )

    def restrict_fields(self, fields):
        """Returns a new MaskedStateObservationSet with a subset of the original fields.

        Args:
            fields(iterable): a subset of the keys corresponding to the state's fields
        """
        return MaskedStateObservationSet(self.state.restrict_fields(fields), self.mask.restrict_fields(fields))

    def to(self, *args, **kwargs):
        """Returns a copy of the MaskedStateObservationSet with its tensordict set as specified by the arguments."""
        return MaskedStateObservationSet(
            self.state.to(*args, **kwargs), self.mask.to(*args, **kwargs), self.masked_value
        )

    @property
    def values(self):
        """Returns all observed values for all fields, in the form of a TensorDict with no batch dimensions.

        Returns:
            TensorDict: observed values
        """
        return mask_state(self.state, self.mask)


class MultipleObservationSet(ObservationSet):
    """ObservationSet consisting of multiple observation types."""

    def __init__(self, observationsets: List[ObservationSet]):
        """
        Args:
            observationsets (List[ObservationSet]): ObservationSet objects to be combined
        """
        self.observationsets = observationsets
        raise NotImplementedError

    def restrict_time_domain(self, tmin, tmax):
        """Returns a new MultipleObservationSet with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        return MultipleObservationSet([o.restrict_time_domain(tmin, tmax) for o in self.observationsets])

    @property
    def values(self):
        """Returns all observed values for all fields, in the form of a TensorDict with no batch dimensions.

        Returns:
            TensorDict: observed values
        """
        raise NotImplementedError


class GriddedPointObservationSet(ObservationSet):
    """ObservationSet consisting of point observations on a grid."""

    def __init__(
        self, x: TensorDict, time_axis: Tensor, idx_t: TensorDict, idx_coords: TensorDict, idx_batch: TensorDict = None
    ):
        """Args:
        x (TensorDict): TensorDict whose fields have rows for each observation
        time_axis (Tensor): time axis of gridded state sequence that has been observed
        idx_t (TensorDict): indices into time_axis for each row of each field of x. Different numbers of observations
          for each field are possible
        idx_coords (TensorDict): indices into grid coordinates for each row of each field of x
        idx_batch (TensorDict, optional): batch indices (0-based indexing) for each row of each field of x. Defaults to
          all zeros if omitted.
        """
        assert isinstance(time_axis, Tensor) and time_axis.ndim == 1, "time_axis must be a 1D tensor"
        assert x.keys() == idx_t.keys() == idx_coords.keys(), "key mismatch"
        idx_batch = torch.zeros_like(idx_t, dtype=int) if idx_batch is None else idx_batch
        assert (
            x.ndim == idx_t.ndim == idx_coords.ndim == idx_batch.ndim == 0
        ), "TensorDicts should not have any batch axes"
        for key, field in x.items():
            assert (
                idx_t[key].ndim == 1
                and idx_batch[key].ndim == 1
                and field.shape[0] == idx_t[key].shape[0] == idx_coords[key].shape[0] == idx_batch[key].shape[0]
            ), f"shape mismatch for field {key}"
            assert (
                torch.all(idx_t[key] < len(time_axis))
                and torch.all(idx_t[key] >= 0)
                and idx_t[key].dtype in [torch.int64, torch.int32]
            ), "invalid time indices"
        self.idx_t, self.idx_coords, self.x, self.time_axis, self.idx_batch = idx_t, idx_coords, x, time_axis, idx_batch

    def restrict_time_domain(self, tmin, tmax):
        """Returns a new GriddedPointObservationSet with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        s, e = sorted_index_range(self.time_axis, tmin, tmax, nonempty=True)
        x, idx_t, idx_coords, idx_batch = TensorDict(), TensorDict(), TensorDict(), TensorDict()
        for key, field in self.x.items():
            ii = torch.logical_and(self.idx_t >= s, self.idx_t < e)
            x[key], idx_t[key], idx_coords[key], idx_batch[key] = (
                field[ii, :],
                self.idx_t[ii],
                self.idx_coords[ii, :],
                self.idx_batch[ii],
            )
        return GriddedPointObservationSet(x, self.time_axis[s:e], idx_t, idx_coords, (tmin, tmax), idx_batch)

    @property
    def values(self):
        """Returns all observed values for all fields, in the form of a TensorDict with no batch dimensions.

        Returns:
            TensorDict: observed values
        """
        return self.x


class PointObservationSet(ObservationSet):
    """ObservationSet consisting of observations at points freely varying through time and space, not snapped to a
    grid."""

    def __init__(
        self, x: TensorDict, t: TensorDict, coords: TensorDict, valid_time_interval: Tuple, idx_batch: TensorDict = None
    ):
        """Args:
        x (TensorDict): TensorDict whose fields have rows for each observation
        t (TensorDict): time coordinates each row of each field of x. Different numbers of observations for each field
          are possible
        valid_time_interval (Tuple): minimum and maximum times for which observation data were collected.
        idx_batch (TensorDict, optional): batch indices (0-based indexing) for each row of each field of x. Defaults to
          all zeros if omitted.
        """
        assert x.keys() == t.keys() == coords.keys(), "key mismatch"
        assert len(valid_time_interval) == 2, "invalid interval"
        idx_batch = torch.zeros_like(t, dtype=int) if idx_batch is None else idx_batch
        assert x.ndim == t.ndim == coords.ndim == idx_batch.ndim == 0, "TensorDicts should not have any batch axes"
        for key, field in x.items():
            assert (
                t[key].ndim == 1
                and idx_batch[key].ndim == 1
                and field.shape[0] == t[key].shape[0] == coords[key].shape[0] == idx_batch[key].shape[0]
            ), f"shape mismatch for field {key}"
            assert torch.all(t[key] < valid_time_interval[1]) and torch.all(t[key] >= valid_time_interval[0])
        self.t, self.coords, self.x, self.valid_time_interval = t, x, coords, valid_time_interval

    def restrict_time_domain(self, tmin, tmax):
        """Returns a new PointObservationSet with a restricted time axis.

        Args:
            tmin (float): beginning of new time domain
            tmax (float): end of new time domain
        """
        assert (
            tmin >= self.valid_time_interval[0] and tmax <= self.valid_time_interval[1] and tmin < tmax
        ), "invalid time interval"

        x, t, coords, idx_batch = TensorDict(), TensorDict(), TensorDict(), TensorDict()
        for key, field in self.x.items():
            ii = torch.logical_and(self.t >= tmin, self.idx_t < tmax)
            x[key], t[key], coords[key], idx_batch[key] = (
                field[ii, :],
                self.t[ii],
                self.coords[ii, :],
                self.idx_batch[ii],
            )
        return PointObservationSet(x, t, coords, (tmin, tmax), idx_batch=idx_batch)

    @property
    def values(self):
        """Returns all observed values for all fields, in the form of a TensorDict with no batch dimensions.

        Returns:
            TensorDict: observed values
        """
        return self.x
