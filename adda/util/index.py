import torch
from torch import Tensor


def sorted_index_range(sorted_1Darray: Tensor, min_val, max_val, nonempty: bool = False):
    """Find indices into sorted array that fall into a chosen interval.

    Args:
        sorted_1Darray (Tensor): sorted 1D Tensor
        min_val (_type_): left boundary of interval
        max_val (_type_): right boundary of interval
        nonempty (bool, optional): Throws an error if not indices are found. Defaults to False.

    Returns:
        idx_start: maximum index such that sorted_1Darray[idx_start] <= min_val, or else len(sorted_1darray)
        idx_end: maximum index such that sorted_1Darray[idx_end] <= max_val, or else len(sorted_1darray)
    """
    assert max_val > min_val, "range of values must be strictly increasing"
    idx_start = torch.searchsorted(sorted_1Darray, min_val, right=False)
    idx_end = torch.searchsorted(sorted_1Darray, max_val, right=False)
    if nonempty:
        assert idx_end > idx_start, "unexpected empty index range"

    return idx_start, idx_end
