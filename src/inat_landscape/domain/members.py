"""Fast set membership for large id arrays."""

import numpy as np
from numpy.typing import NDArray


def in_sorted(values: NDArray, sorted_values: NDArray) -> NDArray[np.bool_]:
    """Tell which values are in `sorted_values`, by binary search.

    np.isin sorts both arrays at each call: with tens of millions of values to look in,
    sort them once and call this for each batch.
    """
    if not len(sorted_values):
        return np.zeros(len(values), dtype=bool)
    at = np.minimum(np.searchsorted(sorted_values, values), len(sorted_values) - 1)
    return sorted_values[at] == values
