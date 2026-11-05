"""Caps on the number of photos per geographic cell, for wide and even coverage."""

import numpy as np
from numpy.typing import NDArray

Ints = NDArray[np.integer]
MAP_RES = 4  # the maps show H3 cells of resolution 4, each with the mean of its finer cells


def cap_first(cells: Ints, keys: Ints, positions: Ints, photo_ids: Ints, cap: int) -> Ints:
    """Return the sorted row indices of the first `cap` photos of each cell.

    "First" is the sample order: (observation key, photo position, photo id). The keys are
    random, so the kept photos are a random sample of the cell. A larger cap keeps a superset.
    """
    _check(cap)
    order = np.lexsort((photo_ids, positions, keys, cells))
    return np.sort(order[_rank_in_group(cells[order]) < cap])


def top_per_cell(cells: Ints, scores: NDArray[np.floating], cap: int) -> NDArray[np.bool_]:
    """Return a mask of the `cap` best-scored photos of each cell (ties: first row first)."""
    _check(cap)
    order = np.lexsort((np.arange(len(cells)), -scores, cells))
    keep = np.zeros(len(cells), dtype=bool)
    keep[order[_rank_in_group(cells[order]) < cap]] = True
    return keep


def group_mean(groups: Ints, values: NDArray[np.floating]) -> tuple[Ints, NDArray[np.float64]]:
    """Return the distinct groups (sorted) and the mean of the values (or rows) of each."""
    found, index = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(found), *values.shape[1:]))
    np.add.at(sums, index, values)
    counts = np.bincount(index)
    return found, sums / np.expand_dims(counts, tuple(range(1, values.ndim)))


def _rank_in_group(sorted_groups: Ints) -> Ints:
    """Return the position of each row inside its run of equal values."""
    n = len(sorted_groups)
    boundary = np.ones(n, dtype=bool)  # pragma: no mutate - any dtype gives the same nonzero
    boundary[1:] = sorted_groups[1:] != sorted_groups[:-1]
    starts = np.flatnonzero(boundary)
    lengths = np.diff(np.append(starts, n))
    return np.arange(n) - np.repeat(starts, lengths)


def _check(cap: int) -> None:
    if cap < 0:
        raise ValueError(f"cap must be >= 0, got {cap}")
