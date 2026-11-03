"""Deterministic sampling based on the random part of UUID v4 identifiers.

An observation is in the sample when the first 8 hex digits of its UUID are below a
prefix derived from the sample fraction. A larger fraction always keeps a superset.
"""

import numpy as np
from numpy.typing import NDArray

_KEY_DIGITS = 8
_KEY_SPACE = 16**_KEY_DIGITS
_KEEP_ALL = "g"  # sorts after every lowercase hex digit string


def sample_prefix(fraction: float) -> str:
    """Return the exclusive upper bound for the UUID key of a sample of `fraction`."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    bound = int(fraction * _KEY_SPACE)
    return _KEEP_ALL if bound >= _KEY_SPACE else f"{bound:0{_KEY_DIGITS}x}"


def in_sample(observation_uuid: str, prefix: str) -> bool:
    """Tell if an observation is in the sample defined by `prefix`."""
    return observation_uuid[:_KEY_DIGITS].lower() < prefix


def uuid_key(observation_uuid: str) -> int:
    """Return the 64-bit sort key of an observation: the first 16 hex digits of its UUID."""
    return int(observation_uuid.replace("-", "")[:16], 16)


def uniform_order(
    keys: NDArray[np.uint64],
    positions: NDArray[np.int64],
    photo_ids: NDArray[np.int64],
    cells: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the photo ids in uniform order, and the rank of each photo in its cell.

    The sample order is (observation key, photo position, photo id). Because UUID v4 values
    are random, it is a random order. The rank of a photo is its place in the sample order of
    its H3 cell. The uniform order is (rank, sample order): first one photo of each cell, then
    a second one, and so on. Thus the first n photos have the same count in each cell (one
    more in some cells), as far as the cells have photos. A larger n gives a superset.
    """
    order = np.lexsort((photo_ids, positions, keys))
    ranks = np.empty_like(order)
    ranks[order] = _ranks_in_groups(cells[order])
    uniform = np.lexsort((np.argsort(order), ranks))
    return photo_ids[uniform], ranks[uniform]


def _ranks_in_groups(groups: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return the place of each item among the earlier items of its group."""
    by_group = np.lexsort((np.arange(len(groups)), groups))  # stable: earlier items first
    sorted_groups = groups[by_group]
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    sizes = np.diff(np.r_[starts, len(groups)])
    ranks = np.empty_like(by_group)
    ranks[by_group] = np.arange(len(groups)) - np.repeat(starts, sizes)
    return ranks


def for_worker(photo_id: int, worker: int, workers: int) -> bool:
    """Tell if a photo belongs to `worker` of `workers` parallel workers."""
    if not 0 <= worker < workers:
        raise ValueError(f"worker must be in [0, {workers}), got {worker}")
    return photo_id % workers == worker
