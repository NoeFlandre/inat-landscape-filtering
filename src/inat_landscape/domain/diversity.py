"""Visual diversity of a set of image embeddings, and a colour for what they show.

The mean cosine distance between the photos of a cell needs only the sum of their unit
vectors: for n unit vectors with sum s, the sum of x_i . x_j over the pairs i != j is
|s|^2 - n. So a cell is a sum and a count, and the shards can be read in any order.
"""

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.floating]
Ints = NDArray[np.integer]


def unit(vectors: Floats) -> NDArray[np.float32]:
    """Return the vectors scaled to length 1 (a zero vector stays zero)."""
    v = np.asarray(vectors, np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, np.finfo(np.float32).tiny)  # a zero row: 0 / tiny = 0


def group_sums(groups: Ints, vectors: Floats) -> tuple[Ints, Floats, Ints]:
    """Return each group that has rows, the sum of its vectors (float64) and its row count."""
    if not len(groups):
        return groups, np.zeros((0, *np.shape(vectors)[1:])), np.zeros(0, np.int64)
    order = np.argsort(groups)
    sorted_groups = groups[order]
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    sums = np.add.reduceat(np.asarray(vectors, np.float64)[order], starts)
    counts = np.diff(np.r_[starts, len(groups)])
    return sorted_groups[starts], sums, counts


def mean_pairwise_distance(sums: Floats, counts: Ints) -> Floats:
    """Return the mean cosine distance between the unit vectors of each group, from their sum
    and count. A group of fewer than 2 vectors gives NaN."""
    n = np.asarray(counts)
    dots = (sums * sums).sum(axis=1) - n  # sum of x_i . x_j over the pairs i != j
    mean = np.divide(dots, n * (n - 1), out=np.full(len(n), np.nan), where=n > 1)
    return 1 - mean


def principal_components(vectors: Floats, k: int = 3) -> Floats:
    """Return the first k principal components of the rows (the projections), with the sign
    chosen so that the largest loading of each axis is positive (a stable result)."""
    x = vectors - vectors.mean(axis=0)
    values, axes = np.linalg.eigh(x.T @ x)
    axes = axes[:, np.argsort(values)[::-1][:k]]
    signs = np.sign(axes[np.abs(axes).argmax(axis=0), np.arange(axes.shape[1])])
    return x @ (axes * signs)  # pragma: no mutate - the signs are 1 or -1: * and / are equal


def to_unit_range(values: Floats, low: float = 2, high: float = 98) -> Floats:
    """Scale each column to [0, 1] between its `low` and `high` percentiles (clipped), so a
    few extreme cells do not wash out the colours. A constant column gives 0."""
    lo, hi = np.percentile(values, [low, high], axis=0)
    scaled = np.divide(values - lo, hi - lo, out=np.zeros(values.shape), where=hi > lo)
    return np.clip(scaled, 0, 1)
