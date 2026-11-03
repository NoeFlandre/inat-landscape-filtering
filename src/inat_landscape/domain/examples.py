"""Example photos for each score band: the mosaic of the card shows what each band keeps."""

import math
import random
from collections.abc import Sequence
from itertools import pairwise

Band = tuple[float, float]
EXAMPLE_LICENSE = "CC0"  # no attribution needed on the mosaic
_BELOW = 2  # bands below the lowest threshold, to show what the thresholds remove


def score_bands(thresholds: Sequence[float]) -> list[Band]:
    """Return the score bands of the mosaic, lowest first.

    The edges are the thresholds, plus two steps below the lowest one. A step is the gap of
    the two lowest thresholds, or half of the only threshold (a hundredth if it is 0). The
    added edges are rounded to 9 decimals, to remove the float noise of the subtraction.
    """
    if not thresholds:
        raise ValueError("the mosaic needs at least one threshold")
    edges = sorted(thresholds)
    step = edges[1] - edges[0] if len(edges) > 1 else abs(edges[0]) / 2 or 0.01
    below = [round(edges[0] - k * step, 9) for k in range(_BELOW, 0, -1)]
    points = [-math.inf, *below, *edges, math.inf]
    return list(pairwise(points))


def band_label(band: Band) -> str:
    """Return the label of a band, for example `0.02 to 0.04`, `< 0.00` or `≥ 0.08`."""
    lo, hi = band
    if lo == -math.inf:
        return f"< {_number(hi)}"
    if hi == math.inf:
        return f"≥ {_number(lo)}"
    return f"{_number(lo)} to {_number(hi)}"


def _number(value: float) -> str:
    return f"{value:.2f}" if round(value, 2) == value else f"{value:g}"


def pick_examples(
    ids: Sequence[int],
    scores: Sequence[float | None],
    licenses: Sequence[str],
    bands: Sequence[Band],
    per_band: int,
    seed: int = 0,
) -> list[list[int]]:
    """Return up to `per_band` CC0 photo ids for each band, sorted, the same for one seed."""
    rng = random.Random(seed)  # noqa: S311 - a fixed choice of example photos, not security
    return [_sample(m, per_band, rng) for m in _members(ids, scores, licenses, bands)]


def _members(
    ids: Sequence[int],
    scores: Sequence[float | None],
    licenses: Sequence[str],
    bands: Sequence[Band],
) -> list[list[int]]:
    """Return the sorted CC0 photo ids of each band."""
    members: list[list[int]] = [[] for _ in bands]
    for photo_id, score, license_ in sorted(zip(ids, scores, licenses, strict=True)):
        index = _band_index(score, bands) if license_ == EXAMPLE_LICENSE else None
        if index is not None:
            members[index].append(photo_id)
    return members


def _band_index(score: float | None, bands: Sequence[Band]) -> int | None:
    if score is None:
        return None
    return next((i for i, (lo, hi) in enumerate(bands) if lo <= score < hi), None)


def _sample(ids: list[int], size: int, rng: random.Random) -> list[int]:
    return sorted(rng.sample(ids, min(size, len(ids))))
