"""Plain-text summaries of score distributions."""

from collections.abc import Sequence
from statistics import quantiles


def histogram(scores: Sequence[float], bins: int) -> list[int]:
    """Count scores in `bins` equal bins over [0, 1]. The last bin includes 1."""
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    counts = [0] * bins
    for s in scores:
        counts[min(max(int(s * bins), 0), bins - 1)] += 1
    return counts


def summarize(scores: Sequence[float], thresholds: Sequence[float], bins: int = 20) -> str:
    """Return a text report: quantiles, counts above thresholds and a histogram."""
    if not scores:
        raise ValueError("no scores to summarize")
    lines = [f"n={len(scores)} min={min(scores):.4f} max={max(scores):.4f}", _quantiles(scores)]
    lines += [f">= {t}: {sum(s >= t for s in scores)}" for t in thresholds]
    return "\n".join(lines + _bars(histogram(scores, bins)))


def key_quantiles(scores: Sequence[float]) -> tuple[tuple[str, float], ...]:
    """Return the median, p90, p99 and maximum of the scores."""
    qs = quantiles(scores, n=100, method="inclusive")
    return (("Median", qs[49]), ("p90", qs[89]), ("p99", qs[98]), ("Max", max(scores)))


def _quantiles(scores: Sequence[float]) -> str:
    qs = quantiles(scores, n=100, method="inclusive")
    return "quantiles: " + " ".join(f"p{p}={qs[p - 1]:.4f}" for p in (50, 90, 95, 99))


def _bars(counts: list[int]) -> list[str]:
    width = 1 / len(counts)
    return [
        f"[{i * width:.2f}, {(i + 1) * width:.2f}] {c:>7} " + "#" * min(c, 60)
        for i, c in enumerate(counts)
    ]


def cap_report(counts: Sequence[int], caps: Sequence[int]) -> str:
    """Return a text report of photos per cell, and of the photos kept under each cap."""
    lines = [f"cells={len(counts)} photos={sum(counts)} max_per_cell={max(counts, default=0)}"]
    lines += [
        f"cap {cap}: {sum(min(c, cap) for c in counts)} photos, "
        f"{sum(c >= cap for c in counts)} cells at the cap"
        for cap in caps
    ]
    return "\n".join(lines)
