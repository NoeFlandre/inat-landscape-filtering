"""Score thresholds and the dataset configs they define."""

from collections.abc import Sequence


def config_name(threshold: float) -> str:
    """Return the config name for a threshold, for example `score_ge_0_30`."""
    text = f"{threshold:.2f}"
    if float(text) != threshold:
        text = repr(threshold)
    return "score_ge_" + text.replace(".", "_")


def keep(scores: Sequence[float], threshold: float) -> list[bool]:
    """Tell which scores reach the threshold (inclusive)."""
    return [s >= threshold for s in scores]


def parse_thresholds(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated list of thresholds in [0, 1]."""
    try:
        values = {float(part) for part in raw.split(",")}
    except ValueError:
        raise ValueError(f"bad threshold list: {raw!r}") from None
    if not all(0 <= v <= 1 for v in values):
        raise ValueError(f"each threshold must be in [0, 1]: {raw!r}")
    return tuple(sorted(values))
