"""Score thresholds and the dataset configs they define."""

_PREFIXES = {"score_landscape": "score_ge_", "score_contrast": "contrast_ge_"}


def config_name(threshold: float, score: str = "score_landscape") -> str:
    """Return the config name for a threshold on a score, for example `score_ge_0_30`."""
    if score not in _PREFIXES:
        raise ValueError(f"no config prefix for score {score!r}")
    text = f"{threshold:.2f}"
    if float(text) != threshold:
        text = repr(threshold)
    return _PREFIXES[score] + text.replace(".", "_")


def parse_thresholds(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated list of thresholds in [0, 1]."""
    try:
        values = {float(part) for part in raw.split(",")}
    except ValueError:
        raise ValueError(f"bad threshold list: {raw!r}") from None
    if not all(0 <= v <= 1 for v in values):
        raise ValueError(f"each threshold must be in [0, 1]: {raw!r}")
    return tuple(sorted(values))
