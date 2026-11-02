"""Deterministic sampling based on the random part of UUID v4 identifiers.

An observation is in the sample when the first 8 hex digits of its UUID are below a
prefix derived from the sample fraction. A larger fraction always keeps a superset.
"""

from dataclasses import dataclass

_KEY_DIGITS = 8
_KEY_SPACE = 16**_KEY_DIGITS
_KEEP_ALL = "g"  # sorts after every lowercase hex digit string


@dataclass(frozen=True, slots=True)
class Candidate:
    """One photo that can enter the sample."""

    observation_uuid: str
    position: int
    photo_id: int


def sample_prefix(fraction: float) -> str:
    """Return the exclusive upper bound for the UUID key of a sample of `fraction`."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    bound = int(fraction * _KEY_SPACE)
    return _KEEP_ALL if bound >= _KEY_SPACE else f"{bound:0{_KEY_DIGITS}x}"


def in_sample(observation_uuid: str, prefix: str) -> bool:
    """Tell if an observation is in the sample defined by `prefix`."""
    return observation_uuid[:_KEY_DIGITS].lower() < prefix


def select(candidates: list[Candidate], n: int) -> list[int]:
    """Return the photo ids of the first `n` candidates in sample order.

    The order is (observation UUID, photo position, photo id). Because UUID v4 values are
    random, the first `n` photos are a random sample, and a larger `n` gives a superset.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return [c.photo_id for c in sorted(candidates, key=_order)[:n]]


def _order(c: Candidate) -> tuple[str, int, int]:
    # lower() and upper() give the same order on hex digits: that mutant is equivalent.
    return (c.observation_uuid.lower(), c.position, c.photo_id)  # pragma: no mutate
