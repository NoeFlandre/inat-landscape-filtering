"""Errors that the CLI reports as a clean message with exit code 1."""

from pathlib import Path


class StageError(RuntimeError):
    """A stage cannot run, for example because an earlier stage did not run."""


def require(path: Path, what: str, stage: str) -> None:
    """Fail if the earlier stage `stage` did not write `path` (its `what`)."""
    if not path.exists():
        raise StageError(f"no {what} at {path}: run `{stage}` first")
