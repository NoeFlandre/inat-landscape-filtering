"""Errors that the CLI reports as a clean message with exit code 1."""


class StageError(RuntimeError):
    """A stage cannot run, for example because an earlier stage did not run."""
