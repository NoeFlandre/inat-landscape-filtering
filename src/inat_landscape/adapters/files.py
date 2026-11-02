"""Local files: atomic Parquet and JSON writes, so a crash never leaves a partial file."""

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_table(table: pa.Table, path: Path) -> None:
    """Write a Parquet file atomically."""
    _atomic(path, lambda tmp: pq.write_table(table, tmp, compression="zstd"))


def read_table(path: Path, columns: list[str] | None = None) -> pa.Table:
    """Read a Parquet file, or only some of its columns."""
    return pq.read_table(path, columns=columns)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a JSON file atomically."""
    _atomic(path, lambda tmp: tmp.write_text(json.dumps(data, indent=2, sort_keys=True)))


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, or return None if it does not exist."""
    return json.loads(path.read_text()) if path.exists() else None


def _atomic(path: Path, write: Callable[[Path], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    tmp = Path(name)
    try:
        write(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_bytes(data: bytes, path: Path) -> None:
    """Write a binary file atomically."""
    _atomic(path, lambda tmp: tmp.write_bytes(data))
