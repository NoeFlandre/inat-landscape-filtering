"""The cached scores of each embedding shard (ADR 0012)."""

from pathlib import Path

import numpy as np
import pyarrow as pa

from inat_landscape.adapters.embedding_store import Pruned, shard_name
from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.domain.members import in_sorted


def carry_scores(cache: Path, pruned: list[Pruned], keep: np.ndarray) -> int:
    """Give the cached scores of each pruned shard to its new shard: the rows of the photos in
    the sorted `keep`. The old cache files go. Return how many cache files were carried."""
    folders = [f for f in cache.iterdir() if f.is_dir()] if cache.exists() else []
    return sum(_carry(folder, pruned, keep) for folder in folders)


def _carry(folder: Path, pruned: list[Pruned], keep: np.ndarray) -> int:
    carried = 0
    for p in pruned:
        old = folder / shard_name(p.path)
        if not old.exists():
            continue
        if p.new:
            table = read_table(old)
            kept = table.filter(pa.array(in_sorted(table["photo_id"].to_numpy(), keep)))
            write_table(kept, folder / shard_name(p.new))
            carried += 1
        old.unlink()
    return carried
