"""Diversity stage: how varied the photos of each H3 cell look, and what they show.

One pass over the embedding shards (ADR 0014). Each cell keeps the sum and the count of the
unit embeddings of its first `per_cell` photos in the uniform order, which are a random
sample of the cell. The sums give the mean cosine distance between the photos of a cell,
and their means give 3 principal components: a colour for what the cell shows.
"""

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.adapters.hub import Hub
from inat_landscape.domain.diversity import (
    group_sums,
    mean_pairwise_distance,
    principal_components,
    unit,
)
from inat_landscape.domain.members import in_sorted
from inat_landscape.pipeline.embed import open_store, read_sample
from inat_landscape.pipeline.errors import StageError

DIVERSITY = "diversity.parquet"
PER_CELL = 50  # photos of each cell: the same count in each cell, so cells compare
MIN_PHOTOS = 20  # a cell with fewer embedded photos has no value


def run_diversity(
    workdir: Path,
    model_id: str,
    hub: Hub,
    *,
    per_cell: int = PER_CELL,
    min_photos: int = MIN_PHOTOS,
    workers: int = 1,
) -> pa.Table:
    """Write and return, for each cell with `min_photos` embedded photos at least: the cell,
    its photo count, its diversity and the 3 principal components of its mean embedding."""
    ids, cells = _chosen(workdir, per_cell)
    sums = _Sums(ids, cells)
    store = open_store(workdir, model_id, hub)
    store.each(sums.adder(store), workers)
    return _write(workdir, sums, min_photos, per_cell)


def _chosen(workdir: Path, per_cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted ids of the first `per_cell` photos of each cell, and their cells.
    A photo of two observations counts once."""
    sample = read_sample(workdir, ["photo_id", "h3_cell", "cell_rank"])
    first = sample.filter(pc.less(sample["cell_rank"], per_cell))
    ids, at = np.unique(first["photo_id"].to_numpy(), return_index=True)
    return ids, first["h3_cell"].to_numpy()[at]


class _Sums:
    """The sum and count of the unit embeddings of each cell, filled shard by shard."""

    def __init__(self, ids: np.ndarray, cells: np.ndarray) -> None:
        self.ids = ids
        self.cells, self.index = np.unique(cells, return_inverse=True)
        self.sums: np.ndarray | None = None
        self.counts = np.zeros(len(self.cells), np.int64)
        self.lock = threading.Lock()

    def adder(self, store: EmbeddingStore) -> Callable[[str], None]:
        """Return a function that adds one shard (thread-safe)."""

        def add(path: str) -> None:
            shard_ids, embeddings = store.read(path)
            shard_ids = np.asarray(shard_ids, np.int64)
            mask = in_sorted(shard_ids, self.ids)
            rows = self.index[np.searchsorted(self.ids, shard_ids[mask])]
            groups, sums, counts = group_sums(rows, unit(embeddings[mask]))
            with self.lock:
                if self.sums is None:
                    self.sums = np.zeros((len(self.cells), embeddings.shape[1]))
                self.sums[groups] += sums
                self.counts[groups] += counts

        return add


def _write(workdir: Path, sums: _Sums, min_photos: int, per_cell: int) -> pa.Table:
    need = max(min_photos, 2)  # one photo has no pair
    keep = sums.counts >= need
    if sums.sums is None or not keep.any():
        raise StageError(f"no cell has {need} embedded photos: run `embed` first")
    total, counts = sums.sums[keep], sums.counts[keep]
    pcs = principal_components(total / counts[:, None], k=3)
    table = pa.table(
        {
            "h3_cell": pa.array(sums.cells[keep]),
            "photos": pa.array(counts),
            "diversity": pa.array(mean_pairwise_distance(total, counts)),
            **{f"pc{i + 1}": pa.array(pcs[:, i]) for i in range(3)},
        }
    ).replace_schema_metadata(
        {b"per_cell": str(per_cell).encode(), b"min_photos": str(need).encode()}
    )
    write_table(table, workdir / DIVERSITY)
    return table


def read_diversity(workdir: Path) -> tuple[pa.Table, dict[str, int]] | None:
    """Return the diversity table and its parameters (`per_cell`, `min_photos`), or None if
    the stage did not run."""
    if not (workdir / DIVERSITY).exists():
        return None
    table = read_table(workdir / DIVERSITY)
    meta = table.schema.metadata or {}
    return table, {k.decode(): int(v) for k, v in meta.items() if k in (b"per_cell", b"min_photos")}
