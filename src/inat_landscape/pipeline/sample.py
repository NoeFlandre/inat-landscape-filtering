"""Sample stage: take the first n candidate photos in uniform order (ADR 0011)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.domain.sampling import uniform_order, uuid_key
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ingest import CANDIDATES

SAMPLE = "sample.parquet"
SAMPLE_IDS = "sample-ids.parquet"  # photo_id and extension only: all that the embed jobs read

__all__ = ["SAMPLE", "SAMPLE_IDS", "StageError", "run_sample"]


def run_sample(workdir: Path, n: int | None) -> pa.Table:
    """Write and return the first `n` candidates in uniform order (all when n is None).

    The uniform order takes one photo of each H3 cell, then a second one, and so on. The
    embed stage works in this order, so a stop at any time leaves about the same count of
    photos in each cell. `cell_rank` is the place of a photo in its cell.
    """
    path = workdir / CANDIDATES
    if not path.exists():
        raise StageError(f"no candidates at {path}: run `ingest` first")
    if n is not None and n < 0:
        raise StageError(f"n must be >= 0, got {n}")
    table = read_table(path)
    ids, ranks = uniform_order(*_columns(table))
    ids, ranks = ids[:n], ranks[:n]
    sample = table.filter(pc.is_in(table["photo_id"], pa.array(ids)))
    sorter = np.argsort(ids)
    where = sorter[np.searchsorted(ids[sorter], sample["photo_id"].to_numpy())]  # place in ids
    sample = sample.append_column("sample_rank", pa.array(where))
    sample = sample.append_column("cell_rank", pa.array(ranks[where]))
    sample = sample.sort_by("sample_rank")
    write_table(sample, workdir / SAMPLE)
    write_table(sample.select(["photo_id", "extension"]), workdir / SAMPLE_IDS)
    return sample


def _columns(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.array([uuid_key(u) for u in table["observation_uuid"].to_pylist()], np.uint64)
    positions = pc.fill_null(table["position"], 0).to_numpy()  # missing position sorts first
    cells = pc.dictionary_encode(table["h3_cell"]).combine_chunks().indices.to_numpy()
    return keys, positions.astype(np.int64), table["photo_id"].to_numpy(), cells.astype(np.int64)
