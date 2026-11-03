"""Sample stage: take the first n candidate photos in uniform order (ADR 0011)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.adapters.inat_dump import open_source, uuid_keys
from inat_landscape.domain.sampling import uniform_order
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ingest import CANDIDATES, IngestParams, Opener, run_index

SAMPLE = "sample.parquet"
SAMPLE_IDS = "sample-ids.parquet"  # photo_id and extension only: all that the embed jobs read

_IDS_SCHEMA = pa.schema([("photo_id", pa.int64()), ("extension", pa.string())])

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
    write_table(_ids(sample), workdir / SAMPLE_IDS)
    return sample


def _columns(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = uuid_keys(table["observation_uuid"])  # vectorized `uuid_key`
    positions = pc.fill_null(table["position"], 0).to_numpy()  # missing position sorts first
    cells = pc.dictionary_encode(table["h3_cell"]).combine_chunks().indices.to_numpy()
    return keys, positions.astype(np.int64), table["photo_id"].to_numpy(), cells.astype(np.int64)


def run_worklist(workdir: Path, params: IngestParams, *, opener: Opener = open_source) -> pa.Table:
    """Write and return `sample-ids.parquet` from the dump index, before the details pass.

    It has the same photos and order as the sample of the same parameters (ADR 0011), so the
    embed jobs can start one dump pass earlier.
    """
    index = run_index(workdir, params, opener=opener)
    selected = index.selected
    cells = pc.dictionary_encode(selected["h3_cell"]).combine_chunks().indices.to_numpy()
    ids, _ = uniform_order(
        selected["observation_key"].to_numpy(),
        selected["position"].to_numpy().astype(np.int64),
        selected["photo_id"].to_numpy(),
        cells.astype(np.int64),
    )
    order = pa.table({"photo_id": ids, "_place": np.arange(len(ids))})
    table = selected.select(["photo_id", "extension"]).join(order, "photo_id")
    table = _ids(table.sort_by("_place"))
    write_table(table, workdir / SAMPLE_IDS)
    return table


def _ids(table: pa.Table) -> pa.Table:
    """Return the photo ids and extensions (short texts: a 32-bit string column holds them)."""
    return table.select(["photo_id", "extension"]).cast(_IDS_SCHEMA)
