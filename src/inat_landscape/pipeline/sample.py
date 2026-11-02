"""Sample stage: take the first n candidate photos in sample order."""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.domain.sampling import Candidate, select
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ingest import CANDIDATES

SAMPLE = "sample.parquet"

__all__ = ["SAMPLE", "StageError", "run_sample"]


def run_sample(workdir: Path, n: int) -> pa.Table:
    """Write and return the sample of `n` photos."""
    path = workdir / CANDIDATES
    if not path.exists():
        raise StageError(f"no candidates at {path}: run `ingest` first")
    table = read_table(path)
    chosen = select(_candidates(table), n)
    order = {photo_id: rank for rank, photo_id in enumerate(chosen)}
    sample = table.filter(pc.is_in(table["photo_id"], pa.array(chosen, pa.int64())))
    ranks = pa.array([order[i] for i in sample["photo_id"].to_pylist()], pa.int64())
    sample = sample.append_column("sample_rank", ranks).sort_by("sample_rank")
    write_table(sample, workdir / SAMPLE)
    return sample


def _candidates(table: pa.Table) -> list[Candidate]:
    positions = pc.fill_null(table["position"], 0).to_pylist()  # missing position sorts first
    columns = zip(
        table["observation_uuid"].to_pylist(), positions, table["photo_id"].to_pylist(), strict=True
    )
    return [Candidate(u, p, i) for u, p, i in columns]
