"""Status: how far the pipeline got in a work directory. Reads local files only."""

from pathlib import Path

import pyarrow.parquet as pq

from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.hub import OfflineHub
from inat_landscape.pipeline.embed import FAILURES, HUB_DIR, embedding_dir
from inat_landscape.pipeline.ingest import CANDIDATES
from inat_landscape.pipeline.sample import SAMPLE
from inat_landscape.pipeline.score import SCORED


def run_status(workdir: Path, model_id: str) -> dict[str, int]:
    """Return the row count of each stage output (0 when a stage did not run)."""
    store = EmbeddingStore(workdir / HUB_DIR, embedding_dir(model_id), OfflineHub())
    return {
        "candidates": _rows(workdir / CANDIDATES),
        "sample": _rows(workdir / SAMPLE),
        "embeddings": len(store.done_ids()),
        "failures": _rows(workdir / FAILURES),
        "scored": _rows(workdir / SCORED),
    }


def _rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows if path.exists() else 0
