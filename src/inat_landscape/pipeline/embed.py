"""Embed stage: download the small photos and store one SigLIP embedding per photo."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.files import read_table, write_table
from inat_landscape.adapters.hub import Hub
from inat_landscape.adapters.images import Fetched, decode, fetch
from inat_landscape.adapters.inat_dump import PHOTO_URL
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ports import Encoder, Fetcher, Log
from inat_landscape.pipeline.sample import SAMPLE

HUB_DIR = "hub"
FAILURES = "failures.parquet"
SCORING_SIZE = "small"
_FAILURE_SCHEMA = pa.schema(
    [
        ("photo_id", pa.int64()),
        ("status", pa.int64()),
        ("error", pa.string()),
        ("permanent", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class EmbedReport:
    """What one embed run did."""

    total: int
    already: int
    skipped: int
    embedded: int
    failed: int


def embedding_dir(model_id: str) -> str:
    """Return the hub folder of the embeddings of one model and image size."""
    return f"embeddings/{model_id.rsplit('/', maxsplit=1)[-1]}/{SCORING_SIZE}"


def photo_url(photo_id: int, size: str, extension: str) -> str:
    """Return the URL of one photo file in the open data bucket."""
    return PHOTO_URL.format(photo_id=photo_id, size=size, extension=extension)


def open_store(workdir: Path, model_id: str, hub: Hub) -> EmbeddingStore:
    """Return the embedding store of `model_id`, synced both ways with the hub."""
    store = EmbeddingStore(workdir / HUB_DIR, embedding_dir(model_id), hub)
    store.pull()
    store.push()
    return store


def read_sample(workdir: Path) -> pa.Table:
    """Return the sample table, or fail if the sample stage did not run."""
    if not (workdir / SAMPLE).exists():
        raise StageError(f"no sample at {workdir / SAMPLE}: run `sample` first")
    return read_table(workdir / SAMPLE)


def run_embed(
    workdir: Path,
    encoder: Encoder,
    hub: Hub,
    *,
    fetcher: Fetcher = fetch,
    chunk: int = 1000,
    log: Log = lambda m: print(m, file=sys.stderr),
) -> EmbedReport:
    """Embed every sampled photo that has no embedding and no permanent failure."""
    sample = read_sample(workdir)
    store = open_store(workdir, encoder.model_id, hub)
    done = store.done_ids()
    failures = _read_failures(workdir)
    ids = sample["photo_id"].to_pylist()
    permanent = _permanent(failures) - done
    todo = _todo(sample.select(["photo_id", "extension"]).to_pylist(), done | permanent)
    embedded = failed = 0
    for part in _chunks(todo, chunk):
        ok, bad = _embed_chunk(part, encoder, store, fetcher)
        embedded, failed = embedded + ok, failed + len(bad)
        _record_failures(workdir, failures, part, bad)
        log(f"embedded {embedded}/{len(todo)}, failed {failed}")
    return EmbedReport(
        total=len(ids),
        already=len(done.intersection(ids)),
        skipped=len(permanent.intersection(ids)),
        embedded=embedded,
        failed=failed,
    )


def _permanent(failures: dict[int, dict]) -> set[int]:
    return {photo_id for photo_id, f in failures.items() if f["permanent"]}


def _todo(rows: list[dict], exclude: set[int]) -> list[dict]:
    return [r for r in rows if r["photo_id"] not in exclude]


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _embed_chunk(
    part: list[dict], encoder: Encoder, store: EmbeddingStore, fetcher: Fetcher
) -> tuple[int, list[dict]]:
    urls = {r["photo_id"]: photo_url(r["photo_id"], SCORING_SIZE, r["extension"]) for r in part}
    ids, images, failed = _decode_all(fetcher(urls))
    if ids:
        store.add(ids, np.asarray(encoder.encode_images(images)))
    return len(ids), failed


def _decode_all(results: list[Fetched]) -> tuple[list[int], list[Image.Image], list[dict]]:
    ids, images, failed = [], [], []
    for result in results:
        image = decode(result.data) if result.data is not None else None
        if image is None:
            failed.append(_failure(result))
        else:
            ids.append(result.photo_id)
            images.append(image)
    return ids, images, failed


def _failure(result: Fetched) -> dict:
    if result.ok:
        return {
            "photo_id": result.photo_id,
            "status": result.status,
            "error": "cannot decode image",
            "permanent": True,
        }
    return {
        "photo_id": result.photo_id,
        "status": result.status,
        "error": result.error,
        "permanent": result.permanent,
    }


def _read_failures(workdir: Path) -> dict[int, dict]:
    path = workdir / FAILURES
    rows = read_table(path).to_pylist() if path.exists() else []
    return {r["photo_id"]: r for r in rows}


def _record_failures(
    workdir: Path, failures: dict[int, dict], tried: list[dict], failed: list[dict]
) -> None:
    for row in tried:
        failures.pop(row["photo_id"], None)
    failures.update({f["photo_id"]: f for f in failed})
    rows = [failures[k] for k in sorted(failures)]
    write_table(pa.Table.from_pylist(rows, schema=_FAILURE_SCHEMA), workdir / FAILURES)
