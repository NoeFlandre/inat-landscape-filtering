"""Embed stage: download the small photos and store one SigLIP embedding per photo."""

import logging
import math
import os
import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
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
from inat_landscape.domain.sampling import Slots, in_slots
from inat_landscape.pipeline.errors import require
from inat_landscape.pipeline.land import LAND, land_only
from inat_landscape.pipeline.ports import Encoder, Fetcher
from inat_landscape.pipeline.sample import SAMPLE, SAMPLE_IDS
from inat_landscape.pipeline.score_cache import carry_scores

_LOG = logging.getLogger(__name__)
HUB_DIR = "hub"
# Decode and resize threads: the cores of the job, at most 16 (a GPU job has a share of a node).
_DECODE_THREADS = min(16, os.process_cpu_count() or 4)
FAILURES = "failures.parquet"
SCORING_SIZE = "small"
ONE_WORKER = Slots(0, 1, 1)  # all the photos
PUSH_SECONDS = 1800.0  # at most one commit of shards per job and half hour (hub: 128/hour)
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


def open_store(workdir: Path, model_id: str, hub: Hub, *, tag: str = "") -> EmbeddingStore:
    """Return the embedding store of `model_id`, after the upload of its local shards."""
    store = EmbeddingStore(workdir / HUB_DIR, embedding_dir(model_id), hub, tag=tag)
    store.push()
    return store


def worker_tag(worker: Slots) -> str:
    """Return the shard tag of a slot range: `w1of4`, `w1-3of40`, or none for one worker."""
    if worker.count == 1:
        return ""
    span = f"{worker.start}" if worker.stop == worker.start + 1 else f"{worker.start}-{worker.stop}"
    return f"w{span}of{worker.count}"


def read_sample(workdir: Path, columns: list[str] | None = None) -> pa.Table:
    """Return the sample table, or fail if the sample stage did not run.

    After the land stage, the photos of the cells at sea are left out (ADR 0013). For the
    photo ids and extensions only, the small `sample-ids.parquet` is enough.
    """
    if _slim_is_enough(workdir, columns):
        return read_table(workdir / SAMPLE_IDS, columns)
    require(workdir / SAMPLE, "sample", "sample")
    if columns is None:
        return land_only(workdir, read_table(workdir / SAMPLE))
    table = read_table(workdir / SAMPLE, list(dict.fromkeys([*columns, "h3_cell"])))
    return land_only(workdir, table).select(columns)


def _slim_is_enough(workdir: Path, columns: list[str] | None) -> bool:
    """Tell if `sample-ids.parquet` has the columns, and no land filter applies."""
    ids_only = bool(columns) and set(columns or ()) <= {"photo_id", "extension"}
    return ids_only and (workdir / SAMPLE_IDS).exists() and not (workdir / LAND).exists()


@dataclass(frozen=True)
class _Work:
    ids: list[int]
    store: EmbeddingStore
    done: set[int]
    failures: dict[int, dict]
    permanent: set[int]
    todo: list[dict]


def _work(workdir: Path, model_id: str, hub: Hub, worker: Slots) -> _Work:
    """Return the photos of `worker` and which of them still need an embedding."""
    sample = read_sample(workdir, columns=["photo_id", "extension"])  # small, for parallel workers
    sample = sample.filter(pa.array(in_slots(sample["photo_id"].to_numpy(), worker)))
    store = open_store(workdir, model_id, hub, tag=worker_tag(worker))
    done = store.remote_ids() | store.done_ids()
    failures = _read_failures(workdir / failures_name(worker))
    # The shared failure file (from `state pull`) also lists the known permanent failures.
    shared = _read_failures(workdir / FAILURES) if worker.count > 1 else {}
    permanent = (_permanent(failures) | _permanent(shared)) - done
    rows = sample.select(["photo_id", "extension"]).to_pylist()
    return _Work(
        sample["photo_id"].to_pylist(),
        store,
        done,
        failures,
        permanent,
        _todo(rows, done | permanent),
    )


def prune_embeddings(
    workdir: Path, model_id: str, hub: Hub, *, workers: int = 1, cache: Path | None = None
) -> tuple[int, int]:
    """Remove from the hub shards the photos that are not in the sample (after the land
    stage: the photos at sea). The cached scores of a pruned shard go to its new shard
    (ADR 0012). Return the count of shards rewritten and of rows removed."""
    keep = np.sort(read_sample(workdir, ["photo_id"])["photo_id"].to_numpy())
    pruned = open_store(workdir, model_id, hub).prune(keep, workers=workers)
    if cache is not None:
        carry_scores(cache, pruned, keep)
    return len(pruned), sum(p.removed for p in pruned)


def plan_embed(workdir: Path, model_id: str, hub: Hub, *, worker: Slots = ONE_WORKER) -> int:
    """Return how many photos of `worker` still need an embedding. Loads no model."""
    return len(_work(workdir, model_id, hub, worker).todo)


@dataclass(frozen=True)
class EmbedOptions:
    """How one embed run splits and batches its work."""

    chunk: int = 1000  # photos per shard
    worker: Slots = ONE_WORKER  # the slot range of this worker
    download_batch: int = 512  # photos downloaded and decoded ahead of the model, per batch
    max_photos: int | None = None  # stop after this many photos (a supervisor restarts the run)
    max_seconds: float | None = None  # stop after the batch that passes this time (job walltime)


def run_embed(
    workdir: Path,
    encoder: Encoder,
    hub: Hub,
    *,
    fetcher: Fetcher = fetch,
    options: EmbedOptions = EmbedOptions(),  # noqa: B008 - frozen dataclass, shared safely
) -> EmbedReport:
    """Embed every sampled photo of the slots of `worker` with no embedding and no permanent
    failure. Parallel workers on other machines share the work through the hub shards."""
    chunk, worker = options.chunk, options.worker
    work = _work(workdir, encoder.model_id, hub, worker)
    todo = work.todo[: options.max_photos]
    shard = _Shard(work.store, chunk)
    try:
        embedded, failed = _embed_all(workdir, encoder, (work, shard), todo, fetcher, options)
    finally:
        shard.flush()
        _try_push(work.store, tries=4, wait=60.0)  # the last shards, also after an error
    return EmbedReport(
        total=len(work.ids),
        already=len(work.done.intersection(work.ids)),
        skipped=len(work.permanent.intersection(work.ids)),
        embedded=embedded,
        failed=failed,
    )


def _embed_all(
    workdir: Path,
    encoder: Encoder,
    state: "tuple[_Work, _Shard]",
    todo: list[dict],
    fetcher: Fetcher,
    options: EmbedOptions,
) -> tuple[int, int]:
    """Embed the photos of `todo` batch by batch. Return the embedded and failed counts."""
    work, shard = state
    deadline, pushed = _deadline(options.max_seconds), time.monotonic()
    embedded = failed = 0
    batch = min(options.chunk, options.download_batch)
    for part, ids, images, bad in _prefetched(todo, fetcher, batch):
        if ids:
            shard.add(ids, np.asarray(encoder.encode_images(images)))
        embedded, failed = embedded + len(ids), failed + len(bad)
        _record_failures(workdir / failures_name(options.worker), work.failures, part, bad)
        _LOG.info("embedded %d/%d, failed %d", embedded, len(todo), failed)
        pushed = _push_now(work.store, pushed)
        if time.monotonic() >= deadline:
            _LOG.info("time limit reached: the next job continues")
            break
    return embedded, failed


def _push_now(store: EmbeddingStore, pushed: float) -> float:
    """Push the written shards when PUSH_SECONDS passed since `pushed`; return the push time."""
    if time.monotonic() - pushed < PUSH_SECONDS:
        return pushed
    _try_push(store)
    return time.monotonic()


def _try_push(store: EmbeddingStore, tries: int = 1, wait: float = 0.0) -> bool:
    """Push the written shards; on a hub error (for example the commit rate limit), log it
    and keep the shards for the next push. Return True when they are on the hub."""
    for attempt in range(tries):
        try:
            _LOG.info("pushed %d shards", store.push())
            return True
        except Exception as exc:
            _LOG.warning("push failed (try %d of %d): %s", attempt + 1, tries, exc)
            time.sleep(wait)
    return False


def _deadline(max_seconds: float | None) -> float:
    """Return the monotonic time after which the run stops (infinite with no limit)."""
    return math.inf if max_seconds is None else time.monotonic() + max_seconds


def _permanent(failures: dict[int, dict]) -> set[int]:
    return {photo_id for photo_id, f in failures.items() if f["permanent"]}


def _todo(rows: list[dict], exclude: set[int]) -> list[dict]:
    return [r for r in rows if r["photo_id"] not in exclude]


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class _Shard:
    """Embeddings waiting to be written; a shard is written every `size` photos."""

    def __init__(self, store: EmbeddingStore, size: int) -> None:
        self.store, self.size = store, size
        self.ids: list[int] = []
        self.rows: list[np.ndarray] = []

    def add(self, ids: list[int], rows: np.ndarray) -> None:
        self.ids += ids
        self.rows.append(rows)
        if len(self.ids) >= self.size:
            self.flush()

    def flush(self) -> None:
        if self.ids:
            self.store.add(self.ids, np.concatenate(self.rows), upload=False)
            self.ids, self.rows = [], []


_Batch = tuple[list[dict], list[int], list[Image.Image], list[dict]]


def _prefetched(todo: list[dict], fetcher: Fetcher, size: int) -> Iterator[_Batch]:
    """Yield (rows, ok ids, images, failures) per batch, downloaded in a background thread.

    The next batches download and decode while the caller encodes the current one.
    """
    batches: queue.Queue[_Batch | Exception | None] = queue.Queue(maxsize=2)
    threading.Thread(target=_produce, args=(todo, fetcher, size, batches), daemon=True).start()
    while (item := batches.get()) is not None:
        if isinstance(item, Exception):
            raise item
        yield item


def _produce(
    todo: list[dict], fetcher: Fetcher, size: int, out: "queue.Queue[_Batch | Exception | None]"
) -> None:
    try:
        for part in _chunks(todo, size):
            urls = {
                r["photo_id"]: photo_url(r["photo_id"], SCORING_SIZE, r["extension"]) for r in part
            }
            out.put((part, *_decode_all(fetcher(urls))))
        out.put(None)
    except Exception as exc:
        out.put(exc)


def _decode_all(results: list[Fetched]) -> tuple[list[int], list[Image.Image], list[dict]]:
    ids, images, failed = [], [], []
    with ThreadPoolExecutor(_DECODE_THREADS) as pool:
        decoded = list(pool.map(_decode_or_none, results))
    for result, image in zip(results, decoded, strict=True):
        if image is None:
            failed.append(_failure(result))
        else:
            ids.append(result.photo_id)
            images.append(image)
    return ids, images, failed


def _decode_or_none(result: Fetched) -> Image.Image | None:
    return decode(result.data) if result.data is not None else None


def _failure(result: Fetched) -> dict:
    """Return the failure row of a photo: a failed download, or a download that does not
    decode (permanent)."""
    error, permanent = (
        ("cannot decode image", True) if result.ok else (result.error, result.permanent)
    )
    return {
        "photo_id": result.photo_id,
        "status": result.status,
        "error": error,
        "permanent": permanent,
    }


def failures_name(worker: Slots) -> str:
    """Return the failure file of a worker: one file each, as parallel workers share a folder."""
    if worker.count == 1:
        return FAILURES
    return f"failures-{worker_tag(worker)[1:].replace('of', '-of-')}.parquet"


def _read_failures(path: Path) -> dict[int, dict]:
    rows = read_table(path).to_pylist() if path.exists() else []
    return {r["photo_id"]: r for r in rows}


def _record_failures(
    path: Path, failures: dict[int, dict], tried: list[dict], failed: list[dict]
) -> None:
    for row in tried:
        failures.pop(row["photo_id"], None)
    failures.update({f["photo_id"]: f for f in failed})
    rows = [failures[k] for k in sorted(failures)]
    write_table(pa.Table.from_pylist(rows, schema=_FAILURE_SCHEMA), path)
