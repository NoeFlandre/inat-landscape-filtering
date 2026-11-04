"""Score stage: zero-shot SigLIP scores from the stored image embeddings."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa

from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.files import read_rows, read_table, write_json, write_table
from inat_landscape.adapters.hub import Hub
from inat_landscape.domain.scoring import (
    CLOSEUP_PROMPTS,
    LANDSCAPE_PROMPTS,
    NEGATIVE_PROMPTS,
    contrast_direction,
    cosine_scores,
    ensemble,
    sigmoid_scores,
)
from inat_landscape.pipeline.embed import open_store, read_sample
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ports import Encoder
from inat_landscape.pipeline.sample import SAMPLE

SCORED = "scored.parquet"
EMBEDDINGS = "embeddings.json"  # count and length of the scored embeddings
SCORE = "score_landscape"
SCORE_ROWS = 20_000  # embeddings scored at a time, to bound the memory


def run_score(
    workdir: Path, encoder: Encoder, hub: Hub, *, cache: Path | None = None, workers: int = 1
) -> pa.Table:
    """Write and return the sample photos that have an embedding, with their scores.

    The shards are scored one at a time in each of `workers` threads (ADR 0011: tens of
    millions of embeddings do not fit in memory). With a `cache` folder, the scores of each
    shard are kept there, and a later run reads them instead of the shard (ADR 0012).
    `embeddings.json` keeps the count and the length of the embeddings.
    """
    sample_ids = read_sample(workdir, columns=["photo_id"])["photo_id"].to_numpy()
    store = open_store(workdir, encoder.model_id, hub)
    directions = _directions(encoder)
    shards = _Shards(store, directions, encoder, _cache_dir(cache, directions, encoder))
    ids, scores, dim, rows = _score_shards(shards, workers)
    if not len(ids):
        raise StageError("no embeddings: run `embed` first")
    counts = {"embedding_rows": rows, "embedding_photos": len(ids), "embedding_dim": dim}
    write_json(counts, workdir / EMBEDDINGS)
    keep = np.isin(ids, sample_ids)
    ids, scores = ids[keep], {name: column[keep] for name, column in scores.items()}
    table = read_rows(workdir / SAMPLE, "photo_id", ids.tolist())
    sorter = np.argsort(ids)
    where = sorter[np.searchsorted(ids[sorter], table["photo_id"].to_numpy())]
    for name, column in scores.items():
        table = table.append_column(name, pa.array(column[where]))
    write_table(table, workdir / SCORED)
    return table


class _Shards:
    """Gives the scores of one shard: from the cache, else from its embeddings."""

    def __init__(
        self,
        store: EmbeddingStore,
        directions: dict[str, tuple[str, np.ndarray]],
        encoder: Encoder,
        cache: Path | None,
    ) -> None:
        self.store, self.directions, self.encoder, self.cache = store, directions, encoder, cache

    def score(self, path: str) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
        """Return the ids, the score columns and the embedding length of the shard."""
        if self.cache is None:
            return self._compute(path)
        cached = self.cache / path.rsplit("/", 1)[-1]
        if cached.exists():
            return _read_scores(cached)
        ids, columns, dim = self._compute(path)
        table = pa.table({"photo_id": ids} | columns)
        write_table(table.replace_schema_metadata({b"dim": str(dim).encode()}), cached)
        return ids, columns, dim

    def _compute(self, path: str) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
        shard_ids, embeddings = self.store.read(path)
        parts = [
            _score_columns(embeddings[i : i + SCORE_ROWS], self.directions, self.encoder)
            for i in range(0, len(shard_ids), SCORE_ROWS)
        ]
        columns = {n: np.concatenate([p[n] for p in parts]) for n in self.directions}
        return np.asarray(shard_ids, np.int64), columns, embeddings.shape[1]


def _read_scores(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Read the cached ids, score columns and embedding length of one shard."""
    table = read_table(path)
    columns = {n: table[n].to_numpy() for n in table.column_names[1:]}
    return table["photo_id"].to_numpy(), columns, int(table.schema.metadata[b"dim"])


def _score_shards(
    shards: _Shards, workers: int
) -> tuple[np.ndarray, dict[str, np.ndarray], int, int]:
    """Score every shard. Return the ids (first copy of each), their scores, the length of the
    embeddings and the count of rows (a photo of two observations can have two rows)."""
    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(shards.score, shards.store.shards()))
    ids = [i for i, _, _ in done]
    dim = max((d for _, _, d in done), default=0)
    return (*_first_copies(ids, [p for _, p, _ in done]), dim, sum(len(i) for i in ids))


def _cache_dir(
    cache: Path | None, directions: dict[str, tuple[str, np.ndarray]], encoder: Encoder
) -> Path | None:
    """Return the cache folder of these text directions and this encoder (no cache: None)."""
    if cache is None:
        return None
    digest = hashlib.sha256(f"{encoder.model_id} {encoder.scale} {encoder.bias}".encode())
    for name, (kind, text) in directions.items():
        digest.update(f"{name} {kind}".encode() + np.asarray(text, np.float64).tobytes())
    return cache / digest.hexdigest()[:16]


def _first_copies(
    ids: list[np.ndarray], parts: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the ids and the scores of the first copy of each id."""
    if not ids:
        return np.zeros(0, np.int64), {}
    all_ids = np.concatenate(ids)
    _, first = np.unique(all_ids, return_index=True)
    first.sort()
    return all_ids[first], {n: np.concatenate([p[n] for p in parts])[first] for n in parts[0]}


def _directions(encoder: Encoder) -> dict[str, tuple[str, np.ndarray]]:
    """Return each score column: its kind (sigmoid or cosine) and its text direction."""
    texts = encoder.encode_texts([*LANDSCAPE_PROMPTS, *CLOSEUP_PROMPTS, *NEGATIVE_PROMPTS])
    a, b = len(LANDSCAPE_PROMPTS), len(LANDSCAPE_PROMPTS) + len(CLOSEUP_PROMPTS)
    landscape, closeup, negative = texts[:a], texts[a:b], texts[b:]
    directions = {
        SCORE: ("sigmoid", ensemble(landscape)),
        "score_closeup": ("sigmoid", ensemble(closeup)),
        "score_contrast": ("cosine", contrast_direction(landscape, negative)),
    }
    return directions | {f"{SCORE}_{i}": ("sigmoid", t) for i, t in enumerate(landscape, start=1)}


def _score_columns(
    images: np.ndarray, directions: dict[str, tuple[str, np.ndarray]], encoder: Encoder
) -> dict[str, np.ndarray]:
    return {
        name: sigmoid_scores(images, text, encoder.scale, encoder.bias)
        if kind == "sigmoid"
        else cosine_scores(images, text)
        for name, (kind, text) in directions.items()
    }
