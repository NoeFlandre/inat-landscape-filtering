"""Score stage: zero-shot SigLIP scores from the stored image embeddings."""

from pathlib import Path

import numpy as np
import pyarrow as pa

from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.files import read_rows, write_json, write_table
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


def run_score(workdir: Path, encoder: Encoder, hub: Hub) -> pa.Table:
    """Write and return the sample photos that have an embedding, with their scores.

    The shards are scored one at a time (ADR 0011: tens of millions of embeddings do not fit
    in memory). `embeddings.json` keeps the count and the length of the embeddings.
    """
    sample_ids = read_sample(workdir, columns=["photo_id"])["photo_id"].to_numpy()
    store = open_store(workdir, encoder.model_id, hub)
    directions = _directions(encoder)
    ids, scores, dim, rows = _score_shards(store, directions, encoder)
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


def _score_shards(
    store: EmbeddingStore, directions: dict[str, tuple[str, np.ndarray]], encoder: Encoder
) -> tuple[np.ndarray, dict[str, np.ndarray], int, int]:
    """Score every shard. Return the ids (first copy of each), their scores, the length of the
    embeddings and the count of rows (a photo of two observations can have two rows)."""
    ids: list[np.ndarray] = []
    parts: list[dict[str, np.ndarray]] = []
    dim = 0
    for shard_ids, embeddings in store.stream():
        dim = embeddings.shape[1]
        ids.append(np.asarray(shard_ids, np.int64))
        parts.extend(
            _score_columns(embeddings[i : i + SCORE_ROWS], directions, encoder)
            for i in range(0, len(shard_ids), SCORE_ROWS)
        )
    return (*_first_copies(ids, parts), dim, sum(len(i) for i in ids))


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
