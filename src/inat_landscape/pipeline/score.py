"""Score stage: zero-shot SigLIP scores from the stored image embeddings."""

from pathlib import Path

import numpy as np
import pyarrow as pa

from inat_landscape.adapters.files import read_rows, write_table
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
SCORE = "score_landscape"
SCORE_ROWS = 20_000  # embeddings scored at a time, to bound the memory


def run_score(workdir: Path, encoder: Encoder, hub: Hub) -> pa.Table:
    """Write and return the sample photos that have an embedding, with their scores."""
    table, images, order = _embedded_sample(workdir, encoder.model_id, hub)
    directions = _directions(encoder)
    parts = [
        _score_columns(images[order[i : i + SCORE_ROWS]], directions, encoder)
        for i in range(0, len(order), SCORE_ROWS)
    ]
    for name in parts[0]:
        table = table.append_column(name, pa.array(np.concatenate([p[name] for p in parts])))
    write_table(table, workdir / SCORED)
    return table


def _embedded_sample(
    workdir: Path, model_id: str, hub: Hub
) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    """Return the sample rows that have an embedding, all embeddings, and for each row the
    position of its embedding. No full copy of the embeddings is made."""
    ids, embeddings = open_store(workdir, model_id, hub).load()
    if not ids:
        raise StageError("no embeddings: run `embed` first")
    read_sample(workdir, columns=["photo_id"])  # fails clearly if the sample is missing
    table = read_rows(workdir / SAMPLE, "photo_id", ids)
    row = {photo_id: n for n, photo_id in enumerate(ids)}
    return table, embeddings, np.array([row[i] for i in table["photo_id"].to_pylist()])


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
