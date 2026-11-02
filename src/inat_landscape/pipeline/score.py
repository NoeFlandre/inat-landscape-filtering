"""Score stage: zero-shot SigLIP scores from the stored image embeddings."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from inat_landscape.adapters.files import write_table
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

SCORED = "scored.parquet"
SCORE = "score_landscape"


def run_score(workdir: Path, encoder: Encoder, hub: Hub) -> pa.Table:
    """Write and return the sample photos that have an embedding, with their scores."""
    table, images = _embedded_sample(workdir, encoder.model_id, hub)
    for name, values in _score_columns(images, encoder).items():
        table = table.append_column(name, pa.array(values))
    write_table(table, workdir / SCORED)
    return table


def _embedded_sample(workdir: Path, model_id: str, hub: Hub) -> tuple[pa.Table, np.ndarray]:
    """Return the sample rows that have an embedding, and their embeddings in row order."""
    sample = read_sample(workdir)
    ids, embeddings = open_store(workdir, model_id, hub).load()
    if not ids:
        raise StageError("no embeddings: run `embed` first")
    table = sample.filter(pc.is_in(sample["photo_id"], pa.array(ids, pa.int64())))
    row = {photo_id: n for n, photo_id in enumerate(ids)}
    return table, embeddings[[row[i] for i in table["photo_id"].to_pylist()]]


def _score_columns(images: np.ndarray, encoder: Encoder) -> dict[str, np.ndarray]:
    texts = encoder.encode_texts([*LANDSCAPE_PROMPTS, *CLOSEUP_PROMPTS, *NEGATIVE_PROMPTS])
    a, b = len(LANDSCAPE_PROMPTS), len(LANDSCAPE_PROMPTS) + len(CLOSEUP_PROMPTS)
    landscape, closeup, negative = texts[:a], texts[a:b], texts[b:]

    def score(text_embedding: np.ndarray) -> np.ndarray:
        return sigmoid_scores(images, text_embedding, encoder.scale, encoder.bias)

    columns = {SCORE: score(ensemble(landscape)), "score_closeup": score(ensemble(closeup))}
    columns["score_contrast"] = cosine_scores(images, contrast_direction(landscape, negative))
    return columns | {f"{SCORE}_{i}": score(t) for i, t in enumerate(landscape, start=1)}
