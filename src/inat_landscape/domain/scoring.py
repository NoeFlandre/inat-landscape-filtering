"""Zero-shot scores: SigLIP sigmoid of image-text cosine similarity."""

import numpy as np
from numpy.typing import NDArray

MODEL_ID = "google/siglip2-base-patch16-224"
LANDSCAPE_PROMPTS = (
    "a photo of a landscape",
    "a wide view of the natural scenery",
    "an aerial-like view of the land",
)
CLOSEUP_PROMPTS = (
    "a close-up photo of a plant",
    "a macro photo of a flower",
    "a photo of a leaf",
)

Matrix = NDArray[np.floating]


def normalize(rows: Matrix) -> NDArray[np.float64]:
    """Scale each row of a 2-D array to unit length (in float64)."""
    rows = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero vector")
    return rows / norms


def ensemble(text_embeddings: Matrix) -> NDArray[np.float64]:
    """Return one unit text embedding: the mean of the unit prompt embeddings."""
    return normalize(normalize(text_embeddings).mean(axis=0, keepdims=True))[0]


def sigmoid_scores(
    image_embeddings: Matrix, text_embedding: Matrix, scale: float, bias: float
) -> NDArray[np.float64]:
    """Return sigmoid(scale * cosine + bias) for each image, as SigLIP computes it."""
    cosine = normalize(image_embeddings) @ normalize(np.atleast_2d(text_embedding))[0]
    logits = scale * cosine + bias
    return 0.5 * (1.0 + np.tanh(0.5 * logits))  # stable sigmoid
