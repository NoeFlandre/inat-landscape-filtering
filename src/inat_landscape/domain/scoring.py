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

NEGATIVE_PROMPTS = (
    "a photo of a single plant",
    "a photo of a single leaf",
    "a photo of a single flower",
    "a photo of a single animal",
    "a natural specimen",
    "animal tracks on the ground",
    "animal droppings",
    "animal bones or a skull",
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


def contrast_direction(positive: Matrix, negative: Matrix) -> NDArray[np.float64]:
    """Return the unit text direction: mean positive prompt minus mean negative prompt."""
    mean_pos = normalize(positive).mean(axis=0)
    mean_neg = normalize(negative).mean(axis=0)
    return normalize((mean_pos - mean_neg)[np.newaxis, :])[0]


def cosine_scores(image_embeddings: Matrix, direction: Matrix) -> NDArray[np.float64]:
    """Return the cosine between each image and a text direction, in [-1, 1]."""
    return normalize(image_embeddings) @ normalize(np.atleast_2d(direction))[0]
