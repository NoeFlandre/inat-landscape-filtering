"""Scoring: SigLIP sigmoid scores from embeddings."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from inat_landscape.domain.scoring import (
    CLOSEUP_PROMPTS,
    LANDSCAPE_PROMPTS,
    NEGATIVE_PROMPTS,
    contrast_direction,
    cosine_scores,
    ensemble,
    normalize,
    sigmoid_scores,
)

finite = st.floats(-10, 10, allow_nan=False, width=32)
vectors = arrays(np.float32, (3, 4), elements=finite).filter(
    lambda a: bool(np.all(np.linalg.norm(a, axis=1) > 1e-3))
)


def test_prompts_match_the_agreed_plan():
    assert LANDSCAPE_PROMPTS == (
        "a photo of a landscape",
        "a wide view of the natural scenery",
        "an aerial-like view of the land",
    )
    assert len(CLOSEUP_PROMPTS) == 3


@given(vectors)
def test_normalize_gives_unit_rows(v):
    assert np.allclose(np.linalg.norm(normalize(v), axis=1), 1.0, atol=1e-4)


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError, match=r"^cannot normalize a zero vector$"):
        normalize(np.zeros((1, 3), dtype=np.float32))


def test_ensemble_is_normalized_mean_of_normalized_rows():
    t = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    result = ensemble(t)
    assert result.shape == (2,)
    assert np.allclose(result, np.array([1.0, 1.0]) / np.sqrt(2))


def test_normalize_computes_in_float64():
    rows = np.array([[3.0, 4.0]], dtype=np.float16)
    out = normalize(rows)
    assert out.dtype == np.float64
    assert out.tolist() == [[0.6, 0.8]]


def test_normalize_needs_2d_input():
    with pytest.raises(np.exceptions.AxisError):
        normalize(np.array([3.0, 4.0]))


@given(vectors, vectors, st.floats(1, 200), st.floats(-20, 5))
def test_scores_are_probabilities(images, texts, scale, bias):
    s = sigmoid_scores(images, texts[0], scale, bias)
    assert s.shape == (3,)
    assert np.all((s >= 0) & (s <= 1))


def test_score_matches_siglip_formula():
    img = np.array([[1.0, 0.0]], dtype=np.float32)
    txt = np.array([0.6, 0.8], dtype=np.float32)
    expected = 1 / (1 + np.exp(-(10 * 0.6 - 2)))
    assert sigmoid_scores(img, txt, 10.0, -2.0) == pytest.approx([expected], rel=1e-6)


def test_score_ignores_embedding_norm():
    txt = np.array([0.6, 0.8], dtype=np.float32)
    a = sigmoid_scores(np.array([[1.0, 1.0]], dtype=np.float32), txt, 50.0, -10.0)
    b = sigmoid_scores(np.array([[7.0, 7.0]], dtype=np.float32), txt, 50.0, -10.0)
    assert a == pytest.approx(b)


def test_score_does_not_overflow_for_large_logits():
    img = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    s = sigmoid_scores(img, np.array([1.0, 0.0], dtype=np.float32), 1000.0, 0.0)
    assert s.tolist() == [1.0, 0.0]


def test_negative_prompts_cover_specimens_and_animal_signs():
    assert "a natural specimen" in NEGATIVE_PROMPTS
    for word in ("plant", "leaf", "flower", "animal", "tracks", "droppings", "bones"):
        assert any(word in p for p in NEGATIVE_PROMPTS), word


def test_contrast_direction_is_unit_mean_positive_minus_mean_negative():
    pos = np.array([[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]])
    neg = np.array([[0.0, 0.0, 1.0], [0.0, 3.0, 4.0]])
    d = contrast_direction(pos, neg)
    expected = np.array([0.6, 0.8, 0.0]) - np.array([0.0, 0.3, 0.9])
    assert np.allclose(d, expected / np.linalg.norm(expected))


def test_cosine_scores_rank_landscape_above_specimen():
    d = contrast_direction(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    s = cosine_scores(np.array([[5.0, 0.0], [0.0, 2.0], [1.0, 1.0]]), d)
    assert s.tolist() == pytest.approx([2**-0.5, -(2**-0.5), 0.0])


def test_contrast_direction_rejects_identical_sets():
    with pytest.raises(ValueError, match="zero"):
        contrast_direction(np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]]))
