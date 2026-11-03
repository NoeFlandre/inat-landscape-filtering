"""Example photos for each score band of the card mosaic."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.examples import band_label, pick_examples, score_bands


def test_bands_add_two_steps_below_the_lowest_threshold():
    assert score_bands([0.04, 0.06, 0.08]) == [
        (-math.inf, 0.0),
        (0.0, 0.02),
        (0.02, 0.04),
        (0.04, 0.06),
        (0.06, 0.08),
        (0.08, math.inf),
    ]


def test_bands_sort_the_thresholds():
    assert score_bands([0.08, 0.04, 0.06]) == score_bands([0.04, 0.06, 0.08])


def test_one_threshold_steps_by_half_of_it():
    assert score_bands([0.002]) == [
        (-math.inf, 0.0),
        (0.0, 0.001),
        (0.001, 0.002),
        (0.002, math.inf),
    ]


def test_one_zero_threshold_steps_by_a_hundredth():
    assert score_bands([0.0])[:3] == [(-math.inf, -0.02), (-0.02, -0.01), (-0.01, 0.0)]


def test_bands_need_a_threshold():
    with pytest.raises(ValueError, match=r"needs at least one threshold$"):
        score_bands([])


@pytest.mark.parametrize(
    ("band", "label"),
    [
        ((-math.inf, 0.0), "< 0.00"),
        ((0.0, 0.02), "0.00 to 0.02"),
        ((0.08, math.inf), "≥ 0.08"),
        ((0.0005, 0.001), "0.0005 to 0.001"),
        ((0.5, math.inf), "≥ 0.50"),
    ],
)
def test_band_label(band, label):
    assert band_label(band) == label


def test_examples_are_cc0_photos_of_each_band():
    ids = [1, 2, 3, 4, 5, 6]
    scores = [-0.1, 0.01, 0.01, 0.05, 0.2, 0.3]
    licenses = ["CC0", "CC0", "CC-BY", "CC0", "CC0", "CC-BY"]
    bands = [(-math.inf, 0.0), (0.0, 0.04), (0.04, math.inf)]
    assert pick_examples(ids, scores, licenses, bands, per_band=2) == [[1], [2], [4, 5]]


def test_examples_skip_missing_scores():
    picked = pick_examples([1, 2], [None, 0.5], ["CC0", "CC0"], [(0.0, math.inf)], per_band=3)
    assert picked == [[2]]


def test_examples_are_the_same_for_the_same_seed_and_change_with_it():
    ids = list(range(1000))
    scores = [0.5] * 1000
    licenses = ["CC0"] * 1000
    bands = [(0.0, math.inf)]
    first = pick_examples(ids, scores, licenses, bands, per_band=6, seed=1)
    assert first == pick_examples(ids[::-1], scores, licenses, bands, per_band=6, seed=1)
    assert first != pick_examples(ids, scores, licenses, bands, per_band=6, seed=2)
    assert first == sorted(first)


def test_a_band_includes_its_lower_edge_and_excludes_its_upper_edge():
    bands = [(0.0, 0.5), (0.5, math.inf)]
    assert pick_examples([1, 2], [0.0, 0.5], ["CC0", "CC0"], bands, per_band=1) == [[1], [2]]


@given(
    st.lists(st.tuples(st.floats(-1, 1), st.sampled_from(["CC0", "CC-BY"])), max_size=50),
    st.integers(1, 5),
)
def test_examples_invariants(rows, per_band):
    ids = list(range(len(rows)))
    bands = score_bands([0.0, 0.5])
    picked = pick_examples(ids, [r[0] for r in rows], [r[1] for r in rows], bands, per_band)
    assert len(picked) == len(bands)
    for (lo, hi), chosen in zip(bands, picked, strict=True):
        assert len(chosen) <= per_band
        assert all(lo <= rows[i][0] < hi and rows[i][1] == "CC0" for i in chosen)


def test_two_thresholds_step_by_their_gap():
    assert score_bands([0.1, 0.12])[:3] == [(-math.inf, 0.06), (0.06, 0.08), (0.08, 0.1)]


def test_added_edges_are_rounded_to_nine_decimals():
    assert score_bands([3e-10, 5e-10])[:2] == [(-math.inf, 0.0), (0.0, 0.0)]


def test_the_default_seed_is_zero():
    ids, scores, licenses = list(range(100)), [0.5] * 100, ["CC0"] * 100
    bands = [(0.0, math.inf)]
    assert pick_examples(ids, scores, licenses, bands, 3) == pick_examples(
        ids, scores, licenses, bands, 3, seed=0
    )


def test_columns_of_different_lengths_are_an_error():
    with pytest.raises(ValueError, match="zip"):
        pick_examples([1, 2], [0.5], ["CC0"], [(0.0, math.inf)], per_band=1)


def test_a_score_outside_every_band_is_left_out():
    assert pick_examples([1], [2.0], ["CC0"], [(0.0, 1.0)], per_band=1) == [[]]
