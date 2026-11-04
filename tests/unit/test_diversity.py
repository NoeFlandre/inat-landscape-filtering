"""Visual diversity of embeddings."""

import itertools

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from inat_landscape.domain.diversity import (
    group_sums,
    mean_pairwise_distance,
    principal_components,
    to_unit_range,
    unit,
)


def test_unit_scales_rows_to_length_one_and_keeps_zero_rows():
    v = unit(np.array([[3.0, 4.0], [0.0, 0.0], [0.3, 0.4]]))
    assert v.dtype == np.float32
    assert np.allclose(v, [[0.6, 0.8], [0.0, 0.0], [0.6, 0.8]])


def test_group_sums_add_the_rows_of_each_group():
    rows = np.array([[1.0], [2.0], [3.0], [4.0]], np.float32)
    groups, sums, counts = group_sums(np.array([2, 0, 2, 5]), rows)
    assert groups.tolist() == [0, 2, 5]
    assert sums.tolist() == [[2.0], [4.0], [4.0]]
    assert sums.dtype == np.float64
    assert counts.tolist() == [1, 2, 1]


def test_group_sums_of_no_row():
    groups, sums, counts = group_sums(np.zeros(0, np.int64), np.zeros((0, 3), np.float32))
    assert (len(groups), sums.shape, len(counts)) == (0, (0, 3), 0)
    assert (sums.dtype, counts.dtype) == (np.float64, np.int64)


def test_same_vectors_have_no_diversity_and_opposite_ones_the_most():
    same = unit(np.array([[1.0, 1.0]] * 3)).sum(axis=0, dtype=np.float64)
    opposite = np.array([[1.0, 0.0], [-1.0, 0.0]]).sum(axis=0)
    got = mean_pairwise_distance(np.array([same, opposite]), np.array([3, 2]))
    assert np.allclose(got, [0.0, 2.0], atol=1e-6)


def test_diversity_needs_a_pair_whatever_the_sum():
    got = mean_pairwise_distance(np.array([[2.0, 0.0], [1.0, 1.0]]), np.array([1, 2]))
    assert np.isnan(got[0])
    assert np.isclose(got[1], 1.0)


def test_a_single_vector_has_no_diversity_value():
    assert np.isnan(mean_pairwise_distance(np.array([[1.0, 0.0]]), np.array([1]))[0])
    assert np.isnan(mean_pairwise_distance(np.array([[0.0, 0.0]]), np.array([0]))[0])


@settings(max_examples=50)
@given(st.lists(st.tuples(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1)), min_size=2))
def test_diversity_is_the_mean_pairwise_cosine_distance(rows):
    v = unit(np.array(rows) + np.array([0.0, 0.0, 2.0]))  # no zero row
    pairs = [1 - float(np.dot(a, b)) for a, b in itertools.permutations(v.astype(np.float64), 2)]
    got = mean_pairwise_distance(v.sum(axis=0, dtype=np.float64)[None], np.array([len(v)]))
    assert np.isclose(got[0], np.mean(pairs), atol=1e-5)


def test_principal_components_follow_the_largest_spread():
    x = np.array([[-2.0, 0.1, 0.0], [2.0, 0.1, 0.0], [-2.0, -0.1, 0.0], [2.0, -0.1, 0.0]])
    pcs = principal_components(x + np.array([10.0, 5.0, 3.0]), k=2)  # not centred
    assert pcs.shape == (4, 2)
    assert np.allclose(pcs, x[:, :2])  # the largest loadings are positive
    assert principal_components(np.diag([5.0, 4.0, 3.0, 2.0, 1.0])).shape == (5, 3)
    assert np.allclose(principal_components(-x, k=1), -principal_components(x, k=1))


def test_principal_components_have_a_stable_sign():
    x = np.array([[1.0, 0.0], [3.0, 0.0], [-4.0, 0.0]])
    assert principal_components(x, k=1)[:, 0].tolist() == [1.0, 3.0, -4.0]


def test_to_unit_range_clips_to_the_percentiles():
    values = np.arange(101, dtype=float)[:, None]
    scaled = to_unit_range(values, 10, 90)
    assert scaled[10, 0] == 0.0
    assert scaled[50, 0] == 0.5
    assert scaled[90, 0] == 1.0
    assert scaled[0, 0] == 0.0
    assert scaled[100, 0] == 1.0
    assert to_unit_range(np.ones((3, 1))).tolist() == [[0.0], [0.0], [0.0]]


def test_to_unit_range_scales_each_column_alone():
    values = np.array([[0.0, 0.0], [5.0, 50.0], [10.0, 100.0]])
    assert to_unit_range(values, 0, 100).tolist() == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]


def test_to_unit_range_defaults_to_the_2nd_and_98th_percentiles():
    values = np.arange(101, dtype=float)[:, None]
    assert to_unit_range(values)[2, 0] == 0.0
    assert to_unit_range(values)[98, 0] == 1.0
    assert 0 < to_unit_range(values)[3, 0] < to_unit_range(values)[97, 0] < 1
