"""Per-cell caps: first photos in sample order (before embedding), top scores (publish)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.cells import cap_first, top_per_cell

rows = st.lists(
    st.tuples(st.integers(0, 4), st.integers(0, 50), st.integers(0, 3)), max_size=60
).map(lambda r: [(c, k, p, i) for i, (c, k, p) in enumerate(r)])


def _arrays(r):
    a = np.array(r, dtype=np.int64).reshape(-1, 4)
    return a[:, 0], a[:, 1].astype(np.uint64), a[:, 2], a[:, 3]


def _naive_cap_first(r, cap):
    kept, seen = [], {}
    for c, _k, _p, i in sorted(r, key=lambda x: (x[1], x[2], x[3])):
        if seen.get(c, 0) < cap:
            seen[c] = seen.get(c, 0) + 1
            kept.append(i)
    return sorted(kept)


@given(rows, st.integers(0, 5))
def test_cap_first_matches_naive_definition(r, cap):
    assert cap_first(*_arrays(r), cap).tolist() == _naive_cap_first(r, cap)


@given(rows, st.integers(0, 5), st.integers(0, 5))
def test_cap_first_larger_cap_keeps_superset(r, a, b):
    small, large = sorted((a, b))
    assert set(cap_first(*_arrays(r), small)) <= set(cap_first(*_arrays(r), large))


@given(rows, st.integers(0, 5), st.randoms())
def test_cap_first_ignores_input_order(r, cap, rnd):
    shuffled = r[:]
    rnd.shuffle(shuffled)
    ids = lambda rows_, idx: sorted(rows_[i][3] for i in idx)  # noqa: E731
    assert ids(r, cap_first(*_arrays(r), cap)) == ids(shuffled, cap_first(*_arrays(shuffled), cap))


def test_cap_first_example():
    cells = np.array([7, 7, 7, 9])
    keys = np.array([30, 10, 20, 5], dtype=np.uint64)
    pos = np.array([0, 1, 0, 0])
    ids = np.array([100, 101, 102, 103])
    assert cap_first(cells, keys, pos, ids, 2).tolist() == [1, 2, 3]


def test_cap_first_empty_and_negative():
    empty = np.array([], dtype=np.int64)
    assert cap_first(empty, empty.astype(np.uint64), empty, empty, 3).tolist() == []
    with pytest.raises(ValueError, match="cap must be >= 0"):
        cap_first(empty, empty.astype(np.uint64), empty, empty, -1)


@given(st.lists(st.tuples(st.integers(0, 3), st.floats(0, 1)), max_size=40), st.integers(0, 4))
def test_top_per_cell_keeps_the_best_scores_of_each_cell(r, cap):
    cells = np.array([c for c, _ in r], dtype=np.int64)
    scores = np.array([s for _, s in r], dtype=float)
    keep = top_per_cell(cells, scores, cap)
    for c in set(cells.tolist()):
        mine = cells == c
        assert keep[mine].sum() == min(cap, mine.sum())
        if keep[mine].any() and (~keep[mine]).any():
            assert scores[mine & keep].min() >= scores[mine & ~keep].max()


def test_top_per_cell_breaks_ties_by_row_order():
    keep = top_per_cell(np.array([1, 1, 1]), np.array([0.5, 0.5, 0.9]), 2)
    assert keep.tolist() == [True, False, True]


def test_top_per_cell_rejects_negative_cap():
    with pytest.raises(ValueError, match="cap must be >= 0"):
        top_per_cell(np.array([1]), np.array([0.1]), -1)
