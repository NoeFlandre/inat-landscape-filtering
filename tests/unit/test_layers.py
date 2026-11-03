"""Complete layers: the cap that every H3 cell has reached so far."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.layers import complete_cap


def _cap(ranks, done, tolerance=0.0):
    return complete_cap(np.array(ranks, np.int64), np.array(done, bool), tolerance)


def test_the_cap_is_the_first_rank_with_a_missing_photo():
    assert _cap([0, 0, 1, 1, 2], [True, True, True, False, False]) == 1


def test_all_done_gives_one_more_than_the_last_rank():
    assert _cap([0, 1, 1, 2], [True] * 4) == 3


def test_nothing_done_gives_zero():
    assert _cap([0, 0, 1], [False] * 3) == 0


def test_an_empty_sample_gives_zero():
    assert _cap([], []) == 0


def test_a_few_failed_downloads_do_not_hold_the_layers_back():
    ranks = [0] * 1000 + [1] * 1000 + [2] * 10
    done = [True] * 999 + [False] + [True] * 1000 + [False] * 10
    assert _cap(ranks, done, tolerance=0.002) == 2
    assert _cap(ranks, done, tolerance=0.0) == 0


def test_a_later_layer_within_the_tolerance_does_not_skip_a_layer_above_it():
    ranks = [0] * 10 + [1] * 10 + [2] * 1000
    done = [True] * 10 + [False] * 10 + [True] * 1000
    assert _cap(ranks, done, tolerance=0.02) == 1


@given(st.lists(st.tuples(st.integers(0, 5), st.booleans()), max_size=60))
def test_every_photo_below_the_cap_is_done_with_no_tolerance(rows):
    ranks = [r for r, _ in rows]
    done = [d for _, d in rows]
    cap = _cap(ranks, done)
    assert all(d for r, d in rows if r < cap)
    assert cap == max(ranks, default=-1) + 1 or any(not d for r, d in rows if r == cap)
