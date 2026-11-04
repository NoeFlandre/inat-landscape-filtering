"""Fast set membership."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.members import in_sorted


def test_in_sorted_finds_the_members():
    found = in_sorted(np.array([5, 1, 9, 3, 10]), np.array([1, 3, 9]))
    assert found.tolist() == [False, True, True, True, False]


def test_nothing_is_in_an_empty_array():
    found = in_sorted(np.array([1, 2]), np.array([], np.int64))
    assert found.dtype == np.bool_
    assert found.tolist() == [False, False]


@given(st.lists(st.integers(-50, 50)), st.lists(st.integers(-50, 50)))
def test_in_sorted_is_isin(values, keep):
    values, keep = np.array(values, np.int64), np.array(keep, np.int64)
    assert in_sorted(values, np.sort(keep)).tolist() == np.isin(values, keep).tolist()


@given(
    st.lists(st.integers(0, 2**64 - 1), max_size=30),
    st.lists(st.integers(0, 2**64 - 1), max_size=30),
)
def test_in_sorted_takes_unsigned_64_bit_values(values, selected):
    values_a = np.array(values, dtype=np.uint64)
    selected_a = np.unique(np.array(selected, dtype=np.uint64))
    assert in_sorted(values_a, selected_a).tolist() == np.isin(values_a, selected_a).tolist()
