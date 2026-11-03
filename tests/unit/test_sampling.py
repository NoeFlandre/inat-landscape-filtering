"""Sampling: deterministic, prefix-based, and stable when the sample grows."""

import random
import uuid
from collections import Counter

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.sampling import (
    Slots,
    for_worker,
    in_sample,
    in_slots,
    sample_prefix,
    uniform_order,
    uuid_key,
)

uuids = st.uuids(version=4).map(str)
fractions = st.floats(min_value=1e-9, max_value=1.0, allow_nan=False)


def test_prefix_is_eight_hex_chars_for_fraction_below_one():
    assert sample_prefix(0.5) == "80000000"
    assert sample_prefix(0.001) == "00418937"


def test_prefix_for_full_fraction_keeps_everything():
    assert in_sample("ffffffff-0000-4000-8000-000000000000", sample_prefix(1.0))


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01])
def test_prefix_rejects_out_of_range_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        sample_prefix(fraction)


def test_in_sample_is_case_insensitive():
    assert in_sample("0000ABCD-0000-4000-8000-000000000000", "0000abce")
    assert not in_sample("0000ABCD-0000-4000-8000-000000000000", "0000abcd")


@given(uuids, fractions, fractions)
def test_larger_fraction_keeps_superset(u, a, b):
    small, large = sorted((a, b))
    if in_sample(u, sample_prefix(small)):
        assert in_sample(u, sample_prefix(large))


def test_kept_share_matches_fraction():
    rng = random.Random(7)
    ids = [str(uuid.UUID(int=rng.getrandbits(128), version=4)) for _ in range(20000)]
    prefix = sample_prefix(0.25)
    share = sum(in_sample(u, prefix) for u in ids) / len(ids)
    assert 0.22 < share < 0.28


photos = st.lists(
    st.tuples(st.integers(0, 2**64 - 1), st.integers(0, 3), st.integers(0, 4)),  # key, pos, cell
    max_size=40,
)


def _order(rows):
    keys = np.array([r[0] for r in rows], dtype=np.uint64)
    positions = np.array([r[1] for r in rows], dtype=np.int64)
    ids = np.arange(len(rows), dtype=np.int64)
    cells = np.array([r[2] for r in rows], dtype=np.int64)
    return uniform_order(keys, positions, ids, cells)


@given(photos, st.integers(0, 50))
def test_any_prefix_of_the_order_is_uniform_over_the_cells(rows, n):
    ids, ranks = _order(rows)
    first = Counter(rows[i][2] for i in ids[:n].tolist())
    left = Counter(rows[i][2] for i in ids[n:].tolist())
    # A cell with photos left has no fewer photos in the prefix than the cap reached so far.
    assert all(first[c] >= max(first.values(), default=0) - 1 for c in left)
    assert sorted(ids.tolist()) == list(range(len(rows)))
    assert ranks.tolist() == sorted(ranks.tolist())


@given(photos)
def test_the_rank_is_the_place_of_a_photo_in_the_sample_order_of_its_cell(rows):
    ids, ranks = _order(rows)
    sample = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1], i))
    seen: Counter = Counter()
    expected = {}
    for i in sample:
        expected[i] = seen[rows[i][2]]
        seen[rows[i][2]] += 1
    assert dict(zip(ids.tolist(), ranks.tolist(), strict=True)) == expected


def test_order_is_rank_then_observation_then_position_then_photo():
    keys = np.array([5, 5, 9, 1], dtype=np.uint64)
    positions = np.array([1, 0, 0, 0], dtype=np.int64)
    ids = np.array([10, 11, 12, 13], dtype=np.int64)
    cells = np.array([7, 7, 7, 8], dtype=np.int64)
    ordered, ranks = uniform_order(keys, positions, ids, cells)
    assert ordered.tolist() == [13, 11, 10, 12]
    assert ranks.tolist() == [0, 0, 1, 2]
    assert ranks.dtype == np.int64


def test_order_does_not_depend_on_the_input_order():
    keys = np.array([3, 1, 2], dtype=np.uint64)
    cells = np.array([0, 0, 1], dtype=np.int64)
    zero = np.zeros(3, dtype=np.int64)
    ids = np.array([1, 2, 3], dtype=np.int64)
    back = uniform_order(keys[::-1], zero, ids[::-1], cells[::-1])[0].tolist()
    assert uniform_order(keys, zero, ids, cells)[0].tolist() == back


def test_empty_order():
    empty = np.array([], dtype=np.int64)
    ids, ranks = uniform_order(empty.astype(np.uint64), empty, empty, empty)
    assert (ids.tolist(), ranks.tolist()) == ([], [])


def test_uuid_key_is_the_first_16_hex_digits():
    assert uuid_key("0123abcd-ef45-4678-9abc-def012345678") == 0x0123ABCDEF454678
    assert uuid_key("0123ABCD-EF45-4678-9ABC-DEF012345678") == 0x0123ABCDEF454678


@given(uuids, uuids)
def test_uuid_key_order_matches_uuid_text_order(a, b):
    if a[:18] != b[:18]:
        assert (uuid_key(a) < uuid_key(b)) == (a[:18] < b[:18])


@given(st.integers(0, 10**9), st.integers(1, 8))
def test_each_photo_belongs_to_exactly_one_worker(photo_id, workers):
    owners = [w for w in range(workers) if for_worker(photo_id, w, workers)]
    assert len(owners) == 1


@pytest.mark.parametrize(("worker", "workers"), [(-1, 2), (2, 2), (0, 0)])
def test_for_worker_rejects_bad_worker(worker, workers):
    with pytest.raises(ValueError, match="worker"):
        for_worker(1, worker, workers)


@given(
    st.lists(st.integers(0, 10**9), max_size=40),
    st.integers(1, 30).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n - 1))),
    st.integers(1, 5),
)
def test_a_slot_range_owns_the_photos_of_its_slots(ids, count_start, width):
    count, start = count_start
    stop = min(start + width, count)
    got = in_slots(np.array(ids, dtype=np.int64), Slots(start, stop, count)).tolist()
    assert got == [start <= i % count < stop for i in ids]


def test_a_single_slot_is_the_old_worker_split():
    ids = np.arange(20, dtype=np.int64)
    assert in_slots(ids, Slots(3, 4, 7)).tolist() == [for_worker(i, 3, 7) for i in range(20)]


@pytest.mark.parametrize(
    ("raw", "slots"),
    [("3/7", Slots(3, 4, 7)), ("2:5/40", Slots(2, 5, 40)), ("0/1", Slots(0, 1, 1))],
)
def test_slots_parse_and_print(raw, slots):
    assert Slots.parse(raw) == slots
    assert str(slots) == raw


@pytest.mark.parametrize("raw", ["3", "5/4", "3:3/7", "2:9/7", "-1/3", "a/b", "4:2/9"])
def test_bad_slots_are_rejected(raw):
    with pytest.raises(ValueError, match="slots"):
        Slots.parse(raw)


def test_slot_ranges_of_the_same_count_overlap_only_when_they_share_a_slot():
    assert Slots(0, 3, 9).overlaps(Slots(2, 5, 9))
    assert not Slots(0, 3, 9).overlaps(Slots(3, 5, 9))
    assert Slots(0, 1, 9).overlaps(Slots(4, 5, 7))  # another count: the photos can overlap


def test_a_range_that_starts_where_the_other_ends_does_not_overlap_it():
    assert not Slots(3, 5, 9).overlaps(Slots(0, 3, 9))


def test_an_empty_range_cannot_be_made():
    with pytest.raises(ValueError, match="slots must be I/N or I:J/N with 0 <= I < J <= N"):
        Slots(3, 3, 7)
