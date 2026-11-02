"""Sampling: deterministic, prefix-based, and stable when the sample grows."""

import random
import uuid

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.sampling import (
    Candidate,
    for_worker,
    in_sample,
    sample_prefix,
    select,
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


candidates = st.lists(
    st.builds(
        Candidate,
        observation_uuid=uuids,
        position=st.integers(0, 5),
        photo_id=st.integers(1, 10**9),
    ),
    unique_by=lambda c: c.photo_id,
    max_size=40,
)


@given(candidates, st.integers(0, 50), st.integers(0, 50))
def test_select_grows_as_superset(cands, a, b):
    small, large = sorted((a, b))
    assert set(select(cands, small)) <= set(select(cands, large))


@given(candidates, st.integers(0, 50))
def test_select_is_order_independent_and_sized(cands, n):
    chosen = select(cands, n)
    assert chosen == select(list(reversed(cands)), n)
    assert len(chosen) == min(n, len(cands))


def test_select_orders_by_observation_then_position_then_photo():
    a = Candidate("00000000-0000-4000-8000-000000000000", 1, 9)
    b = Candidate("00000000-0000-4000-8000-000000000000", 0, 10)
    c = Candidate("10000000-0000-4000-8000-000000000000", 0, 1)
    assert select([c, a, b], 3) == [10, 9, 1]


def test_select_rejects_negative_n():
    with pytest.raises(ValueError, match="n must be >= 0"):
        select([], -1)


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
