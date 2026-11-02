"""Threshold configs: names and nested subsets."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.thresholds import config_name, keep, parse_thresholds

scores = st.lists(st.floats(0, 1, allow_nan=False), max_size=30)
thresholds = st.floats(0, 1, allow_nan=False)


@pytest.mark.parametrize(
    ("t", "name"),
    [
        (0.3, "score_ge_0_30"),
        (0.05, "score_ge_0_05"),
        (0.125, "score_ge_0_125"),
        (1.0, "score_ge_1_00"),
    ],
)
def test_config_name(t, name):
    assert config_name(t) == name


@given(scores, thresholds, thresholds)
def test_higher_threshold_keeps_subset(s, a, b):
    low, high = sorted((a, b))
    kept_high = {i for i, k in enumerate(keep(s, high)) if k}
    kept_low = {i for i, k in enumerate(keep(s, low)) if k}
    assert kept_high <= kept_low


def test_keep_is_inclusive():
    assert keep([0.29, 0.3, 0.31], 0.3) == [False, True, True]


def test_parse_thresholds_sorts_and_dedups():
    assert parse_thresholds("0.5,0.1,0.5") == (0.1, 0.5)


@pytest.mark.parametrize("raw", ["", "abc", "1.5", "-0.1", "0.1,,0.2"])
def test_parse_thresholds_rejects_bad_input(raw):
    with pytest.raises(ValueError, match="threshold"):
        parse_thresholds(raw)


def test_parse_thresholds_accepts_both_bounds():
    assert parse_thresholds("1,0") == (0.0, 1.0)
