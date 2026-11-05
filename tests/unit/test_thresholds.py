"""Threshold configs: names and the parsing of a threshold list."""

import pytest

from inat_landscape.domain.thresholds import config_name, parse_thresholds


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


def test_parse_thresholds_sorts_and_dedups():
    assert parse_thresholds("0.5,0.1,0.5") == (0.1, 0.5)


@pytest.mark.parametrize("raw", ["", "abc", "1.5", "-0.1", "0.1,,0.2"])
def test_parse_thresholds_rejects_bad_input(raw):
    with pytest.raises(ValueError, match="threshold"):
        parse_thresholds(raw)


def test_parse_thresholds_accepts_both_bounds():
    assert parse_thresholds("1,0") == (0.0, 1.0)


def test_config_name_depends_on_the_score_column():
    assert config_name(0.05, "score_contrast") == "contrast_ge_0_05"
    assert config_name(0.05, "score_landscape") == "score_ge_0_05"
    with pytest.raises(ValueError, match="score"):
        config_name(0.05, "score_closeup")
