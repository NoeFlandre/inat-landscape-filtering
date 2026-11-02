"""Score summary used to pick thresholds."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.stats import histogram, summarize

scores = st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=200)


@given(scores, st.integers(1, 20))
def test_histogram_counts_every_score_once(s, bins):
    counts = histogram(s, bins)
    assert len(counts) == bins
    assert sum(counts) == len(s)


def test_histogram_puts_one_in_last_bin():
    assert histogram([0.0, 0.5, 1.0], 2) == [1, 2]


def test_histogram_rejects_zero_bins():
    with pytest.raises(ValueError, match="bins"):
        histogram([0.5], 0)


def test_summarize_reports_quantiles_and_counts_above():
    text = summarize([0.0, 0.1, 0.2, 0.9], thresholds=(0.15, 0.5), bins=2)
    assert "n=4" in text
    assert ">= 0.15: 2" in text
    assert ">= 0.5: 1" in text
    assert "[0.50, 1.00]" in text


def test_summarize_rejects_empty_scores():
    with pytest.raises(ValueError, match=r"^no scores to summarize$"):
        summarize([], thresholds=(), bins=2)


def test_summarize_exact_report():
    scores = [i / 100 for i in range(101)]
    lines = summarize(scores, thresholds=(0.5,), bins=4).splitlines()
    assert lines[0] == "n=101 min=0.0000 max=1.0000"
    assert lines[1] == "quantiles: p50=0.5000 p90=0.9000 p95=0.9500 p99=0.9900"
    assert lines[2] == ">= 0.5: 51"
    assert lines[3] == "[0.00, 0.25]      25 " + "#" * 25
    assert lines[6] == "[0.75, 1.00]      26 " + "#" * 26
    assert len(lines) == 7


def test_summarize_single_score_and_bar_cap():
    lines = summarize([0.2] * 61, thresholds=(), bins=1).splitlines()
    assert lines[1] == "quantiles: p50=0.2000 p90=0.2000 p95=0.2000 p99=0.2000"
    assert lines[2] == "[0.00, 1.00]      61 " + "#" * 60


def test_summarize_one_score():
    assert (
        summarize([0.3], thresholds=(), bins=1).splitlines()[1].startswith("quantiles: p50=0.3000")
    )


def test_summarize_uses_twenty_bins_by_default():
    assert len(summarize([0.5, 0.6], thresholds=()).splitlines()) == 2 + 20
