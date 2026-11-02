"""Sample stage: first n candidates in sample order."""

import io

import pytest

from inat_landscape.pipeline.ingest import run_ingest
from inat_landscape.pipeline.sample import SAMPLE, StageError, run_sample
from tests import fakes


@pytest.fixture
def workdir(tmp_path):
    run_ingest(tmp_path, "fake", 0.25, opener=lambda _s: io.BytesIO(fakes.make_dump()))
    return tmp_path


def test_sample_takes_first_n_in_order(workdir):
    table = run_sample(workdir, 3)
    assert table["photo_id"].to_pylist() == [101, 102, 103]
    assert (workdir / SAMPLE).exists()


def test_sample_larger_than_candidates_keeps_all(workdir):
    assert run_sample(workdir, 100).num_rows == 4


def test_sample_needs_ingest_first(tmp_path):
    with pytest.raises(StageError, match="ingest"):
        run_sample(tmp_path, 3)
