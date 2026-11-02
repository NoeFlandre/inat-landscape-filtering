"""Sample stage: first n candidates in sample order (all by default)."""

import io

import pytest

from inat_landscape.pipeline.ingest import IngestParams, run_ingest
from inat_landscape.pipeline.sample import SAMPLE, StageError, run_sample
from tests import fakes


@pytest.fixture
def workdir(tmp_path):
    params = IngestParams(source="fake", fraction=0.25, res=5, cap=1000)
    run_ingest(tmp_path, params, opener=lambda _s: io.BytesIO(fakes.make_dump()))
    return tmp_path


def test_sample_takes_first_n_in_order(workdir):
    table = run_sample(workdir, 3)
    assert table["photo_id"].to_pylist() == [101, 102, 103]
    assert table["sample_rank"].to_pylist() == [0, 1, 2]
    assert (workdir / SAMPLE).exists()


def test_sample_without_n_keeps_all_candidates(workdir):
    assert run_sample(workdir, None).num_rows == 4


def test_sample_larger_than_candidates_keeps_all(workdir):
    assert run_sample(workdir, 100).num_rows == 4


def test_sample_needs_ingest_first(tmp_path):
    with pytest.raises(StageError, match="ingest"):
        run_sample(tmp_path, 3)
