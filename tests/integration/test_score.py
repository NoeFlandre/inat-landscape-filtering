"""Score stage."""

import numpy as np
import pytest

from inat_landscape.adapters.images import fetch
from inat_landscape.domain.scoring import LANDSCAPE_PROMPTS
from inat_landscape.pipeline.embed import run_embed
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.score import SCORED, run_score
from tests.fakes import FakeEncoder, FakeHub
from tests.integration.pipeline_helpers import prepared, serve_photos


@pytest.fixture
def embedded(tmp_path, server, monkeypatch):
    _, handler, base = server
    monkeypatch.setattr(
        "inat_landscape.pipeline.embed.PHOTO_URL", base + "/photos/{photo_id}/{size}.{extension}"
    )
    serve_photos(handler, skip=(103,))
    prepared(tmp_path / "w")
    (tmp_path / "remote").mkdir()
    hub = FakeHub(tmp_path / "remote")
    run_embed(tmp_path / "w", FakeEncoder(), hub, fetcher=lambda u: fetch(u, backoff=0.0))
    return tmp_path / "w", hub


def test_scores_only_embedded_photos_in_sample_order(embedded):
    workdir, hub = embedded
    table = run_score(workdir, FakeEncoder(), hub)
    assert table["photo_id"].to_pylist() == [101, 102, 107]
    assert (workdir / SCORED).exists()


def test_score_columns_are_probabilities(embedded):
    workdir, hub = embedded
    table = run_score(workdir, FakeEncoder(), hub)
    names = ["score_landscape", "score_closeup"] + [
        f"score_landscape_{i}" for i in range(1, len(LANDSCAPE_PROMPTS) + 1)
    ]
    for name in names:
        values = np.array(table[name].to_pylist())
        assert np.all((values >= 0) & (values <= 1)), name


def test_score_needs_embeddings(tmp_path):
    prepared(tmp_path)
    (tmp_path / "remote").mkdir()
    with pytest.raises(StageError, match="embed"):
        run_score(tmp_path, FakeEncoder(), FakeHub(tmp_path / "remote"))


def test_scores_do_not_depend_on_the_scoring_chunk(embedded, monkeypatch):
    workdir, hub = embedded
    whole = run_score(workdir, FakeEncoder(), hub)
    monkeypatch.setattr("inat_landscape.pipeline.score.SCORE_ROWS", 1)
    chunked = run_score(workdir, FakeEncoder(), hub)
    assert whole.column_names == chunked.column_names
    for name in whole.column_names:
        if name.startswith("score_"):  # last-bit differences between BLAS paths
            assert np.allclose(whole[name].to_numpy(), chunked[name].to_numpy(), atol=1e-12)
        else:
            assert whole[name].equals(chunked[name]), name
