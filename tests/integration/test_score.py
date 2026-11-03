"""Score stage."""

import json

import numpy as np
import pytest

from inat_landscape.adapters.images import fetch
from inat_landscape.domain.scoring import LANDSCAPE_PROMPTS
from inat_landscape.pipeline.embed import HUB_DIR, open_store, run_embed
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.score import EMBEDDINGS, SCORED, run_score
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
    assert table["photo_id"].to_pylist() == [101, 107, 102]  # uniform order
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


def test_score_streams_the_shards_from_a_new_machine_and_keeps_the_first_copy(embedded, tmp_path):
    workdir, hub = embedded
    other = tmp_path / "other"
    prepared(other)
    store = open_store(other, FakeEncoder.model_id, hub)
    store.add([101], np.full((1, 4), 7.0))  # a second copy of 101, in a later shard name
    first = run_score(workdir, FakeEncoder(), hub)
    again = run_score(other, FakeEncoder(), hub)
    assert again["photo_id"].to_pylist() == first["photo_id"].to_pylist()
    assert sorted(again["score_contrast"].to_pylist()) == sorted(
        first["score_contrast"].to_pylist()
    )
    assert json.loads((other / EMBEDDINGS).read_text()) == {"embedding_dim": 4, "embedding_rows": 3}
    assert not list((other / HUB_DIR).rglob(".stream/**/*.parquet"))
