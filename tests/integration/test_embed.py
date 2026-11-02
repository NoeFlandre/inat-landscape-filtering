"""Embed stage: resumable, never embeds a photo twice."""

import pytest

from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.files import read_table
from inat_landscape.adapters.hub import OfflineHub
from inat_landscape.adapters.images import fetch
from inat_landscape.pipeline.embed import FAILURES, HUB_DIR, embedding_dir, run_embed
from inat_landscape.pipeline.errors import StageError
from tests.fakes import FakeEncoder, FakeHub
from tests.integration.pipeline_helpers import prepared, serve_photos


@pytest.fixture
def setup(tmp_path, server, monkeypatch):
    _, handler, base = server
    monkeypatch.setattr(
        "inat_landscape.pipeline.embed.PHOTO_URL", base + "/photos/{photo_id}/{size}.{extension}"
    )
    prepared(tmp_path / "w")
    (tmp_path / "remote").mkdir()
    return tmp_path / "w", handler, FakeHub(tmp_path / "remote")


def _embed(workdir, hub, encoder, **kw):
    return run_embed(workdir, encoder, hub, fetcher=lambda u: fetch(u, backoff=0.0), chunk=2, **kw)


def test_embeds_all_sampled_photos_in_chunks(setup):
    workdir, handler, hub = setup
    serve_photos(handler)
    report = _embed(workdir, hub, FakeEncoder())
    assert (report.total, report.already, report.embedded, report.failed) == (4, 0, 4, 0)
    assert len(hub.uploads) == 2
    store = EmbeddingStore(workdir / HUB_DIR, embedding_dir(FakeEncoder.model_id), OfflineHub())
    assert store.done_ids() == {101, 102, 103, 107}


def test_second_run_embeds_nothing(setup):
    workdir, handler, hub = setup
    serve_photos(handler)
    _embed(workdir, hub, FakeEncoder())
    encoder = FakeEncoder()
    report = _embed(workdir, hub, encoder)
    assert (report.already, report.embedded) == (4, 0)
    assert encoder.image_calls == 0


def test_new_machine_resumes_from_hub(setup, tmp_path):
    workdir, handler, hub = setup
    serve_photos(handler)
    _embed(workdir, hub, FakeEncoder())
    other = tmp_path / "other"
    prepared(other)
    encoder = FakeEncoder()
    assert _embed(other, hub, encoder).already == 4
    assert encoder.image_calls == 0


def test_missing_photo_is_recorded_and_skipped_next_time(setup):
    workdir, handler, hub = setup
    serve_photos(handler, skip=(102,))
    report = _embed(workdir, hub, FakeEncoder())
    assert (report.embedded, report.failed) == (3, 1)
    failures = read_table(workdir / FAILURES).to_pylist()
    assert failures == [{"photo_id": 102, "status": 404, "error": "Not Found", "permanent": True}]
    encoder = FakeEncoder()
    again = _embed(workdir, hub, encoder)
    assert (again.embedded, again.failed, again.skipped) == (0, 0, 1)


def test_corrupt_image_is_a_permanent_failure(setup):
    workdir, handler, hub = setup
    serve_photos(handler)
    handler.routes["/photos/103/small.jpg"] = b"garbage"
    report = _embed(workdir, hub, FakeEncoder())
    assert report.failed == 1
    [row] = read_table(workdir / FAILURES).to_pylist()
    assert row["photo_id"] == 103
    assert row["error"] == "cannot decode image"


def test_transient_failure_is_retried_on_next_run(setup):
    workdir, handler, hub = setup
    serve_photos(handler)
    handler.flaky["/photos/101/small.jpg"] = 100
    assert _embed(workdir, hub, FakeEncoder()).failed == 1
    handler.flaky.clear()
    again = _embed(workdir, hub, FakeEncoder())
    assert (again.embedded, again.failed) == (1, 0)
    assert read_table(workdir / FAILURES).num_rows == 0


def test_embed_needs_sample(tmp_path):
    with pytest.raises(StageError, match="sample"):
        run_embed(tmp_path, FakeEncoder(), OfflineHub())


def test_chunk_where_every_photo_fails_writes_no_shard(setup):
    workdir, _handler, hub = setup
    report = _embed(workdir, hub, FakeEncoder())
    assert (report.embedded, report.failed) == (0, 4)
    assert hub.uploads == []


def test_online_run_uploads_shards_made_offline(setup):
    # Regression: shards written with --offline were never uploaded by a later online run.
    workdir, handler, hub = setup
    serve_photos(handler)
    _embed(workdir, OfflineHub(), FakeEncoder())
    encoder = FakeEncoder()
    report = _embed(workdir, hub, encoder)
    assert (report.already, report.embedded, encoder.image_calls) == (4, 0, 0)
    assert len(hub.uploads) == 2


def test_workers_split_the_photos_without_overlap(setup, tmp_path):
    workdir, handler, hub = setup
    serve_photos(handler)
    other = tmp_path / "other"
    prepared(other)
    assert _embed(workdir, hub, FakeEncoder(), worker=(0, 2)).embedded == 1  # 102
    assert _embed(other, hub, FakeEncoder(), worker=(1, 2)).embedded == 3  # 101, 103, 107
    encoder = FakeEncoder()
    assert _embed(workdir, hub, encoder).already == 4
    assert encoder.image_calls == 0
