"""Embed stage: resumable, never embeds a photo twice."""

import pytest

from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.files import read_table
from inat_landscape.adapters.hub import OfflineHub
from inat_landscape.adapters.images import fetch
from inat_landscape.pipeline.embed import (
    FAILURES,
    HUB_DIR,
    EmbedOptions,
    embedding_dir,
    plan_embed,
    run_embed,
)
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
    options = EmbedOptions(chunk=2, **kw)
    return run_embed(
        workdir, encoder, hub, fetcher=lambda u: fetch(u, backoff=0.0), options=options
    )


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


def test_plan_counts_the_work_left_without_the_model(setup):
    workdir, handler, hub = setup
    serve_photos(handler)
    assert plan_embed(workdir, FakeEncoder.model_id, hub, worker=(1, 2)) == 3
    _embed(workdir, hub, FakeEncoder(), worker=(1, 2))
    assert plan_embed(workdir, FakeEncoder.model_id, hub, worker=(1, 2)) == 0
    assert plan_embed(workdir, FakeEncoder.model_id, hub) == 1


def test_shards_gather_several_download_batches(setup):
    workdir, handler, hub = setup
    serve_photos(handler)
    options = EmbedOptions(chunk=3, download_batch=1)
    report = run_embed(
        workdir, FakeEncoder(), hub, fetcher=lambda u: fetch(u, backoff=0.0), options=options
    )
    assert report.embedded == 4
    assert len(hub.uploads) == 2  # 3 photos, then 1


def test_a_download_error_stops_the_run(setup):
    workdir, _handler, hub = setup

    def broken(_urls):
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        run_embed(workdir, FakeEncoder(), hub, fetcher=broken, options=EmbedOptions(chunk=2))


def test_parallel_workers_keep_separate_failure_files(setup):
    workdir, handler, hub = setup
    serve_photos(handler, skip=(101, 102))
    _embed(workdir, hub, FakeEncoder(), worker=(0, 2))  # 102 fails
    _embed(workdir, hub, FakeEncoder(), worker=(1, 2))  # 101 fails
    zero = read_table(workdir / "failures-0-of-2.parquet")["photo_id"].to_pylist()
    one = read_table(workdir / "failures-1-of-2.parquet")["photo_id"].to_pylist()
    assert (zero, one) == ([102], [101])
    assert not (workdir / FAILURES).exists()


def test_max_photos_stops_the_run_early_with_its_work_saved(setup):
    workdir, handler, hub = setup
    serve_photos(handler)
    report = _embed(workdir, hub, FakeEncoder(), max_photos=2)
    assert report.embedded == 2
    assert plan_embed(workdir, FakeEncoder.model_id, hub) == 2
    assert _embed(workdir, hub, FakeEncoder()).embedded == 2


def test_workers_skip_the_permanent_failures_of_the_shared_failure_file(setup):
    # Bug: a worker read only its own failure file. On a fresh node it tried the known 404
    # photos again, so its plan never reached 0 and its job chain never ended.
    workdir, handler, hub = setup
    serve_photos(handler, skip=(102,))
    _embed(workdir, hub, FakeEncoder())  # one worker: 102 goes to the shared failure file
    (workdir / "failures-0-of-2.parquet").unlink(missing_ok=True)
    assert plan_embed(workdir, FakeEncoder.model_id, hub, worker=(0, 2)) == 0
    report = _embed(workdir, hub, FakeEncoder(), worker=(0, 2))
    assert (report.embedded, report.failed, report.skipped) == (0, 0, 1)


def test_a_worker_tags_its_shards_and_reads_no_other_tagged_shard(setup):
    workdir, handler, hub = setup
    serve_photos(handler)
    _embed(workdir, hub, FakeEncoder(), worker=(1, 2))
    names = [p.rsplit("/", 1)[-1] for p in hub.list_files("embeddings/")]
    assert names
    assert all(n.startswith("part-w1of2-") for n in names)
    assert plan_embed(workdir, FakeEncoder.model_id, hub, worker=(0, 2)) == 1  # 102 only
