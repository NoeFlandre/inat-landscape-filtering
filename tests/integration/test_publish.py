"""Publish stage: configs, images, card, upload."""

import json

import pyarrow.parquet as pq
import pytest

from inat_landscape.adapters.images import fetch
from inat_landscape.domain.card import fields
from inat_landscape.domain.scoring import LANDSCAPE_PROMPTS
from inat_landscape.pipeline.embed import HUB_DIR, run_embed
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.publish import run_publish
from inat_landscape.pipeline.score import run_score
from tests.fakes import FakeEncoder, FakeHub
from tests.integration.pipeline_helpers import prepared, serve_photos


@pytest.fixture
def scored(tmp_path, server, monkeypatch):
    _, handler, base = server
    url = base + "/photos/{photo_id}/{size}.{extension}"
    monkeypatch.setattr("inat_landscape.pipeline.embed.PHOTO_URL", url)
    serve_photos(handler)
    prepared(tmp_path / "w")
    (tmp_path / "remote").mkdir()
    hub = FakeHub(tmp_path / "remote")
    run_embed(tmp_path / "w", FakeEncoder(), hub, fetcher=lambda u: fetch(u, backoff=0.0))
    table = run_score(tmp_path / "w", FakeEncoder(), hub)
    scores = sorted(table["score_landscape"].to_pylist())
    return tmp_path / "w", hub, handler, scores


def _publish(workdir, hub, thresholds):
    return run_publish(
        workdir,
        thresholds,
        hub,
        model_id=FakeEncoder.model_id,
        fetcher=lambda u: fetch(u, backoff=0.0),
    )


def test_writes_all_config_with_urls_and_no_images(scored):
    workdir, hub, _, _ = scored
    _publish(workdir, hub, (0.0,))
    table = pq.read_table(workdir / HUB_DIR / "data/all/part-00000.parquet")
    assert table.num_rows == 4
    assert "image" not in table.column_names
    row = table.to_pylist()[0]
    assert row["url_medium"].endswith("/photos/101/medium.jpg")
    assert row["url_small"].endswith("/photos/101/small.jpg")


def test_threshold_configs_are_nested_and_carry_images(scored):
    workdir, hub, _, scores = scored
    low, high = scores[1], scores[3]
    report = _publish(workdir, hub, (low, high))
    assert [s.rows for s in report.subsets] == [3, 1]
    path = workdir / HUB_DIR / f"data/{report.subsets[1].config}/part-00000.parquet"
    table = pq.read_table(path)
    image = table["image"].to_pylist()[0]
    assert image["bytes"][:2] == b"\xff\xd8"
    assert (
        image["path"].endswith(".jpg")
        or image["path"].endswith(".jpeg")
        or image["path"].endswith(".png")
    )
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    assert features["image"] == {"_type": "Image"}
    assert features["latitude"] == {"dtype": "float64", "_type": "Value"}


def test_card_figures_and_upload(scored):
    workdir, hub, _, _ = scored
    _publish(workdir, hub, (0.0,))
    card = (workdir / HUB_DIR / "README.md").read_text()
    assert "config_name: score_ge_0_00" in card
    assert (
        "**4 iNaturalist photos of plants and unidentified observations (4 with a location)" in card
    )
    assert "| plantae | 2 | 2 |" in card
    assert "This gave 4 candidate photos" in card
    for name in ("hero", "map", "scores"):
        png = workdir / HUB_DIR / "assets" / f"{name}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert f"](assets/{name}.png)" in card
    [(allow, delete)] = hub.folder_uploads
    assert allow == ["README.md", "data/**", "assets/**"]
    assert delete == ["data/**", "assets/**"]
    assert {"README.md", "assets/map.png"} <= set(hub.list_files(""))


def test_card_documents_exactly_the_published_columns(scored):
    workdir, hub, _, _ = scored
    report = _publish(workdir, hub, (0.0,))
    path = workdir / HUB_DIR / f"data/{report.subsets[0].config}/part-00000.parquet"
    columns = set(pq.read_schema(path).names)
    documented = {n for f in fields(len(LANDSCAPE_PROMPTS)) for n in f.names}
    assert columns == documented


def test_stale_threshold_configs_are_removed(scored):
    workdir, hub, _, _ = scored
    _publish(workdir, hub, (0.0,))
    _publish(workdir, hub, (0.5,))
    names = sorted(p.name for p in (workdir / HUB_DIR / "data").iterdir())
    assert names == ["all", "score_ge_0_50"]


def test_medium_images_are_cached(scored):
    workdir, hub, handler, _ = scored
    _publish(workdir, hub, (0.0,))
    handler.routes.clear()
    report = _publish(workdir, hub, (0.0,))
    assert report.subsets[0].rows == 4
    assert report.images_failed == 0


def test_photo_without_medium_image_is_left_out_of_image_configs(scored):
    workdir, hub, handler, _ = scored
    del handler.routes["/photos/102/medium.jpeg"]
    report = _publish(workdir, hub, (0.0,))
    assert report.subsets[0].rows == 3
    assert report.images_failed == 1


def test_publish_needs_scores(tmp_path):
    with pytest.raises(StageError, match="score"):
        run_publish(tmp_path, (0.1,), FakeHub(tmp_path), model_id="m")
