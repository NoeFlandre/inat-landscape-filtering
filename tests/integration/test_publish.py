"""Publish stage: configs, images, card, upload."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from inat_landscape.adapters import figures
from inat_landscape.adapters.files import read_table
from inat_landscape.adapters.images import fetch
from inat_landscape.domain.card import fields
from inat_landscape.domain.scoring import LANDSCAPE_PROMPTS
from inat_landscape.pipeline.diversity import run_diversity
from inat_landscape.pipeline.embed import HUB_DIR, EmbedOptions, run_embed
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.land import run_land
from inat_landscape.pipeline.publish import PublishSpec, run_maps, run_publish
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


def _partly_scored(tmp_path, server, monkeypatch, max_photos):
    _, handler, base = server
    url = base + "/photos/{photo_id}/{size}.{extension}"
    monkeypatch.setattr("inat_landscape.pipeline.embed.PHOTO_URL", url)
    serve_photos(handler)
    prepared(tmp_path / "w")
    (tmp_path / "remote").mkdir()
    hub = FakeHub(tmp_path / "remote")
    options = EmbedOptions(max_photos=max_photos)
    run_embed(
        tmp_path / "w", FakeEncoder(), hub, fetcher=lambda u: fetch(u, backoff=0.0), options=options
    )
    run_score(tmp_path / "w", FakeEncoder(), hub)
    return tmp_path / "w", hub


def _publish(workdir, hub, thresholds, score="score_landscape", cap=100, **kw):
    spec = PublishSpec(tuple(thresholds), score=score, cap_per_cell=cap, **kw)
    return run_publish(
        workdir, spec, hub, model_id=FakeEncoder.model_id, fetcher=lambda u: fetch(u, backoff=0.0)
    )


def _remote(hub, config):
    """Read a config from the fake hub: all its part files, in order."""
    parts = sorted((hub.remote / "data" / config).glob("part-*.parquet"))
    return pa.concat_tables([pq.read_table(p) for p in parts]) if parts else None


def test_writes_all_config_with_urls_and_no_images(scored):
    workdir, hub, _, _ = scored
    _publish(workdir, hub, (0.0,))
    table = pq.read_table(workdir / HUB_DIR / "data/all/part-00000.parquet")
    assert table.num_rows == 4
    assert "image" not in table.column_names
    row = table.to_pylist()[0]
    assert row["url_medium"].endswith("/photos/101/medium.jpg")
    assert row["url_small"].endswith("/photos/101/small.jpg")


def test_image_configs_are_nested_carry_images_and_leave_no_local_parts(scored):
    workdir, hub, _, scores = scored
    low, high = scores[1], scores[3]
    report = _publish(workdir, hub, (low, high))
    assert [s.rows for s in report.subsets] == [3, 1]
    table = _remote(hub, report.subsets[1].config)
    image = table["image"].to_pylist()[0]
    assert image["bytes"][:2] == b"\xff\xd8"
    assert image["path"].rsplit(".", 1)[-1] in {"jpg", "jpeg", "png"}
    features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    assert features["image"] == {"_type": "Image"}
    assert features["latitude"] == {"dtype": "float64", "_type": "Value"}
    assert not list((workdir / HUB_DIR / "data").glob("score_ge_*/*.parquet"))


def test_image_parts_are_uploaded_one_commit_each(scored):
    workdir, hub, _, _ = scored
    hub.folder_uploads.clear()  # the embed run of the fixture committed its shards
    report = _publish(workdir, hub, (0.0, 0.5), part_rows=1)
    assert [
        len(list((hub.remote / "data" / s.config).glob("*.parquet"))) for s in report.subsets
    ] == [4, 4]
    # 4 part commits (each with both configs), then the card, `all` and figures
    assert len(hub.folder_uploads) == 5
    assert hub.folder_uploads[-1] == (
        ["README.md", "data/**", "assets/**"],
        ["data/all/**", "assets/**"],
    )
    assert _remote(hub, report.subsets[0].config).num_rows == 4


def test_metadata_only_publish_has_no_image_bytes(scored):
    workdir, hub, _, _ = scored
    hub.folder_uploads.clear()  # the embed run of the fixture committed its shards
    report = _publish(workdir, hub, (0.0,), images=False)
    local = pq.read_table(workdir / HUB_DIR / f"data/{report.subsets[0].config}/part-00000.parquet")
    assert "image" not in local.column_names
    assert local.num_rows == 4
    assert hub.folder_uploads == [(["README.md", "data/**", "assets/**"], ["data/**", "assets/**"])]
    card = (workdir / HUB_DIR / "README.md").read_text()
    assert "come with the URL of their 500 px image" in card


def test_card_figures_and_upload(scored):
    workdir, hub, _, _ = scored
    _publish(workdir, hub, (0.0,))
    card = (workdir / HUB_DIR / "README.md").read_text()
    assert "config_name: score_ge_0_00" in card
    assert "**4 iNaturalist photos of all taxa (4 with a location)" in card
    assert "| plantae | 2 | 2 |" in card
    assert "This gave 4 photos." in card
    assert "The run is complete: 4 of the 4 sample photos have an embedding (0 downloads" in card
    assert "come with the URL of their 500 px image" not in card
    for name in ("hero", "map", "scores"):
        png = workdir / HUB_DIR / "assets" / f"{name}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert f"](assets/{name}.png)" in card
    assert {"README.md", "assets/map.png"} <= set(hub.list_files(""))


def test_card_documents_exactly_the_published_columns(scored):
    workdir, hub, _, _ = scored
    report = _publish(workdir, hub, (0.0,))
    columns = set(_remote(hub, report.subsets[0].config).column_names)
    documented = {n for f in fields(len(LANDSCAPE_PROMPTS)) for n in f.names}
    assert columns == documented


def test_card_without_images_documents_exactly_the_published_columns(scored):
    workdir, hub, _, _ = scored
    report = _publish(workdir, hub, (0.0,), images=False)
    columns = set(_remote(hub, report.subsets[0].config).column_names)
    documented = {n for f in fields(len(LANDSCAPE_PROMPTS), with_images=False) for n in f.names}
    assert columns == documented


def test_stale_threshold_configs_are_removed(scored):
    workdir, hub, _, _ = scored
    _publish(workdir, hub, (0.0,))
    _publish(workdir, hub, (0.5,))
    assert _remote(hub, "score_ge_0_00") is None
    assert _remote(hub, "score_ge_0_50") is not None
    _publish(workdir, hub, (0.5,), images=False)
    names = sorted(p.name for p in (workdir / HUB_DIR / "data").iterdir())
    assert names == ["all", "score_ge_0_50"]


def test_photo_without_medium_image_is_left_out_of_image_configs(scored):
    workdir, hub, handler, _ = scored
    del handler.routes["/photos/102/medium.jpeg"]
    report = _publish(workdir, hub, (0.0,))
    assert report.subsets[0].rows == 3
    assert report.images_failed == 1
    assert 102 not in _remote(hub, report.subsets[0].config)["photo_id"].to_pylist()


def test_an_empty_config_still_has_one_file(scored):
    workdir, hub, _, _ = scored
    report = _publish(workdir, hub, (0.0, 0.999999))
    assert report.subsets[1].rows == 0
    assert _remote(hub, report.subsets[1].config).num_rows == 0


def test_publish_needs_scores(tmp_path):
    with pytest.raises(StageError, match="score"):
        run_publish(tmp_path, PublishSpec((0.1,)), FakeHub(tmp_path), model_id="m")


def test_cap_per_cell_keeps_the_best_scored_photos(scored):
    workdir, hub, _, _ = scored
    report = _publish(workdir, hub, (0.0,), cap=1)
    table = _remote(hub, report.subsets[0].config).to_pylist()
    cells = [r["h3_cell"] for r in table]
    assert len(cells) == len(set(cells)) == 3  # 4 photos in 3 cells
    plant = max(
        (
            r
            for r in pq.read_table(workdir / "scored.parquet").to_pylist()
            if r["photo_id"] in (101, 102)
        ),
        key=lambda r: r["score_landscape"],
    )
    assert plant["photo_id"] in {r["photo_id"] for r in table}


def test_contrast_score_configs(scored):
    workdir, hub, _, _ = scored
    report = _publish(workdir, hub, (0.0,), score="score_contrast")
    assert report.subsets[0].config == "contrast_ge_0_00"
    card = (workdir / HUB_DIR / "README.md").read_text()
    assert "`score_contrast` ≥ 0.0, best 100 per cell" in card


def test_card_shows_a_mosaic_of_example_photos_per_score_band(scored):
    workdir, hub, _, _ = scored
    _publish(workdir, hub, (0.0,))
    card = (workdir / HUB_DIR / "README.md").read_text()
    mosaic = workdir / HUB_DIR / "assets" / "examples.jpg"
    assert mosaic.read_bytes()[:2] == b"\xff\xd8"  # JPEG
    assert "](assets/examples.jpg)" in card
    assert "assets/examples.jpg" in set(hub.list_files(""))


def test_publish_keeps_only_the_complete_layers_while_the_run_goes_on(
    tmp_path, server, monkeypatch
):
    workdir, hub = _partly_scored(tmp_path, server, monkeypatch, max_photos=3)
    _publish(workdir, hub, (0.0,))
    published = read_table(workdir / HUB_DIR / "data" / "all" / "part-00000.parquet")
    assert sorted(published["photo_id"].to_pylist()) == [101, 103, 107]  # layer 0; 102 is next
    card = (workdir / HUB_DIR / "README.md").read_text()
    assert "each H3 cell has its first 1 photos" in card


def test_publish_with_no_complete_layer_is_an_error(tmp_path, server, monkeypatch):
    workdir, hub = _partly_scored(tmp_path, server, monkeypatch, max_photos=2)
    with pytest.raises(StageError, match="no complete layer"):
        _publish(workdir, hub, (0.0,))


def test_publish_after_the_land_stage_leaves_out_the_cells_at_sea(scored, monkeypatch):
    workdir, hub, _, _ = scored
    square = [[3.0, 45.3], [3.5, 45.3], [3.5, 45.7], [3.0, 45.7], [3.0, 45.3]]
    run_land(workdir, lambda: [{"type": "Polygon", "coordinates": [square]}])
    run_score(workdir, FakeEncoder(), hub)
    maps = []
    monkeypatch.setattr(figures, "world_map", lambda path, cells, **k: maps.append(cells))
    _publish(workdir, hub, (0.0,))
    card = (workdir / HUB_DIR / "README.md").read_text()
    assert "**2 iNaturalist photos of all taxa (2 with a location)" in card
    assert "the 2 photos of 2 cells at sea are left out." in card
    assert sorted(_remote(hub, "all")["photo_id"].to_pylist()) == [101, 102]
    assert maps == [([3.25], [45.5], [2])]  # one cell on land, with 101 and 102


def test_publish_draws_the_diversity_maps_after_the_diversity_stage(scored):
    workdir, hub, _, _ = scored
    run_diversity(workdir, FakeEncoder.model_id, hub, min_photos=2)
    _publish(workdir, hub, (0.0,))
    card = (workdir / HUB_DIR / "README.md").read_text()
    assert "](assets/diversity.png)" in card
    assert "50 random photos of the cell" in card
    assert "](assets/look-detail.png)" in card
    for name in ("diversity", "look", "look-detail"):
        assert (workdir / HUB_DIR / "assets" / f"{name}.png").read_bytes()[:4] == b"\x89PNG"


DIVERSITY_MAPS = ["assets/diversity.png", "assets/look.png", "assets/look-detail.png"]


def test_maps_only_uploads_the_diversity_maps(scored):
    workdir, hub, _, _ = scored
    run_diversity(workdir, FakeEncoder.model_id, hub, min_photos=2)
    assert run_maps(workdir, hub) == DIVERSITY_MAPS
    assert hub.folder_uploads[-1] == (DIVERSITY_MAPS, [])
    assert set(DIVERSITY_MAPS) <= set(hub.list_files("assets/"))
    assert "README.md" not in hub.list_files("")


def test_maps_only_draws_the_cell_map_again_from_the_local_all_config(scored, monkeypatch):
    workdir, hub, _, _ = scored
    _publish(workdir, hub, (0.0,))
    run_diversity(workdir, FakeEncoder.model_id, hub, min_photos=2)
    maps = []
    monkeypatch.setattr(figures, "world_map", lambda path, cells, **k: maps.append((cells, k)))
    assert run_maps(workdir, hub) == ["assets/map.png", *DIVERSITY_MAPS]
    assert len(maps) == 1
    assert sum(maps[0][0][2]) == len(_remote(hub, "all"))
    assert maps[0][1]["cap"] == 1000  # the cap of the run


def test_maps_draw_the_land_grid_in_grey_after_the_land_stage(scored, monkeypatch):
    workdir, hub, _, _ = scored
    square = [[3.0, 45.3], [3.5, 45.3], [3.5, 45.7], [3.0, 45.7], [3.0, 45.3]]
    run_land(workdir, lambda: [{"type": "Polygon", "coordinates": [square]}])
    run_diversity(workdir, FakeEncoder.model_id, hub, min_photos=1)
    lands = []
    monkeypatch.setattr(figures, "look_map", lambda path, *a, **k: lands.append(k["land"]))
    run_maps(workdir, hub)
    lon, lat = lands[0]
    assert len(lon) == len(lat) > 7  # the res 6 children of the land cells
    assert np.all((lon > 2.7) & (lon < 3.8) & (lat > 45.0) & (lat < 46.0))  # near the square
    assert lands[1] is lands[0]  # computed once for both colour maps


def test_maps_only_needs_the_diversity_stage(scored):
    workdir, hub, _, _ = scored
    with pytest.raises(StageError, match="diversity"):
        run_maps(workdir, hub)
