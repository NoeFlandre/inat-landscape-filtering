"""Land stage: only the H3 cells that touch land are kept (ADR 0013)."""

import json

import h3
import numpy as np
import pytest

from inat_landscape.adapters.land import LAND_SOURCES, fetch_land, land_cells
from inat_landscape.pipeline.embed import read_sample
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.land import LAND_REPORT, on_land, run_land
from tests.integration.pipeline_helpers import prepared

# A small square around the photos 101 and 102 (45.5 N, 3.25 E); 103 and 107 are "at sea".
SQUARE = {
    "type": "Polygon",
    "coordinates": [[[3.0, 45.3], [3.5, 45.3], [3.5, 45.7], [3.0, 45.7], [3.0, 45.3]]],
}


def test_land_cells_are_the_cells_that_touch_the_polygon():
    cells = land_cells([SQUARE], 5)
    assert h3.str_to_int(h3.latlng_to_cell(45.5, 3.25, 5)) in cells
    assert h3.str_to_int(h3.latlng_to_cell(45.3, 3.0, 5)) in cells  # a corner: overlap
    assert h3.str_to_int(h3.latlng_to_cell(10.0, 20.0, 5)) not in cells
    assert list(cells) == sorted(cells)


def test_land_cells_take_multipolygons():
    other = {"type": "Polygon", "coordinates": [[[20, 10], [20.1, 10], [20.1, 10.1], [20, 10]]]}
    multi = {"type": "MultiPolygon", "coordinates": [SQUARE["coordinates"], other["coordinates"]]}
    assert set(land_cells([multi], 5)) == set(land_cells([SQUARE, other], 5))


def test_run_land_keeps_the_sample_cells_on_land(tmp_path):
    prepared(tmp_path)
    report = run_land(tmp_path, lambda: [SQUARE])
    assert (report.sea_cells, report.sea_photos) == (2, 2)
    assert json.loads((tmp_path / LAND_REPORT).read_text()) == {
        "sources": list(LAND_SOURCES),
        "sea_cells": 2,
        "sea_photos": 2,
    }
    assert sorted(read_sample(tmp_path)["photo_id"].to_pylist()) == [101, 102]
    assert sorted(read_sample(tmp_path, ["photo_id"])["photo_id"].to_pylist()) == [101, 102]
    assert read_sample(tmp_path, ["photo_id"]).column_names == ["photo_id"]


def test_without_the_land_stage_every_cell_stays(tmp_path):
    prepared(tmp_path)
    assert on_land(tmp_path, np.array([1, 2])) is None
    assert len(read_sample(tmp_path, ["photo_id"])) == 4


def test_run_land_needs_a_sample(tmp_path):
    with pytest.raises(StageError, match="sample"):
        run_land(tmp_path, lambda: [SQUARE])


def test_fetch_land_reads_the_geometries_of_each_file(server):
    _, handler, base = server
    collection = {"type": "FeatureCollection", "features": [{"geometry": SQUARE}]}
    handler.routes["/land.geojson"] = json.dumps(collection).encode()
    assert fetch_land([base + "/land.geojson"] * 2) == [SQUARE, SQUARE]
