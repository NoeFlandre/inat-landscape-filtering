"""H3 cells, their parents and their outlines."""

import numpy as np

from inat_landscape.adapters.geo import h3_cells, outlines, parents

POINTS = np.array([45.5, -12.0, 45.6]), np.array([3.25, 130.0, 3.3])


def test_parents_hold_the_cells():
    cells = h3_cells(*POINTS, 5)
    found = parents(cells, 4)
    assert found.dtype == np.int64
    assert found.tolist() == h3_cells(*POINTS, 4).tolist()
    assert parents(np.zeros(0, np.int64), 4).tolist() == []


def test_parents_of_unsigned_ids():
    cells = h3_cells(*POINTS, 5)
    assert parents(cells.astype(np.uint64), 4).tolist() == parents(cells, 4).tolist()


def test_outlines_are_the_corners_around_each_cell():
    lat, lon = np.array([45.5, -12.0]), np.array([3.25, 130.0])
    rings = outlines(h3_cells(lat, lon, 4))
    assert [len(r) for r in rings] == [6, 6]
    for ring, x, y in zip(rings, lon, lat, strict=True):
        assert np.allclose(ring.mean(axis=0), [x, y], atol=0.3)  # longitude first
    assert outlines(np.zeros(0, np.int64)) == []
