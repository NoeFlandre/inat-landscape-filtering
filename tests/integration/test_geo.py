"""H3 cells."""

import numpy as np
from h3.api import basic_int as h3

from inat_landscape.adapters.geo import h3_cells


def test_h3_cells_match_the_h3_library():
    lat = np.array([45.5, -12.0, 0.0])
    lon = np.array([3.25, 130.0, 0.0])
    cells = h3_cells(lat, lon, 5)
    assert cells.dtype == np.int64
    assert cells.tolist() == [h3.latlng_to_cell(a, b, 5) for a, b in zip(lat, lon, strict=True)]


def test_h3_cells_empty():
    assert h3_cells(np.array([]), np.array([]), 5).tolist() == []
