"""H3 cells and their centres."""

import numpy as np

from inat_landscape.adapters.geo import cell_centers, children, h3_cells


def test_cell_centers_are_inside_their_cells():
    lat, lon = np.array([45.5, -12.0]), np.array([3.25, 130.0])
    cells = h3_cells(lat, lon, 5)
    centre_lat, centre_lon = cell_centers(cells)
    assert h3_cells(centre_lat, centre_lon, 5).tolist() == cells.tolist()
    assert np.allclose(centre_lat, lat, atol=0.2)
    assert np.allclose(centre_lon, lon, atol=0.2)


def test_cell_centers_of_no_cell():
    lat, lon = cell_centers(np.zeros(0, np.int64))
    assert (len(lat), len(lon)) == (0, 0)


def test_children_are_the_finer_cells_of_each_cell():
    cells = h3_cells(np.array([45.5, -12.0]), np.array([3.25, 130.0]), 5)
    finer = children(cells, 6)
    assert len(finer) == 14  # 7 children each (no pentagon here)
    assert sorted(set(h3_cells(*cell_centers(finer), 5).tolist())) == sorted(cells.tolist())
    assert h3_cells(*cell_centers(finer), 6).tolist() == finer.tolist()
    assert len(children(np.zeros(0, np.uint64), 6)) == 0
