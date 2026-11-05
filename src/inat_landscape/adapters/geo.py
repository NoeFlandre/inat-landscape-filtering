"""H3 cells of coordinates, their parents and their outlines."""

import numpy as np
from h3.api import basic_int as h3
from numpy.typing import NDArray

_cell = np.frompyfunc(h3.latlng_to_cell, 3, 1)
_parent = np.frompyfunc(h3.cell_to_parent, 2, 1)


def h3_cells(lat: NDArray[np.floating], lon: NDArray[np.floating], res: int) -> NDArray[np.int64]:
    """Return the H3 cell id of each point at resolution `res`."""
    if len(lat) == 0:
        return np.zeros(0, dtype=np.int64)
    return _cell(lat, lon, res).astype(np.int64)


def parents(cells: NDArray[np.integer], res: int) -> NDArray[np.int64]:
    """Return the cell of resolution `res` that holds each of `cells`."""
    if len(cells) == 0:
        return np.zeros(0, dtype=np.int64)
    return _parent(cells.astype(object), res).astype(np.int64)


def outlines(cells: NDArray[np.integer]) -> list[NDArray[np.float64]]:
    """Return the corners of each cell, as rows of (longitude, latitude)."""
    return [np.array(h3.cell_to_boundary(int(c)), dtype=float)[:, ::-1] for c in cells]
