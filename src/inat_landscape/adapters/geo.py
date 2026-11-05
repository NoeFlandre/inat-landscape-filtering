"""H3 cells of coordinates."""

from itertools import chain

import numpy as np
from h3.api import basic_int as h3
from numpy.typing import NDArray

_cell = np.frompyfunc(h3.latlng_to_cell, 3, 1)


def h3_cells(lat: NDArray[np.floating], lon: NDArray[np.floating], res: int) -> NDArray[np.int64]:
    """Return the H3 cell id of each point at resolution `res`."""
    if len(lat) == 0:
        return np.zeros(0, dtype=np.int64)
    return _cell(lat, lon, res).astype(np.int64)


def cell_centers(cells: NDArray[np.integer]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the latitude and longitude of the centre of each H3 cell."""
    points = np.array([h3.cell_to_latlng(int(c)) for c in cells], dtype=float).reshape(-1, 2)
    return points[:, 0], points[:, 1]


def children(cells: NDArray[np.integer], res: int) -> NDArray[np.uint64]:
    """Return the cells of resolution `res` inside each of `cells`."""
    found = chain.from_iterable(h3.cell_to_children(int(c), res) for c in cells)
    return np.fromiter(found, np.uint64)
