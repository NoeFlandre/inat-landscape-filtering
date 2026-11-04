"""Land polygons (Natural Earth) and the H3 cells that touch them."""

from collections.abc import Iterable

import h3
import httpx
import numpy as np

# Natural Earth 1:10m land and minor islands, at a fixed release (public domain).
LAND_SOURCES = tuple(
    f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/{name}.geojson"
    for name in ("ne_10m_land", "ne_10m_minor_islands")
)


def fetch_land(urls: Iterable[str] = LAND_SOURCES) -> list[dict]:
    """Download the land GeoJSON files and return their geometries."""
    geometries = []
    for url in urls:
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        geometries.extend(f["geometry"] for f in response.json()["features"])
    return geometries


def land_cells(geometries: Iterable[dict], res: int) -> np.ndarray:
    """Return the sorted H3 cells (as integers) of resolution `res` that touch the land."""
    cells: set[int] = set()
    for geometry in geometries:
        shape = h3.geo_to_h3shape(geometry)
        found = h3.h3shape_to_cells_experimental(shape, res, contain="overlap")
        cells.update(h3.str_to_int(c) for c in found)
    return np.array(sorted(cells), np.uint64)
