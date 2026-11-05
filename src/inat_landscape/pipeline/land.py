"""Land stage: keep only the H3 cells that touch land (ADR 0013)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from inat_landscape.adapters.files import read_table, write_json, write_table
from inat_landscape.adapters.land import LAND_SOURCES, fetch_land, land_cells
from inat_landscape.domain.members import in_sorted
from inat_landscape.pipeline.errors import require
from inat_landscape.pipeline.ingest import DEFAULT_RES, ingest_params
from inat_landscape.pipeline.sample import SAMPLE

LAND = "land.parquet"  # the sample cells that touch land; the other cells are left out
LAND_REPORT = "land.json"
LAND_GRID = "land-grid.parquet"  # all the land cells: the maps draw them in grey


@dataclass(frozen=True)
class LandReport:
    """The sample cells and photos that are left out because they are at sea."""

    sea_cells: int
    sea_photos: int


def run_land(workdir: Path, fetch: Callable[[], list[dict]] = fetch_land) -> LandReport:
    """Write the sample cells that touch land, and the count of what is left out.

    The sample and the candidates stay as they are: `read_sample` and `score` drop the
    photos of the other cells.
    """
    require(workdir / SAMPLE, "sample", "sample")
    res = ingest_params(workdir).get("res", DEFAULT_RES)
    sample_cells = read_table(workdir / SAMPLE, ["h3_cell"])["h3_cell"].to_numpy()
    cells, counts = np.unique(sample_cells, return_counts=True)
    grid = land_cells(fetch(), res)
    on_land = np.isin(cells.astype(np.uint64), grid)
    write_table(pa.table({"h3_cell": pa.array(cells[on_land])}), workdir / LAND)
    write_table(pa.table({"h3_cell": pa.array(grid)}), workdir / LAND_GRID)
    report = LandReport(int((~on_land).sum()), int(counts[~on_land].sum()))
    write_json({"sources": list(LAND_SOURCES), **vars(report)}, workdir / LAND_REPORT)
    return report


def on_land(workdir: Path, cells: np.ndarray) -> np.ndarray | None:
    """Return which of `cells` touch land, or None when the land stage did not run."""
    if not (workdir / LAND).exists():
        return None
    land = np.sort(read_table(workdir / LAND)["h3_cell"].to_numpy())
    return in_sorted(np.asarray(cells, land.dtype), land)


def land_only(workdir: Path, table: pa.Table) -> pa.Table:
    """Leave out the rows of the cells at sea, if the land stage ran."""
    keep = on_land(workdir, table["h3_cell"].to_numpy())
    return table if keep is None else table.filter(pa.array(keep))
