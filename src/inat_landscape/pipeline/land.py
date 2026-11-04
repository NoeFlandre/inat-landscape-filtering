"""Land stage: keep only the H3 cells that touch land (ADR 0013)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from inat_landscape.adapters.files import read_json, read_table, write_json, write_table
from inat_landscape.adapters.land import LAND_SOURCES, fetch_land, land_cells
from inat_landscape.domain.members import in_sorted
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ingest import MANIFEST
from inat_landscape.pipeline.sample import SAMPLE

LAND = "land.parquet"  # the sample cells that touch land; the other cells are left out
LAND_REPORT = "land.json"


@dataclass(frozen=True)
class LandReport:
    """The sample cells and photos that are left out because they are at sea."""

    sea_cells: int
    sea_photos: int


def run_land(workdir: Path, fetch: Callable[[], list[dict]] = fetch_land) -> LandReport:
    """Write the sample cells that touch land, and the count of what is left out.

    The sample and the candidates stay as they are: `read_sample` and `publish` drop the
    photos of the other cells.
    """
    if not (workdir / SAMPLE).exists():
        raise StageError(f"no sample at {workdir / SAMPLE}: run `sample` first")
    res = (read_json(workdir / MANIFEST) or {}).get("params", {}).get("res", 5)
    sample_cells = read_table(workdir / SAMPLE, ["h3_cell"])["h3_cell"].to_numpy()
    cells, counts = np.unique(sample_cells, return_counts=True)
    on_land = np.isin(cells.astype(np.uint64), land_cells(fetch(), res))
    write_table(pa.table({"h3_cell": pa.array(cells[on_land])}), workdir / LAND)
    report = LandReport(int((~on_land).sum()), int(counts[~on_land].sum()))
    write_json({"sources": list(LAND_SOURCES), **vars(report)}, workdir / LAND_REPORT)
    return report


def on_land(workdir: Path, cells: np.ndarray) -> np.ndarray | None:
    """Return which of `cells` touch land, or None when the land stage did not run."""
    if not (workdir / LAND).exists():
        return None
    land = np.sort(read_table(workdir / LAND)["h3_cell"].to_numpy())
    return in_sorted(np.asarray(cells, land.dtype), land)
