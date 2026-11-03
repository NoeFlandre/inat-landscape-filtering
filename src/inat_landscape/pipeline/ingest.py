"""Ingest stage: three cached passes over the dump stream.

1. taxa: read the taxa table, to know which taxa are eligible;
2. index: count eligible photos per H3 cell and select the first `cap` of each cell;
3. details: read the full metadata of the selected photos only.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from inat_landscape.adapters.files import read_json, read_table, write_json, write_table
from inat_landscape.adapters.inat_dump import (
    DumpSlice,
    PhotoIndex,
    Selection,
    index_photos,
    open_source,
    read_dump,
    recap,
    scan_taxa,
)
from inat_landscape.domain.sampling import sample_prefix
from inat_landscape.domain.taxa import LIFE, TAXA_SCOPE, Group, classify

TAXA = "ingest/taxa.parquet"
SELECTED = "ingest/selected.parquet"
COUNTS = "ingest/cell_counts.parquet"
CANDIDATES = "ingest/candidates.parquet"
MANIFEST = "ingest/manifest.json"
_TAXA_MANIFEST = "ingest/taxa.json"
_INDEX_MANIFEST = "ingest/index.json"

Opener = Callable[[str], BinaryIO]


@dataclass(frozen=True)
class IngestParams:
    """What defines an ingest run."""

    source: str
    fraction: float
    res: int
    cap: int
    taxa: str = TAXA_SCOPE  # the taxa scope (ADR 0004)


def run_index(
    workdir: Path, params: IngestParams, *, force: bool = False, opener: Opener = open_source
) -> PhotoIndex:
    """Return the per-cell counts and the capped selection, computing them if needed."""
    stored = read_json(workdir / _INDEX_MANIFEST)
    if not force and _covers(stored, params):
        cached = PhotoIndex(read_table(workdir / SELECTED), read_table(workdir / COUNTS))
        return recap(cached, params.cap)
    taxa = _taxa(workdir, params.source, opener)
    with opener(params.source) as stream:
        index = index_photos(
            stream,
            eligible_taxa(taxa),
            sample_prefix(params.fraction),
            res=params.res,
            cap=params.cap,
        )
    write_table(index.selected, workdir / SELECTED)
    write_table(index.counts, workdir / COUNTS)
    write_json(asdict(params), workdir / _INDEX_MANIFEST)
    return index


def run_ingest(
    workdir: Path, params: IngestParams, *, force: bool = False, opener: Opener = open_source
) -> pa.Table:
    """Write the candidate photos with full metadata, or reuse them if they exist."""
    stored = read_json(workdir / MANIFEST)
    if not force and stored is not None and stored.get("params") == asdict(params):
        return read_table(workdir / CANDIDATES)
    index = run_index(workdir, params, force=force, opener=opener)
    selected = index.selected
    selection = Selection.of(
        selected["observation_key"].to_numpy(), selected["photo_id"].to_numpy()
    )
    with opener(params.source) as stream:
        table = build_candidates(read_dump(stream, selection), selected)
    write_table(table, workdir / CANDIDATES)
    write_json({"params": asdict(params), "candidates": table.num_rows}, workdir / MANIFEST)
    return table


def _covers(stored: dict | None, params: IngestParams) -> bool:
    """Tell if a stored index has the same pool and a cap at least as large."""
    wanted = asdict(params)
    return (
        stored is not None
        and all(stored.get(k) == wanted[k] for k in ("source", "fraction", "res", "taxa"))
        and stored.get("cap", -1) >= params.cap
    )


def eligible_taxa(taxa: pa.Table) -> NDArray[np.int64]:
    """Return the sorted ids of the taxa that the dataset keeps: all of them (ADR 0004)."""
    return np.sort(taxa["taxon_id"].to_numpy(zero_copy_only=False).astype(np.int64))


def _taxa(workdir: Path, source: str, opener: Opener) -> pa.Table:
    if read_json(workdir / _TAXA_MANIFEST) == {"source": source}:
        return read_table(workdir / TAXA)
    with opener(source) as stream:
        taxa = scan_taxa(stream)
    write_table(taxa, workdir / TAXA)
    write_json({"source": source}, workdir / _TAXA_MANIFEST)
    return taxa


def build_candidates(dump: DumpSlice, selected: pa.Table) -> pa.Table:
    """Join the dump tables, and add the H3 cell and the taxon group of each photo."""
    taxa = dump.taxa.rename_columns(["taxon_id", "ancestry", "taxon_name"])
    observers = dump.observers.rename_columns(["observer_id", "login", "observer_name"])
    table = (
        dump.photos.join(dump.observations, "observation_uuid", join_type="inner")
        .join(taxa, "taxon_id", join_type="left outer")
        .join(observers, "observer_id", join_type="left outer")
        .join(selected.select(["photo_id", "h3_cell"]), "photo_id", join_type="inner")
    )
    table = table.append_column("group", groups(table["taxon_id"], taxa))
    table = table.drop_columns(["ancestry"])
    table = table.set_column(
        table.schema.get_field_index("license"), "license", pc.utf8_upper(table["license"])
    )
    table = table.append_column("attribution", attributions(table))
    return _large_strings(table).sort_by([("photo_id", "ascending")])


def _large_strings(table: pa.Table) -> pa.Table:
    """Return the table with 64-bit string offsets: a column of tens of millions of texts
    passes the 2 GB of `string` when a sort or a take joins its chunks."""
    schema = pa.schema(
        f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f for f in table.schema
    )
    return table.cast(schema)


def groups(taxon_ids: pa.Array | pa.ChunkedArray, taxa: pa.Table) -> pa.Array:
    """Return the group of each taxon id (`classify`), computed once for each taxon.

    `taxa` has the `taxon_id` and `ancestry` of the taxa. An id that is not in `taxa` has no
    known ancestry.
    """
    pairs = zip(taxa["taxon_id"].to_pylist(), taxa["ancestry"].to_pylist(), strict=True)
    names = pa.array([classify(t, a).value for t, a in pairs], pa.string())
    found = pc.take(names, pc.index_in(taxon_ids, taxa["taxon_id"]))
    unknown = pc.if_else(  # classify(id, None): no taxon or Life is unidentified, else other
        pc.fill_null(pc.equal(taxon_ids, LIFE), True), Group.UNIDENTIFIED.value, Group.OTHER.value
    )
    return pc.coalesce(found, unknown)


def attributions(table: pa.Table) -> pa.Array:
    """Return the attribution text of each photo (`attribution`), for whole columns at once."""
    code = pc.utf8_upper(table["license"])
    fallback = pc.binary_join_element_wise(
        "iNaturalist user ", pc.cast(table["observer_id"], pa.string()), ""
    )
    who = pc.coalesce(_text(table["observer_name"]), _text(table["login"]), fallback)
    cc0 = pc.binary_join_element_wise(who, ", no rights reserved (CC0)", "")
    other = pc.binary_join_element_wise("© ", who, ", some rights reserved (", code, ")", "")
    return pc.if_else(pc.equal(code, "CC0"), cc0, other)


def _text(column: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Return the column with empty strings as nulls (`name or login` in the domain rule)."""
    return pc.if_else(pc.equal(column, ""), pa.scalar(None, pa.string()), column)
