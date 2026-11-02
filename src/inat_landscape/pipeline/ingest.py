"""Ingest stage: stream the dump once and write the candidate photos table."""

from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pyarrow.compute as pc

from inat_landscape.adapters.files import read_json, read_table, write_json, write_table
from inat_landscape.adapters.inat_dump import DumpSlice, open_source, read_dump
from inat_landscape.domain.licensing import attribution
from inat_landscape.domain.sampling import sample_prefix
from inat_landscape.domain.taxa import classify

CANDIDATES = "ingest/candidates.parquet"
MANIFEST = "ingest/manifest.json"


def run_ingest(
    workdir: Path,
    source: str,
    fraction: float,
    *,
    force: bool = False,
    opener: Callable[[str], BinaryIO] = open_source,
) -> pa.Table:
    """Write the candidates of the `fraction` sample, or reuse them if they exist."""
    manifest = {"source": source, "fraction": fraction}
    previous = read_json(workdir / MANIFEST)
    if not force and previous is not None and _same(previous, manifest):
        return read_table(workdir / CANDIDATES)
    with opener(source) as stream:
        table = build_candidates(read_dump(stream, sample_prefix(fraction)))
    write_table(table, workdir / CANDIDATES)
    write_json(manifest | {"candidates": table.num_rows}, workdir / MANIFEST)
    return table


def _same(previous: dict, manifest: dict) -> bool:
    return all(previous.get(k) == v for k, v in manifest.items())


def build_candidates(dump: DumpSlice) -> pa.Table:
    """Join the dump tables and keep photos of the kept taxon groups."""
    taxa = dump.taxa.rename_columns(["taxon_id", "ancestry", "taxon_name"])
    observers = dump.observers.rename_columns(["observer_id", "login", "observer_name"])
    table = (
        dump.photos.join(dump.observations, "observation_uuid", join_type="inner")
        .join(taxa, "taxon_id", join_type="left outer")
        .join(observers, "observer_id", join_type="left outer")
    )
    groups = [
        classify(t, a)
        for t, a in zip(table["taxon_id"].to_pylist(), table["ancestry"].to_pylist(), strict=True)
    ]
    table = table.append_column("group", pa.array([g and g.value for g in groups], pa.string()))
    table = table.filter(pc.is_valid(table["group"])).drop_columns(["ancestry"])
    table = table.set_column(
        table.schema.get_field_index("license"), "license", pc.utf8_upper(table["license"])
    )
    table = table.append_column("attribution", pa.array(_attributions(table), pa.string()))
    return table.sort_by([("photo_id", "ascending")])


def _attributions(table: pa.Table) -> list[str]:
    rows = zip(
        table["license"].to_pylist(),
        table["observer_name"].to_pylist(),
        table["login"].to_pylist(),
        table["observer_id"].to_pylist(),
        strict=True,
    )
    return [
        attribution(lic, name, login or f"iNaturalist user {oid}") for lic, name, login, oid in rows
    ]
