"""Stream the iNaturalist Open Data metadata dump and keep a small sampled slice.

The dump is a gzipped tar with four tab-separated files. It is too large to store, so
this module reads it once as a stream and filters each file while it parses it.
"""

import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv

from inat_landscape.domain.licensing import ALLOWED_LICENSES

SNAPSHOT_URL = (
    "https://inaturalist-open-data.s3.amazonaws.com/metadata/inaturalist-open-data-20261027.tar.gz"
)
PHOTO_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{extension}"

_SCHEMAS: dict[str, dict[str, pa.DataType]] = {
    "observations.csv": {
        "observation_uuid": pa.string(),
        "observer_id": pa.int64(),
        "latitude": pa.float64(),
        "longitude": pa.float64(),
        "positional_accuracy": pa.int64(),
        "taxon_id": pa.int64(),
        "quality_grade": pa.string(),
        "observed_on": pa.string(),
        "anomaly_score": pa.float64(),
    },
    "photos.csv": {
        "photo_uuid": pa.string(),
        "photo_id": pa.int64(),
        "observation_uuid": pa.string(),
        "extension": pa.string(),
        "license": pa.string(),
        "width": pa.int64(),
        "height": pa.int64(),
        "position": pa.int64(),
    },
    "taxa.csv": {"taxon_id": pa.int64(), "ancestry": pa.string(), "name": pa.string()},
    "observers.csv": {"observer_id": pa.int64(), "login": pa.string(), "name": pa.string()},
}
_BLOCK_SIZE = 64 << 20

Predicate = Callable[[pa.RecordBatch], pa.Array]


class DumpError(RuntimeError):
    """The dump does not have the expected content or order."""


@dataclass(frozen=True)
class DumpSlice:
    """The sampled part of the dump."""

    observations: pa.Table
    photos: pa.Table
    taxa: pa.Table
    observers: pa.Table


def open_source(source: str) -> BinaryIO:
    """Open the dump from an https URL or a local path."""
    if source.startswith("https://"):
        return urllib.request.urlopen(source)  # noqa: S310 - scheme checked above
    return Path(source).open("rb")


def read_dump(stream: BinaryIO, prefix: str) -> DumpSlice:
    """Read the dump stream and keep the observations in the sample of `prefix`.

    Photos, taxa and observers are kept only when a kept observation refers to them.
    Photos are also kept only when their licence is allowed.
    """
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        tables = _read_members(tar, prefix)
    missing = sorted(set(_SCHEMAS) - set(tables))
    if missing:
        raise DumpError(f"dump has no {', '.join(missing)}")
    return DumpSlice(
        observations=tables["observations.csv"],
        photos=tables["photos.csv"],
        taxa=tables["taxa.csv"],
        observers=tables["observers.csv"],
    )


def _read_members(tar: tarfile.TarFile, prefix: str) -> dict[str, pa.Table]:
    tables: dict[str, pa.Table] = {}
    for member in tar:
        name = Path(member.name).name
        handle = tar.extractfile(member) if name in _SCHEMAS else None
        if handle is not None:
            predicate = _predicate(name, prefix, tables.get("observations.csv"))
            tables[name] = _read_filtered(handle, name, predicate)
    return tables


def _predicate(name: str, prefix: str, observations: pa.Table | None) -> Predicate:
    if name == "observations.csv":
        return lambda b: pc.less(
            pc.utf8_lower(pc.utf8_slice_codeunits(b["observation_uuid"], 0, 8)), prefix
        )
    if observations is None:
        raise DumpError(f"{name} comes before observations.csv in the dump")
    if name == "photos.csv":
        uuids = observations["observation_uuid"].combine_chunks()
        allowed = pa.array(ALLOWED_LICENSES)
        return lambda b: pc.and_(
            pc.is_in(b["observation_uuid"], uuids),
            pc.is_in(pc.utf8_upper(b["license"]), allowed),
        )
    key = "taxon_id" if name == "taxa.csv" else "observer_id"
    ids = pc.unique(observations[key].combine_chunks()).drop_null()
    return lambda b: pc.is_in(b[key], ids)


def _read_filtered(handle: IO[bytes], name: str, predicate: Predicate) -> pa.Table:
    schema = _SCHEMAS[name]
    reader = csv.open_csv(
        handle,
        read_options=csv.ReadOptions(block_size=_BLOCK_SIZE),
        parse_options=csv.ParseOptions(delimiter="\t", invalid_row_handler=lambda _row: "skip"),
        convert_options=csv.ConvertOptions(
            column_types=schema, include_columns=list(schema), strings_can_be_null=True
        ),
    )
    batches = [batch.filter(predicate(batch)) for batch in reader]
    return pa.Table.from_batches(batches, schema=pa.schema(schema))
