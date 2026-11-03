"""Stream the iNaturalist Open Data metadata dump and keep a small sampled slice.

The dump is a gzipped tar with four tab-separated files. It is too large to store, so
this module reads it once as a stream and filters each file while it parses it.
"""

import contextlib
import http.client
import io
import tarfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, BinaryIO, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray
from pyarrow import csv

from inat_landscape.adapters.geo import h3_cells
from inat_landscape.domain.cells import cap_first
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
_UUID_LENGTH = 36
_HEX = np.zeros(256, dtype=np.uint64)
for _i, _c in enumerate("0123456789abcdef"):
    _HEX[ord(_c)] = _HEX[ord(_c.upper())] = _i
_SHIFTS = np.arange(60, -1, -4, dtype=np.uint64)

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


@dataclass(frozen=True)
class Selection:
    """Observations (by key) and photos (by id) to read in full."""

    observation_keys: NDArray[np.uint64]  # sorted
    photo_ids: pa.Array

    @classmethod
    def of(cls, observation_keys: object, photo_ids: object) -> "Selection":
        keys = np.unique(np.asarray(observation_keys, dtype=np.uint64))
        return cls(keys, pa.array(np.asarray(photo_ids, dtype=np.int64)))


@dataclass(frozen=True)
class PhotoIndex:
    """Eligible photos per H3 cell, and the first `cap` of each cell in sample order."""

    selected: pa.Table  # photo_id, observation_key, position, h3_cell
    counts: pa.Table  # h3_cell, photos (before the cap)


def open_source(source: str) -> BinaryIO:
    """Open the dump from an https URL or a local path.

    An https download resumes where it stopped (HTTP Range) when the connection breaks: the
    37 GB stream takes about an hour, and a reset would lose the whole pass.
    """
    if source.startswith("https://"):
        return cast(BinaryIO, io.BufferedReader(_ResumableHttp(source), 1 << 20))
    return Path(source).open("rb")


_RETRY_WAIT = 5.0  # seconds between two tries
_RETRIES = 20  # tries in a row with no byte read, before the error goes up


class _ResumableHttp(io.RawIOBase):
    """A read-only HTTP body that reopens at its position after a connection error."""

    def __init__(self, url: str) -> None:
        self.url, self.position, self.response = url, 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        for attempt in range(_RETRIES):
            try:
                if self.response is None:
                    self.response = self._open()
                n = self.response.readinto(buffer)
            except (OSError, http.client.HTTPException):
                self._drop(attempt)
                continue
            self.position += n
            return n
        raise ConnectionResetError(104, f"{self.url}: {_RETRIES} tries failed")

    def _open(self) -> Any:
        headers = {"Range": f"bytes={self.position}-"}
        request = urllib.request.Request(self.url, headers=headers)  # noqa: S310 - https only
        return urllib.request.urlopen(request, timeout=120)  # noqa: S310 - https only

    def _drop(self, attempt: int) -> None:
        if self.response is not None:
            with contextlib.suppress(Exception):
                self.response.close()
        self.response = None
        time.sleep(_RETRY_WAIT * min(attempt + 1, 6))

    def close(self) -> None:
        if self.response is not None:
            self.response.close()
        super().close()


def read_dump(stream: BinaryIO, selection: Selection) -> DumpSlice:
    """Read the selected observations and photos, and the taxa and observers they refer to.

    Photos are also kept only when their licence is allowed.
    """
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        tables = _read_members(tar, selection)
    missing = sorted(set(_SCHEMAS) - set(tables))
    if missing:
        raise DumpError(f"dump has no {', '.join(missing)}")
    return DumpSlice(
        observations=tables["observations.csv"],
        photos=tables["photos.csv"],
        taxa=tables["taxa.csv"],
        observers=tables["observers.csv"],
    )


def _read_members(tar: tarfile.TarFile, selection: Selection) -> dict[str, pa.Table]:
    tables: dict[str, pa.Table] = {}
    for name, handle in _members(tar, set(_SCHEMAS)):
        predicate = _predicate(name, selection, tables.get("observations.csv"))
        tables[name] = _read_filtered(handle, name, predicate)
    return tables


def _predicate(name: str, selection: Selection, observations: pa.Table | None) -> Predicate:
    if name == "observations.csv":
        return lambda b: _key_in(b["observation_uuid"], selection.observation_keys)
    if observations is None:
        raise DumpError(f"{name} comes before observations.csv in the dump")
    if name == "photos.csv":
        return lambda b: pc.and_(pc.is_in(b["photo_id"], selection.photo_ids), _allowed(b))
    key = "taxon_id" if name == "taxa.csv" else "observer_id"
    ids = pc.unique(observations[key].combine_chunks()).drop_null()
    return lambda b: pc.is_in(b[key], ids)


def scan_taxa(stream: BinaryIO) -> pa.Table:
    """Read only the taxa table of the dump."""
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for _name, handle in _members(tar, {"taxa.csv"}):
            return _read_filtered(handle, "taxa.csv", lambda b: pc.is_valid(b["taxon_id"]))
    raise DumpError("dump has no taxa.csv")


def index_photos(
    stream: BinaryIO,
    eligible_taxa: NDArray[np.int64],
    prefix: str,
    *,
    res: int,
    cap: int,
    compact_rows: int = 20_000_000,
) -> PhotoIndex:
    """Count eligible photos per H3 cell and keep the first `cap` of each cell.

    An observation is eligible when it is in the sample of `prefix`, has a location, and has
    no taxon or a taxon in `eligible_taxa`. A photo is eligible when its observation is and its
    licence is allowed. Only the observation and photo tables are read.
    """
    cells: _ObservationCells | None = None
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for name, handle in _members(tar, {"observations.csv", "photos.csv"}):
            if name == "photos.csv":
                return _photo_index(handle, _require(cells), cap, compact_rows)
            cells = _observation_cells(handle, eligible_taxa, prefix, res)
    raise DumpError("dump has no photos.csv")


def _members(tar: tarfile.TarFile, names: set[str]) -> Iterator[tuple[str, IO[bytes]]]:
    """Yield the name and content of each regular file of the stream named in `names`."""
    for member in tar:
        name = Path(member.name).name
        handle = tar.extractfile(member) if name in names else None
        if handle is not None:
            yield name, handle


def _require(cells: "_ObservationCells | None") -> "_ObservationCells":
    if cells is None:
        raise DumpError("photos.csv comes before observations.csv in the dump")
    return cells


def uuid_keys(uuids: pa.Array | pa.ChunkedArray) -> NDArray[np.uint64]:
    """Return the 64-bit sort key of each UUID: its first 16 hex digits."""
    digits = pc.utf8_slice_codeunits(pc.replace_substring(uuids, "-", ""), 0, 16)
    if isinstance(digits, pa.ChunkedArray):
        digits = digits.combine_chunks()
    offsets = np.frombuffer(digits.buffers()[1], dtype=np.int32)[digits.offset :]
    raw = np.frombuffer(digits.buffers()[2], dtype=np.uint8)[offsets[0] : offsets[len(digits)]]
    nibbles = _HEX[raw.reshape(-1, 16)]
    return np.bitwise_or.reduce(nibbles << _SHIFTS, axis=1).astype(np.uint64)


@dataclass(frozen=True)
class _ObservationCells:
    keys: NDArray[np.uint64]  # sorted
    cells: NDArray[np.int64]


def _observation_cells(
    handle: IO[bytes], eligible_taxa: NDArray[np.int64], prefix: str, res: int
) -> _ObservationCells:
    keys, cells = [], []
    for batch in _batches(handle, "observations.csv"):
        rows = batch.filter(_eligible_observation(batch, eligible_taxa, prefix))
        keys.append(uuid_keys(rows["observation_uuid"]))
        cells.append(h3_cells(rows["latitude"].to_numpy(), rows["longitude"].to_numpy(), res))
    all_keys = np.concatenate(keys) if keys else np.zeros(0, np.uint64)
    order = np.argsort(all_keys, kind="stable")
    all_cells = np.concatenate(cells) if cells else np.zeros(0, np.int64)
    return _ObservationCells(all_keys[order], all_cells[order])


def _eligible_observation(
    batch: pa.RecordBatch, eligible_taxa: NDArray[np.int64], prefix: str
) -> pa.Array:
    uuid = batch["observation_uuid"]
    in_sample = pc.less(pc.utf8_lower(pc.utf8_slice_codeunits(uuid, 0, 8)), prefix)
    located = pc.and_(pc.is_valid(batch["latitude"]), pc.is_valid(batch["longitude"]))
    taxon = pc.or_(
        pc.is_null(batch["taxon_id"]), pc.is_in(batch["taxon_id"], pa.array(eligible_taxa))
    )
    valid = pc.equal(pc.utf8_length(uuid), _UUID_LENGTH)
    return pc.fill_null(pc.and_(pc.and_(in_sample, located), pc.and_(taxon, valid)), False)


def _photo_index(
    handle: IO[bytes], obs: _ObservationCells, cap: int, compact_rows: int
) -> PhotoIndex:
    parts: list[pa.Table] = []
    counts = pa.table({"h3_cell": pa.array([], pa.int64()), "photos": pa.array([], pa.int64())})
    buffered = 0
    for batch in _batches(handle, "photos.csv"):
        part = _matched_photos(batch, obs)
        counts = _add_counts(counts, part["h3_cell"])
        parts.append(part)
        buffered += part.num_rows
        if buffered >= compact_rows:
            parts, buffered = [_compact(pa.concat_tables(parts), cap)], 0
    selected = _compact(pa.concat_tables(parts), cap) if parts else _empty_selection()
    return PhotoIndex(selected, counts.sort_by("h3_cell"))


def _matched_photos(batch: pa.RecordBatch, obs: _ObservationCells) -> pa.Table:
    rows = batch.filter(
        pc.fill_null(pc.and_(_allowed(batch), _valid_uuid(batch["observation_uuid"])), False)
    )
    keys = uuid_keys(rows["observation_uuid"])
    at = np.minimum(np.searchsorted(obs.keys, keys), max(len(obs.keys) - 1, 0))
    found = obs.keys[at] == keys if len(obs.keys) else np.zeros(len(keys), dtype=bool)
    return pa.table(
        {
            "photo_id": rows["photo_id"].to_numpy(zero_copy_only=False)[found],
            "observation_key": keys[found],
            "position": pc.fill_null(rows["position"], 0).to_numpy(zero_copy_only=False)[found],
            "h3_cell": obs.cells[at[found]],
        }
    )


def recap(index: PhotoIndex, cap: int) -> PhotoIndex:
    """Return the index with at most `cap` photos per cell; the counts do not change."""
    return PhotoIndex(_compact(index.selected, cap), index.counts)


def _compact(table: pa.Table, cap: int) -> pa.Table:
    columns = [table[c].to_numpy() for c in ("h3_cell", "observation_key", "position", "photo_id")]
    return table.take(pa.array(cap_first(*columns, cap)))


def _add_counts(counts: pa.Table, cells: pa.ChunkedArray | pa.Array) -> pa.Table:
    new = pa.table({"h3_cell": cells}).group_by("h3_cell").aggregate([("h3_cell", "count")])
    new = new.rename_columns(["h3_cell", "photos"])
    merged = pa.concat_tables([counts, new.select(["h3_cell", "photos"])])
    return (
        merged.group_by("h3_cell")
        .aggregate([("photos", "sum")])
        .rename_columns(["h3_cell", "photos"])
    )


def _empty_selection() -> pa.Table:
    return pa.table(
        {
            "photo_id": pa.array([], pa.int64()),
            "observation_key": pa.array([], pa.uint64()),
            "position": pa.array([], pa.int64()),
            "h3_cell": pa.array([], pa.int64()),
        }
    )


def _key_in(uuids: pa.Array, sorted_keys: NDArray[np.uint64]) -> pa.Array:
    valid = pc.fill_null(_valid_uuid(uuids), False).to_numpy(zero_copy_only=False)
    keys = uuid_keys(uuids.filter(pa.array(valid)))
    hit = np.zeros(len(uuids), dtype=bool)
    hit[np.flatnonzero(valid)] = np.isin(keys, sorted_keys, assume_unique=False)
    return pa.array(hit)


def _valid_uuid(uuids: pa.Array) -> pa.Array:
    return pc.equal(pc.utf8_length(uuids), _UUID_LENGTH)


def _allowed(batch: pa.RecordBatch) -> pa.Array:
    return pc.is_in(pc.utf8_upper(batch["license"]), pa.array(ALLOWED_LICENSES))


def _batches(handle: IO[bytes], name: str) -> csv.CSVStreamingReader:
    schema = _SCHEMAS[name]
    return csv.open_csv(
        handle,
        read_options=csv.ReadOptions(block_size=_BLOCK_SIZE),
        parse_options=csv.ParseOptions(delimiter="\t", invalid_row_handler=lambda _row: "skip"),
        convert_options=csv.ConvertOptions(
            column_types=schema, include_columns=list(schema), strings_can_be_null=True
        ),
    )


def _read_filtered(handle: IO[bytes], name: str, predicate: Predicate) -> pa.Table:
    batches = [batch.filter(predicate(batch)) for batch in _batches(handle, name)]
    return pa.Table.from_batches(batches, schema=pa.schema(_SCHEMAS[name]))
