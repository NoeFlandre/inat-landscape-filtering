"""Local file helpers: atomic writes."""

import pyarrow as pa
import pytest

from inat_landscape.adapters.files import (
    read_json,
    read_table,
    write_batches,
    write_json,
    write_table,
)


def test_table_round_trip_creates_parent_dirs(tmp_path):
    path = tmp_path / "a" / "b.parquet"
    write_table(pa.table({"x": [1, 2]}), path)
    assert read_table(path)["x"].to_pylist() == [1, 2]
    assert [p.name for p in path.parent.iterdir()] == ["b.parquet"]


def test_json_round_trip(tmp_path):
    write_json({"k": 1}, tmp_path / "m.json")
    assert read_json(tmp_path / "m.json") == {"k": 1}


def test_read_json_missing_returns_none(tmp_path):
    assert read_json(tmp_path / "none.json") is None


def test_failed_write_leaves_no_partial_file(tmp_path):
    path = tmp_path / "t.parquet"
    with pytest.raises(AttributeError):
        write_table("not a table", path)  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


def test_write_batches_streams_tables_into_one_file(tmp_path):
    schema = pa.schema([("x", pa.int64())])
    parts = (pa.table({"x": [i, i + 1]}, schema=schema) for i in (0, 2, 4))
    write_batches(parts, schema, tmp_path / "s.parquet")
    assert read_table(tmp_path / "s.parquet")["x"].to_pylist() == [0, 1, 2, 3, 4, 5]


def test_write_batches_with_no_batch_writes_an_empty_file(tmp_path):
    schema = pa.schema([("x", pa.int64())])
    write_batches(iter(()), schema, tmp_path / "e.parquet")
    assert read_table(tmp_path / "e.parquet").num_rows == 0
