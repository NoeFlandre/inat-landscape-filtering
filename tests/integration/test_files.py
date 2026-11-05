"""Local file helpers: atomic writes."""

import pyarrow as pa
import pytest

from inat_landscape.adapters.files import read_json, read_table, write_json, write_table


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
