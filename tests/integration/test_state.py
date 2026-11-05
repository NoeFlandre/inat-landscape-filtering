"""Pipeline state: move the work files between machines through `state/` of a hub repo."""

import pytest

from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.state import STATE_FILES, pull_state, push_state
from tests.fakes import FakeHub


@pytest.fixture
def hub(tmp_path):
    (tmp_path / "remote").mkdir()
    return FakeHub(tmp_path / "remote")


def test_push_then_pull_moves_existing_state_files(tmp_path, hub):
    src = tmp_path / "a"
    (src / "ingest").mkdir(parents=True)
    (src / "sample.parquet").write_bytes(b"sample")
    (src / "ingest" / "manifest.json").write_text("{}")
    assert push_state(src, hub) == ["ingest/manifest.json", "sample.parquet"]
    assert hub.uploads == ["state/ingest/manifest.json", "state/sample.parquet"]
    dst = tmp_path / "b"
    assert pull_state(dst, hub) == ["ingest/manifest.json", "sample.parquet"]
    assert (dst / "sample.parquet").read_bytes() == b"sample"


def test_push_with_nothing_to_push_is_an_error(tmp_path, hub):
    with pytest.raises(StageError, match="nothing to push"):
        push_state(tmp_path, hub)


def test_pull_with_empty_remote_is_an_error(tmp_path, hub):
    with pytest.raises(StageError, match="no state"):
        pull_state(tmp_path, hub)


def test_state_files_cover_the_hand_off_points():
    assert set(STATE_FILES) == {
        "ingest/taxa.parquet",
        "ingest/taxa.json",
        "ingest/index.json",
        "ingest/selected.parquet",
        "ingest/cell_counts.parquet",
        "ingest/manifest.json",
        "ingest/candidates.parquet",
        "sample.parquet",
        "sample-ids.parquet",
        "scored.parquet",
        "failures.parquet",
        "land.parquet",
        "land.json",
        "land-grid.parquet",
        "diversity.parquet",
    }


def test_pull_only_some_files(tmp_path, hub):
    src = tmp_path / "a"
    src.mkdir()
    (src / "sample.parquet").write_bytes(b"big")
    (src / "sample-ids.parquet").write_bytes(b"small")
    push_state(src, hub)
    dst = tmp_path / "b"
    assert pull_state(dst, hub, only=["sample-ids.parquet", "failures.parquet"]) == [
        "sample-ids.parquet"
    ]
    assert not (dst / "sample.parquet").exists()
