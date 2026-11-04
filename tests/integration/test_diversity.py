"""Diversity stage: one pass over the shards gives a value and a colour per cell."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from inat_landscape.adapters.hub import OfflineHub
from inat_landscape.domain.diversity import mean_pairwise_distance, unit
from inat_landscape.pipeline.diversity import DIVERSITY, read_diversity, run_diversity
from inat_landscape.pipeline.embed import open_store, read_sample
from inat_landscape.pipeline.errors import StageError
from tests.fakes import FakeEncoder, FakeHub
from tests.integration.pipeline_helpers import prepared

VECTORS = {101: [1.0, 0.0, 0.0], 102: [0.0, 1.0, 0.0], 103: [0.0, 0.0, 2.0], 107: [3.0, 0.0, 0.0]}


@pytest.fixture
def embedded(tmp_path):
    prepared(tmp_path)
    (tmp_path / "remote").mkdir()
    hub = FakeHub(tmp_path / "remote")
    store = open_store(tmp_path, FakeEncoder.model_id, hub)
    store.add([101, 103], np.array([VECTORS[101], VECTORS[103]]))
    store.add([102, 107], np.array([VECTORS[102], VECTORS[107]]))
    return tmp_path, hub


def test_diversity_of_each_cell_with_enough_photos(embedded):
    workdir, hub = embedded
    table = run_diversity(workdir, FakeEncoder.model_id, hub, min_photos=2, workers=2)
    sample = read_sample(workdir, ["photo_id", "h3_cell"]).to_pydict()
    cell = dict(zip(sample["photo_id"], sample["h3_cell"], strict=True))[101]
    assert table["h3_cell"].to_pylist() == [cell]  # 101 and 102 share a cell; 103, 107: alone
    assert table["photos"].to_pylist() == [2]
    pair = unit(np.array([VECTORS[101], VECTORS[102]])).sum(axis=0, dtype=np.float64)
    expected = mean_pairwise_distance(pair[None], np.array([2]))
    assert np.isclose(table["diversity"][0].as_py(), expected[0])
    assert table.column_names == ["h3_cell", "photos", "diversity", "pc1", "pc2", "pc3"]
    assert (workdir / DIVERSITY).exists()


def test_each_cell_counts_only_its_first_photos(embedded):
    workdir, hub = embedded
    with pytest.raises(StageError, match="2 embedded photos"):  # one photo has no pair
        run_diversity(workdir, FakeEncoder.model_id, hub, per_cell=1, min_photos=1)
    run_diversity(workdir, FakeEncoder.model_id, hub, per_cell=2, min_photos=1)
    found = read_diversity(workdir)
    assert found is not None
    assert found[1] == {"per_cell": 2, "min_photos": 2}
    assert pq.read_metadata(workdir / DIVERSITY).num_rows == 1


def test_diversity_needs_cells_with_enough_photos(embedded):
    workdir, hub = embedded
    with pytest.raises(StageError, match="embed"):
        run_diversity(workdir, FakeEncoder.model_id, hub, min_photos=3)


def test_diversity_needs_embeddings(tmp_path):
    prepared(tmp_path)
    with pytest.raises(StageError, match="embed"):
        run_diversity(tmp_path, FakeEncoder.model_id, OfflineHub(), min_photos=1)


def test_no_diversity_before_the_stage(tmp_path):
    assert read_diversity(tmp_path) is None
