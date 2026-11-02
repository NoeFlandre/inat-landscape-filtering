"""Embedding store: shards are the record of done work, mirrored to the hub."""

import numpy as np
import pytest

from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.hub import OfflineHub
from tests.fakes import FakeHub

DIR = "embeddings/model/small"


@pytest.fixture
def hub(tmp_path):
    (tmp_path / "remote").mkdir()
    return FakeHub(tmp_path / "remote")


def test_add_writes_float16_shard_and_uploads(tmp_path, hub):
    store = EmbeddingStore(tmp_path / "a", DIR, hub)
    path = store.add([3, 1], np.array([[1, 2], [3, 4]], np.float32))
    assert path.parent == store.local_dir
    assert hub.uploads == [f"{DIR}/{path.name}"]
    ids, emb = store.load()
    assert ids == [3, 1]
    assert emb.dtype == np.float32
    assert emb.tolist() == [[1, 2], [3, 4]]


def test_shard_name_is_deterministic(tmp_path):
    a = EmbeddingStore(tmp_path / "a", DIR, OfflineHub()).add([1, 2], np.ones((2, 2)))
    b = EmbeddingStore(tmp_path / "b", DIR, OfflineHub()).add([1, 2], np.zeros((2, 2)))
    assert a.name == b.name


def test_new_machine_pulls_remote_shards_and_sees_done_ids(tmp_path, hub):
    EmbeddingStore(tmp_path / "first", DIR, hub).add([1, 2], np.ones((2, 3)))
    second = EmbeddingStore(tmp_path / "second", DIR, hub)
    assert second.done_ids() == set()
    assert second.pull() == 1
    assert second.done_ids() == {1, 2}
    assert second.pull() == 0


def test_load_keeps_first_copy_of_duplicate_ids(tmp_path):
    store = EmbeddingStore(tmp_path, DIR, OfflineHub())
    store.add([1, 2], np.array([[1.0], [2.0]]))
    store.add([2, 3], np.array([[9.0], [3.0]]))
    ids, emb = store.load()
    assert sorted(ids) == [1, 2, 3]
    assert len(emb) == 3


def test_load_empty_store(tmp_path):
    ids, emb = EmbeddingStore(tmp_path, DIR, OfflineHub()).load()
    assert ids == []
    assert emb.shape == (0, 0)


def test_add_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="length"):
        EmbeddingStore(tmp_path, DIR, OfflineHub()).add([1], np.ones((2, 2)))


def test_push_uploads_only_local_shards_missing_on_remote(tmp_path, hub):
    store = EmbeddingStore(tmp_path / "a", DIR, OfflineHub())
    store.add([1], np.ones((1, 2)))
    store.add([2], np.ones((1, 2)))
    online = EmbeddingStore(tmp_path / "a", DIR, hub)
    assert online.push() == 2
    assert online.push() == 0
    assert len(hub.uploads) == 2


def test_dim_is_the_embedding_length(tmp_path):
    store = EmbeddingStore(tmp_path, DIR, OfflineHub())
    assert store.dim() == 0
    store.add([1], np.ones((1, 5)))
    assert store.dim() == 5
