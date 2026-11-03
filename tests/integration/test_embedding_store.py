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
    [(ids, emb)] = list(store.stream())
    assert ids == [3, 1]
    assert emb.dtype == np.float16
    assert emb.tolist() == [[1, 2], [3, 4]]


def test_shard_name_is_deterministic(tmp_path):
    a = EmbeddingStore(tmp_path / "a", DIR, OfflineHub()).add([1, 2], np.ones((2, 2)))
    b = EmbeddingStore(tmp_path / "b", DIR, OfflineHub()).add([1, 2], np.zeros((2, 2)))
    assert a.name == b.name


def test_new_machine_sees_the_remote_ids_and_has_no_local_shard(tmp_path, hub):
    EmbeddingStore(tmp_path / "first", DIR, hub).add([1, 2], np.ones((2, 3)))
    second = EmbeddingStore(tmp_path / "second", DIR, hub)
    assert second.done_ids() == set()
    assert second.remote_ids() == {1, 2}


def test_stream_reads_local_shards_in_place(tmp_path):
    store = EmbeddingStore(tmp_path, DIR, OfflineHub())
    store.add([1, 2], np.array([[1.0], [2.0]]))
    assert [ids for ids, _ in store.stream()] == [[1, 2]]
    assert store.done_ids() == {1, 2}  # the local shard is kept


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


def test_a_tagged_store_names_its_shards_with_the_tag(tmp_path):
    store = EmbeddingStore(tmp_path, DIR, OfflineHub(), tag="w1of4")
    assert store.add([1], np.ones((1, 2))).name.startswith("part-w1of4-")


def test_remote_ids_read_the_untagged_shards_and_the_shards_of_the_tag_only(tmp_path, hub):
    EmbeddingStore(tmp_path / "old", DIR, hub).add([1, 2], np.ones((2, 2)))
    EmbeddingStore(tmp_path / "a", DIR, hub, tag="w0of2").add([4], np.ones((1, 2)))
    EmbeddingStore(tmp_path / "b", DIR, hub, tag="w1of2").add([3], np.ones((1, 2)))
    fresh = EmbeddingStore(tmp_path / "new", DIR, hub, tag="w0of2")
    assert fresh.remote_ids() == {1, 2, 4}
    assert not fresh.local_dir.exists()  # only the photo_id column is read, nothing is stored
    assert EmbeddingStore(tmp_path / "all", DIR, hub).remote_ids() == {1, 2, 3, 4}


def test_stream_yields_each_remote_shard_once_and_keeps_no_copy(tmp_path, hub):
    EmbeddingStore(tmp_path / "a", DIR, hub, tag="w0of2").add([1, 2], np.ones((2, 3)))
    EmbeddingStore(tmp_path / "b", DIR, hub).add([3], np.zeros((1, 3)))
    store = EmbeddingStore(tmp_path / "c", DIR, hub)
    parts = sorted((ids, emb.shape, emb.dtype) for ids, emb in store.stream())
    assert parts == [([1, 2], (2, 3), np.float16), ([3], (1, 3), np.float16)]
    assert not list((tmp_path / "c").rglob("*.parquet"))


def test_stream_of_an_empty_store_yields_nothing(tmp_path, hub):
    assert list(EmbeddingStore(tmp_path, DIR, hub).stream()) == []


def test_a_slot_reads_the_shards_of_another_worker_count(tmp_path, hub):
    EmbeddingStore(tmp_path / "a", DIR, hub, tag="w1of2").add([5], np.ones((1, 2)))
    EmbeddingStore(tmp_path / "b", DIR, hub, tag="w2of4").add([6], np.ones((1, 2)))
    EmbeddingStore(tmp_path / "c", DIR, hub, tag="w1of4").add([7], np.ones((1, 2)))
    # A new count N can start at any time: the work of the old count is seen, the work of
    # the other slots of the same count cannot overlap and is not read.
    assert EmbeddingStore(tmp_path / "d", DIR, hub, tag="w1of4").remote_ids() == {5, 7}


def test_a_slot_range_reads_the_shards_of_the_ranges_it_overlaps(tmp_path, hub):
    EmbeddingStore(tmp_path / "a", DIR, hub, tag="w0-2of9").add([1], np.ones((1, 2)))
    EmbeddingStore(tmp_path / "b", DIR, hub, tag="w2-4of9").add([2], np.ones((1, 2)))
    EmbeddingStore(tmp_path / "c", DIR, hub, tag="w3of9").add([3], np.ones((1, 2)))
    assert EmbeddingStore(tmp_path / "d", DIR, hub, tag="w1-3of9").remote_ids() == {1, 2}


def test_a_snapshot_of_the_done_ids_replaces_the_reads_of_the_shards_it_covers(tmp_path, hub):
    EmbeddingStore(tmp_path / "a", DIR, hub).add([1, 2], np.ones((2, 2)))
    EmbeddingStore(tmp_path / "b", DIR, hub, tag="w0of2").add([4], np.ones((1, 2)))
    assert EmbeddingStore(tmp_path / "s", DIR, hub).snapshot() == 3
    EmbeddingStore(tmp_path / "c", DIR, hub, tag="w1of2").add([5], np.ones((1, 2)))
    hub.id_reads = 0
    fresh = EmbeddingStore(tmp_path / "d", DIR, hub, tag="w1of2")
    assert fresh.remote_ids() == {1, 5}  # the photos of slot 1 of 2 (odd ids)
    assert hub.id_reads == 1  # only the shard made after the snapshot


def test_the_snapshot_is_not_an_embedding_shard(tmp_path, hub):
    store = EmbeddingStore(tmp_path / "a", DIR, hub)
    store.add([1], np.ones((1, 2)))
    store.snapshot()
    assert [ids for ids, _ in EmbeddingStore(tmp_path / "b", DIR, hub).stream()] == [[1]]


def test_a_second_snapshot_reads_only_the_new_shards(tmp_path, hub):
    EmbeddingStore(tmp_path / "a", DIR, hub).add([1], np.ones((1, 2)))
    EmbeddingStore(tmp_path / "s", DIR, hub).snapshot()
    EmbeddingStore(tmp_path / "b", DIR, hub).add([2], np.ones((1, 2)))
    hub.id_reads = 0
    assert EmbeddingStore(tmp_path / "t", DIR, hub).snapshot() == 2
    assert hub.id_reads == 1
