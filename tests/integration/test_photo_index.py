"""Streaming photo index: taxa scan, per-cell counts, capped selection, uuid keys."""

import io

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.adapters.geo import h3_cells
from inat_landscape.adapters.inat_dump import (
    DumpError,
    Selection,
    _id_in,
    index_photos,
    read_dump,
    scan_taxa,
    uuid_keys,
)
from inat_landscape.domain.sampling import sample_prefix, uuid_key
from tests import fakes

PLANT_TAXA = np.array([47126, 48460, 52775], dtype=np.int64)


def _stream(members=None):
    return io.BytesIO(fakes.make_dump(members))


@given(st.lists(st.uuids(version=4).map(str), max_size=20))
def test_uuid_keys_match_the_domain_key(uuids):
    keys = uuid_keys(pa.array(uuids, pa.string()))
    assert keys.dtype == np.uint64
    assert keys.tolist() == [uuid_key(u) for u in uuids]


def test_uuid_keys_accept_upper_case_and_chunked_arrays():
    u = "0A000000-0000-4000-8000-000000000001"
    chunked = pa.chunked_array([[u], [u.lower()]])
    assert uuid_keys(chunked).tolist() == [uuid_key(u)] * 2


def test_scan_taxa_returns_the_taxa_table():
    taxa = scan_taxa(_stream())
    assert sorted(taxa["taxon_id"].to_pylist()) == [3, 47126, 48460, 52775]


def test_scan_taxa_without_taxa_is_an_error():
    members = dict(fakes.DEFAULT_MEMBERS)
    del members["taxa.csv"]
    with pytest.raises(DumpError, match=r"taxa\.csv"):
        scan_taxa(_stream(members))


def test_index_counts_eligible_photos_per_cell_and_selects_first_per_cell():
    index = index_photos(_stream(), PLANT_TAXA, sample_prefix(0.25), res=5, cap=1)
    cells = dict(
        zip(index.counts["h3_cell"].to_pylist(), index.counts["photos"].to_pylist(), strict=True)
    )
    plant_cell, noid_cell, life_cell = h3_cells(
        np.array([45.5, -12.0, 10.0]), np.array([3.25, 130.0, 20.0]), 5
    ).tolist()
    # bird (3) excluded, unsampled f... excluded, CC-BY-SA excluded
    assert cells == {plant_cell: 2, noid_cell: 1, life_cell: 1}
    selected = index.selected.to_pylist()
    assert sorted(r["photo_id"] for r in selected) == [101, 103, 107]  # cap 1: position 0 first
    assert {r["h3_cell"] for r in selected} == set(cells)


def test_index_skips_observations_without_location():
    members = dict(fakes.DEFAULT_MEMBERS)
    members["observations.csv"] = [
        line.replace("\t45.5\t3.25\t", "\t\t\t") for line in members["observations.csv"]
    ]
    index = index_photos(_stream(members), PLANT_TAXA, sample_prefix(0.25), res=5, cap=10)
    assert 101 not in index.selected["photo_id"].to_pylist()


def test_index_compacts_without_changing_the_result():
    small = index_photos(_stream(), PLANT_TAXA, sample_prefix(0.25), res=5, cap=1, compact_rows=1)
    large = index_photos(_stream(), PLANT_TAXA, sample_prefix(0.25), res=5, cap=1)
    assert small.selected.sort_by("photo_id").equals(large.selected.sort_by("photo_id"))
    assert small.counts.sort_by("h3_cell").equals(large.counts.sort_by("h3_cell"))


def test_index_needs_observations_before_photos():
    members = {
        "photos.csv": fakes.DEFAULT_MEMBERS["photos.csv"],
        "observations.csv": fakes.DEFAULT_MEMBERS["observations.csv"],
    }
    with pytest.raises(DumpError, match=r"observations\.csv"):
        index_photos(_stream(members), PLANT_TAXA, sample_prefix(1.0), res=5, cap=1)


def test_index_without_photos_is_an_error():
    members = {"observations.csv": fakes.DEFAULT_MEMBERS["observations.csv"]}
    with pytest.raises(DumpError, match=r"photos\.csv"):
        index_photos(_stream(members), PLANT_TAXA, sample_prefix(1.0), res=5, cap=1)


def test_read_dump_keeps_only_the_selection():
    selection = Selection.of([uuid_key(fakes.OBS_PLANT)], [102])
    dump = read_dump(_stream(), selection)
    assert dump.observations["observation_uuid"].to_pylist() == [fakes.OBS_PLANT]
    assert dump.photos["photo_id"].to_pylist() == [102]
    assert dump.taxa["taxon_id"].to_pylist() == [52775]
    assert dump.observers["observer_id"].to_pylist() == [10]


def test_index_with_an_empty_photos_table():
    members = dict(fakes.DEFAULT_MEMBERS)
    members["photos.csv"] = [fakes.PHOTO_HEADER]
    index = index_photos(_stream(members), PLANT_TAXA, sample_prefix(1.0), res=5, cap=1)
    assert index.selected.num_rows == 0
    assert index.counts.num_rows == 0


def test_index_reads_a_plain_tar_like_the_gzipped_one():
    gz = index_photos(_stream(), PLANT_TAXA, sample_prefix(0.25), res=5, cap=1)
    plain = index_photos(
        io.BytesIO(fakes.make_dump(gzip=False)), PLANT_TAXA, sample_prefix(0.25), res=5, cap=1
    )
    assert plain.selected.equals(gz.selected)
    assert plain.counts.equals(gz.counts)


def test_the_index_keeps_the_extension_of_each_photo():
    index = index_photos(_stream(), PLANT_TAXA, sample_prefix(0.25), res=5, cap=10)
    selected = index.selected
    rows = sorted(
        zip(selected["photo_id"].to_pylist(), selected["extension"].to_pylist(), strict=True)
    )
    assert rows == [(101, "jpg"), (102, "jpeg"), (103, "jpg"), (107, "png")]


@given(
    st.lists(st.one_of(st.none(), st.integers(-5, 30)), max_size=40),
    st.lists(st.integers(-5, 30), max_size=20),
)
def test_id_lookup_matches_is_in(ids, selected):
    column = pa.array(ids, pa.int64())
    expected = pc.fill_null(pc.is_in(column, pa.array(selected, pa.int64())), False)
    got = _id_in(column, np.unique(np.array(selected, dtype=np.int64)))
    assert pc.fill_null(got, False).to_pylist() == expected.to_pylist()
