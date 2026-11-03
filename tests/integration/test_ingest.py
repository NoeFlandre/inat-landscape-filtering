"""Ingest stage: cached passes (taxa, index, details) -> candidate photos."""

import io
import json

import numpy as np
import pyarrow as pa

from inat_landscape.pipeline.ingest import (
    CANDIDATES,
    COUNTS,
    MANIFEST,
    IngestParams,
    eligible_taxa,
    run_index,
    run_ingest,
)
from tests import fakes

PARAMS = IngestParams(source="fake://dump", fraction=0.25, res=5, cap=1000)


def _opener(calls):
    def open_(source):
        calls.append(source)
        return io.BytesIO(fakes.make_dump())

    return open_


def _run(tmp_path, calls, params=PARAMS, **kw):
    return run_ingest(tmp_path, params, opener=_opener(calls), **kw)


def test_candidates_keep_plants_and_unidentified_only(tmp_path):
    table = _run(tmp_path, [])
    assert table["photo_id"].to_pylist() == [101, 102, 103, 107]
    assert table["group"].to_pylist() == ["plantae", "plantae", "unidentified", "unidentified"]


def test_candidates_carry_location_cell_taxon_and_attribution(tmp_path):
    row = _run(tmp_path, []).to_pylist()[0]
    assert (row["latitude"], row["longitude"], row["positional_accuracy"]) == (45.5, 3.25, 20)
    assert row["h3_cell"] == 599534425792839679
    assert row["taxon_name"] == "Bellis perennis"
    assert row["observed_on"] == "2021-05-01"
    assert row["attribution"] == "© Ana B, some rights reserved (CC-BY)"


def test_licence_is_normalized_to_upper_case(tmp_path):
    assert _run(tmp_path, [])["license"].to_pylist()[-1] == "CC-BY"


def test_cap_applies_before_details(tmp_path):
    params = IngestParams(source="fake://dump", fraction=0.25, res=5, cap=1)
    assert _run(tmp_path, [], params)["photo_id"].to_pylist() == [101, 103, 107]


def test_index_writes_counts_and_is_cached(tmp_path):
    calls = []
    index = run_index(tmp_path, PARAMS, opener=_opener(calls))
    assert sorted(index.counts["photos"].to_pylist()) == [1, 1, 2]
    assert (tmp_path / COUNTS).exists()
    run_index(tmp_path, PARAMS, opener=_opener(calls))
    assert len(calls) == 2  # taxa + index, then nothing


def test_manifest_and_full_reuse(tmp_path):
    calls = []
    _run(tmp_path, calls)
    assert len(calls) == 3  # taxa, index, details
    assert json.loads((tmp_path / MANIFEST).read_text())["candidates"] == 4
    assert (tmp_path / CANDIDATES).exists()
    assert _run(tmp_path, calls).num_rows == 4
    assert len(calls) == 3


def test_new_fraction_reuses_taxa_but_reruns_index_and_details(tmp_path):
    calls = []
    _run(tmp_path, calls)
    _run(tmp_path, calls, IngestParams(source="fake://dump", fraction=0.5, res=5, cap=1000))
    assert len(calls) == 5


def test_force_reruns_index_and_details(tmp_path):
    calls = []
    _run(tmp_path, calls)
    _run(tmp_path, calls, force=True)
    assert len(calls) == 5


def test_every_taxon_is_eligible():
    taxa = pa.table(
        {
            "taxon_id": [3, 47126, 48460, 52775],
            "ancestry": ["48460/1/2", "48460", None, "48460/47126/211194"],
            "name": ["Aves", "Plantae", "Life", "x"],
        }
    )
    assert eligible_taxa(taxa).tolist() == [3, 47126, 48460, 52775]
    assert eligible_taxa(taxa).dtype == np.int64


def test_smaller_cap_is_derived_from_a_cached_larger_cap(tmp_path):
    calls = []
    run_index(tmp_path, PARAMS, opener=_opener(calls))
    small = IngestParams(source="fake://dump", fraction=0.25, res=5, cap=1)
    derived = run_index(tmp_path, small, opener=_opener(calls))
    assert len(calls) == 2  # no new pass over the dump
    direct = run_index(tmp_path / "direct", small, opener=_opener([]))
    assert derived.selected.sort_by("photo_id").equals(direct.selected.sort_by("photo_id"))
    assert derived.counts.equals(direct.counts)


def test_larger_cap_or_other_resolution_reads_the_dump_again(tmp_path):
    calls = []
    run_index(
        tmp_path,
        IngestParams(source="fake://dump", fraction=0.25, res=5, cap=1),
        opener=_opener(calls),
    )
    run_index(tmp_path, PARAMS, opener=_opener(calls))
    run_index(
        tmp_path,
        IngestParams(source="fake://dump", fraction=0.25, res=6, cap=1000),
        opener=_opener(calls),
    )
    assert len(calls) == 4  # taxa once, then one index pass each
