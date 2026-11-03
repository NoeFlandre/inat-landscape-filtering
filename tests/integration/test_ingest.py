"""Ingest stage: cached passes (taxa, index, details) -> candidate photos."""

import io
import json

import numpy as np
import pyarrow as pa
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.licensing import attribution
from inat_landscape.domain.taxa import classify
from inat_landscape.pipeline.ingest import (
    CANDIDATES,
    COUNTS,
    MANIFEST,
    IngestParams,
    attributions,
    eligible_taxa,
    groups,
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


def test_candidates_carry_their_group(tmp_path):
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


def test_an_index_of_another_taxa_scope_is_not_reused(tmp_path):
    calls = []
    _run(tmp_path, calls)
    manifest = tmp_path / "ingest" / "index.json"
    stored = json.loads(manifest.read_text())
    assert stored["taxa"] == "all"
    del stored["taxa"]  # an index of the plants-only scope had no taxa key
    manifest.write_text(json.dumps(stored))
    (tmp_path / "ingest" / "manifest.json").unlink()
    before = len(calls)
    _run(tmp_path, calls)
    assert len(calls) > before + 1  # the index pass runs again, then the details pass


names = st.one_of(st.none(), st.just(""), st.text(max_size=5))


@given(
    st.lists(
        st.tuples(st.sampled_from(["CC0", "CC-BY", "CC-BY-NC"]), names, names, st.integers(1, 99)),
        max_size=30,
    )
)
def test_vectorized_attributions_equal_the_domain_rule(rows):
    table = pa.table(
        {
            "license": pa.array([r[0] for r in rows], pa.string()),
            "observer_name": pa.array([r[1] for r in rows], pa.string()),
            "login": pa.array([r[2] for r in rows], pa.string()),
            "observer_id": pa.array([r[3] for r in rows], pa.int64()),
        }
    )
    expected = [attribution(c, n, lg or f"iNaturalist user {i}") for c, n, lg, i in rows]
    assert attributions(table).to_pylist() == expected


@given(
    st.lists(
        st.tuples(
            st.one_of(st.none(), st.sampled_from([3, 47126, 48460, 52775, 999, 47170])),
        ),
        max_size=30,
    )
)
def test_vectorized_groups_equal_the_domain_rule(rows):
    taxa = pa.table(
        {
            "taxon_id": [3, 47126, 48460, 52775, 47170],
            "ancestry": ["48460/1/2", "48460", None, "48460/47126/211194", "48460"],
        }
    )
    ancestry = dict(zip(taxa["taxon_id"].to_pylist(), taxa["ancestry"].to_pylist(), strict=True))
    ids = [r[0] for r in rows]
    expected = [classify(t, ancestry.get(t)).value for t in ids]
    assert groups(pa.array(ids, pa.int64()), taxa).to_pylist() == expected
