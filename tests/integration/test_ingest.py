"""Ingest stage: dump -> candidate photos table."""

import io
import json

from inat_landscape.pipeline.ingest import CANDIDATES, MANIFEST, run_ingest
from tests import fakes


def _opener(calls):
    def open_(source):
        calls.append(source)
        return io.BytesIO(fakes.make_dump())

    return open_


def _run(tmp_path, calls, **kw):
    return run_ingest(tmp_path, "fake://dump", 0.25, opener=_opener(calls), **kw)


def test_candidates_keep_plants_and_unidentified_only(tmp_path):
    table = _run(tmp_path, [])
    assert table["photo_id"].to_pylist() == [101, 102, 103, 107]
    assert table["group"].to_pylist() == ["plantae", "plantae", "unidentified", "unidentified"]


def test_candidates_carry_location_taxon_and_attribution(tmp_path):
    row = _run(tmp_path, []).to_pylist()[0]
    assert row["latitude"] == 45.5
    assert row["longitude"] == 3.25
    assert row["positional_accuracy"] == 20
    assert row["taxon_name"] == "Bellis perennis"
    assert row["observed_on"] == "2021-05-01"
    assert row["attribution"] == "© Ana B, some rights reserved (CC-BY)"
    assert row["license"] == "CC-BY"


def test_licence_is_normalized_to_upper_case(tmp_path):
    assert _run(tmp_path, [])["license"].to_pylist()[-1] == "CC-BY"


def test_writes_candidates_and_manifest(tmp_path):
    _run(tmp_path, [])
    assert (tmp_path / CANDIDATES).exists()
    manifest = json.loads((tmp_path / MANIFEST).read_text())
    assert manifest == {"source": "fake://dump", "fraction": 0.25, "candidates": 4}


def test_second_run_with_same_parameters_reuses_output(tmp_path):
    calls = []
    _run(tmp_path, calls)
    again = _run(tmp_path, calls)
    assert calls == ["fake://dump"]
    assert again.num_rows == 4


def test_force_or_new_fraction_reads_the_dump_again(tmp_path):
    calls = []
    _run(tmp_path, calls)
    _run(tmp_path, calls, force=True)
    run_ingest(tmp_path, "fake://dump", 0.5, opener=_opener(calls))
    assert len(calls) == 3
