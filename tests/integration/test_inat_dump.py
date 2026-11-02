"""Streaming reader for the iNaturalist metadata dump."""

import io

import pytest

from inat_landscape.adapters.inat_dump import DumpError, open_source, read_dump
from inat_landscape.domain.sampling import sample_prefix
from tests import fakes


def _read(members=None):
    return read_dump(io.BytesIO(fakes.make_dump(members)), sample_prefix(0.25))


def test_keeps_only_sampled_observations():
    dump = _read()
    assert sorted(dump.observations["observation_uuid"].to_pylist()) == sorted(
        [fakes.OBS_PLANT, fakes.OBS_NO_ID, fakes.OBS_BIRD, fakes.OBS_LIFE]
    )


def test_keeps_photos_of_sampled_observations_with_allowed_licence():
    assert sorted(_read().photos["photo_id"].to_pylist()) == [101, 102, 103, 105, 107]


def test_keeps_only_referenced_taxa_and_observers():
    dump = _read()
    assert sorted(dump.taxa["taxon_id"].to_pylist()) == [3, 48460, 52775]
    assert sorted(dump.observers["observer_id"].to_pylist()) == [10, 11, 12]


def test_parses_quoted_names_and_empty_values():
    dump = _read()
    names = dict(
        zip(dump.observers["login"].to_pylist(), dump.observers["name"].to_pylist(), strict=True)
    )
    assert names["cyd"] == 'Cyd "C" D'
    assert names["bob"] in (None, "")


def test_ignores_unknown_members():
    members = dict(fakes.DEFAULT_MEMBERS) | {"README.txt": ["hello"]}
    assert _read(members).photos.num_rows == 5


def test_photos_before_observations_is_an_error():
    members = {
        "photos.csv": fakes.DEFAULT_MEMBERS["photos.csv"],
        "observations.csv": fakes.DEFAULT_MEMBERS["observations.csv"],
    }
    with pytest.raises(DumpError, match=r"observations\.csv"):
        _read(members)


def test_missing_member_is_an_error():
    members = dict(fakes.DEFAULT_MEMBERS)
    del members["taxa.csv"]
    with pytest.raises(DumpError, match=r"taxa\.csv"):
        _read(members)


def test_skips_malformed_rows():
    members = dict(fakes.DEFAULT_MEMBERS)
    members["photos.csv"] = [*members["photos.csv"], "broken\trow"]
    assert _read(members).photos.num_rows == 5


def test_open_source_reads_local_path(tmp_path):
    path = tmp_path / "dump.tar.gz"
    path.write_bytes(fakes.make_dump())
    with open_source(str(path)) as stream:
        assert read_dump(stream, sample_prefix(0.25)).photos.num_rows == 5


def test_open_source_uses_urlopen_for_https(monkeypatch):
    seen = []
    monkeypatch.setattr("urllib.request.urlopen", lambda url: seen.append(url) or io.BytesIO())
    open_source("https://example.org/dump.tar.gz")
    assert seen == ["https://example.org/dump.tar.gz"]
