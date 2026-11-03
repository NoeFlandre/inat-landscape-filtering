"""Streaming reader for the iNaturalist metadata dump."""

import io

import pytest

from inat_landscape.adapters.inat_dump import DumpError, Selection, open_source, read_dump
from inat_landscape.domain.sampling import uuid_key
from tests import fakes

SAMPLED = [fakes.OBS_PLANT, fakes.OBS_NO_ID, fakes.OBS_BIRD, fakes.OBS_LIFE]
ALL = Selection.of([uuid_key(u) for u in SAMPLED], [101, 102, 103, 104, 105, 107])


def _read(members=None):
    return read_dump(io.BytesIO(fakes.make_dump(members)), ALL)


def test_keeps_only_selected_observations():
    dump = _read()
    assert sorted(dump.observations["observation_uuid"].to_pylist()) == sorted(
        [fakes.OBS_PLANT, fakes.OBS_NO_ID, fakes.OBS_BIRD, fakes.OBS_LIFE]
    )


def test_keeps_selected_photos_with_allowed_licence():
    assert sorted(_read().photos["photo_id"].to_pylist()) == [101, 102, 103, 107]  # 105: CC-BY-SA


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
    assert _read(members).photos.num_rows == 4


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
    assert _read(members).photos.num_rows == 4


def test_open_source_reads_local_path(tmp_path):
    path = tmp_path / "dump.tar.gz"
    path.write_bytes(fakes.make_dump())
    with open_source(str(path)) as stream:
        assert read_dump(stream, ALL).photos.num_rows == 4


class _Flaky:
    """urlopen stand-in: each response breaks after `cut` bytes, until the last part."""

    def __init__(self, data: bytes, cut: int) -> None:
        self.data, self.cut, self.ranges = data, cut, []

    def __call__(self, request, timeout=None):
        start = int((request.get_header("Range") or "bytes=0-")[6:-1])
        self.ranges.append(start)
        part = self.data[start : start + self.cut]
        broken = start + self.cut < len(self.data)
        return _Response(part, broken)


class _Response(io.BytesIO):
    def __init__(self, data: bytes, broken: bool) -> None:
        super().__init__(data)
        self.broken = broken

    def readinto(self, buffer):
        n = super().readinto(buffer)
        if n == 0 and self.broken:
            raise ConnectionResetError(104, "Connection reset by peer")
        return n


def test_open_source_resumes_an_https_download_after_a_reset(monkeypatch):
    data = fakes.make_dump()
    flaky = _Flaky(data, cut=97)
    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr("inat_landscape.adapters.inat_dump._RETRY_WAIT", 0.0)
    with open_source("https://example.org/dump.tar.gz") as stream:
        assert read_dump(stream, ALL).photos.num_rows == 4
    assert flaky.ranges[:3] == [0, 97, 194]


def test_open_source_gives_up_after_many_resets_without_progress(monkeypatch):
    def broken(request, timeout=None):
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr("urllib.request.urlopen", broken)
    monkeypatch.setattr("inat_landscape.adapters.inat_dump._RETRY_WAIT", 0.0)
    with pytest.raises(ConnectionResetError), open_source("https://example.org/d") as stream:
        stream.read(10)
