"""Acceptance: the full pilot flow through the CLI, with fake network and model."""

import io
import json
import subprocess
import sys

import pytest

from inat_landscape.adapters.encoder import SiglipEncoder
from inat_landscape.adapters.hub import HfHub, OfflineHub
from inat_landscape.adapters.images import fetch
from inat_landscape.cli import Deps, main
from tests import fakes
from tests.fakes import FakeEncoder, FakeHub
from tests.integration.pipeline_helpers import serve_photos


@pytest.fixture
def deps(tmp_path, server, monkeypatch):
    _, handler, base = server
    monkeypatch.setattr(
        "inat_landscape.pipeline.embed.PHOTO_URL", base + "/photos/{photo_id}/{size}.{extension}"
    )
    serve_photos(handler)
    (tmp_path / "remote").mkdir()
    hub = FakeHub(tmp_path / "remote")
    return Deps(
        opener=lambda _s: io.BytesIO(fakes.make_dump()),
        encoder=lambda _model, _batch: FakeEncoder(),
        hub=lambda *_args: hub,
        fetcher=lambda urls, **kw: fetch(urls, backoff=0.0, **kw),
    )


def _run(deps, tmp_path, *args):
    return main(["--workdir", str(tmp_path / "w"), *args], deps)


def test_full_pilot_flow(deps, tmp_path, capsys):
    assert _run(deps, tmp_path, "measure", "--fraction", "0.25", "--caps", "1,10") == 0
    assert _run(deps, tmp_path, "ingest", "--fraction", "0.25") == 0
    assert _run(deps, tmp_path, "sample") == 0
    assert _run(deps, tmp_path, "embed", "--worker", "1/2", "--concurrency", "4") == 0
    assert _run(deps, tmp_path, "embed", "--worker", "0/2", "--plan") == 0
    assert _run(deps, tmp_path, "embed", "--worker", "0/2") == 0
    assert _run(deps, tmp_path, "score", "--thresholds", "0.1") == 0
    assert _run(deps, tmp_path, "publish", "--thresholds", "0,0.99", "--cap-per-cell", "1") == 0
    assert _run(deps, tmp_path, "publish", "--thresholds", "0", "--score", "score_contrast") == 0
    assert _run(deps, tmp_path, "state", "push") == 0
    assert main(["--workdir", str(tmp_path / "w2"), "state", "pull"], deps) == 0
    assert (tmp_path / "w2" / "scored.parquet").exists()
    assert _run(deps, tmp_path, "status", "--model", FakeEncoder.model_id) == 0
    out = capsys.readouterr().out
    assert "cells=3 photos=4 max_per_cell=2" in out
    assert "cap 1: 3 photos, 3 cells at the cap" in out
    assert "candidates=4" in out
    assert "sample=4" in out
    assert "score_contrast: " in out
    assert "score_ge_0_00=3" in out
    assert "contrast_ge_0_00=" in out
    assert "embedded=3" in out
    assert "embedded=1" in out
    assert "todo=1" in out
    assert "n=4" in out
    status = json.loads(out.strip().splitlines()[-1])
    assert status == {"candidates": 4, "sample": 4, "embeddings": 4, "failures": 0, "scored": 4}


def test_stage_out_of_order_exits_1_with_message(deps, tmp_path, capsys):
    assert _run(deps, tmp_path, "sample", "--n", "3") == 1
    assert "run `ingest` first" in capsys.readouterr().err


def test_unknown_score_exits_2(deps, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(deps, tmp_path, "publish", "--thresholds", "0.1", "--score", "score_closeup")
    assert exc.value.code == 2


def test_bad_thresholds_exit_2(deps, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(deps, tmp_path, "publish", "--thresholds", "2")
    assert exc.value.code == 2


def test_help_lists_commands():
    result = subprocess.run(
        [sys.executable, "-m", "inat_landscape.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for command in ("measure", "ingest", "sample", "embed", "score", "publish", "state", "status"):
        assert command in result.stdout


def test_no_command_exits_2():
    result = subprocess.run(
        [sys.executable, "-m", "inat_landscape.cli"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2


def test_default_deps_build_real_adapters_lazily():
    real = Deps()
    assert isinstance(real.hub("u/r", True), OfflineHub)
    assert isinstance(real.hub("u/r", False), HfHub)
    private = real.hub("u/r", False, True)
    assert isinstance(private, HfHub)
    assert private.private
    encoder = real.encoder("some/model", 8)
    assert isinstance(encoder, SiglipEncoder)
    assert (encoder.model_id, encoder.batch_size) == ("some/model", 8)


@pytest.mark.parametrize("worker", ["2/2", "x", "1"])
def test_bad_worker_exits_2(deps, tmp_path, worker):
    with pytest.raises(SystemExit) as exc:
        _run(deps, tmp_path, "embed", "--worker", worker)
    assert exc.value.code == 2


def test_bad_caps_exit_2(deps, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(deps, tmp_path, "measure", "--caps", "a,b")
    assert exc.value.code == 2
