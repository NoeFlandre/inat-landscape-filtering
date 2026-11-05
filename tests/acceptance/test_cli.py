"""Acceptance: the full pilot flow through the CLI, with fake network and model."""

import io
import json
import logging
import subprocess
import sys
from dataclasses import replace

import pytest

from inat_landscape.adapters.encoder import SiglipEncoder
from inat_landscape.adapters.hub import REPO_ID, HfHub, OfflineHub
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
        encoder=lambda *_args: FakeEncoder(),
        hub=lambda *_args: hub,
        fetcher=lambda urls, **kw: fetch(urls, backoff=0.0, **kw),
        land=lambda: [_square(lon, lat) for lon, lat in ((3.25, 45.5), (130, -12), (20, 10))],
    )


def _square(lon, lat):
    corners = [(lon - 0.1, lat - 0.1), (lon + 0.1, lat - 0.1), (lon + 0.1, lat + 0.1)]
    return {"type": "Polygon", "coordinates": [[*corners, (lon - 0.1, lat - 0.1)]]}


def _run(deps, tmp_path, *args):
    return main(["--workdir", str(tmp_path / "w"), *args], deps)


def test_full_pilot_flow(deps, tmp_path, capsys):
    assert _run(deps, tmp_path, "measure", "--fraction", "0.25", "--caps", "1,10") == 0
    assert _run(deps, tmp_path, "worklist", "--fraction", "0.25") == 0
    assert _run(deps, tmp_path, "ingest", "--fraction", "0.25") == 0
    assert _run(deps, tmp_path, "sample") == 0
    assert (
        _run(
            deps,
            tmp_path,
            "embed",
            "--worker",
            "1/2",
            "--concurrency",
            "4",
            "--fetch-processes",
            "2",
            "--no-compile",
        )
        == 0
    )
    assert _run(deps, tmp_path, "embed", "--worker", "0/2", "--plan") == 0
    assert _run(deps, tmp_path, "embed", "--snapshot") == 0
    assert _run(deps, tmp_path, "embed", "--worker", "0/2", "--max-photos", "1000") == 0
    assert _run(deps, tmp_path, "land") == 0
    assert _run(deps, tmp_path, "embed", "--prune", "--workers", "2") == 0
    assert _run(deps, tmp_path, "score", "--thresholds", "0.1") == 0
    diversity = ["diversity", "--workers", "2", "--min-photos", "2"]
    diversity += ["--model", FakeEncoder.model_id]
    assert _run(deps, tmp_path, *diversity) == 0
    assert _run(deps, tmp_path, "publish", "--thresholds", "0", "--maps-only") == 0
    cache = str(tmp_path / "cache")
    assert _run(deps, tmp_path, "score", "--cache", cache, "--workers", "2") == 0
    assert list((tmp_path / "cache").rglob("*.parquet"))
    assert _run(deps, tmp_path, "publish", "--thresholds", "0,0.99", "--cap-per-cell", "1") == 0
    assert _run(deps, tmp_path, "publish", "--thresholds", "0", "--score", "score_contrast") == 0
    assert _run(deps, tmp_path, "publish", "--thresholds", "0", "--no-images") == 0
    assert _run(deps, tmp_path, "state", "push") == 0
    assert main(["--workdir", str(tmp_path / "w2"), "state", "pull"], deps) == 0
    assert (tmp_path / "w2" / "scored.parquet").exists()
    assert _run(deps, tmp_path, "status", "--model", FakeEncoder.model_id) == 0
    out = capsys.readouterr().out
    assert "cells=3 photos=4 max_per_cell=2" in out
    assert "worklist=4" in out
    assert "snapshot=" in out
    assert "sea_cells=0 sea_photos=0" in out
    assert "pruned_shards=0 pruned_rows=0" in out
    assert "diversity_cells=" in out
    assert "maps=assets/diversity.png,assets/look.png,assets/look-detail.png" in out
    assert "cap 1: 3 photos, 3 cells at the cap" in out
    assert "candidates=4" in out
    assert "sample=4" in out
    assert "images=no" in out
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
    for command in (
        "measure",
        "ingest",
        "worklist",
        "sample",
        "embed",
        "score",
        "publish",
        "state",
        "status",
    ):
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
    encoder = real.encoder("some/model", 8, True)
    assert isinstance(encoder, SiglipEncoder)
    assert (encoder.model_id, encoder.batch_size, encoder.compile) == ("some/model", 8, True)


@pytest.mark.parametrize("worker", ["2/2", "x", "1"])
def test_bad_worker_exits_2(deps, tmp_path, worker):
    with pytest.raises(SystemExit) as exc:
        _run(deps, tmp_path, "embed", "--worker", worker)
    assert exc.value.code == 2


def test_bad_caps_exit_2(deps, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(deps, tmp_path, "measure", "--caps", "a,b")
    assert exc.value.code == 2


def test_dump_file_reads_a_local_copy_and_keeps_the_source(deps, tmp_path, capsys):
    local = tmp_path / "dump.tar"
    local.write_bytes(fakes.make_dump(gzip=False))
    args = ["--source", "https://example.org/dump.tar.gz", "--dump-file", str(local)]
    assert _run(deps, tmp_path, "worklist", "--fraction", "0.25", *args) == 0
    assert "worklist=4" in capsys.readouterr().out
    index = json.loads((tmp_path / "w" / "ingest" / "index.json").read_text())
    assert index["source"] == "https://example.org/dump.tar.gz"


def test_http_clients_log_no_line_per_request(deps, tmp_path):
    # Bug: httpx logged each photo request at INFO; on the NFS /home of Grid'5000 the log
    # writes stalled the download loop at about 60 photos per second.
    _run(deps, tmp_path, "status", "--model", FakeEncoder.model_id)
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name


def test_state_goes_to_the_dataset_repo(deps, tmp_path):
    repos = []
    hub = deps.hub()
    deps = replace(deps, hub=lambda repo_id, *_args: (repos.append(repo_id), hub)[1])
    assert _run(deps, tmp_path, "ingest", "--fraction", "0.25") == 0
    assert _run(deps, tmp_path, "state", "push") == 0
    assert repos[-1] == REPO_ID
    assert "state/ingest/manifest.json" in hub.list_files("state/")
