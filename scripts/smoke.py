"""Smoke test: run the installed CLI on the fixture dump, offline, without the real model.

It checks the commands that need no network: ingest, sample and status.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.fakes import make_dump


def run(*args: str) -> str:
    out = subprocess.run(["inat-landscape", *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


with tempfile.TemporaryDirectory() as tmp:
    dump = Path(tmp) / "dump.tar.gz"
    dump.write_bytes(make_dump())
    work = str(Path(tmp) / "work")
    assert (
        run(
            "--workdir",
            work,
            "ingest",
            "--source",
            str(dump),
            "--fraction",
            "0.25",
            "--cap",
            "1000",
        )
        == "candidates=4"
    )
    assert run("--workdir", work, "sample", "--n", "3") == "sample=3"
    status = json.loads(run("--workdir", work, "status"))
    assert status["candidates"] == 4, status
    assert status["sample"] == 3, status
print("smoke ok")
