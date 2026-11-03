"""CPU supervisor script: regression tests for its shell logic."""

import os
import re
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "cpu_supervisor.sh"


def _token_block() -> str:
    text = SCRIPT.read_text()
    match = re.search(r"# The token file wins.*?\nexport HF_TOKEN\n", text, re.S)
    assert match, "token block not found"
    return match.group(0)


def _run(block: str, env: dict[str, str]) -> str:
    bash = shutil.which("bash") or "/bin/bash"
    command = [bash, "-c", block + 'printf %s "$HF_TOKEN"']
    return subprocess.run(command, env=env, capture_output=True, text=True, check=True).stdout  # noqa: S603 - fixed command


def test_token_file_wins_over_an_old_environment_token(tmp_path):
    # Bug: the supervisor used an old invalid HF_TOKEN from the launching shell (HTTP 401).
    (tmp_path / ".hf_token").write_text("good")
    env = {**os.environ, "HOME": str(tmp_path), "HF_TOKEN": "old"}
    assert _run(_token_block(), env) == "good"


def test_environment_token_is_kept_without_a_token_file(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "HF_TOKEN": "env"}
    assert _run(_token_block(), env) == "env"
