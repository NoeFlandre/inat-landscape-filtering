"""Grid'5000 GPU finder: which GPUs are free, and one worker slot for each."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASH = shutil.which("bash") or "/bin/bash"
SCRIPT = Path(__file__).parents[2] / "scripts" / "g5k" / "free_gpus.py"
_spec = importlib.util.spec_from_file_location("free_gpus", SCRIPT)
assert _spec
assert _spec.loader
free_gpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(free_gpus)


def _core(host, gpu, *, cluster="chuc", model="A100-SXM4-40GB", jobs=None, **kw):
    return {
        "host": f"{host}.lille.grid5000.fr",
        "cluster": cluster,
        "gpu": gpu,
        "gpu_model": model,
        "state": kw.get("state", "Alive"),
        "jobs": jobs,
        "exotic": kw.get("exotic", "NO"),
        "cpuarch": kw.get("cpuarch", "x86_64"),
        "production": kw.get("production", "NO"),
    }


def test_a_gpu_is_free_when_no_core_of_it_has_a_job():
    cores = [
        _core("chuc1", 1),
        _core("chuc1", 1),
        _core("chuc1", 2),
        _core("chuc1", 2, jobs="123"),
        _core("chuc2", 3, state="Absent"),
        _core("chuc2", 4),
    ]
    [found] = free_gpus.clusters(cores)
    assert (found["cluster"], found["free"]) == ("chuc", 2)
    assert found["options"] == ["-p", "cluster='chuc'"]


def test_cores_without_a_gpu_are_ignored():
    assert free_gpus.clusters([_core("chuc1", None)]) == []


def test_production_and_exotic_clusters_get_their_queue_and_type():
    cores = [
        _core("grue1", 1, cluster="grue", model="Tesla T4", production="YES"),
        _core("sirius1", 1, cluster="sirius", exotic="YES"),
    ]
    found = {c["cluster"]: c["options"] for c in free_gpus.clusters(cores)}
    assert found["grue"] == ["-q", "production", "-p", "cluster='grue'"]
    assert found["sirius"] == ["-t", "exotic", "-p", "cluster='sirius'"]


def test_gpus_that_the_cuda_build_cannot_use_are_left_out():
    for model in [
        "Radeon Instinct MI50 32GB",
        "AMD Instinct MI300X",
        "Instinct MI210",
        "AGX Xavier",
        "GH200 480GB",
        "RTX PRO 6000 Blackwell Server Edition",
    ]:
        assert not free_gpus.usable(model), model
    usable = ["A100-SXM4-40GB", "Tesla T4", "TITAN RTX", "L40S", "H100 NVL", "Tesla V100"]
    for model in [*usable, "GeForce GTX 1080 Ti", "Tesla P100-PCIE-16GB", "TITAN X (Pascal)"]:
        assert free_gpus.usable(model), model


def test_plan_gives_each_free_gpu_one_free_slot_of_the_global_count():
    found = {
        "lille": [{"cluster": "chuc", "free": 2, "options": ["-p", "cluster='chuc'"]}],
        "nancy": [{"cluster": "grue", "free": 1, "options": ["-q", "production"]}],
    }
    assert free_gpus.plan(found, count=8, slots=[1, 4, 6]) == [
        {"slot": "1/8", "site": "lille", "options": ["-p", "cluster='chuc'"]},
        {"slot": "4/8", "site": "lille", "options": ["-p", "cluster='chuc'"]},
        {"slot": "6/8", "site": "nancy", "options": ["-q", "production"]},
    ]


def test_plan_stops_when_the_free_slots_or_the_free_gpus_end():
    found = {"lille": [{"cluster": "chuc", "free": 3, "options": []}]}
    assert [s["slot"] for s in free_gpus.plan(found, count=4, slots=[2])] == ["2/4"]
    assert len(free_gpus.plan(found, count=9, slots=list(range(9)))) == 3


def test_plan_with_no_free_gpu_is_empty():
    assert free_gpus.plan({"lille": []}, count=4, slots=[0, 1]) == []


def test_plan_command_reads_the_sites_and_prints_one_slot_per_line(tmp_path):
    site = tmp_path / "lille.jsonl"
    site.write_text(json.dumps({"cluster": "chuc", "free": 2, "options": ["-p", "x"]}) + "\n")
    command = [sys.executable, str(SCRIPT), "plan", "4", "1,3", str(site)]
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert [json.loads(line)["slot"] for line in out.splitlines()] == ["1/4", "3/4"]


def test_plan_command_with_no_free_slot_prints_nothing(tmp_path):
    site = tmp_path / "lille.jsonl"
    site.write_text(json.dumps({"cluster": "chuc", "free": 2, "options": []}) + "\n")
    command = [sys.executable, str(SCRIPT), "plan", "4", "", str(site)]
    assert subprocess.run(command, capture_output=True, text=True, check=True).stdout == ""  # noqa: S603


def test_the_job_scripts_are_valid_bash():
    for script in [*sorted(SCRIPT.parent.glob("*.sh")), SCRIPT.parent / "embed.oar"]:
        subprocess.run([BASH, "-n", str(script)], check=True)  # noqa: S603 - fixed command


def test_a_slot_is_required_to_submit():
    run = subprocess.run(  # noqa: S603 - fixed command
        [BASH, str(SCRIPT.parent / "submit.sh")], capture_output=True, text=True, check=False
    )
    assert run.returncode != 0
    assert "usage" in run.stderr


def test_gpus_of_non_x86_nodes_are_left_out():
    cores = [_core("drac1", 1, cluster="drac", model="Tesla P100-SXM2-16GB", cpuarch="ppc64le")]
    assert free_gpus.clusters(cores) == []


def test_seats_lists_one_line_per_free_gpu(tmp_path):
    site = tmp_path / "lille.jsonl"
    site.write_text(json.dumps({"cluster": "chuc", "free": 2, "options": ["-p", "x"]}) + "\n")
    command = [sys.executable, str(SCRIPT), "seats", str(site)]
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603
    assert [json.loads(line) for line in out.splitlines()] == [
        {"site": "lille", "options": ["-p", "x"]},
        {"site": "lille", "options": ["-p", "x"]},
    ]
