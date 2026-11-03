#!/usr/bin/env python3
"""Find the free GPUs of a Grid'5000 site, and give each free GPU one worker slot.

Standard library only: it runs on the frontends, which do not have the project environment.

    python3 free_gpus.py                    # on a frontend: free GPUs per cluster (JSON lines)
    python3 free_gpus.py plan SITE.jsonl…   # one slot I/N per free GPU, with one global N
"""

import json
import subprocess
import sys
from pathlib import Path

# The cu126 PyTorch build runs on NVIDIA x86 GPUs from Volta (7.0) to Hopper (9.0).
# These words mark AMD GPUs, ARM boards, Pascal or older GPUs, and Blackwell GPUs.
_UNUSABLE = ("Radeon", "Instinct", "AGX", "GH200", "Blackwell", "GTX", "P100", "P6000", "TITAN X")


def usable(model: str) -> bool:
    """Tell if the cu126 PyTorch build can use this GPU model."""
    return not any(word in model for word in _UNUSABLE)


def clusters(cores: list[dict]) -> list[dict]:
    """Return the free usable GPUs of each cluster, with the oarsub options to get them.

    `cores` are OAR resources (one per core). A GPU is free when no core of it has a job.
    """
    found: dict[str, dict] = {}
    for (cluster, _, _), (core, free) in sorted(_gpus(cores).items()):
        if usable(core.get("gpu_model") or ""):
            entry = found.setdefault(
                cluster, {"cluster": cluster, "free": 0, "options": _options(core)}
            )
            entry["free"] += free
    return [entry for entry in found.values() if entry["free"]]


def _gpus(cores: list[dict]) -> dict[tuple, tuple[dict, bool]]:
    """Return (one core, free) for each GPU, keyed by (cluster, host, gpu)."""
    gpus: dict[tuple, tuple[dict, bool]] = {}
    for core in (c for c in cores if c.get("gpu") is not None):
        key = (core["cluster"], core["host"], core["gpu"])
        busy = core.get("state") != "Alive" or bool(core.get("jobs"))
        gpus[key] = (core, gpus.get(key, (core, True))[1] and not busy)
    return gpus


def _options(core: dict) -> list[str]:
    options = ["-q", "production"] if core.get("production") == "YES" else []
    options += ["-t", "exotic"] if core.get("exotic") == "YES" else []
    return [*options, "-p", f"cluster='{core['cluster']}'"]


def plan(found: dict[str, list[dict]]) -> list[dict]:
    """Give each free GPU of each site one slot I/N. N is the count of all free GPUs."""
    seats = [
        (site, entry["options"])
        for site, entries in found.items()
        for entry in entries
        for _ in range(entry["free"])
    ]
    return [
        {"slot": f"{i}/{len(seats)}", "site": site, "options": options}
        for i, (site, options) in enumerate(seats)
    ]


def _oar_cores() -> list[dict]:
    command = ["oarnodes", "-J", "--sql", "gpu_count > 0"]
    data = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    return list(data.values()) if isinstance(data, dict) else data


def main(argv: list[str]) -> None:
    if argv[:1] == ["plan"]:
        found = {
            Path(p).stem: [json.loads(line) for line in Path(p).read_text().splitlines() if line]
            for p in argv[1:]
        }
        lines = plan(found)
    else:
        lines = clusters(_oar_cores())
    for line in lines:
        print(json.dumps(line))


if __name__ == "__main__":
    main(sys.argv[1:])
