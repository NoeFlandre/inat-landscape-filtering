#!/usr/bin/env python3
"""Find the free GPUs of a Grid'5000 site, and give each free GPU one worker slot.

Standard library only: it runs on the frontends, which do not have the project environment.

    python3 free_gpus.py                    # on a frontend: free GPUs per cluster (JSON lines)
    python3 free_gpus.py plan N SLOTS SITE.jsonl…   # give free slots of N (SLOTS: 1,4,…) to GPUs
    python3 free_gpus.py seats SITE.jsonl…          # one line per free GPU: site, options, weight
    python3 free_gpus.py assign N FREE SITE.jsonl…  # runs of free slots (FREE: 1,4,…) per GPU
"""

import json
import subprocess
import sys
from pathlib import Path

# The cu126 PyTorch build runs on NVIDIA x86 GPUs from Maxwell (sm_50) to Hopper (sm_90).
# These words mark AMD GPUs, ARM boards and Blackwell GPUs (sm_120).
_UNUSABLE = ("Radeon", "Instinct", "AGX", "GH200", "Blackwell")


# Speed of each GPU model in units of about 120 photos/s (embed, measured on 2026-11-03).
# A job owns as many worker slots as its weight, so all slots advance at about one speed.
_WEIGHTS = (
    ("L40S", 7), ("A40", 7), ("A100", 7), ("H100", 7), ("H200", 7),
    ("RTX 6000 Ada", 5), ("RTX 8000", 5), ("A6000", 5), ("L4", 4),
    ("RTX 6000", 3), ("V100", 3),
    ("A5000", 2), ("2080 Ti", 2), ("T4", 2), ("TITAN RTX", 2),
    ("GTX", 1), ("P100", 1), ("TITAN X", 1),
)  # fmt: skip


def weight(model: str) -> int:
    """Return the speed weight of a GPU model (2 for a model not in the table)."""
    return next((w for name, w in _WEIGHTS if name in model), 2)


def usable(model: str) -> bool:
    """Tell if the cu126 PyTorch build can use this GPU model."""
    return not any(word in model for word in _UNUSABLE)


def clusters(cores: list[dict]) -> list[dict]:
    """Return the free usable GPUs of each cluster, with the oarsub options to get them.

    `cores` are OAR resources (one per core). A GPU is free when no core of it has a job.
    """
    found: dict[str, dict] = {}
    for (cluster, _, _), (core, free) in sorted(_gpus(cores).items()):
        if usable(core.get("gpu_model") or "") and core.get("cpuarch", "x86_64") == "x86_64":
            entry = found.setdefault(
                cluster,
                {
                    "cluster": cluster,
                    "free": 0,
                    "options": _options(core),
                    "weight": weight(core.get("gpu_model") or ""),
                },
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


def seats(found: dict[str, list[dict]]) -> list[tuple[str, list[str], int]]:
    """Return (site, oarsub options, weight) once for each free GPU, fastest first."""
    every = [
        (site, entry["options"], entry.get("weight", 1))
        for site, entries in found.items()
        for entry in entries
        for _ in range(entry["free"])
    ]
    return sorted(every, key=lambda seat: -seat[2])


def assign(seats: list[tuple[str, list[str], int]], *, count: int, free: list[int]) -> list[dict]:
    """Give each GPU a run of consecutive free slots of `count`, as long as its weight."""
    runs, left = [], sorted(free)
    for site, options, size in seats:
        if not left:
            break
        run = [left.pop(0)]
        while left and len(run) < size and left[0] == run[-1] + 1:
            run.append(left.pop(0))
        span = f"{run[0]}" if len(run) == 1 else f"{run[0]}:{run[-1] + 1}"
        runs.append({"site": site, "slots": f"{span}/{count}", "options": options})
    return runs


def plan(found: dict[str, list[dict]], *, count: int, slots: list[int]) -> list[dict]:
    """Give the free `slots` of `count` worker slots to the free GPUs, one slot each."""
    return [
        {"slot": f"{slot}/{count}", "site": site, "options": options}
        for slot, (site, options, _) in zip(slots, seats(found), strict=False)
    ]


def free_slots(count: int, busy: list[str]) -> list[int]:
    """Return the slots of `count` that no busy name holds (3-of-9, or 3-6-of-9: slots 3 to 5)."""
    held = set()
    for name in busy:
        if name.strip():
            span = name.strip().lstrip("-").rsplit("-of-", 1)[0].split("-")
            held.update(range(int(span[0]), int(span[-1]) + (len(span) == 1)))
    return [i for i in range(count) if i not in held]


def _oar_cores() -> list[dict]:
    command = ["oarnodes", "-J", "--sql", "gpu_count > 0"]
    data = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    return list(data.values()) if isinstance(data, dict) else data


def _read(paths: list[str]) -> dict[str, list[dict]]:
    return {
        Path(p).stem: [json.loads(line) for line in Path(p).read_text().splitlines() if line]
        for p in paths
    }


def _slots(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s]


def _free(argv: list[str]) -> list[str]:
    free = free_slots(int(argv[0]), Path(argv[1]).read_text().splitlines())
    return [",".join(str(i) for i in free)]


def _seats(argv: list[str]) -> list[dict]:
    return [{"site": s, "options": o, "weight": w} for s, o, w in seats(_read(argv))]


def _assign(argv: list[str]) -> list[dict]:
    return assign(seats(_read(argv[2:])), count=int(argv[0]), free=_slots(argv[1]))


def _plan(argv: list[str]) -> list[dict]:
    return plan(_read(argv[2:]), count=int(argv[0]), slots=_slots(argv[1]))


_COMMANDS = {"free": _free, "seats": _seats, "assign": _assign, "plan": _plan}


def main(argv: list[str]) -> None:
    command = _COMMANDS.get(argv[0]) if argv else None
    lines = command(argv[1:]) if command else clusters(_oar_cores())
    for line in lines:
        print(line if isinstance(line, str) else json.dumps(line))


if __name__ == "__main__":
    main(sys.argv[1:])
