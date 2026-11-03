"""Pipeline state: hand the work files to another machine through a hub repository.

The state goes under `state/` in a private repository, apart from the public dataset.
Embeddings are not part of it: they already live in the public dataset.
"""

from collections.abc import Sequence
from pathlib import Path

from inat_landscape.adapters.hub import Hub
from inat_landscape.pipeline.errors import StageError

STATE_FILES = (
    "ingest/taxa.parquet",  # the index of the dump (two passes): a retry does not read it again
    "ingest/taxa.json",
    "ingest/index.json",
    "ingest/selected.parquet",
    "ingest/cell_counts.parquet",
    "ingest/manifest.json",
    "ingest/candidates.parquet",
    "sample.parquet",
    "sample-ids.parquet",
    "scored.parquet",
    "failures.parquet",
)
_PREFIX = "state/"


def push_state(workdir: Path, hub: Hub) -> list[str]:
    """Upload the state files that exist in the work directory. Return their names."""
    present = [name for name in STATE_FILES if (workdir / name).exists()]
    if not present:
        raise StageError(f"nothing to push in {workdir}")
    for name in present:
        hub.upload_file(workdir / name, _PREFIX + name, f"Update {name}")
    return present


def pull_state(workdir: Path, hub: Hub, *, only: Sequence[str] = STATE_FILES) -> list[str]:
    """Download the state files of the hub (or the files of `only`) into the work directory.

    Return their names.
    """
    names = [n for n in _remote_state(hub) if n in only]
    if not names:
        raise StageError("no state on the hub: run `state push` first")
    for name in names:
        _pull_one(workdir, hub, name)
    return names


def _pull_one(workdir: Path, hub: Hub, name: str) -> None:
    staging = workdir / ".state"
    hub.download(_PREFIX + name, staging)
    target = workdir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    (staging / _PREFIX / name).replace(target)


def _remote_state(hub: Hub) -> list[str]:
    names = [path.removeprefix(_PREFIX) for path in hub.list_files(_PREFIX)]
    return [name for name in names if name in STATE_FILES]
