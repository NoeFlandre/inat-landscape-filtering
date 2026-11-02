"""Build a work directory with a sample and an image server for stage tests."""

import io

from inat_landscape.pipeline.ingest import IngestParams, run_ingest
from inat_landscape.pipeline.sample import run_sample
from tests import fakes
from tests.conftest import jpeg

COLORS = {101: (10, 200, 10), 102: (200, 10, 10), 103: (10, 10, 200), 107: (90, 90, 90)}


def prepared(workdir, n=4):
    params = IngestParams(source="fake", fraction=0.25, res=5, cap=1000)
    run_ingest(workdir, params, opener=lambda _s: io.BytesIO(fakes.make_dump()))
    return run_sample(workdir, n)


def serve_photos(handler, sizes=("small", "medium"), skip=()):
    ext = {101: "jpg", 102: "jpeg", 103: "jpg", 107: "png"}
    for pid, color in COLORS.items():
        if pid in skip:
            continue
        for size in sizes:
            handler.routes[f"/photos/{pid}/{size}.{ext[pid]}"] = jpeg(color)
