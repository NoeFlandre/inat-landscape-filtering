"""Publish stage: build the dataset configs and the card, and upload them."""

import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from inat_landscape.adapters.files import read_table, write_bytes, write_table
from inat_landscape.adapters.hf_parquet import image_column, with_hf_features
from inat_landscape.adapters.hub import Hub
from inat_landscape.adapters.images import fetch
from inat_landscape.adapters.inat_dump import SNAPSHOT_URL
from inat_landscape.domain.card import CardInfo, Subset, render_card
from inat_landscape.domain.scoring import CLOSEUP_PROMPTS, LANDSCAPE_PROMPTS
from inat_landscape.domain.thresholds import config_name
from inat_landscape.pipeline.embed import HUB_DIR, SCORING_SIZE, embedding_dir, photo_url
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ports import Fetcher, Log
from inat_landscape.pipeline.score import SCORE, SCORED

IMAGE_SIZE = "medium"
CACHE = Path("cache") / IMAGE_SIZE
PART = "part-00000.parquet"


@dataclass(frozen=True)
class PublishReport:
    """What one publish run wrote."""

    total: int
    subsets: tuple[Subset, ...]
    images_failed: int


def run_publish(
    workdir: Path,
    thresholds: Sequence[float],
    hub: Hub,
    *,
    model_id: str,
    fetcher: Fetcher = fetch,
    log: Log = lambda m: print(m, file=sys.stderr),
) -> PublishReport:
    """Write `all` and one config per threshold under the hub folder, then upload them."""
    if not (workdir / SCORED).exists():
        raise StageError(f"no scores at {workdir / SCORED}: run `score` first")
    table = _with_urls(read_table(workdir / SCORED))
    root = workdir / HUB_DIR
    shutil.rmtree(root / "data", ignore_errors=True)
    write_table(with_hf_features(table), root / "data" / "all" / PART)
    kept = table.filter(pc.greater_equal(table[SCORE], min(thresholds)))
    images = _medium_images(workdir, kept, fetcher)
    log(f"medium images: {len(images)} ok, {kept.num_rows - len(images)} failed")
    subsets = tuple(_write_subset(root, table, t, images) for t in thresholds)
    card = render_card(
        CardInfo(
            snapshot=SNAPSHOT_URL,
            model_id=model_id,
            landscape_prompts=LANDSCAPE_PROMPTS,
            closeup_prompts=CLOSEUP_PROMPTS,
            total=table.num_rows,
            subsets=subsets,
            licenses=_count(table["license"]),
            embedding_dir=embedding_dir(model_id),
        )
    )
    (root / "README.md").write_text(card)
    hub.upload_folder(
        root,
        f"Publish {table.num_rows} scored photos",
        allow=["README.md", "data/**"],
        delete=["data/**"],
    )
    return PublishReport(table.num_rows, subsets, kept.num_rows - len(images))


def _with_urls(table: pa.Table) -> pa.Table:
    rows = zip(table["photo_id"].to_pylist(), table["extension"].to_pylist(), strict=True)
    pairs = [(photo_url(i, SCORING_SIZE, e), photo_url(i, IMAGE_SIZE, e)) for i, e in rows]
    table = table.append_column("url_small", pa.array([p[0] for p in pairs], pa.string()))
    return table.append_column("url_medium", pa.array([p[1] for p in pairs], pa.string()))


def _medium_images(workdir: Path, table: pa.Table, fetcher: Fetcher) -> dict[int, bytes]:
    """Return the medium image bytes of each photo, from the cache or downloaded."""
    pairs = zip(table["photo_id"].to_pylist(), table["extension"].to_pylist(), strict=True)
    paths = {i: workdir / CACHE / f"{i}.{e}" for i, e in pairs}
    _download_missing(paths, fetcher)
    return {i: p.read_bytes() for i, p in paths.items() if p.exists()}


def _download_missing(paths: dict[int, Path], fetcher: Fetcher) -> None:
    missing = {
        i: photo_url(i, IMAGE_SIZE, p.suffix[1:]) for i, p in paths.items() if not p.exists()
    }
    for result in fetcher(missing):
        if result.data is not None:
            write_bytes(result.data, paths[result.photo_id])


def _write_subset(
    root: Path, table: pa.Table, threshold: float, images: dict[int, bytes]
) -> Subset:
    ids = pa.array(list(images), pa.int64())
    subset = table.filter(
        pc.and_(pc.greater_equal(table[SCORE], threshold), pc.is_in(table["photo_id"], ids))
    )
    rows = zip(subset["photo_id"].to_pylist(), subset["extension"].to_pylist(), strict=True)
    pairs = [(images[i], f"{i}.{e}") for i, e in rows]
    column = image_column([b for b, _ in pairs], [p for _, p in pairs])
    name = config_name(threshold)
    write_table(
        with_hf_features(subset.append_column("image", column)), root / "data" / name / PART
    )
    return Subset(name, threshold, subset.num_rows)


def _count(column: pa.ChunkedArray) -> dict[str, int]:
    return {v["values"]: v["counts"] for v in pc.value_counts(column).to_pylist()}
