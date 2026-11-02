"""Publish stage: build the dataset configs and the card, and upload them."""

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from inat_landscape.adapters import figures
from inat_landscape.adapters.embedding_store import EmbeddingStore
from inat_landscape.adapters.files import read_json, read_table, write_bytes, write_table
from inat_landscape.adapters.hf_parquet import image_column, with_hf_features
from inat_landscape.adapters.hub import REPO_ID, Hub, OfflineHub
from inat_landscape.adapters.images import fetch
from inat_landscape.adapters.inat_dump import SNAPSHOT_URL
from inat_landscape.domain.card import CardInfo, Subset, breakdown, render_card
from inat_landscape.domain.scoring import CLOSEUP_PROMPTS, LANDSCAPE_PROMPTS
from inat_landscape.domain.stats import key_quantiles
from inat_landscape.domain.thresholds import config_name
from inat_landscape.pipeline.embed import HUB_DIR, SCORING_SIZE, embedding_dir, photo_url
from inat_landscape.pipeline.errors import StageError
from inat_landscape.pipeline.ingest import CANDIDATES, MANIFEST
from inat_landscape.pipeline.ports import Fetcher
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
    repo_id: str = REPO_ID,
) -> PublishReport:
    """Write `all`, one config per threshold, the figures and the card; then upload them."""
    if not (workdir / SCORED).exists():
        raise StageError(f"no scores at {workdir / SCORED}: run `score` first")
    table = _with_urls(read_table(workdir / SCORED))
    root = workdir / HUB_DIR
    for folder in ("data", "assets"):
        shutil.rmtree(root / folder, ignore_errors=True)
    write_table(with_hf_features(table), root / "data" / "all" / PART)
    kept = table.filter(pc.greater_equal(table[SCORE], min(thresholds)))
    images = _medium_images(workdir, kept, fetcher)
    subsets = tuple(_write_subset(root, table, t, images) for t in thresholds)
    _figures(root / "assets", workdir, table, thresholds)
    info = _card_info(workdir, table, subsets, model_id=model_id, repo_id=repo_id)
    (root / "README.md").write_text(render_card(info))
    hub.upload_folder(
        root,
        f"Publish {table.num_rows} scored photos",
        allow=["README.md", "data/**", "assets/**"],
        delete=["data/**", "assets/**"],
    )
    return PublishReport(table.num_rows, subsets, kept.num_rows - len(images))


def _card_info(
    workdir: Path, table: pa.Table, subsets: tuple[Subset, ...], *, model_id: str, repo_id: str
) -> CardInfo:
    manifest = read_json(workdir / MANIFEST) or {}
    scores = table[SCORE].to_pylist()
    thresholds = [s.threshold for s in subsets]
    store = EmbeddingStore(workdir / HUB_DIR, embedding_dir(model_id), OfflineHub())
    return CardInfo(
        repo_id=repo_id,
        snapshot=manifest.get("source", SNAPSHOT_URL),
        model_id=model_id,
        landscape_prompts=LANDSCAPE_PROMPTS,
        closeup_prompts=CLOSEUP_PROMPTS,
        fraction=manifest.get("fraction", 0.0),
        candidates=manifest.get("candidates", table.num_rows),
        total=table.num_rows,
        subsets=subsets,
        groups=breakdown(table["group"].to_pylist(), scores, thresholds),
        licenses=breakdown(table["license"].to_pylist(), scores, thresholds),
        quantiles=key_quantiles(scores),
        years=_years(table),
        embedding_dir=embedding_dir(model_id),
        embedding_dim=store.dim(),
        no_location=table["latitude"].null_count,
    )


def _years(table: pa.Table) -> tuple[str, str]:
    """Return the first and last observation years."""
    dates = pc.drop_null(table["observed_on"])
    return str(pc.min(dates))[:4], str(pc.max(dates))[:4]


def _figures(folder: Path, workdir: Path, table: pa.Table, thresholds: Sequence[float]) -> None:
    pool = read_table(workdir / CANDIDATES).select(["longitude", "latitude"])
    kept = table.filter(pc.greater_equal(table[SCORE], min(thresholds)))

    def xy(t: pa.Table) -> tuple[list[float], list[float]]:
        return t["longitude"].to_pylist(), t["latitude"].to_pylist()

    figures.pipeline(folder / "hero.png")
    figures.world_map(
        folder / "map.png",
        xy(pool),
        xy(table),
        xy(kept),
        f"score_landscape ≥ {min(thresholds)} ({kept.num_rows:,} photos)",
    )
    figures.score_histogram(folder / "scores.png", table[SCORE].to_pylist(), thresholds)


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
