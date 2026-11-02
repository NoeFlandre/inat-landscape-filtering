"""Publish stage: build the dataset configs and the card, and upload them."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
from inat_landscape.domain.cells import top_per_cell
from inat_landscape.domain.scoring import CLOSEUP_PROMPTS, LANDSCAPE_PROMPTS, NEGATIVE_PROMPTS
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
class PublishSpec:
    """Which score defines the configs, their thresholds, and the cap per H3 cell."""

    thresholds: tuple[float, ...]
    score: str = SCORE
    cap_per_cell: int = 100


@dataclass(frozen=True)
class PublishReport:
    """What one publish run wrote."""

    total: int
    subsets: tuple[Subset, ...]
    images_failed: int


def run_publish(
    workdir: Path,
    spec: PublishSpec,
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
    lowest = _kept_mask(table, spec, spec.thresholds[0])
    images = _medium_images(workdir, table.filter(pa.array(lowest)), fetcher)
    ids = pa.array(list(images), pa.int64())
    has_image = pc.is_in(table["photo_id"], ids).to_numpy(zero_copy_only=False)
    masks = [_kept_mask(table, spec, t) & has_image for t in spec.thresholds]
    subsets = tuple(
        _write_subset(root, table.filter(pa.array(m)), spec, t, images)
        for m, t in zip(masks, spec.thresholds, strict=True)
    )
    _figures(root / "assets", workdir, table, spec, masks[0])
    info = _card_info(
        workdir,
        table,
        list(zip(subsets, masks, strict=True)),
        spec,
        model_id=model_id,
        repo_id=repo_id,
    )
    (root / "README.md").write_text(render_card(info))
    hub.upload_folder(
        root,
        f"Publish {table.num_rows} scored photos",
        allow=["README.md", "data/**", "assets/**"],
        delete=["data/**", "assets/**"],
    )
    return PublishReport(table.num_rows, subsets, int((lowest & ~has_image).sum()))


def _kept_mask(table: pa.Table, spec: PublishSpec, threshold: float) -> np.ndarray:
    """Photos at or above the threshold, at most `cap_per_cell` per cell, best scored first."""
    scores = table[spec.score].to_numpy()
    above = np.flatnonzero(scores >= threshold)
    cells = table["h3_cell"].to_numpy()[above]
    mask = np.zeros(table.num_rows, dtype=bool)
    mask[above[top_per_cell(cells, scores[above], spec.cap_per_cell)]] = True
    return mask


def _card_info(
    workdir: Path,
    table: pa.Table,
    published: list[tuple[Subset, np.ndarray]],
    spec: PublishSpec,
    *,
    model_id: str,
    repo_id: str,
) -> CardInfo:
    stored = read_json(workdir / MANIFEST) or {}
    params = stored.get("params", {})
    store = EmbeddingStore(workdir / HUB_DIR, embedding_dir(model_id), OfflineHub())
    kept = [m.tolist() for _, m in published]
    return CardInfo(
        repo_id=repo_id,
        snapshot=params.get("source", SNAPSHOT_URL),
        model_id=model_id,
        landscape_prompts=LANDSCAPE_PROMPTS,
        closeup_prompts=CLOSEUP_PROMPTS,
        negative_prompts=NEGATIVE_PROMPTS,
        fraction=params.get("fraction", 0.0),
        candidates=stored.get("candidates", table.num_rows),
        total=table.num_rows,
        subsets=tuple(s for s, _ in published),
        groups=breakdown(table["group"].to_pylist(), kept),
        licenses=breakdown(table["license"].to_pylist(), kept),
        quantiles=key_quantiles(table[spec.score].to_pylist()),
        years=_years(table),
        embedding_dir=embedding_dir(model_id),
        embedding_dim=store.dim(),
        embedding_rows=len(store.done_ids()),
        no_location=table["latitude"].null_count,
        score=spec.score,
        cap_per_cell=spec.cap_per_cell,
        embed_cap=params.get("cap", 0),
        h3_res=params.get("res", 0),
    )


def _years(table: pa.Table) -> tuple[str, str]:
    """Return the first and last observation years."""
    dates = pc.drop_null(table["observed_on"])
    return str(pc.min(dates))[:4], str(pc.max(dates))[:4]


def _figures(
    folder: Path, workdir: Path, table: pa.Table, spec: PublishSpec, lowest: np.ndarray
) -> None:
    pool = read_table(workdir / CANDIDATES).select(["longitude", "latitude"])
    kept = table.filter(pa.array(lowest))

    def xy(t: pa.Table) -> tuple[list[float], list[float]]:
        return t["longitude"].to_pylist(), t["latitude"].to_pylist()

    figures.pipeline(folder / "hero.png")
    label = f"{spec.score} ≥ {spec.thresholds[0]}, best {spec.cap_per_cell} per cell"
    figures.world_map(
        folder / "map.png", xy(pool), xy(table), xy(kept), f"{label} ({kept.num_rows:,} photos)"
    )
    figures.score_histogram(
        folder / "scores.png",
        table[spec.score].to_pylist(),
        spec.thresholds,
        label=spec.score,
        log=spec.score == SCORE,
    )


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
    root: Path, subset: pa.Table, spec: PublishSpec, threshold: float, images: dict[int, bytes]
) -> Subset:
    rows = zip(subset["photo_id"].to_pylist(), subset["extension"].to_pylist(), strict=True)
    pairs = [(images[i], f"{i}.{e}") for i, e in rows]
    column = image_column([b for b, _ in pairs], [p for _, p in pairs])
    name = config_name(threshold, spec.score)
    write_table(
        with_hf_features(subset.append_column("image", column)), root / "data" / name / PART
    )
    return Subset(name, threshold, subset.num_rows)
