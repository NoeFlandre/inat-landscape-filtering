"""Publish stage: build the dataset configs and the card, and upload them."""

import shutil
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image

from inat_landscape.adapters import figures
from inat_landscape.adapters.files import (
    read_json,
    read_table,
    write_table,
)
from inat_landscape.adapters.geo import parents
from inat_landscape.adapters.hf_parquet import image_column, with_hf_features
from inat_landscape.adapters.hub import REPO_ID, Hub
from inat_landscape.adapters.images import decode, fetch
from inat_landscape.adapters.inat_dump import SNAPSHOT_URL
from inat_landscape.domain.card import CardInfo, Subset, breakdown, render_card
from inat_landscape.domain.cells import MAP_RES, group_mean, top_per_cell
from inat_landscape.domain.diversity import to_unit_range
from inat_landscape.domain.examples import band_label, pick_examples, score_bands
from inat_landscape.domain.layers import complete_cap
from inat_landscape.domain.scoring import CLOSEUP_PROMPTS, LANDSCAPE_PROMPTS, NEGATIVE_PROMPTS
from inat_landscape.domain.stats import key_quantiles
from inat_landscape.domain.thresholds import config_name
from inat_landscape.pipeline.diversity import COMPONENTS, read_diversity
from inat_landscape.pipeline.embed import (
    HUB_DIR,
    SCORING_SIZE,
    embedding_dir,
    photo_url,
    read_sample,
)
from inat_landscape.pipeline.errors import StageError, require
from inat_landscape.pipeline.ingest import DEFAULT_RES, MANIFEST, ingest_params
from inat_landscape.pipeline.land import LAND, LAND_GRID, LAND_REPORT
from inat_landscape.pipeline.ports import Fetcher
from inat_landscape.pipeline.score import EMBEDDINGS, SCORE, SCORED

IMAGE_SIZE = "medium"
PART = "part-00000.parquet"
LAYER_TOLERANCE = 0.002  # share of failed downloads that a complete layer can miss
EXAMPLES_PER_BAND = 6  # photos in each row of the card mosaic
STAGING = Path("publish-parts")  # one image part at a time, deleted after its upload
DIVERSITY_MAPS = ("assets/diversity.png", "assets/look.png")


@dataclass(frozen=True)
class PublishSpec:
    """Which score defines the configs, their thresholds, and the cap per H3 cell."""

    thresholds: tuple[float, ...]
    score: str = SCORE
    cap_per_cell: int = 100
    images: bool = True
    part_rows: int = 2000  # rows of the lowest config per image part (about 300 MB)


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
    """Write `all`, one config per threshold, the figures and the card; then upload them.

    With `spec.images`, the threshold configs carry the 500 px image and are uploaded part by
    part (bounded disk); otherwise they hold the metadata only (quick refresh during a run).
    """
    table, layer_cap = _write_all(workdir)
    root = workdir / HUB_DIR
    kept = [_kept_mask(table, spec, t) for t in spec.thresholds]
    names = [config_name(t, spec.score) for t in spec.thresholds]
    has_image, delete = _write_configs(workdir, table, (names, kept), spec, (hub, fetcher))
    masks = [m & has_image for m in kept]
    subsets = tuple(
        Subset(n, t, int(m.sum())) for n, t, m in zip(names, spec.thresholds, masks, strict=True)
    )
    _figures(root / "assets", workdir, table, spec)
    _examples(root / "assets" / "examples.jpg", table, spec, fetcher)
    info = _card_info(
        workdir,
        table,
        list(zip(subsets, masks, strict=True)),
        spec,
        model_id=model_id,
        repo_id=repo_id,
    )
    (root / "README.md").write_text(render_card(replace(info, layer_cap=layer_cap)))
    hub.upload_folder(
        root,
        f"Publish {table.num_rows} scored photos",
        allow=["README.md", "data/**", "assets/**"],
        delete=delete,
    )
    return PublishReport(table.num_rows, subsets, int((kept[0] & ~has_image).sum()))


def run_maps(workdir: Path, hub: Hub) -> list[str]:
    """Draw the maps again and upload them alone: the card does not change.

    The map of photos per cell is drawn too if the `all` config is in the work directory.
    Return the uploaded files.
    """
    if read_diversity(workdir) is None:
        raise StageError(f"no diversity at {workdir}: run `diversity` first")
    root, params = workdir / HUB_DIR, ingest_params(workdir)
    res = params.get("res", DEFAULT_RES)
    folder, land = root / "assets", _land_cells(workdir)
    maps = list(DIVERSITY_MAPS)
    published = root / "data" / "all"
    if published.exists():
        columns = ["h3_cell"]
        cap = params.get("cap", PublishSpec.cap_per_cell)
        _world_map(folder, read_table(published, columns), cap, res, land)
        maps.insert(0, "assets/map.png")
    _diversity_maps(folder, workdir, res, land)
    hub.upload_folder(root, "Update the maps", allow=maps, delete=[])
    return maps


def _write_all(workdir: Path) -> tuple[pa.Table, int | None]:
    """Clear the local configs and figures, and write `all` (the published photos, with URLs).

    With a uniform sample (ADR 0011), only the complete layers are published: the photos of
    rank below the cap that every cell has reached. Return the table and that cap.
    """
    require(workdir / SCORED, "scores", "score")
    table, layer_cap = _complete_layers(workdir, read_table(workdir / SCORED))
    table = _with_urls(table)
    root = workdir / HUB_DIR
    for folder in ("data", "assets"):
        shutil.rmtree(root / folder, ignore_errors=True)
    write_table(with_hf_features(table), root / "data" / "all" / PART)
    return table, layer_cap


def _complete_layers(workdir: Path, scored: pa.Table) -> tuple[pa.Table, int | None]:
    sample = read_sample(workdir)
    if "cell_rank" not in sample.column_names:
        return scored, None  # a sample of before ADR 0011: every scored photo is published
    ranks = sample["cell_rank"].to_numpy()
    done = pc.is_in(sample["photo_id"], scored["photo_id"]).to_numpy(zero_copy_only=False)
    cap = complete_cap(ranks, done, LAYER_TOLERANCE)
    if cap == 0:
        raise StageError("no complete layer yet: every cell needs its first photo")
    return scored.filter(pc.less(scored["cell_rank"], cap)), cap


Configs = tuple[list[str], list[np.ndarray]]  # config names, and the rows each one keeps


def _write_configs(
    workdir: Path, table: pa.Table, configs: Configs, spec: PublishSpec, io: tuple[Hub, Fetcher]
) -> tuple[np.ndarray, list[str]]:
    """Write the threshold configs; return which rows have an image, and the remote files
    that the final upload replaces."""
    hub, fetcher = io
    if spec.images:
        has_image = _upload_image_parts(workdir, table, configs, spec.part_rows, hub, fetcher)
        return has_image, ["data/all/**", "assets/**"]
    for name, mask in zip(*configs, strict=True):
        subset = with_hf_features(table.filter(pa.array(mask)))
        write_table(subset, workdir / HUB_DIR / "data" / name / PART)
    return np.ones(table.num_rows, dtype=bool), ["data/**", "assets/**"]


def _upload_image_parts(
    workdir: Path, table: pa.Table, configs: Configs, part_rows: int, hub: Hub, fetcher: Fetcher
) -> np.ndarray:
    """Upload the threshold configs with their images, one commit per part of the lowest
    config; return which rows have an image. The configs on the hub are replaced."""
    names, kept = configs
    for config in {f.split("/")[1] for f in hub.list_files("data/")} - {"all"}:
        hub.delete_folder(f"data/{config}", f"Remove config {config} before the image build")
    rows = np.flatnonzero(kept[0])
    has_image = np.zeros(table.num_rows, dtype=bool)
    staging = workdir / STAGING
    for p, start in enumerate(range(0, max(len(rows), 1), part_rows)):
        part = rows[start : start + part_rows]
        images = _medium_images(table.take(pa.array(part, pa.int64())), fetcher)
        has_image[part] = np.isin(table["photo_id"].to_numpy()[part], list(images))
        for name, mask in zip(names, kept, strict=True):
            chosen = part[mask[part] & has_image[part]]
            subset = _with_images(table.take(pa.array(chosen, pa.int64())), images)
            write_table(with_hf_features(subset), staging / "data" / name / f"part-{p:05d}.parquet")
        hub.upload_folder(staging, f"Add image part {p}", allow=["data/**"], delete=[])
        shutil.rmtree(staging)
    return has_image


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
    embeddings = read_json(workdir / EMBEDDINGS) or {}
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
        embedding_dim=embeddings.get("embedding_dim", 0),
        embedding_rows=embeddings.get("embedding_rows", 0),
        embedding_photos=embeddings.get("embedding_photos", embeddings.get("embedding_rows", 0)),
        no_location=table["latitude"].null_count,
        score=spec.score,
        cap_per_cell=spec.cap_per_cell,
        embed_cap=params.get("cap", 0),
        h3_res=params.get("res", 0),
        with_images=spec.images,
        **_sea(workdir),
        **_diversity_info(workdir),
    )


def _diversity_info(workdir: Path) -> dict:
    """Return the photos per cell of the diversity maps (none: no diversity stage)."""
    found = read_diversity(workdir)
    if found is None:
        return {}
    params = found[1]
    return {"diversity_per_cell": params["per_cell"], "diversity_min_photos": params["min_photos"]}


def _sea(workdir: Path) -> dict:
    """Return the photos and cells at sea that the land stage left out (none: no stage)."""
    land = read_json(workdir / LAND_REPORT)
    if land is None:
        return {}
    return {"sea_photos": land["sea_photos"], "sea_cells": land["sea_cells"]}


def _years(table: pa.Table) -> tuple[str, str]:
    """Return the first and last observation years."""
    dates = pc.drop_null(table["observed_on"])
    return str(pc.min(dates))[:4], str(pc.max(dates))[:4]


def _figures(folder: Path, workdir: Path, table: pa.Table, spec: PublishSpec) -> None:
    params = ingest_params(workdir)
    figures.pipeline(folder / "hero.png", with_images=spec.images, land=(workdir / LAND).exists())
    cap, res = params.get("cap", spec.cap_per_cell), params.get("res", DEFAULT_RES)
    land = _land_cells(workdir)
    _world_map(folder, table, cap, res, land)
    _diversity_maps(folder, workdir, res, land)
    figures.score_histogram(
        folder / "scores.png",
        table[spec.score].to_pylist(),
        spec.thresholds,
        label=spec.score,
        log=spec.score == SCORE,
    )


def _world_map(folder: Path, table: pa.Table, cap: int, res: int, land: np.ndarray | None) -> None:
    """Draw the count of photos of each cell (photos with no location are not on the map)."""
    cells, counts = np.unique(pc.drop_null(table["h3_cell"]).to_numpy(), return_counts=True)
    figures.world_map(folder / "map.png", _on_map(cells, counts), cap=cap, res=res, land=land)


def _on_map(cells: np.ndarray, values: np.ndarray) -> figures.Cells:
    """Return the map cells, each with the mean of the values of its cells."""
    return group_mean(parents(cells, MAP_RES), np.asarray(values, float))


def _land_cells(workdir: Path) -> np.ndarray | None:
    """Return the map cells that hold land cells, if the land stage ran."""
    if not (workdir / LAND_GRID).exists():
        return None
    return np.unique(parents(read_table(workdir / LAND_GRID)["h3_cell"].to_numpy(), MAP_RES))


def _diversity_maps(folder: Path, workdir: Path, res: int, land: np.ndarray | None) -> None:
    """Draw the visual diversity and the colour of each cell, if the diversity stage ran."""
    found = read_diversity(workdir)
    if found is None:
        return
    table, params = found
    cells = table["h3_cell"].to_numpy()
    figures.diversity_map(
        folder / "diversity.png",
        _on_map(cells, table["diversity"].to_numpy()),
        per_cell=params["per_cell"],
        res=res,
        land=land,
    )
    pcs = np.stack([table[f"pc{i}"].to_numpy() for i in range(1, COMPONENTS + 1)], axis=1)
    figures.look_map(folder / "look.png", _on_map(cells, to_unit_range(pcs)), land=land)


def _examples(path: Path, table: pa.Table, spec: PublishSpec, fetcher: Fetcher) -> None:
    """Draw a few random CC0 photos (240 px) of each score band, for the card."""
    bands = score_bands(spec.thresholds)
    columns = table.select(["photo_id", spec.score, "license"]).to_pydict()
    picked = pick_examples(*columns.values(), bands=bands, per_band=EXAMPLES_PER_BAND)
    images = _example_images(table, list(chain.from_iterable(picked)), fetcher)
    rows = [
        (band_label(band), _present(chosen, images))
        for band, chosen in zip(bands, picked, strict=True)
    ]
    figures.examples_mosaic(path, rows, per_band=EXAMPLES_PER_BAND, label=spec.score)


def _present(ids: list[int], images: dict[int, Image.Image]) -> list[Image.Image]:
    return [images[i] for i in ids if i in images]


def _example_images(table: pa.Table, ids: list[int], fetcher: Fetcher) -> dict[int, Image.Image]:
    """Download and decode the 240 px photos of `ids`. A photo that fails is left out."""
    chosen = table.filter(pc.is_in(table["photo_id"], pa.array(ids, pa.int64())))
    urls = _small_urls(chosen)
    decoded = {f.photo_id: decode(f.data or b"") for f in fetcher(urls)}
    return {i: image for i, image in decoded.items() if image is not None}


def _small_urls(rows: pa.Table) -> dict[int, str]:
    return {i: photo_url(i, SCORING_SIZE, e) for i, e in _id_extensions(rows)}


def _id_extensions(rows: pa.Table) -> list[tuple[int, str]]:
    """Return the photo id and the file extension of each row."""
    return list(zip(rows["photo_id"].to_pylist(), rows["extension"].to_pylist(), strict=True))


def _with_urls(table: pa.Table) -> pa.Table:
    pairs = [
        (photo_url(i, SCORING_SIZE, e), photo_url(i, IMAGE_SIZE, e))
        for i, e in _id_extensions(table)
    ]
    table = table.append_column("url_small", pa.array([p[0] for p in pairs], pa.string()))
    return table.append_column("url_medium", pa.array([p[1] for p in pairs], pa.string()))


def _medium_images(rows: pa.Table, fetcher: Fetcher) -> dict[int, bytes]:
    """Download the medium image of each row; failed downloads are left out."""
    results = fetcher({i: photo_url(i, IMAGE_SIZE, e) for i, e in _id_extensions(rows)})
    return {r.photo_id: r.data for r in results if r.data is not None}


def _with_images(rows: pa.Table, images: dict[int, bytes]) -> pa.Table:
    names = [(i, f"{i}.{e}") for i, e in _id_extensions(rows)]
    column = image_column([images[i] for i, _ in names], [n for _, n in names])
    return rows.append_column("image", column)
