"""PNG figures for the dataset card (matplotlib, no display needed)."""

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

BG = "#fbfbf9"
INK = "#24343f"
MUTED = "#5b6670"
LINE = "#5f6f78"
GREENS = ("#e4eed9", "#b9d59c", "#86b766", "#5a9640", "#3a7429", "#22531a", "#123510")
BLUES = ("#b9d3f5", "#86b4ef", "#5596e6", "#2b78d6", "#1c5aa8", "#11407e", "#0b2f61")
KEPT = "#b8622f"
LAND_GREY = "#e8e8e6"  # land with no value: not mistaken for the sea (white)
PANEL = {
    "collect": ("#f6f7f9", "#cfd6dc"),
    "score": ("#f8f7fa", "#d3d0db"),
    "dataset": ("#f4f8f5", "#c9d8cd"),
}
CHIP = {
    "purple": ("#e9e2f0", "#a99ab8"),
    "blue": ("#dde9ef", "#9fb7c4"),
    "green": ("#e3ecdc", "#a6b997"),
    "sand": ("#f1e6d8", "#c7ae8f"),
}

_Axis = Sequence[float] | np.ndarray
Points = tuple[_Axis, _Axis]  # longitudes, latitudes
CellCounts = tuple[Sequence[float], Sequence[float], Sequence[int]]  # lon, lat, photos per cell
CellValues = tuple[np.ndarray, np.ndarray, np.ndarray]  # lon, lat, one value per cell
_MAX_LINEAR_BINS = 12
_EXTENT = (-180, 180, -60, 84)  # west, east, south, north: the maps leave out Antarctica
_MAP_SIZE = (16, 8.6)
_HEXAGON = 2.0  # degrees: the hexagons of the maps
_CHIP_FONT = 12
_MIN_FONT = 6  # a chip label does not get smaller than this
Box = tuple[float, float, float, float]  # x, y, width, height

mpl.rcParams.update(
    {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


def collect_steps(*, land: bool = False) -> tuple[str, ...]:
    """Return the steps of the Collect panel of the pipeline diagram."""
    return ("Metadata stream", "UUID sample", "All taxa", *(("On land",) if land else ()))


def dataset_rows(*, with_images: bool) -> tuple[str, ...]:
    """Return the configs of the Dataset panel of the pipeline diagram."""
    content = "500 px image" if with_images else "500 px URL"
    return ("all · metadata + scores", f"score ≥ t · + {content}", "embeddings · 768-d")


def pipeline(path: Path, *, with_images: bool = True, land: bool = False) -> None:
    """Draw the pipeline diagram: collect, score, dataset."""
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.set(xlim=(0, 160), ylim=(0, 64))
    ax.axis("off")
    for x, title in ((4, "Collect"), (58, "Score"), (112, "Dataset")):
        ax.text(x, 60, title, fontsize=20, fontweight="bold", color=INK, va="center")
    _collect(ax, land=land)
    _score(ax)
    _dataset(ax, with_images=with_images)
    for x0, x1 in ((48, 57), (102, 111)):
        ax.annotate(
            "", (x1, 31), (x0, 31), arrowprops={"arrowstyle": "-|>", "color": LINE, "lw": 1.4}
        )
    _save(fig, path)


def _collect(ax: plt.Axes, *, land: bool) -> None:
    _panel(ax, (4, 4, 44, 50), "collect")
    ax.add_patch(Rectangle((7, 28), 15, 15, color="#e5e9ec"))
    ax.add_patch(Rectangle((9, 31), 11, 9, fill=False, ec="#2f6b70", lw=2))
    ax.plot([10, 13, 15, 16.5, 19], [32, 36, 34, 35.5, 32], color="#2f6b70", lw=2)
    ax.add_patch(plt.Circle((17.5, 38), 0.9, fill=False, ec="#2f6b70", lw=1.6))
    ax.text(
        14.5,
        25,
        "iNaturalist\nOpen Data",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=INK,
        linespacing=1.3,
    )
    steps = collect_steps(land=land)
    gap = 33 / len(steps)  # the chips fill the panel from y = 49 down to y = 16
    centers = [49 - gap * (i + 0.5) for i in range(len(steps))]
    ax.plot([22, 25], [35.5, 35.5], color=LINE, lw=1.2)
    ax.plot([25, 25], [centers[-1], centers[0]], color=LINE, lw=1.2)
    for y, label, tone in zip(centers, steps, ("purple", "blue", "green", "sand"), strict=False):
        _chip(ax, (27.5, y - gap * 0.4, 19, gap * 0.8), label, tone)
        ax.annotate(
            "", (27.5, y), (25, y), arrowprops={"arrowstyle": "-|>", "color": LINE, "lw": 1.2}
        )
    ax.text(
        26,
        10,
        "CC0 · CC-BY · CC-BY-NC",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#3f7a4a",
    )


def _score(ax: plt.Axes) -> None:
    _panel(ax, (58, 4, 44, 50), "score")
    _chip(ax, (62, 42, 16, 8), "240 px photo", "sand")
    _chip(ax, (82, 42, 16, 8), "Prompts", "purple")
    _chip(ax, (66, 24, 28, 9), "SigLIP2\nsigmoid(scale·cos + bias)", "blue")
    for head, tail in (
        ((70, 33), (70, 42)),
        ((88, 33), (90, 42)),
        ((70, 13), (80, 24)),
        ((90, 13), (80, 24)),
    ):
        ax.annotate("", head, tail, arrowprops={"arrowstyle": "-|>", "color": LINE, "lw": 1})
    ax.text(70, 10, "LANDSCAPE", ha="center", fontsize=12, fontweight="bold", color="#3f7a4a")
    ax.text(90, 10, "CLOSE-UP", ha="center", fontsize=12, fontweight="bold", color="#9a5b4e")


def _dataset(ax: plt.Axes, *, with_images: bool) -> None:
    _panel(ax, (112, 4, 44, 50), "dataset")
    rows = zip(dataset_rows(with_images=with_images), ("blue", "green", "purple"), strict=True)
    for i, (label, tone) in enumerate(rows):
        _chip(ax, (117, 41 - i * 12, 34, 8), label, tone)
    for k, (dx, dy) in enumerate(((0, 0), (5, 3), (9, -1))):
        x, y = 125 + dx, 8 + dy
        ax.plot(x, y + 2, marker="v", ms=14, color=KEPT, zorder=3 + k)
        ax.plot(x, y + 2.6, marker="o", ms=4, color="white", zorder=4 + k)
    ax.text(140, 10, "geolocated\nphotos", va="center", fontsize=12, fontweight="bold", color=INK)


def _panel(ax: plt.Axes, box: Box, tone: str) -> None:
    x, y, w, h = box
    face, edge = PANEL[tone]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=face, ec=edge, lw=1.2
        )
    )


def _chip(ax: plt.Axes, box: Box, label: str, tone: str) -> None:
    x, y, w, h = box
    face, edge = CHIP[tone]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=face, ec=edge, lw=1.2
        )
    )
    text = ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=_CHIP_FONT,
        fontweight="bold",
        color=INK,
        linespacing=1.3,
    )
    _fit(ax, text, w - 2, h - 1)


def _fit(ax: plt.Axes, text: plt.Text, width: float, height: float) -> None:
    """Make the font smaller until the text fits in width x height (data units)."""
    size = float(_CHIP_FONT)
    inverse = ax.transData.inverted()
    while size > _MIN_FONT:
        box = text.get_window_extent().transformed(inverse)
        if box.width <= width and box.height <= height:
            return
        size -= 0.5
        text.set_fontsize(size)


def world_map(
    path: Path, cells: CellCounts, *, cap: int, res: int, land: Points | None = None
) -> None:
    """Draw how full the H3 cells are against the cap (mean per 2° hexagon)."""
    fig, ax = plt.subplots(figsize=_MAP_SIZE)
    _land(ax, land, _HEXAGON)
    bounds = _cap_bounds(cap)
    lon, lat, count = cells
    filled = _hexbin(ax, (lon, lat), count, _palette(BLUES, len(bounds) - 1), _norm(bounds))
    labels = [_bin_label(lo, hi) for lo, hi in pairwise(bounds)]
    label = f"Scored photos per H3 cell (resolution {res}, cap {cap}), mean per 2° hexagon"
    _colorbar(fig, ax, filled, bounds, labels, label)
    _world(ax)
    _save(fig, path)


def diversity_map(
    path: Path, cells: CellValues, *, per_cell: int, res: int, land: Points | None = None
) -> None:
    """Draw the visual diversity of each H3 cell (mean per 2° hexagon), in 10 quantile bins."""
    fig, ax = plt.subplots(figsize=_MAP_SIZE)
    _land(ax, land, _HEXAGON)
    lon, lat, values = cells
    bounds = _quantile_bounds(np.asarray(values, float), 10)
    filled = _hexbin(ax, (lon, lat), values, _palette(GREENS, len(bounds) - 1), _norm(bounds))
    labels = [f"{lo:.2f}" for lo in bounds[:-1]]
    label = (
        f"Visual diversity per H3 cell (resolution {res}): mean cosine distance between the "
        f"embeddings of {per_cell} photos, mean per 2° hexagon"
    )
    _colorbar(fig, ax, filled, bounds, labels, label)
    _world(ax)
    _save(fig, path)


def look_map(  # noqa: PLR0913 - the points, their colours and how to draw them
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    rgb: np.ndarray,
    *,
    land: Points | None = None,
    size: float = _HEXAGON,
    dpi: int = 150,
) -> None:
    """Draw the colour of each H3 cell (mean per `size`° hexagon; 2° as in the other maps):
    similar colours, similar photos. A smaller `size` and a larger `dpi` show more detail."""
    fig, ax = plt.subplots(figsize=_MAP_SIZE)
    _land(ax, land, size)
    black = ListedColormap(["#000000"])
    channels = [_hexbin(ax, (lon, lat), rgb[:, c], black, size=size) for c in range(3)]
    colors = np.stack([np.asarray(c.get_array()) for c in channels], axis=1)
    for extra in channels[1:]:  # same hexagons: keep one collection, with the RGB colours
        extra.remove()
    channels[0].set_array(None)
    channels[0].set_facecolor(colors)
    ax.text(
        0.5,
        -0.02,
        "What the photos show: the 3 first principal components of the mean embedding of each "
        "H3 cell (resolution 5)\nas red, green and blue, "
        f"mean per {size:g}° hexagon (similar colours: similar photos)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=13,
        color=MUTED,
    )
    _world(ax)
    _save(fig, path, dpi)


def _quantile_bounds(values: np.ndarray, bins: int) -> list[float]:
    """Return the edges of `bins` bins with the same count of values (fewer if values repeat).
    The largest value is in the last bin."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return [float(edges[0]), *edges[1:-1].tolist(), float(np.nextafter(edges[-1], np.inf))]


def _palette(colors: Sequence[str], n: int) -> ListedColormap:
    return ListedColormap(LinearSegmentedColormap.from_list("ramp", colors)(np.linspace(0, 1, n)))


def _norm(bounds: Sequence[float]) -> BoundaryNorm:
    return BoundaryNorm(bounds, len(bounds) - 1)


def _land(ax: plt.Axes, land: Points | None, size: float) -> None:
    """Draw in light grey, under the values, each hexagon that has a land point."""
    if land is None:
        return
    lon, lat = (np.asarray(a, float) for a in land)
    ax.hexbin(
        lon,
        lat,
        gridsize=_grid(size),
        extent=_EXTENT,
        linewidths=0,
        mincnt=1,
        cmap=ListedColormap([LAND_GREY]),
        zorder=0,
    )


def _grid(size: float) -> tuple[int, int]:
    """Return the hexagons across and down the map extent, for hexagons `size` degrees wide."""
    return round((_EXTENT[1] - _EXTENT[0]) / size), round((_EXTENT[3] - _EXTENT[2]) / size)


def _world(ax: plt.Axes) -> None:
    ax.set(xlim=_EXTENT[:2], ylim=_EXTENT[2:], aspect="equal")
    ax.axis("off")


def _colorbar(
    fig: plt.Figure, ax: plt.Axes, mappable, bounds: Sequence[float], labels: list[str], label: str
) -> None:
    bar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        fraction=0.04,
        pad=0.02,
        aspect=45,
        ticks=[(lo + hi) / 2 for lo, hi in pairwise(bounds)],
    )
    bar.ax.set_xticklabels(labels)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=12, colors=MUTED)
    bar.set_label(label, fontsize=13, color=MUTED)


def _cap_bounds(cap: int) -> list[float]:
    """Bin edges from 1 to the cap: one bin per value for small caps, log bins otherwise."""
    if cap <= _MAX_LINEAR_BINS:
        return [float(v) for v in range(1, cap + 2)]
    return sorted({float(round(v)) for v in np.geomspace(1, cap + 1, _MAX_LINEAR_BINS)})


def _bin_label(lo: float, hi: float) -> str:
    return f"{lo:.0f}" if hi - lo <= 1 else f"{lo:.0f}-{hi - 1:.0f}"


def _hexbin(
    ax: plt.Axes,
    points: Points,
    values: Sequence[float] | np.ndarray,
    cmap: ListedColormap,
    norm: BoundaryNorm | None = None,
    *,
    size: float = _HEXAGON,
):
    """Draw the mean of `values` in each `size`° hexagon that has a point."""
    return ax.hexbin(
        np.asarray(points[0], float),
        np.asarray(points[1], float),
        C=np.asarray(values, dtype=float),
        reduce_C_function=np.mean,
        gridsize=_grid(size),
        extent=_EXTENT,
        linewidths=0,
        mincnt=1,
        cmap=cmap,
        norm=norm,
    )


def score_histogram(
    path: Path, scores: Sequence[float], thresholds: Sequence[float], *, label: str, log: bool
) -> None:
    """Draw the distribution of scores, on a log axis when `log`, with the thresholds."""
    values = np.asarray(scores, dtype=float)
    if log:
        values = np.clip(values, 1e-9, 1)
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.hist(values, bins=_bins(values, log), color=BLUES[3], edgecolor=BG, linewidth=0.5)
    if log:
        ax.set_xscale("log")
    _threshold_lines(ax, values, thresholds)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=12, left=False)
    ax.set_yticks([])
    ax.set_xlabel(f"{label} ({'log' if log else 'linear'} scale)", fontsize=13, color=MUTED)
    _save(fig, path)


def _threshold_lines(ax: plt.Axes, values: np.ndarray, thresholds: Sequence[float]) -> None:
    top = ax.get_ylim()[1]
    for i, t in enumerate(thresholds):
        n = int((values >= t).sum())
        ax.axvline(t, color=KEPT, lw=1.6)
        ax.text(
            t,
            top * (0.95 - 0.17 * i),
            f" ≥ {t}: {n:,} photos",
            color=KEPT,
            fontsize=12,
            va="top",
            bbox={"fc": BG, "ec": "none", "pad": 2},
        )


def _bins(values: np.ndarray, log: bool) -> list[float]:
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return [low - 0.5, high + 0.5]
    edges = np.logspace(np.log10(low), np.log10(high), 70) if log else np.linspace(low, high, 70)
    return edges.tolist()


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def examples_mosaic(
    path: Path, rows: Sequence[tuple[str, Sequence[Image.Image]]], *, per_band: int, label: str
) -> None:
    """Draw one row of square photo tiles for each score band, lowest band at the bottom."""
    tile = 1.25  # inches; with 100 dpi, a tile is 125 px: small, yet the scene shows
    fig, axes = plt.subplots(
        len(rows), per_band, figsize=(per_band * tile + 1.4, len(rows) * tile), squeeze=False
    )
    for ax_row, (band, images) in zip(axes, reversed(rows), strict=True):
        for i, ax in enumerate(ax_row):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i < len(images):
                ax.imshow(_square(images[i]))
        ax_row[0].set_ylabel(band, rotation=0, ha="right", va="center", fontsize=11, color=INK)
    fig.suptitle(f"{label}: random CC0 photos of each band", fontsize=12, color=MUTED)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.17, right=0.99, bottom=0.01, top=0.93)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 80})
    plt.close(fig)


def _square(image: Image.Image) -> Image.Image:
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.convert("RGB").crop((left, top, left + side, top + side))
