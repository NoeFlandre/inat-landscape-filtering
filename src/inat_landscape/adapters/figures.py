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

BG = "#fbfbf9"
INK = "#24343f"
MUTED = "#5b6670"
LINE = "#5f6f78"
BLUES = ("#b9d3f5", "#86b4ef", "#5596e6", "#2b78d6", "#1c5aa8", "#11407e", "#0b2f61")
KEPT = "#b8622f"
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

mpl.rcParams.update(
    {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    }
)


def pipeline(path: Path) -> None:
    """Draw the pipeline diagram: collect, score, dataset."""
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.set(xlim=(0, 160), ylim=(0, 64))
    ax.axis("off")
    for x, title in ((4, "Collect"), (58, "Score"), (112, "Dataset")):
        ax.text(x, 60, title, fontsize=20, fontweight="bold", color=INK, va="center")
    _collect(ax)
    _score(ax)
    _dataset(ax)
    for x0, x1 in ((48, 57), (102, 111)):
        ax.annotate(
            "", (x1, 31), (x0, 31), arrowprops={"arrowstyle": "-|>", "color": LINE, "lw": 1.4}
        )
    _save(fig, path)


def _collect(ax: plt.Axes) -> None:
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
    centers = (44, 33, 22)
    ax.plot([22, 25], [35.5, 35.5], color=LINE, lw=1.2)
    ax.plot([25, 25], [centers[-1], centers[0]], color=LINE, lw=1.2)
    for y, label, tone in zip(
        centers,
        ("Metadata stream", "UUID sample", "Plants + no ID"),
        ("purple", "blue", "green"),
        strict=True,
    ):
        _chip(ax, (27.5, y - 3.5, 19, 7), label, tone)
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


def _dataset(ax: plt.Axes) -> None:
    _panel(ax, (112, 4, 44, 50), "dataset")
    rows = (
        ("all · metadata + scores", "blue"),
        ("score ≥ t · + 500 px image", "green"),
        ("embeddings · 768-d", "purple"),
    )
    for i, (label, tone) in enumerate(rows):
        _chip(ax, (117, 41 - i * 12, 34, 8), label, tone)
    for k, (dx, dy) in enumerate(((0, 0), (5, 3), (9, -1))):
        x, y = 125 + dx, 8 + dy
        ax.plot(x, y + 2, marker="v", ms=14, color=KEPT, zorder=3 + k)
        ax.plot(x, y + 2.6, marker="o", ms=4, color="white", zorder=4 + k)
    ax.text(140, 10, "geolocated\nphotos", va="center", fontsize=12, fontweight="bold", color=INK)


Points = tuple[Sequence[float | None], Sequence[float | None]]  # longitudes, latitudes
CellCounts = tuple[Sequence[float], Sequence[float], Sequence[int]]  # lon, lat, photos per cell
_MAX_LINEAR_BINS = 12
Box = tuple[float, float, float, float]  # x, y, width, height


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
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=INK,
        linespacing=1.3,
    )


def world_map(path: Path, pool: Points, cells: CellCounts, *, cap: int, res: int) -> None:
    """Draw how full the H3 cells are against the cap (mean per 2° hexagon), over the full
    sample in grey."""
    fig, ax = plt.subplots(figsize=(16, 8.6))
    _hexbin(ax, pool, ListedColormap(["#e6e6e3"]))
    bounds = _cap_bounds(cap)
    colors = LinearSegmentedColormap.from_list("blues", BLUES)(np.linspace(0, 1, len(bounds) - 1))
    norm = BoundaryNorm(bounds, len(bounds) - 1)
    lon, lat, count = cells
    filled = _hexbin(ax, (lon, lat), ListedColormap(colors), norm, values=count)
    ax.set(xlim=(-180, 180), ylim=(-60, 84), aspect="equal")
    ax.axis("off")
    bar = fig.colorbar(
        filled,
        ax=ax,
        orientation="horizontal",
        fraction=0.04,
        pad=0.02,
        aspect=45,
        ticks=[(lo + hi) / 2 for lo, hi in pairwise(bounds)],
    )
    bar.ax.set_xticklabels([_bin_label(lo, hi) for lo, hi in pairwise(bounds)])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=12, colors=MUTED)
    bar.set_label(
        f"Scored photos per H3 cell (resolution {res}, cap {cap}), mean per 2° hexagon "
        "(grey: the full sample)",
        fontsize=13,
        color=MUTED,
    )
    _save(fig, path)


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
    cmap: ListedColormap,
    norm: BoundaryNorm | None = None,
    values: Sequence[float] | None = None,
):
    x, y = _finite(points) if values is None else (np.asarray(points[0]), np.asarray(points[1]))
    return ax.hexbin(
        x,
        y,
        C=None if values is None else np.asarray(values, dtype=float),
        reduce_C_function=np.mean,
        gridsize=(180, 72),
        extent=(-180, 180, -60, 84),
        linewidths=0,
        mincnt=1,
        cmap=cmap,
        norm=norm,
    )


def _finite(points: Points) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([[np.nan if v is None else v for v in axis] for axis in points], dtype=float)
    keep = np.isfinite(xy).all(axis=0)
    return xy[0, keep], xy[1, keep]


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


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
