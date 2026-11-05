"""Card figures: each function writes a PNG or a JPEG."""

import numpy as np
from PIL import Image

from inat_landscape.adapters import figures

PNG = b"\x89PNG\r\n\x1a\n"


def test_pipeline_figure(tmp_path):
    figures.pipeline(tmp_path / "a" / "hero.png")
    assert (tmp_path / "a" / "hero.png").read_bytes()[:8] == PNG


def test_world_map(tmp_path):
    cells = ([0.0, 20.0], [0.0, 10.0], [3, 10])
    figures.world_map(tmp_path / "map.png", cells, cap=10, res=5)
    assert (tmp_path / "map.png").read_bytes()[:8] == PNG


def test_world_map_with_a_large_cap(tmp_path):
    cells = ([0.0], [0.0], [700])
    figures.world_map(tmp_path / "m.png", cells, cap=1000, res=5)
    assert (tmp_path / "m.png").read_bytes()[:8] == PNG


def test_cap_bounds():
    assert figures._cap_bounds(3) == [1.0, 2.0, 3.0, 4.0]
    big = figures._cap_bounds(1000)
    assert big[0] == 1.0
    assert big[-1] == 1001.0
    assert len(big) <= 12
    assert figures._bin_label(1.0, 2.0) == "1"
    assert figures._bin_label(10.0, 32.0) == "10-31"


def test_score_histogram_handles_zero_and_one(tmp_path):
    figures.score_histogram(
        tmp_path / "s.png", [0.0, 1e-6, 0.002, 1.0], [0.001], label="s", log=True
    )
    assert (tmp_path / "s.png").read_bytes()[:8] == PNG


def test_score_histogram_linear_with_negative_scores(tmp_path):
    figures.score_histogram(tmp_path / "c.png", [-0.2, 0.0, 0.1], [0.05], label="c", log=False)
    assert (tmp_path / "c.png").read_bytes()[:8] == PNG


def test_score_histogram_with_one_value(tmp_path):
    figures.score_histogram(tmp_path / "o.png", [0.3, 0.3], [0.3], label="o", log=False)
    assert (tmp_path / "o.png").read_bytes()[:8] == PNG


def test_examples_mosaic_is_a_small_jpeg_with_one_row_per_band(tmp_path):
    red, wide = Image.new("RGB", (240, 180), "red"), Image.new("RGB", (240, 120), "blue")
    rows = [("< 0.00", [red, wide]), ("0.00 to 0.02", []), ("≥ 0.08", [wide])]
    path = tmp_path / "examples.jpg"
    figures.examples_mosaic(path, rows, per_band=3, label="score_contrast")
    with Image.open(path) as image:
        assert image.format == "JPEG"
        assert image.height > image.width / 3  # three rows of square tiles
    assert path.stat().st_size < 300_000


def test_pipeline_figure_names_all_taxa_and_the_threshold_content():
    assert figures.collect_steps() == ("Metadata stream", "UUID sample", "All taxa")
    assert figures.collect_steps(land=True)[-1] == "On land"
    assert figures.dataset_rows(with_images=True)[1] == "score ≥ t · + 500 px image"
    assert figures.dataset_rows(with_images=False)[1] == "score ≥ t · + 500 px URL"


def test_pipeline_figure_without_images(tmp_path):
    figures.pipeline(tmp_path / "hero.png", with_images=False)
    assert (tmp_path / "hero.png").read_bytes()[:8] == PNG


def test_diversity_map(tmp_path):
    cells = (np.array([0.0, 20.0, -40.0]), np.array([0.0, 10.0, 30.0]), np.array([0.2, 0.5, 0.9]))
    figures.diversity_map(tmp_path / "d.png", cells, per_cell=50, res=5)
    assert (tmp_path / "d.png").read_bytes()[:8] == PNG


def test_look_map(tmp_path):
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    figures.look_map(tmp_path / "l.png", np.array([0.0, 100.0]), np.array([0.0, -30.0]), rgb)
    assert (tmp_path / "l.png").read_bytes()[:8] == PNG


def test_diversity_map_of_one_value(tmp_path):
    cells = (np.array([0.0]), np.array([0.0]), np.array([0.4]))
    figures.diversity_map(tmp_path / "d.png", cells, per_cell=50, res=5)
    assert (tmp_path / "d.png").read_bytes()[:8] == PNG


def test_chip_text_fits_its_box(tmp_path):
    fig, ax = figures.plt.subplots(figsize=(16, 6.4))
    ax.set(xlim=(0, 160), ylim=(0, 64))
    figures._chip(ax, (0, 0, 14, 5), "Metadata stream", "blue")
    text = ax.texts[0]
    box = text.get_window_extent().transformed(ax.transData.inverted())
    assert box.width <= 12
    assert float(text.get_fontsize()) < 12
    figures.plt.close(fig)


def test_pipeline_figure_with_the_land_step(tmp_path):
    figures.pipeline(tmp_path / "hero.png", with_images=False, land=True)
    assert (tmp_path / "hero.png").read_bytes()[:8] == PNG


LAND = (np.array([0.0, 10.0, 30.0]), np.array([0.0, 5.0, 40.0]))  # lon, lat of land points


def test_maps_draw_the_land_without_value_in_light_grey(tmp_path):
    cells = (np.array([0.0]), np.array([0.0]), np.array([0.4]))
    figures.world_map(tmp_path / "m.png", ([0.0], [0.0], [3]), cap=10, res=5, land=LAND)
    figures.diversity_map(tmp_path / "d.png", cells, per_cell=50, res=5, land=LAND)
    rgb = np.array([[1.0, 0.0, 0.0]])
    figures.look_map(tmp_path / "l.png", cells[0], cells[1], rgb, land=LAND)
    for name in ("m", "d", "l"):
        assert (tmp_path / f"{name}.png").read_bytes()[:8] == PNG


def test_land_is_under_the_values_with_the_same_hexagons():
    fig, ax = figures.plt.subplots()
    figures._land(ax, LAND, 0.5)
    grey = ax.collections[0]
    assert grey.get_zorder() < 1  # under the values
    assert np.asarray(grey.get_offsets()).shape == (3, 2)
    grey_rgba = figures.mpl.colors.to_rgba(figures.LAND_GREY)
    assert np.allclose(grey.get_cmap()(0.5), grey_rgba)  # one colour for every hexagon
    figures._land(ax, None, 0.5)
    assert len(ax.collections) == 1
    figures.plt.close(fig)


def test_hexagon_size_sets_the_grid():
    assert figures._grid(2.0) == (180, 72)
    assert figures._grid(0.25) == (1440, 576)


def test_detailed_look_map_is_larger(tmp_path):
    lon, lat = np.array([0.0, 100.0]), np.array([0.0, -30.0])
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    figures.look_map(tmp_path / "a.png", lon, lat, rgb)
    figures.look_map(tmp_path / "b.png", lon, lat, rgb, size=0.25, dpi=300)
    with Image.open(tmp_path / "a.png") as a, Image.open(tmp_path / "b.png") as b:
        assert b.width > 1.9 * a.width
