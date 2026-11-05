"""Card figures: each function writes a PNG or a JPEG."""

import numpy as np
from PIL import Image

from inat_landscape.adapters import figures
from inat_landscape.adapters.geo import h3_cells
from inat_landscape.domain.cells import MAP_RES
from inat_landscape.domain.projection import equal_earth

PNG = b"\x89PNG\r\n\x1a\n"


def test_pipeline_figure(tmp_path):
    figures.pipeline(tmp_path / "a" / "hero.png")
    assert (tmp_path / "a" / "hero.png").read_bytes()[:8] == PNG


def _cells(lat, lon):
    return h3_cells(np.array(lat, float), np.array(lon, float), MAP_RES)


A, B, C = _cells([0.0, 10.0, 30.0], [0.0, 20.0, -40.0])


def test_world_map(tmp_path):
    cells = (np.array([A, B]), np.array([3.0, 10.0]))
    figures.world_map(tmp_path / "map.png", cells, cap=10, res=5)
    assert (tmp_path / "map.png").read_bytes()[:8] == PNG


def test_world_map_with_a_large_cap(tmp_path):
    cells = (np.array([A]), np.array([700.0]))
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
    cells = (np.array([A, B, C]), np.array([0.2, 0.5, 0.9]))
    figures.diversity_map(tmp_path / "d.png", cells, per_cell=50, res=5)
    assert (tmp_path / "d.png").read_bytes()[:8] == PNG


def test_look_map(tmp_path):
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    figures.look_map(tmp_path / "l.png", (np.array([A, C]), rgb))
    assert (tmp_path / "l.png").read_bytes()[:8] == PNG


def test_diversity_map_of_one_value(tmp_path):
    cells = (np.array([A]), np.array([0.4]))
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


LAND = np.array([A, B, C])  # land cells


def test_maps_draw_the_land_without_value_in_light_grey(tmp_path):
    cells = (np.array([A]), np.array([0.4]))
    counts = (np.array([A]), np.array([3.0]))
    figures.world_map(tmp_path / "m.png", counts, cap=10, res=5, land=LAND)
    figures.diversity_map(tmp_path / "d.png", cells, per_cell=50, res=5, land=LAND)
    figures.look_map(tmp_path / "l.png", (np.array([A]), np.array([[1.0, 0.0, 0.0]])), land=LAND)
    for name in ("m", "d", "l"):
        assert (tmp_path / f"{name}.png").read_bytes()[:8] == PNG


def test_land_with_no_value_is_grey_under_the_values():
    fig, ax = figures.plt.subplots()
    shapes = figures._draw(ax, (np.array([A]), np.array([0.4])), LAND)
    grey = ax.collections[0]
    assert grey.get_zorder() < shapes.get_zorder()
    assert len(grey.get_paths()) == 2  # B and C: A has a value
    assert np.allclose(grey.get_facecolor(), figures.mpl.colors.to_rgba(figures.LAND_GREY))
    assert np.asarray(shapes.get_array()).tolist() == [0.4]
    figures._draw(ax, (np.array([A]), np.array([0.4])), None)
    assert len(ax.collections) == 3
    figures.plt.close(fig)


def test_cells_are_their_hexagons_on_the_equal_earth_projection():
    fig, ax = figures.plt.subplots()
    shapes = figures._draw(ax, (np.array([B]), np.array([[0.0, 0.0, 1.0]])), None)
    corners = np.asarray(shapes.get_paths()[0].vertices)[:6]
    x, y = equal_earth([20.0], [10.0])
    assert np.allclose(corners.mean(axis=0), [x[0], y[0]], atol=0.01)
    assert np.allclose(shapes.get_facecolor()[0], [0.0, 0.0, 1.0, 1.0])
    figures.plt.close(fig)


def test_a_cell_over_the_180th_meridian_is_drawn_on_both_sides():
    (east,) = _cells([0.0], [180.0])
    fig, ax = figures.plt.subplots()
    shapes = figures._draw(ax, (np.array([east, A]), np.array([0.2, 0.7])), None)
    assert np.asarray(shapes.get_array()).tolist() == [0.2, 0.2, 0.7]
    sides = [np.asarray(p.vertices)[:, 0].mean() for p in shapes.get_paths()[:2]]
    assert sorted(np.sign(sides)) == [-1, 1]
    figures.plt.close(fig)


def test_outline_with_continuous_longitudes():
    ring = np.array([[179.0, 0.0], [-179.0, 1.0], [-179.0, -1.0]])
    left, right = figures._sides(ring)
    assert left[:, 0].tolist() == [179.0, 181.0, 181.0]
    assert right[:, 0].tolist() == [-181.0, -179.0, -179.0]
    west = np.array([[-179.0, 0.0], [179.0, 1.0]])
    assert [s[:, 0].tolist() for s in figures._sides(west)] == [[-179.0, -181.0], [181.0, 179.0]]
    inside = np.array([[10.0, 0.0], [11.0, 1.0]])
    assert [s.tolist() for s in figures._sides(inside)] == [inside.tolist()]


def test_maps_are_detailed_and_end_at_the_frame(tmp_path):
    figures.look_map(tmp_path / "l.png", (np.array([A]), np.array([[1.0, 0.0, 0.0]])))
    with Image.open(tmp_path / "l.png") as image:
        assert image.width > 4000
    edge = figures.equal_earth([180.0], [0.0])[0][0]
    assert np.isclose(figures._FRAME[:, 0].max(), edge)
    assert figures._FRAME[:, 1].min() == figures.equal_earth([0.0], [-60.0])[1][0]
