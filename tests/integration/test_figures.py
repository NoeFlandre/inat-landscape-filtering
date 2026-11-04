"""Card figures: each function writes a PNG or a JPEG."""

from PIL import Image

from inat_landscape.adapters import figures

PNG = b"\x89PNG\r\n\x1a\n"


def test_pipeline_figure(tmp_path):
    figures.pipeline(tmp_path / "a" / "hero.png")
    assert (tmp_path / "a" / "hero.png").read_bytes()[:8] == PNG


def test_world_map_skips_missing_coordinates(tmp_path):
    pool = ([0.0, 10.0, None], [0.0, 5.0, None])
    cells = ([0.0, 20.0], [0.0, 10.0], [3, 10])
    figures.world_map(tmp_path / "map.png", pool, cells, cap=10, res=5)
    assert (tmp_path / "map.png").read_bytes()[:8] == PNG


def test_world_map_with_a_large_cap(tmp_path):
    cells = ([0.0], [0.0], [700])
    figures.world_map(tmp_path / "m.png", ([0.0], [0.0]), cells, cap=1000, res=5)
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
    assert figures.dataset_rows(with_images=True)[1] == "score ≥ t · + 500 px image"
    assert figures.dataset_rows(with_images=False)[1] == "score ≥ t · + 500 px URL"


def test_pipeline_figure_without_images(tmp_path):
    figures.pipeline(tmp_path / "hero.png", with_images=False)
    assert (tmp_path / "hero.png").read_bytes()[:8] == PNG
