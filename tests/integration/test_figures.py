"""Card figures: each function writes a PNG."""

from inat_landscape.adapters import figures

PNG = b"\x89PNG\r\n\x1a\n"


def test_pipeline_figure(tmp_path):
    figures.pipeline(tmp_path / "a" / "hero.png")
    assert (tmp_path / "a" / "hero.png").read_bytes()[:8] == PNG


def test_world_map_skips_missing_coordinates(tmp_path):
    pool = ([0.0, 10.0, None], [0.0, 5.0, None])
    figures.world_map(tmp_path / "map.png", pool, ([0.0], [0.0]), ([None], [None]), "kept")
    assert (tmp_path / "map.png").read_bytes()[:8] == PNG


def test_score_histogram_handles_zero_and_one(tmp_path):
    figures.score_histogram(tmp_path / "s.png", [0.0, 1e-6, 0.002, 1.0], [0.001])
    assert (tmp_path / "s.png").read_bytes()[:8] == PNG
