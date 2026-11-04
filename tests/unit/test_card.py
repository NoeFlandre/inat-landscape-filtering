"""Dataset card: generated from one place, from the run results."""

from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.card import (
    Breakdown,
    CardInfo,
    Subset,
    breakdown,
    fields,
    render_card,
    size_category,
    snapshot_label,
)

GOLDEN = Path(__file__).parent.parent / "golden" / "card.md"


def _info():
    return CardInfo(
        repo_id="u/r",
        snapshot="https://x/metadata/inaturalist-open-data-20261027.tar.gz",
        model_id="google/siglip2-base-patch16-224",
        landscape_prompts=("a", "b"),
        closeup_prompts=("c", "d"),
        negative_prompts=("e", "f"),
        fraction=0.001,
        candidates=1234,
        total=10,
        subsets=(
            Subset("score_ge_0_10", 0.1, 4),
            Subset("score_ge_0_50", 0.5, 2),
            Subset("score_ge_0_90", 0.9, 1),
        ),
        groups=(Breakdown("plantae", 9, (4, 2, 1)), Breakdown("unidentified", 1, (0, 0, 0))),
        licenses=(Breakdown("CC-BY", 6, (3, 1, 1)), Breakdown("CC0", 4, (1, 1, 0))),
        quantiles=(("p50", 0.00001), ("max", 0.0135)),
        years=("2001-01-01", "2026-10-01"),
        embedding_dir="embeddings/siglip2-base-patch16-224/small",
        embedding_dim=768,
        embedding_rows=12,
        no_location=2,
        score="score_landscape",
        cap_per_cell=100,
        embed_cap=1000,
        h3_res=5,
    )


def test_card_matches_golden_file():
    assert render_card(_info()) == GOLDEN.read_text()


def test_card_yaml_lists_every_config_with_a_train_split():
    card = render_card(_info())
    assert card.startswith("---\nlicense: [cc0-1.0, cc-by-4.0, cc-by-nc-4.0]\n")
    for name in ("all", "score_ge_0_10", "score_ge_0_50", "score_ge_0_90", "embeddings"):
        assert f"- config_name: {name}\n  data_files:\n  - split: train\n" in card


@pytest.mark.parametrize(
    ("n", "label"),
    [
        (0, "n<1K"),
        (999, "n<1K"),
        (1000, "1K<n<10K"),
        (10_000, "10K<n<100K"),
        (5_000_000, "1M<n<10M"),
        (10_000_000, "n>10M"),
    ],
)
def test_size_category(n, label):
    assert size_category(n) == label


def test_snapshot_label():
    assert snapshot_label("https://x/inaturalist-open-data-20261027.tar.gz") == "2026-10-27"
    assert snapshot_label("https://x/latest.tar.gz") == "`latest.tar.gz`"
    assert snapshot_label("dump") == "`dump`"


def test_breakdown_counts_and_orders_largest_first():
    scores = [0.9, 0.1, 0.2, 0.5]
    result = breakdown(["b", "a", "b", "c"], [[s >= t for s in scores] for t in (0.15, 0.5)])
    assert result == (
        Breakdown("b", 2, (2, 1)),
        Breakdown("a", 1, (0, 0)),
        Breakdown("c", 1, (1, 1)),
    )


cats = st.lists(st.tuples(st.sampled_from("xyz"), st.floats(0, 1)), max_size=30)


@given(cats, st.lists(st.floats(0, 1), max_size=3))
def test_breakdown_invariants(rows, thresholds):
    names = [c for c, _ in rows]
    result = breakdown(names, [[s >= t for _, s in rows] for t in thresholds])
    assert sum(b.rows for b in result) == len(rows)
    for b in result:
        assert all(k <= b.rows for k in b.kept)
        assert len(b.kept) == len(thresholds)


def test_fields_name_each_prompt_column():
    names = [n for f in fields(3) for n in f.names]
    assert [n for n in names if n.startswith("score_landscape_")] == [
        "score_landscape_1",
        "score_landscape_2",
        "score_landscape_3",
    ]
    assert len(names) == len(set(names))


def test_breakdown_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match=r"^each kept mask must have one value per photo$"):
        breakdown(["a", "b"], [[True]])


def test_missing_locations_line_only_when_some_are_missing():
    assert "Missing locations" in render_card(_info())
    assert "Missing locations" not in render_card(replace(_info(), no_location=0))


def test_no_missing_location_line_leaves_the_list_intact():
    card = render_card(replace(_info(), no_location=0))
    assert "flag for this.\n* **What the location means.**" in card


def test_unknown_resolution_has_a_fallback_area():
    assert "(resolution 9, see H3)" in render_card(replace(_info(), h3_res=9))


def test_progress_line():
    assert "The run is in progress: 10 of the 1,234 sample photos are scored." in render_card(
        _info()
    )
    done = replace(_info(), candidates=10)
    assert "All 10 sample photos are scored." in render_card(done)


def test_a_card_without_images_points_to_the_image_urls():
    card = render_card(replace(_info(), with_images=False))
    assert "and come with the URL of their 500 px image (`url_medium`)" in card
    assert "with the 500 px image" not in card
    assert "added at the end of the run" not in card
    assert "# strictest threshold\n" in card
    assert "best 100 per cell, with the URL of the 500 px image |" in card


def test_a_card_with_images_says_so():
    card = render_card(_info())
    assert "and come with their 500 px image" in card
    assert "# strictest threshold, with images" in card


def test_card_acknowledges_grid5000():
    card = render_card(_info())
    section = card.split("## Acknowledgements\n", 1)[1]
    assert "Grid'5000 testbed" in section
    assert "https://www.grid5000.fr" in section


def test_card_states_the_layer_cap_while_the_dataset_grows():
    info = _info()
    card = render_card(replace(info, layer_cap=50, embed_cap=1000, total=info.candidates - 1))
    assert "each H3 cell has its first 50 photos" in card
    assert "on the way to 1,000 per cell" in card


def test_card_with_all_layers_does_not_talk_of_layers():
    info = _info()
    card = render_card(replace(info, layer_cap=3, embed_cap=1000, total=info.candidates))
    assert "grows by layers" not in card


def test_a_finished_run_says_it_is_complete_with_its_failed_downloads():
    info = _info()
    done = replace(info, layer_cap=1000, embed_cap=1000, total=info.candidates - 2)
    card = render_card(done)
    assert "grows by layers" not in card
    assert (
        f"The run is complete: {info.candidates - 2:,} of the {info.candidates:,} sample " in card
    )
    assert "photos have an embedding (2 downloads failed)." in card


def test_a_card_without_images_documents_no_image_column():
    card = render_card(replace(_info(), with_images=False))
    assert "| `image` |" not in card
    assert "keep the URL of their image](assets/hero.png)" in card


def test_a_card_with_images_documents_the_image_column():
    card = render_card(_info())
    assert "| `image` | image | The 500 px photo (threshold configs only) |" in card
    assert "keep their image](assets/hero.png)" in card


def test_fields_without_images_have_no_image_column():
    assert "image" not in {n for f in fields(3, with_images=False) for n in f.names}
    assert "image" in {n for f in fields(3) for n in f.names}


def test_card_counts_the_photos_with_two_embedding_rows():
    card = render_card(replace(_info(), embedding_rows=12, embedding_photos=10))
    assert "| `embeddings` | 12 |" in card
    assert (
        "one row per photo, but 2 photos have two rows "
        "(a photo of two observations, embedded twice; keep one): `photo_id`"
    ) in card
    one = render_card(replace(_info(), embedding_rows=12, embedding_photos=11))
    assert "but 1 photos have two rows" in one


def test_card_without_copies_has_one_embedding_row_per_photo():
    card = render_card(replace(_info(), embedding_rows=12, embedding_photos=12))
    assert "The `embeddings` config has one row per photo:" in card
    assert "two rows" not in card
