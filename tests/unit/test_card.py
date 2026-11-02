"""Dataset card: generated from one place, lists configs and licences."""

from pathlib import Path

from inat_landscape.domain.card import CardInfo, Subset, render_card


def _info():
    return CardInfo(
        snapshot="https://x/inaturalist-open-data-20261027.tar.gz",
        model_id="google/siglip2-base-patch16-224",
        landscape_prompts=("a", "b"),
        closeup_prompts=("c", "d"),
        total=10,
        subsets=(Subset("score_ge_0_10", 0.1, 4), Subset("score_ge_0_50", 0.5, 1)),
        licenses={"CC-BY": 6, "CC0": 4},
        embedding_dir="embeddings/siglip2-base-patch16-224/small",
    )


def test_card_has_yaml_configs_for_each_subset_and_embeddings():
    card = render_card(_info())
    assert card.startswith("---\n")
    assert "- config_name: all\n  data_files: data/all/*.parquet\n  default: true" in card
    assert "- config_name: score_ge_0_10\n  data_files: data/score_ge_0_10/*.parquet" in card
    emb = "embeddings/siglip2-base-patch16-224/small/*.parquet"
    assert f"- config_name: embeddings\n  data_files: {emb}" in card


def test_card_lists_counts_prompts_and_licences():
    card = render_card(_info())
    assert "| `score_ge_0_50` | 0.5 | 1 |" in card
    assert "| CC-BY | 6 |" in card
    assert '"a"' in card
    assert "20261027" in card
    assert "license: other" in card


def test_card_yaml_block_is_well_formed_and_followed_by_body():
    card = render_card(_info())
    assert "\nconfigs:\n- config_name: all\n" in card
    assert "\n---\n\n# iNaturalist landscape filtering\n" in card


def test_card_matches_golden_file():
    golden = Path(__file__).parent.parent / "golden" / "card.md"
    assert render_card(_info()) == golden.read_text()
