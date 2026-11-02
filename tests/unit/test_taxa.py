"""Taxa filter: keep Plantae and unidentified observations only."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.taxa import LIFE, PLANTAE, Group, classify


def test_missing_taxon_is_unidentified():
    assert classify(None, None) is Group.UNIDENTIFIED


def test_life_root_is_unidentified():
    assert classify(LIFE, None) is Group.UNIDENTIFIED


def test_plantae_itself_is_plantae():
    assert classify(PLANTAE, str(LIFE)) is Group.PLANTAE


def test_plant_descendant_is_plantae():
    assert classify(52775, "48460/47126/211194/47125") is Group.PLANTAE


@pytest.mark.parametrize(
    ("taxon_id", "ancestry"),
    [
        (3, "48460/1/2355/3"),  # bird
        (47170, "48460/47170"),  # fungi
        (999, "48460/1/471260"),  # id that only contains 47126 as a substring
        (999, None),  # unknown ancestry
    ],
)
def test_other_taxa_are_excluded(taxon_id, ancestry):
    assert classify(taxon_id, ancestry) is None


ancestries = st.lists(st.integers(1, 10**6), min_size=1, max_size=8).map(
    lambda ids: "/".join(map(str, ids))
)


@given(st.integers(1, 10**6), ancestries)
def test_plantae_iff_47126_is_an_ancestor_or_self(taxon_id, ancestry):
    expected = taxon_id == PLANTAE or str(PLANTAE) in ancestry.split("/")
    assert (classify(taxon_id, ancestry) is Group.PLANTAE) == (expected and taxon_id != LIFE)
