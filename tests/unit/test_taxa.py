"""Taxon groups: every observation is kept, in the group of its kingdom."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inat_landscape.domain.taxa import KINGDOMS, LIFE, PLANTAE, Group, classify


def test_missing_taxon_is_unidentified():
    assert classify(None, None) is Group.UNIDENTIFIED


def test_life_root_is_unidentified():
    assert classify(LIFE, None) is Group.UNIDENTIFIED


def test_plantae_itself_is_plantae():
    assert classify(PLANTAE, str(LIFE)) is Group.PLANTAE


def test_plant_descendant_is_plantae():
    assert classify(52775, "48460/47126/211194/47125") is Group.PLANTAE


@pytest.mark.parametrize(
    ("taxon_id", "ancestry", "group"),
    [
        (3, "48460/1/2355/3", Group.ANIMALIA),  # bird
        (47170, "48460", Group.FUNGI),
        (48222, "48460", Group.CHROMISTA),
        (47686, "48460", Group.PROTOZOA),
        (67333, "48460", Group.BACTERIA),
        (151817, "48460", Group.ARCHAEA),
        (131236, "48460", Group.VIRUSES),
    ],
)
def test_other_kingdoms_are_kept_in_their_group(taxon_id, ancestry, group):
    assert classify(taxon_id, ancestry) is group


@pytest.mark.parametrize(
    ("taxon_id", "ancestry"),
    [
        (999, "48460/471260/11"),  # ids that only contain 1 or 47126 as a substring
        (999, None),  # unknown ancestry
    ],
)
def test_a_taxon_of_no_known_kingdom_is_other(taxon_id, ancestry):
    assert classify(taxon_id, ancestry) is Group.OTHER


def test_every_kingdom_has_a_group():
    assert {Group(name) for name in KINGDOMS.values()} <= set(Group)


ancestries = st.lists(st.integers(1, 10**6), min_size=1, max_size=8).map(
    lambda ids: "/".join(map(str, ids))
)


@given(st.integers(1, 10**6), ancestries)
def test_every_observation_has_a_group(taxon_id, ancestry):
    assert classify(taxon_id, ancestry) in set(Group)


@given(st.integers(1, 10**6), ancestries)
def test_the_group_is_the_first_kingdom_from_the_root(taxon_id, ancestry):
    lineage = [int(t) for t in ancestry.split("/")] + [taxon_id]
    first = next((KINGDOMS[t] for t in lineage if t in KINGDOMS), "other")
    expected = Group.UNIDENTIFIED if taxon_id == LIFE else Group(first)
    assert classify(taxon_id, ancestry) is expected


def test_a_kingdom_above_another_id_wins():
    assert classify(PLANTAE, "48460/1") is Group.ANIMALIA
