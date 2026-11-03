"""Taxon groups: every observation is kept, in the group of its kingdom (ADR 0004)."""

from enum import StrEnum

LIFE = 48460
PLANTAE = 47126
KINGDOMS = {
    1: "animalia",
    PLANTAE: "plantae",
    47170: "fungi",
    48222: "chromista",
    47686: "protozoa",
    67333: "bacteria",
    151817: "archaea",
    131236: "viruses",
}


class Group(StrEnum):
    """Observation groups: the kingdom, or no identification."""

    ANIMALIA = "animalia"
    PLANTAE = "plantae"
    FUNGI = "fungi"
    CHROMISTA = "chromista"
    PROTOZOA = "protozoa"
    BACTERIA = "bacteria"
    ARCHAEA = "archaea"
    VIRUSES = "viruses"
    UNIDENTIFIED = "unidentified"
    OTHER = "other"  # a taxon with no known kingdom in its ancestry


def classify(taxon_id: int | None, ancestry: str | None) -> Group:
    """Return the group of a taxon.

    `ancestry` is the iNaturalist ancestry string: ancestor ids joined by "/".
    """
    if taxon_id in (None, LIFE):
        return Group.UNIDENTIFIED
    # Any default that is not a taxon id is equivalent, so the line is not mutated.
    ancestors = (ancestry or "").split("/")  # pragma: no mutate
    lineage = {str(taxon_id), *ancestors}
    kingdom = next((name for i, name in KINGDOMS.items() if str(i) in lineage), "other")
    return Group(kingdom)
