"""Taxon groups that the pipeline keeps."""

from enum import StrEnum

LIFE = 48460
PLANTAE = 47126


class Group(StrEnum):
    """Kept observation groups."""

    PLANTAE = "plantae"
    UNIDENTIFIED = "unidentified"


def classify(taxon_id: int | None, ancestry: str | None) -> Group | None:
    """Return the kept group of a taxon, or None if the pipeline drops it.

    `ancestry` is the iNaturalist ancestry string: ancestor ids joined by "/".
    """
    if taxon_id in (None, LIFE):
        return Group.UNIDENTIFIED
    # Any default that is not a taxon id is equivalent, so the line is not mutated.
    ancestors = (ancestry or "").split("/")  # pragma: no mutate
    lineage = {str(taxon_id), *ancestors}
    return Group.PLANTAE if str(PLANTAE) in lineage else None
