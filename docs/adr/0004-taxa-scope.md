# ADR 0004: Taxa in scope

Status: accepted (2026-11-02)

## Context

Most iNaturalist photos are close views of organisms. Landscape photos are rare.
Some taxon groups almost never have landscape photos (for example insects).

## Decision

For the first run, keep only:

- Plantae (taxon 47126 and its descendants). Habitat and vegetation photos are frequent in this group.
- Observations with no taxon, or with the root taxon Life (48460). Users often upload a view with no organism to identify.

## Consequences

- The scope is small and has a high share of landscape photos.
- Birds and mammals are the first candidates for the next extension.
- The rule is in `domain/taxa.py`. A change of scope is a change in one place.
