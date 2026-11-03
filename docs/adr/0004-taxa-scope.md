# ADR 0004: Taxa in scope

Status: accepted (2026-11-02), changed (2026-11-03)

## Context

Most iNaturalist photos are close views of organisms. Landscape photos are rare.
Some taxon groups almost never have landscape photos (for example insects).

The first run kept only Plantae and the observations with no taxon. The cap-10 dataset of that run showed that the scores and the thresholds separate landscape photos from close views well. Landscape photos are also in observations of animals and fungi (for example a habitat photo of a bird observation).

## Decision

- Keep all taxa.
- `group` is the kingdom of the taxon: `animalia`, `plantae`, `fungi`, `chromista`, `protozoa`, `bacteria`, `archaea` or `viruses`. Observations with no taxon, or with the root taxon Life (48460), are `unidentified`. A taxon with no known kingdom in its ancestry is `other`.

## Consequences

- The pool is larger: the cap per H3 cell (ADR 0007) sets the cost of the embeddings.
- A user can select a group with a filter on `group`.
- The rule is in `domain/taxa.py`. A change of scope is a change in one place.
