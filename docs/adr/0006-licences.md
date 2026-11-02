# ADR 0006: Licences

Status: accepted (2026-11-02)

## Context

The Open Data bucket has photos with CC0, CC-BY and CC-BY-NC licences.
CC-BY and CC-BY-NC require attribution. CC-BY-NC does not allow commercial use.

## Decision

- Keep the three licences.
- Store the licence and the attribution text on each row. The text follows the iNaturalist Open Data README.
- Set `license: other` on the dataset card. Explain the per-photo licences on the card.

## Consequences

- The dataset is as large as possible.
- A user who needs commercial use must remove the CC-BY-NC rows with a filter on `license`.
