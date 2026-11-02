# ADR 0007: Cap photos per H3 cell

Status: accepted (2026-11-02)

## Context

iNaturalist photos are dense in North America and Europe. The dataset must cover the world as widely as possible. Diego Marcos asked to cap the photos per area. The biases of interest are at a large scale.

## Decision

- Give each photo the [H3](https://h3geo.org) cell of its location at resolution 5 (about 250 km²).
- Drop photos with no location. A photo with no cell cannot have a cap.
- **Before embedding:** keep at most 1,000 photos per cell, the first in sample order. The order is random, so the kept photos are a random sample of the cell.
- **At publish:** each threshold config keeps at most 100 photos per cell, the best scored.
- `ingest` reads the dump in three cached passes: taxa, then a per-cell index (count and cap), then the full metadata of the kept photos only. Each pass keeps its output, so a run does not read a pass again.
- `measure` runs the first two passes and prints the photos per cell, and the photos kept under each cap.

## Consequences

- Dense cells cost at most 1,000 embeddings. Sparse cells keep all their photos.
- The index keeps observation keys and cells as numpy arrays. With about 100 million eligible observations, this needs about 2 GB of memory.
- A larger cap keeps a superset, because the order does not change.
- The top 100 by score of a set above a threshold are also in the top 100 above a lower threshold. Thus the threshold configs stay nested.
