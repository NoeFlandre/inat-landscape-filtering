# ADR 0011: Embed in uniform order over the H3 cells

Status: accepted (2026-11-03)

## Context

The run with all taxa and a cap of 1,000 photos per H3 cell has tens of millions of photos. It takes many hours on the free GPUs, and it can stop before the end. The dataset must stay as uniform over the cells as possible at each moment: if the work went one area at a time, a stop would leave some areas full and others empty.

## Decision

- The rank of a photo is its place in the sample order (ADR 0007) of its H3 cell.
- The sample is in uniform order: rank first, then sample order. Thus the work takes one photo of each cell, then a second photo of each cell, and so on up to the cap.
- `embed` works in sample order, so it follows the uniform order. The worker split by photo id (ADR 0009) keeps the order in each worker.
- The sample has a `cell_rank` column.

## Consequences

- At each moment, the done photos have the same count in each cell, with one more in some cells, except for the cells that have fewer photos than that count. The cap reached so far grows from 1 to 1,000.
- A sample of the first n photos (`sample --n`) is also uniform, and a larger n gives a superset.
- The embeddings of the cap-10 run are in the new pool: they are done work, wherever their rank is.
