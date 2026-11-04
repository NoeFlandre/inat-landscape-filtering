# ADR 0014: Maps of visual diversity and of what the photos show

Status: accepted (2026-11-04)

## Context

The card map gives the count of photos in each H3 cell. It does not tell what the photos show, or how varied they are. The embeddings can tell both.

## Decision

- The `diversity` stage reads all the embedding shards once, in parallel threads (`EmbeddingStore.each`, also used by `score`).
- Each cell uses the photos of `cell_rank` below 50 (`--per-cell`): a random sample of the same size in each cell (ADR 0011). Thus the dense cells do not get more weight. A cell with fewer than 20 embedded photos has no value (`--min-photos`).
- A cell keeps only the sum and the count of the unit embeddings of its photos. For n unit vectors with sum s, the mean cosine distance over the pairs is 1 - (|s|² - n) / (n (n - 1)). Thus there is no loop over pairs, and the shards can be read in any order.
- Diversity: that mean cosine distance. Colour: the 3 first principal components of the mean embeddings of the cells, scaled between their 2nd and 98th percentiles, as red, green and blue.
- `diversity.parquet` keeps, for each cell: `h3_cell`, `photos`, `diversity`, `pc1`, `pc2`, `pc3`. `publish` draws `assets/diversity.png` and `assets/look.png` from it, and the card explains the three maps in one paragraph.

## Consequences

- One more pass over the shards: about the time of `score` without its cache.
- The diversity depends on the model: it is what SigLIP2 sees as different, not a measure of the land cover.
- The colours of the PCA map have no fixed meaning. Only similar against different colours tells something.
