# ADR 0014: Maps of visual diversity and of what the photos show

Status: accepted (2026-11-04)

## Context

The card map gives the count of photos in each H3 cell. It does not tell what the photos show, or how varied they are. The embeddings can tell both.

## Decision

- The `diversity` stage reads all the embedding shards once, in parallel threads (`EmbeddingStore.each`, also used by `score`).
- Each cell uses the photos of `cell_rank` below 50 (`--per-cell`): a random sample of the same size in each cell (ADR 0011). Thus the dense cells do not get more weight. A cell with fewer than 20 embedded photos has no value (`--min-photos`).
- A cell keeps only the sum and the count of the unit embeddings of its photos. For n unit vectors with sum s, the mean cosine distance over the pairs is 1 - (|s|² - n) / (n (n - 1)). Thus there is no loop over pairs, and the shards can be read in any order.
- Diversity: that mean cosine distance. Colour: the 3 first principal components of the mean embeddings of the cells, scaled between their 2nd and 98th percentiles, as red, green and blue.
- The colour map uses the same 2° hexagons as the other maps: each hexagon has the mean red, green and blue of its cells.
- A fourth map shows the same colours in detail: hexagons of 0.25° in place of 2°, at 400 dpi. It uses the same principal components: there is no second computation.
- On each map, light grey is land with no value. White is sea. The grey comes from `land-grid.parquet`: all the land cells of the `land` stage (ADR 0013), not only the sample cells. The maps use the centres of their children (resolution 6), so that each small hexagon of the detailed map on land has a point.
- `publish --maps-only` draws the maps again and uploads only them: the card, the configs and the scores do not change. It draws the map of photos per cell too if the `all` config is in the work directory.
- `diversity.parquet` keeps, for each cell: `h3_cell`, `photos`, `diversity`, `pc1`, `pc2`, `pc3`. `publish` draws `assets/diversity.png`, `assets/look.png` and `assets/look-detail.png` from it, and the card explains each map.

## Consequences

- One more pass over the shards: about the time of `score` without its cache.
- The diversity depends on the model: it is what SigLIP2 sees as different, not a measure of the land cover.
- The colours of the PCA map have no fixed meaning. Only similar against different colours tells something.
