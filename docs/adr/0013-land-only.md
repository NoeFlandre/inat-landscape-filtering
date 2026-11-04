# ADR 0013: Keep only the cells that touch land

Status: accepted (2026-11-04)

## Context

The map of the card showed H3 cells at sea: observations of marine life, from boats or by divers. The dataset is for land use and land cover, so these photos are not useful.

## Decision

- The `land` stage downloads the Natural Earth 1:10m land and minor islands polygons (release v5.1.2, public domain). It writes `land.parquet`: the sample H3 cells that touch land (H3 "overlap" mode). `land.json` gives the count of cells and photos at sea.
- `read_sample` leaves out the photos of the other cells. Thus `score`, `publish` and the map use only the cells on land. The sample and the candidates do not change: to undo, delete `land.parquet`.
- `embed --prune` removes the photos at sea from the embedding shards on the hub. A shard with removed rows is replaced by a shard of its kept rows, with the same worker tag, 50 shards for each commit. The done-ids snapshot is deleted.

## Consequences

- A cell that touches land keeps all its photos, also the photos on the water near the coast. The card states it.
- A photo is not tested alone: the coast of Natural Earth 1:10m is not precise enough for a photo on a beach.
- `embed --prune --cache FOLDER` gives the cached scores of each pruned shard (ADR 0012) to its new shard. Thus the next `score` does not download the shards again.
- The prune sorts the kept ids once and finds the rows of each shard by binary search (`domain/members.py`). `np.isin` sorted the 73 million kept ids again for each shard: one shard took minutes, and 16 threads used 56 GB of memory.
