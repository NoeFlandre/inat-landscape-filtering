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
- The score cache (ADR 0012) does not know the new shards: the next `score` reads them again.
