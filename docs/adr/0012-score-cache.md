# ADR 0012: Keep the scores of each shard

Status: accepted (2026-11-04)

## Context

With all taxa, `score` reads about 74 million embeddings from the hub. It read one shard at a time, and took almost 2 hours. If a later step of the job stopped, for example when the node had too little memory, the next job read all the shards again.

## Decision

- `score --workers N` downloads and scores N shards at the same time, in threads. The order of the shards does not change, so the result does not change.
- `score --cache FOLDER` writes the scores of each shard to `FOLDER/KEY/SHARD.parquet` when the shard is done. A later run reads these scores and does not download the shard.
- KEY is a hash of the model, of its scale and bias, and of the text directions. A change of a prompt gives a new KEY, so old scores are not used.
- `publish.oar` uses 8 workers and the cache `~/.inat-landscape/score-cache` on the NFS `/home` of the site.

## Consequences

- The work of `score` is kept shard by shard. A job that stops does not lose it.
- The cache of all taxa uses about 4 GB of the `/home` quota of the site.
- A job on a different site does not see the cache: use the same site again.
