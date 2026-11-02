# ADR 0005: Storage and resume

Status: accepted (2026-11-02)

## Context

The pipeline must not calculate an embedding two times. The work machine can stop at any time. A new machine must continue the work.

## Decision

- The embedding shards on the Hugging Face dataset are the only record of done work.
- `embed` downloads the remote shards and uploads the local-only shards first. Then it embeds only the photos with no embedding.
- `embed` writes one shard for each chunk of 1000 photos. It uploads the shard at once.
- The name of a shard is a hash of its photo ids. A shard name does not change between runs.
- All local writes are atomic (a temporary file, then a rename).
- The local `work/hub/` folder has the same layout as the Hugging Face repository. `publish` uploads it. Files with no change do not upload again.

## Consequences

- A stop during a chunk loses only that chunk.
- `done_ids` reads all shards. At a very large scale, this needs an index.
