# ADR 0001: Data source and snapshot

Status: accepted (2026-11-02)

## Context

We need iNaturalist photos with location, taxon and licence, worldwide, at scale.
The iNaturalist API has rate limits.
The iNaturalist Open Data bucket on AWS has the metadata of all openly licensed photos. It also has the photos at several sizes.

## Decision

- Use the iNaturalist Open Data bucket.
- Pin the snapshot `inaturalist-open-data-20261027.tar.gz` (36.7 GB compressed).
- Read the dump as a stream in one pass. Do not store it. The work disk is too small for the dump.
- Keep a sample of observations while the stream is read. An observation is in the sample when the first 8 hex digits of its UUID are less than `fraction × 16^8`.

## Consequences

- The sample is deterministic. A larger fraction gives a superset.
- The sample selection is random, because UUID v4 values are random.
- The tar order is `observations.csv`, then the other files. The reader filters photos, taxa and observers with the kept observations. It stops with an error if the order is different.
- A new fraction needs a new pass over the dump (about 25 minutes).
