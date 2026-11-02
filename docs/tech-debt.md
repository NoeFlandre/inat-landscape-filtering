# Technical debt

Each item tells why it exists and how to remove it.

## Obscured coordinates have no flag

- **Why:** the Open Data dump has no column for obscured coordinates. iNaturalist moves some locations to a random point in a cell of about 0.2 degrees.
- **Effect:** some rows have a location error of up to about 20 km.
- **Cleanup:** get the `obscured` flag from the iNaturalist API or a GBIF export. Join it on `observation_uuid`.

## Ingest cannot resume part-way

- **Why:** a gzip stream cannot seek. A stop during `ingest` loses the pass (about 25 minutes).
- **Cleanup:** if the dump grows, split the read by tar member and save each filtered member.

## Embed runs download and model one after the other

- **Why:** for the pilot, the model takes most of the time. The download of a chunk takes a few seconds.
- **Cleanup:** at scale, download the next chunk while the model runs. Decode JPEG on the GPU.

## `done_ids` reads every shard

- **Why:** this is simple and correct for thousands of shards.
- **Cleanup:** keep an index file of done ids, or a shard manifest.

## The thresholds have no label-based check

- **Why:** the pilot uses only zero-shot scores. There are no human labels.
- **Cleanup:** label a test set. Measure precision and recall for each threshold.

## The real model has no test in CI

- **Why:** the real model is 375 MB. CI uses a tiny random SigLIP model.
- **Cleanup:** none needed. The smoke run on real data checks the real model.
