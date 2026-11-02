# Technical debt

Each item tells why it exists and how to remove it.

## Obscured coordinates have no flag

- **Why:** the Open Data dump has no column for obscured coordinates. iNaturalist moves some locations to a random point in a cell of about 0.2 degrees.
- **Effect:** some rows have a location error of up to about 20 km.
- **Cleanup:** get the `obscured` flag from the iNaturalist API or a GBIF export. Join it on `observation_uuid`.

## Ingest cannot resume inside a pass

- **Why:** a gzip stream cannot seek. A stop during a pass loses that pass (about 20 to 30 minutes). Completed passes are kept.
- **Cleanup:** save the filtered output of each tar member as soon as it is read.

## The details pass and publish keep their tables in memory

- **Why:** this is simple, and it is correct up to a few million photos.
- **Effect:** with tens of millions of capped photos, the details join and the medium images in `publish` can exceed the memory.
- **Cleanup:** write the details and the configs in parts, and stream the images.

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

## The card states the pilot's visual check as fixed text

- **Why:** the visual check of each score band is a human judgement, not a value the code can compute.
- **Effect:** a run with other thresholds shows a sentence about the pilot thresholds.
- **Cleanup:** pass the review notes to `publish` as a file, or label a test set and compute precision per threshold.
