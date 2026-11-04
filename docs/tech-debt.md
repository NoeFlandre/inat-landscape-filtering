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
- **Effect:** with tens of millions of capped photos, the details join and the `all` config in `publish` can exceed the memory. The images are already in parts (ADR 0010).
- **Cleanup:** write the details and the `all` config in parts.

## Photos decode on the CPU

- **Why:** a background thread downloads and decodes the next photos while the model runs, and a thread pool resizes them. This keeps one GPU busy.
- **Cleanup:** if the CPU of a GPU node becomes the limit, decode the JPEG files on the GPU.

## The done-ids snapshot is made by hand

- **Why:** `embed --snapshot` writes one index file of the done ids (`index/done-ids.parquet`). A job reads the snapshot and only the shards that came after it. This keeps the Hugging Face reads low. A job that runs `--snapshot` must be started by hand (or by a loop on the frontend), for example once each hour.
- **Effect:** if nobody makes a new snapshot, each job reads more shards, and Hugging Face can refuse the reads (HTTP 429).
- **Cleanup:** let the last job of each hour write the snapshot.

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
