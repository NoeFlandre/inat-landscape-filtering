# Usage

## Install

1. Install [uv](https://docs.astral.sh/uv/).
2. Run `uv sync` (CPU). On NVIDIA GPUs, run `uv sync --no-default-groups --group dev --group cu126`.

To use Docker, build the image with `docker build -t inat-landscape .`.

## Run the pilot

Run the commands in this order.

```bash
uv run inat-landscape measure                       # photos per H3 cell, metadata only
uv run inat-landscape ingest                        # full pool, 1,000 photos per cell at most
uv run inat-landscape sample                        # all candidates (or --n N)
uv run inat-landscape embed
uv run inat-landscape score
uv run inat-landscape publish --thresholds 0.0005,0.001,0.002 --cap-per-cell 100
```

- `measure` and `ingest` read the dump (about 37 GB) as a stream. They do not store it. Options: `--fraction` (default 1), `--res` (H3 resolution, default 5), `--cap` (photos per cell, default 1000).
- `publish --score score_contrast` uses the negative-prompt score for the configs.
- `embed` and `publish` write to the Hugging Face repository. Set `HF_TOKEN` to a token with write access.
- To work without the Hugging Face repository, add `--offline` to `embed`, `score` and `publish`.

- `embed --worker I/N` embeds one share of the photos, for parallel jobs. See [Grid'5000](grid5000.md).
- `state push` and `state pull` move the work files through a private Hugging Face repository.

Use `uv run inat-landscape status` to see the row count of each step.
Use `uv run inat-landscape COMMAND --help` to see all options.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | The command is successful. |
| 1 | The command failed. The error message is on stderr. |
| 2 | The command line is not correct. |

## Resume

- `ingest` reads the dump in three passes: taxa, per-cell index, details. Each pass keeps its output and does not run again with the same parameters. Use `--force` to run the index and the details again.
- `embed` reads the embedding shards on the Hugging Face repository first. It does not embed a photo that has an embedding.
- `embed` uploads local shards that are not on the Hugging Face repository. Thus shards from an `--offline` run go to the repository at the next online run.
- `embed` records failed downloads in `failures.parquet`. It does not try a photo with a permanent failure (for example HTTP 404) again. It tries a photo with a temporary failure again.
- `publish` downloads the 500 px photos in parts of 2,000 and uploads each part. It keeps no photo on the disk. `publish --no-images` writes the metadata only. Use it for a quick refresh during a run.

## Work directory

```
work/
  ingest/taxa.parquet         taxa table of the dump
  ingest/cell_counts.parquet  eligible photos per H3 cell
  ingest/selected.parquet     photos kept by the cap per cell
  ingest/candidates.parquet   full metadata of the kept photos
  sample.parquet              the first N candidates
  failures.parquet            failed downloads
  scored.parquet              sample photos with scores
  publish-parts/              the image part that is uploading
  hub/                        local copy of the Hugging Face dataset
```
