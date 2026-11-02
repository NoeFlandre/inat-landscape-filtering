# Usage

## Install

1. Install [uv](https://docs.astral.sh/uv/).
2. Run `uv sync`.

To use Docker, build the image with `docker build -t inat-landscape .`.

## Run the pilot

Run the commands in this order.

```bash
uv run inat-landscape ingest --fraction 0.001
uv run inat-landscape sample --n 10000
uv run inat-landscape embed
uv run inat-landscape score
uv run inat-landscape publish --thresholds 0.05,0.1,0.2
```

- `ingest` downloads about 37 GB as a stream. It does not store the dump.
- `embed` and `publish` write to the Hugging Face repository. Set `HF_TOKEN` to a token with write access.
- To work without the Hugging Face repository, add `--offline` to `embed`, `score` and `publish`.

Use `uv run inat-landscape status` to see the row count of each step.
Use `uv run inat-landscape COMMAND --help` to see all options.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | The command is successful. |
| 1 | The command failed. The error message is on stderr. |
| 2 | The command line is not correct. |

## Resume

- `ingest` does not run again if the work directory has the same source and fraction. Use `--force` to run it again.
- `embed` reads the embedding shards on the Hugging Face repository first. It does not embed a photo that has an embedding.
- `embed` uploads local shards that are not on the Hugging Face repository. Thus shards from an `--offline` run go to the repository at the next online run.
- `embed` records failed downloads in `failures.parquet`. It does not try a photo with a permanent failure (for example HTTP 404) again. It tries a photo with a temporary failure again.
- `publish` keeps the downloaded 500 px photos in `cache/medium`.

## Work directory

```
work/
  ingest/candidates.parquet   candidate photos of the sample
  ingest/manifest.json        source and fraction of the candidates
  sample.parquet              the first N candidates
  failures.parquet            failed downloads
  scored.parquet              sample photos with scores
  cache/medium/               500 px photos
  hub/                        local copy of the Hugging Face dataset
```
