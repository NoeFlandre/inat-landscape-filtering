# inat-landscape-filtering

Find iNaturalist photos that show a landscape. Keep their geolocation. Use them for land use and land cover work.

- Dataset: [NoeFlandre/inat-landscape-filtering](https://huggingface.co/datasets/NoeFlandre/inat-landscape-filtering)
- Documentation: `docs/` (build with `uv run mkdocs serve`)

## Quick start

```bash
uv sync
uv run inat-landscape --help
uv run inat-landscape ingest --fraction 0.001
uv run inat-landscape sample --n 10000
uv run inat-landscape embed
uv run inat-landscape score
uv run inat-landscape publish --thresholds 0.0005,0.001,0.002
```

## Quality

Run `make qa`. It runs ruff, ty, the tests, the architecture check, the CRAP check, mutation testing and a smoke test.
