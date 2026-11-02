---
license: other
license_name: per-image-creative-commons
pretty_name: iNaturalist landscape filtering
task_categories:
- image-classification
- zero-shot-image-classification
tags:
- inaturalist
- landscape
- land-cover
- land-use
- geospatial
- siglip
configs:
- config_name: all
  data_files: data/all/*.parquet
  default: true
- config_name: score_ge_0_10
  data_files: data/score_ge_0_10/*.parquet
- config_name: score_ge_0_50
  data_files: data/score_ge_0_50/*.parquet
- config_name: embeddings
  data_files: embeddings/siglip2-base-patch16-224/small/*.parquet
---

# iNaturalist landscape filtering

iNaturalist photos with a zero-shot score that tells if the photo shows a landscape.
Use the photos for land use and land cover work. Each photo keeps its geolocation.

Code: <https://github.com/NoeFlandre/inat-landscape-filtering>

## Source

- Metadata: iNaturalist Open Data, snapshot `https://x/inaturalist-open-data-20261027.tar.gz`.
- Photos: Plantae observations and observations with no identification.
- Sample: a deterministic sample of 10 scored photos (pilot run).

## Configs

| Config | Rows |
|---|---|
| `all` | 10 (metadata and scores, no image bytes) |
| `embeddings` | one `google/siglip2-base-patch16-224` embedding (float16) per photo |

Threshold configs keep the photos with `score_landscape >= threshold`. They include the
`medium` image (longest side 500 px).

| Config | Threshold | Rows |
|---|---|---|
| `score_ge_0_10` | 0.1 | 4 |
| `score_ge_0_50` | 0.5 | 1 |

## Scores

The model is `google/siglip2-base-patch16-224`. The pipeline scores the `small` image (longest side
240 px). Each score is `sigmoid(scale * cosine + bias)`, as SigLIP computes it. A score is
not a calibrated probability. Look at the photos before you select a threshold.

- `score_landscape`: mean text embedding of the prompts "a", "b".
- `score_landscape_1` ... `score_landscape_2`: one score for each
  of these prompts, in the same order.
- `score_closeup`: mean text embedding of the prompts "c", "d".

## Licences

Each photo has its own licence, in the `license` column. The `attribution` column gives
the attribution text that the licence requires. CC-BY-NC photos are not for commercial
use: remove them with a filter on `license` if you need commercial use.

| Licence | Photos |
|---|---|
| CC-BY | 6 |
| CC0 | 4 |

## Known limits

- Obscured coordinates: iNaturalist moves the location of some observations (threatened
  taxa, or a user choice) to a random point in a cell of about 0.2 degrees. The source
  data has no flag for this. Use `positional_accuracy` with care.
- The coordinates give the position of the observer or of the organism, not of the view.
- Zero-shot scores have no human validation yet.
