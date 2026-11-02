---
license: [cc0-1.0, cc-by-4.0, cc-by-nc-4.0]
pretty_name: iNaturalist Landscape Filtering
task_categories: [image-classification, zero-shot-image-classification]
tags: [inaturalist, landscape, land-use, land-cover, geospatial, remote-sensing, ground-level-imagery, siglip]
size_categories: [n<1K]
configs:
- config_name: all
  data_files:
  - split: train
    path: data/all/*.parquet
  default: true
- config_name: score_ge_0_10
  data_files:
  - split: train
    path: data/score_ge_0_10/*.parquet
- config_name: score_ge_0_50
  data_files:
  - split: train
    path: data/score_ge_0_50/*.parquet
- config_name: score_ge_0_90
  data_files:
  - split: train
    path: data/score_ge_0_90/*.parquet
- config_name: embeddings
  data_files:
  - split: train
    path: embeddings/siglip2-base-patch16-224/small/*.parquet
---

![iNaturalist Landscape Filtering: a deterministic sample of iNaturalist photos of plants and unidentified observations is scored for landscape content by SigLIP2; photos above each threshold keep their image](assets/hero.png)

# iNaturalist Landscape Filtering

**10 iNaturalist photos of plants and unidentified observations (8 with a location), each with a zero-shot SigLIP2 score that tells if the photo shows a landscape.**

**4 photos reach the lowest threshold (0.1 on `score_landscape`, at most 100 per H3 cell) and come with their 500 px image. All 10 come with a 768-dimension [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) image embedding.**

The photos come from the [iNaturalist Open Data](https://github.com/inaturalist/inaturalist-open-data) snapshot of 2026-10-27, observed from 2001-01-01 to 2026-10-01. Every row is one photo with its location, date, taxon, licence and scores. The goal is a set of ground-level photos that show land use and land cover, to pair with remote sensing. This is a pilot: the scores have no human validation yet.

![Photos per cell of 2° and photos at or above 0.1](assets/map.png)

## Quick start

```python
from datasets import load_dataset

photos = load_dataset("u/r", "score_ge_0_90", split="train")  # strictest threshold, with images
scores = load_dataset("u/r", split="train")  # all photos: metadata and scores
emb = load_dataset("u/r", "embeddings", split="train")  # photo_id, embedding
```

Pick your own threshold, or join the embeddings, with DuckDB:

```sql
SELECT a.photo_id, a.latitude, a.longitude, a.score_landscape, e.embedding
FROM 'hf://datasets/u/r/data/all/*.parquet' a
JOIN 'hf://datasets/u/r/embeddings/siglip2-base-patch16-224/small/*.parquet' e USING (photo_id)
WHERE a.score_landscape >= 0.001 AND a.license <> 'CC-BY-NC';
```

## Contents

| Config | Rows | Content |
|---|---:|---|
| `all` (default) | 10 | Metadata and scores, no image |
| `score_ge_0_10` | 4 | `score_landscape` ≥ 0.1, best 100 per cell, with the 500 px image |
| `score_ge_0_50` | 2 | `score_landscape` ≥ 0.5, best 100 per cell, with the 500 px image |
| `score_ge_0_90` | 1 | `score_landscape` ≥ 0.9, best 100 per cell, with the 500 px image |
| `embeddings` | 10 | `photo_id` and the SigLIP2 image embedding (768 float16 values) |

![Distribution of score_landscape, with the thresholds](assets/scores.png)

Score quantiles, over all 10 photos:

| p50 | max |
|---:|---:|
| 1e-05 | 0.013 |

| Group | Photos | ≥ 0.1 | ≥ 0.5 | ≥ 0.9 |
|---|---:|---:|---:|---:|
| plantae | 9 | 4 | 2 | 1 |
| unidentified | 1 | 0 | 0 | 0 |

| Licence | Photos | ≥ 0.1 | ≥ 0.5 | ≥ 0.9 |
|---|---:|---:|---:|---:|
| CC-BY | 6 | 3 | 1 | 1 |
| CC0 | 4 | 1 | 1 | 0 |

## Fields

| Field | Type | Description |
|---|---|---|
| `photo_id`, `photo_uuid` | int / string | iNaturalist photo ids |
| `observation_uuid` | string | iNaturalist observation id |
| `latitude`, `longitude` | float | Location of the observation (WGS 84) |
| `positional_accuracy` | int | Accuracy of the location, in metres (often empty) |
| `observed_on` | string | Date of the observation (`YYYY-MM-DD`) |
| `taxon_id`, `taxon_name` | int / string | Taxon of the observation (empty if none) |
| `group` | string | `plantae` or `unidentified` |
| `quality_grade` | string | `research`, `needs_id` or `casual` |
| `anomaly_score` | float | iNaturalist location anomaly score (often empty) |
| `position` | int | Position of the photo in its observation (0 is the first) |
| `width`, `height`, `extension` | int / string | Size and format of the original |
| `license` | string | `CC0`, `CC-BY` or `CC-BY-NC` |
| `attribution` | string | Attribution text that the licence requires |
| `observer_id`, `login`, `observer_name` | int / string | Photographer |
| `url_small`, `url_medium` | string | 240 px and 500 px photo on the open data bucket |
| `score_landscape` | float | Main score, see [Scores](#scores) |
| `score_landscape_1`, `score_landscape_2` | float | Score of each landscape prompt, in order |
| `score_closeup` | float | Score of the close-up prompts |
| `score_contrast` | float | Landscape minus specimen prompts: cosine in [-1, 1] |
| `h3_cell` | int | [H3](https://h3geo.org) cell of the location |
| `sample_rank` | int | Position of the photo in the sample order |
| `image` | image | The 500 px photo (threshold configs only) |

The `embeddings` config has one row per photo: `photo_id` and `embedding` (list of 768 float16). Join it to the other configs on `photo_id`.

## Files

```
data/all/*.parquet                      all photos: metadata and scores
data/<config>/*.parquet                 photos at or above a threshold, with the 500 px image
embeddings/siglip2-base-patch16-224/small/*.parquet   photo_id and SigLIP2 embedding
assets/                                 figures of this card
```

Configs: `all` (default), `score_ge_0_10`, `score_ge_0_50`, `score_ge_0_90`, `embeddings`.

## Scores

Each score is `sigmoid(scale * cosine + bias)` between the image embedding and a text embedding, as SigLIP computes it. A prompt set uses the mean of its unit text embeddings.

* `score_landscape` uses these prompts; `score_landscape_<i>` is the score of prompt `i` alone:
   1. `a`
   2. `b`
* `score_closeup` uses `c`, `d`.
* `score_contrast` is the cosine between the image and the direction "mean landscape prompt minus mean specimen prompt", with the specimen prompts `e`, `f`. It is high for a view of the land and low for a close view of one organism or of its traces.

The scores are not calibrated probabilities, and they are small: most photos score below 0.0001. Their order is what matters. In the pilot (November 2026), a visual check of random photos in each score band set the thresholds 0.0005, 0.001 and 0.002: at or above 0.002, almost all photos are scene views; between 0.0005 and 0.001, about half are close views of vegetation or ground.

## Methodology

1. **Source.** The metadata dump [`inaturalist-open-data-20261027.tar.gz`](https://x/metadata/inaturalist-open-data-20261027.tar.gz) is read once as a stream (36.7 GB compressed); it is not stored.
2. **Sample.** An observation is in the sample when the first 8 hex digits of its UUID are below `0.001 * 16^8`. UUID v4 values are random, so the sample is random and deterministic, and a larger fraction gives a superset. This gave 1,234 candidate photos. The dataset holds the first 10 in (observation UUID, photo position, photo id) order; a larger sample keeps them. Within each [H3](https://h3geo.org) cell (resolution 5, about 250 km²), only the first 1,000 photos in that order enter the sample, so dense areas do not dominate. Photos with no location are left out.
3. **Taxa.** Only Plantae (taxon 47126 and its descendants) and observations with no taxon or the root taxon Life. Habitat and vegetation photos are frequent there.
4. **Licences.** Only photos under CC0, CC-BY or CC-BY-NC, the licences of the open data bucket.
5. **Embeddings.** The `small` photo (240 px longest side) goes through [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224). Download and decode cost more than the model at scale; the model input is 224 px anyway.
6. **Scores.** Zero-shot, from the stored embeddings (see [Scores](#scores)).
7. **Thresholds.** 0.1, 0.5, 0.9 on `score_landscape`. Each threshold config keeps at most 100 photos per cell, the best scored, with their `medium` image (500 px).

Code, with the decisions and their tradeoffs: [github.com/NoeFlandre/inat-landscape-filtering](https://github.com/NoeFlandre/inat-landscape-filtering).

## Known limits

* **Obscured locations.** iNaturalist moves some locations (threatened taxa, user choice) to a random point in a cell of about 0.2°. The source has no flag for this.
* **Missing locations.** 2 photos have no coordinates. Filter on `latitude IS NOT NULL`.
* **What the location means.** It is where the observer or the organism was, not the area in view.
* **No validation.** The thresholds come from a visual check, not from labels.
* **Bias.** iNaturalist users are most active in North America and Europe; the map shows it.

## License

Each photo keeps its own licence, in `license`, and the attribution it requires, in `attribution`:

* `CC0`: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), no rights reserved.
* `CC-BY`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* `CC-BY-NC`: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), no commercial use. Remove these rows with a filter on `license` for commercial use.

Metadata: iNaturalist Open Data. Scores and embeddings: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). To have a photo removed, open a discussion on this repository.
