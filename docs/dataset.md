# Dataset

The dataset is [NoeFlandre/inat-landscape-filtering](https://huggingface.co/datasets/NoeFlandre/inat-landscape-filtering).

## Layout

```
README.md                                       dataset card
data/all/*.parquet                              all scored photos: metadata and scores
data/<config>/*.parquet                         photos at or above a threshold, best 100 per H3 cell, with the 500 px image
embeddings/siglip2-base-patch16-224/small/*.parquet   photo_id and float16 embedding
```

## Columns

| Column | Content |
|---|---|
| `photo_id`, `photo_uuid`, `observation_uuid` | iNaturalist identifiers |
| `latitude`, `longitude`, `positional_accuracy` | location of the observation, accuracy in metres |
| `observed_on` | date of the observation |
| `taxon_id`, `taxon_name`, `group` | taxon; `group` is `plantae` or `unidentified` |
| `quality_grade`, `anomaly_score` | iNaturalist quality data |
| `position` | position of the photo in its observation (0 is the first photo) |
| `width`, `height`, `extension` | size and format of the original photo |
| `license`, `attribution`, `login`, `observer_name`, `observer_id` | licence and the attribution it requires |
| `url_small`, `url_medium` | URLs of the 240 px and 500 px photos |
| `score_landscape` | main score: mean of the landscape prompts |
| `score_landscape_1` to `score_landscape_3` | score of each landscape prompt |
| `score_closeup` | score of the close-up prompts |
| `score_contrast` | cosine with "landscape minus specimen" prompts, in [-1, 1] |
| `h3_cell` | H3 cell of the location (resolution 5) |
| `sample_rank` | position of the photo in the sample order |
| `image` | the 500 px photo (threshold configs only) |

## Scores

A score is `sigmoid(scale * cosine + bias)`.
SigLIP calculates its logits this way.
A score is not a calibrated probability.
Look at the photos before you select a threshold.
