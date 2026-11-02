# iNaturalist landscape filtering

This project finds iNaturalist photos that show a landscape.
Use these photos for land use and land cover work.
Each photo keeps its geolocation.

The project gives each photo a zero-shot score from the SigLIP2 model.
It publishes the scores and the photos to a Hugging Face dataset:
[NoeFlandre/inat-landscape-filtering](https://huggingface.co/datasets/NoeFlandre/inat-landscape-filtering).

## Pipeline

| Step | Command | Result |
|---|---|---|
| 1 | `ingest` | Reads the iNaturalist metadata dump once. Keeps a deterministic sample. |
| 2 | `sample` | Selects the first N candidate photos. |
| 3 | `embed` | Downloads the 240 px photos. Stores one embedding for each photo. |
| 4 | `score` | Calculates the scores. Prints the score distribution. |
| 5 | `publish` | Writes the dataset configs and card. Uploads them. |

Each step can run again. A step does not do work that is already done.
