# ADR 0015: The pipeline state goes in the dataset repository

Status: accepted (2026-11-05). Replaces the private state repository of ADR 0009.

## Context

The work files of the pipeline (the dump index, the candidates, the sample, the scores, the land cells and the diversity) were in a private repository, `NoeFlandre/inat-landscape-filtering-state`. Thus the dataset was not complete without a second repository.

## Decision

- `state push` and `state pull` use `state/` in the dataset repository (`--state-repo`, default: the dataset).
- `state/` is not a config: the card lists the configs with explicit `data_files`, so the dataset viewer and `load_dataset` do not read `state/`. The card gives the file tree of `state/`.
- `publish` deletes only in `data/` and `assets/`. It does not touch `state/`.
- The private state repository was deleted after its files were copied to `state/`.

## Consequences

- The dataset is self-contained: with the code, it is possible to run the pipeline again or to extend the sample without the iNaturalist dump. The candidates are a dated snapshot that iNaturalist can replace.
- The dataset is about 14 GB larger. The state has the photos at sea of the sample: they are not in a config.
