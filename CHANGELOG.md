# Changelog

## 1.0.0 (2026-11-05)

The first stable release. Use it to go back to a known good state.

- Code: git tag `v1.0.0`.
- Dataset: tag `v1.0.0` of [NoeFlandre/inat-landscape-filtering](https://huggingface.co/datasets/NoeFlandre/inat-landscape-filtering) (commit `74a16ee`).

Content:

- 73,033,520 iNaturalist photos of all taxa, on land, each with a zero-shot SigLIP2 landscape score and a 768-d image embedding.
- Configs: `all`, one config per proposed threshold (0.04, 0.06, 0.08 on `score_contrast`, at most 1000 photos per H3 cell), and `embeddings`.
- Three maps on the Equal Earth projection, with H3 cells of resolution 4 (ADR 0016).
- `state/`: the work files to run the pipeline again without the iNaturalist dump (ADR 0015).
