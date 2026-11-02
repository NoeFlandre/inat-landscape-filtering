# ADR 0009: Embed on Grid'5000 with parallel short jobs

Status: accepted (2026-11-02)

## Context

With a cap of 10 photos per H3 cell, the sample has 1.8 million photos. The CPU machine embeds about 7 photos per second (3 days). Grid'5000 has GPUs. Its usage policy limits daytime use, but jobs of 1 hour or less do not count in the daytime quota.

## Decision

- `embed --worker I/N` keeps the photos with `photo_id mod N = I`. N jobs run in parallel without overlap.
- Each job has a walltime of 1 hour. The embeddings on Hugging Face are the record of done work, so a new job continues the work of an old one.
- The jobs read the sample from a private Hugging Face repository (`state pull`). Grid'5000 does not read the metadata dump.
- PyTorch comes from a uv dependency group: `cpu` (default) or `cu126` (Grid'5000).
- The work directory and the caches are on the node-local `/tmp`. `/home` keeps only the repository and its environment.
- Shards have 50,000 photos: about 37 commits on Hugging Face instead of 1,800.

## Consequences

- The run time goes down with the number of workers, and the jobs respect the policy at all hours.
- Each job reads the shards of the other workers to know what is done. This is a few GB per job.
- `score` and `publish` run outside Grid'5000.
