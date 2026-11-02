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

## Speed (2026-11-02)

- The vision model runs under bf16 autocast on CPUs with AMX or AVX-512 BF16, and under bf16 or fp16 on GPUs. `torch.compile` is on by default (`--no-compile` to stop it).
- Photos are resized with PIL, as the model processor does, in a thread pool. A background thread downloads and decodes the next photos while the model runs.
- On the 4-core CPU of the work machine: 5.6 photos per second before, about 16 after. The embeddings stay within a cosine of 0.997 (median 0.9999) of the fp32 embeddings, which is the noise of the float16 storage.
- `torch.compile` freezes the weights (`torch._inductor.config.freezing`): the weights are folded into the graph and laid out for AMX. Measured on 1,024 real photos: 15.4 → 25.2 photos per second, with the same distance to the fp32 embeddings as before (min cosine 0.9989, median 0.9999). Batch 128 (21.2) and `max-autotune` (21.5) were slower and are not used.
- On a CPU, several single-thread workers beat one multi-thread worker: on 4 cores and 4,096 real photos, 1 × 4 threads gave 15.8 photos per second, 2 × 2 gave 17.4, and 4 × 1 gave 21.5 (load and compile included). The Python steps of one worker then overlap with the model of the others. Each worker reads only two columns of the sample and keeps its own failure file, and `score` works in chunks of 100,000 embeddings, so 4 workers and the hourly refresh fit in 15 GB of memory.
