# ADR 0009: Embed on Grid'5000 with parallel short jobs

Status: accepted (2026-11-02)

## Context

With a cap of 10 photos per H3 cell, the sample has 1.8 million photos. The CPU machine embeds about 7 photos per second (3 days). Grid'5000 has GPUs. Its usage policy limits daytime use, but jobs of 1 hour or less do not count in the daytime quota.

## Decision

- `embed --worker I/N` keeps the photos with `photo_id mod N = I`. N jobs run in parallel without overlap.
- Each job has a walltime of 1 hour. The embeddings on Hugging Face are the record of done work, so a new job continues the work of an old one.
- The jobs read the sample from a private Hugging Face repository (`state pull`; now `state/` of the dataset, ADR 0015). Grid'5000 does not read the metadata dump.
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

## Multi-site run (2026-11-03)

- `scripts/g5k/free_gpus.py` reads the OAR resources of each site and finds the free GPUs that the cu126 PyTorch build can use. `scripts/g5k/submit_all.sh` gives each free GPU one slot `I/N`, with one global N, and submits one job chain for each slot on the site of its GPU. A slot has one chain only, and the script stops if chains of an earlier run are still queued or running.
- Each chain keeps the oarsub options of its cluster (queue, job type, cluster). Some production clusters admit the user only in the besteffort queue: there, the chain uses besteffort jobs.
- A chain stops when no photo is left, or when a job made no progress.
- Most `/home` directories had less than 2 GB free. Thus the Python environment is on the node `/tmp`, in one folder for each slot (jobs can share a node). Each job installs it again. `/home` keeps only small slot files (`~/.inat-landscape/`).
- The workers also skip the permanent failures of the shared `failures.parquet`. Before, a worker on a new node tried the known 404 photos again, so its count of photos left never became 0.
- Result: 9 slots on the free GPUs of three Grenoble clusters (vercors9, vercors10 and vercors16; slot 8 waited for a busy Rennes GPU and moved to a Grenoble L4) embedded the last 1,004,353 photos in about 1 hour, at about 200 photos per second on each L4. 4 slots ran as besteffort jobs. The retry pass tried the 10 failed downloads again, without success. The final dataset has 1,817,780 embedded photos of 1,817,793 (13 failed downloads).
- `scripts/g5k/publish.oar` runs `score` and `publish` on a CPU node: every 20 minutes without images during the run, and once with the images at the end.
- The final publish with images takes more than 1 hour. The production queue refused a walltime extension and stopped the first try at 1 hour. Thus the final publish runs in the default queue, with 1-hour extensions in the last 10 minutes of the job.

## All-taxa run (2026-11-03 to 2026-11-04)

- The sample of all taxa, with a cap of 1,000 photos per H3 cell, has 74,215,277 photos. The run used 135 weighted slots (`I:J/N`, ADR 0011) on the free GPUs of four sites (Grenoble, Nancy, Rennes, Sophia). A fast GPU owns a run of more slots, so all slots advance at about one speed.
- A done-ids snapshot (`embed --snapshot`, one index file each hour) keeps the Hugging Face reads low. Before the snapshot, many jobs read all shards at the same time and Hugging Face refused the reads (HTTP 429). The shard pushes of a job go in one commit each 30 minutes, below the limit of 128 commits each hour for a repository.
- Result: 74,214,068 of the 74,215,277 sample photos have an embedding. The retry pass tried the 1,196 failed downloads again, without success; with 13 photos skipped earlier, 1,209 photos have no embedding.
- A photo can be in two observations. When two slots embedded the same photo, it has two rows in the `embeddings` config (15,837 photos). `score` keeps the first copy, and the card gives the count.
- The final `score` and `publish` (without images) took about 2.5 hours on one CPU node. The default-queue CPU nodes of Grenoble were busy for hours, so the job ran in the production queue of Nancy, with a walltime of 4 hours.
- A second `publish` stopped on a node with 96 GB of memory: the system stopped the process (exit 137) after `score`. A node with 192 GB is necessary (`memnode>=190000`).
- With 8 parallel shard reads and the score cache (ADR 0012), `score` took 32 minutes in place of about 2 hours, and the total job took about 1 hour on a Rennes production node.
- After the run, `~/.hf_token` was deleted on all nine sites. The private state repository stays (until ADR 0015).
- After the land filter and the diversity maps (ADR 0013, ADR 0014), all the data on the nine sites was deleted (only `~/.ssh` and the shell profiles stay), with the second token.
