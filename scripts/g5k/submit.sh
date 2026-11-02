#!/usr/bin/env bash
# Submit N parallel short embedding jobs on Grid'5000, one GPU each.
#
# Usage policy (https://www.grid5000.fr/w/Grid5000:UsagePolicy): a job of one hour or less,
# submitted less than 10 minutes before it starts, does not count in the daytime quota.
# So each job has a walltime of 1 hour. The jobs resume from the embeddings on the hub:
# run this script again until `embed` reports nothing left to do.
#
# Usage, on a frontend, from the repository:  ./scripts/g5k/submit.sh [WORKERS]
set -euo pipefail
workers="${1:-4}"
for ((i = 0; i < workers; i++)); do
    oarsub -l "host=1/gpu=1,walltime=1:00:00" "./scripts/g5k/embed.oar $i/$workers"
done
echo "submitted $workers job(s) of 1 hour"
