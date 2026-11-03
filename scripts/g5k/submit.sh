#!/usr/bin/env bash
# Start the job chain of one worker slot I/N on this site, with 1-hour GPU jobs.
#
# Usage policy (https://www.grid5000.fr/w/Grid5000:UsagePolicy): a job of one hour or less,
# submitted less than 10 minutes before it starts, does not count in the daytime quota.
# So each job has a walltime of 1 hour, and each job queues the next job of its slot.
#
# Usage, on a frontend, from the repository:  ./scripts/g5k/submit.sh I/N [OARSUB OPTIONS…]
#   for example: ./scripts/g5k/submit.sh 3/16 -q production -p "cluster='grue'"
# The options select the GPUs (queue, job type, properties). The next jobs use them again.
set -euo pipefail
worker="${1:?usage: submit.sh I/N [oarsub options]}"
shift
slot="${worker/\//-of-}"
keep="$HOME/.inat-landscape/$slot"
# One chain for each slot: do not start a second chain of the same slot on this site.
if oarstat -u -f -J 2>/dev/null | grep -q "\"name\" *: *\"inat-embed-$slot\""; then
    echo "slot $worker has a job chain on this site already" >&2
    exit 1
fi
mkdir -p "$keep" "$HOME/.inat-landscape/logs"
printf '%s\n' "$@" > "$keep/oarsub.options"
rm -f "$keep/todo"
oarsub -n "inat-embed-$slot" "$@" -l "gpu=1,walltime=1:00:00" \
    -O "$HOME/.inat-landscape/logs/%jobid%.out" -E "$HOME/.inat-landscape/logs/%jobid%.err" \
    "$PWD/scripts/g5k/embed.oar $worker"
