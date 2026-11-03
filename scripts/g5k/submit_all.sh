#!/usr/bin/env bash
# Start a job chain for free worker slots of N on the free GPUs of all Grid'5000 sites.
#
# Run it on a machine that reaches each site frontend with `ssh SITE` (a frontend, or a
# computer with the SSH configuration of https://www.grid5000.fr/w/SSH). It does only light
# work on the frontends: it reads the OAR resources, updates the repository, and submits.
#
# Usage: ./scripts/g5k/submit_all.sh N [SITE…]      (default: all sites with GPUs)
#
# Run it again at any time (for example every 15 minutes): it starts only the slots that have
# no queued or running job and that are not done, on the GPUs that are free now. So each new
# job starts at once, and the run grows when GPUs become free. Each site must have
# ~/.hf_token (mode 600). A slot has one chain only; the script stops if a chain of another
# count N is queued or running, because slots of two counts overlap.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
count="${1:?usage: submit_all.sh N [SITE…]}"
shift
sites=("$@")
[ ${#sites[@]} -gt 0 ] || sites=(grenoble lille lyon nancy nantes rennes sophia toulouse luxembourg)
branch="claude/new-session-8l7p9c"
repo="https://github.com/NoeFlandre/inat-landscape-filtering"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

: > "$tmp/busy"
for site in "${sites[@]}"; do
    ssh "$site" test -s .hf_token || { echo "$site has no ~/.hf_token" >&2; exit 1; }
    # Slots with a queued or running job, and slots that are done.
    ssh "$site" "oarstat -u -f -J 2>/dev/null | grep -o '\"name\" *: *\"inat-embed-[0-9]*-of-[0-9]*\"' | grep -o '[0-9]*-of-[0-9]*'; ls -d .inat-landscape/*-of-$count/done 2>/dev/null | grep -o '[0-9]*-of-[0-9]*'" >> "$tmp/busy" || true
    ssh "$site" python3 - < "$here/free_gpus.py" > "$tmp/$site.jsonl"
done
if grep -v -- "-of-$count\$" "$tmp/busy" | grep -q .; then
    echo "chains of another worker count are queued or running: wait, or delete them" >&2
    exit 1
fi
free_slots="$(python3 -c '
import sys
busy = {int(line.split("-")[0]) for line in open(sys.argv[2]) if line.strip()}
print(",".join(str(i) for i in range(int(sys.argv[1])) if i not in busy))
' "$count" "$tmp/busy")"
python3 "$here/free_gpus.py" plan "$count" "$free_slots" "$tmp"/*.jsonl > "$tmp/plan.jsonl"
echo "free slots: $(tr ',' '\n' <<< "$free_slots" | grep -c . || true), free GPUs used: $(wc -l < "$tmp/plan.jsonl")"
[ -s "$tmp/plan.jsonl" ] || exit 0

for site in $(python3 -c 'import json,sys; print(*sorted({json.loads(l)["site"] for l in sys.stdin}))' < "$tmp/plan.jsonl"); do
    ssh "$site" "if [ -d inat-landscape-filtering/.git ]; then git -C inat-landscape-filtering pull -q --ff-only; else git clone -q -b $branch $repo inat-landscape-filtering; fi"
done
# One submission for each slot, on the site of its GPU, with the oarsub options of its cluster.
python3 -c '
import json, shlex, sys
for line in sys.stdin:
    s = json.loads(line)
    args = " ".join(shlex.quote(a) for a in [s["slot"], *s["options"]])
    print(s["site"], shlex.quote("cd inat-landscape-filtering && ./scripts/g5k/submit.sh " + args))
' < "$tmp/plan.jsonl" | while read -r site command; do
    echo "$site: $(ssh -n "$site" "eval $command" 2>&1 | tail -1)"
done
