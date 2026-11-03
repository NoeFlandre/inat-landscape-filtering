#!/usr/bin/env bash
# Start one job chain for each free GPU on all Grid'5000 sites, with one global count N.
#
# Run it on a machine that reaches each site frontend with `ssh SITE` (a frontend, or a
# computer with the SSH configuration of https://www.grid5000.fr/w/SSH). It does only light
# work on the frontends: it reads the OAR resources, updates the repository, and submits.
#
# Usage: ./scripts/g5k/submit_all.sh [SITE…]      (default: all sites with GPUs)
#
# Each site must have ~/.hf_token (mode 600). Slot I/N goes to one GPU: no two chains share
# a slot. The script stops if a chain of an earlier run is still queued or running, because
# slots of two different counts N overlap.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
sites=("$@")
[ ${#sites[@]} -gt 0 ] || sites=(grenoble lille lyon nancy nantes rennes sophia toulouse luxembourg)
branch="claude/new-session-8l7p9c"
repo="https://github.com/NoeFlandre/inat-landscape-filtering"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for site in "${sites[@]}"; do
    if ssh "$site" 'oarstat -u -f -J 2>/dev/null' | grep -q '"name" *: *"inat-embed-'; then
        echo "$site has inat-embed jobs: wait for them to end, or delete them with oardel" >&2
        exit 1
    fi
    ssh "$site" test -s .hf_token || { echo "$site has no ~/.hf_token" >&2; exit 1; }
    ssh "$site" python3 - < "$here/free_gpus.py" > "$tmp/$site.jsonl"
    echo "$site: $(cat "$tmp/$site.jsonl" | tr '\n' ' ')"
done
python3 "$here/free_gpus.py" plan "$tmp"/*.jsonl > "$tmp/plan.jsonl"
[ -s "$tmp/plan.jsonl" ] || { echo "no free GPU" >&2; exit 1; }

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
    echo "$site: $(ssh -n "$site" "eval $command" | tail -1)"
done
echo "submitted $(wc -l < "$tmp/plan.jsonl") slot(s)"
