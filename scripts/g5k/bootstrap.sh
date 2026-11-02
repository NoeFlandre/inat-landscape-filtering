#!/usr/bin/env bash
# One-time setup and start on a Grid'5000 frontend with GPUs (for example Lille or Lyon),
# from a browser terminal (https://intranet.grid5000.fr/notebooks) or an SSH session:
#
#   curl -fsSL https://raw.githubusercontent.com/NoeFlandre/inat-landscape-filtering/claude/new-session-8l7p9c/scripts/g5k/bootstrap.sh | bash -s -- 4
#
# It clones the repository, asks once for a Hugging Face write token (not shown, saved with
# mode 600 in ~/.hf_token), and submits WORKERS parallel 1-hour GPU jobs.
set -euo pipefail
workers="${1:-4}"
branch="claude/new-session-8l7p9c"
dir="$HOME/inat-landscape-filtering"

if [ -d "$dir/.git" ]; then
    git -C "$dir" pull --ff-only
else
    git clone -b "$branch" https://github.com/NoeFlandre/inat-landscape-filtering "$dir"
fi
if [ ! -s "$HOME/.hf_token" ]; then
    printf "Hugging Face write token (hidden): " > /dev/tty
    read -rs token < /dev/tty
    echo > /dev/tty
    (umask 077 && printf "%s" "$token" > "$HOME/.hf_token")
fi
cd "$dir" && ./scripts/g5k/submit.sh "$workers"
