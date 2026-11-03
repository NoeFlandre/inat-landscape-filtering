#!/usr/bin/env bash
# Supervise 4 CPU embedding workers until the sample is fully embedded, refresh the HF
# dataset hourly (metadata only), then finish (retry pass for failed downloads, final score and
# publish with the 500 px images, uploaded in parts).
# Workers exit every 20,000 photos to free memory, and are restarted here. A refresh runs only
# with at least 4.5 GB RAM and 3 GB disk free. The HF token is read from ~/.hf_token.
# Usage: nohup scripts/cpu_supervisor.sh >> work/full/supervisor.log 2>&1 &
cd "$(dirname "$0")/.."
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.hf_token")}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
W=work/full; N=4
declare -A seen
log() { echo "$(date -u +%H:%M) $*"; }
pid_of() { pgrep -f "python3 .*embed --worker $1/$N " | head -1; }
commits() { grep -c "commit/main" "$W/embed-w$1.log" 2>/dev/null || echo 0; }
todo() { uv run --no-sync inat-landscape --workdir $W embed --worker "$1/$N" --plan 2>/dev/null | sed -n 's/^todo=//p'; }
start() {
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup uv run --no-sync inat-landscape --workdir $W embed \
        --worker "$1/$N" --chunk 10000 --batch-size 64 --concurrency 32 --download-batch 512 \
        --max-photos 20000 >> "$W/embed-w$1.log" 2>&1 &
}
publish() {  # $1: --images or --no-images
    if ! uv run --no-sync inat-landscape --workdir $W score > "$W/score-last.log" 2>&1; then
        log "score failed (exit $?): publish skipped"; return
    fi
    grep -E "^n=" "$W/score-last.log"
    uv run --no-sync inat-landscape --workdir $W publish --score score_contrast \
        --thresholds 0.04,0.06,0.08 --cap-per-cell 10 "$1" | grep -E "^all=|contrast_ge"
}
free_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
last_refresh=$(date +%s)  # first refresh one hour after start
while true; do
    finished=0
    for ((i = 0; i < N; i++)); do
        p=$(pid_of "$i")
        if [ -n "$p" ]; then
            c=$(commits "$i"); rss=$(( $(awk '/VmRSS/ {print $2}' /proc/$p/status 2>/dev/null || echo 0) / 1024 ))
            if [ "$c" -gt "${seen[$i]:-$c}" ] && [ "$rss" -gt 3300 ]; then
                log "worker $i recycled after a shard save (${rss} MB)"; kill "$p"
            fi
            seen[$i]=$c
        else
            left=$(todo "$i")
            if [ "${left:-1}" = 0 ]; then finished=$((finished + 1))
            else log "worker $i (re)started: ${left:-?} photos left"; start "$i"; fi
        fi
    done
    if [ "$finished" -eq "$N" ]; then
        log "all workers finished: retry pass for failed downloads"
        uv run --no-sync inat-landscape --workdir $W embed --chunk 10000 --batch-size 64 \
            --concurrency 32 --download-batch 512 2>&1 | grep -E "^total="
        log "final publish with images"; publish --images; log "DONE"; exit 0
    fi
    if [ $(( $(date +%s) - last_refresh )) -ge 3600 ]; then
        if [ "$(free_mb)" -ge 4500 ] && [ "$(df -m --output=avail / | tail -1)" -ge 3000 ]; then log "refresh"; publish --no-images; last_refresh=$(date +%s)
        else log "refresh skipped: $(free_mb) MB RAM, $(df -m --output=avail / | tail -1) MB disk free"; fi
    fi
    sleep 30
done
