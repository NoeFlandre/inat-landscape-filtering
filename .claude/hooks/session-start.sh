#!/bin/bash
# SessionStart: install the Python environment, then relaunch the CPU embedding run if the VM
# restarted while it was in progress (work/full has a sample, and no supervisor is running).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Sync only if a file that decides the environment changed since the last successful sync.
hashes=.venv/.session-start.sha256
if ! sha256sum --check --status "$hashes" 2>/dev/null; then
  rm -f "$hashes"
  uv sync --locked
  sha256sum .claude/hooks/session-start.sh pyproject.toml uv.lock > "$hashes" || rm -f "$hashes"
fi

# Relaunch the run after a VM restart. The token is read from ~/.hf_token, never from the repo.
supervisor_running() {
  ps -eo args | awk '$1 ~ /bash$/ && $2 ~ /(cpu_supervisor|supervisor[0-9]*)\.sh$/ {found = 1} END {exit !found}'
}
if [ -f work/full/sample.parquet ] && [ -s "$HOME/.hf_token" ] && ! supervisor_running; then
  setsid nohup scripts/cpu_supervisor.sh >> work/full/supervisor.log 2>&1 < /dev/null &
  echo "Relaunched the CPU embedding supervisor (log: work/full/supervisor.log)."
fi
