#!/bin/bash
# SessionStart: install the Python environment.
# The CPU run is handed over to Grid'5000: the hook does not relaunch it (two writers would
# embed the same photos twice).
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
