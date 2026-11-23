#!/usr/bin/env bash
# Boot hook. Launch the HTTP service, then wait until /health succeeds.
# The platform runs this before configured terminals, so the server cannot
# be left to the http terminal alone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.cursor/health.sh"

PORT="${WATERMARKS_SERVER_PORT:-8765}"
LOG="${TMPDIR:-/tmp}/watermarks-remover-http-${PORT}.log"
url="http://127.0.0.1:${PORT}/health"

# nohup inherits this process environment (including WATERMARKS_SERVER_API_KEY).
# A tmux session would not: the tmux server keeps the environment it was first
# started with, so a later boot would drop the key and the port.
if ! health_ok; then
  nohup bash .cursor/serve.sh >/dev/null 2>&1 &
fi

for _ in $(seq 1 30); do
  if health_ok; then
    echo "watermarks-remover health ok at ${url}"
    exit 0
  fi
  sleep 1
done

echo "error: HTTP service did not answer ${url}" >&2
if [[ -f "$LOG" ]]; then
  tail -n 40 "$LOG" >&2
fi
exit 1
