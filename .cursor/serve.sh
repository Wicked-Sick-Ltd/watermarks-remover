#!/usr/bin/env bash
# Run the core HTTP service in the foreground. A second invocation follows the
# log instead of binding the port again.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.cursor/health.sh"

PORT="${WATERMARKS_SERVER_PORT:-8765}"
HOST="${WATERMARKS_SERVER_HOST:-127.0.0.1}"
PYTHON="${WATERMARKS_PYTHON:-$ROOT/.venv/bin/python}"
SCRIPT="${WATERMARKS_SERVER_SCRIPT:-$ROOT/service/scripts/server.py}"
runtime="${TMPDIR:-/tmp}"
LOG="${runtime}/watermarks-remover-http-${PORT}.log"
LOCK="${runtime}/watermarks-remover-http-${PORT}.lock"
PIDFILE="${runtime}/watermarks-remover-http-${PORT}.pid"

touch "$LOG"

follow_log() {
  echo "watermarks-remover HTTP service already running; following ${LOG}"
  exec tail -n +1 -f "$LOG"
}

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK"
  if ! flock -n 9; then
    follow_log
  fi
fi

if health_ok; then
  if command -v flock >/dev/null 2>&1; then
    flock -u 9 || true
  fi
  follow_log
fi

echo $$ >"$PIDFILE"
export PYTHONUNBUFFERED=1
"$PYTHON" "$SCRIPT" --host "$HOST" --port "$PORT" 2>&1 | tee -a "$LOG"
