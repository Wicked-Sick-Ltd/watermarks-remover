#!/usr/bin/env bash
# Boot hook. The HTTP service itself is the `http` terminal so it keeps a
# named session. Terminals are not launched unless `start` exists, so this
# script only waits until GET /health succeeds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

url="http://127.0.0.1:8765/health"
for _ in $(seq 1 30); do
  if curl -fsS "$url" >/dev/null; then
    echo "watermarks-remover health ok at ${url}"
    exit 0
  fi
  sleep 1
done

echo "error: HTTP service did not answer ${url}" >&2
exit 1
