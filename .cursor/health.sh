# Shared by start.sh and serve.sh. Sourced, not executed.
# Probes loopback /health. Sends WATERMARKS_SERVER_API_KEY as a bearer token
# when that variable is set; otherwise the request stays unauthenticated.
health_ok() {
  local port="${WATERMARKS_SERVER_PORT:-8765}"
  local -a args=(-fsS --max-time 2)
  if [[ -n "${WATERMARKS_SERVER_API_KEY:-}" ]]; then
    args+=(-H "Authorization: Bearer ${WATERMARKS_SERVER_API_KEY}")
  fi
  curl "${args[@]}" "http://127.0.0.1:${port}/health" >/dev/null 2>&1
}
