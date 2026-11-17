#!/usr/bin/env bash
# Idempotent, non-interactive bootstrap for Cursor Cloud Agents.
# Runs from any cwd; installs the core toolchain only (no optional ML backends).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DEBIAN_FRONTEND=noninteractive

apt_missing=0
if ! command -v python3 >/dev/null 2>&1; then
  apt_missing=1
fi
if ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
  apt_missing=1
fi
for tool in curl ffmpeg ffprobe qpdf exiftool gs; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    apt_missing=1
  fi
done

if [[ "$apt_missing" -eq 1 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "error: system packages are missing and sudo is not available" >&2
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    ghostscript \
    libimage-exiftool-perl \
    python3 \
    python3-pip \
    python3-venv \
    qpdf
fi

# Same pin as service/Dockerfile. Optional for import/tests (they skip when
# absent) but part of the core image toolchain, so a bad download fails closed.
C2PATOOL_VERSION="c2patool-v0.27.15"
C2PATOOL_SHA256="7a035b727a6cdda8ad08d98fe94b3c20febee4aea4e7c1de02571b295730faf0"
if ! command -v c2patool >/dev/null 2>&1; then
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/c2patool.tar.gz" \
    "https://github.com/contentauth/c2pa-rs/releases/download/${C2PATOOL_VERSION}/${C2PATOOL_VERSION}-x86_64-unknown-linux-gnu.tar.gz"
  echo "${C2PATOOL_SHA256}  $tmp/c2patool.tar.gz" | sha256sum -c -
  tar -xzf "$tmp/c2patool.tar.gz" -C "$tmp"
  sudo install -m 0755 "$tmp/c2patool/c2patool" /usr/local/bin/c2patool
  rm -rf "$tmp"
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "error: Python 3.10+ is required (found $(python3 --version 2>&1))" >&2
  exit 1
fi

python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --no-input -r requirements-dev.txt

# Core install, tests, and the loopback HTTP service need no secrets.
# WATERMARKS_SERVER_API_KEY is required only when MARKLLM_DIR is set: server.py
# refuses to boot in that configuration. HF_TOKEN is only for gated model
# downloads, which this script does not perform.
if [[ -n "${MARKLLM_DIR:-}" && -z "${WATERMARKS_SERVER_API_KEY:-}" ]]; then
  echo "warning: MARKLLM_DIR is set but WATERMARKS_SERVER_API_KEY is missing; the HTTP service will not start until WATERMARKS_SERVER_API_KEY is set" >&2
fi

echo "install ok: $(.venv/bin/python --version) ($(command -v c2patool))"
