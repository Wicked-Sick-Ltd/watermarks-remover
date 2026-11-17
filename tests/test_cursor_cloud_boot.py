"""Cloud-agent boot must bring the HTTP service up itself and authenticate /health.

`start` runs before configured terminals, so a hook that only polls a server
launched by the `http` terminal exits before anything is listening. When
WATERMARKS_SERVER_API_KEY is set, /health returns 401 unless the probe sends
that bearer token.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
FLOCK = shutil.which("flock")
GIT = shutil.which("git")

TRACKED_CURSOR_FILES = (
    ".cursor/environment.json",
    ".cursor/health.sh",
    ".cursor/install.sh",
    ".cursor/serve.sh",
    ".cursor/start.sh",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _curl(port: int, token: str | None) -> subprocess.CompletedProcess[str]:
    cmd = ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "2"]
    if token is not None:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])
    cmd.append(f"http://127.0.0.1:{port}/health")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)


def _kill_tree(pid: int) -> None:
    children = Path(f"/proc/{pid}/task/{pid}/children")
    if children.is_file():
        for child in children.read_text(encoding="utf-8").split():
            _kill_tree(int(child))
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _stop(tmp_path: Path, port: int) -> None:
    pidfile = tmp_path / f"watermarks-remover-http-{port}.pid"
    if not pidfile.is_file():
        return
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    if pid > 0:
        _kill_tree(pid)
        time.sleep(0.1)


def test_install_script_includes_exiftool_office_zip_support() -> None:
    """exiftool needs Archive::Zip; the core image installs it explicitly."""
    script = (ROOT / ".cursor" / "install.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "service" / "Dockerfile").read_text(encoding="utf-8")
    assert "libarchive-zip-perl" in dockerfile
    assert "libarchive-zip-perl" in script
    assert "Archive::Zip" in script


@pytest.mark.skipif(GIT is None, reason="git is required")
def test_cursor_gitignore_tracks_only_environment_files() -> None:
    """Un-ignoring .cursor/ must not make local editor files committable."""
    for path in TRACKED_CURSOR_FILES:
        result = subprocess.run(
            [GIT, "check-ignore", "-q", "--no-index", path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 1, f"{path} should be tracked, check-ignore said otherwise"

    for path in (".cursor/notes.local", ".cursor/rules/local.mdc"):
        result = subprocess.run(
            [GIT, "check-ignore", "-q", "--no-index", path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, f"{path} should stay ignored"


@pytest.mark.skipif(
    BASH is None or FLOCK is None or shutil.which("curl") is None,
    reason="boot script needs bash, flock, and curl",
)
def test_start_launches_service_and_sends_bearer_token(tmp_path: Path) -> None:
    """start.sh reaches /health with the configured bearer token and no terminal."""
    port = _free_port()
    token = "cloud-boot-test-token"  # noqa: S105 — fixture bearer, not a credential
    stub = tmp_path / "stub_server.py"
    stub.write_text(
        textwrap.dedent(
            """\
            import os
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            args = sys.argv[1:]
            port = int(args[args.index("--port") + 1])
            key = os.environ.get("WATERMARKS_SERVER_API_KEY", "")


            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if key and self.headers.get("Authorization") != f"Bearer {key}":
                        self.send_response(401)
                        self.end_headers()
                        return
                    body = b'{"ok": true}'
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, fmt, *log_args):
                    return


            ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "TMPDIR": str(tmp_path),
            "WATERMARKS_SERVER_PORT": str(port),
            "WATERMARKS_SERVER_API_KEY": token,
            "WATERMARKS_PYTHON": sys.executable,
            "WATERMARKS_SERVER_SCRIPT": str(stub),
        }
    )
    try:
        refused = _curl(port, token)
        assert refused.returncode != 0

        result = subprocess.run(
            [BASH, str(ROOT / ".cursor" / "start.sh")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "health ok" in result.stdout
        assert token not in result.stdout
        assert token not in result.stderr

        denied = _curl(port, None)
        assert denied.stdout.strip() == "401"
        allowed = _curl(port, token)
        assert allowed.stdout.strip() == "200"

        again = subprocess.run(
            [BASH, str(ROOT / ".cursor" / "start.sh")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
        assert again.returncode == 0, again.stderr
        assert token not in again.stdout
    finally:
        _stop(tmp_path, port)
