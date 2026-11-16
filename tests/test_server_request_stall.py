"""Unfinished HTTP requests must not pin service worker threads.

The stdlib server reads the request line, headers, and (for POST) the
Content-Length body on one thread per connection. These tests open real TCP
connections to that server and withhold the rest of the request. They fail
while a handler stays blocked, and they pass once every unfinished read is
abandoned within MAX_HANDLER_WAIT seconds. That includes a client that keeps
sending a byte often enough that an idle socket timeout never fires, and
enough stalled clients to exhaust the process file-descriptor limit.

Sidecar copies of the same handler are included:
synthid_score_server.py (/health, /score) and synthid_text_server.py
(/health, /watermark, /watermark/batch). A finished request with a wrong
bearer token is not a stall: authorization runs only after the request line
and headers have been read.

Run:
    python3 -m pytest -q tests/test_server_request_stall.py
"""

from __future__ import annotations

import contextlib
import http.client
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server
import synthid_score_server
import synthid_text_server

# A handler must abandon an unfinished request within this many seconds.
# The observation window is a little longer so a wait of exactly
# MAX_HANDLER_WAIT is visible to the client.
MAX_HANDLER_WAIT = 5.0
OBSERVE_SECONDS = MAX_HANDLER_WAIT + 2.0
# Obvious placeholder. Not a deployed credential.
PLACEHOLDER_API_KEY = "placeholder-api-key"
# Small enough that a test can exhaust it, large enough for the interpreter
# plus the listening socket. Every real deployment has some finite limit.
SERVER_NOFILE = 48

CORE_GETS = ("/health", "/capabilities", "/openapi.json")
CORE_POSTS = (
    "/inspect",
    "/detect",
    "/clean",
    "/watermark",
    "/inspect/batch",
    "/detect/batch",
    "/clean/batch",
    "/watermark/batch",
)


class _Running:
    def __init__(self, handler_cls):
        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.socks: list[socket.socket] = []

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def stop(self) -> None:
        for sock in self.socks:
            with contextlib.suppress(OSError):
                sock.close()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _connect(port: int, payload: bytes) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=2)
    if payload:
        sock.sendall(payload)
    return sock


def _request_line(method: str, path: str) -> bytes:
    # No terminating newline: the handler stays inside readline.
    return f"{method} {path} HTTP/1.1".encode()


def _partial_headers(method: str, path: str) -> bytes:
    # The request line is complete. The header block never reaches a blank
    # line, and the last header line itself has no newline.
    return f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nX-Stall: ".encode()


def _partial_body(path: str, *, expect_continue: bool = False, length: int = 64) -> bytes:
    # Headers are complete and Content-Length is inside the service cap, but
    # the body is shorter than that length. The peer stays open. Callers that
    # keep sending must pass a length those extra bytes cannot finish.
    extra = "Expect: 100-continue\r\n" if expect_continue else ""
    head = f"POST {path} HTTP/1.1\r\nHost: localhost\r\n{extra}Content-Length: {length}\r\n\r\n"
    return head.encode() + b"{"


def _drip(sock: socket.socket, stop: threading.Event) -> threading.Thread:
    """Keep the kernel receive buffer moving without ever finishing the read."""

    def run() -> None:
        while not stop.wait(0.05):
            try:
                sock.sendall(b"x")
            except OSError:
                return

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _assert_all_released(named: list[tuple[str, socket.socket]]) -> None:
    """Fail if any peer is still unanswered when the observation window ends."""
    deadline = time.monotonic() + OBSERVE_SECONDS
    pending = list(named)
    failures: list[str] = []
    for _, sock in pending:
        sock.setblocking(False)
    while pending and time.monotonic() < deadline:
        readable, _, _ = select.select([sock for _, sock in pending], [], [], 0.1)
        readable_ids = {id(sock) for sock in readable}
        still: list[tuple[str, socket.socket]] = []
        for name, sock in pending:
            if id(sock) not in readable_ids:
                still.append((name, sock))
                continue
            try:
                data = sock.recv(256)
            except BlockingIOError:
                still.append((name, sock))
                continue
            except OSError:
                # The server aborted the connection.
                continue
            if data == b"" or data.startswith(b"HTTP/"):
                continue
            failures.append(f"{name}: unexpected bytes {data!r}")
        pending = still
    failures.extend(f"{name}: still blocked with no response" for name, _ in pending)
    assert not failures, (
        "unfinished requests still occupy worker threads past "
        f"{MAX_HANDLER_WAIT:.0f}s:\n" + "\n".join(failures)
    )


def _track(
    running: _Running, name: str, payload: bytes, named: list[tuple[str, socket.socket]]
) -> socket.socket:
    sock = _connect(running.port, payload)
    running.socks.append(sock)
    named.append((name, sock))
    return sock


def test_unfinished_requests_release_worker_threads():
    """Request-line, header, and body reads must end while the peer stays open."""
    core = _Running(server.Handler)
    score = _Running(synthid_score_server.Handler)
    text = _Running(synthid_text_server.Handler)
    named: list[tuple[str, socket.socket]] = []
    try:
        for path in CORE_GETS:
            _track(core, f"GET {path} request line", _request_line("GET", path), named)
            _track(core, f"GET {path} headers", _partial_headers("GET", path), named)
        for path in CORE_POSTS:
            _track(core, f"POST {path} request line", _request_line("POST", path), named)
            _track(core, f"POST {path} headers", _partial_headers("POST", path), named)
            _track(core, f"POST {path} body", _partial_body(path), named)
        _track(
            core,
            "POST /clean expect-continue body",
            _partial_body("/clean", expect_continue=True),
            named,
        )
        for label, running, posts in (
            ("synthid-score", score, ("/score",)),
            ("synthid-text", text, ("/watermark", "/watermark/batch")),
        ):
            _track(
                running,
                f"{label} GET /health request line",
                _request_line("GET", "/health"),
                named,
            )
            _track(
                running,
                f"{label} GET /health headers",
                _partial_headers("GET", "/health"),
                named,
            )
            for path in posts:
                _track(
                    running,
                    f"{label} POST {path} request line",
                    _request_line("POST", path),
                    named,
                )
                _track(
                    running,
                    f"{label} POST {path} headers",
                    _partial_headers("POST", path),
                    named,
                )
                _track(
                    running,
                    f"{label} POST {path} body",
                    _partial_body(path),
                    named,
                )
        _assert_all_released(named)
    finally:
        core.stop()
        score.stop()
        text.stop()


def test_missing_bearer_token_does_not_prevent_header_stall():
    """The token is checked only after the request line and headers are read."""
    previous_key = server.API_KEY
    server.API_KEY = PLACEHOLDER_API_KEY
    core = _Running(server.Handler)
    named: list[tuple[str, socket.socket]] = []
    try:
        for path in (*CORE_GETS, *CORE_POSTS):
            method = "GET" if path in CORE_GETS else "POST"
            _track(
                core,
                f"{method} {path} request line without bearer token",
                _request_line(method, path),
                named,
            )
            _track(
                core,
                f"{method} {path} headers without bearer token",
                _partial_headers(method, path),
                named,
            )
        _assert_all_released(named)
    finally:
        server.API_KEY = previous_key
        core.stop()


def test_slow_byte_drip_releases_worker_threads():
    """Bytes that never finish the read must not refresh the wait forever."""
    core = _Running(server.Handler)
    score = _Running(synthid_score_server.Handler)
    text = _Running(synthid_text_server.Handler)
    stop = threading.Event()
    drippers: list[threading.Thread] = []
    named: list[tuple[str, socket.socket]] = []
    try:
        cases = (
            ("core request line", core, b""),
            ("core headers", core, _partial_headers("GET", "/openapi.json")),
            ("core inspect body", core, _partial_body("/inspect", length=100000)),
            ("core detect body", core, _partial_body("/detect", length=100000)),
            ("core clean body", core, _partial_body("/clean", length=100000)),
            ("core watermark body", core, _partial_body("/watermark", length=100000)),
            ("core inspect batch body", core, _partial_body("/inspect/batch", length=100000)),
            ("core detect batch body", core, _partial_body("/detect/batch", length=100000)),
            ("core clean batch body", core, _partial_body("/clean/batch", length=100000)),
            ("core watermark batch body", core, _partial_body("/watermark/batch", length=100000)),
            ("score body", score, _partial_body("/score", length=100000)),
            ("text watermark body", text, _partial_body("/watermark", length=100000)),
            ("text batch body", text, _partial_body("/watermark/batch", length=100000)),
        )
        for name, running, prelude in cases:
            sock = _track(running, name, prelude, named)
            drippers.append(_drip(sock, stop))
        _assert_all_released(named)
    finally:
        stop.set()
        for dripper in drippers:
            dripper.join(timeout=1)
        core.stop()
        score.stop()
        text.stop()


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _spawn_server(script: Path, port: int, extra_args: list[str] | None = None) -> subprocess.Popen:
    """Run a service entrypoint with a small open-file limit."""
    extra = extra_args or []
    launcher = (
        "import os, resource, sys\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
        f"resource.setrlimit(resource.RLIMIT_NOFILE, ({SERVER_NOFILE}, hard))\n"
        "os.execv(sys.executable, [sys.executable, sys.argv[1], "
        "'--host', '127.0.0.1', '--port', sys.argv[2], *sys.argv[3:]])\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", launcher, str(script), str(port), *extra],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _probe_health(port: int, headers: dict[str, str] | None, timeout: float) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", "/health", headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def _wait_until_listening(
    port: int, proc: subprocess.Popen, headers: dict[str, str] | None = None
) -> None:
    deadline = time.monotonic() + 20
    last_error = "server did not start"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"server exited before listening (code {proc.returncode})")
        try:
            status, _body = _probe_health(port, headers, 0.5)
        except (TimeoutError, OSError) as exc:
            last_error = str(exc)
        else:
            if status == 200 or status in (401, 503):
                return
            last_error = f"HTTP {status}"
        time.sleep(0.05)
    pytest.fail(last_error)


def _health_while_held(port: int, headers: dict[str, str] | None = None) -> str | None:
    """Return None if /health completes before the stall budget runs out."""
    deadline = time.monotonic() + OBSERVE_SECONDS
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = _probe_health(port, headers, 0.5)
        except (TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if status == 200 and b'"ok"' in body:
                return None
            last = f"status {status} body {body[:80]!r}"
        time.sleep(0.05)
    return last


def _fill_until_saturated(port: int, payload: bytes) -> list[socket.socket]:
    socks = []
    for _ in range(SERVER_NOFILE + 16):
        try:
            socks.append(_connect(port, payload))
        except OSError:
            break
    return socks


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def test_repeated_stalls_do_not_block_health():
    """Holding unfinished connections open must not keep /health down."""
    pytest.importorskip("resource")
    failures: list[str] = []

    port = _free_port()
    proc = _spawn_server(SCRIPTS / "server.py", port)
    stalls: list[socket.socket] = []
    try:
        _wait_until_listening(port, proc)
        # Incomplete headers block before authorization and before routing, so
        # this shape covers every GET and POST. The listening process is the
        # real server.py entrypoint.
        stalls.extend(_fill_until_saturated(port, _partial_headers("POST", "/clean")))
        if len(stalls) < 8:
            failures.append(
                f"could not open enough stalled connections to fill the server ({len(stalls)})"
            )
        reason = _health_while_held(port)
        if reason is not None:
            failures.append(
                f"GET /health did not complete within {OBSERVE_SECONDS:.0f}s while "
                f"{len(stalls)} unfinished connections were held open: {reason}"
            )
    finally:
        for sock in stalls:
            sock.close()
        _kill(proc)

    # Same exhaustion with a required bearer token. The stalled clients never
    # send the token; the health probe does.
    port = _free_port()
    proc = _spawn_server(
        SCRIPTS / "server.py",
        port,
        ["--api-key", PLACEHOLDER_API_KEY],
    )
    stalls = []
    headers = {"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"}
    try:
        _wait_until_listening(port, proc, headers)
        denied = _connect(
            port,
            b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        try:
            denied.settimeout(2)
            denied_bytes = denied.recv(128)
        finally:
            denied.close()
        if not (
            denied_bytes.startswith(b"HTTP/1.0 401") or denied_bytes.startswith(b"HTTP/1.1 401")
        ):
            failures.append(f"finished request without a token was not refused: {denied_bytes!r}")
        stalls.extend(_fill_until_saturated(port, _partial_headers("GET", "/health")))
        stalls.extend(_fill_until_saturated(port, _partial_body("/inspect")))
        if len(stalls) < 8:
            failures.append(
                "could not open enough unauthenticated stalled connections "
                f"to fill the server ({len(stalls)})"
            )
        reason = _health_while_held(port, headers)
        if reason is not None:
            failures.append(
                "GET /health with the configured bearer token did not complete within "
                f"{OBSERVE_SECONDS:.0f}s while unfinished, unauthenticated connections "
                f"were held open: {reason}"
            )
    finally:
        for sock in stalls:
            sock.close()
        _kill(proc)

    assert not failures, "\n".join(failures)


def test_sidecar_stalls_do_not_block_their_health():
    """The score and text sidecars share the unbounded read."""
    pytest.importorskip("resource")
    cases = (
        (SCRIPTS / "synthid_score_server.py", _partial_body("/score")),
        (SCRIPTS / "synthid_text_server.py", _partial_headers("POST", "/watermark")),
    )
    failures: list[str] = []
    for script, payload in cases:
        port = _free_port()
        proc = _spawn_server(script, port)
        stalls: list[socket.socket] = []
        try:
            _wait_until_listening(port, proc)
            stalls.extend(_fill_until_saturated(port, payload))
            if len(stalls) < 8:
                failures.append(
                    f"{script.name}: could not open enough stalled connections ({len(stalls)})"
                )
            reason = _health_while_held(port)
            if reason is not None:
                failures.append(
                    f"{script.name} GET /health did not complete within "
                    f"{OBSERVE_SECONDS:.0f}s while {len(stalls)} unfinished connections "
                    f"were held open: {reason}"
                )
        finally:
            for sock in stalls:
                sock.close()
            _kill(proc)
    assert not failures, "\n".join(failures)
