"""Bounded socket I/O for the stdlib HTTP services.

``http.server`` reads the request line, headers, and body with blocking
socket calls and no timeout, so one peer that withholds bytes keeps a worker
thread (and a file descriptor) forever, and ThreadingHTTPServer accepts
without limit. server.py and the SynthID sidecars share this module:

* ``GuardedRequestHandler`` paces every read and write on the connection.
  Each direction must move ``MIN_TRANSFER_RATE * IO_WINDOW_SECONDS`` bytes
  within every ``IO_WINDOW_SECONDS`` window, or finish inside it. The request
  window opens at accept and the response window at its first byte; a byte
  does not reopen the window, so a slow drip is cut off like silence.
* ``ThreadingHTTPServer`` caps concurrent connections below the process
  file-descriptor limit. At the cap, a new connection evicts the connection
  still sending its request at the lowest recent rate; if every slot is past
  its request, the newcomer gets a 503.

Stdlib only: the sidecar images copy this file next to their server script.
"""

from __future__ import annotations

import contextlib
import http.server
import io
import os
import socket
import threading
import time

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

IO_WINDOW_SECONDS = float(os.environ.get("WATERMARKS_IO_WINDOW_SECONDS", "4"))
MIN_TRANSFER_RATE = float(os.environ.get("WATERMARKS_MIN_TRANSFER_RATE", str(16 << 10)))
MAX_CONNECTIONS = int(os.environ.get("WATERMARKS_MAX_CONNECTIONS", "64"))
# Descriptors kept free for listening, accepting a connection to refuse it,
# and the temp files / subprocess pipes a handler opens.
_FD_HEADROOM = 8

_BUSY_BODY = b'{"ok": false, "error": "server busy"}'
_BUSY_RESPONSE = (
    b"HTTP/1.0 503 Service Unavailable\r\n"
    b"Content-Type: application/json; charset=utf-8\r\n"
    b"Content-Length: " + str(len(_BUSY_BODY)).encode() + b"\r\n"
    b"Connection: close\r\n\r\n" + _BUSY_BODY
)


class _Pace:
    """Byte accounting for one direction of a connection."""

    def __init__(self, started: float | None) -> None:
        self.window_start = started
        self.window_bytes = 0
        self._prev_start = started
        self._prev_bytes = 0

    def remaining(self, what: str) -> float:
        if self.window_start is None:
            self.window_start = self._prev_start = time.monotonic()
        remaining = self.window_start + IO_WINDOW_SECONDS - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{what} stalled below the minimum transfer rate")
        return remaining

    def advance(self, n: int) -> None:
        self.window_bytes += n
        if self.window_bytes >= MIN_TRANSFER_RATE * IO_WINDOW_SECONDS:
            self._prev_start, self._prev_bytes = self.window_start, self.window_bytes
            self.window_start = time.monotonic()
            self.window_bytes = 0

    def restart(self) -> None:
        self.window_start = self._prev_start = time.monotonic()
        self.window_bytes = self._prev_bytes = 0

    def recent_rate(self) -> float:
        """Bytes per second over the current and previous window."""
        elapsed = time.monotonic() - (self._prev_start or 0.0)
        return (self._prev_bytes + self.window_bytes) / max(elapsed, 1e-3)


class _Connection:
    """Per-connection transfer state, shared by the handler and server."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.inbound = _Pace(time.monotonic())
        self.outbound = _Pace(None)
        self.reading = True
        self.evicted = False
        self._lock = threading.Lock()

    def finish_reading(self) -> None:
        with self._lock:
            if self.evicted:
                raise TimeoutError("connection evicted while sending its request")
            self.reading = False

    def try_evict(self) -> bool:
        with self._lock:
            if not self.reading or self.evicted:
                return False
            self.evicted = True
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        return True


class _PacedReader(io.RawIOBase):
    def __init__(self, conn: _Connection) -> None:
        self._conn = conn

    def readable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._conn.sock.fileno()

    def restart(self) -> None:
        """Open a fresh window, e.g. before the next request on the same socket."""
        self._conn.inbound.restart()

    def readinto(self, buffer) -> int:  # type: ignore[override]
        conn = self._conn
        if conn.evicted:
            raise TimeoutError("connection evicted while sending its request")
        conn.sock.settimeout(conn.inbound.remaining("request"))
        n = conn.sock.recv_into(buffer)
        if n == 0 and conn.evicted:
            raise TimeoutError("connection evicted while sending its request")
        conn.inbound.advance(n)
        return n


class _PacedWriter(io.BufferedIOBase):
    def __init__(self, conn: _Connection) -> None:
        self._conn = conn

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._conn.sock.fileno()

    def write(self, data) -> int:  # type: ignore[override]
        conn = self._conn
        if conn.reading:
            conn.finish_reading()
        with memoryview(data) as view, view.cast("B") as raw:
            total = len(raw)
            offset = 0
            while offset < total:
                conn.sock.settimeout(conn.outbound.remaining("response"))
                n = conn.sock.send(raw[offset:])
                offset += n
                conn.outbound.advance(n)
        return total


class GuardedRequestHandler(http.server.BaseHTTPRequestHandler):
    """BaseHTTPRequestHandler whose socket reads and writes cannot block forever.

    Subclasses read a request body with ``read_body`` so the connection stops
    counting as an unfinished request once the body is in.
    """

    def setup(self) -> None:
        self.connection = self.request
        register = getattr(self.server, "guard_connection", None)
        self._guard = register(self.request) if register else _Connection(self.request)
        self.rfile = io.BufferedReader(_PacedReader(self._guard))
        self.wfile = _PacedWriter(self._guard)

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        raw = (self.headers.get("Content-Length") or "").strip()
        if not raw.isdigit() or int(raw) == 0:
            self._guard.finish_reading()
        return True

    def read_body(self, length: int) -> bytes:
        data = self.rfile.read(length)
        self._guard.finish_reading()
        return data


def paced_reader(sock: socket.socket) -> io.BufferedReader:
    """Buffered reader on *sock* held to the request pacing, window opening now.

    Reads raise TimeoutError once the peer falls behind; ``reader.raw.restart()``
    opens a new window for the next request on the same socket.
    """
    return io.BufferedReader(_PacedReader(_Connection(sock)))


def _connection_cap() -> int:
    if resource is None:
        return MAX_CONNECTIONS
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return MAX_CONNECTIONS
    try:
        in_use = len(os.listdir("/proc/self/fd"))
    except OSError:
        in_use = 16
    # Evicted connections hold their descriptor until their thread closes it,
    # so the hard ceiling below is twice the cap.
    return max(1, min(MAX_CONNECTIONS, (soft - in_use - _FD_HEADROOM) // 2))


class ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """ThreadingHTTPServer with a bounded number of live connections."""

    request_queue_size = 128

    def __init__(self, *args, **kwargs) -> None:
        self._guard_lock = threading.Lock()
        self._guard_conns: dict[socket.socket, _Connection] = {}
        super().__init__(*args, **kwargs)
        self.max_connections = _connection_cap()

    def guard_connection(self, sock: socket.socket) -> _Connection:
        with self._guard_lock:
            conn = self._guard_conns.get(sock)
            if conn is None:
                conn = self._guard_conns[sock] = _Connection(sock)
            return conn

    def verify_request(self, request, client_address) -> bool:
        with self._guard_lock:
            live = [c for c in self._guard_conns.values() if not c.evicted]
            admit = len(self._guard_conns) < 2 * self.max_connections
            if admit and len(live) >= self.max_connections:
                candidates = sorted(
                    (c for c in live if c.reading), key=lambda c: c.inbound.recent_rate()
                )
                admit = any(c.try_evict() for c in candidates)
            if admit:
                self._guard_conns[request] = _Connection(request)
                return True
        with contextlib.suppress(OSError):
            request.setblocking(False)
            request.send(_BUSY_RESPONSE)
        return False

    def close_request(self, request) -> None:
        try:
            super().close_request(request)
        finally:
            with self._guard_lock:
                self._guard_conns.pop(request, None)
