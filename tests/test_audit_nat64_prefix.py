"""Well-known NAT64 targets must be refused before the audit client connects.

``audit_website`` keeps a resolved address when ``ipaddress`` reports
``is_global``. On Python 3.12 that is true for the well-known NAT64 prefix
``64:ff9b::/96`` (the local-use prefix ``64:ff9b:1::/48`` is already private).
A name the operator audits can therefore resolve to an embedded link-local,
loopback, or private IPv4 address, and the client will open a TCP connection
and print the response body in the audit report.

These tests fail while that prefix is accepted. They pass once every resolve,
including a later lookup of the same name, is rejected before ``connect``.
"""

from __future__ import annotations

import contextlib
import errno
import ipaddress
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service" / "scripts"))

import audit_website

# RFC 6052: 64:ff9b::/96 carries the IPv4 address in the low 32 bits.
_WK_PREFIX = int(ipaddress.IPv6Address("64:ff9b::"))
# 192.0.0.9 is globally reachable per the stdlib exceptions list, and is only
# contacted after this process has assigned it on loopback.
_PUBLIC_V4 = "192.0.0.9"
_SENTINEL = "nat64-audit-marker"
_EMBEDDED_V4 = (
    "169.254.169.254",
    "127.0.0.1",
    "10.1.2.3",
    "172.16.5.5",
    "192.168.1.50",
    str(ipaddress.IPv4Address(0)),
    "100.64.1.1",
    "1.1.1.1",
)
_IP = shutil.which("ip") or "ip"
_SUDO = shutil.which("sudo") or "sudo"
_TEE = shutil.which("tee") or "tee"
_ADDED: list[tuple[str, str]] = []


def _wk(ipv4: str) -> str:
    embedded = int(ipaddress.IPv4Address(ipv4))
    return str(ipaddress.IPv6Address(_WK_PREFIX | embedded))


def _local_use(ipv4: str) -> str:
    prefix = int(ipaddress.IPv6Address("64:ff9b:1::"))
    return str(ipaddress.IPv6Address(prefix | int(ipaddress.IPv4Address(ipv4))))


def _is_ip(host: object) -> bool:
    text = str(host).split("%", 1)[0]
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


def _norm(host: object) -> str:
    text = str(host).split("%", 1)[0]
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return text


def _family(ip: str) -> int:
    version = ipaddress.ip_address(ip).version
    return socket.AF_INET6 if version == 6 else socket.AF_INET


def _bindable(ip: str) -> bool:
    family = _family(ip)
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((ip, 0))
    except OSError:
        return False
    else:
        return True
    finally:
        sock.close()


def _route_is_local(ip: str) -> bool:
    """True only when the kernel would deliver this address on loopback."""
    flag = "-6" if ":" in ip else "-4"
    proc = subprocess.run(
        [_IP, flag, "route", "get", ip],
        capture_output=True,
        text=True,
        check=False,
    )
    text = proc.stdout
    return proc.returncode == 0 and (" dev lo " in f" {text} " or text.startswith("local "))


def _ensure_local(ip: str) -> bool:
    if _bindable(ip) and _route_is_local(ip):
        return True
    if not sys.platform.startswith("linux"):
        return False
    prefix = "/128" if ":" in ip else "/32"
    added = subprocess.run(
        [_SUDO, "-n", _IP, "addr", "add", f"{ip}{prefix}", "dev", "lo"],
        capture_output=True,
        text=True,
        check=False,
    )
    if added.returncode == 0:
        _ADDED.append((ip, prefix))
    return _bindable(ip) and _route_is_local(ip)


@pytest.fixture(scope="module", autouse=True)
def _assigned_loopback_addresses():
    needed = [_wk(item) for item in _EMBEDDED_V4]
    needed.append(_PUBLIC_V4)
    for ip in needed:
        _ensure_local(ip)
    yield
    for ip, prefix in _ADDED:
        subprocess.run(
            [_SUDO, "-n", _IP, "addr", "del", f"{ip}{prefix}", "dev", "lo"],
            capture_output=True,
            text=True,
            check=False,
        )
    _ADDED.clear()


def _page() -> bytes:
    html = (
        '<html><head><meta name="generator" content="'
        + _SENTINEL
        + '"></head><body>ok</body></html>'
    )
    return html.encode()


def _sitemap(loc: str) -> bytes:
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{loc}</loc></url></urlset>"
    )
    return xml.encode()


def _response(status: int, body: bytes, headers: dict[str, str]) -> bytes:
    reason = {200: "OK", 302: "Found", 404: "Not Found"}.get(status, "OK")
    fields = {"Content-Length": str(len(body)), "Connection": "close", **headers}
    head = "\r\n".join(
        [f"HTTP/1.1 {status} {reason}", *(f"{key}: {value}" for key, value in fields.items())]
    )
    return (head + "\r\n\r\n").encode() + body


def _read_path(conn: socket.socket) -> str:
    conn.settimeout(2)
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    line = data.split(b"\r\n", 1)[0]
    parts = line.split(b" ")
    if len(parts) < 2:
        return "/"
    return parts[1].decode("ascii", "replace").split("?", 1)[0]


class _HttpServer:
    def __init__(self, ip: str, port: int | None = None):
        self.ip = ip
        self.requests: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._routes: dict[str, tuple[int, bytes, dict[str, str]]] = {}
        self._default: tuple[int, bytes, dict[str, str]] | None = None
        family = _family(ip)
        self._sock = socket.socket(family, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            self._sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        self._sock.bind((ip, 0 if port is None else port))
        self.port = int(self._sock.getsockname()[1])
        self._sock.listen(32)
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def route(self, path: str, status: int, body: bytes, headers: dict[str, str]) -> None:
        self._routes[path] = (status, body, headers)

    def default(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self._default = (status, body, headers)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                try:
                    path = _read_path(conn)
                except OSError:
                    continue
                with self._lock:
                    self.requests.append(path)
                    spec = self._routes.get(path, self._default)
                if spec is None:
                    spec = (404, b"", {})
                status, body, headers = spec
                try:
                    conn.sendall(_response(status, body, headers))
                except OSError:
                    continue

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=2)


@contextlib.contextmanager
def _listening(ip: str, port: int | None = None):
    if not (_bindable(ip) and _route_is_local(ip)):
        yield None
        return
    server = _HttpServer(ip, port)
    try:
        yield server
    finally:
        server.close()


def _guard(monkeypatch: pytest.MonkeyPatch, allowed: set[str]) -> list[str]:
    """Record connect destinations. Complete the handshake only on loopback."""
    attempts: list[str] = []
    real_connect = socket.socket.connect
    allowed_norm = {_norm(ip) for ip in allowed}

    def connect(self: socket.socket, address: object) -> None:
        if not isinstance(address, tuple) or not address:
            raise OSError(errno.ENETUNREACH, "blocked by test")
        ip = _norm(address[0])
        attempts.append(ip)
        if ip not in allowed_norm or not _route_is_local(ip):
            raise OSError(errno.ENETUNREACH, "blocked by test")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def _fetch(url: str) -> tuple[bytes | None, BaseException | None]:
    try:
        body, _content_type = audit_website.fetch(url, 2, 1 << 20)
    except Exception as exc:
        return None, exc
    return body, None


def _assert_blocked(
    attempts: list[str],
    error: BaseException | None,
    body: bytes | None,
    server: _HttpServer | None,
    forbidden: set[str],
    report: str = "",
) -> None:
    forbidden_norm = {_norm(ip) for ip in forbidden}
    hit = [ip for ip in attempts if ip in forbidden_norm]
    leaked = _SENTINEL in report or (body is not None and _SENTINEL.encode() in body)
    detail = (
        f"attempts={attempts} error={type(error).__name__ if error else None}: {error} "
        f"requests={None if server is None else server.requests} leaked={leaked}"
    )
    if leaked and report:
        detail = f"{detail}\n{report}"
    assert not hit, detail
    assert not leaked, detail
    assert isinstance(error, ValueError), detail
    if server is not None:
        assert server.requests == [], detail


def _bracket(ip: str, port: int, path: str) -> str:
    return f"http://[{ip}]:{port}{path}"


@pytest.mark.parametrize("embedded", _EMBEDDED_V4)
def test_literal_well_known_nat64_is_refused_before_connect(embedded, monkeypatch):
    nat64 = _wk(embedded)
    with _listening(nat64) as server:
        if server is None:
            pytest.fail(f"{nat64} is not a loopback address; refusing to connect outward")
        server.default(200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {nat64})
        url = _bracket(nat64, server.port, "/page.html")
        body, error = _fetch(url)
        _assert_blocked(attempts, error, body, server, {nat64, embedded})


def test_same_origin_redirect_is_refused_before_connect(monkeypatch):
    nat64 = _wk("169.254.169.254")
    with _listening(nat64) as server:
        if server is None:
            pytest.fail(f"{nat64} is not a loopback address; refusing to connect outward")
        server.route(
            "/start",
            302,
            b"",
            {"Location": "/landed.html"},
        )
        server.route("/landed.html", 200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {nat64})
        body, error = _fetch(_bracket(nat64, server.port, "/start"))
        _assert_blocked(attempts, error, body, server, {nat64, "169.254.169.254"})


def test_absolute_same_origin_redirect_is_refused_before_connect(monkeypatch):
    nat64 = _wk("10.1.2.3")
    with _listening(nat64) as server:
        if server is None:
            pytest.fail(f"{nat64} is not a loopback address; refusing to connect outward")
        target = _bracket(nat64, server.port, "/landed.html")
        server.route("/start", 302, b"", {"Location": target})
        server.route("/landed.html", 200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {nat64})
        body, error = _fetch(_bracket(nat64, server.port, "/start"))
        _assert_blocked(attempts, error, body, server, {nat64, "10.1.2.3"})


def _install_hosts(address: str, hostname: str) -> str:
    path = Path("/etc/hosts")
    original = path.read_text()
    line = f"{address} {hostname}"
    if line not in original.splitlines():
        updated = original if original.endswith("\n") or original == "" else original + "\n"
        updated += line + "\n"
        subprocess.run(
            [_SUDO, "-n", _TEE, str(path)],
            input=updated,
            text=True,
            capture_output=True,
            check=True,
        )
    infos = socket.getaddrinfo(hostname, 9, type=socket.SOCK_STREAM)
    got = {_norm(item[4][0]) for item in infos}
    if got != {address}:
        subprocess.run(
            [_SUDO, "-n", _TEE, str(path)],
            input=original,
            text=True,
            capture_output=True,
            check=False,
        )
        pytest.fail(f"resolver returned {got} for {hostname}, expected {address}")
    return original


def _restore_hosts(original: str) -> None:
    subprocess.run(
        [_SUDO, "-n", _TEE, "/etc/hosts"],
        input=original,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture
def nat64_hostname():
    address = _wk("169.254.169.254")
    hostname = "wk-nat64.audit.test"
    if not (_bindable(address) and _route_is_local(address)):
        pytest.fail(f"{address} is not loopback; refusing to publish it in the resolver")
    original = _install_hosts(address, hostname)
    try:
        yield hostname, address
    finally:
        _restore_hosts(original)


def test_dns_well_known_nat64_is_refused_before_connect(nat64_hostname, monkeypatch):
    hostname, address = nat64_hostname
    with _listening(address) as server:
        assert server is not None
        server.default(200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {address})
        # Every lookup, including the one inside socket.create_connection, must
        # still see this hosts entry. Re-resolving the name is not a fix.
        url = f"http://{hostname}:{server.port}/page.html"
        body, error = _fetch(url)
        _assert_blocked(attempts, error, body, server, {address, "169.254.169.254"})


def test_dns_redirect_reresolve_is_refused_before_second_connect(monkeypatch):
    """First answer is global; the next lookup of the same name is NAT64.

    ``socket.create_connection`` resolves again (stdlib socket.py). Numeric
    lookups stay on the real resolver. Only the attacker-controlled name is
    sequenced, and a second lookup must still be rejected before connect.
    """
    nat64 = _wk("169.254.169.254")
    public = _PUBLIC_V4
    if not (_route_is_local(public) and _route_is_local(nat64)):
        pytest.fail("rebinding listeners are not both on loopback")
    hostname = "rebinding.nat64.audit.test"
    port_holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_holder.bind(("127.0.0.1", 0))
    port = int(port_holder.getsockname()[1])
    port_holder.close()
    real_getaddrinfo = socket.getaddrinfo
    lookups = {"hostname": 0}

    def getaddrinfo(host, port_arg, *args, **kwargs):
        if _is_ip(host):
            return real_getaddrinfo(host, port_arg, *args, **kwargs)
        lookups["hostname"] += 1
        chosen = public if lookups["hostname"] == 1 else nat64
        return real_getaddrinfo(chosen, port_arg, *args, **kwargs)

    with _listening(public, port) as public_server, _listening(nat64, port) as nat64_server:
        if public_server is None or nat64_server is None:
            pytest.fail("could not bind the rebinding pair on loopback")
        location = f"http://{hostname}:{port}/landed.html"
        public_server.route("/start", 302, b"", {"Location": location})
        nat64_server.route("/landed.html", 200, _page(), {"Content-Type": "text/html"})
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
        attempts = _guard(monkeypatch, {public, nat64})
        body, error = _fetch(f"http://{hostname}:{port}/start")
        _assert_blocked(
            [ip for ip in attempts if ip != _norm(public)],
            error,
            body,
            nat64_server,
            {nat64, "169.254.169.254"},
        )
        assert public_server.requests == ["/start"]
        assert lookups["hostname"] >= 1


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", argv)
    return audit_website.main()


def test_sitemap_report_omits_nat64_body(monkeypatch, capsys):
    nat64 = _wk("169.254.169.254")
    with _listening(nat64) as server:
        if server is None:
            pytest.fail(f"{nat64} is not loopback; refusing to request it")
        page = _bracket(nat64, server.port, "/page.html")
        sitemap = _bracket(nat64, server.port, "/sitemap.xml")
        server.route("/sitemap.xml", 200, _sitemap(page), {"Content-Type": "application/xml"})
        server.route("/page.html", 200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {nat64})
        rc = _run_main(
            monkeypatch,
            ["audit_website.py", "--sitemap", sitemap, "--timeout", "2"],
        )
        report = capsys.readouterr().out
        _assert_blocked(
            attempts,
            ValueError("refused") if rc == 2 else None,
            None,
            server,
            {nat64},
            report,
        )
        assert _SENTINEL not in report
        assert rc == 2


def test_base_discovery_omits_nat64_body(monkeypatch, capsys):
    nat64 = _wk("192.168.1.50")
    with _listening(nat64) as server:
        if server is None:
            pytest.fail(f"{nat64} is not loopback; refusing to request it")
        base = f"http://[{nat64}]:{server.port}"
        page = f"{base}/page.html"
        server.route(
            "/sitemap.xml",
            200,
            _sitemap(page),
            {"Content-Type": "application/xml"},
        )
        server.route("/page.html", 200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {nat64})
        rc = _run_main(
            monkeypatch,
            ["audit_website.py", "--base", base, "--timeout", "2"],
        )
        report = capsys.readouterr().out
        _assert_blocked(
            attempts,
            ValueError("refused") if rc == 2 else None,
            None,
            server,
            {nat64, "192.168.1.50"},
            report,
        )
        assert _SENTINEL not in report
        assert rc == 2


def test_dns_sitemap_report_omits_nat64_body(nat64_hostname, monkeypatch, capsys):
    hostname, address = nat64_hostname
    with _listening(address) as server:
        assert server is not None
        page = f"http://{hostname}:{server.port}/page.html"
        sitemap = f"http://{hostname}:{server.port}/sitemap.xml"
        server.route("/sitemap.xml", 200, _sitemap(page), {"Content-Type": "application/xml"})
        server.route("/page.html", 200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {address})
        rc = _run_main(
            monkeypatch,
            ["audit_website.py", "--sitemap", sitemap, "--timeout", "2"],
        )
        report = capsys.readouterr().out
        _assert_blocked(
            attempts,
            ValueError("refused") if rc == 2 else None,
            None,
            server,
            {address, "169.254.169.254"},
            report,
        )
        assert _SENTINEL not in report
        assert rc == 2


def test_dns_base_discovery_omits_nat64_body(nat64_hostname, monkeypatch, capsys):
    hostname, address = nat64_hostname
    with _listening(address) as server:
        assert server is not None
        base = f"http://{hostname}:{server.port}"
        page = f"{base}/page.html"
        server.route("/sitemap.xml", 200, _sitemap(page), {"Content-Type": "application/xml"})
        server.route("/page.html", 200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {address})
        rc = _run_main(monkeypatch, ["audit_website.py", "--base", base, "--timeout", "2"])
        report = capsys.readouterr().out
        _assert_blocked(
            attempts,
            ValueError("refused") if rc == 2 else None,
            None,
            server,
            {address, "169.254.169.254"},
            report,
        )
        assert _SENTINEL not in report
        assert rc == 2


def test_local_use_nat64_prefix_never_connects(monkeypatch):
    address = _local_use("169.254.169.254")
    attempts = _guard(monkeypatch, set())
    body, error = _fetch(f"http://[{address}]/blocked")
    _assert_blocked(attempts, error, body, None, {address, "169.254.169.254"})


def test_ipv4_mapped_metadata_never_connects(monkeypatch):
    attempts = _guard(monkeypatch, set())
    body, error = _fetch("http://[::ffff:169.254.169.254]/blocked")
    _assert_blocked(attempts, error, body, None, {"169.254.169.254", "::ffff:169.254.169.254"})


def test_ipv4_metadata_literal_never_connects(monkeypatch):
    attempts = _guard(monkeypatch, set())
    body, error = _fetch("http://169.254.169.254/blocked")
    _assert_blocked(attempts, error, body, None, {"169.254.169.254"})


def test_cross_origin_redirect_does_not_reach_nat64(monkeypatch):
    nat64 = _wk("172.16.5.5")
    public = _PUBLIC_V4
    if not (_route_is_local(public) and _route_is_local(nat64)):
        pytest.fail("cross-origin listeners are not both on loopback")
    with _listening(public) as public_server, _listening(nat64) as nat64_server:
        if public_server is None or nat64_server is None:
            pytest.fail("could not bind cross-origin listeners on loopback")
        target = _bracket(nat64, nat64_server.port, "/secret.html")
        public_server.route("/start", 302, b"", {"Location": target})
        nat64_server.default(200, _page(), {"Content-Type": "text/html"})
        attempts = _guard(monkeypatch, {public, nat64})
        body, error = _fetch(f"http://{public}:{public_server.port}/start")
        _assert_blocked(
            [ip for ip in attempts if ip != _norm(public)],
            error,
            body,
            nat64_server,
            {nat64, "172.16.5.5"},
        )
        assert public_server.requests == ["/start"]
