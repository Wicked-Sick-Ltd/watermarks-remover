"""c2patool must never reach the network while inspecting an uploaded file.

c2pa-rs follows the XMP `dcterms:provenance` URL of an asset that carries no
embedded manifest (`verify.remote_manifest_fetch` defaults to true), so without
an offline settings file an uploaded JPEG or PDF makes the service send a GET to
any host and port the uploader names, and the HTTP status or connection error
comes back in `tools.c2patool.snippet`.
"""

from __future__ import annotations

import base64
import http.client
import json
import shutil
import socket
import struct
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import image_meta
import server
from test_pdf_deep_images import _minimal_pdf


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _xmp_with_provenance(url: str) -> bytes:
    return (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dcterms="http://purl.org/dc/terms/" '
        f'dcterms:provenance="{url}"/>'
        "</rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    ).encode()


def _jpeg_with_provenance(url: str) -> bytes:
    """A 1x1 JPEG with no JUMBF box and an XMP APP1 naming a remote manifest."""
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + _xmp_with_provenance(url)
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return (
        b"\xff\xd8"
        + app1
        + b"\xff\xdb\x00\x43\x00"
        + b"\x00" * 64
        + b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xd9"
    )


def _pdf_with_provenance(url: str) -> bytes:
    return _minimal_pdf(xmp=_xmp_with_provenance(url))


def test_c2patool_runs_with_network_lookups_disabled(monkeypatch, tmp_path):
    """Every c2patool run is pinned to a settings file that turns the network off.

    c2patool silently falls back to its network-enabled defaults when
    `--settings` names a missing file, so the file has to exist and carry the
    offline values at the moment the tool runs.
    """
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        settings = Path(cmd[cmd.index("--settings") + 1])
        seen["settings_path"] = settings
        seen["settings"] = tomllib.loads(settings.read_text(encoding="utf-8"))
        seen["siblings"] = sorted(p.name for p in settings.parent.iterdir())
        return _Completed(1, stderr="Error: No claim found\n")

    monkeypatch.setattr(
        image_meta, "which", lambda c: "/fake/bin/c2patool" if c == "c2patool" else None
    )
    monkeypatch.setattr(image_meta.subprocess, "run", fake_run)
    path = tmp_path / "x.jpg"
    path.write_bytes(_jpeg_with_provenance("http://127.0.0.1:9/internal/path"))

    entry = image_meta.run_optional_tools(path)["c2patool"]

    assert "--settings" in seen.get("cmd", []), "c2patool ran with its default settings"
    assert seen["cmd"][-1] == str(path)
    settings = seen["settings"]
    assert settings["verify"]["remote_manifest_fetch"] is False
    assert settings["verify"]["ocsp_fetch"] is False
    # An explicit empty allow list makes the SDK resolver refuse every URL,
    # which also covers did:web, OCSP and timestamp lookups.
    assert settings["core"]["allowed_network_hosts"] == []
    # c2patool loads trust-list sidecars from the settings file's directory.
    assert seen["siblings"] == [seen["settings_path"].name]
    assert not seen["settings_path"].exists(), "settings file left behind"
    assert entry["ok"] is True and entry["has_manifest"] is False


def test_c2patool_is_not_run_without_its_settings_file(monkeypatch, tmp_path):
    """If the offline settings cannot be written, c2patool must not run at all."""
    calls: list = []

    def broken_tempdir(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(
        image_meta, "which", lambda c: "/fake/bin/c2patool" if c == "c2patool" else None
    )
    monkeypatch.setattr(image_meta.tempfile, "TemporaryDirectory", broken_tempdir)
    monkeypatch.setattr(
        image_meta.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Completed(0)
    )
    path = tmp_path / "x.jpg"
    path.write_bytes(_jpeg_with_provenance("http://127.0.0.1:9/internal/path"))

    entry = image_meta.run_optional_tools(path)["c2patool"]

    assert calls == []
    assert entry["ok"] is False
    assert entry["has_manifest"] is False


def _real_c2patool() -> str | None:
    exe = shutil.which("c2patool")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if r.returncode == 0 else None


needs_c2patool = pytest.mark.skipif(_real_c2patool() is None, reason="c2patool not installed")


class _Listener:
    """A loopback TCP listener standing in for an internal service."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.requests: list[bytes] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self.sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(2)
                try:
                    self.requests.append(conn.recv(4096))
                    conn.sendall(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                except OSError:
                    pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.sock.close()


@pytest.fixture
def listener():
    lst = _Listener()
    yield lst
    lst.close()


@pytest.fixture(scope="module")
def conn():
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=120)
    yield c
    c.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _inspect(conn, data: bytes, name: str) -> dict:
    conn.request(
        "POST",
        "/inspect",
        body=json.dumps({"file": base64.b64encode(data).decode("ascii"), "name": name}).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    body = json.loads(resp.read())
    assert resp.status == 200, body
    return body


@needs_c2patool
@pytest.mark.parametrize(
    ("name", "build"),
    [("x.jpg", _jpeg_with_provenance), ("x.pdf", _pdf_with_provenance)],
)
def test_inspect_upload_does_not_fetch_xmp_provenance_url(conn, listener, name, build):
    """POST /inspect with an XMP-declared remote manifest makes no outbound request."""
    url = f"http://127.0.0.1:{listener.port}/internal/path"

    body = _inspect(conn, build(url), name)

    ct = body["report"]["tools"]["c2patool"]
    assert ct["available"] is True
    assert listener.requests == [], (
        f"c2patool fetched the uploaded file's provenance URL: {listener.requests!r}; "
        f"snippet={ct.get('snippet')!r}"
    )
    assert "http status" not in (ct.get("snippet") or "").lower()
