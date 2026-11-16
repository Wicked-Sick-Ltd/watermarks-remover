"""Report lists stay bounded when a file repeats a flagged element.

Inspectors and strippers record one finding/action per C2PA-like chunk,
segment, box or tag. The element count is attacker-controlled (a 12-byte
empty PNG chunk is enough), so without a cap one upload turns into a
findings list -- and then a JSON response -- many times its own size.
"""

from __future__ import annotations

import base64
import copy
import http.client
import json
import struct
import sys
import threading
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import av_meta
import container_meta
import image_meta
import server

REPEATS = 20_000
# Well above the service cap, far below REPEATS: a bounded report passes, a
# per-element report does not.
BOUND = 1_000

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_chunk(ctype: bytes, payload: bytes = b"") -> bytes:
    crc = zlib.crc32(payload, zlib.crc32(ctype)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _png_many_cabx(n: int = REPEATS) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        PNG_SIG
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"caBX") * n
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00"))
        + _png_chunk(b"IEND")
    )


def _jpeg_many_app11(n: int = REPEATS) -> bytes:
    return b"\xff\xd8" + b"\xff\xeb\x00\x02" * n + b"\xff\xd9"


def _webp_many_c2pa(n: int = REPEATS) -> bytes:
    body = b"WEBP" + b"VP8L" + struct.pack("<I", 0) + (b"C2PA" + struct.pack("<I", 0)) * n
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _avif_many_jumb(n: int = REPEATS) -> bytes:
    ftyp = struct.pack(">I", 20) + b"ftyp" + b"avif" + b"\x00\x00\x00\x00" + b"avif"
    return ftyp + (struct.pack(">I", 8) + b"jumb") * n


def _gif_many_c2pa_app_ext(n: int = REPEATS) -> bytes:
    header = b"GIF89a" + struct.pack("<HHBBB", 1, 1, 0, 0, 0)
    ext = b"\x21\xff\x0b" + b"c2pa0000000" + b"\x00"
    return header + ext * n + b"\x3b"


def _wav_many_c2pa(n: int = REPEATS) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 8000, 1, 8)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + (b"C2PA" + struct.pack("<I", 0)) * n
        + b"data"
        + struct.pack("<I", 2)
        + b"\x80\x80"
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _html_many_data_ai(n: int = REPEATS) -> str:
    return "<html><body>" + '<p data-ai="1">x</p>' * n + "</body></html>"


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


def _post(conn, path: str, payload: dict) -> tuple[int, bytes]:
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_inspect_endpoint_png_with_many_cabx_chunks_returns_bounded_report(conn):
    data = _png_many_cabx()
    status, raw = _post(
        conn, "/inspect", {"file": base64.b64encode(data).decode("ascii"), "name": "x.png"}
    )
    assert status == 200
    body = json.loads(raw)
    report = body["report"]
    assert report["format"] == "png"
    assert report["has_c2pa"] is True
    assert len(report["findings"]) < BOUND
    assert len(report["findings_confidence"]) == len(report["findings"])
    assert report["findings"][0] == "PNG chunk caBX (possible C2PA container)"
    assert any("omitted" in f for f in report["findings"])
    # The response must not scale with the number of repeated chunks.
    assert len(raw) < len(data)


def test_clean_endpoint_png_with_many_cabx_chunks_drops_all_and_bounds_actions(conn):
    data = _png_many_cabx()
    status, raw = _post(
        conn, "/clean", {"file": base64.b64encode(data).decode("ascii"), "name": "x.png"}
    )
    assert status == 200, raw[:500]
    body = json.loads(raw)
    cleaned = base64.b64decode(body["cleaned"])
    assert b"caBX" not in cleaned
    assert len(body["report"]["actions"]) < BOUND


def test_strip_png_drops_every_cabx_chunk_with_bounded_actions():
    cleaned, actions = image_meta.strip_png(_png_many_cabx())
    assert b"caBX" not in cleaned
    assert cleaned.endswith(_png_chunk(b"IEND"))
    assert len(actions) < BOUND
    assert actions[0] == "drop chunk caBX"
    assert any("omitted" in a for a in actions)


@pytest.mark.parametrize(
    "inspect, data",
    [
        (image_meta.inspect_png, _png_many_cabx()),
        (image_meta.inspect_jpeg, _jpeg_many_app11()),
        (image_meta.inspect_webp, _webp_many_c2pa()),
        (lambda d: image_meta.inspect_isobmff(d, "avif"), _avif_many_jumb()),
        (image_meta.inspect_gif, _gif_many_c2pa_app_ext()),
        (av_meta._inspect_wav, _wav_many_c2pa()),
    ],
    ids=["png", "jpeg", "webp", "avif", "gif", "wav"],
)
def test_binary_inspectors_bound_findings(inspect, data):
    has_c2pa, _has_ai, findings = inspect(data)
    assert has_c2pa is True
    assert len(findings) < BOUND


@pytest.mark.parametrize(
    "strip, data, marker",
    [
        (image_meta.strip_jpeg, _jpeg_many_app11(), b"\xff\xeb"),
        (image_meta.strip_webp, _webp_many_c2pa(), b"C2PA"),
        (lambda d: image_meta.strip_isobmff(d, "avif"), _avif_many_jumb(), b"jumb"),
        (image_meta.strip_gif, _gif_many_c2pa_app_ext(), b"c2pa"),
    ],
    ids=["jpeg", "webp", "avif", "gif"],
)
def test_binary_strippers_bound_actions(strip, data, marker):
    cleaned, actions = strip(data)
    assert marker not in cleaned
    assert len(actions) < BOUND


def test_wav_strip_bounds_actions():
    cleaned, actions = av_meta._strip_wav(_wav_many_c2pa(), strip_all_metadata=True)
    assert b"C2PA" not in cleaned
    assert len(actions) < BOUND


def test_html_inspect_bounds_findings():
    _has_c2pa, has_ai, findings, _extra = container_meta.inspect_html(_html_many_data_ai())
    assert has_ai is True
    assert len(findings) < BOUND


def test_inspect_image_keeps_tool_findings_after_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(
        image_meta,
        "run_optional_tools",
        lambda *_a, **_k: {"c2patool": {"has_manifest": True}},
    )
    data = _png_many_cabx()
    path = tmp_path / "x.png"
    path.write_bytes(data)
    report = image_meta.inspect_image(path, data=data, run_synthid=False)
    assert "c2patool reports a C2PA-related manifest" in report.findings
    assert len(report.findings) < BOUND


def test_report_list_cap_counts_omitted_entries_and_survives_copies():
    from common import ReportList

    entries = ReportList(cap=3)
    for i in range(5):
        entries.append(f"e{i}")
    entries.extend(["e5", "e6"])
    entries += ["e7"]
    entries.insert(0, "e8")
    assert entries[:3] == ["e0", "e1", "e2"]
    assert len(entries) == 4
    assert entries[-1] == "no further entries listed: 6 omitted (report capped at 3)"
    for clone in (copy.copy(entries), copy.deepcopy(entries)):
        assert list(clone) == list(entries)
        clone.append("more")
        assert len(clone) == 4
        assert clone[-1] == "no further entries listed: 7 omitted (report capped at 3)"
    assert json.loads(json.dumps(entries)) == list(entries)


def test_report_list_keeps_first_entry_of_each_new_key_past_the_cap():
    from common import ReportList

    entries = ReportList(cap=2, key=lambda s: s[0])
    entries.extend(["a1", "a2", "a3", "b1", "a4", "b2", "c1"])
    assert entries == [
        "a1",
        "a2",
        "b1",
        "c1",
        "no further entries listed: 3 omitted (report capped at 2)",
    ]
    clone = copy.deepcopy(entries)
    clone.extend(["b3", "d1"])
    assert clone[-2:] == ["d1", "no further entries listed: 4 omitted (report capped at 2)"]


def test_capped_findings_keep_late_probable_finding_actionable(tmp_path):
    import audit_lib

    page = (
        "<html><head>"
        + '<meta name="generator" content="WordPress 6.5">' * 500
        + '<meta name="ai-generator" content="ChatGPT">'
        + "</head><body>x</body></html>"
    )
    path = tmp_path / "page.html"
    path.write_text(page, encoding="utf-8")
    item = audit_lib.scan_file(path)
    assert len(item["findings"]) < BOUND
    assert "probable" in item["confidence"]
    assert audit_lib.is_actionable(item) is True


def test_capped_markdown_findings_keep_late_c2pa_hint():
    text = "".join(f"<!-- generated by ChatGPT {i} -->\n" for i in range(500))
    text += "<!-- c2pa manifest generated by ChatGPT -->\n"
    has_c2pa, has_ai, findings, _details = container_meta.inspect_markdown(text)
    assert has_ai is True
    assert len(findings) < BOUND
    assert has_c2pa is True


def test_capped_actions_keep_late_mutating_action():
    from common import actions_list, result_has_changes

    actions = actions_list()
    for i in range(500):
        actions.append(f"kept {i} bytes of truncated tail (file truncated)")
    actions.append("drop chunk caBX")
    assert len(actions) < BOUND
    assert "drop chunk caBX" in actions
    assert result_has_changes({"actions": list(actions)}) is True
    overflowed = actions_list()
    for i in range(500):
        overflowed.append(f"kept {i} bytes")
    assert result_has_changes({"actions": list(overflowed)}) is False
