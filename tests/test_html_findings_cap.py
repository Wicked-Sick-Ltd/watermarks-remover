"""HTML inspection must not record one finding per repeated match.

POST /inspect of a .html (or .htm) body calls inspect_html, which appends a
truncated copy of every data-ai attribute, AI meta tag, provenance comment,
JSON-LD block, and embedded image data URI. ContainerInspectReport.to_dict
then builds a confidence string per finding, and the HTTP handler pretty-prints
both lists. These tests bound that output. They fail while every match is kept.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import sys
import threading
import tracemalloc
import zipfile
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server
from common import classify_finding_confidence
from container_meta import inspect_container, inspect_html

# Match findings recorded for one HTML document. One extra list entry is
# allowed so a fix can append a single truncation note.
HTML_INSPECT_FINDINGS_CAP = 32

# Retained allocations while scanning a document that repeats a short match
# thousands of times. Keeping one string per match is over a megabyte here;
# stopping at the cap retains a few kilobytes.
_PEAK_BYTES = 200_000


def _peak_bytes(text: str) -> tuple[int, bool, list[str]]:
    tracemalloc.start()
    try:
        _c2pa, has_ai, findings, _details = inspect_html(text)
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak, has_ai, findings


def _assert_capped(findings: list[str]) -> None:
    assert len(findings) <= HTML_INSPECT_FINDINGS_CAP + 1


def test_small_html_still_reports_each_data_ai_attribute():
    html = 'data-ai="one" data-ai-model="two" data-ai_src="three"'
    _c2pa, has_ai, findings, _details = inspect_html(html)
    attr = [f for f in findings if f.startswith("attr:")]
    assert has_ai is True
    assert len(attr) == 3
    assert any("one" in f for f in attr)
    assert any("two" in f for f in attr)
    assert any("three" in f for f in attr)


def test_data_ai_finding_count_does_not_grow_with_repetition():
    short = inspect_html('data-ai="x"' * (HTML_INSPECT_FINDINGS_CAP + 1))
    long = inspect_html('data-ai="x"' * (HTML_INSPECT_FINDINGS_CAP * 30))
    assert short[1] is True
    assert long[1] is True
    _assert_capped(short[2])
    _assert_capped(long[2])
    assert len(long[2]) == len(short[2])


def test_inspect_html_stops_before_a_late_data_ai_attribute():
    html = (
        'data-ai="EARLY"'
        + ('data-ai="x"' * HTML_INSPECT_FINDINGS_CAP)
        + 'data-ai="LATE"'
        + ('data-ai="y"' * 1000)
    )
    _c2pa, has_ai, findings, _details = inspect_html(html)
    assert has_ai is True
    _assert_capped(findings)
    assert any("EARLY" in finding for finding in findings)
    assert all("LATE" not in finding for finding in findings)


def test_repeated_data_ai_attributes_do_not_accumulate_findings():
    text = 'data-ai="x"' * 20_000
    peak, has_ai, findings = _peak_bytes(text)
    assert has_ai is True
    assert findings and findings[0].startswith("attr:")
    _assert_capped(findings)
    assert peak < _PEAK_BYTES


def test_repeated_ai_meta_tags_are_capped():
    text = '<meta name="generator" content="claude">' * 8_000
    peak, has_ai, findings = _peak_bytes(text)
    assert has_ai is True
    _assert_capped(findings)
    assert peak < _PEAK_BYTES


def test_repeated_provenance_comments_are_capped():
    text = "<!-- claude -->" * 8_000
    peak, has_ai, findings = _peak_bytes(text)
    assert has_ai is True
    _assert_capped(findings)
    assert peak < _PEAK_BYTES


def test_repeated_jsonld_blocks_are_capped():
    block = '<script type="application/ld+json">{"@type":"DigitalSourceType"}</script>'
    _c2pa, has_ai, findings, _details = inspect_html(block * 400)
    assert has_ai is True
    _assert_capped(findings)


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _embedded_png_html(copies: int) -> str:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    png = (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", b"Comment\x00c2pa test contentcredentials")
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00"))
        + _png_chunk(b"IEND", b"")
    )
    tag = '<img src="data:image/png;base64,' + base64.b64encode(png).decode("ascii") + '">'
    return tag * copies


def test_repeated_embedded_data_uris_are_capped():
    _c2pa, has_ai, findings, _details = inspect_html(_embedded_png_html(80))
    assert has_ai is True
    _assert_capped(findings)


@pytest.mark.parametrize("name", ["x.html", "x.htm"])
def test_inspect_container_html_suffix_caps_serialized_findings(name: str):
    data = ('data-ai="x"' * 2_000).encode("ascii")
    report = inspect_container(Path(name), data=data).to_dict()
    assert report["format"] == "html"
    assert report["has_ai_metadata"] is True
    _assert_capped(report["findings"])
    assert report["findings_confidence"] == [
        classify_finding_confidence(f) for f in report["findings"]
    ]
    assert len(report["findings_confidence"]) == len(report["findings"])


def test_epub_html_member_reuses_the_html_findings_cap():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ch.xhtml", 'data-ai="x"' * 2_000)
    report = inspect_container(Path("book.epub"), data=buf.getvalue()).to_dict()
    assert report["format"] == "epub"
    assert report["has_ai_metadata"] is True
    attr = [f for f in report["findings"] if "attr:" in f]
    assert attr
    assert len(attr) <= HTML_INSPECT_FINDINGS_CAP + 1
    _assert_capped([f for f in report["findings"] if f.startswith("ch.xhtml:")])


def _post(conn, path: str, payload: dict) -> tuple[int, bytes]:
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    return resp.status, resp.read()


@pytest.fixture()
def http_conn():
    # Default WATERMARKS_SERVER_API_KEY is empty, so /inspect is unauthenticated.
    assert server.API_KEY == ""
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    try:
        yield conn
    finally:
        conn.close()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _html_file(copies: int) -> dict[str, str]:
    raw = ('data-ai="x"' * copies).encode("ascii")
    return {"name": "x.html", "file": base64.b64encode(raw).decode("ascii")}


def test_post_inspect_pretty_json_stays_bounded(http_conn):
    status, raw = _post(http_conn, "/inspect", _html_file(2_000))
    assert status == 200
    # The handler pretty-prints. A capped report is a few kilobytes; one
    # finding per attribute is tens of kilobytes at this size and gigabytes
    # at the 256 MiB input cap.
    assert len(raw) < 12_000
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["kind"] == "container"
    report = body["report"]
    assert report["format"] == "html"
    assert report["has_ai_metadata"] is True
    _assert_capped(report["findings"])
    assert len(report["findings_confidence"]) == len(report["findings"])


def test_post_inspect_htm_suffix_is_the_same_path(http_conn):
    raw_file = ('data-ai="x"' * 2_000).encode("ascii")
    status, raw = _post(
        http_conn,
        "/inspect",
        {"name": "x.htm", "file": base64.b64encode(raw_file).decode("ascii")},
    )
    assert status == 200
    assert len(raw) < 12_000
    report = json.loads(raw)["report"]
    assert report["format"] == "html"
    _assert_capped(report["findings"])


def test_post_inspect_batch_caps_each_html_file(http_conn):
    status, raw = _post(
        http_conn,
        "/inspect/batch",
        {"files": [_html_file(2_000), {"name": "y.htm", "file": _html_file(2_000)["file"]}]},
    )
    assert status == 200
    assert len(raw) < 24_000
    body = json.loads(raw)
    assert body["ok"] is True
    assert len(body["results"]) == 2
    for result in body["results"]:
        assert result["ok"] is True
        _assert_capped(result["report"]["findings"])
        assert len(result["report"]["findings_confidence"]) == len(result["report"]["findings"])
