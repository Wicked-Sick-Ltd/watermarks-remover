"""WebP RIFF parsing must not retain every chunk.

A WebP of repeating zero-length chunks stays far under the request body cap
(``MAX_BODY_BYTES`` / ``MAX_INPUT_BYTES``) while ``_webp_chunks`` appends a
tuple and a copied fourcc for every chunk. ``POST /inspect`` and ``POST /clean``
of a ``.webp`` run that parser, and so does every container path that calls
``inspect_webp`` or ``strip_webp``.

These tests fail while that list is built. A fix that walks chunks in place
(or stops after a small retained-chunk cap, before the rest are copied) keeps
the traced peak from growing by a Python object per chunk. A small WebP with a
real C2PA chunk must still be reported and stripped.
"""

from __future__ import annotations

import base64
import gc
import http.client
import io
import json
import struct
import subprocess
import sys
import tempfile
import threading
import tracemalloc
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_lib
import audit_website
import container_meta
import server
from common import which
from image_meta import clean_image, inspect_image, inspect_webp, strip_webp

# Large enough that today's ~100 bytes of tuple/fourcc overhead per empty
# chunk clears the allowance, and small enough to stay well under the
# 256 MiB input cap. 40_000 chunks is 320_012 bytes.
CHUNK_COUNT = 40_000
SLOPE_SMALL = 2_000
# Streaming the 8-byte chunk into an output buffer costs 8 bytes/chunk.
# Retaining the (fourcc, payload, padding) tuple costs about 100. 16 sits
# between those, so a walk that does not keep every chunk passes and the
# current list does not.
MAX_BYTES_PER_EXTRA_CHUNK = 16
# Same-size control vs bomb. The control pays for the file bytes once;
# the bomb's extra cost is the chunk list.
SAME_SIZE_ALLOWANCE = 1 << 20

needs_qpdf = pytest.mark.skipif(which("qpdf") is None, reason="qpdf not installed")


def _empty_chunk_webp(count: int, fourcc: bytes = b"JUNK") -> bytes:
    """RIFF/WEBP whose chunks are ``count`` copies of an 8-byte zero-length chunk."""
    body = b"WEBP" + (fourcc + struct.pack("<I", 0)) * count
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _single_payload_webp(total_size: int) -> bytes:
    """One ``VP8 `` chunk whose file length matches ``total_size``."""
    payload_len = total_size - 20
    if payload_len < 0 or payload_len % 2:
        raise AssertionError(f"no even single-chunk WebP of {total_size} bytes")
    body = b"WEBP" + b"VP8 " + struct.pack("<I", payload_len) + (b"\x00" * payload_len)
    data = b"RIFF" + struct.pack("<I", len(body)) + body
    if len(data) != total_size:
        raise AssertionError((len(data), total_size))
    return data


def _webp_chunk(fourcc: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) & 1 else b""
    return fourcc + struct.pack("<I", len(payload)) + payload + padding


def _marked_webp() -> bytes:
    body = b"WEBP" + b"".join(
        (
            _webp_chunk(b"VP8X", b"\x04" + b"\x00" * 9),
            _webp_chunk(b"VP8 ", b"image-data"),
            _webp_chunk(b"C2PA", b"jumb c2pa manifest"),
        )
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _peak(fn) -> int:
    """High-water traced bytes of ``fn``, including a refusal raised mid-parse.

    ``ValueError`` is the refusal the cleaners already use for a malformed
    WebP. A cap that raises one still has to do it before the chunk list
    exists; the peak is what the assertion checks.
    """
    gc.collect()
    tracemalloc.start()
    try:
        fn()
    except ValueError:
        # Malformed-input refusal is expected during this measurement.
        # The peak is what the assertion checks, including a cap that raises.
        pass
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak


def _quiet_peak(fn, trials: int = 3) -> int:
    """Lowest traced peak across a few trials.

    tracemalloc's high-water mark only rises. A same-size bomb takes longer
    to walk than the single-chunk control, so other threads in a full suite
    (the HTTP server in particular) can push one sample over the allowance
    on 3.13 without a chunk tuple being retained. A parser that keeps a
    tuple per chunk is high on every trial, so the minimum still fails.
    """
    return min(_peak(fn) for _ in range(trials))


def _assert_slope(label: str, parse) -> None:
    small = _empty_chunk_webp(SLOPE_SMALL)
    large = _empty_chunk_webp(CHUNK_COUNT)
    parse(small)
    parse(large)
    peak_small = _peak(lambda: parse(small))
    peak_large = _peak(lambda: parse(large))
    extra = CHUNK_COUNT - SLOPE_SMALL
    delta = peak_large - peak_small
    limit = extra * MAX_BYTES_PER_EXTRA_CHUNK
    assert delta < limit, (
        f"{label}: peak grew by {delta} bytes across {extra} extra empty chunks "
        f"({delta / extra:.1f} bytes/chunk, limit {MAX_BYTES_PER_EXTRA_CHUNK})"
    )


def _assert_same_size(label: str, invoke, bomb: bytes, control: bytes, measure=_peak) -> None:
    assert len(bomb) == len(control)
    invoke(control)
    invoke(bomb)
    peak_control = measure(lambda: invoke(control))
    peak_bomb = measure(lambda: invoke(bomb))
    delta = peak_bomb - peak_control
    assert delta < SAME_SIZE_ALLOWANCE, (
        f"{label}: empty-chunk WebP peaked {peak_bomb} bytes, "
        f"same-size single chunk peaked {peak_control} "
        f"(delta {delta}, allowance {SAME_SIZE_ALLOWANCE})"
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _post(conn: http.client.HTTPConnection, path: str, payload: dict) -> tuple[int, dict]:
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    raw = resp.read()
    return resp.status, json.loads(raw) if raw else {}


@pytest.fixture(scope="module")
def conn() -> http.client.HTTPConnection:
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=120)
    yield client
    client.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def test_small_webp_c2pa_is_still_detected_and_removed():
    """A handful of real chunks must keep working once the bomb is bounded."""
    data = _marked_webp()
    has_c2pa, has_ai, findings = inspect_webp(data)
    assert has_c2pa and has_ai
    assert "WebP C2PA chunk" in findings
    cleaned, actions = strip_webp(data)
    assert any("C2PA" in action for action in actions)
    assert inspect_webp(cleaned)[0] is False


def test_inspect_webp_memory_does_not_scale_with_empty_chunks():
    _assert_slope("inspect_webp", inspect_webp)


def test_strip_webp_memory_does_not_scale_with_empty_chunks():
    _assert_slope("strip_webp", strip_webp)


def _inspect_image(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.webp"
        path.write_bytes(data)
        inspect_image(path, data=data, run_synthid=False, include_exiftool=False)


def _clean_image(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "in.webp"
        src.write_bytes(data)
        clean_image(src, root / "out.webp")


def _scan_file(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.webp"
        path.write_bytes(data)
        audit_lib.scan_file(path, display_name="in.webp", check_stylometry=False)


def _inspect_remote(data: bytes) -> None:
    audit_website.inspect_remote("https://example.com/shot.webp", data, "image/webp")


def _html(data: bytes) -> str:
    return f'<html><img src="data:image/webp;base64,{_b64(data)}"></html>'


def _inspect_html(data: bytes) -> None:
    container_meta.inspect_html(_html(data))


def _clean_html(data: bytes) -> None:
    container_meta.clean_html(_html(data))


def _inspect_markdown(data: bytes) -> None:
    container_meta.inspect_markdown(_html(data))


def _clean_markdown(data: bytes) -> None:
    container_meta.clean_markdown(_html(data))


def _svg(data: bytes) -> bytes:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f'<image href="data:image/webp;base64,{_b64(data)}"/>'
        "</svg>"
    ).encode("ascii")


def _inspect_svg(data: bytes) -> None:
    container_meta.inspect_svg(_svg(data))


def _clean_svg(data: bytes) -> None:
    container_meta.clean_svg(_svg(data))


def _zip_with(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, data)
    return buf.getvalue()


def _inspect_docx(data: bytes) -> None:
    container_meta.inspect_docx(_zip_with("word/media/shot.webp", data))


def _clean_docx(data: bytes) -> None:
    container_meta.clean_docx(_zip_with("word/media/shot.webp", data))


def _inspect_xlsx(data: bytes) -> None:
    container_meta.inspect_xlsx(_zip_with("xl/media/shot.webp", data))


def _clean_xlsx(data: bytes) -> None:
    container_meta.clean_xlsx(_zip_with("xl/media/shot.webp", data))


def _inspect_pptx(data: bytes) -> None:
    container_meta.inspect_pptx(_zip_with("ppt/media/shot.webp", data))


def _clean_pptx(data: bytes) -> None:
    container_meta.clean_pptx(_zip_with("ppt/media/shot.webp", data))


def _clean_epub(data: bytes) -> None:
    container_meta.clean_epub(_zip_with("OEBPS/images/shot.webp", data))


def _pdf_with_webp(root: Path, data: bytes) -> Path:
    src = root / "shot.webp"
    src.write_bytes(data)
    out = root / "in.pdf"
    qpdf = which("qpdf")
    assert qpdf is not None
    subprocess.run(
        [
            qpdf,
            "--empty",
            "--add-attachment",
            str(src),
            "--filename=shot.webp",
            "--key=shot.webp",
            "--",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def _inspect_pdf(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = _pdf_with_webp(Path(tmp), data)
        container_meta.inspect_pdf(pdf, pdf.read_bytes())


def _clean_pdf(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf = _pdf_with_webp(root, data)
        container_meta.clean_pdf(pdf, root / "out.pdf", deep_images="never")


EMBEDDED = [
    ("inspect_image", _inspect_image),
    ("clean_image", _clean_image),
    ("audit_lib.scan_file", _scan_file),
    ("audit_website.inspect_remote", _inspect_remote),
    ("inspect_html", _inspect_html),
    ("clean_html", _clean_html),
    ("inspect_markdown", _inspect_markdown),
    ("clean_markdown", _clean_markdown),
    ("inspect_svg", _inspect_svg),
    ("clean_svg", _clean_svg),
    ("inspect_docx", _inspect_docx),
    ("clean_docx", _clean_docx),
    ("inspect_xlsx", _inspect_xlsx),
    ("clean_xlsx", _clean_xlsx),
    ("inspect_pptx", _inspect_pptx),
    ("clean_pptx", _clean_pptx),
    ("clean_epub", _clean_epub),
]


@pytest.mark.parametrize("label,invoke", EMBEDDED, ids=[label for label, _invoke in EMBEDDED])
def test_callers_do_not_retain_every_empty_webp_chunk(label: str, invoke) -> None:
    bomb = _empty_chunk_webp(CHUNK_COUNT)
    control = _single_payload_webp(len(bomb))
    _assert_same_size(label, invoke, bomb, control)


@needs_qpdf
@pytest.mark.parametrize("label,invoke", [("inspect_pdf", _inspect_pdf), ("clean_pdf", _clean_pdf)])
def test_pdf_attachment_webp_does_not_retain_every_empty_chunk(label: str, invoke) -> None:
    bomb = _empty_chunk_webp(CHUNK_COUNT)
    control = _single_payload_webp(len(bomb))
    _assert_same_size(label, invoke, bomb, control)


@pytest.mark.parametrize("path", ["/inspect", "/clean", "/inspect/batch", "/clean/batch"])
def test_http_webp_does_not_retain_every_empty_chunk(path, conn, monkeypatch):
    """The attack body is under both caps and still reaches the chunk list.

    No API key is required unless one is configured. Clearing it here forces
    the default open check so a developer env key cannot hide the parser.
    ``MARKLLM_DIR`` makes that empty key refuse every request, so a 401 would
    pass the memory comparison without parsing. It is cleared, and a non-200
    response fails the test.
    """
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.delenv("MARKLLM_DIR", raising=False)
    bomb = _empty_chunk_webp(CHUNK_COUNT)
    control = _single_payload_webp(len(bomb))
    assert len(bomb) <= server.MAX_INPUT_BYTES
    assert (server.MAX_INPUT_BYTES - 12) // 8 >= 1_000_000

    def invoke(data: bytes) -> None:
        payload = {"file": _b64(data), "name": "shot.webp"}
        body = {"files": [payload]} if path.endswith("/batch") else payload
        raw = json.dumps(body).encode("utf-8")
        assert len(raw) <= server.MAX_BODY_BYTES
        status, response = _post(conn, path, body)
        assert status == 200, response
        assert response.get("ok") is True

    # One tracemalloc sample on 3.13 can clear the allowance while the
    # in-process server thread allocates. The allowance itself is unchanged.
    _assert_same_size(path, invoke, bomb, control, measure=_quiet_peak)


def test_http_marked_webp_is_still_inspected(conn, monkeypatch):
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.delenv("MARKLLM_DIR", raising=False)
    status, body = _post(conn, "/inspect", {"file": _b64(_marked_webp()), "name": "shot.webp"})
    assert status == 200
    assert body["report"]["has_c2pa"] is True
    assert body["report"]["format"] == "webp"
