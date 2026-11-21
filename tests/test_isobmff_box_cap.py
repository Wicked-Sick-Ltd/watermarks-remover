"""Spec: ISOBMFF walks must refuse a box flood before retaining every box.

``_parse_isobmff_boxes`` appends one ``(fourcc, payload, size, header)`` tuple
per box and nothing stops that list. An 8-byte ``free`` box is a legal box, and
``MAX_INPUT_BYTES`` (default 256 MiB) still admits tens of millions of them, so
POST /inspect and POST /clean can build a multi-gigabyte object graph from a
body that stays under ``MAX_BODY_BYTES``.

These tests fail while that walk retains every box. They pass once every walk
refuses, with ``ValueError``, before it retains more than ``MAX_RETAINED_BOXES``
records. Raising only after the full list exists still fails the traced-memory
ceiling. A file-size cutoff that rejects a large single ``mdat`` also fails.

The ceiling is calibrated on this interpreter from walks of 256 and 512 boxes
(both must remain legal). It is the traced cost of ``MAX_RETAINED_BOXES``
records, plus a 1.5x slack and, where a caller must read the upload, three
copies of those buffers.

Not covered here: PDF attachment extraction goes through qpdf
(``_pdf_show_attachment``), which is not installed in this environment. Once
the bytes are extracted, ``_inspect_attachment_bytes`` and
``_clean_attachment_bytes`` are tested. ODT inspect/clean and EPUB inspect do
not call the ISOBMFF walker. ``xlsx`` / ``pptx`` share the OOXML helpers with
``docx`` and are parametrized below.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import sys
import tracemalloc
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import av_meta
import common
import container_meta
import image_meta
import server

# A normal media file is a handful of boxes. 512 free boxes plus the ftyp
# header must still parse; a walk may not retain more than this many records.
MIN_ACCEPTED_FREE_BOXES = 512
MAX_RETAINED_BOXES = 4096
# Far past the cap, still a few hundred KiB — well under the upload limits.
BOMB_FREE_BOXES = 32_768

_AV_NAMES = ("clip.mp4", "clip.mov", "clip.m4a", "clip.m4v")
_IMAGE_NAMES = (
    ("photo.avif", b"avif"),
    ("photo.heic", b"heic"),
    ("photo.heif", b"heif"),
)


def _box(fourcc: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc + payload


_FREE = _box(b"free")
assert len(_FREE) == 8


def _ftyp(brand: bytes) -> bytes:
    return _box(b"ftyp", brand + b"\x00\x00\x00\x00" + brand)


def _top_level(n: int, brand: bytes) -> bytes:
    """ftyp plus *n* minimum-size free boxes. *n* + 1 boxes in total."""
    return _ftyp(brand) + _FREE * n


def _meta_bomb(n: int, brand: bytes = b"avif") -> bytes:
    """ftyp plus one meta box whose children are *n* free boxes.

    Top-level box count is 2. The nested walk is what has to be capped.
    """
    children = b"\x00\x00\x00\x00" + _FREE * n
    return _ftyp(brand) + _box(b"meta", children)


def _moov_bomb(n: int) -> bytes:
    """ftyp plus one moov whose children are *n* free boxes.

    ``inspect_isobmff`` does not open moov. ``_inspect_moov_udta`` /
    ``_strip_moov_udta`` do.
    """
    return _ftyp(b"isom") + _box(b"moov", _FREE * n)


def _traced(fn):
    tracemalloc.start()
    try:
        try:
            value = fn()
        except Exception as exc:
            return exc, tracemalloc.get_traced_memory()[1], None
        return None, tracemalloc.get_traced_memory()[1], value
    finally:
        tracemalloc.stop()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# Populated on first use. A list, not a global rebinding, so the cache can be
# filled without ``global``.
_CALIBRATION: list[tuple[int, float]] = []


def _calibration() -> tuple[int, float]:
    """(peak of a 1024-free-box walk, traced bytes per additional box).

    The two walks are both under ``MAX_RETAINED_BOXES``. The median of several
    samples keeps a concurrent thread's allocations from inverting the slope.
    """
    if _CALIBRATION:
        return _CALIBRATION[0]
    lows: list[float] = []
    pers: list[float] = []
    for _ in range(5):
        _, peak_lo, _ = _traced(lambda: image_meta._parse_isobmff_boxes(_top_level(1024, b"avif")))
        _, peak_hi, _ = _traced(lambda: image_meta._parse_isobmff_boxes(_top_level(2048, b"avif")))
        lows.append(float(peak_lo))
        pers.append((peak_hi - peak_lo) / 1024)
    per = _median(pers)
    assert per > 0, pers
    result = (int(_median(lows)), per)
    _CALIBRATION.append(result)
    return result


def _budget(*copies: bytes) -> int:
    peak_lo, per = _calibration()
    retained = peak_lo + per * (MAX_RETAINED_BOXES - 1024)
    # 1.5x covers the capped list plus a short-lived prefix rebuild.
    # Callers that have to read the upload may also hold a few copies of it.
    return int(retained * 1.5) + 100_000 + sum(len(part) for part in copies) * 3


def _assert_refused(err, peak: int, *copies: bytes) -> None:
    assert isinstance(err, ValueError), err
    budget = _budget(*copies)
    assert peak <= budget, f"walk retained too much before refusing: peak={peak} budget={budget}"


def _assert_bounded(peak: int, *copies: bytes) -> None:
    budget = _budget(*copies)
    assert peak <= budget, f"walk retained every box: peak={peak} budget={budget}"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _post(conn, path: str, payload: dict) -> tuple[int, dict]:
    raw = json.dumps(payload).encode("utf-8")
    assert len(raw) <= server.MAX_BODY_BYTES
    conn.request(
        "POST",
        path,
        body=raw,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


@pytest.fixture(scope="module")
def conn():
    import http.client
    import threading

    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield client
    client.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def test_upload_limits_still_admit_tens_of_millions_of_minimum_boxes():
    """The body cap is not a box cap. This is the precondition the walk must close."""
    assert len(_FREE) == 8
    assert common.MAX_INPUT_BYTES // len(_FREE) >= 10_000_000
    assert server.MAX_BODY_BYTES >= common.MAX_INPUT_BYTES


def test_walk_at_accepted_count_parses_every_box():
    data = _top_level(MIN_ACCEPTED_FREE_BOXES, b"avif")
    boxes, end = image_meta._parse_isobmff_boxes(data)
    assert len(boxes) == MIN_ACCEPTED_FREE_BOXES + 1
    assert end == len(data)
    err, _peak, _value = _traced(lambda: image_meta.inspect_isobmff(data, "avif"))
    assert err is None
    err, _peak, cleaned = _traced(lambda: image_meta.strip_isobmff(data, "avif"))
    assert err is None
    assert isinstance(cleaned[0], bytes)
    assert len(cleaned[0]) == len(data)


def test_single_large_mdat_is_not_a_box_flood():
    """A box-count cap must not be implemented as a max file size of cap*8."""
    payload = b"\x00" * (MAX_RETAINED_BOXES * 8)
    data = _ftyp(b"avif") + _box(b"mdat", payload)
    assert len(data) > MAX_RETAINED_BOXES * 8
    boxes, end = image_meta._parse_isobmff_boxes(data)
    assert len(boxes) == 2
    assert end == len(data)
    err, _peak, _value = _traced(lambda: image_meta.inspect_isobmff(data, "avif"))
    assert err is None


def test_top_level_flood_is_refused_before_the_list_exists():
    data = _top_level(BOMB_FREE_BOXES, b"avif")
    assert len(data) < common.MAX_INPUT_BYTES
    err, peak, value = _traced(lambda: image_meta._parse_isobmff_boxes(data))
    assert value is None
    _assert_refused(err, peak, data)


def test_inspect_and_strip_refuse_top_level_flood():
    data = _top_level(BOMB_FREE_BOXES, b"avif")
    for fn in (
        lambda: image_meta.inspect_isobmff(data, "avif"),
        lambda: image_meta.strip_isobmff(data, "avif"),
    ):
        err, peak, value = _traced(fn)
        assert value is None
        _assert_refused(err, peak, data)


def test_meta_children_are_capped_too():
    # The outer walk sees two boxes (ftyp + meta). The cap has to apply to the
    # children; a top-level-only limit of 4096 would not fire.
    assert len(image_meta._parse_isobmff_boxes(_meta_bomb(3))[0]) == 2
    data = _meta_bomb(BOMB_FREE_BOXES)
    err, peak, value = _traced(lambda: image_meta.inspect_isobmff(data, "avif"))
    assert value is None
    _assert_refused(err, peak, data)
    err, peak, value = _traced(lambda: image_meta.strip_isobmff(data, "avif"))
    assert value is None
    _assert_refused(err, peak, data)


def test_moov_children_are_capped_on_inspect_and_strip():
    data = _moov_bomb(BOMB_FREE_BOXES)
    # Top-level inspect does not open moov; the MP4 path must still refuse.
    err, peak, value = _traced(lambda: av_meta._inspect_mp4(data))
    assert value is None
    _assert_refused(err, peak, data)
    err, peak, value = _traced(lambda: av_meta._strip_mp4(data, strip_all_metadata=True))
    assert value is None
    _assert_refused(err, peak, data)


@pytest.mark.parametrize("name,brand", _IMAGE_NAMES)
def test_clean_image_refuses_before_writing_output(tmp_path, name, brand):
    data = _top_level(BOMB_FREE_BOXES, brand)
    src = tmp_path / name
    dest = tmp_path / f"out{Path(name).suffix}"
    src.write_bytes(data)
    err, peak, value = _traced(lambda: image_meta.clean_image(src, dest, strip_all_metadata=True))
    assert value is None
    _assert_refused(err, peak, data)
    assert not dest.exists()
    assert src.read_bytes() == data


@pytest.mark.parametrize("name", _AV_NAMES)
def test_clean_av_refuses_before_writing_output(tmp_path, name):
    data = _top_level(BOMB_FREE_BOXES, b"isom")
    src = tmp_path / name
    dest = tmp_path / f"out{Path(name).suffix}"
    src.write_bytes(data)
    err, peak, value = _traced(lambda: av_meta.clean_av(src, dest))
    assert value is None
    _assert_refused(err, peak, data)
    assert not dest.exists()
    assert src.read_bytes() == data


def test_clean_av_refuses_moov_flood_before_writing_output(tmp_path):
    data = _moov_bomb(BOMB_FREE_BOXES)
    src = tmp_path / "clip.mp4"
    dest = tmp_path / "out.mp4"
    src.write_bytes(data)
    err, peak, value = _traced(lambda: av_meta.clean_av(src, dest))
    assert value is None
    _assert_refused(err, peak, data)
    assert not dest.exists()


def test_attachment_image_and_av_walks_refuse(tmp_path):
    image = _top_level(BOMB_FREE_BOXES, b"avif")
    video = _top_level(BOMB_FREE_BOXES, b"isom")
    image_path = tmp_path / "att-0.avif"
    video_path = tmp_path / "att-1.mp4"
    image_path.write_bytes(image)
    video_path.write_bytes(video)
    err, peak, value = _traced(
        lambda: container_meta._inspect_attachment_bytes(image_path, image, "image", 0)
    )
    assert value is None
    _assert_refused(err, peak, image)
    err, peak, value = _traced(
        lambda: container_meta._inspect_attachment_bytes(video_path, video, "av", 0)
    )
    assert value is None
    _assert_refused(err, peak, video)
    err, peak, value = _traced(
        lambda: container_meta._clean_attachment_bytes(image_path, image, "image", "always", 0)
    )
    assert value is None
    _assert_refused(err, peak, image)
    assert not image_path.with_name(image_path.name + ".cleaned").exists()
    err, peak, value = _traced(
        lambda: container_meta._clean_attachment_bytes(video_path, video, "av", "always", 0)
    )
    assert value is None
    _assert_refused(err, peak, video)
    assert not video_path.with_name(video_path.name + ".cleaned").exists()


def _html_wrapping(data: bytes, mime: str) -> str:
    return f'<html><img src="data:{mime};base64,{_b64(data)}"></html>'


def test_html_and_markdown_inspect_propagate_the_refusal():
    data = _top_level(BOMB_FREE_BOXES, b"avif")
    html = _html_wrapping(data, "image/avif")
    md = f"![x](data:image/avif;base64,{_b64(data)})\n"
    # The data-URI text is decoded inside the call, so the budget allows copies
    # of that text as well as the decoded image. It does not allow one tuple
    # per free box.
    err, peak, value = _traced(lambda: container_meta.inspect_html(html))
    assert value is None
    _assert_refused(err, peak, data, html.encode())
    err, peak, value = _traced(lambda: container_meta.inspect_markdown(md))
    assert value is None
    _assert_refused(err, peak, data, md.encode())


def test_embedded_clean_and_svg_do_not_retain_the_flood():
    """These callers swallow the walker's error. The allocation still has to stop."""
    data = _top_level(BOMB_FREE_BOXES, b"heic")
    html = _html_wrapping(data, "image/heic")
    md = f"![x](data:image/heic;base64,{_b64(data)})\n"
    svg = (
        '<?xml version="1.0"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg">'
        f'<image href="data:image/heic;base64,{_b64(data)}"/></svg>'
    ).encode()
    _err, peak, _value = _traced(lambda: container_meta.clean_html(html))
    _assert_bounded(peak, data, html.encode())
    _err, peak, _value = _traced(lambda: container_meta.clean_markdown(md))
    _assert_bounded(peak, data, md.encode())
    _err, peak, _value = _traced(lambda: container_meta.inspect_svg(svg))
    _assert_bounded(peak, data, svg)
    # clean_svg reaches this helper only after _strip_xml_declarations, which
    # appends one string per character of the document. That list dwarfs a
    # capped box walk, so the bound is on the helper that actually calls
    # strip_isobmff. clean_svg (and clean_html / clean_markdown above) swallow
    # the refusal and keep the original URI.
    text = svg.decode()
    _err, peak, _value = _traced(lambda: container_meta._clean_embedded_data_uris(text))
    _assert_bounded(peak, data, text.encode())


def _ooxml(media_name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("[Content_Types].xml", "<Types></Types>")
        zf.writestr(media_name, payload)
    return buf.getvalue()


@pytest.mark.parametrize(
    "fmt,media",
    [
        ("docx", "word/media/bomb.avif"),
        ("xlsx", "xl/media/bomb.heic"),
        ("pptx", "ppt/media/bomb.avif"),
    ],
)
def test_ooxml_embedded_media_does_not_retain_the_flood(fmt, media):
    brand = b"heic" if media.endswith(".heic") else b"avif"
    payload = _top_level(BOMB_FREE_BOXES, brand)
    archive = _ooxml(media, payload)
    inspect = getattr(container_meta, f"inspect_{fmt}")
    _err, peak, _value = _traced(lambda: inspect(archive))
    _assert_bounded(peak, payload, archive)
    _err, peak, _value = _traced(lambda: container_meta._scrub_ooxml_zip(archive, fmt))
    _assert_bounded(peak, payload, archive)


def test_epub_clean_does_not_retain_the_flood():
    payload = _top_level(BOMB_FREE_BOXES, b"avif")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", "<container/>")
        zf.writestr("OEBPS/content.opf", "<package></package>")
        zf.writestr("OEBPS/bomb.avif", payload)
    archive = buf.getvalue()
    _err, peak, _value = _traced(lambda: container_meta.clean_epub(archive))
    _assert_bounded(peak, payload, archive)


def test_http_small_avif_inspect_still_succeeds(conn):
    data = _top_level(4, b"avif")
    status, body = _post(conn, "/inspect", {"file": _b64(data), "name": "photo.avif"})
    assert status == 200
    assert body["ok"] is True
    assert body["kind"] == "image"
    assert body["report"]["format"] == "avif"


@pytest.mark.parametrize("name", _AV_NAMES)
def test_http_inspect_clean_and_detect_refuse_av_flood(conn, name):
    data = _top_level(BOMB_FREE_BOXES, b"isom")
    assert len(data) < common.MAX_INPUT_BYTES
    status, body = _post(conn, "/inspect", {"file": _b64(data), "name": name})
    assert status == 400
    assert body["ok"] is False
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": name})
    assert status == 400
    assert body["ok"] is False
    status, body = _post(conn, "/detect", {"file": _b64(data), "name": name})
    assert status == 400
    assert body["ok"] is False


@pytest.mark.parametrize("name,brand", _IMAGE_NAMES)
def test_http_inspect_and_clean_refuse_image_flood(conn, name, brand):
    data = _top_level(BOMB_FREE_BOXES, brand)
    status, body = _post(conn, "/inspect", {"file": _b64(data), "name": name})
    assert status == 400
    assert body["ok"] is False
    status, body = _post(conn, "/clean", {"file": _b64(data), "name": name})
    assert status == 400
    assert body["ok"] is False


def test_http_inspect_refuses_html_and_markdown_data_uri(conn):
    data = _top_level(BOMB_FREE_BOXES, b"avif")
    html = _html_wrapping(data, "image/avif").encode()
    md = f"![x](data:image/avif;base64,{_b64(data)})\n".encode()
    status, body = _post(conn, "/inspect", {"file": _b64(html), "name": "page.html"})
    assert status == 400
    assert body["ok"] is False
    status, body = _post(conn, "/inspect", {"file": _b64(md), "name": "notes.md"})
    assert status == 400
    assert body["ok"] is False


def test_http_batch_inspect_and_clean_report_the_refusal(conn):
    data = _top_level(BOMB_FREE_BOXES, b"avif")
    status, body = _post(
        conn,
        "/inspect/batch",
        {"files": [{"file": _b64(data), "name": "photo.avif"}]},
    )
    assert status == 200
    assert body["ok"] is True
    assert body["results"][0]["ok"] is False
    assert body["results"][0].get("error")
    heic = _top_level(BOMB_FREE_BOXES, b"heic")
    status, body = _post(
        conn,
        "/clean/batch",
        {"files": [{"file": _b64(heic), "name": "photo.heic"}]},
    )
    assert status == 200
    assert body["results"][0]["ok"] is False
    assert "cleaned" not in body["results"][0]


def test_http_batch_mp4_clean_refuses(conn):
    data = _moov_bomb(BOMB_FREE_BOXES)
    status, body = _post(
        conn,
        "/clean/batch",
        {"files": [{"file": _b64(data), "name": "clip.mp4"}]},
    )
    assert status == 200
    assert body["results"][0]["ok"] is False
    assert "cleaned" not in body["results"][0]
