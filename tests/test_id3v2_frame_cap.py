"""ID3v2 parsers must stop at a frame cap instead of retaining every frame.

`_parse_id3v2_frames` appends one `(frame_id, payload)` tuple per v2.3/v2.4
frame and every inspect/clean caller walks that list. A tag filled with
empty 10-byte frames is still a legal MP3 (or a FLAC when ``fLaC`` follows
the tag, or a WAV ``id3`` chunk). At the default body cap that is about
30 million frames, which is enough to exhaust the worker.

Contract these tests lock in:

- At most ``MAX_RETAINED_ID3_FRAMES`` frames are retained. A tag with more
  frames must be reported as truncated (the existing truncated-tag wording).
  Building the full list and then slicing it still fails the allocation test.
- A tag with a few frames and a large payload, larger than the over-cap
  empty-frame fixture, still parses. The bound is the frame count.
- ``_strip_wav`` does not expand frames into a list, so WAV cleaning is not
  part of this contract. PDF/ZIP attachment routing and the CLIs call
  ``inspect_av`` / ``clean_av``, which are covered here.

The cap tests fail until the parser enforces this.
"""

from __future__ import annotations

import base64
import gc
import http.client
import json
import struct
import sys
import threading
import tracemalloc
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import av_meta
import server

# Generous for real tags (text frames, artwork, chapters) and far below the
# millions of empty frames that fit under MAX_BODY_BYTES.
MAX_RETAINED_ID3_FRAMES = 8192
OVER_FRAMES = MAX_RETAINED_ID3_FRAMES * 4
_EXTRA_FRAMES = OVER_FRAMES - MAX_RETAINED_ID3_FRAMES
# Below one pointer per extra frame (8 bytes) and one tuple (~100 bytes).
_RESIDUAL_BYTES_PER_EXTRA_FRAME = 3


def _synchsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _frame(frame_id: bytes, payload: bytes, major: int) -> bytes:
    size = _synchsafe(len(payload)) if major == 4 else struct.pack(">I", len(payload))
    return frame_id + size + b"\x00\x00" + payload


def _tag_from_frames(frames: bytes, major: int) -> bytes:
    return b"ID3" + bytes([major, 0, 0]) + _synchsafe(len(frames)) + frames


def _empty_tag(frame_count: int, major: int = 3, tail: bytes = b"") -> bytes:
    """v2.3/v2.4 tag of ``frame_count`` zero-length TIT2 frames, then ``tail``."""
    body = _frame(b"TIT2", b"", major) * frame_count
    return _tag_from_frames(body, major) + tail


def _wav_with_id3(id3: bytes) -> bytes:
    chunk = b"id3 " + struct.pack("<I", len(id3)) + id3
    if len(id3) & 1:
        chunk += b"\x00"
    body = b"WAVE" + chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _frame_list(parsed: object) -> list:
    if parsed is None:
        return []
    if isinstance(parsed, tuple) and len(parsed) >= 3 and isinstance(parsed[2], list):
        return parsed[2]
    raise AssertionError(f"unexpected ID3 parse result: {type(parsed)!r}")


def _texts(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_texts(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_texts(v) for v in value)
    return ""


def _assert_truncated(text: str) -> None:
    assert "truncat" in text.lower(), f"expected a truncated-tag error, got: {text[:500]!r}"


def _signal_text(value: object) -> str:
    """Findings, notes, actions, and errors. Paths and file bytes are ignored.

    Pytest names the tmp directory after the test, and a base64 body can
    spell "truncat" by chance. Neither is a truncated-tag report.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, av_meta.AVInspectReport):
        return _texts(value.findings) + " " + _texts(value.notes)
    if isinstance(value, dict):
        parts = []
        for key in ("error", "findings", "notes", "actions", "post_findings"):
            if key in value:
                parts.append(_texts(value[key]))
        report = value.get("report")
        if isinstance(report, dict):
            parts.append(_signal_text(report))
        results = value.get("results")
        if isinstance(results, list):
            parts.extend(_signal_text(item) for item in results)
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(item, str) for item in value):
            # (_strip_id3v2 / _strip_flac) -> (cleaned_bytes, actions). The
            # cleaned bytes are not a message; actions are.
            return value[1]
        return " ".join(_signal_text(item) for item in value)
    return ""


def _outcome_text(fn) -> str:
    try:
        result = fn()
    except ValueError as exc:
        return str(exc)
    return _signal_text(result)


def _peak(fn) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        fn()
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak


def _assert_allocation_bounded(run, accounted_bytes: int) -> None:
    """``run(False)`` is the cap-sized input; ``run(True)`` is the over-cap input.

    ``accounted_bytes`` is the extra input copy the caller already makes
    (file read, WAV chunk slice). Residual growth past that must stay below
    one pointer per extra frame, so buffering every frame and then dropping
    the tail still fails.
    """
    base = _peak(lambda: run(False))
    over = _peak(lambda: run(True))
    residual = (over - base) - accounted_bytes
    limit = _EXTRA_FRAMES * _RESIDUAL_BYTES_PER_EXTRA_FRAME
    assert residual < limit, (
        f"retained {residual} bytes for {_EXTRA_FRAMES} extra frames "
        f"(limit {limit}); frames are still materialized past the cap"
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _post(conn: http.client.HTTPConnection, path: str, payload: dict) -> tuple[int, dict]:
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
def conn() -> http.client.HTTPConnection:
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    yield client
    client.close()
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def test_empty_frame_bomb_fits_under_the_body_cap_and_exceeds_millions() -> None:
    """The default body cap admits millions of empty frames. No buffer is built."""
    overhead = len(b'{"file":"","name":"x.mp3"}')
    raw_max = ((server.MAX_BODY_BYTES - overhead) // 4) * 3
    max_frames = (raw_max - 10) // 10
    assert max_frames > 1_000_000
    assert max_frames > MAX_RETAINED_ID3_FRAMES
    sample = _empty_tag(OVER_FRAMES)
    envelope = len(json.dumps({"file": _b64(sample), "name": "x.mp3"}).encode("utf-8"))
    assert envelope <= server.MAX_BODY_BYTES
    assert len(sample) <= server.MAX_INPUT_BYTES


def test_small_id3_marker_still_reported() -> None:
    payload = b"Generated by OpenAI"
    data = _tag_from_frames(_frame(b"TXXX", payload, 3), 3)
    report = av_meta.inspect_av(Path("note.mp3"), data=data)
    assert report.format == "mp3"
    assert report.has_ai_metadata is True
    assert "truncat" not in _signal_text(report).lower()
    parsed = _frame_list(av_meta._parse_id3v2_frames(data))
    assert len(parsed) == 1
    assert payload in parsed[0][1]


def test_few_large_frames_are_kept() -> None:
    """Frame cap, not a byte cap: one large payload stays under the frame limit."""
    payload = b"\x11" * 400_000
    data = _tag_from_frames(
        _frame(b"TIT2", b"title", 3) + _frame(b"TXXX", payload, 3),
        3,
    )
    assert len(data) > len(_empty_tag(OVER_FRAMES))
    parsed = _frame_list(av_meta._parse_id3v2_frames(data))
    assert len(parsed) == 2
    assert parsed[1][1] == payload
    report = av_meta.inspect_av(Path("art.mp3"), data=data)
    assert "truncat" not in _signal_text(report).lower()


@pytest.mark.parametrize("major", [3, 4])
def test_parse_retains_at_most_the_frame_cap(major: int) -> None:
    parsed = av_meta._parse_id3v2_frames(_empty_tag(OVER_FRAMES, major))
    frames = _frame_list(parsed)
    assert len(frames) <= MAX_RETAINED_ID3_FRAMES, (
        f"v2.{major} parser retained {len(frames)} frames"
    )


@pytest.mark.parametrize("major", [3, 4])
def test_parse_allocation_stops_at_the_frame_cap(major: int) -> None:
    base = _empty_tag(MAX_RETAINED_ID3_FRAMES, major)
    over = _empty_tag(OVER_FRAMES, major)

    def run(big: bool) -> None:
        av_meta._parse_id3v2_frames(over if big else base)

    _assert_allocation_bounded(run, accounted_bytes=0)


@pytest.mark.parametrize("major", [3, 4])
def test_inspect_mp3_reports_truncated_tag(major: int) -> None:
    data = _empty_tag(OVER_FRAMES, major)
    text = _outcome_text(lambda: av_meta.inspect_av(Path("bomb.mp3"), data=data))
    _assert_truncated(text)


def test_inspect_mp3_allocation_stops_at_the_frame_cap() -> None:
    base = _empty_tag(MAX_RETAINED_ID3_FRAMES)
    over = _empty_tag(OVER_FRAMES)

    def run(big: bool) -> None:
        av_meta.inspect_av(Path("bomb.mp3"), data=over if big else base)

    _assert_allocation_bounded(run, accounted_bytes=0)


def test_inspect_flac_leading_id3_reports_truncated_tag() -> None:
    data = _empty_tag(OVER_FRAMES, tail=b"fLaC")
    assert av_meta.detect_av_format(_empty_tag(2, tail=b"fLaC")) == "flac"
    text = _outcome_text(lambda: av_meta.inspect_av(Path("bomb.flac"), data=data))
    _assert_truncated(text)


def test_inspect_flac_allocation_stops_at_the_frame_cap() -> None:
    base = _empty_tag(MAX_RETAINED_ID3_FRAMES, tail=b"fLaC")
    over = _empty_tag(OVER_FRAMES, tail=b"fLaC")

    def run(big: bool) -> None:
        av_meta.inspect_av(Path("bomb.flac"), data=over if big else base)

    _assert_allocation_bounded(run, accounted_bytes=0)


def test_inspect_wav_id3_chunk_reports_truncated_tag() -> None:
    data = _wav_with_id3(_empty_tag(OVER_FRAMES))
    text = _outcome_text(lambda: av_meta.inspect_av(Path("bomb.wav"), data=data))
    _assert_truncated(text)


def test_inspect_wav_allocation_stops_at_the_frame_cap() -> None:
    base = _wav_with_id3(_empty_tag(MAX_RETAINED_ID3_FRAMES))
    over = _wav_with_id3(_empty_tag(OVER_FRAMES))
    accounted = len(over) - len(base)

    def run(big: bool) -> None:
        av_meta.inspect_av(Path("bomb.wav"), data=over if big else base)

    _assert_allocation_bounded(run, accounted_bytes=accounted)


@pytest.mark.parametrize("strip_all", [True, False])
def test_clean_mp3_reports_truncated_tag(tmp_path: Path, strip_all: bool) -> None:
    src = tmp_path / "bomb.mp3"
    dest = tmp_path / "out.mp3"
    src.write_bytes(_empty_tag(OVER_FRAMES))
    text = _outcome_text(
        lambda: av_meta.clean_av(src, dest, strip_all_metadata=strip_all)
    )
    _assert_truncated(text)


@pytest.mark.parametrize("strip_all", [True, False])
def test_clean_mp3_allocation_stops_at_the_frame_cap(tmp_path: Path, strip_all: bool) -> None:
    base = _empty_tag(MAX_RETAINED_ID3_FRAMES)
    over = _empty_tag(OVER_FRAMES)
    base_path = tmp_path / "base.mp3"
    over_path = tmp_path / "over.mp3"
    base_path.write_bytes(base)
    over_path.write_bytes(over)
    dest_base = tmp_path / "base-out.mp3"
    dest_over = tmp_path / "over-out.mp3"

    def run(big: bool) -> None:
        av_meta.clean_av(
            over_path if big else base_path,
            dest_over if big else dest_base,
            strip_all_metadata=strip_all,
        )

    _assert_allocation_bounded(run, accounted_bytes=len(over) - len(base))


@pytest.mark.parametrize("strip_all", [True, False])
def test_clean_flac_reports_truncated_tag(tmp_path: Path, strip_all: bool) -> None:
    src = tmp_path / "bomb.flac"
    dest = tmp_path / "out.flac"
    src.write_bytes(_empty_tag(OVER_FRAMES, tail=b"fLaC"))
    text = _outcome_text(
        lambda: av_meta.clean_av(src, dest, strip_all_metadata=strip_all)
    )
    _assert_truncated(text)


@pytest.mark.parametrize("strip_all", [True, False])
def test_clean_flac_allocation_stops_at_the_frame_cap(tmp_path: Path, strip_all: bool) -> None:
    base = _empty_tag(MAX_RETAINED_ID3_FRAMES, tail=b"fLaC")
    over = _empty_tag(OVER_FRAMES, tail=b"fLaC")
    base_path = tmp_path / "base.flac"
    over_path = tmp_path / "over.flac"
    base_path.write_bytes(base)
    over_path.write_bytes(over)

    def run(big: bool) -> None:
        av_meta.clean_av(
            over_path if big else base_path,
            tmp_path / ("over-out.flac" if big else "base-out.flac"),
            strip_all_metadata=strip_all,
        )

    _assert_allocation_bounded(run, accounted_bytes=len(over) - len(base))


def test_strip_id3_selective_and_full_report_truncated_tag() -> None:
    data = _empty_tag(OVER_FRAMES)
    for strip_all in (True, False):
        text = _outcome_text(lambda strip_all=strip_all: av_meta._strip_id3v2(data, strip_all_metadata=strip_all))
        _assert_truncated(text)


def test_strip_flac_selective_and_full_report_truncated_tag() -> None:
    data = _empty_tag(OVER_FRAMES, tail=b"fLaC")
    for strip_all in (True, False):
        text = _outcome_text(
            lambda strip_all=strip_all: av_meta._strip_flac(data, strip_all_metadata=strip_all)
        )
        _assert_truncated(text)


def test_http_small_mp3_marker_still_reported(conn: http.client.HTTPConnection) -> None:
    data = _tag_from_frames(_frame(b"TXXX", b"Generated by OpenAI", 3), 3)
    status, body = _post(conn, "/inspect", {"file": _b64(data), "name": "note.mp3"})
    assert status == 200
    assert body["ok"] is True
    assert body["kind"] == "av"
    assert body["report"]["has_ai_metadata"] is True
    assert "truncat" not in _signal_text(body).lower()


def _http_body(kind: str) -> bytes:
    if kind == "mp3":
        return _empty_tag(OVER_FRAMES)
    if kind == "flac":
        return _empty_tag(OVER_FRAMES, tail=b"fLaC")
    if kind == "wav":
        return _wav_with_id3(_empty_tag(OVER_FRAMES))
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("path", "name", "kind", "options"),
    [
        ("/inspect", "bomb.mp3", "mp3", None),
        ("/inspect", "bomb.flac", "flac", None),
        ("/inspect", "bomb.wav", "wav", None),
        ("/detect", "bomb.mp3", "mp3", None),
        ("/detect", "bomb.flac", "flac", None),
        ("/clean", "bomb.mp3", "mp3", None),
        ("/clean", "bomb.mp3", "mp3", {"keep_non_ai_metadata": True}),
        ("/clean", "bomb.flac", "flac", None),
        ("/clean", "bomb.flac", "flac", {"keep_non_ai_metadata": True}),
    ],
)
def test_http_over_cap_tag_reports_truncated(
    conn: http.client.HTTPConnection,
    path: str,
    name: str,
    kind: str,
    options: dict | None,
) -> None:
    body: dict = {"file": _b64(_http_body(kind)), "name": name}
    if options is not None:
        body["options"] = options
    status, response = _post(conn, path, body)
    assert status in (200, 400), response
    _assert_truncated(_signal_text(response))


def test_http_batch_inspect_detect_and_clean_report_truncated(
    conn: http.client.HTTPConnection,
) -> None:
    mp3 = _empty_tag(OVER_FRAMES)
    flac = _empty_tag(OVER_FRAMES, tail=b"fLaC")
    entries = [
        {"file": _b64(mp3), "name": "bomb.mp3"},
        {"file": _b64(flac), "name": "bomb.flac"},
    ]
    for path in ("/inspect/batch", "/detect/batch", "/clean/batch"):
        status, response = _post(conn, path, {"files": entries})
        assert status in (200, 400), response
        _assert_truncated(_signal_text(response))
        results = response.get("results")
        if isinstance(results, list):
            assert results, response
            for item in results:
                _assert_truncated(_signal_text(item))
