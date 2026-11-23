"""Resource limits for per-frame video purification.

These tests are the spec for the unbounded frame-extraction bug. They fail while
one clean of a short, high-frame-count video may write every decoded PNG and may
spend a fresh purification timeout on every frame. They pass once both of these
hold:

- A single video creates at most ``MAX_EXTRACTED_FRAMES`` frame PNGs. Refusing
  the upload, or sampling down to that many frames, is acceptable. Writing the
  frames and deleting them afterwards is not: the count is taken while the
  extraction directory still exists.
- ``video_purify``'s ``timeout`` is one budget for extraction, every selected
  frame, and remux together. Omitting it uses that same budget at the default
  (3600 seconds), which is what ``POST /clean`` and ``POST /clean/batch`` do.
  The budget is read from ``time.monotonic`` (``time.time`` and
  ``time.perf_counter`` move by the same offset in these tests).

The per-file child rlimit (``common._CHILD_RLIMIT_FSIZE``) is not a frame-count
cap. The bomb fixture is a few seconds long so a duration check alone does not
satisfy the frame cap.
"""

from __future__ import annotations

import base64
import http.client
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import clean_video
import common
import server

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed")

# Small enough that a normal clip (the existing 40-frame pipeline) still fits,
# and far below the six-digit sequence ffmpeg is asked to write.
MAX_EXTRACTED_FRAMES = 64
# Default per-call timeout on video_purify. The whole request may spend this
# much clock time, not this much per frame.
TOTAL_BUDGET_SECONDS = 3600
# Each successful purifier invocation moves the clock forward by this much.
BUDGET_STEP_SECONDS = 200
# Under the frame cap, so a frame cap alone does not make the budget test pass.
DEADLINE_FRAMES = 48
# At or under the cap, and cheap per frame: every frame must still be purified.
SHORT_CLIP_FRAMES = 40
BOMB_FRAMES = 300

_PURIFIER_SCRIPTS = ("clean_ctrlregen.py", "markdiffusion_harness.py")


def _high_frame_mp4(path: Path, frames: int, *, rate: int = 100) -> None:
    """Build a short H.264 file with exactly ``frames`` coded frames.

    Solid 32x32 frames compress to a few kilobytes in the container and expand
    to one PNG per frame when ffmpeg decodes with ``-fps_mode vfr``.
    """
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=32x32:r={rate}",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _duration_seconds(path: Path) -> float:
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


class _FrameCounter:
    """Highest number of ``frame_*.png`` files seen during a purify call."""

    def __init__(self) -> None:
        self.pngs = 0
        self.max_file = 0

    def _note(self, paths: list[Path]) -> None:
        sizes: list[int] = []
        for path in paths:
            try:
                sizes.append(path.stat().st_size)
            except OSError:
                continue
        self.pngs = max(self.pngs, len(sizes))
        if sizes:
            self.max_file = max(self.max_file, *sizes)

    def note_dir(self, directory: Path) -> None:
        self._note(list(directory.glob("frame_*.png")))

    def note_tree(self, root: Path) -> None:
        self._note(list(root.rglob("frame_*.png")))


@pytest.fixture
def frame_counter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Count frame PNGs as ffmpeg writes them, before the temp dir is removed."""
    counter = _FrameCounter()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(clean_video.tempfile, "tempdir", str(scratch))

    real_run = clean_video._run_ffmpeg

    def wrapped(cmd: list[str], timeout: int, what: str):
        rc, err = real_run(cmd, timeout, what)
        if what == "ffmpeg frame extraction":
            for arg in cmd:
                text = str(arg)
                if "frame_%06d.png" in text:
                    counter.note_dir(Path(text).parent)
        counter.note_tree(scratch)
        return rc, err

    monkeypatch.setattr(clean_video, "_run_ffmpeg", wrapped)

    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            counter.note_tree(scratch)
            stop.wait(0.01)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    yield counter
    stop.set()
    thread.join(timeout=1)
    counter.note_tree(scratch)


def _assert_frame_cap(counter: _FrameCounter) -> None:
    fsize = common._CHILD_RLIMIT_FSIZE
    assert counter.pngs <= MAX_EXTRACTED_FRAMES, (
        f"wrote {counter.pngs} frame PNGs, largest {counter.max_file} bytes, "
        f"each under the {fsize}-byte per-file rlimit; cap is {MAX_EXTRACTED_FRAMES}"
    )
    if counter.pngs:
        assert counter.max_file < fsize


def _backend_dir(tmp_path: Path, remove_pixel: str) -> Path:
    directory = tmp_path / f"{remove_pixel}-backend"
    directory.mkdir()
    return directory


def _purify_kwargs(remove_pixel: str, backend: Path) -> dict[str, str]:
    if remove_pixel == "ctrlregen":
        return {"ctrlregen_dir": str(backend)}
    return {"markdiffusion_dir": str(backend)}


def _install_clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Move time.monotonic / time.time / time.perf_counter together."""
    offset = {"seconds": 0.0}
    real_mono = time.monotonic
    real_perf = time.perf_counter
    real_time = time.time

    def monotonic() -> float:
        return real_mono() + offset["seconds"]

    def perf_counter() -> float:
        return real_perf() + offset["seconds"]

    def wall() -> float:
        return real_time() + offset["seconds"]

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(time, "perf_counter", perf_counter)
    monkeypatch.setattr(time, "time", wall)
    return offset


def _install_purifier(monkeypatch: pytest.MonkeyPatch, offset: dict[str, float] | None):
    """Run every step of run_* up to the model process, then return success.

    The real adapter builds the command and passes ``timeout`` to
    ``subprocess.run``. Only the model process is replaced, because the
    CtrlRegen / MarkDiffusion checkouts are not installed here. When ``offset``
    is set, each invocation spends ``BUDGET_STEP_SECONDS`` on the patched clock.
    """
    calls: list[int | None] = []
    real_run = subprocess.run

    def wrapped(cmd, *args, **kwargs):
        text = " ".join(str(part) for part in cmd) if isinstance(cmd, (list, tuple)) else ""
        if any(script in text for script in _PURIFIER_SCRIPTS):
            timeout = kwargs.get("timeout")
            calls.append(timeout if isinstance(timeout, int) else None)
            if offset is not None:
                offset["seconds"] += BUDGET_STEP_SECONDS
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", wrapped)
    return calls


def _assert_budget(calls: list[int | None], budget: int) -> None:
    limit = budget // BUDGET_STEP_SECONDS + 1
    assert len(calls) <= limit, (
        f"{len(calls)} purification calls each charged {BUDGET_STEP_SECONDS}s "
        f"on the clock; a {budget}s total budget allows at most {limit}"
    )


@needs_ffmpeg
@pytest.mark.parametrize("remove_pixel", ["ctrlregen", "diffusion"])
def test_video_purify_caps_extracted_frames(tmp_path, frame_counter, remove_pixel):
    src = tmp_path / "bomb.mp4"
    _high_frame_mp4(src, BOMB_FRAMES)
    assert _duration_seconds(src) < 5
    assert src.stat().st_size < 256 << 20

    result = clean_video.video_purify(
        src,
        tmp_path / "out.mp4",
        remove_pixel=remove_pixel,
        **_purify_kwargs(remove_pixel, _backend_dir(tmp_path, remove_pixel)),
    )
    _assert_frame_cap(frame_counter)
    # A refusal is fine. A success must be a sample, not the whole bomb.
    if result.get("available"):
        assert result["frames_total"] <= MAX_EXTRACTED_FRAMES
        assert result["frames_purified"] <= MAX_EXTRACTED_FRAMES


@needs_ffmpeg
@pytest.mark.parametrize("remove_pixel", ["ctrlregen", "diffusion"])
def test_default_timeout_is_one_budget_for_every_frame(tmp_path, monkeypatch, remove_pixel):
    src = tmp_path / "clip.mp4"
    _high_frame_mp4(src, DEADLINE_FRAMES, rate=24)
    offset = _install_clock(monkeypatch)
    calls = _install_purifier(monkeypatch, offset)

    clean_video.video_purify(
        src,
        tmp_path / "out.mp4",
        remove_pixel=remove_pixel,
        **_purify_kwargs(remove_pixel, _backend_dir(tmp_path, remove_pixel)),
    )
    _assert_budget(calls, TOTAL_BUDGET_SECONDS)


@needs_ffmpeg
def test_explicit_timeout_is_one_budget_for_every_frame(tmp_path, monkeypatch):
    src = tmp_path / "clip.mp4"
    _high_frame_mp4(src, DEADLINE_FRAMES, rate=24)
    budget = 600
    offset = _install_clock(monkeypatch)
    calls = _install_purifier(monkeypatch, offset)

    clean_video.video_purify(
        src,
        tmp_path / "out.mp4",
        remove_pixel="ctrlregen",
        timeout=budget,
        ctrlregen_dir=str(_backend_dir(tmp_path, "ctrlregen")),
    )
    _assert_budget(calls, budget)


@needs_ffmpeg
def test_short_clip_under_the_cap_purifies_every_frame(tmp_path, monkeypatch, frame_counter):
    """A clip under the cap still has every frame purified when each one is cheap."""
    src = tmp_path / "short.mp4"
    _high_frame_mp4(src, SHORT_CLIP_FRAMES, rate=20)
    calls = _install_purifier(monkeypatch, offset=None)

    result = clean_video.video_purify(
        src,
        tmp_path / "out.mp4",
        remove_pixel="ctrlregen",
        ctrlregen_dir=str(_backend_dir(tmp_path, "ctrlregen")),
    )
    assert result["available"] is True
    assert result["frames_total"] == SHORT_CLIP_FRAMES
    assert result["frames_purified"] == SHORT_CLIP_FRAMES
    assert len(calls) == SHORT_CLIP_FRAMES
    assert frame_counter.pngs == SHORT_CLIP_FRAMES


def _serve():
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, thread


def _post(conn: http.client.HTTPConnection, path: str, payload: dict) -> tuple[int, dict]:
    conn.request(
        "POST",
        path,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, json.loads(data) if data else {}


@needs_ffmpeg
def test_clean_post_caps_frames_for_ctrlregen(tmp_path, monkeypatch, frame_counter):
    """Unauthenticated POST /clean reaches extraction when the backend dir exists.

    The default API key is empty (server.API_KEY), so this request sends no
    Authorization header. clean_av only rewrites metadata boxes; the coded
    frames are what ffmpeg then expands.
    """
    monkeypatch.setattr(server, "API_KEY", "")
    backend = _backend_dir(tmp_path, "ctrlregen")
    monkeypatch.setenv("NOAI_WATERMARK_DIR", str(backend))
    monkeypatch.delenv("MARKDIFFUSION_DIR", raising=False)

    src = tmp_path / "bomb.mp4"
    _high_frame_mp4(src, BOMB_FRAMES)
    body = {
        "file": base64.b64encode(src.read_bytes()).decode("ascii"),
        "name": "clip.mp4",
        "options": {"remove_pixel": "ctrlregen"},
    }
    srv, thread = _serve()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=120)
        status, payload = _post(conn, "/clean", body)
        conn.close()
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)

    _assert_frame_cap(frame_counter)
    if status == 200 and payload.get("report", {}).get("pixel_removal", {}).get("available"):
        pix = payload["report"]["pixel_removal"]
        assert pix["frames_total"] <= MAX_EXTRACTED_FRAMES


@needs_ffmpeg
def test_clean_batch_caps_frames_for_diffusion(tmp_path, monkeypatch, frame_counter):
    """POST /clean/batch uses the same purify path for a .mov upload."""
    monkeypatch.setattr(server, "API_KEY", "")
    backend = _backend_dir(tmp_path, "diffusion")
    monkeypatch.setenv("MARKDIFFUSION_DIR", str(backend))
    monkeypatch.delenv("NOAI_WATERMARK_DIR", raising=False)

    src = tmp_path / "bomb.mov"
    _high_frame_mp4(src, BOMB_FRAMES)
    body = {
        "files": [
            {
                "file": base64.b64encode(src.read_bytes()).decode("ascii"),
                "name": "clip.mov",
                "options": {"remove_pixel": "diffusion"},
            }
        ]
    }
    srv, thread = _serve()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=120)
        _post(conn, "/clean/batch", body)
        conn.close()
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)

    _assert_frame_cap(frame_counter)


@needs_ffmpeg
def test_cli_caps_extracted_frames(tmp_path, monkeypatch, frame_counter):
    backend = _backend_dir(tmp_path, "ctrlregen")
    monkeypatch.setenv("NOAI_WATERMARK_DIR", str(backend))
    src = tmp_path / "bomb.mp4"
    _high_frame_mp4(src, BOMB_FRAMES)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr(
        sys,
        "argv",
        ["clean_video.py", str(src), "-o", str(dest), "--remove-pixel", "ctrlregen"],
    )
    clean_video.main()
    _assert_frame_cap(frame_counter)


@needs_ffmpeg
def test_capped_extract_does_not_keep_the_full_soundtrack(tmp_path, monkeypatch):
    """A successful purify past the frame cap must not keep the rest of the audio."""
    src = tmp_path / "long.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=100",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-frames:v",
            str(BOMB_FRAMES),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    source_duration = _duration_seconds(src)
    assert source_duration > 2

    def _pass_through(_frame, _output, **_kwargs):
        return {"available": True, "bytes_out": 1}

    monkeypatch.setattr(clean_video, "run_ctrlregen_clean", _pass_through)
    dest = tmp_path / "out.mp4"
    result = clean_video.video_purify(
        src,
        dest,
        remove_pixel="ctrlregen",
        ctrlregen_dir=str(_backend_dir(tmp_path, "ctrlregen")),
    )
    assert result["available"] is True
    assert result["frames_total"] <= MAX_EXTRACTED_FRAMES
    assert _duration_seconds(dest) < source_duration - 1
