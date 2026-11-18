"""SynthID scoring must reject oversized images before OpenCV allocates them.

``score_file`` decodes with ``cv2.imread`` (default ``IMREAD_COLOR``) and never
measures the bitmap that header declares. OpenCV 5.0.0, the version pinned by
``service/scripts/requirements-synthid-scorer.txt`` (``opencv-python==5.0.0.93``),
allocates ``width * height * 3`` bytes after a header check whose default cap
is ``1 << 30`` pixels (``modules/imgcodecs/src/loadsave.cpp``,
``CV_IO_MAX_IMAGE_PIXELS``). A few hundred bytes whose IHDR or SOF stays under
that cap still become hundreds of megabytes, and the same path reaches
multiple gigabytes as the declared size approaches the cap.

Pillow 13.0.0 (the scorer image's ``Pillow==13.0.0``) sets
``Image.MAX_IMAGE_PIXELS`` to ``1024**3 // 4 // 3`` and raises
``DecompressionBombError`` above twice that, 178_956_970. These tests require
the scorer to refuse, before decoding, when ``width * height * 3`` exceeds
that hard limit. The ``3`` is the channel count ``cv2.imread`` allocates:
grayscale files come back as 3-channel arrays too.

The refusal has to be visible on stderr (or the sidecar JSON error) as a
pixel/dimension budget error, and the scoring process's peak RSS has to stay
within 64 MiB of scoring a 64x64 image. A decode that later returns "could
not load image", including one whose allocation hit a resource limit, still
fails.

Covered entry points, all of which call ``score_file`` on the uploaded bytes
when the scorer is configured and the API key is empty (the default):

- ``score_synthid.score_file`` and the ``score_synthid.py`` CLI
- sidecar ``POST /score`` (``synthid_score_server.py``)
- core ``POST /inspect``, ``/detect``, ``/clean`` and the matching ``/batch``
  routes, which forward the bytes when ``WATERMARKS_SYNTHID_SCORER_URL`` is set

Not covered here: WebP, BMP, GIF, TIFF, AVIF, and HEIC use the same
``score_file`` call once classified as images. ``inspect_image.py`` and
``clean_image.py`` call the same ``inspect_image`` / ``clean_image`` functions
the core routes do. Compose memory limits and a required API key are not a
substitute: a client who can already upload an image still reaches ``imread``.
"""

from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
THIS_FILE = Path(__file__).resolve()

# Pillow 13.0.0 src/PIL/Image.py: DecompressionBombError above 2 * MAX_IMAGE_PIXELS.
PILLOW_HARD_PIXEL_LIMIT = 2 * (1024 * 1024 * 1024 // 4 // 3)
# Default cv2.imread flag is IMREAD_COLOR, so the bitmap is 3 channels even
# when the container declares one.
DECODED_CHANNELS = 3
OVER_SIDE = 8000
UNDER_SIDE = 64
RSS_SLACK_KIB = 64 * 1024

assert PILLOW_HARD_PIXEL_LIMIT == 178_956_970
assert OVER_SIDE * OVER_SIDE * DECODED_CHANNELS > PILLOW_HARD_PIXEL_LIMIT
# A width*height check against Pillow's hard limit would still accept this
# side length. The scorer has to count the channels imread will allocate.
assert OVER_SIDE * OVER_SIDE <= PILLOW_HARD_PIXEL_LIMIT
assert UNDER_SIDE * UNDER_SIDE * DECODED_CHANNELS <= PILLOW_HARD_PIXEL_LIMIT

_STUB_EXTRACTOR = """\
class _Result:
    def __init__(self):
        self.is_watermarked = False
        self.confidence = 0.0
        self.phase_match = 0.0
        self.multi_scale_consistency = None
        self.details = {}


class RobustSynthIDExtractor:
    def detect_from_v4_codebook(self, rgb, codebook, model=None):
        return _Result()
"""

_STUB_CODEBOOK = """\
class SpectralCodebookV4:
    def load(self, path):
        return None
"""

# 8x8 baseline JPEG from OpenCV's encoder (631 bytes). Tests patch the SOF0
# marker (FF C0) so the header declares OVER_SIDE without growing the file.
# libjpeg still returns the full bitmap for that header.
_TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYF"
    "BgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoK"
    "CgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAIAAgDASIA"
    "AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQA"
    "AAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3"
    "ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWm"
    "p6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEA"
    "AwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSEx"
    "BhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElK"
    "U1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3"
    "uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+ii"
    "igD/2Q=="
)

_CORE_ROUTES = (
    "/inspect",
    "/detect",
    "/clean",
    "/inspect/batch",
    "/detect/batch",
    "/clean/batch",
)


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _compressed_png(side: int, *, color_type: int) -> bytes:
    """Valid PNG of ``side`` x ``side`` whose IDAT is compressed zeros.

    A truncated IDAT makes OpenCV 5 fail inside ``readHeader`` and never
    allocate. The bomb has to be a real image so ``cv2.imread`` builds the
    bitmap the IHDR describes.
    """
    channels = 1 if color_type == 0 else 3
    ihdr = struct.pack(">IIBBBBB", side, side, 8, color_type, 0, 0, 0)
    row = b"\x00" + bytes(side * channels)
    compressor = zlib.compressobj(9)
    compressed = bytearray()
    for _ in range(side):
        compressed += compressor.compress(row)
    compressed += compressor.flush()
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", bytes(compressed))
        + _png_chunk(b"IEND", b"")
    )


def _jpeg_with_sof(side: int) -> bytes:
    """Same tiny JPEG, with the SOF0 height and width replaced."""
    data = bytearray(_TINY_JPEG)
    marker = data.find(b"\xff\xc0")
    if marker < 0:
        raise RuntimeError("SOF0 missing from the baseline JPEG")
    # FF C0, length(2), precision(1), height(2), width(2), ...
    data[marker + 5 : marker + 9] = struct.pack(">HH", side, side)
    return bytes(data)


def _make_upstream(root: Path) -> Path:
    """Checkout layout ``score_file`` requires before it reaches ``imread``."""
    upstream = root / "upstream"
    extraction = upstream / "src" / "extraction"
    extraction.mkdir(parents=True)
    (extraction / "robust_extractor.py").write_text(_STUB_EXTRACTOR)
    (extraction / "synthid_bypass_v4.py").write_text(_STUB_CODEBOOK)
    artifacts = upstream / "artifacts"
    artifacts.mkdir()
    # Presence is checked before the decoder runs. The stub load() ignores it.
    (artifacts / "spectral_codebook_v4.npz").write_bytes(b"")
    return upstream


def _child_env(upstream: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["REVERSE_SYNTHID_DIR"] = str(upstream)
    env["PYTHONUNBUFFERED"] = "1"
    # Empty key is the default: ``_authorized`` accepts every caller.
    env["WATERMARKS_SYNTHID_SCORER_API_KEY"] = ""
    env.pop("WATERMARKS_SYNTHID_SCORER_URL", None)
    # Product images do not set these. A developer override would hide the bug.
    for name in (
        "OPENCV_IO_MAX_IMAGE_PIXELS",
        "OPENCV_IO_MAX_IMAGE_WIDTH",
        "OPENCV_IO_MAX_IMAGE_HEIGHT",
    ):
        env.pop(name, None)
    return env


def _peak_rss_kib() -> int:
    """Peak resident set of this process, in KiB.

    Linux publishes the unit on the VmHWM line (``kB``). Darwin's
    ``getrusage`` reports bytes. Windows has no ``resource`` module;
    ``GetProcessMemoryInfo`` reports bytes.
    """
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("VmHWM:"):
                continue
            parts = line.split()
            value = int(parts[1])
            unit = parts[2].lower() if len(parts) > 2 else "kb"
            if unit == "kb":
                return value
            if unit == "mb":
                return value * 1024
            raise RuntimeError(f"unexpected VmHWM unit: {line!r}")
        raise RuntimeError("VmHWM missing from /proc/self/status")
    if sys.platform == "win32":
        return _windows_peak_rss_kib()
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # BSD/Darwin ru_maxrss is bytes. Other Unix platforms that lack /proc
    # follow the Linux getrusage convention of kilobytes.
    if sys.platform == "darwin":
        return raw // 1024
    return raw


def _cv2_importable() -> bool:
    try:
        import cv2
    except ImportError:
        return False
    return hasattr(cv2, "imread")


def _mentions_budget(*parts: object) -> bool:
    blob = "\n".join(str(part) for part in parts if part is not None).lower()
    talks_about_size = any(word in blob for word in ("pixel", "dimension", "decompression bomb"))
    talks_about_refusal = any(
        word in blob for word in ("exceed", "budget", "limit", "reject", "refus")
    )
    return talks_about_size and talks_about_refusal


def _score_succeeded(payload: object) -> bool:
    return (
        isinstance(payload, dict) and payload.get("available") is True and "confidence" in payload
    )


def _assert_budget_refusal(result: dict, baseline_kib: int) -> None:
    text_parts: list[object] = [result.get("stderr"), result.get("stdout")]
    if result.get("payload") is not None:
        text_parts.append(json.dumps(result["payload"]))
    if result.get("body") is not None:
        text_parts.append(json.dumps(result["body"]))
    problems: list[str] = []
    if not _mentions_budget(*text_parts):
        problems.append("response is not a pixel/dimension budget refusal")
    if _score_succeeded(result.get("payload")) or _score_succeeded(result.get("body")):
        problems.append("image was scored")
    peak = result["maxrss_kib"]
    if peak >= baseline_kib + RSS_SLACK_KIB:
        problems.append(
            f"peak RSS {peak} KiB exceeded the {UNDER_SIDE}px baseline "
            f"{baseline_kib} KiB by more than {RSS_SLACK_KIB} KiB"
        )
    assert not problems, (
        "over-budget image reached the decoder: "
        + "; ".join(problems)
        + f" detail={ {k: result.get(k) for k in ('exit_code', 'status', 'stderr', 'maxrss_kib')} }"
    )


def _run_score(image: Path, upstream: Path, *, cli: bool = False) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(THIS_FILE),
            "cli" if cli else "score",
            str(image),
            str(upstream),
        ],
        capture_output=True,
        text=True,
        env=_child_env(upstream),
        timeout=180,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "exit_code": proc.returncode,
            "payload": None,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            # A killed worker never reports a peak. Treat that as unbounded.
            "maxrss_kib": baseline_fallback_unbounded(),
            "crashed": True,
        }
    reported = json.loads(proc.stdout)
    reported["stderr"] = (reported.get("stderr") or "") + proc.stderr
    reported.setdefault("crashed", False)
    return reported


def baseline_fallback_unbounded() -> int:
    return 1 << 30


def _windows_peak_rss_kib() -> int:
    """Peak working set via kernel32, with a real 64-bit argument prototype.

    ``ctypes.windll`` defaults to 32-bit argument slots. On 64-bit Windows that
    truncates the process handle and the counter pointer, and the call fails.
    """
    import ctypes.wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    current = kernel32.GetCurrentProcess
    current.restype = ctypes.wintypes.HANDLE
    query = kernel32.K32GetProcessMemoryInfo
    query.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.POINTER(_Counters),
        ctypes.wintypes.DWORD,
    ]
    query.restype = ctypes.wintypes.BOOL
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if not query(current(), ctypes.byref(counters), counters.cb):
        raise OSError(f"GetProcessMemoryInfo failed ({ctypes.get_last_error()})")
    return int(counters.PeakWorkingSetSize) // 1024


def _start_sidecar(upstream: Path) -> tuple[subprocess.Popen[str], Path]:
    """Start the sidecar with files for the port, RSS, and stderr.

    A stdout or stderr pipe can fill and block the child before it publishes
    its port. macOS does that during startup. Files do not.
    """
    fd, name = tempfile.mkstemp(prefix="wm-sidecar-port-")
    os.close(fd)
    port_path = Path(name)
    rss_path = port_path.with_name(port_path.name + ".rss")
    err_path = port_path.with_name(port_path.name + ".err")
    err_handle = err_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(THIS_FILE),
            "serve",
            str(upstream),
            str(port_path),
            str(rss_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=err_handle,
        stdin=subprocess.DEVNULL,
        env=_child_env(upstream),
    )
    err_handle.close()
    proc._rss_path = rss_path  # type: ignore[attr-defined]
    proc._err_path = err_path  # type: ignore[attr-defined]
    return proc, port_path


def _wait_port(proc: subprocess.Popen[str], port_path: Path, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = ""
        if port_path.is_file():
            text = port_path.read_text(encoding="ascii", errors="replace").strip()
        if text.isdigit():
            return int(text)
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    err = _captured_stderr(proc).strip()
    marker = (
        port_path.read_text(encoding="ascii", errors="replace").strip()
        if port_path.is_file()
        else ""
    )
    raise AssertionError(
        f"sidecar produced no port (returncode={proc.poll()}) marker={marker!r} stderr={err[:500]!r}"
    )


def _captured_stderr(proc: subprocess.Popen[str]) -> str:
    err_path = getattr(proc, "_err_path", None)
    if isinstance(err_path, Path) and err_path.is_file():
        return err_path.read_text(encoding="utf-8", errors="replace")
    return ""


def _read_stdout_line(proc: subprocess.Popen[str], timeout: float) -> str:
    """Wait until the sidecar's RSS file has one ``RSS <kib>`` line."""
    rss_path = getattr(proc, "_rss_path", None)
    if not isinstance(rss_path, Path):
        raise AssertionError("sidecar has no RSS file")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if rss_path.is_file():
            text = rss_path.read_text(encoding="ascii", errors="replace").strip()
            if text.startswith("RSS "):
                return text + "\n"
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    err = _captured_stderr(proc).strip()
    raise AssertionError(
        f"sidecar did not report RSS (returncode={proc.poll()}) stderr={err[:500]!r}"
    )


@contextlib.contextmanager
def _sidecar(upstream: Path):
    proc, port_path = _start_sidecar(upstream)
    try:
        port = _wait_port(proc, port_path, 30)
        yield proc, port
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        port_path.unlink(missing_ok=True)


def _post_sidecar(proc: subprocess.Popen[str], port: int, data: bytes) -> dict:
    body = json.dumps({"file": base64.b64encode(data).decode("ascii")}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
    try:
        conn.request(
            "POST",
            "/score",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    except (OSError, http.client.HTTPException) as exc:
        return {
            "status": None,
            "body": None,
            "stderr": str(exc),
            "maxrss_kib": baseline_fallback_unbounded(),
            "crashed": True,
        }
    finally:
        conn.close()
    rss_line = _read_stdout_line(proc, 30)
    if not rss_line.startswith("RSS "):
        raise AssertionError(f"sidecar did not report RSS: {rss_line!r}")
    parsed: object
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else None
    except json.JSONDecodeError:
        parsed = raw.decode("utf-8", errors="replace")
    return {
        "status": status,
        "body": parsed,
        "payload": parsed,
        "stderr": "",
        "stdout": "",
        "maxrss_kib": int(rss_line.split()[1]),
        "crashed": False,
    }


def _finish_sidecar(proc: subprocess.Popen[str]) -> str:
    """Stderr captured after the process has exited."""
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    text = _captured_stderr(proc)
    for attr in ("_err_path", "_rss_path"):
        path = getattr(proc, attr, None)
        if isinstance(path, Path):
            path.unlink(missing_ok=True)
    return text


def _core_request(route: str, data: bytes, name: str) -> dict:
    encoded = base64.b64encode(data).decode("ascii")
    entry = {"file": encoded, "name": name}
    if route.endswith("/batch"):
        return {"files": [entry]}
    return entry


def _response_scored_synthid(body: object) -> bool:
    if not isinstance(body, dict):
        return False
    items = body.get("results") if isinstance(body.get("results"), list) else [body]
    for item in items:
        if not isinstance(item, dict):
            continue
        report = item.get("report") if isinstance(item.get("report"), dict) else {}
        for key in ("synthid", "synthid_before", "synthid_after"):
            if _score_succeeded(report.get(key)):
                return True
        detections = item.get("detections")
        if isinstance(detections, list):
            for det in detections:
                if (
                    isinstance(det, dict)
                    and det.get("detector") == "synthid"
                    and _score_succeeded(det)
                ):
                    return True
    return False


@pytest.fixture(scope="module")
def upstream(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_upstream(tmp_path_factory.mktemp("synthid-upstream"))


@pytest.fixture(scope="module")
def under_budget_png(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("small") / "small.png"
    path.write_bytes(_compressed_png(UNDER_SIDE, color_type=2))
    return path


@pytest.fixture(scope="module")
def score_baseline(upstream: Path, under_budget_png: Path) -> dict:
    return _run_score(under_budget_png, upstream)


def test_under_budget_image_is_not_refused_for_size(score_baseline: dict) -> None:
    """A 64x64 image stays under the budget and must still be scorable."""
    assert not _mentions_budget(score_baseline.get("stderr"), score_baseline.get("payload"))
    if _cv2_importable():
        assert score_baseline["exit_code"] == 0
        assert _score_succeeded(score_baseline["payload"])
    else:
        assert score_baseline["exit_code"] == 3


def test_score_file_rejects_rgb_png_before_allocating(
    tmp_path: Path, upstream: Path, score_baseline: dict
) -> None:
    image = tmp_path / "bomb.png"
    data = _compressed_png(OVER_SIDE, color_type=2)
    assert len(data) < 256 * 1024
    image.write_bytes(data)
    result = _run_score(image, upstream)
    _assert_budget_refusal(result, score_baseline["maxrss_kib"])


def test_score_file_rejects_grayscale_png_before_allocating(
    tmp_path: Path, upstream: Path, score_baseline: dict
) -> None:
    """Grayscale IHDR still decodes to 3 channels under the default imread flag."""
    image = tmp_path / "gray.png"
    data = _compressed_png(OVER_SIDE, color_type=0)
    assert len(data) < 256 * 1024
    image.write_bytes(data)
    result = _run_score(image, upstream)
    _assert_budget_refusal(result, score_baseline["maxrss_kib"])


@pytest.mark.parametrize("prefix", [b"\x00", b"\xff\x00"], ids=["stray-byte", "stuffed-ff00"])
def test_score_file_rejects_jpeg_sof_after_padding(
    prefix: bytes, tmp_path: Path, upstream: Path, score_baseline: dict
) -> None:
    """A stray byte or a stuffed FF 00 before the markers must not hide the SOF."""
    image = tmp_path / "padded.jpg"
    data = _jpeg_with_sof(OVER_SIDE)
    assert data[:2] == b"\xff\xd8"
    image.write_bytes(data[:2] + prefix + data[2:])
    result = _run_score(image, upstream)
    _assert_budget_refusal(result, score_baseline["maxrss_kib"])


def test_small_jpeg_with_stuffed_byte_is_not_refused_for_size(
    tmp_path: Path, upstream: Path
) -> None:
    image = tmp_path / "small.jpg"
    assert _TINY_JPEG[:2] == b"\xff\xd8"
    image.write_bytes(_TINY_JPEG[:2] + b"\xff\x00" + _TINY_JPEG[2:])
    result = _run_score(image, upstream)
    assert not _mentions_budget(result.get("stderr"), result.get("payload"))


def test_score_file_rejects_jpeg_sof_before_allocating(
    tmp_path: Path, upstream: Path, score_baseline: dict
) -> None:
    image = tmp_path / "bomb.jpg"
    data = _jpeg_with_sof(OVER_SIDE)
    assert len(data) < 256 * 1024
    image.write_bytes(data)
    result = _run_score(image, upstream)
    _assert_budget_refusal(result, score_baseline["maxrss_kib"])


def test_score_synthid_cli_rejects_jpeg_sof_before_allocating(
    tmp_path: Path, upstream: Path, score_baseline: dict
) -> None:
    image = tmp_path / "bomb.jpg"
    image.write_bytes(_jpeg_with_sof(OVER_SIDE))
    result = _run_score(image, upstream, cli=True)
    _assert_budget_refusal(result, score_baseline["maxrss_kib"])


@pytest.fixture(scope="module")
def sidecar_baseline(upstream: Path, under_budget_png: Path) -> dict:
    with _sidecar(upstream) as (proc, port):
        result = _post_sidecar(proc, port, under_budget_png.read_bytes())
        result["stderr"] = _finish_sidecar(proc)
    return result


def test_sidecar_score_rejects_rgb_png(upstream: Path, sidecar_baseline: dict) -> None:
    data = _compressed_png(OVER_SIDE, color_type=2)
    assert len(data) < 256 * 1024
    with _sidecar(upstream) as (proc, port):
        result = _post_sidecar(proc, port, data)
        result["stderr"] = _finish_sidecar(proc) + "\n" + result.get("stderr", "")
    _assert_budget_refusal(result, sidecar_baseline["maxrss_kib"])
    # No Authorization header: the default empty API key must not be what stops this.
    assert result["status"] != 401


def test_sidecar_score_rejects_jpeg_sof(upstream: Path, sidecar_baseline: dict) -> None:
    data = _jpeg_with_sof(OVER_SIDE)
    with _sidecar(upstream) as (proc, port):
        result = _post_sidecar(proc, port, data)
        result["stderr"] = _finish_sidecar(proc) + "\n" + result.get("stderr", "")
    _assert_budget_refusal(result, sidecar_baseline["maxrss_kib"])
    assert result["status"] != 401


@pytest.mark.parametrize("route", _CORE_ROUTES)
def test_core_forwards_over_budget_jpeg_to_scorer(
    route: str, upstream: Path, sidecar_baseline: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core inspect/detect/clean (and their batch routes) forward the bytes.

    The allocation happens in the sidecar. Refusing inside core after the
    sidecar has already decoded still fails the RSS check.
    """
    data = _jpeg_with_sof(OVER_SIDE)
    name = "bomb.jpg"
    with _sidecar(upstream) as (proc, port):
        url = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("WATERMARKS_SYNTHID_SCORER_URL", url)
        monkeypatch.delenv("WATERMARKS_SYNTHID_SCORER_API_KEY", raising=False)
        sys.path.insert(0, str(SCRIPTS))
        import server

        monkeypatch.setattr(server, "API_KEY", "")
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=180)
        try:
            payload = _core_request(route, data, name)
            raw = json.dumps(payload).encode("utf-8")
            conn.request(
                "POST",
                route,
                body=raw,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            core_raw = resp.read()
            core_status = resp.status
        finally:
            conn.close()
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
        result = {
            "status": None,
            "body": None,
            "stderr": "",
            "maxrss_kib": baseline_fallback_unbounded(),
        }
        try:
            rss_line = _read_stdout_line(proc, 30)
        except AssertionError:
            rss_line = ""
        if rss_line.startswith("RSS "):
            result["maxrss_kib"] = int(rss_line.split()[1])
        result["stderr"] = _finish_sidecar(proc)
        try:
            core_body = json.loads(core_raw.decode("utf-8")) if core_raw else None
        except json.JSONDecodeError:
            core_body = core_raw.decode("utf-8", errors="replace")
        result["core_status"] = core_status
        result["core_body"] = core_body
    _assert_budget_refusal(result, sidecar_baseline["maxrss_kib"])
    assert not _response_scored_synthid(result["core_body"])
    assert result["core_status"] != 401


def test_core_inspect_forwards_over_budget_png(
    upstream: Path, sidecar_baseline: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _compressed_png(OVER_SIDE, color_type=2)
    assert len(data) < 256 * 1024
    with _sidecar(upstream) as (proc, port):
        monkeypatch.setenv("WATERMARKS_SYNTHID_SCORER_URL", f"http://127.0.0.1:{port}")
        monkeypatch.delenv("WATERMARKS_SYNTHID_SCORER_API_KEY", raising=False)
        sys.path.insert(0, str(SCRIPTS))
        import server

        monkeypatch.setattr(server, "API_KEY", "")
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=180)
        try:
            raw = json.dumps(_core_request("/inspect", data, "bomb.png")).encode("utf-8")
            conn.request(
                "POST",
                "/inspect",
                body=raw,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            core_raw = resp.read()
            core_status = resp.status
        finally:
            conn.close()
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
        rss_line = _read_stdout_line(proc, 30)
        result = {
            "status": None,
            "body": None,
            "stderr": _finish_sidecar(proc),
            "maxrss_kib": int(rss_line.split()[1]),
            "core_status": core_status,
            "core_body": json.loads(core_raw.decode("utf-8")),
        }
    _assert_budget_refusal(result, sidecar_baseline["maxrss_kib"])
    assert not _response_scored_synthid(result["core_body"])


def _worker_score(image: str, upstream: str, *, cli: bool) -> int:
    sys.path.insert(0, str(SCRIPTS))
    import score_synthid

    stdout = io.StringIO()
    stderr = io.StringIO()
    if cli:
        sys.argv = [
            "score_synthid.py",
            image,
            "--upstream-dir",
            upstream,
            "--json",
        ]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = score_synthid.main()
    else:
        with contextlib.redirect_stderr(stderr):
            code, payload = score_synthid.score_file(Path(image), upstream_dir=upstream)
        stdout.write(json.dumps(payload) if payload is not None else "")
    try:
        rss = _peak_rss_kib()
    except OSError as exc:
        stderr.write(f"\n{exc}\n")
        rss = baseline_fallback_unbounded()
    json.dump(
        {
            "exit_code": code,
            "payload": json.loads(stdout.getvalue()) if stdout.getvalue().strip() else None,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "maxrss_kib": rss,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


def _worker_serve(upstream: str) -> int:
    marker = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    rss_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    if marker is not None:
        marker.write_text("starting\n", encoding="ascii")
    os.environ["REVERSE_SYNTHID_DIR"] = upstream
    os.environ["WATERMARKS_SYNTHID_SCORER_API_KEY"] = ""
    sys.path.insert(0, str(SCRIPTS))
    import synthid_score_server as sidecar

    sidecar.API_KEY = ""

    class _MeasuringHandler(sidecar.Handler):
        def do_POST(self) -> None:
            super().do_POST()
            try:
                rss = _peak_rss_kib()
            except OSError as exc:
                sys.stderr.write(f"{exc}\n")
                sys.stderr.flush()
                rss = baseline_fallback_unbounded()
            if rss_path is not None:
                rss_path.write_text(f"RSS {rss}\n", encoding="ascii")

    httpd = sidecar.ThreadingHTTPServer(("127.0.0.1", 0), _MeasuringHandler)
    if marker is not None:
        marker.write_text(f"{httpd.server_address[1]}\n", encoding="ascii")
    httpd.serve_forever()
    return 0


def _main() -> int:
    mode = sys.argv[1]
    if mode in ("score", "cli"):
        return _worker_score(sys.argv[2], sys.argv[3], cli=mode == "cli")
    if mode == "serve":
        return _worker_serve(sys.argv[2])
    raise SystemExit(f"unknown worker mode {mode}")


if __name__ == "__main__":
    raise SystemExit(_main())
