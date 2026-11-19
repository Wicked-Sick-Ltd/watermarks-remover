#!/usr/bin/env python3
"""Optional SynthID pixel-domain scorer backed by an external reverse-SynthID checkout.

This script does NOT vendor upstream code. It imports the scorer from a
user-provided checkout (https://github.com/aloshdenny/reverse-SynthID) at
runtime, using that environment's optional dependencies (numpy, opencv,
scipy, PyWavelets, scikit-learn, Pillow).

Exit codes:
  0  scored successfully
  1  scorer runtime error
  2  bad input (missing/unreadable image, bad args, over decoded-size budget)
  3  scorer unavailable (not configured / missing deps / missing codebook)

``cv2.imread`` (IMREAD_COLOR) allocates ``width * height * 3`` bytes, including
for grayscale files. A header whose declared bitmap exceeds Pillow 13's
decompression-bomb hard limit (twice ``Image.MAX_IMAGE_PIXELS``) is refused
from the PNG IHDR or JPEG SOF before that decode, and before OpenCV is imported.

The scoring logic lives in :func:score_file so the CLI and the HTTP
sidecar (synthid_score_server.py) share one implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Pillow 13 src/PIL/Image.py: DecompressionBombError above 2 * MAX_IMAGE_PIXELS.
# MAX_IMAGE_PIXELS is 1024**3 // 4 // 3. cv2.imread's default flag is IMREAD_COLOR,
# so the bitmap is 3 channels even when the container declares one.
_DECODED_CHANNELS = 3
_PILLOW_HARD_PIXEL_LIMIT = 2 * (1024 * 1024 * 1024 // 4 // 3)
_JPEG_SOF = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    """Width and height from a PNG IHDR. Does not read the compressed image."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """Width and height from the first JPEG SOF marker. Stops before entropy data.

    Only a byte after ``0xFF`` is a marker. A stray non-FF byte is skipped, and
    ``FF 00`` is a stuffed byte. Treating either as a length-prefixed segment
    walks off the real SOF and the later decode still allocates the bitmap.
    """
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = handle.read(1)
            if not byte:
                return None
            if byte != b"\xff":
                continue
            while True:
                nxt = handle.read(1)
                if not nxt:
                    return None
                if nxt != b"\xff":
                    break
            marker = nxt[0]
            if marker == 0x00:
                continue
            if marker in (0xD9, 0xDA):
                return None
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) < 2:
                return None
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                return None
            payload_length = segment_length - 2
            if marker in _JPEG_SOF:
                if payload_length < 5:
                    return None
                payload = handle.read(5)
                if len(payload) < 5:
                    return None
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                return width, height
            handle.seek(payload_length, 1)


def _refuse_over_decoded_budget(path: Path) -> bool:
    """Print a pixel-budget refusal and return True when the header is too large.

    The check uses the header alone. It does not decode, and it runs before
    OpenCV is imported so a missing ``cv2`` cannot hide an over-budget image.
    """
    with path.open("rb") as handle:
        sig = handle.read(8)
    if sig == b"\x89PNG\r\n\x1a\n":
        dims = _png_dimensions(path)
    elif sig[:2] == b"\xff\xd8":
        dims = _jpeg_dimensions(path)
    else:
        dims = None
    if dims is None:
        return False
    width, height = dims
    if width <= 0 or height <= 0:
        return False
    decoded = width * height * _DECODED_CHANNELS
    if decoded <= _PILLOW_HARD_PIXEL_LIMIT:
        return False
    print(
        f"image dimensions {width}x{height} exceed the decoded pixel budget "
        f"({_DECODED_CHANNELS} channels, limit {_PILLOW_HARD_PIXEL_LIMIT} pixels)",
        file=sys.stderr,
    )
    return True


def resolve_upstream(raw: str | None) -> Path | None:
    if not raw:
        return None
    upstream = Path(raw).expanduser().resolve()
    if not upstream.is_dir():
        return None
    return upstream


def score_file(
    path: Path,
    *,
    upstream_dir: str | None = None,
    codebook: Path | None = None,
    model: str | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Score *path* with the reverse-SynthID extractor.

    Returns (exit_code, payload) matching the CLI exit-code contract:
    0 = scored (payload present), 2 = bad input (payload None),
    3 = scorer unavailable (payload None). Errors are printed to stderr so
    callers parsing stdout JSON are never corrupted.
    """
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2, None

    if _refuse_over_decoded_budget(path):
        return 2, None

    raw_upstream = upstream_dir or os.environ.get("REVERSE_SYNTHID_DIR")
    upstream = resolve_upstream(str(raw_upstream) if raw_upstream else None)
    if upstream is None:
        print(
            "SynthID scorer not configured: set REVERSE_SYNTHID_DIR or pass --upstream-dir",
            file=sys.stderr,
        )
        return 3, None

    extraction = upstream / "src" / "extraction"
    if not extraction.is_dir():
        print(f"upstream extraction dir not found: {extraction}", file=sys.stderr)
        return 3, None

    codebook_path = codebook or upstream / "artifacts" / "spectral_codebook_v4.npz"
    codebook_path = Path(codebook_path).expanduser().resolve()
    if not codebook_path.is_file():
        print(f"codebook not found: {codebook_path}", file=sys.stderr)
        return 3, None

    sys.path.insert(0, str(extraction))
    try:
        import cv2
        from robust_extractor import RobustSynthIDExtractor
        from synthid_bypass_v4 import SpectralCodebookV4
    except ImportError as e:
        print(f"optional scorer dependencies missing: {e}", file=sys.stderr)
        return 3, None

    try:
        img = cv2.imread(str(path))
        if img is None:
            print(f"could not load image: {path}", file=sys.stderr)
            return 2, None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Upstream prints progress ("CodebookV4 loaded: ...") straight to
        # stdout, which corrupts --json for any caller that parses us
        # (image_meta.py json.loads our stdout). Keep stdout ours alone.
        with contextlib.redirect_stdout(sys.stderr):
            codebook_v4 = SpectralCodebookV4()
            codebook_v4.load(str(codebook_path))

            extractor = RobustSynthIDExtractor()
            result = extractor.detect_from_v4_codebook(rgb, codebook_v4, model=model)
    except Exception as e:
        print(f"scorer error: {e}", file=sys.stderr)
        return 1, None

    payload = {
        "available": True,
        "upstream_dir": str(upstream),
        "codebook": str(codebook_path),
        "model": model,
        "profile_key": result.details.get("profile_key"),
        "exact_match": result.details.get("exact_match"),
        "is_watermarked": result.is_watermarked,
        "confidence": result.confidence,
        "phase_match": result.phase_match,
        "per_channel_scores": result.details.get("per_channel_scores"),
        "per_channel_n": result.details.get("per_channel_n"),
        "multi_scale_consistency": result.multi_scale_consistency,
    }
    return 0, payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Image to score (PNG/JPEG/etc.)")
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=None,
        help="reverse-SynthID checkout root (default: $REVERSE_SYNTHID_DIR)",
    )
    p.add_argument(
        "--codebook",
        type=Path,
        default=None,
        help="spectral_codebook_v4.npz path (default: <upstream>/artifacts/)",
    )
    p.add_argument("--model", type=str, default=None, help="Optional model hint")
    p.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = p.parse_args()

    code, payload = score_file(
        args.path,
        upstream_dir=str(args.upstream_dir) if args.upstream_dir else None,
        codebook=args.codebook,
        model=args.model,
    )
    if code != 0:
        return code

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        label = "yes" if payload["is_watermarked"] else "no"
        print(f"SynthID score: confidence {payload['confidence']:.3f} (watermarked: {label})")
        print(f"  phase_match: {payload['phase_match']:.3f}")
        if payload.get("profile_key"):
            print(f"  profile: {payload['profile_key']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
