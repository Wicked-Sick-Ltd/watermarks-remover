"""Stylometry must not hold one Python object per token of an upload.

POST /inspect and POST /detect run score_text_stylometry over every text file.
Materializing each word (and each sentence / phrase match) as its own list
entry made peak memory a large multiple of the upload, so a body of short
repeated words near MAX_INPUT_BYTES could push the threaded server into swap or
an OOM kill. The upload itself already sits in memory a few times over (base64
body, bytes, decoded str); the scorer must stay within a small constant factor
of that, independent of how many tokens the text holds.
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server
from score_stylometry import score_text_stylometry

UPLOAD_BYTES = 600_000
# Peak traced allocation per upload byte. The unbounded tokenizer ran at
# roughly 30-56x on these inputs; a streaming pass needs only transient copies
# of the decoded text.
MAX_PEAK_RATIO = 6.0

SHAPES = {
    "one-letter words": "a ",
    "two-letter words": "ab ",
    "one-word sentences": "ab. ",
    "one-word lines": "ab.\n",
    "repeated cadence phrase": "boasts ",
}


def _upload(unit: str) -> bytes:
    return (unit * (UPLOAD_BYTES // len(unit))).encode("ascii")


def _peak_ratio(fn, data: bytes) -> float:
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / len(data)


@pytest.mark.parametrize("unit", list(SHAPES.values()), ids=list(SHAPES))
def test_detect_text_upload_memory_is_bounded(unit: str) -> None:
    data = _upload(unit)
    result: dict = {}

    def run() -> None:
        result.update(server._detect_payload(data, "x.txt"))

    ratio = _peak_ratio(run, data)
    assert ratio < MAX_PEAK_RATIO, f"/detect peak was {ratio:.1f}x the upload"
    styl = next(d for d in result["detections"] if d["detector"] == "stylometry")
    assert styl["word_count"] == len(data) // len(unit)


def test_inspect_text_upload_memory_is_bounded() -> None:
    data = _upload("ab ")
    ratio = _peak_ratio(lambda: server._inspect_payload(data, "x.txt", run_detect=False), data)
    assert ratio < MAX_PEAK_RATIO, f"/inspect peak was {ratio:.1f}x the upload"


def test_streaming_scorer_reports_full_counts() -> None:
    text = "ab " * 120_000 + "It boasts a rich tapestry of ideas. " * 50
    report = score_text_stylometry(text)
    assert report.word_count == 120_000 + 7 * 50
    assert report.sentence_count == 50
    markers = {m["phrase"]: m for m in report.to_dict()["matched_markers"]}
    boasts = markers["boasts (copula avoidance)"]
    assert boasts["count"] == 50
    assert len(boasts["samples"]) == 3
    assert len(boasts["spans"]) == 10
