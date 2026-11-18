"""Markdown/HTML inspection must not keep one finding per repeated match.

/inspect decodes a .md/.html upload and walks its comments, <meta> tags, and
data-ai attributes. A body made of thousands of short provenance comments used
to produce one findings string per comment (and one regex match object per
closing tag held for the whole walk), so memory and the JSON response grew with
the comment count rather than staying near the input size.
"""

from __future__ import annotations

import json
import re
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server
from container_meta import (
    _MARKUP_COMMENT_CLOSE_RE,
    _MARKUP_COMMENT_OPEN_RE,
    _drop_provenance_comments,
    _iter_tag_blocks,
    inspect_html,
    inspect_markdown,
)

REPEATS = 20_000
# Far above any real document's per-kind finding count, far below REPEATS.
FINDINGS_BOUND = 50


def _peak_bytes(fn, *args):
    tracemalloc.start()
    try:
        fn(*args)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _findings_via_inspect(name: str, body: bytes) -> list[str]:
    res = server._inspect_payload(body, name, False)
    json.dumps(res)
    assert res["kind"] == "container"
    assert res["report"]["has_ai_metadata"] is True
    return res["report"]["findings"]


def test_inspect_markdown_upload_of_repeated_comments_is_bounded():
    findings = _findings_via_inspect("notes.md", b"<!-- claude -->\n" * REPEATS)
    comment_findings = [f for f in findings if f.startswith("comment:")]
    assert 1 <= len(comment_findings) <= FINDINGS_BOUND
    assert len(findings) <= FINDINGS_BOUND
    assert any(f"{REPEATS} total" in f for f in comment_findings)


def test_inspect_html_upload_of_repeated_comments_is_bounded():
    findings = _findings_via_inspect("page.html", b"<!-- claude -->\n" * REPEATS)
    comment_findings = [f for f in findings if f.startswith("comment:")]
    assert 1 <= len(comment_findings) <= FINDINGS_BOUND
    assert len(findings) <= FINDINGS_BOUND


def test_inspect_html_repeated_meta_and_data_ai_are_bounded():
    unit = (
        '<meta name="generator" content="ChatGPT"><meta name="generator" content="WordPress 6.4">'
    )
    unit += '<p data-ai="1">x</p>'
    _c2pa, has_ai, findings, _ = inspect_html(unit * REPEATS)
    assert has_ai
    for prefix in ("meta:", "info: cms generator:", "attr:"):
        kind = [f for f in findings if f.startswith(prefix)]
        assert 1 <= len(kind) <= FINDINGS_BOUND, prefix
        assert any(f"{REPEATS} total" in f for f in kind), prefix
    assert len(findings) <= 3 * FINDINGS_BOUND


def test_c2pa_comment_past_the_sample_limit_still_sets_flag():
    tail = "<!-- C2PA manifest attached -->"
    md_c2pa, _ai, _f, _ = inspect_markdown("<!-- claude -->\n" * REPEATS + tail)
    html_c2pa, _ai, _f, _ = inspect_html("<!-- claude -->\n" * REPEATS + tail)
    assert md_c2pa is True
    assert html_c2pa is True


def test_markdown_inspection_memory_stays_near_input_size():
    text = "<!-- claude -->\n" * REPEATS
    assert _peak_bytes(inspect_markdown, text) < 2 * len(text)


def test_html_inspection_memory_stays_near_input_size():
    text = "<!-- claude -->\n" * REPEATS
    assert _peak_bytes(inspect_html, text) < 2 * len(text)


def test_tag_block_walk_does_not_hold_every_closing_match():
    text = "<!-- claude -->\n" * REPEATS

    def walk(t):
        for _ in _iter_tag_blocks(t, _MARKUP_COMMENT_OPEN_RE, _MARKUP_COMMENT_CLOSE_RE):
            pass

    assert _peak_bytes(walk, text) < len(text) // 4


def test_comment_walk_matches_lazy_regex_semantics():
    lazy = re.compile(r"<!--.*?--!?>", re.DOTALL)
    samples = [
        "",
        "<!-->",
        "<!--->",
        "<!---->",
        "<!-- a --><!-- b -->",
        "<!-- a --!><!-- b",
        "--> <!-- a -->",
        "<!-- <!-- nested --> -->",
        "<!-- unclosed",
        "x<!-- a -->y<!--b-->z<!--",
        "<!-- a --> --> <!-- b --!>",
    ]
    for sample in samples:
        expected = [m.span() for m in lazy.finditer(sample)]
        got = [
            (os_, ce)
            for os_, _oe, _cs, ce in _iter_tag_blocks(
                sample, _MARKUP_COMMENT_OPEN_RE, _MARKUP_COMMENT_CLOSE_RE
            )
        ]
        assert got == expected, (sample, expected, got)


def test_clean_still_drops_every_repeated_comment():
    text = "<!-- claude -->\n" * REPEATS
    cleaned, n = _drop_provenance_comments(text)
    assert n == REPEATS
    assert cleaned == "\n" * REPEATS
