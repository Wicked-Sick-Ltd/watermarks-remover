#!/usr/bin/env python3
"""Zero-LLM statistical & stylometric AI-text detector.

Evaluates text for high-frequency AI cadence markers, sentence-length
burstiness variance, lexical diversity (MATTR), and structural uniformity.
Stdlib-only, no PyTorch or external model dependencies.

Exit codes:
  0  clean (score < threshold)
  1  suspicious / AI stylometric signals detected (score >= threshold)
  2  bad input (missing file, binary data, bad args)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from common import (  # noqa: E402
    classify_finding_confidence,
    emit_json,
    read_text_input,
)

DEFAULT_THRESHOLD = 0.65
MIN_SAMPLE_WORDS = 30
FULL_WEIGHT_WORDS = 100

# High-frequency formulaic transition markers, hedging verbs, and structural
# boilerplate commonly overrepresented in AI-generated text across frontier LLMs.
AI_PHRASE_PATTERNS: tuple[tuple[str, str, float], ...] = (
    # (regex_pattern, human_label, weight)
    (r"\bdelve(?:s|d)?\s+into\b", "delve into", 1.2),
    (r"\ba\s+testament\s+to\b", "a testament to", 1.1),
    (r"\brich\s+tapestry(?:\s+of)?\b", "rich tapestry", 1.3),
    (r"\bplays?\s+a\s+(?:pivotal|crucial|vital|key)\s+role\b", "plays a pivotal/crucial role", 1.0),
    (
        r"\bin\s+(?:today'?s|the)\s+"
        r"(?:(?:fast-paced|ever-evolving|digital|rapidly\s+changing)\s+)*"
        r"(?:world|landscape|era|environment)\b",
        "in today's fast-paced world/landscape",
        1.4,
    ),
    (
        r"\bit\s+is\s+(?:important|essential|crucial|worth\s+noting)\s+to\s+"
        r"(?:note|remember|consider|highlight)\b",
        "it is important/crucial to note",
        0.9,
    ),
    (
        r"\bnot\s+only\b[\w\s,]+\bbut\s+(?:also\s+)?(?:serves\s+to|acts\s+as|highlights)\b",
        "not only ... but also serves to",
        0.8,
    ),
    (
        r"\bserve(?:s|d)?\s+as\s+a\s+(?:beacon|reminder|catalyst|cornerstone)\b",
        "serves as a beacon/catalyst/cornerstone",
        1.1,
    ),
    (
        r"\bunderscore(?:s|d)?\s+the\s+(?:importance|need|significance)\b",
        "underscores the importance/need",
        0.9,
    ),
    (
        r"\bfoster(?:s|ing|ed)?\s+a\s+(?:sense|culture|deeper\s+understanding)\b",
        "fosters a sense/culture",
        0.9,
    ),
    (
        r"\bseamlessly\s+(?:integrates?|integrated|blends?|combine[sd]?)\b",
        "seamlessly integrates/blends",
        1.0,
    ),
    (
        r"\bnavigat(?:e|ing|es|ed)\s+the\s+(?:complexities|intricacies|nuances)\b",
        "navigating the complexities/nuances",
        1.0,
    ),
    (r"\bmultifaceted\s+(?:nature|approach|landscape)\b", "multifaceted nature/approach", 1.0),
    (r"\bharness(?:ing|ed|es)?\s+the\s+power\s+of\b", "harnessing the power of", 1.0),
    (r"\ba\s+myriad\s+of\b", "a myriad of", 0.8),
    (r"\bparadigm\s+shift\b", "paradigm shift", 0.9),
    (r"\bholistic\s+(?:approach|view|perspective)\b", "holistic approach/perspective", 0.9),
    (r"\bin\s+conclusion\b[,\s]", "in conclusion", 0.8),
    (r"\bto\s+summarize\b[,\s]", "to summarize", 0.8),
    (r"\bultimately\b[,\s]", "ultimately,", 0.6),
    (r"\bfurthermore\b[,\s]", "furthermore,", 0.6),
    (r"\bmoreover\b[,\s]", "moreover,", 0.6),
    (r"\bas\s+an\s+ai\b", "as an AI", 1.5),
    (r"\bi\s+hope\s+this\s+helps\b", "I hope this helps", 1.2),
    # Patterns folded in from blader/humanizer (Wikipedia "Signs of AI writing")
    # and conorbronsdon/avoid-ai-writing's replacement table. These are the
    # regex-detectable subset; the full prose category list lives in
    # skill references/detectors.md. Weights stay moderate: this is a gauge,
    # not a verdict, so a single hit should not dominate the composite.
    (r"\bstands?\s+as\s+a\s+testament\b", "stands as a testament to", 1.1),
    (
        r"\bmark(?:s|ing)?\s+an?\s+"
        r"(?:indelible|pivotal|significant|new)\s+(?:moment|chapter|milestone)\b",
        "marking a pivotal moment/chapter",
        1.0,
    ),
    (
        r"\b(?:reflecting|symbolizing|showcasing|underscoring)\s+(?:the|a|its)\b",
        "shallow -ing analysis (reflecting/symbolizing/showcasing)",
        0.9,
    ),
    (r"\b(?:nestled|vibrant|breathtaking)\b", "sales language (nestled/vibrant)", 0.8),
    (r"\b(?:game[- ]changer|game-changing)\b", "game-changer", 0.9),
    (r"\b(?:unparalleled|unprecedented)\b", "unparalleled/unprecedented", 0.7),
    (r"\b(?:world-class|state-of-the-art|cutting-edge)\b", "world-class/state-of-the-art", 0.8),
    (r"\b(?:revolutionary|groundbreaking)\b", "revolutionary/groundbreaking", 0.6),
    (r"\b(?:leverag(?:e|ing|ed|es)|utiliz(?:e|ing|ed|es))\b", "leverage/utilize", 0.8),
    (r"\bboasts?\b", "boasts (copula avoidance)", 0.6),
    (r"\bit['\u2019]?s\s+not\s+just\b", "it's not just X, it's Y", 0.8),
    (r"\b(?:not\s+just\b.{0,60}\bbut\s+also\b)", "not just X but also Y", 0.8),
    (r"\bdive(?:s|d)?\s+into\b", "dive into", 0.7),
    (r"\blet['\u2019]?s\s+(?:dive\s+in|get\s+started)\b", "let's dive in", 0.7),
    (r"\bin\s+order\s+to\b", "in order to", 0.6),
    (r"\bdue\s+to\s+the\s+fact\s+that\b", "due to the fact that", 0.8),
    (r"\bit(?:['\u2019]s| is)\s+worth\s+noting\s+that\b", "it is worth noting that", 0.8),
    (r"\b(?:needless\s+to\s+say|it\s+goes\s+without\s+saying)\b", "needless to say", 0.8),
    (r"\bthe\s+future\s+looks\s+bright\b", "the future looks bright", 0.9),
    (
        r"\b(?:sure\s+thing!?|great\s+question!?|happy\s+to\s+help)\b",
        "assistant chatter (sure thing/great question)",
        0.6,
    ),
)

RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
RE_WORDS = re.compile(r"\b[\w'-]+\b", re.UNICODE)


@dataclass
class MarkerMatch:
    phrase: str
    count: int
    weight: float
    samples: list[str] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)  # start/end char offsets

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "count": self.count,
            "weight": self.weight,
            "samples": self.samples,
            "spans": [list(s) for s in self.spans],
        }


@dataclass
class StylometryReport:
    path: str
    word_count: int
    sentence_count: int
    burstiness_cv: float | None
    lexical_diversity: float
    ai_ngram_density: float
    matched_markers: list[dict[str, Any]]
    score: float | None
    confidence_level: str | None  # CLEAN | LOW | MEDIUM | HIGH, or None when uncalibrated
    density_tier: str  # low | medium | high, or uncalibrated when not scored
    status: str  # ok | insufficient_length
    findings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "burstiness_cv": round(self.burstiness_cv, 4)
            if self.burstiness_cv is not None
            else None,
            "lexical_diversity": round(self.lexical_diversity, 4),
            "ai_ngram_density": round(self.ai_ngram_density, 4),
            "matched_markers": self.matched_markers,
            "score": round(self.score, 4) if self.score is not None else None,
            "confidence_level": self.confidence_level,
            "density_tier": self.density_tier,
            "status": self.status,
            "findings": self.findings,
            "findings_confidence": [classify_finding_confidence(f) for f in self.findings],
            "notes": self.notes,
        }


# Runs between str.splitlines() boundaries. Matching runs instead of calling
# splitlines() keeps one line alive at a time rather than a list of all of them.
RE_LINE = re.compile(r"[^\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]+")
RE_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")

MARKER_SAMPLE_LIMIT = 3
MARKER_SPAN_LIMIT = 10

# Uploads reach the scorer at up to MAX_INPUT_BYTES, which is room for over a
# hundred million short words or sentences. Every pass below is a generator or
# a fixed-size window: holding one Python object per token, sentence or match
# made peak memory tens of times the upload and could OOM the threaded server.


def iter_sentences(text: str) -> Iterator[str]:
    """Yield sentences one at a time, skipping blank lines and fenced code blocks."""
    in_code_block = False
    for line in RE_LINE.finditer(text):
        trimmed = line.group(0).strip()
        if trimmed.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not trimmed:
            continue
        start = 0
        for brk in RE_SENTENCE_BREAK.finditer(trimmed):
            piece = trimmed[start : brk.start()].strip()
            if piece:
                yield piece
            start = brk.end()
        piece = trimmed[start:].strip()
        if piece:
            yield piece


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences while ignoring blank lines and code block markers."""
    return list(iter_sentences(text))


def iter_words(text: str) -> Iterator[str]:
    """Yield normalized alphanumeric word tokens one at a time."""
    for m in RE_WORDS.finditer(text):
        yield m.group(0).lower()


def extract_words(text: str) -> list[str]:
    """Extract normalized alphanumeric word tokens."""
    return list(iter_words(text))


def _count_words(text: str) -> int:
    return sum(1 for _ in RE_WORDS.finditer(text))


def _sentence_stats(sentences: Iterable[str]) -> tuple[int, float, float, float | None]:
    """Return (sentence_count, mean, std_dev, cv) in one pass over *sentences*."""
    sentence_count = 0
    n = 0
    total = 0
    total_sq = 0
    for s in sentences:
        sentence_count += 1
        length = _count_words(s)
        if length > 0:
            n += 1
            total += length
            total_sq += length * length

    if sentence_count == 0:
        return 0, 0.0, 0.0, None
    if n < 2:
        return sentence_count, float(total), 0.0, None

    mean_len = total / n
    # Integer sums keep the variance exact until the single final division.
    variance = (n * total_sq - total * total) / (n * (n - 1))
    std_dev = math.sqrt(variance)
    cv = (std_dev / mean_len) if mean_len > 0 else 0.0
    return sentence_count, mean_len, std_dev, cv


def compute_burstiness(sentences: Iterable[str]) -> tuple[float, float, float | None]:
    """Compute mean sentence word length, standard deviation, and coefficient of variation (CV).

    CV is ``None`` when it cannot be measured (no sentences, or fewer than two
    with words): a one-or-zero-sample CV is undefined, and reporting it as 0.0
    made the caller's tiering read "perfectly uniform" — the strongest
    LLM-likeness signal — for text whose body yielded no sentences at all (#132).
    """
    _, mean_len, std_dev, cv = _sentence_stats(sentences)
    return mean_len, std_dev, cv


def _word_stats(words: Iterable[str], window_size: int = 50) -> tuple[int, float]:
    """Return (word_count, MATTR) in one pass, holding only *window_size* words."""
    it = iter(words)
    window = deque(islice(it, window_size))
    n = len(window)
    if n == 0:
        return 0, 0.0
    current_window = Counter(window)
    if n < window_size:
        return n, len(current_window) / n

    total_ttr = len(current_window) / window_size
    num_windows = 1
    for entering_word in it:
        leaving_word = window.popleft()
        current_window[leaving_word] -= 1
        if current_window[leaving_word] == 0:
            del current_window[leaving_word]

        window.append(entering_word)
        current_window[entering_word] += 1
        total_ttr += len(current_window) / window_size
        num_windows += 1

    return n + num_windows - 1, total_ttr / num_windows


def compute_mattr(words: Iterable[str], window_size: int = 50) -> float:
    """Compute Moving-Average Type-Token Ratio (MATTR) across sliding windows."""
    return _word_stats(words, window_size)[1]


def scan_ai_phrases(text: str) -> list[MarkerMatch]:
    """Find and tally high-frequency AI cadence phrases."""
    matches: list[MarkerMatch] = []
    for pattern, label, weight in AI_PHRASE_PATTERNS:
        count = 0
        samples: list[str] = []
        offsets: list[tuple[int, int]] = []
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if count < MARKER_SAMPLE_LIMIT:
                samples.append(m.group(0))
            if count < MARKER_SPAN_LIMIT:
                offsets.append((m.start(), m.end()))
            count += 1
        if count:
            matches.append(
                MarkerMatch(
                    phrase=label,
                    count=count,
                    weight=weight,
                    samples=samples,
                    spans=offsets,
                )
            )
    return matches


def classify_density(score: float | None) -> str:
    """Map a composite score to an edit-decision tier.

    ``high`` means the text sits at or above the suspicious threshold, so the
    skill's rewrite pass should engage; ``medium``/``low`` mean the measurable
    AI-density signals are weaker and the skill should mainly verify rather than
    rewrite. ``uncalibrated`` is returned for samples too short to score.
    """
    if score is None:
        return "uncalibrated"
    if score >= DEFAULT_THRESHOLD:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def score_text_stylometry(text: str, path: str = "<text>") -> StylometryReport:
    """Run full multi-dimensional stylometric analysis and return a structured report."""
    word_count, mattr = _word_stats(iter_words(text))
    sentence_count, _, _, cv = _sentence_stats(iter_sentences(text))

    findings: list[str] = []
    notes: list[str] = []

    # 1. Length Guard
    if word_count < MIN_SAMPLE_WORDS:
        marker_matches = scan_ai_phrases(text)
        for m in marker_matches:
            findings.append(f"AI phrase marker '{m.phrase}' found ({m.count}x)")
        notes.append(
            f"Sample contains {word_count} words; statistical stylometry is "
            f"uncalibrated below {MIN_SAMPLE_WORDS} words, so no score is reported"
        )
        return StylometryReport(
            path=path,
            word_count=word_count,
            sentence_count=sentence_count,
            burstiness_cv=None,
            lexical_diversity=mattr,
            ai_ngram_density=0.0,
            matched_markers=[m.to_dict() for m in marker_matches],
            score=None,
            confidence_level=None,
            density_tier="uncalibrated",
            status="insufficient_length",
            findings=findings,
            notes=notes,
        )

    # 2. Metric Calculations
    marker_matches = scan_ai_phrases(text)

    # N-gram density: weighted marker instances per 100 words
    total_marker_weight = sum(m.count * m.weight for m in marker_matches)
    ngram_density = (total_marker_weight / (word_count / 100.0)) if word_count > 0 else 0.0

    # 3. Component Sub-scores (0.0 to 1.0)
    # Burstiness subscore: low CV (<0.35) is strongly characteristic of LLMs;
    # high CV (>0.60) is human
    burstiness_score: float | None
    if cv is None:
        # Fewer than two parseable sentences: burstiness is unmeasurable, not
        # maximally LLM-like. The composite renormalizes over the remaining
        # components instead of crediting the strongest signal (#132).
        burstiness_score = None
    elif cv < 0.25:
        burstiness_score = 0.95
    elif cv < 0.35:
        burstiness_score = 0.80
    elif cv < 0.45:
        burstiness_score = 0.50
    elif cv < 0.55:
        burstiness_score = 0.25
    else:
        burstiness_score = 0.05

    # N-gram subscore: >1.5 weighted matches per 100 words is very high
    if ngram_density >= 2.0:
        ngram_score = 1.0
    elif ngram_density >= 1.0:
        ngram_score = 0.75
    elif ngram_density >= 0.5:
        ngram_score = 0.45
    elif ngram_density > 0:
        ngram_score = 0.20
    else:
        ngram_score = 0.0

    # Lexical uniformity subscore: LLMs cluster tightly around MATTR 0.65-0.78 for 50-word windows
    diversity_score = 0.4 if 0.68 <= mattr <= 0.76 else 0.1

    # 4. Composite Scoring & Small-Sample Dampening
    if burstiness_score is None:
        notes.append(
            "Sentence burstiness unavailable (fewer than 2 parsed sentences — "
            "e.g. body wrapped in a code fence); composite renormalized over "
            "AI-phrase density and lexical diversity"
        )
        raw_composite = ((ngram_score * 0.45) + (diversity_score * 0.10)) / 0.55
    else:
        raw_composite = (burstiness_score * 0.45) + (ngram_score * 0.45) + (diversity_score * 0.10)

    # Dampening factor: scales smoothly from 0.4 at MIN_SAMPLE_WORDS up to 1.0 at FULL_WEIGHT_WORDS
    if word_count < FULL_WEIGHT_WORDS:
        dampener = 0.4 + 0.6 * (
            (word_count - MIN_SAMPLE_WORDS) / (FULL_WEIGHT_WORDS - MIN_SAMPLE_WORDS)
        )
        notes.append(
            f"Sample word count ({word_count}) is in calibration range "
            f"({MIN_SAMPLE_WORDS}-{FULL_WEIGHT_WORDS}); score dampened by factor {dampener:.2f}"
        )
    else:
        dampener = 1.0

    final_score = min(1.0, max(0.0, raw_composite * dampener))

    # 5. Classify Findings & Confidence Tiers
    if marker_matches:
        for m in marker_matches:
            findings.append(f"AI cadence phrase '{m.phrase}' ({m.count}x)")

    if cv is not None and cv < 0.35 and sentence_count >= 3:
        findings.append(f"Unnaturally uniform sentence cadence (CV={cv:.2f} < 0.35)")

    if ngram_density >= 1.0:
        findings.append(f"Elevated AI formulaic transition density ({ngram_density:.2f}/100w)")

    if final_score >= 0.75:
        confidence = "HIGH"
    elif final_score >= 0.50:
        confidence = "MEDIUM"
    elif final_score >= 0.25:
        confidence = "LOW"
    else:
        confidence = "CLEAN"

    return StylometryReport(
        path=path,
        word_count=word_count,
        sentence_count=sentence_count,
        burstiness_cv=cv,
        lexical_diversity=mattr,
        ai_ngram_density=ngram_density,
        matched_markers=[m.to_dict() for m in marker_matches],
        score=final_score,
        confidence_level=confidence,
        density_tier=classify_density(final_score),
        status="ok",
        findings=findings,
        notes=notes,
    )


def print_human_stylometry_report(report: StylometryReport, explain: bool = False) -> None:
    """Print clean human-readable output to stdout."""
    print(f"=== Stylometric AI-Text Report: {report.path} ===")
    print("Gauge only: measures lexical and cadence signals, not authorship.")
    print(f"Status:             {report.status}")
    if report.confidence_level is not None:
        print(f"Confidence Level:   {report.confidence_level}")
    else:
        print("Confidence Level:   n/a")
    if report.score is not None:
        print(f"Stylometry Score:   {report.score * 100:.1f}% (score: {report.score:.3f})")
    else:
        print("Stylometry Score:   n/a (sample too short to score)")
    print(f"Density Tier:       {report.density_tier}")
    print(f"Word Count:         {report.word_count}")
    print(f"Sentence Count:     {report.sentence_count}")
    cv_display = (
        f"{report.burstiness_cv:.3f}"
        if report.burstiness_cv is not None
        else "n/a (fewer than 2 sentences)"
    )
    print(f"Sentence CV:        {cv_display}")
    print(f"Lexical Diversity:  {report.lexical_diversity:.3f} (MATTR)")
    print(f"AI Marker Density:  {report.ai_ngram_density:.3f} / 100 words")

    if report.findings:
        print("\nFindings:")
        for f in report.findings:
            print(f"  - {f}")

    if explain and report.matched_markers:
        print("\nMatched Phrases Detail:")
        for m in report.matched_markers:
            print(f"  * {m['phrase']} (occurrences: {m['count']}, weight: {m['weight']})")
            if m.get("samples"):
                print(f'    sample: "{m["samples"][0]}"')

    if report.notes:
        print("\nNotes:")
        for n in report.notes:
            print(f"  * {n}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", nargs="?", default="-", help="Text file to score ('-' for stdin)")
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Score threshold to trigger exit code 1 (default: {DEFAULT_THRESHOLD})",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON output")
    p.add_argument(
        "--explain", action="store_true", help="Include detailed matched phrase breakdown"
    )

    args = p.parse_args()

    # read_text_input raises SystemExit(2) on unreadable/oversized/non-regular input.
    text = read_text_input(args.path)

    input_label = "<stdin>" if args.path == "-" else args.path
    report = score_text_stylometry(text, path=input_label)

    if args.json:
        emit_json(report.to_dict())
    else:
        print_human_stylometry_report(report, explain=args.explain)

    if report.status == "ok" and report.score is not None and report.score >= args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
