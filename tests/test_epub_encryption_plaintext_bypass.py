"""EPUB CipherReference entries must not exempt plaintext members.

A META-INF/encryption.xml CipherReference URI is not evidence that a member
is ciphertext. These tests are the spec for that bypass: they fail while a
package document, chapter, image, or other part can be listed there and left
in the clear, and they pass once those parts are inspected and cleaned and an
unreadable encrypted member is an inconclusive scan.

The checks below drive the entry points the product actually runs: the
container inspectors and cleaners, the inspect and clean CLIs, the pre-commit
gate, the directory audit, the website-audit item, the written-file hook, and
the HTTP /inspect, /clean, /inspect/batch, and /clean/batch routes.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import struct
import subprocess
import sys
import threading
import zipfile
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import audit_dir
import audit_website
import check_staged
import clean_staged
import server
from audit_lib import is_actionable, scan_file
from container_meta import clean_container, clean_epub, inspect_container, inspect_epub

INSPECT_PY = SCRIPTS / "inspect_file.py"
CLEAN_PY = SCRIPTS / "clean_file.py"
HOOK_PY = SCRIPTS / "hook_written_file.py"

AES_256_CBC = "http://www.w3.org/2001/04/xmlenc#aes256-cbc"
ZWSP = "\u200b"
OPAQUE = b"opaque-payload-not-markup"


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _c2pa_png() -> bytes:
    """1x1 PNG whose tEXt chunk carries a C2PA marker the image scan recognizes."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00")
    text = b"Comment\x00c2pa test contentcredentials"
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tEXt", text)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _cipher_uri(name: str, style: str) -> str:
    if style == "relative":
        return f"../{name}"
    if style == "bare":
        return name
    raise AssertionError(style)


def _encryption_xml(names: tuple[str, ...], *, style: str, method: str | None) -> str:
    blocks = []
    for name in names:
        method_el = f'<enc:EncryptionMethod Algorithm="{method}"/>' if method is not None else ""
        blocks.append(
            "<enc:EncryptedData>"
            f"{method_el}"
            "<enc:CipherData>"
            f'<enc:CipherReference URI="{_cipher_uri(name, style)}"/>'
            "</enc:CipherData>"
            "</enc:EncryptedData>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
        'xmlns:enc="http://www.w3.org/2001/04/xmlenc#">' + "".join(blocks) + "</encryption>"
    )


def _opf(*, creator: str, generator: str) -> str:
    generator_meta = f'<meta name="generator" content="{generator}"/>' if generator else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:identifier id="uid">urn:uuid:1234</dc:identifier>'
        "<dc:title>Test Book</dc:title>"
        f"<dc:creator>{creator}</dc:creator>"
        f"{generator_meta}</metadata>"
        "<manifest>"
        '<item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="img" href="images/cover.png" media-type="image/png"/>'
        '<item id="extra" href="../META-INF/metadata.xml" media-type="application/xml"/>'
        "</manifest>"
        '<spine><itemref idref="c1"/></spine>'
        "</package>"
    )


def _xhtml(*, generator: str, body: str) -> str:
    generator_meta = f'<meta name="generator" content="{generator}"/>' if generator else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>Chapter 1</title>{generator_meta}</head>"
        f"<body><p>{body}</p></body></html>"
    )


def build_epub(
    parts: dict[str, bytes],
    encrypted: tuple[str, ...] = (),
    *,
    style: str = "relative",
    method: str | None = None,
) -> bytes:
    """Build an EPUB. Listed members are named by CipherReference and left as given."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        if encrypted:
            zf.writestr(
                "META-INF/encryption.xml",
                _encryption_xml(encrypted, style=style, method=method),
            )
        for name, raw in parts.items():
            zf.writestr(name, raw)
    return buf.getvalue()


def marked_parts() -> dict[str, bytes]:
    """Package document, chapter, image, and a side file, each with its own mark."""
    return {
        "OEBPS/content.opf": _opf(creator="OpenAI", generator="ChatGPT").encode(),
        "OEBPS/chapter1.xhtml": _xhtml(
            generator="ChatGPT",
            body=f"Chapter one {ZWSP} with a hidden mark",
        ).encode(),
        "OEBPS/images/cover.png": _c2pa_png(),
        "META-INF/metadata.xml": b"<metadata>c2pa contentcredentials</metadata>",
    }


def marked_epub(
    *, style: str = "relative", method: str | None = None, exempt: bool = True
) -> bytes:
    parts = marked_parts()
    encrypted = tuple(parts) if exempt else ()
    return build_epub(parts, encrypted, style=style, method=method)


def _read(data: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return zf.read(name)


def marks_remain(data: bytes) -> list[str]:
    """Provenance that must not survive a clean of a plaintext book."""
    left: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        if "OEBPS/content.opf" in names and b"OpenAI" in zf.read("OEBPS/content.opf"):
            left.append("opf creator")
        if "OEBPS/chapter1.xhtml" in names:
            xhtml = zf.read("OEBPS/chapter1.xhtml")
            if b"ChatGPT" in xhtml:
                left.append("xhtml generator")
            if ZWSP.encode() in xhtml:
                left.append("xhtml invisible mark")
        if (
            "OEBPS/images/cover.png" in names
            and b"c2pa" in zf.read("OEBPS/images/cover.png").lower()
        ):
            left.append("png c2pa")
        if "META-INF/metadata.xml" in names and b"c2pa" in zf.read("META-INF/metadata.xml").lower():
            left.append("side-file c2pa")
    return left


def _inspected(findings: list[str], name: str) -> bool:
    """True when *name* contributed a real finding, not only an encryption skip."""
    return any(name in finding and "encrypted content" not in finding for finding in findings)


def assert_marks_detected(data: bytes) -> None:
    """The same book with no encryption manifest is the control for these marks."""
    control = marked_epub(exempt=False)
    control_c2pa, control_ai, control_findings, _details = inspect_epub(control)
    assert control_ai is True
    assert control_c2pa is True
    assert _inspected(control_findings, "content.opf")
    assert _inspected(control_findings, "chapter1.xhtml")
    assert _inspected(control_findings, "cover.png")
    assert _inspected(control_findings, "metadata.xml")
    assert inspect_container(Path("book.epub"), data=control).layer_a_total > 0

    has_c2pa, has_ai, findings, _details = inspect_epub(data)
    assert has_ai is True
    assert has_c2pa is True
    assert _inspected(findings, "content.opf")
    assert _inspected(findings, "chapter1.xhtml")
    assert _inspected(findings, "cover.png")
    assert _inspected(findings, "metadata.xml")

    report = inspect_container(Path("book.epub"), data=data)
    assert report.has_ai_metadata is True
    assert report.has_c2pa is True
    assert report.layer_a_total > 0
    assert _inspected(report.findings, "chapter1.xhtml")


def assert_marks_removed(data: bytes) -> None:
    cleaned, _actions = clean_epub(data)
    assert marks_remain(cleaned) == []
    # Re-inspect the cleaned bytes the same way clean_container does, and
    # require that report to agree with the bytes.
    report = inspect_container(Path("book.cleaned.epub"), data=cleaned)
    assert report.has_ai_metadata is False
    assert report.has_c2pa is False
    assert report.layer_a_total == 0


@pytest.mark.parametrize(
    ("style", "method"),
    [
        ("relative", None),
        ("bare", None),
        ("relative", AES_256_CBC),
    ],
)
def test_cipher_reference_does_not_exempt_parseable_plaintext(style: str, method: str | None):
    """A CipherReference URI, with or without an encryption method, does not
    exempt members whose bytes still parse as the declared content type."""
    data = marked_epub(style=style, method=method)
    assert_marks_detected(data)
    assert_marks_removed(data)


def test_layer_a_only_chapter_is_not_exempt(tmp_path: Path):
    """Invisible Unicode is reported by the container body scan, which has its
    own encryption skip, separate from the per-member metadata scan."""
    chapter = "OEBPS/chapter1.xhtml"
    data = build_epub(
        {
            "OEBPS/content.opf": _opf(creator="Jane Doe", generator="").encode(),
            chapter: _xhtml(generator="", body=f"Visible text {ZWSP} only").encode(),
        },
        (chapter,),
    )
    report = inspect_container(tmp_path / "book.epub", data=data)
    assert report.layer_a_total > 0
    assert any(finding.startswith("layer-a") for finding in report.findings)
    item_path = tmp_path / "book.epub"
    item_path.write_bytes(data)
    item = scan_file(item_path)
    assert item["suspicious_total"] > 0
    assert is_actionable(item)


def test_precommit_directory_and_website_audits_reject_listed_plaintext(tmp_path, monkeypatch):
    book = tmp_path / "book.epub"
    book.write_bytes(marked_epub())

    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(book)])
    assert check_staged.main() == 1

    monkeypatch.setattr(sys, "argv", ["audit_dir.py", str(tmp_path), "--json"])
    assert audit_dir.main() == 1

    remote = audit_website.inspect_remote(
        "https://example.com/book.epub",
        book.read_bytes(),
        "application/epub+zip",
    )
    assert remote["has_ai_metadata"] is True
    assert remote["has_c2pa"] is True
    assert remote["suspicious_total"] > 0
    assert is_actionable(remote)


def test_inspect_cli_rejects_listed_plaintext(tmp_path: Path):
    book = tmp_path / "book.epub"
    book.write_bytes(marked_epub())
    proc = subprocess.run(
        [sys.executable, str(INSPECT_PY), str(book), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["has_ai_metadata"] is True
    assert report["has_c2pa"] is True
    assert report["suspicious_total"] > 0


def _run_hook(path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for var in (
        "WATERMARKS_HOOK_MODE",
        "CLAUDE_PLUGIN_OPTION_HOOK_MODE",
        "WATERMARKS_HOOK_ROOT",
        "CLAUDE_PROJECT_DIR",
    ):
        env.pop(var, None)
    env["WATERMARKS_HOOK_ROOT"] = str(path.parent)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": str(path)},
        "cwd": str(path.parent),
    }
    return subprocess.run(
        [sys.executable, str(HOOK_PY), "--mode", mode],
        input=json.dumps(payload),
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
        check=False,
    )


def test_written_file_hook_check_rejects_listed_plaintext(tmp_path: Path):
    book = tmp_path / "book.epub"
    book.write_bytes(marked_epub())
    proc = _run_hook(book, "check")
    assert proc.returncode == 2
    shown = proc.stdout + proc.stderr
    assert any(
        token in shown for token in ("ChatGPT", "OpenAI", "c2pa", "layer-a", "contentcredentials")
    )


def test_written_file_hook_clean_strips_listed_plaintext(tmp_path: Path):
    book = tmp_path / "book.epub"
    book.write_bytes(marked_epub())
    proc = _run_hook(book, "clean")
    assert marks_remain(book.read_bytes()) == [], proc.stdout + proc.stderr


def test_clean_cli_and_staged_hook_strip_listed_plaintext(tmp_path, monkeypatch):
    book = tmp_path / "book.epub"
    book.write_bytes(marked_epub())
    dest = tmp_path / "book.cleaned.epub"
    proc = subprocess.run(
        [sys.executable, str(CLEAN_PY), str(book), "-o", str(dest), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dest.is_file()
    assert marks_remain(dest.read_bytes()) == [], proc.stdout
    report = json.loads(proc.stdout)
    assert report["still_has_ai_metadata"] is False
    assert report["still_has_c2pa"] is False
    assert proc.returncode == 0

    staged = tmp_path / "staged.epub"
    staged.write_bytes(marked_epub())
    monkeypatch.setattr(sys, "argv", ["clean_staged.py", str(staged)])
    clean_staged.main()
    assert marks_remain(staged.read_bytes()) == []


def test_http_inspect_and_clean_do_not_exempt_listed_plaintext():
    data = marked_epub()
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
    try:
        payload = {"file": base64.b64encode(data).decode("ascii"), "name": "book.epub"}
        conn.request(
            "POST",
            "/inspect",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        assert response.status == 200
        report = body["report"]
        assert report["has_ai_metadata"] is True
        assert report["has_c2pa"] is True
        assert report["suspicious_total"] > 0
        assert body["suspicious"]["verdict"] is True

        conn.request(
            "POST",
            "/clean",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        assert response.status == 200
        cleaned = base64.b64decode(body["cleaned"])
        assert marks_remain(cleaned) == []
        assert body["report"]["still_has_ai_metadata"] is False
        assert body["report"]["still_has_c2pa"] is False

        batch = {"files": [payload]}
        conn.request(
            "POST",
            "/inspect/batch",
            body=json.dumps(batch).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        assert response.status == 200
        batched = body["results"][0]
        assert batched["report"]["has_ai_metadata"] is True
        assert batched["report"]["has_c2pa"] is True
        assert batched["suspicious"]["verdict"] is True

        conn.request(
            "POST",
            "/clean/batch",
            body=json.dumps(batch).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = json.loads(response.read())
        assert response.status == 200
        batched = body["results"][0]
        assert marks_remain(base64.b64decode(batched["cleaned"])) == []
        assert batched["report"]["still_has_ai_metadata"] is False
        assert batched["report"]["still_has_c2pa"] is False
    finally:
        conn.close()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def test_opaque_encrypted_member_is_inconclusive(tmp_path, monkeypatch):
    """A real encryption method whose bytes do not parse as the chapter's
    content type may be copied through, but that copy is not a clean scan.
    """
    chapter = "OEBPS/chapter1.xhtml"
    data = build_epub(
        {
            "OEBPS/content.opf": _opf(creator="Jane Doe", generator="").encode(),
            chapter: OPAQUE,
        },
        (chapter,),
        method=AES_256_CBC,
    )
    book = tmp_path / "book.epub"
    book.write_bytes(data)

    monkeypatch.setattr(sys, "argv", ["check_staged.py", str(book)])
    assert check_staged.main() != 0

    inspected = subprocess.run(
        [sys.executable, str(INSPECT_PY), str(book), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert inspected.returncode != 0

    hook = _run_hook(book, "check")
    assert hook.returncode == 2

    dest = tmp_path / "out.epub"
    cleaned = subprocess.run(
        [sys.executable, str(CLEAN_PY), str(book), "-o", str(dest), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if dest.is_file() and _read(dest.read_bytes(), chapter) == OPAQUE:
        assert cleaned.returncode != 0
    elif dest.is_file():
        assert OPAQUE not in dest.read_bytes()
    else:
        assert cleaned.returncode != 0


def test_clean_container_does_not_report_clean_while_plaintext_marks_remain(tmp_path: Path):
    src = tmp_path / "book.epub"
    src.write_bytes(marked_epub())
    dest = tmp_path / "book.cleaned.epub"
    result = clean_container(src, dest)
    assert marks_remain(dest.read_bytes()) == []
    assert result["still_has_ai_metadata"] is False
    assert result["still_has_c2pa"] is False
    again = inspect_container(dest)
    assert again.has_ai_metadata is False
    assert again.has_c2pa is False
    assert again.layer_a_total == 0
