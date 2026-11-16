"""The skills' bash examples must keep file names out of shell parsing.

An agent follows SKILL.md as written: it fills each path placeholder with the
file the user named and runs the block in its shell. These tests do the same
with real files whose names carry shell metacharacters, and fail if the shell
runs anything the name smuggled in or splits the path into several arguments.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "skills"
CLEAN_SKILL = SKILLS_DIR / "clean-user-facing-text"
MARKS_SKILL = SKILLS_DIR / "remove-ai-marks"

PATH_PLACEHOLDERS = ("INPUT", "OUTPUT", "FINAL", "DIR")
BASH_BLOCK = re.compile(r"^[ \t]*```bash[ \t]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)
CANARY = "PWNED"

HOSTILE_STEMS = (
    f"draft;touch {CANARY};x",
    f"draft$(touch {CANARY})x",
    f"draft`touch {CANARY}`x",
    f"draft|touch {CANARY}|x",
    f"draft && touch {CANARY} && x",
    f"draft >{CANARY} x",
    f"-draft;touch {CANARY}",
)


def _bash_blocks(skill: Path) -> list[str]:
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    return [textwrap.dedent(block) for block in BASH_BLOCK.findall(text)]


def _bare(name: str) -> re.Pattern[str]:
    # A placeholder word the agent is meant to overwrite with the path itself,
    # as opposed to a "$NAME" variable reference or a NAME=... assignment.
    return re.compile(rf"(?<![\w$]){re.escape(name)}(?![\w=])")


def _references(block: str, name: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", block) is not None


def _shell_literal(path: str) -> str:
    """Bind a path the way 'Passing file paths safely' tells the agent to."""
    assert "'" not in path and not any(ord(ch) < 0x20 for ch in path)
    if path.startswith("-"):
        path = "./" + path
    return f"'{path}'"


def _agent_script(block: str, paths: dict[str, str], aliases: dict[str, str]) -> str:
    script = re.sub(r"(?<!\w)PYTHON(?!\w)", lambda _: f"'{sys.executable}'", block)
    for placeholder, name in {**{n: n for n in paths}, **aliases}.items():
        script = _bare(placeholder).sub(lambda _, value=paths[name]: value, script)
    preamble = "".join(f"{name}={_shell_literal(value)}\n" for name, value in paths.items())
    return preamble + script


BASH = shutil.which("bash")
needs_posix_bash = pytest.mark.skipif(
    os.name == "nt" or BASH is None,
    reason="needs bash and POSIX file names (Windows forbids |, > and other test characters)",
)


def _run_bash(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    # The blocks exit non-zero by design when inspect finds suspicious input.
    return subprocess.run(
        [BASH, "-c", script],
        cwd=cwd,
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _run_clean_blocks(blocks: list[str], cwd: Path, paths: dict[str, str]) -> None:
    for block in blocks:
        used = {name: value for name, value in paths.items() if _references(block, name)}
        result = _run_bash(
            _agent_script(block, used, {}),
            cwd=cwd,
            env={"SCRIPTS": str(CLEAN_SKILL / "scripts")},
        )
        # Inspect exits 1 on suspicious input, so a split path shows up only here.
        assert "cannot read" not in result.stderr, result.stderr
        assert "error:" not in result.stderr, result.stderr
    assert not (cwd / CANARY).exists(), "a command smuggled in the file name ran"


def _clean_paths(stem: str) -> dict[str, str]:
    return {"INPUT": f"{stem}.md", "OUTPUT": f"{stem}.cleaned.md", "FINAL": f"{stem}.final.md"}


@needs_posix_bash
def test_clean_block_runner_catches_a_split_inspect_path(tmp_path):
    paths = _clean_paths("q3 report")
    (tmp_path / paths["INPUT"]).write_text("Hello\u200b world.\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"cannot read|error:"):
        _run_clean_blocks(
            ['PYTHON "$SCRIPTS/inspect_text.py" --stylometry --json $INPUT\n'], tmp_path, paths
        )


@needs_posix_bash
@pytest.mark.parametrize("stem", HOSTILE_STEMS)
def test_clean_skill_passes_hostile_file_names_as_one_argument(tmp_path, stem):
    paths = _clean_paths(stem)
    (tmp_path / paths["INPUT"]).write_text("Hello\u200b world.\n", encoding="utf-8")

    blocks = [b for b in _bash_blocks(CLEAN_SKILL) if any(_references(b, n) for n in paths)]
    assert blocks
    _run_clean_blocks(blocks, tmp_path, paths)

    assert (tmp_path / paths["OUTPUT"]).read_text(encoding="utf-8") == "Hello world.\n"
    assert (tmp_path / paths["FINAL"]).read_text(encoding="utf-8") == "Hello world.\n"


@needs_posix_bash
@pytest.mark.parametrize("stem", HOSTILE_STEMS)
def test_remove_ai_marks_sends_hostile_file_names_as_data(tmp_path, stem):
    work = tmp_path / "work"
    work.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        '#!/bin/sh\nfor arg in "$@"; do printf \'%s\\0\' "$arg"; done >> "$CURL_LOG"\n'
        "printf '\\036' >> \"$CURL_LOG\"\n",
        encoding="utf-8",
    )
    fake_curl.chmod(fake_curl.stat().st_mode | stat.S_IXUSR)

    name = f"{stem}.md"
    content = b"Hello\xe2\x80\x8b world.\n"
    (work / name).write_bytes(content)

    blocks = [
        b
        for b in _bash_blocks(MARKS_SKILL)
        if "base64 <" in b and (_references(b, "INPUT") or _bare("path").search(b))
    ]
    assert blocks
    for block in blocks:
        _run_bash(
            _agent_script(block, {"INPUT": name}, {"path": "INPUT"}),
            cwd=work,
            env={
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "CURL_LOG": str(curl_log),
                "WM": "http://127.0.0.1:9",
            },
        )

    assert not (work / CANARY).exists(), "a command smuggled in the file name ran"
    calls = [c for c in curl_log.read_bytes().split(b"\x1e") if c]
    assert len(calls) == len(blocks)
    for call in calls:
        argv = call.decode("utf-8").split("\0")
        payload = json.loads(argv[argv.index("-d") + 1])
        assert payload["name"] == name
        assert base64.b64decode(payload["file"]) == content


def _unquoted_path_uses(block: str) -> list[str]:
    found = []
    for name in PATH_PLACEHOLDERS:
        found += [f"bare {name}" for _ in _bare(name).finditer(block)]
        for match in re.finditer(rf"\$(?:\{{{name}\}}|{name}(?!\w))", block):
            before = block[match.start() - 1 : match.start()]
            after = block[match.end() : match.end() + 1]
            if (before, after) != ('"', '"'):
                found.append(f"unquoted {match.group()}")
    return found


@pytest.mark.parametrize(
    "block",
    [
        'PYTHON "$SCRIPTS/inspect_text.py" --json $INPUT\n',
        "$INPUT",
        'cat "$INPUT" >${OUTPUT}\n',
        "PYTHON clean_text.py INPUT\n",
    ],
)
def test_quoting_check_flags_unquoted_path_uses(block):
    assert _unquoted_path_uses(block)


@pytest.mark.parametrize("skill", [CLEAN_SKILL, MARKS_SKILL], ids=lambda p: p.name)
def test_skill_bash_blocks_only_use_quoted_path_variables(skill):
    for block in _bash_blocks(skill):
        assert not _unquoted_path_uses(block), block


@pytest.mark.parametrize("skill", [CLEAN_SKILL, MARKS_SKILL], ids=lambda p: p.name)
def test_skill_tells_the_agent_to_refuse_unquotable_paths(skill):
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    section = text.split("### Passing file paths safely", 1)
    assert len(section) == 2
    rule = section[1].split("\n## ", 1)[0]
    assert "single-quoted" in rule
    assert "control character" in rule
    assert "quotation mark" in rule
