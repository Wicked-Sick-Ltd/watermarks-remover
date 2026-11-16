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


def _run_bash(script: str, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        timeout=120,
    )


@pytest.mark.parametrize("stem", HOSTILE_STEMS)
def test_clean_skill_passes_hostile_file_names_as_one_argument(tmp_path, stem):
    paths = {
        "INPUT": f"{stem}.md",
        "OUTPUT": f"{stem}.cleaned.md",
        "FINAL": f"{stem}.final.md",
    }
    (tmp_path / paths["INPUT"]).write_text("Hello\u200b world.\n", encoding="utf-8")

    blocks = [b for b in _bash_blocks(CLEAN_SKILL) if any(_references(b, n) for n in paths)]
    assert blocks
    for block in blocks:
        used = {name: value for name, value in paths.items() if _references(block, name)}
        _run_bash(
            _agent_script(block, used, {}),
            cwd=tmp_path,
            env={"SCRIPTS": str(CLEAN_SKILL / "scripts")},
        )

    assert not (tmp_path / CANARY).exists(), "a command smuggled in the file name ran"
    assert (tmp_path / paths["OUTPUT"]).read_text(encoding="utf-8") == "Hello world.\n"
    assert (tmp_path / paths["FINAL"]).read_text(encoding="utf-8") == "Hello world.\n"


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

    blocks = [b for b in _bash_blocks(MARKS_SKILL) if _references(b, "INPUT") or _bare("path").search(b)]
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


@pytest.mark.parametrize("skill", [CLEAN_SKILL, MARKS_SKILL], ids=lambda p: p.name)
def test_skill_bash_blocks_only_use_quoted_path_variables(skill):
    for block in _bash_blocks(skill):
        for name in PATH_PLACEHOLDERS:
            assert not _bare(name).search(block), f"bare {name} placeholder in:\n{block}"
            for before, after in re.findall(rf"(.)\${name}(.)", block):
                assert (before, after) == ('"', '"'), f"unquoted ${name} in:\n{block}"


@pytest.mark.parametrize("skill", [CLEAN_SKILL, MARKS_SKILL], ids=lambda p: p.name)
def test_skill_tells_the_agent_to_refuse_unquotable_paths(skill):
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    section = text.split("### Passing file paths safely", 1)
    assert len(section) == 2
    rule = section[1].split("\n## ", 1)[0]
    assert "single-quoted" in rule
    assert "control character" in rule
    assert "quotation mark" in rule
