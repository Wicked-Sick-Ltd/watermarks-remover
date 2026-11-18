"""Caller-chosen paths in remove-ai-marks shell recipes must not be executed.

``install_skill.py`` copies ``skills/remove-ai-marks/SKILL.md`` onto the agent
host. The copied file's bash recipes for ``/inspect``, ``/detect``, and
``/clean`` tell the agent where to put the file path. These tests install that
skill, bind a directory entry whose name contains shell syntax the way the
recipe writes the path, and run the recipe with bash.

Binding rules (this is what a fix has to satisfy):

- The operand of ``base64 <`` and of ``basename`` is the path slot.
- A parameter expansion (``$name``, ``${name}``, ``"$name"``, ``"${name}"``,
  ``$1``, ``"$1"``) is data. The caller path is assigned to that parameter and
  the recipe text is not edited.
- Any other operand is a placeholder the recipe tells the agent to overwrite
  (``path``, ``INPUT``, ``notes.md``, ``notes.txt``, ``shot.png``, and the
  same token in the JSON ``name``). The caller path is spliced into that text,
  including inside quotes the recipe itself wrote. Command substitution still
  runs inside double quotes, and a ``'`` in the path closes a single-quoted
  splice.

A recipe passes only when none of the payloads below create ``CANARY``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_NAME = "remove-ai-marks"
SKILL_MD = ROOT / "skills" / SKILL_NAME / "SKILL.md"
CANARY_NAME = "CANARY"

# Payloads are legal single path components: no slash. ``touch CANARY`` runs
# only when the shell parses the name as syntax. Quote-breakout names also fire
# when that same token sits inside the quotes the recipe already wrote, and
# when a fix only wraps a spliced path in quotes.
PAYLOADS: tuple[tuple[str, str], ...] = (
    ("command_substitution", "$(touch CANARY)"),
    ("shell_separator", "x;touch CANARY;y"),
    ("single_quote_breakout", "x';touch CANARY;'"),
    ("double_quote_breakout", 'x";touch CANARY;"'),
)

_OPERAND = (
    r"(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'"
    r"|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*"
    r"|\$[0-9]+|[^\s)]+)"
)
_BASE64_OPERAND = re.compile(rf"base64\s+<\s+({_OPERAND})")
_BASENAME_OPERAND = re.compile(rf"\bbasename\s+({_OPERAND})")
_ENDPOINT = re.compile(r"/(inspect|detect|clean)\b")
_NAMED_PARAM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_POSITIONAL = re.compile(r"^[0-9]+$")
_EXPANSION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
# Set by the test harness, not by the caller path.
_HARNESS_VARS = frozenset({"WM", "AUTH_HEADER"})


@dataclass(frozen=True)
class Recipe:
    """One fenced bash recipe copied onto the agent host."""

    start: int
    endpoint: str
    body: str

    def label(self) -> str:
        return f"L{self.start} {self.endpoint}"


def _bash_fences(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    fences: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("```bash"):
            index += 1
            start = index + 1
            body: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                body.append(lines[index])
                index += 1
            fences.append((start, "\n".join(body)))
        index += 1
    return fences


def file_curl_recipes(text: str) -> list[Recipe]:
    """Bash fences that post a caller-supplied file to the watermark service."""
    recipes: list[Recipe] = []
    for start, body in _bash_fences(text):
        match = _ENDPOINT.search(body)
        if match is None:
            continue
        recipes.append(Recipe(start, f"/{match.group(1)}", body))
    return recipes


def _parameter(operand: str) -> tuple[str, str] | None:
    """Return ``("name", name)`` or ``("positional", n)`` when the slot is data."""
    quoted = False
    raw = operand
    if operand.startswith('"') and operand.endswith('"') and len(operand) >= 2:
        quoted = True
        raw = operand[1:-1]
    elif operand.startswith("'"):
        return None
    name = None
    if raw.startswith("${") and raw.endswith("}"):
        name = raw[2:-1]
    elif raw.startswith("$"):
        name = raw[1:]
    if name is None:
        return None
    if _NAMED_PARAM.fullmatch(name):
        return ("name", name)
    if not quoted and _POSITIONAL.fullmatch(name):
        return ("positional", name)
    if quoted and _POSITIONAL.fullmatch(name):
        return ("positional", name)
    return None


def bind_caller_path(body: str, payload: str) -> tuple[str, dict[str, str]]:
    """Apply the recipe's own path slots. See the module docstring."""
    operands = [match.group(1) for match in _BASE64_OPERAND.finditer(body)]
    operands += [match.group(1) for match in _BASENAME_OPERAND.finditer(body)]
    env: dict[str, str] = {}
    tokens: list[str] = []
    for operand in operands:
        param = _parameter(operand)
        if param is not None:
            kind, name = param
            if kind == "name":
                env[name] = payload
            continue
        quoted = (operand.startswith('"') and operand.endswith('"')) or (
            operand.startswith("'") and operand.endswith("'")
        )
        tokens.append(operand[1:-1] if quoted else operand)
    bound = body
    for token in sorted(set(tokens), key=len, reverse=True):
        pattern = re.compile(rf"(?<!\$)(?<!{{){re.escape(token)}")
        bound = pattern.sub(lambda _match, value=payload: value, bound)
    if not tokens and not env:
        # No ``base64``/``basename`` slot. A helper recipe still receives the
        # caller path through every parameter it expands, except the harness
        # variables this test sets itself.
        for match in _EXPANSION.finditer(body):
            name = match.group(1) or match.group(2)
            if name not in _HARNESS_VARS:
                env[name] = payload
    return bound, env


def _run_recipe(body: str, payload: str) -> Path:
    """Run ``body`` with ``payload`` bound as the caller path. Return the work dir."""
    work = Path(tempfile.mkdtemp(prefix="wm-skill-path-"))
    (work / payload).write_bytes(b"downloaded file")
    script, env = bind_caller_path(body, payload)
    child = os.environ.copy()
    child.pop("WATERMARKS_SERVER_API_KEY", None)
    child["WM"] = "http://127.0.0.1:9"
    child.update(env)
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required to run the skill recipe")
    subprocess.run(
        [bash, "--noprofile", "--norc", "-c", script, "bash", payload],
        cwd=work,
        env=child,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return work


RECIPES = file_curl_recipes(SKILL_MD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def installed_skill_text(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Install the skill the way Cursor, Claude Code, and Codex receive it."""
    home = tmp_path_factory.mktemp("agent-home")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "install_skill.py"),
            "--skill",
            SKILL_NAME,
            "--target",
            "cursor",
            "--cursor-home",
            str(home),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    installed = home / "skills" / SKILL_NAME / "SKILL.md"
    return installed.read_text(encoding="utf-8")


def test_skill_documents_inspect_and_clean_shell_recipes() -> None:
    endpoints = {recipe.endpoint for recipe in RECIPES}
    assert "/inspect" in endpoints
    assert "/clean" in endpoints


def test_installer_copies_skill_recipes_unchanged(installed_skill_text: str) -> None:
    assert file_curl_recipes(installed_skill_text) == RECIPES


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda recipe: recipe.label())
def test_file_recipe_does_not_execute_caller_path(
    recipe: Recipe, installed_skill_text: str
) -> None:
    """A downloaded filename must not run while this recipe builds the curl body."""
    installed = {item.label(): item for item in file_curl_recipes(installed_skill_text)}
    body = installed[recipe.label()].body
    fired: list[str] = []
    for kind, payload in PAYLOADS:
        work = _run_recipe(body, payload)
        if (work / CANARY_NAME).exists():
            fired.append(f"{kind} ({payload})")
    assert not fired, f"{recipe.label()} executed the caller path: {', '.join(fired)}"


def test_quoted_parameter_expansion_is_not_shell_source() -> None:
    """A quoted expansion is data. This is the shape that must keep passing."""
    body = "\n".join(
        [
            "file_b64=$(base64 < \"$path\" | tr -d '\\n')",
            'name=$(basename "$path")',
            (
                'curl -s -X POST "${AUTH_HEADER[@]}" "$WM/inspect" '
                + "-H 'Content-Type: application/json' \\"
            ),
            '  -d "{\\"file\\": \\"${file_b64}\\", \\"name\\": \\"${name}\\"}"',
        ]
    )
    for _kind, payload in PAYLOADS:
        work = _run_recipe(body, payload)
        assert not (work / CANARY_NAME).exists()
