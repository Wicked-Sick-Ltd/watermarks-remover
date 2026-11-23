"""In-place backup must not copy a symlink target into a new file.

``backup_path`` is shared by ``clean_file.py``, ``clean_text.py``,
``clean_image.py``, the ``clean_staged.py`` pre-commit wrapper, and the
vendored skill copy of the helper. These tests are the spec for that helper:
they fail while a symlink source is followed, and they pass once the backup
refuses the final component before creating or writing any file.

``clean_audio.py``, ``clean_video.py``, and ``clean_ctrlregen.py`` do not call
``backup_path`` and are not part of this spec.
"""

from __future__ import annotations

import builtins
import importlib.util
import io
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
SKILL_SCRIPTS = ROOT / "skills" / "clean-user-facing-text" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common

# Obvious placeholder. Not a credential.
SECRET = b"PLACEHOLDER_CREDENTIAL_MATERIAL\n"
PUBLIC = b"ordinary file body\n"

_CLEARED_ENV = (
    "WATERMARKS_SYNTHID_SCORER_URL",
    "REVERSE_SYNTHID_DIR",
    "NOAI_WATERMARK_DIR",
    "MARKDIFFUSION_DIR",
)


def _load_skill_common():
    spec = importlib.util.spec_from_file_location(
        "skill_common_symlink_backup",
        SKILL_SCRIPTS / "common.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _secret(tmp_path: Path) -> Path:
    private = tmp_path / "private"
    private.mkdir()
    private.chmod(0o700)
    secret = private / "credential.txt"
    secret.write_bytes(SECRET)
    secret.chmod(0o600)
    return secret


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def _workspace_link(tmp_path: Path, name: str, secret: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    link = workspace / name
    _symlink(link, secret)
    return workspace, link


def _materialized(workspace: Path) -> list[str]:
    rows: list[str] = []
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        data = path.read_bytes()
        rows.append(
            f"{path.relative_to(workspace)} mode={mode:04o} "
            f"contains_secret={SECRET in data} size={len(data)}"
        )
    return rows


def _assert_symlink_not_materialized(workspace: Path, link: Path, secret: Path) -> None:
    leaked = _materialized(workspace)
    assert leaked == [], "in-place clean created a file from the symlink:\n" + "\n".join(leaked)
    assert link.is_symlink()
    assert os.readlink(link) == str(secret)
    assert secret.read_bytes() == SECRET
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in _CLEARED_ENV:
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )


class _CreateSpy:
    """Record every O_CREAT so a backup that is created and then deleted still fails."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self._real_open = os.open

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        if flags & os.O_CREAT:
            self.created.append(os.fspath(path))
        return self._real_open(path, flags, mode, dir_fd=dir_fd)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(os, "open", self.open)


class _SwapAfterFirstLookup:
    """After the first non-O_NOFOLLOW lookup, replace the source with a symlink.

    A check that stats the path and then reads it again follows the swapped
    link. One ``os.open`` with ``O_NOFOLLOW`` is a single lookup: it either
    opens the regular file that was there, or fails if the swap already
    happened, and it never copies the link target.
    """

    def __init__(self, source: Path, secret: Path) -> None:
        self.source = Path(os.path.abspath(source))
        self.secret = Path(os.path.abspath(secret))
        self.phase = "armed"
        self._real_open = os.open
        self._real_stat = os.stat
        self._real_io_open = io.open

    def _hit(self, path) -> bool:
        try:
            return os.path.abspath(os.fspath(path)) == str(self.source)
        except (TypeError, ValueError):
            return False

    def _arm(self) -> None:
        if self.phase == "armed":
            self.phase = "pending_swap"

    def _swap(self) -> None:
        if self.phase != "pending_swap":
            return
        self.phase = "done"
        os.unlink(self.source)
        os.symlink(self.secret, self.source)

    def stat(self, path, *, dir_fd=None, follow_symlinks=True):
        hit = self._hit(path)
        if hit:
            self._arm()
        try:
            return self._real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        finally:
            if hit:
                self._swap()

    def lstat(self, path, *, dir_fd=None):
        return self.stat(path, dir_fd=dir_fd, follow_symlinks=False)

    def os_open(self, path, flags, mode=0o777, *, dir_fd=None):
        hit = self._hit(path)
        nofollow = bool(hit and (flags & getattr(os, "O_NOFOLLOW", 0)))
        if hit and not nofollow:
            self._arm()
        try:
            return self._real_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            if hit and not nofollow:
                self._swap()

    def io_open(self, file, mode="r", *args, **kwargs):
        hit = self._hit(file)
        if hit:
            self._arm()
        try:
            return self._real_io_open(file, mode, *args, **kwargs)
        finally:
            if hit:
                self._swap()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pathlib

        monkeypatch.setattr(os, "stat", self.stat)
        monkeypatch.setattr(os, "lstat", self.lstat)
        monkeypatch.setattr(os, "open", self.os_open)
        monkeypatch.setattr(io, "open", self.io_open)
        monkeypatch.setattr(builtins, "open", self.io_open)
        monkeypatch.setattr(pathlib.io, "open", self.io_open)


def _assert_swap_does_not_copy_secret(checkout: Path, secret: Path) -> None:
    assert secret.read_bytes() == SECRET
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    leaked = _materialized(checkout)
    secret_copies = [row for row in leaked if "contains_secret=True" in row]
    assert secret_copies == [], (
        "backup followed a symlink swapped in after the check:\n" + "\n".join(leaked)
    )


def _call_backup(backup_path, src: Path) -> None:
    try:
        backup_path(src)
    except (OSError, SystemExit):
        return


@pytest.mark.parametrize("loader", ["service", "skill"])
def test_backup_path_refuses_symlink_without_creating_a_file(
    tmp_path: Path, monkeypatch, loader: str
):
    secret = _secret(tmp_path)
    workspace, link = _workspace_link(tmp_path, "notes.md", secret)
    spy = _CreateSpy()
    spy.install(monkeypatch)
    backup = common.backup_path if loader == "service" else _load_skill_common().backup_path
    _call_backup(backup, link)
    assert spy.created == [], f"created files before refusing the symlink: {spy.created}"
    _assert_symlink_not_materialized(workspace, link, secret)


@pytest.mark.parametrize("loader", ["service", "skill"])
def test_backup_path_does_not_follow_a_symlink_swapped_in_after_the_check(
    tmp_path: Path, monkeypatch, loader: str
):
    secret = _secret(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = checkout / "notes.md"
    source.write_bytes(PUBLIC)
    swap = _SwapAfterFirstLookup(source, secret)
    swap.install(monkeypatch)
    backup = common.backup_path if loader == "service" else _load_skill_common().backup_path
    _call_backup(backup, source)
    _assert_swap_does_not_copy_secret(checkout, secret)
    bak = checkout / "notes.md.bak"
    if bak.exists() and not bak.is_symlink():
        assert bak.read_bytes() == PUBLIC


@pytest.mark.parametrize(
    ("script", "name"),
    [
        ("clean_file.py", "notes.md"),
        ("clean_file.py", "notes.txt"),
        ("clean_file.py", "photo.png"),
        ("clean_text.py", "notes.txt"),
        ("clean_image.py", "photo.png"),
        ("clean_staged.py", "notes.md"),
    ],
)
def test_service_in_place_clean_does_not_copy_symlink_target(
    tmp_path: Path, script: str, name: str
):
    secret = _secret(tmp_path)
    workspace, link = _workspace_link(tmp_path, name, secret)
    # clean_staged.py's only mode is in-place; it has no --in-place flag.
    args = [str(link)] if script == "clean_staged.py" else [str(link), "--in-place"]
    result = _run(SCRIPTS / script, *args)
    _assert_symlink_not_materialized(workspace, link, secret)
    assert SECRET.decode() not in result.stdout
    assert SECRET.decode() not in result.stderr


def test_skill_clean_text_in_place_does_not_copy_symlink_target(tmp_path: Path):
    secret = _secret(tmp_path)
    workspace, link = _workspace_link(tmp_path, "notes.txt", secret)
    result = _run(SKILL_SCRIPTS / "clean_text.py", str(link), "--in-place")
    _assert_symlink_not_materialized(workspace, link, secret)
    assert SECRET.decode() not in result.stdout
    assert SECRET.decode() not in result.stderr


def test_regular_file_in_place_still_backs_up_that_file(tmp_path: Path):
    workspace = tmp_path / "checkout"
    workspace.mkdir()
    source = workspace / "notes.txt"
    source.write_bytes(PUBLIC)
    result = _run(SCRIPTS / "clean_text.py", str(source), "--in-place")
    assert result.returncode == 0
    bak = workspace / "notes.txt.bak"
    assert bak.is_file()
    assert not bak.is_symlink()
    assert bak.read_bytes() == PUBLIC
    assert source.is_file()
    assert not source.is_symlink()
