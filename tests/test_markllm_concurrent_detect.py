"""Spec for uncapped MarkLLM text-detection processes.

These tests fail while the vulnerability is present. They pass once text
detection cannot be driven into an unbounded set of MarkLLM children.

Contract:
- With MARKLLM_DIR set and the service API key left empty (the default),
  every text-detection entry point refuses the request and does not spawn
  detect_text_watermark.py. A browser-simple Content-Type (text/plain) is
  enough to deliver the JSON body today, so the refusal has to happen before
  any child is created.
- With a key configured and presented, at most one MarkLLM detection child
  may be alive at a time, including when WATERMARKS_MARKLLM_PORT names a
  worker that is down and detection falls back to a one-shot process.
- That child gets a finite address-space limit even when
  WATERMARKS_MARKLLM_RLIMIT_AS is unset.

The checkout's .venv interpreter is a local stand-in that holds the process
open, records the argv the parent built, then execs the real interpreter on
detect_text_watermark.py. torch/transformers are not installed here, so the
real script stops at the model import inside _load_algorithm.

Not covered here:
- rewrite_text.py's CLI detector (operator flags, not this HTTP surface)
- POST /watermark, which loads MarkLLM in-process under a lock and a
  two-model cache rather than spawning one process per request
- image, audio, and container detection, which do not call MarkLLM
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import resource
import socket
import threading
import time
from pathlib import Path

import pytest

import server

PLACEHOLDER_API_KEY = "placeholder-api-key"
CONCURRENCY = 4
TEXT = b"short note for text detection\n"

# Holds the child the parent actually execs, then replaces itself with the
# real interpreter so detect_text_watermark.py runs with the same argv tail.
_STUB = """\
#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

import resource

state = Path(os.environ["MARKLLM_SPAWN_STATE"])
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
record = {
    "pid": os.getpid(),
    "argv": sys.argv,
    "rlimit_as": [soft, hard],
}
(state / f"{os.getpid()}.json").write_text(json.dumps(record))
release = state / "release"
deadline = time.time() + 20
while time.time() < deadline and not release.exists():
    time.sleep(0.02)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
"""

OPERATIONS = (
    "detect",
    "detect_batch",
    "inspect",
    "inspect_batch",
    "clean",
    "clean_batch",
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _request_for(operation: str) -> tuple[str, dict]:
    one = {"file": _b64(TEXT), "name": "notes.txt"}
    if operation == "detect":
        return "/detect", one
    if operation == "detect_batch":
        return "/detect/batch", {"files": [dict(one)]}
    if operation == "inspect":
        return "/inspect", {**one, "detect": True}
    if operation == "inspect_batch":
        return "/inspect/batch", {"files": [dict(one)], "detect": True}
    if operation == "clean":
        return "/clean", {
            **one,
            "options": {"detect_before": True, "detect_after": True},
        }
    if operation == "clean_batch":
        return "/clean/batch", {
            "files": [
                {
                    **one,
                    "options": {"detect_before": True, "detect_after": True},
                }
            ]
        }
    raise AssertionError(operation)


def _make_checkout(tmp_path: Path, state: Path) -> Path:
    upstream = tmp_path / "MarkLLM"
    (upstream / "watermark").mkdir(parents=True)
    (upstream / "config").mkdir()
    (upstream / "config" / "KGW.json").write_text('{"threshold": 4.0}\n', encoding="utf-8")
    stub = upstream / ".venv" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(0o755)
    state.mkdir()
    return upstream


def _closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _live_children(state: Path) -> int:
    alive = 0
    for path in state.glob("*.json"):
        if Path(f"/proc/{path.stem}").exists():
            alive += 1
    return alive


def _records(state: Path) -> list[dict]:
    rows = []
    for path in sorted(state.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _post(
    port: int,
    path: str,
    payload: dict,
    *,
    api_key: str | None,
    content_type: str,
) -> tuple[int | str, dict | str]:
    headers = {"Content-Type": content_type}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=25)
    try:
        raw = json.dumps(payload).encode("utf-8")
        conn.request("POST", path, body=raw, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        try:
            parsed: dict | str = json.loads(data) if data else {}
        except json.JSONDecodeError:
            parsed = data.decode("utf-8", errors="replace")
        return resp.status, parsed
    except Exception as exc:
        return "error", str(exc)
    finally:
        conn.close()


def _burst(port: int, path: str, payload: dict, *, api_key: str | None, content_type: str):
    results: list[tuple[int | str, dict | str] | None] = [None] * CONCURRENCY

    def run(index: int) -> None:
        results[index] = _post(
            port,
            path,
            payload,
            api_key=api_key,
            content_type=content_type,
        )

    threads = [threading.Thread(target=run, args=(i,)) for i in range(CONCURRENCY)]
    for thread in threads:
        thread.start()
    return threads, results


def _join(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(timeout=25)


@pytest.fixture
def markllm_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """HTTP service with MarkLLM enabled and the memory cap left unset."""
    state = tmp_path / "spawn-state"
    upstream = _make_checkout(tmp_path, state)
    for name in (
        "WATERMARKS_MARKLLM_RLIMIT_AS",
        "WATERMARKS_MARKLLM_PORT",
        "WATERMARKS_MARKLLM_TIMEOUT",
        "WATERMARKS_MARKLLM_SCHEME",
        "WATERMARKS_GUMBEL_KEY",
        "WATERMARKS_SERVER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MARKLLM_DIR", str(upstream))
    monkeypatch.setenv("MARKLLM_SPAWN_STATE", str(state))

    previous_key = server.API_KEY
    server.API_KEY = ""
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "port": httpd.server_address[1],
            "state": state,
        }
    finally:
        (state / "release").write_text("1", encoding="utf-8")
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        server.API_KEY = previous_key


def _wait_peak(state: Path, seconds: float) -> int:
    peak = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        peak = max(peak, _live_children(state))
        if peak >= 2:
            time.sleep(0.3)
            peak = max(peak, _live_children(state))
            break
        time.sleep(0.02)
    return peak


def _assert_real_detect_command(records: list[dict]) -> None:
    assert records, "expected the service to run MarkLLM detection"
    for record in records:
        argv = record["argv"]
        assert any(str(arg).endswith("detect_text_watermark.py") for arg in argv), argv
        assert "detect" in argv
        assert argv[argv.index("--scheme") + 1] == "kgw"
        assert "--json" in argv


@pytest.mark.parametrize("operation", OPERATIONS)
def test_open_api_rejects_text_detection_without_spawning(markllm_service, operation):
    """Empty API key is the default. Reaching text detection must not start a child."""
    path, payload = _request_for(operation)
    state: Path = markllm_service["state"]
    threads, results = _burst(
        markllm_service["port"],
        path,
        payload,
        api_key=None,
        content_type="text/plain",
    )
    try:
        peak = _wait_peak(state, 3)
    finally:
        (state / "release").write_text("1", encoding="utf-8")
        _join(threads)
    statuses = [item[0] for item in results if item is not None]
    assert peak == 0, f"spawned MarkLLM children before auth refused the call: {statuses}"
    assert not _records(state), statuses
    assert statuses == [401] * CONCURRENCY, results


@pytest.mark.parametrize("operation", OPERATIONS)
def test_concurrent_text_detection_runs_one_markllm_child(markllm_service, operation):
    """Authorized callers still share one in-flight MarkLLM detection."""
    server.API_KEY = PLACEHOLDER_API_KEY
    path, payload = _request_for(operation)
    state: Path = markllm_service["state"]
    threads, results = _burst(
        markllm_service["port"],
        path,
        payload,
        api_key=PLACEHOLDER_API_KEY,
        content_type="application/json",
    )
    try:
        peak = _wait_peak(state, 8)
    finally:
        (state / "release").write_text("1", encoding="utf-8")
        _join(threads)

    records = _records(state)
    _assert_real_detect_command(records)
    bodies = json.dumps([item[1] for item in results if item is not None])
    assert "dependencies missing" in bodies, bodies[:800]
    if operation in ("clean", "clean_batch"):
        assert bodies.count("dependencies missing") >= 2, bodies[:800]
    assert peak <= 1, (
        f"{operation}: {peak} MarkLLM children ran at once "
        f"(records={len(records)}, statuses={[item[0] for item in results if item]})"
    )


def test_markllm_child_address_space_is_limited_by_default(markllm_service):
    """WATERMARKS_MARKLLM_RLIMIT_AS unset must still cap the child address space."""
    server.API_KEY = PLACEHOLDER_API_KEY
    path, payload = _request_for("detect")
    state: Path = markllm_service["state"]
    threads, results = _burst(
        markllm_service["port"],
        path,
        payload,
        api_key=PLACEHOLDER_API_KEY,
        content_type="application/json",
    )
    # One child is enough; release as soon as it has recorded its limit.
    deadline = time.time() + 8
    while time.time() < deadline and not any(state.glob("*.json")):
        time.sleep(0.02)
    (state / "release").write_text("1", encoding="utf-8")
    _join(threads)

    records = _records(state)
    _assert_real_detect_command(records)
    limit = records[0]["rlimit_as"][0]
    assert isinstance(limit, int)
    assert 0 < limit < resource.RLIM_INFINITY, (
        f"child address space is not limited ({limit!r}); statuses="
        f"{[item[0] for item in results if item]}"
    )


def test_unreachable_markllm_worker_does_not_lift_the_cap(markllm_service, monkeypatch):
    """A dead WATERMARKS_MARKLLM_PORT falls back to a subprocess that stays capped."""
    monkeypatch.setenv("WATERMARKS_MARKLLM_PORT", str(_closed_port()))
    server.API_KEY = PLACEHOLDER_API_KEY
    path, payload = _request_for("detect")
    state: Path = markllm_service["state"]
    threads, results = _burst(
        markllm_service["port"],
        path,
        payload,
        api_key=PLACEHOLDER_API_KEY,
        content_type="application/json",
    )
    try:
        peak = _wait_peak(state, 8)
    finally:
        (state / "release").write_text("1", encoding="utf-8")
        _join(threads)

    records = _records(state)
    _assert_real_detect_command(records)
    bodies = json.dumps([item[1] for item in results if item is not None])
    assert "dependencies missing" in bodies
    assert peak <= 1, (
        f"worker fallback started {peak} children at once "
        f"(records={len(records)})"
    )
