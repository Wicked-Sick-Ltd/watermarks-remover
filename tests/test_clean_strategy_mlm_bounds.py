"""Bounds for Layer B ``mlm`` infill.

These tests are the spec for the unbounded ``roberta-large`` fill-mask path.
They fail while a request can load that model, retokenize every added part, or
run fill-mask with no wall-clock budget and no text/window cap. They pass once
those limits exist.

Required limits, exercised below:

* A request must not construct the fill-mask pipeline. Loading ``roberta-large``
  belongs at startup or behind an operator preload, not on attacker input.
* One request may not run fill-mask on more than ``MLM_TEXT_CHAR_BUDGET``
  characters, and may not invoke the pipeline more than ``MLM_WINDOW_BUDGET``
  times. Reject or leave the text unchanged; do not start inference first.
* ``apply_strategy``'s ``timeout`` (the budget ``/clean`` already uses for LLM
  rewrites, and ``rewrite_text.py --timeout``) must abort an ``mlm`` step.
  ``TimeoutError`` is fine; ``/clean`` already turns it into a 400.
* Chunking must tokenize a linear amount of text. Retokenizing the growing
  chunk on every whitespace-separated part is the quadratic cost.

The stand-in ``transformers`` package mirrors Hugging Face transformers
5.15.0 (``service/scripts/requirements-markllm.txt``): ``pipeline("fill-mask")``
loads the named model on the caller thread, and ``Pipeline.__call__`` runs
``run_single`` to completion with no timeout parameter. ``time.sleep`` stands
in for that synchronous forward. It does not honor a timeout kwarg, because
``FillMaskPipeline.__call__`` does not either.
"""

from __future__ import annotations

import base64
import http.client
import importlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rewrite_text
import server

# Modest per-request ceilings. Inputs under both still have to honor timeout.
MLM_TEXT_CHAR_BUDGET = 8192
MLM_WINDOW_BUDGET = 4
# Shorter than a blocked forward, longer than a timeout abort.
INFER_DELAY_SECONDS = 4.0
TIMEOUT_BUDGET_SECONDS = 0.4
TIMEOUT_SLACK_SECONDS = 2.5

_STUB = '''\
import threading
import time

load_calls = []
infer_calls = []
tokenize_inputs = []
INFER_DELAY = 0.0
started = threading.Event()


class _Ids:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class _Tokenizer:
    mask_token = "<mask>"
    mask_token_id = 1

    def __call__(self, text, **kwargs):
        tokenize_inputs.append(text)
        return {"input_ids": _Ids(len(text))}


class _Pipeline:
    """Synchronous fill-mask call, matching Pipeline.run_single."""

    def __init__(self):
        self.tokenizer = _Tokenizer()

    def __call__(self, inputs, top_k=1, **kwargs):
        started.set()
        infer_calls.append(len(inputs) if isinstance(inputs, str) else -1)
        if INFER_DELAY:
            time.sleep(INFER_DELAY)
        return [{"score": 1.0, "token": 2, "token_str": " term", "sequence": ""}]


def pipeline(task, **kwargs):
    load_calls.append({"task": task, "model": kwargs.get("model")})
    return _Pipeline()
'''

_LLM_ENV = (
    "WATERMARKS_REWRITE_BACKEND",
    "WATERMARKS_REWRITE_MODEL",
    "WATERMARKS_REWRITE_BASE_URL",
    "WATERMARKS_REWRITE_API_KEY",
    "WATERMARKS_REWRITE_ALLOW_REMOTE",
    "WATERMARKS_REWRITE_TEMPERATURE",
    "WATERMARKS_REWRITE_REASONING_EFFORT",
)


@pytest.fixture(autouse=True)
def _clear_mlm_cache():
    rewrite_text._MLM_CACHE.clear()
    yield
    rewrite_text._MLM_CACHE.clear()


@pytest.fixture
def no_llm(monkeypatch: pytest.MonkeyPatch):
    """mlm-only must reach infill with the rewrite backend unset."""
    for key in _LLM_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def tf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install the synchronous fill-mask stand-in where importlib can find it."""
    root = tmp_path / "tfsrc"
    package = root / "transformers"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_STUB)
    monkeypatch.syspath_prepend(str(root))
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()
    imported = importlib.import_module("transformers")
    yield imported


@pytest.fixture
def short_budget(monkeypatch: pytest.MonkeyPatch):
    """Point the rewrite budget ``/clean`` inherits at a testable value.

    ``_apply_layer_b`` omits ``timeout``, so the HTTP path uses
    ``apply_strategy``'s default. Shrinking that default is the budget the
    request is supposed to enforce.
    """
    defaults = dict(rewrite_text.apply_strategy.__kwdefaults__ or {})
    defaults["timeout"] = TIMEOUT_BUDGET_SECONDS
    monkeypatch.setattr(rewrite_text.apply_strategy, "__kwdefaults__", defaults)


@pytest.fixture
def http_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server, "API_KEY", "")
    srv = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _preload(tf, delay: float) -> None:
    tf.INFER_DELAY = delay
    tf.started.clear()
    pipe = tf.pipeline("fill-mask", model="roberta-large")
    tf.load_calls.clear()
    tf.infer_calls.clear()
    tf.tokenize_inputs.clear()
    rewrite_text._MLM_CACHE.clear()
    rewrite_text._MLM_CACHE["pipeline"] = pipe
    rewrite_text._MLM_CACHE["mask_token"] = pipe.tokenizer.mask_token


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def _over_char_budget() -> str:
    unit = "word "
    text = unit * ((MLM_TEXT_CHAR_BUDGET // len(unit)) + 2)
    assert len(text) > MLM_TEXT_CHAR_BUDGET
    return text.strip()


def _over_window_budget() -> str:
    # Character-length tokenization yields one 512-token window per 512 chars.
    text = "word " * ((512 * (MLM_WINDOW_BUDGET + 2)) // 5)
    assert len(text) <= MLM_TEXT_CHAR_BUDGET
    return text.strip()


def _body(text: str, strategy: str | None = "mlm@1") -> dict:
    payload: dict = {
        "file": base64.b64encode(text.encode()).decode("ascii"),
        "name": "notes.txt",
    }
    if strategy is not None:
        payload["options"] = {"strategy": strategy}
    return payload


def _post(srv: server.ThreadingHTTPServer, path: str, payload: dict) -> tuple[float, int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=30)
    raw = json.dumps(payload).encode()
    started = time.monotonic()
    try:
        conn.request(
            "POST",
            path,
            body=raw,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        data = resp.read()
        elapsed = time.monotonic() - started
        return elapsed, resp.status, json.loads(data) if data else {}
    finally:
        conn.close()


def _assert_timed_out(tf, started: float, exc: BaseException | None, completed: bool) -> None:
    elapsed = time.monotonic() - started
    assert tf.started.is_set(), "fill-mask never started; the request never reached inference"
    assert not completed, f"mlm finished in {elapsed:.2f}s with no timeout"
    assert elapsed < TIMEOUT_SLACK_SECONDS, f"caller stayed blocked for {elapsed:.2f}s"
    assert isinstance(exc, (TimeoutError, ValueError, RuntimeError))


def _run_bounded(tf, fn):
    wall = time.monotonic()
    completed = False
    exc: BaseException | None = None
    try:
        fn()
        completed = True
    except (TimeoutError, ValueError, RuntimeError) as e:
        exc = e
    _assert_timed_out(tf, wall, exc, completed)


def _swallow(fn):
    """A fix may reject or return the text unchanged. Inference is the failure."""
    try:
        return fn()
    except (TimeoutError, ValueError, RuntimeError, SystemExit):
        return None


# --- model load stays off the request path ---------------------------------


def test_apply_strategy_does_not_load_model_on_request(tf, no_llm):
    _swallow(
        lambda: rewrite_text.apply_strategy(
            "word word word word",
            [("mlm", 1.0)],
            backend="print-prompt",
            model=None,
            base_url=None,
            api_key=None,
        )
    )
    assert tf.load_calls == []


def test_rewrite_does_not_load_model_on_request(tf, no_llm):
    _swallow(
        lambda: rewrite_text.rewrite(
            "word word word word",
            backend="ollama",
            model="local",
            base_url="http://127.0.0.1:11434",
            api_key=None,
            tactic="mlm",
            lang="French",
            original_lang="English",
            timeout=TIMEOUT_BUDGET_SECONDS,
            layer_a_after=False,
            temperature=0.0,
            candidates=1,
            max_loops=1,
            rewrite_level=1.0,
        )
    )
    assert tf.load_calls == []


def test_clean_payload_does_not_load_model_on_request(tf, no_llm):
    _swallow(
        lambda: server._clean_payload(b"word word word word", "notes.txt", {"strategy": "mlm@1"})
    )
    assert tf.load_calls == []


def test_default_strategy_does_not_load_model_on_request(tf, no_llm, monkeypatch):
    monkeypatch.setattr(server, "_DEFAULT_STRATEGY", "mlm@1")
    _swallow(lambda: server._clean_payload(b"word word word word", "notes.txt", {}))
    assert tf.load_calls == []


def test_clean_batch_does_not_load_model_on_request(tf, no_llm, http_server):
    elapsed, status, body = _post(
        http_server,
        "/clean/batch",
        {"files": [_body("word word word word")]},
    )
    assert tf.load_calls == [], (elapsed, status, body, tf.load_calls)


def test_cli_strategy_does_not_load_model(tf, no_llm, monkeypatch, tmp_path, capsys):
    path = tmp_path / "notes.txt"
    path.write_text("word word word word\n")
    seen: dict[str, list] = {}
    real_read = rewrite_text.read_text_input

    def wrapped(file_path, **kwargs):
        # A startup load may run before the input is read. A load after this
        # point is the request pulling roberta-large for this file.
        seen["before"] = list(tf.load_calls)
        return real_read(file_path, **kwargs)

    monkeypatch.setattr(rewrite_text, "read_text_input", wrapped)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rewrite_text.py", str(path), "--strategy", "mlm@1", "--timeout", "0.4"],
    )
    _swallow(rewrite_text.main)
    assert tf.load_calls[len(seen.get("before", [])) :] == []
    capsys.readouterr()


# --- character and window caps ----------------------------------------------


def test_apply_strategy_rejects_text_over_char_budget(tf, no_llm):
    _preload(tf, delay=0)
    text = _over_char_budget()
    _swallow(
        lambda: rewrite_text.apply_strategy(
            text,
            [("mlm", 1.0)],
            backend="print-prompt",
            model=None,
            base_url=None,
            api_key=None,
        )
    )
    assert tf.infer_calls == []
    assert tf.load_calls == []


def test_clean_payload_rejects_text_over_char_budget(tf, no_llm):
    _preload(tf, delay=0)
    text = _over_char_budget()
    _swallow(lambda: server._clean_payload(text.encode(), "notes.txt", {"strategy": "mlm@1"}))
    assert tf.infer_calls == []


def test_clean_batch_rejects_text_over_char_budget(tf, no_llm, http_server):
    _preload(tf, delay=0)
    _elapsed, _status, body = _post(
        http_server,
        "/clean/batch",
        {"files": [_body(_over_char_budget())]},
    )
    assert tf.infer_calls == [], body


def test_apply_strategy_caps_fill_mask_windows(tf, no_llm):
    _preload(tf, delay=0)
    _swallow(
        lambda: rewrite_text.apply_strategy(
            _over_window_budget(),
            [("mlm", 1.0)],
            backend="print-prompt",
            model=None,
            base_url=None,
            api_key=None,
        )
    )
    assert len(tf.infer_calls) <= MLM_WINDOW_BUDGET


def test_clean_payload_caps_fill_mask_windows(tf, no_llm):
    _preload(tf, delay=0)
    _swallow(
        lambda: server._clean_payload(
            _over_window_budget().encode(), "notes.txt", {"strategy": "mlm@1"}
        )
    )
    assert len(tf.infer_calls) <= MLM_WINDOW_BUDGET


def test_rewrite_caps_fill_mask_windows(tf):
    _preload(tf, delay=0)
    _swallow(
        lambda: rewrite_text.rewrite(
            _over_window_budget(),
            backend="ollama",
            model="local",
            base_url="http://127.0.0.1:11434",
            api_key=None,
            tactic="mlm",
            lang="French",
            original_lang="English",
            timeout=30.0,
            layer_a_after=False,
            temperature=0.0,
            candidates=1,
            max_loops=1,
            rewrite_level=1.0,
        )
    )
    assert len(tf.infer_calls) <= MLM_WINDOW_BUDGET


# --- tokenization is linear --------------------------------------------------


def test_mlm_infill_tokenizes_a_chunk_once(tf, no_llm):
    _preload(tf, delay=0)
    text = _words(80)
    rewrite_text.apply_strategy(
        text,
        [("mlm", 1.0)],
        backend="print-prompt",
        model=None,
        base_url=None,
        api_key=None,
    )
    tokenized = sum(len(chunk) for chunk in tf.tokenize_inputs)
    assert tokenized <= 2 * len(text), tokenized


def test_rewrite_tokenizes_a_chunk_once(tf):
    _preload(tf, delay=0)
    text = _words(80)
    rewrite_text.rewrite(
        text,
        backend="ollama",
        model="local",
        base_url="http://127.0.0.1:11434",
        api_key=None,
        tactic="mlm",
        lang="French",
        original_lang="English",
        timeout=30.0,
        layer_a_after=False,
        temperature=0.0,
        candidates=1,
        max_loops=1,
        rewrite_level=1.0,
    )
    tokenized = sum(len(chunk) for chunk in tf.tokenize_inputs)
    assert tokenized <= 2 * len(text), tokenized


# --- wall-clock budget -------------------------------------------------------


def test_apply_strategy_mlm_honors_timeout(tf, no_llm):
    _preload(tf, delay=INFER_DELAY_SECONDS)

    def run():
        rewrite_text.apply_strategy(
            "word word word word",
            [("mlm", 1.0)],
            backend="print-prompt",
            model=None,
            base_url=None,
            api_key=None,
            timeout=TIMEOUT_BUDGET_SECONDS,
        )

    _run_bounded(tf, run)


def test_apply_strategy_mixed_mlm_honors_timeout(tf, no_llm, monkeypatch):
    _preload(tf, delay=INFER_DELAY_SECONDS)
    monkeypatch.setattr(
        rewrite_text, "_generate_once", lambda *args, **kwargs: "word word word word"
    )

    def run():
        rewrite_text.apply_strategy(
            "word word word word",
            [("paraphrase", 0.8), ("mlm", 1.0)],
            backend="ollama",
            model="local",
            base_url="http://127.0.0.1:11434",
            api_key=None,
            timeout=TIMEOUT_BUDGET_SECONDS,
        )

    _run_bounded(tf, run)


def test_rewrite_mlm_honors_timeout(tf):
    _preload(tf, delay=INFER_DELAY_SECONDS)

    def run():
        rewrite_text.rewrite(
            "word word word word",
            backend="ollama",
            model="local",
            base_url="http://127.0.0.1:11434",
            api_key=None,
            tactic="mlm",
            lang="French",
            original_lang="English",
            timeout=TIMEOUT_BUDGET_SECONDS,
            layer_a_after=False,
            temperature=0.0,
            candidates=1,
            max_loops=1,
            rewrite_level=1.0,
        )

    _run_bounded(tf, run)


def test_clean_payload_mlm_honors_timeout(tf, no_llm, short_budget):
    _preload(tf, delay=INFER_DELAY_SECONDS)

    def run():
        server._clean_payload(b"word word word word", "notes.txt", {"strategy": "mlm@1"})

    _run_bounded(tf, run)


def test_http_clean_mlm_honors_timeout(tf, no_llm, short_budget, http_server):
    _preload(tf, delay=INFER_DELAY_SECONDS)
    elapsed, status, body = _post(http_server, "/clean", _body("word word word word"))
    assert tf.started.is_set(), body
    assert status == 400, (status, body)
    assert elapsed < TIMEOUT_SLACK_SECONDS


def test_http_clean_batch_mlm_honors_timeout(tf, no_llm, short_budget, http_server):
    _preload(tf, delay=INFER_DELAY_SECONDS)
    elapsed, status, body = _post(
        http_server,
        "/clean/batch",
        {"files": [_body("word word word word")]},
    )
    assert tf.started.is_set(), body
    assert status == 200
    assert body["results"][0]["ok"] is False
    assert elapsed < TIMEOUT_SLACK_SECONDS


def test_cli_strategy_mlm_honors_timeout(tf, no_llm, monkeypatch, tmp_path, capsys):
    _preload(tf, delay=INFER_DELAY_SECONDS)
    path = tmp_path / "notes.txt"
    path.write_text("word word word word\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rewrite_text.py",
            str(path),
            "--strategy",
            "mlm@1",
            "--timeout",
            str(TIMEOUT_BUDGET_SECONDS),
        ],
    )
    wall = time.monotonic()
    code = rewrite_text.main()
    elapsed = time.monotonic() - wall
    capsys.readouterr()
    assert tf.started.is_set()
    assert code != 0
    assert elapsed < TIMEOUT_SLACK_SECONDS
