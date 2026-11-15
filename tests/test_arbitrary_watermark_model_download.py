"""Spec tests for caller-chosen Hugging Face model ids on text watermarking.

An unauthenticated client of POST /watermark or POST /watermark/batch must not
be able to choose which causal LM the server downloads and loads. The operator
default (``facebook/opt-1.3b`` when ``WATERMARKS_ALLOWED_TEXT_MODELS`` is unset)
is the only model id a request may cause ``from_pretrained`` to see. These tests
call the real HTTP handlers and the real transformers 5.15.0 / huggingface_hub
1.27.0 loaders (the versions pinned in ``service/scripts/requirements-markllm.txt``),
pointed at a local stand-in of the Hub so a forbidden id writes its weight file
into ``HF_HUB_CACHE`` without fetching a multi-gigabyte repository.

They fail while a request body can still select that repository. A fix passes
them by refusing the id before ``AutoTokenizer.from_pretrained`` and
``AutoModelForCausalLM.from_pretrained``, including when a bearer token is
presented and when the core client's timeout has already expired.

Not covered: ``detect_text_watermark.py`` CLI ``--model``. That value is the
operator's own argv, not a request body from a client of the HTTP service.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import types
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import server  # type: ignore
import synthid_text_server  # type: ignore
import text_watermark  # type: ignore

# The built-in default used when the operator has not configured an allowlist.
# A request may name this id. Every other id in these tests is attacker input.
OPERATOR_DEFAULT_MODEL = "facebook/opt-1.3b"
COMMIT = "a" * 40
PLACEHOLDER_API_KEY = "placeholder-not-a-secret"


def _require_ml_stack() -> Any:
    try:
        import huggingface_hub
        import huggingface_hub.constants as hf_constants
        import torch
        import transformers
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        pytest.fail(
            "these tests need the MarkLLM pins (transformers==5.15.0, "
            "huggingface_hub==1.27.0, tokenizers==0.22.2) and torch==2.15.0 "
            f"so the real from_pretrained path can run: {exc}"
        )
    return {
        "hf_constants": hf_constants,
        "huggingface_hub": huggingface_hub,
        "torch": torch,
        "transformers": transformers,
        "Tokenizer": Tokenizer,
        "WordLevel": WordLevel,
        "Whitespace": Whitespace,
        "GPT2Config": GPT2Config,
        "GPT2LMHeadModel": GPT2LMHeadModel,
    }


def _build_tiny_repo(stack: dict[str, Any], dest: Path) -> dict[str, bytes]:
    """A real tiny GPT-2 checkpoint, saved by the pinned transformers, for the local Hub."""
    dest.mkdir(parents=True, exist_ok=True)
    config = stack["GPT2Config"](
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = stack["GPT2LMHeadModel"](config)
    model.save_pretrained(dest)
    tokenizer = stack["Tokenizer"](
        stack["WordLevel"]({"<pad>": 0, "<eos>": 1, "<unk>": 2, "hello": 3}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = stack["Whitespace"]()
    tokenizer.save(str(dest / "tokenizer.json"))
    (dest / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 16,
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<pad>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    (dest / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    return {path.name: path.read_bytes() for path in dest.iterdir() if path.is_file()}


class _HubState:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.hits: list[dict[str, Any]] = []
        self.weight_delay_s = 0.0
        self.lock = threading.Lock()


HUB = _HubState()


class _HubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _lock_state(self) -> dict[str, bool]:
        return {
            "local_model_lock": text_watermark._LOCAL_MODEL_LOCK.locked(),
            "sidecar_model_lock": synthid_text_server._MODEL_LOCK.locked(),
        }

    def _send(self, status: int, body: bytes, headers: dict[str, str], head: bool) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head and body:
            self.wfile.write(body)

    def _serve(self, head: bool) -> None:
        path = self.path.split("?", 1)[0]
        parts = [part for part in path.split("/") if part]
        if parts[:1] == ["api"]:
            # huggingface_hub 1.27 fetches `{endpoint}/api/agent-harnesses` and
            # treats the JSON object as a registry (`_detect_agent.py`). A list
            # is cached and then crashes the next download before any file GET.
            if parts == ["api", "agent-harnesses"]:
                body = b'{"standardEnvVars": [], "harnesses": {}}'
            else:
                body = b"[]"
            self._send(200, body, {"Content-Type": "application/json", "X-Repo-Commit": COMMIT}, head)
            return
        if "resolve" not in parts:
            self._send(
                404,
                b"",
                {"X-Error-Code": "RepoNotFound", "X-Repo-Commit": COMMIT},
                head,
            )
            return
        index = parts.index("resolve")
        repo = "/".join(parts[:index])
        filename = "/".join(parts[index + 2 :])
        with HUB.lock:
            HUB.hits.append(
                {
                    "method": self.command,
                    "repo": repo,
                    "filename": filename,
                    **self._lock_state(),
                }
            )
        if filename == "model.safetensors" and self.command == "GET" and HUB.weight_delay_s:
            time.sleep(HUB.weight_delay_s)
        data = HUB.files.get(filename)
        if data is None:
            self._send(
                404,
                b"",
                {"X-Error-Code": "EntryNotFound", "X-Repo-Commit": COMMIT},
                head,
            )
            return
        etag = hashlib.sha256(data).hexdigest()
        # HEAD must advertise the real size. huggingface_hub refuses a file
        # whose Content-Length it could not read from the metadata response.
        self._send(
            200,
            data,
            {
                "ETag": f'"{etag}"',
                "X-Repo-Commit": COMMIT,
                "Content-Type": "application/octet-stream",
            },
            head,
        )

    def do_HEAD(self) -> None:
        self._serve(True)

    def do_GET(self) -> None:
        self._serve(False)


class _LoadSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._guard = threading.Lock()

    def install(self, monkeypatch: pytest.MonkeyPatch, transformers: Any) -> None:
        spy = self

        def wrap(kind: str, original: Any):
            def patched(model_id: str, *args: Any, **kwargs: Any) -> Any:
                with spy._guard:
                    spy.calls.append(
                        {
                            "kind": kind,
                            "model": model_id,
                            "local_files_only": kwargs.get("local_files_only", False),
                            "local_model_lock": text_watermark._LOCAL_MODEL_LOCK.locked(),
                            "sidecar_model_lock": synthid_text_server._MODEL_LOCK.locked(),
                        }
                    )
                return original(model_id, *args, **kwargs)

            return patched

        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            wrap("tokenizer", transformers.AutoTokenizer.from_pretrained),
        )
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM,
            "from_pretrained",
            wrap("model", transformers.AutoModelForCausalLM.from_pretrained),
        )

    def reset(self) -> None:
        with self._guard:
            self.calls.clear()


SPY = _LoadSpy()


def _install_markllm_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import stand-ins for the two MarkLLM symbols ``_load_algorithm`` imports.

    ``TransformersConfig`` and ``AutoWatermark.load`` match THU-BPM/MarkLLM
    ``c45ddc40f7b761beabe55a1b8dc4690e531d1c6d`` (``utils/transformers_config.py``,
    ``watermark/auto_watermark.py``): the config object only stores the already
    constructed model, and ``load`` runs after both ``from_pretrained`` calls.
    Neither symbol chooses the repository id.
    """
    utils_pkg = types.ModuleType("utils")
    utils_tc = types.ModuleType("utils.transformers_config")

    class TransformersConfig:
        def __init__(
            self,
            model: Any,
            tokenizer: Any,
            vocab_size: int | None = None,
            device: str = "cuda",
            *args: Any,
            **kwargs: Any,
        ) -> None:
            self.device = device
            self.model = model
            self.tokenizer = tokenizer
            self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size
            self.gen_kwargs: dict[str, Any] = {}
            self.gen_kwargs.update(kwargs)

    utils_tc.TransformersConfig = TransformersConfig
    utils_pkg.transformers_config = utils_tc

    watermark_pkg = types.ModuleType("watermark")
    auto_wm = types.ModuleType("watermark.auto_watermark")

    class AutoWatermark:
        @staticmethod
        def load(
            algorithm_name: str,
            algorithm_config: str | None = None,
            transformers_config: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            class _Watermark:
                def __init__(self) -> None:
                    self.config = transformers_config

                def generate_watermarked_text(self, text: str) -> str:
                    return f"{text} [wm]"

            return _Watermark()

    auto_wm.AutoWatermark = AutoWatermark
    watermark_pkg.auto_watermark = auto_wm
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.transformers_config", utils_tc)
    monkeypatch.setitem(sys.modules, "watermark", watermark_pkg)
    monkeypatch.setitem(sys.modules, "watermark.auto_watermark", auto_wm)


def _start_http(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, HTTPConnection]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    conn = HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
    httpd._stop_thread = thread  # type: ignore[attr-defined]
    return httpd, conn


def _stop_http(httpd: ThreadingHTTPServer, conn: HTTPConnection) -> None:
    conn.close()
    httpd.shutdown()
    httpd.server_close()
    thread = getattr(httpd, "_stop_thread", None)
    if thread is not None:
        thread.join(timeout=5)


def _post(
    conn: HTTPConnection,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    merged = {"Content-Type": "application/json"}
    if headers:
        merged.update(headers)
    conn.request("POST", path, body=json.dumps(payload).encode("utf-8"), headers=merged)
    response = conn.getresponse()
    raw = response.read()
    body = json.loads(raw) if raw else {}
    return response.status, body


def _prepare_upstream(tmp_path: Path) -> Path:
    upstream = tmp_path / "markllm"
    config = upstream / "config"
    config.mkdir(parents=True)
    (config / "SynthID.json").write_text('{"algorithm_name": "SynthID"}', encoding="utf-8")
    return upstream


def _forbidden_cache_dirs(cache_dir: Path, forbidden: set[str]) -> list[Path]:
    found: list[Path] = []
    if not cache_dir.exists():
        return found
    for repo in forbidden:
        folder_name = "models--" + repo.replace("/", "--")
        found.extend(path for path in cache_dir.rglob(folder_name) if path.is_dir())
    return found


def _assert_forbidden_models_not_loaded(cache_dir: Path, forbidden: set[str]) -> None:
    """Fail if a forbidden id was passed to from_pretrained or written into the Hub cache.

    Recording the call is not enough on its own: a download that finishes before
    the handler returns an error is still a failure, and so is a load that
    happens while either process-wide model lock is held.
    """
    forbidden_calls = [call for call in SPY.calls if str(call["model"]) in forbidden]
    forbidden_hits = [hit for hit in HUB.hits if hit["repo"] in forbidden]
    cache_dirs = _forbidden_cache_dirs(cache_dir, forbidden)
    details = {
        "from_pretrained": forbidden_calls,
        "hub_requests": [
            {
                "method": hit["method"],
                "repo": hit["repo"],
                "filename": hit["filename"],
                "local_model_lock": hit["local_model_lock"],
                "sidecar_model_lock": hit["sidecar_model_lock"],
            }
            for hit in forbidden_hits
        ],
        "cache_dirs": [str(path) for path in cache_dirs],
    }
    assert not forbidden_calls and not forbidden_hits and not cache_dirs, (
        "a caller-supplied model id reached the Hugging Face loader or its cache: "
        + json.dumps(details, default=str)
    )


@pytest.fixture(scope="module")
def ml_stack(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    stack = _require_ml_stack()
    repo = tmp_path_factory.mktemp("tiny-gpt2")
    HUB.files = _build_tiny_repo(stack, repo)
    assert "model.safetensors" in HUB.files
    assert len(HUB.files["model.safetensors"]) > 0
    return stack


@pytest.fixture
def attack_env(ml_stack: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache_dir = tmp_path / "hf-cache"
    cache_dir.mkdir()
    hub_httpd, _hub_conn = _start_http(_HubHandler)
    endpoint = f"http://127.0.0.1:{hub_httpd.server_address[1]}"
    constants = ml_stack["hf_constants"]
    monkeypatch.setattr(constants, "ENDPOINT", endpoint)
    monkeypatch.setattr(
        constants,
        "HUGGINGFACE_CO_URL_TEMPLATE",
        endpoint + "/{repo_id}/resolve/{revision}/{filename}",
    )
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache_dir))
    monkeypatch.setattr(constants, "HF_HOME", str(cache_dir))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    monkeypatch.setattr(constants, "AGENT_HARNESSES_PATH", str(cache_dir / ".agent_harnesses.json"))
    from huggingface_hub.utils import _detect_agent

    monkeypatch.setattr(_detect_agent, "_registry", None)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("WATERMARKS_ALLOWED_TEXT_MODELS", raising=False)
    monkeypatch.delenv("WATERMARKS_SYNTHID_TEXT_URL", raising=False)
    monkeypatch.delenv("MARKLLM_DIR", raising=False)
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(synthid_text_server, "API_KEY", "")
    upstream = _prepare_upstream(tmp_path)
    monkeypatch.setattr(synthid_text_server, "MARKLLM_DIR", str(upstream))
    _install_markllm_stubs(monkeypatch)
    SPY.install(monkeypatch, ml_stack["transformers"])
    HUB.hits.clear()
    HUB.weight_delay_s = 0.0
    SPY.reset()
    text_watermark._LOCAL_MODEL_CACHE.clear()
    synthid_text_server._MODEL_CACHE.clear()

    core_httpd, core = _start_http(server.Handler)
    side_httpd, side = _start_http(synthid_text_server.Handler)
    try:
        yield {
            "cache_dir": cache_dir,
            "upstream": upstream,
            "core": core,
            "sidecar": side,
            "sidecar_url": f"http://127.0.0.1:{side_httpd.server_address[1]}",
        }
    finally:
        _stop_http(core_httpd, core)
        _stop_http(side_httpd, side)
        _stop_http(hub_httpd, _hub_conn)


def _watermark_body(model: str, *, offline: bool | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {"model": model}
    if offline is not None:
        options["offline"] = offline
    return {"text": "hello", "options": options}


def test_core_local_watermark_does_not_download_caller_model(attack_env, monkeypatch):
    forbidden = "attacker-org/giant-weights-a"
    monkeypatch.setenv("MARKLLM_DIR", str(attack_env["upstream"]))
    status, body = _post(attack_env["core"], "/watermark", _watermark_body(forbidden))
    _assert_forbidden_models_not_loaded(attack_env["cache_dir"], {forbidden})
    assert status != 200 or body.get("report", {}).get("model") != forbidden


def test_core_local_batch_does_not_download_each_caller_model(attack_env, monkeypatch):
    models = [
        "attacker-org/giant-weights-b",
        "attacker-org/giant-weights-c",
        "attacker-org/giant-weights-d",
    ]
    monkeypatch.setenv("MARKLLM_DIR", str(attack_env["upstream"]))
    status, body = _post(
        attack_env["core"],
        "/watermark/batch",
        {"files": [{"name": f"{i}.txt", "text": "hello", "options": {"model": model}} for i, model in enumerate(models)]},
    )
    _assert_forbidden_models_not_loaded(attack_env["cache_dir"], set(models))
    assert status != 200 or not any(
        (item.get("report") or {}).get("model") in models for item in body.get("results", [])
    )


def test_padded_and_local_path_models_are_not_loaded(attack_env, monkeypatch, tmp_path):
    local_repo = tmp_path / "local-evil-model"
    local_repo.mkdir()
    (local_repo / "config.json").write_text('{"model_type": "gpt2"}', encoding="utf-8")
    padded = "  attacker-org/giant-weights-pad  "
    stripped = padded.strip()
    monkeypatch.setenv("MARKLLM_DIR", str(attack_env["upstream"]))
    _post(attack_env["core"], "/watermark", _watermark_body(padded))
    _post(attack_env["core"], "/watermark", _watermark_body(str(local_repo)))
    _assert_forbidden_models_not_loaded(
        attack_env["cache_dir"],
        {padded, stripped, str(local_repo)},
    )


def test_explicit_offline_false_still_does_not_fetch_caller_model(attack_env, monkeypatch):
    forbidden = "attacker-org/giant-weights-online"
    monkeypatch.setenv("MARKLLM_DIR", str(attack_env["upstream"]))
    _post(attack_env["core"], "/watermark", _watermark_body(forbidden, offline=False))
    _assert_forbidden_models_not_loaded(attack_env["cache_dir"], {forbidden})


def test_bearer_token_does_not_authorize_a_caller_model(attack_env, monkeypatch):
    forbidden = "attacker-org/giant-weights-authed"
    monkeypatch.setattr(server, "API_KEY", PLACEHOLDER_API_KEY)
    monkeypatch.setenv("MARKLLM_DIR", str(attack_env["upstream"]))
    status, _body = _post(
        attack_env["core"],
        "/watermark",
        _watermark_body(forbidden),
        headers={"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"},
    )
    assert status != 401
    _assert_forbidden_models_not_loaded(attack_env["cache_dir"], {forbidden})


def test_core_forwards_caller_model_into_sidecar_loader(attack_env, monkeypatch):
    forbidden = "attacker-org/giant-weights-sidecar"
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", attack_env["sidecar_url"])
    _post(attack_env["core"], "/watermark", _watermark_body(forbidden))
    _assert_forbidden_models_not_loaded(attack_env["cache_dir"], {forbidden})


def test_sidecar_batch_downloads_every_caller_model(attack_env):
    models = ["attacker-org/giant-weights-side-b", "attacker-org/giant-weights-side-c"]
    _post(
        attack_env["sidecar"],
        "/watermark/batch",
        {"files": [{"text": "hello", "options": {"model": model}} for model in models]},
    )
    _assert_forbidden_models_not_loaded(attack_env["cache_dir"], set(models))


def test_core_timeout_does_not_leave_a_sidecar_download(attack_env, monkeypatch):
    """The core urllib timeout is clamped to at least one second and does not cancel the sidecar.

    ``resolve_timeout`` floors the deadline at 1s. The weight GET is held longer
    than that, then this test waits until that hold has finished. A download
    that lands after the core client has already given up still fails the test.
    """
    forbidden = "attacker-org/giant-weights-slow"
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_URL", attack_env["sidecar_url"])
    monkeypatch.setenv("WATERMARKS_SYNTHID_TEXT_TIMEOUT", "1")
    HUB.weight_delay_s = 2.5
    _post(attack_env["core"], "/watermark", _watermark_body(forbidden))
    time.sleep(HUB.weight_delay_s + 0.5)
    _assert_forbidden_models_not_loaded(attack_env["cache_dir"], {forbidden})
