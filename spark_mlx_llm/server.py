"""The MLX-LM 0.31.3 HTTP service, with Spark loading/generation adapters."""

import json
import logging
import sys
import uuid
from dataclasses import fields
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from threading import Lock
from unittest.mock import patch

from mlx_lm import server as upstream
from mlx_lm.models.cache import LRUPromptCache

from ._server_backend import ModelProvider, ResponseGenerator
from .config import GenerationConfig

logger = logging.getLogger(__name__)
_CONFIG_FIELDS = {item.name for item in fields(GenerationConfig)}
# These fields belong to the upstream request parser, including its validation,
# aliases, defaults and chat-template semantics.
_UPSTREAM_CONFIG = {
    "max_tokens", "temperature", "top_p", "top_k", "min_p", "seed",
    "chat_template_kwargs", "prefill_step_size",
}


def print_upstream_help():
    """Show upstream options alongside the additive Spark CLI options."""
    with patch.object(sys, "argv", ["spark-mlx server", "--help"]):
        try:
            upstream.main()
        except SystemExit as error:
            if error.code:
                raise


class APIHandler(upstream.APIHandler):
    """Keep upstream methods, parameter parsing and response formatting intact."""

    spark_config = None
    served_model_name = None

    def _set_completion_headers(self, status_code=200):
        request = getattr(self, "_spark_request", None)
        if status_code == 404 and getattr(request, "spark_context_error", False):
            status_code = 400
        return super()._set_completion_headers(status_code)

    def end_headers(self):
        super().end_headers()
        self._response_started = True

    def _set_stream_headers(self, status_code=200):
        self._stream_started = True
        return super()._set_stream_headers(status_code)

    def do_POST(self):
        try:
            super().do_POST()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, TypeError, AssertionError) as error:
            if getattr(self, "_response_started", False):
                if getattr(self, "_stream_started", False):
                    self.wfile.write(("data: " + json.dumps({"error": str(error)})
                                      + "\n\ndata: [DONE]\n\n").encode())
                    self.wfile.flush()
                self.close_connection = True
                return
            # Non-streaming upstream handlers may have buffered a 200 status
            # before generation failed. Replace those unsent headers together.
            self._headers_buffer = []
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(error)}).encode())

    def validate_model_parameters(self):
        GenerationConfig.reject_http_kv_options(self.body)
        if "max_draft_tokens" in self.body:
            self.num_draft_tokens = self.body["max_draft_tokens"]
        super().validate_model_parameters()
        # Only Spark additions are validated here. The old HTTP fields continue
        # through MLX-LM's parser rather than a restrictive allowlist.
        overrides = {
            name: value for name, value in self.body.items()
            if name in _CONFIG_FIELDS - _UPSTREAM_CONFIG
        }
        base = self.spark_config or GenerationConfig.http_defaults()
        base.overlay(overrides)
        self._spark_overrides = overrides
        thinking = {
            name: overrides[name] for name in ("enable_thinking", "reasoning_effort")
            if name in overrides
        }
        if thinking:
            # Upstream merges these request arguments over the service defaults.
            # Keep its nested form, with explicit top-level Spark values winning.
            self.chat_template_kwargs = {
                **dict(self.chat_template_kwargs or {}), **thinking,
            }
        if "seed" not in self.body and base.seed is not None:
            self.seed = base.seed
        if "model" not in self.body and self.served_model_name:
            self.requested_model = self.served_model_name

    def handle_completion(self, request, stop_words):
        request.spark_overrides = self._spark_overrides
        request.spark_max_tokens_explicit = (
            self.body.get("max_completion_tokens") is not None or "max_tokens" in self.body
        )
        self._spark_request = request
        return super().handle_completion(request, stop_words)

    def handle_chat_completions(self):
        if "messages" in self.body:
            return super().handle_chat_completions()
        # Retain the additional prompt-input form introduced by the Spark API.
        if "prompt" not in self.body:
            raise ValueError("Request did not contain messages or prompt")
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"
        return upstream.CompletionRequest(
            "text", self.body["prompt"], [], self.body.get("tools"), None
        )

    def handle_models_request(self):
        if not self.served_model_name:
            return super().handle_models_request()
        # This is the same named-model override used by the original wrapper.
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps({"object": "list", "data": [{
            "id": self.served_model_name, "object": "model",
            "created": self.created, "owned_by": "local",
        }]}).encode())
        self.wfile.flush()


def _namespace(model, config, allowed_origins, **options):
    if isinstance(allowed_origins, str):
        allowed_origins = allowed_origins.split(",")
    return SimpleNamespace(**{
        "model": model, "adapter_path": None, "draft_model": None,
        "pipeline": False, "trust_remote_code": False, "chat_template": "",
        "use_default_chat_template": False,
        "chat_template_args": config.template_kwargs(),
        "allowed_origins": list(allowed_origins),
        "max_tokens": config.max_tokens, "temp": config.temperature,
        "top_p": config.top_p, "top_k": config.top_k, "min_p": config.min_p,
        "num_draft_tokens": config.max_draft_tokens,
        "prefill_step_size": config.prefill_step_size,
        "prompt_cache_size": 10, "prompt_cache_bytes": None,
        "decode_concurrency": 32, "prompt_concurrency": 8,
        **options,
    })


def make_handler(
    model, tokenizer, *, config=None, served_model_name=None, allowed_origins=("*",),
    **upstream_options,
):
    """Bind the common service to an already loaded model (including a draft)."""
    config = config or GenerationConfig.http_defaults()
    config.validate_http()
    GenerationConfig.reject_http_kv_options(upstream_options)
    if served_model_name is not None and not served_model_name.strip():
        raise ValueError("served_model_name must not be empty")
    args = _namespace(
        getattr(model, "_spark_load_options", {}).get("model", "preloaded"),
        config, allowed_origins, **upstream_options,
    )
    provider = ModelProvider(args, config=config, served_model_name=served_model_name)
    provider.preloaded = (model, tokenizer)
    lock = Lock()
    generator = None

    class Handler(APIHandler):
        spark_config = config

        def __init__(self, *args, **kwargs):
            nonlocal generator
            with lock:
                if generator is None:
                    generator = ResponseGenerator(provider, LRUPromptCache(provider.cli_args.prompt_cache_size))
            super().__init__(generator, *args, system_fingerprint="spark-mlx-llm", **kwargs)

        @classmethod
        def close(cls):
            if generator is not None:
                generator.stop_and_join()

    Handler.served_model_name = served_model_name
    return Handler


def serve(
    model, tokenizer, *, config=None, host="127.0.0.1", port=8080,
    served_model_name=None, allowed_origins=("*",), **upstream_options,
):
    handler = make_handler(
        model, tokenizer, config=config, served_model_name=served_model_name,
        allowed_origins=allowed_origins, **upstream_options,
    )
    try:
        with ThreadingHTTPServer((host, port), handler) as server:
            logger.info("Ready at http://%s:%s/v1", host, port)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        handler.close()


def serve_cli(args, config, remaining=()):
    """Delegate the CLI to upstream, forwarding all its existing server options."""
    config.validate_http()
    import mlx.core as mx
    from ._target_loader import _resolve_model_path, load as load_target
    from .registration import register_model

    register_model()
    if args.served_model_name is not None:
        if not args.served_model_name.strip():
            raise ValueError("served_model_name must not be empty")
        if args.model is None:
            raise ValueError("--served-model-name requires --model")
    if args.device != "auto":
        mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    load_options = {
        "use_draft": args.use_draft, "draft_quantization": args.draft_quantization,
        "draft_readout": args.draft_readout, "dtype": args.dtype,
        "model_cache": args.model_cache, "model_cache_dir": args.model_cache_dir,
        "w4_qmv_backend": args.w4_qmv_backend,
        "model_config": {
            name: getattr(args, name) for name in
            ("fused_quantized_sdpa", "fused_quantized_sdpa_prefill", "tuned_bf16_sdpa", "verification_bf16_sdpa")
            if getattr(args, name) is not None
        },
    }
    argv = ["spark-mlx server", "--host", args.host,
            "--port", str(args.port), "--temp", str(config.temperature),
            "--top-p", str(config.top_p), "--top-k", str(config.top_k),
            "--min-p", str(config.min_p), "--max-tokens", str(config.max_tokens),
            "--num-draft-tokens", str(config.max_draft_tokens),
            "--prefill-step-size", str(config.prefill_step_size),
            "--allowed-origins", ",".join(args.allowed_origins),
            "--chat-template-args", json.dumps(config.template_kwargs())]
    if args.model is not None:
        argv.extend(("--model", args.model))
    if args.draft_model is not None:
        argv.extend(("--draft-model", args.draft_model))
    argv.extend(remaining)
    original_load = upstream.load
    original_run_http = upstream._run_http_server

    def ordinary_load(path, *, adapter_path=None, tokenizer_config=None, **kwargs):
        # Preserve the original Spark loader's defaults and in-memory INT4 path.
        source = _resolve_model_path(path, kwargs.get("revision"))
        raw_path = source / "config.json" if source.is_dir() else None
        raw = json.loads(raw_path.read_text()) if raw_path and raw_path.is_file() else {}
        if raw.get("model_type") != "spark2_5" and source.suffix.lower() != ".gguf":
            return original_load(path, adapter_path=adapter_path, tokenizer_config=tokenizer_config, **kwargs)
        from .model_source import model_cache_enabled, prepare_model_source

        if model_cache_enabled(source, args.model_cache) and not (
            args.dtype or args.w4_qmv_backend or load_options["model_config"]
        ):
            prepared = prepare_model_source(source, cache_dir=args.model_cache_dir)
            return original_load(str(prepared), adapter_path=adapter_path,
                                 tokenizer_config=tokenizer_config, **kwargs)
        model, tokenizer = load_target(
            source, tokenizer_config=tokenizer_config,
            **{key: value for key, value in load_options.items()
               if key not in ("use_draft", "draft_quantization", "draft_readout")},
        )
        if adapter_path is not None:
            from mlx_lm.utils import load_adapters
            model = load_adapters(model, adapter_path)
            model.eval()
        return model, tokenizer

    class Handler(APIHandler):
        spark_config = config
        served_model_name = args.served_model_name

    def provider(cli_args):
        return ModelProvider(
            cli_args, config=config, load_options=load_options,
            served_model_name=args.served_model_name,
        )

    def run_http(host, port, generator):
        return original_run_http(host, port, generator, handler_class=Handler)

    with patch.object(sys, "argv", argv), \
         patch.object(upstream, "load", ordinary_load), \
         patch.object(upstream, "ModelProvider", provider), \
         patch.object(upstream, "ResponseGenerator", ResponseGenerator), \
         patch.object(upstream, "_run_http_server", run_http):
        result = upstream.main()
        return 0 if result is None else result
