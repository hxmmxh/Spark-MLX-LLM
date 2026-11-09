"""Single-request DSpark HTTP service with chat completions and SSE streaming."""

import argparse
import json
import logging
import math
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import chain

from ..dspark.config import DEFAULT_MAX_DRAFT_TOKENS
from ..model_source import add_model_cache_arguments, model_cache_enabled

logger = logging.getLogger(__name__)


def build_parser(*, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--model", required=True, help="BF16/INT4/MLX directory or Spark target GGUF"
    )
    parser.add_argument(
        "--draft-model", help="Standalone Spark DSpark GGUF or Safetensors directory"
    )
    parser.add_argument("--served-model-name", default="dspark")
    add_model_cache_arguments(parser)
    parser.add_argument(
        "--draft-quantization",
        choices=("checkpoint", "bf16", "int8", "int4"),
        default=None,
        help="Default: int8 for floating drafts, preserve stored quantization; checkpoint keeps floating weights",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-context-length", type=int, default=262144)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", "--temp", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--draft-temperature", type=float, default=1.0)
    parser.add_argument("--draft-top-p", type=float, default=1.0)
    parser.add_argument("--draft-top-k", type=int, default=0)
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        help=(
            "Draft candidates per round "
            f"(default: up to {DEFAULT_MAX_DRAFT_TOKENS}; 0 disables drafting)"
        ),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--reasoning-effort", choices=("low", "high"), default="low")
    parser.add_argument(
        "--draft-readout", choices=("checkpoint", "embedding"), default="checkpoint"
    )
    parser.add_argument(
        "--w4-qmv-backend",
        help="Default: checkpoint setting, otherwise dSpark multirow",
    )
    parser.add_argument("--w8-qmv-backend", default="native")
    return parser


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sampling(body, defaults, prefix=""):
    values = {}
    for name in ("temperature", "top_p", "top_k"):
        key = prefix + name
        value = body.get(key, getattr(defaults, key))
        if name == "top_k":
            _integer(value, key)
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (name == "top_p" and not 0 < value <= 1)
        ):
            raise ValueError(f"Invalid {key}")
        values[name] = value
    return values


def prepare_request(pair, defaults, body):
    """Validate the context budget before allocating inference caches."""
    if not isinstance(body, dict):
        raise TypeError("Request must be a JSON object")
    if body.get("model", defaults.served_model_name) not in (
        defaults.served_model_name,
        "default_model",
    ):
        raise ValueError(f"Unknown model; use {defaults.served_model_name!r}")
    messages = body.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            for message in messages
        )
    ):
        raise ValueError("messages must contain system/user/assistant text messages")
    if body.get("tools") or body.get("n", 1) != 1 or body.get("stop"):
        raise ValueError(
            "This endpoint supports one text completion without tools or stop overrides"
        )
    if type(body.get("stream", False)) is not bool:
        raise ValueError("stream must be a boolean")
    if not isinstance(body.get("stream_options", {}), dict):
        raise TypeError("stream_options must be an object")
    effort = body.get("reasoning_effort", defaults.reasoning_effort)
    if effort not in ("low", "high"):
        raise ValueError("reasoning_effort must be low or high")
    seed = body.get("seed")
    if seed is not None:
        _integer(seed, "seed")
    target = _sampling(body, defaults)
    draft = _sampling(body, defaults, "draft_")
    ids = list(
        pair.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=defaults.thinking,
            reasoning_effort=effort,
        )
    )
    remaining = defaults.max_context_length - len(ids)
    if not ids or remaining <= 0:
        raise ValueError(
            f"Prompt must leave space in the {defaults.max_context_length}-token context"
        )
    maximum = body.get("max_completion_tokens", body.get("max_tokens"))
    maximum = min(defaults.max_tokens, remaining) if maximum is None else maximum
    _integer(maximum, "max_tokens", minimum=1)
    if maximum > remaining:
        raise ValueError(
            f"{len(ids)} prompt + {maximum} output tokens exceeds "
            f"max_context_length={defaults.max_context_length}"
        )
    return ids, maximum, target, draft, seed


def make_handler(pair, defaults, stream_generate=None):
    from ..dspark.metrics import GenerationResult
    from ..dspark.tokenization import eos_token_ids

    if stream_generate is None:
        from ..dspark.generation import stream_generate
    from ..dspark.sampling import SamplingOptions

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._json(
                    200,
                    {
                        "status": "ok",
                        "max_context_length": defaults.max_context_length,
                        "draft_quantization": (
                            getattr(pair, "draft_quantization", None) or {}
                        ).get("policy", defaults.draft_quantization),
                    },
                )
            elif self.path == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": defaults.served_model_name,
                                "object": "model",
                                "created": 0,
                                "owned_by": "local",
                                "max_context_length": defaults.max_context_length,
                            }
                        ],
                    },
                )
            else:
                self._json(404, {"error": {"message": "Unknown endpoint"}})

        def do_POST(self):
            streaming = False
            events = None
            try:
                if self.path != "/v1/chat/completions":
                    self._json(404, {"error": {"message": "Unknown endpoint"}})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024**2:
                    raise ValueError("Expected a JSON request body of at most 32 MiB")
                body = json.loads(self.rfile.read(length))
                ids, maximum, target, draft, seed = prepare_request(
                    pair, defaults, body
                )
                ident, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
                base = {
                    "id": ident,
                    "created": created,
                    "model": defaults.served_model_name,
                }
                events = stream_generate(
                    pair,
                    ids,
                    max_new_tokens=maximum,
                    mode="dspark",
                    sampling=SamplingOptions(**target),
                    draft_sampling=SamplingOptions(**draft),
                    seed=seed,
                    apply_chat_template=False,
                    max_draft_tokens=defaults.max_draft_tokens,
                    confidence_threshold=defaults.confidence_threshold,
                    prefill_chunk_size=defaults.prefill_chunk_size,
                    asynchronous_draft=True,
                    async_context=True,
                )
                # Resolve prefill errors before opening the SSE response.
                first = next(events)
                if body.get("stream", False):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    streaming = True

                def send(value):
                    payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()

                def chunk(delta, reason=None):
                    return {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": reason,
                            }
                        ],
                    }

                detokenizer = pair.tokenizer.detokenizer
                detokenizer.reset()
                eos = eos_token_ids(pair)
                if streaming:
                    send(chunk({"role": "assistant", "content": ""}))
                result = None
                for event in chain((first,), events):
                    if isinstance(event, GenerationResult):
                        result = event
                    elif streaming and event.token_id not in eos:
                        detokenizer.add_token(event.token_id)
                        if detokenizer.last_segment:
                            send(chunk({"content": detokenizer.last_segment}))
                if result is None:
                    raise RuntimeError("DSpark returned no generation result")
                usage = {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": len(result.token_ids),
                    "total_tokens": result.prompt_tokens + len(result.token_ids),
                }
                if streaming:
                    detokenizer.finalize()
                    if detokenizer.last_segment:
                        send(chunk({"content": detokenizer.last_segment}))
                    send(chunk({}, result.finish_reason))
                    if body.get("stream_options", {}).get("include_usage"):
                        send(
                            {
                                **base,
                                "object": "chat.completion.chunk",
                                "choices": [],
                                "usage": usage,
                            }
                        )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self._json(
                        200,
                        {
                            **base,
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": result.text,
                                    },
                                    "finish_reason": result.finish_reason,
                                }
                            ],
                            "usage": usage,
                        },
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                logger.exception("DSpark request failed")
                value = {"error": {"message": str(error)}}
                if streaming:
                    try:
                        send(value)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self._json(
                        400 if isinstance(error, (ValueError, TypeError)) else 500,
                        value,
                    )
            finally:
                if events is not None:
                    events.close()

    return Handler


def main(argv=None, *, prog=None):
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    if not args.served_model_name.strip():
        parser.error("--served-model-name must not be empty")
    for name in ("max_context_length", "max_tokens", "prefill_chunk_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in [0, 1]")
    _sampling({}, args)
    _sampling({}, args, "draft_")
    from ..dspark.loader import load_pair

    pair = load_pair(
        args.model,
        model_cache_dir=args.model_cache_dir,
        model_cache=model_cache_enabled(args.model, args.model_cache),
        draft_path=args.draft_model,
        draft_quantization=args.draft_quantization,
        draft_readout=args.draft_readout,
        model_config={
            name: value
            for name, value in {
                "w4_qmv_backend": args.w4_qmv_backend,
                "w8_qmv_backend": args.w8_qmv_backend,
            }.items()
            if value is not None
        },
    )
    if args.max_context_length > pair.config.get(
        "max_position_embeddings", args.max_context_length
    ):
        parser.error("--max-context-length exceeds the model's max_position_embeddings")
    if (
        args.max_draft_tokens is not None
        and not 0 <= args.max_draft_tokens <= pair.drafter.config.max_draft_tokens
    ):
        parser.error("--max-draft-tokens exceeds the trained draft slots")
    if args.confidence_threshold and pair.drafter.confidence_head is None:
        parser.error("Confidence scheduling requires a confidence head")
    logging.basicConfig(level=logging.INFO)
    # Keep all MLX work and the shared detokenizer on one serving thread.
    with HTTPServer((args.host, args.port), make_handler(pair, args)) as server:
        logger.info(
            "DSpark ready at http://%s:%s/v1 (context=%s)",
            args.host,
            args.port,
            args.max_context_length,
        )
        logger.info("Draft precision: %s", pair.draft_quantization["policy"])
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
