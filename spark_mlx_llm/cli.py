"""One command and configuration contract for all Spark inference."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys

from .config import GenerationConfig
from .requests import resolve_request


def _model_arguments(parser, *, required=True):
    from .model_source import add_model_cache_arguments

    parser.add_argument("--model", required=required)
    parser.add_argument("--draft-model")
    parser.add_argument("--use-draft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draft-quantization", choices=("checkpoint", "bf16", "int8", "int4"))
    parser.add_argument("--draft-readout", choices=("checkpoint", "embedding"), default="checkpoint")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--w4-qmv-backend")
    for name in ("fused_quantized_sdpa", "fused_quantized_sdpa_prefill", "tuned_bf16_sdpa", "verification_bf16_sdpa"):
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    add_model_cache_arguments(parser)


def build_parser():
    parser = argparse.ArgumentParser(prog="spark-mlx", description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "chat", "server", "benchmark"):
        entry = commands.add_parser(command, allow_abbrev=False)
        _model_arguments(entry, required=command != "server")
        GenerationConfig.add_arguments(entry)
        entry.add_argument("--print-config", action="store_true", help="Validate and print requested settings without loading a model")
        if command in ("generate", "benchmark"):
            source = entry.add_mutually_exclusive_group()
            source.add_argument("--prompt")
            source.add_argument("--input-file", type=Path)
            source.add_argument("--request-file", type=Path, help="Replay the same JSON body accepted by HTTP")
            entry.add_argument("--json", action="store_true")
            entry.add_argument("--stats", action="store_true")
            if command == "benchmark":
                entry.add_argument("--trials", type=int, default=3)
                entry.add_argument("--warmup-tokens", type=int, default=128)
        if command in ("generate", "chat"):
            entry.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
        if command == "chat":
            entry.add_argument("--system-prompt")
        if command == "server":
            entry.add_argument("--host", default="127.0.0.1")
            entry.add_argument("--port", type=int, default=8080)
            entry.add_argument("--served-model-name")
            entry.add_argument("--allowed-origins", type=lambda value: value.split(","), default=["*"],
                               help="Comma-separated CORS origins (default: *)")
    commands.add_parser("convert", add_help=False, help="Convert a checkpoint; use convert --help")
    doctor = commands.add_parser("doctor")
    _model_arguments(doctor, required=False)
    GenerationConfig.add_arguments(doctor)
    doctor.add_argument("--gpu", action="store_true")
    doctor.add_argument("--lock-timeout", type=float, default=0)
    doctor.add_argument("--json", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--model", required=True)
    return parser


def _load(args):
    import mlx.core as mx
    from .loader import load

    if args.device != "auto":
        mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    return load(
        args.model, draft_model=args.draft_model, use_draft=args.use_draft,
        draft_quantization=args.draft_quantization, draft_readout=args.draft_readout,
        dtype=args.dtype, model_cache=args.model_cache, model_cache_dir=args.model_cache_dir,
        w4_qmv_backend=args.w4_qmv_backend,
        model_config={name: getattr(args, name) for name in
                      ("fused_quantized_sdpa", "fused_quantized_sdpa_prefill", "tuned_bf16_sdpa", "verification_bf16_sdpa")
                      if getattr(args, name) is not None},
    )


def _request(args, config):
    if args.request_file is not None:
        return resolve_request(json.loads(args.request_file.read_text()), config)
    prompt = args.input_file.read_text() if args.input_file is not None else args.prompt or "你好"
    return prompt, None, config, getattr(args, "stream", False)


def _emit(model, tokenizer, prompt, *, config, tools=None, stream=False, as_json=False, stats=False):
    from .generation import generate

    value = generate(model, tokenizer, prompt, config=config, tools=tools, stream=stream)
    if not stream:
        print(json.dumps(value.to_dict(), ensure_ascii=False) if as_json else value.text)
        result = value
    else:
        result = None
        with closing(value):
            for chunk in value:
                if as_json:
                    print(json.dumps(chunk.to_dict(), ensure_ascii=False), flush=True)
                elif chunk.text:
                    print(chunk.text, end="", flush=True)
                if chunk.result is not None:
                    result = chunk.result
        if not as_json:
            print()
        if result is None:
            raise RuntimeError("Generation ended without a result")
    if stats:
        print(json.dumps(result.metrics(), ensure_ascii=False), file=sys.stderr)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "convert":
        from .commands.convert import main as convert

        return convert(argv[1:], prog="spark-mlx convert")
    parser = build_parser()
    if argv and argv[0] == "server" and any(flag in argv for flag in ("--help", "-h")):
        from .server import print_upstream_help
        print_upstream_help()
    if argv and argv[0] == "server":
        args, upstream_options = parser.parse_known_args(argv)
    else:
        args = parser.parse_args(argv)
    if args.command == "inspect":
        from .dspark.checkpoint import inspect_checkpoint

        print(json.dumps(inspect_checkpoint(args.model).summary(), indent=2))
        return 0
    config = GenerationConfig.from_args(parser, args)
    if args.command == "doctor":
        from .diagnostics import doctor
        return doctor(parser, args, config)
    try:
        if args.command in ("generate", "benchmark"):
            prompt, tools, config, stream = _request(args, config)
            if args.command == "benchmark" and stream:
                raise ValueError("benchmark measures non-streaming generation; request stream must be false")
            if args.command == "benchmark" and (args.trials <= 0 or args.warmup_tokens < 0):
                raise ValueError("trials must be positive and warmup_tokens nonnegative")
        if getattr(args, "print_config", False):
            print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "server":
            from .server import serve_cli
            return serve_cli(args, config, upstream_options)
        model, tokenizer = _load(args)
        if args.command == "generate":
            _emit(model, tokenizer, prompt, config=config, tools=tools, stream=stream, as_json=args.json, stats=args.stats)
        elif args.command == "benchmark":
            from .benchmark import benchmark

            print(json.dumps(benchmark(model, tokenizer, prompt, config=config, tools=tools,
                                       trials=args.trials, warmup_tokens=args.warmup_tokens), ensure_ascii=False, indent=2))
        elif args.command == "chat":
            messages = [{"role": "system", "content": args.system_prompt}] if args.system_prompt else []
            initial = list(messages)
            while True:
                try:
                    query = input(">> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if query == "/quit":
                    break
                if query == "/reset":
                    messages = list(initial)
                    continue
                if not query.strip():
                    continue
                pending = [*messages, {"role": "user", "content": query}]
                result = _emit(model, tokenizer, pending, config=config, stream=args.stream)
                messages = [*pending, {"role": "assistant", "content": result.text}]
    except KeyboardInterrupt:
        return 130
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    return 0
