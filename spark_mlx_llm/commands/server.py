"""Serve Spark models, selecting DSpark when a separate or merged draft is present."""

import argparse
import json
import sys
from pathlib import Path

from mlx_lm import server as mlx_server

from ..generation import GenerationConfig
from ..loader import load_without_model_cache
from ..model_source import (
    add_model_cache_arguments,
    model_cache_enabled,
    prepare_local_cli_model,
)
from ..registration import register_model


def build_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        usage=argparse.SUPPRESS,
        description="Shared Spark configuration (generate, chat and server):",
    )
    GenerationConfig.add_arguments(parser, include_kv=False)
    parser.add_argument(
        "--model", help="Local source/MLX directory, target GGUF file, or HF repo"
    )
    parser.add_argument(
        "--draft-model",
        help="Enable DSpark with a matching draft GGUF or Safetensors directory; "
        "add --help for DSpark options",
    )
    parser.add_argument(
        "--served-model-name",
        help="Model name exposed to HTTP clients "
        "(ordinary default: default_model; DSpark default: dspark)",
    )
    add_model_cache_arguments(parser)
    return parser


def _uses_dspark(args):
    if args.draft_model is not None:
        return True
    if args.model is None:
        return False
    path = Path(args.model).expanduser()
    if path.is_dir() and (
        (path / "model.safetensors.index.json").is_file()
        or any(path.glob("model*.safetensors"))
    ):
        from ..dspark.checkpoint import inspect_checkpoint

        return bool(inspect_checkpoint(path).draft_tensors)
    return False


def _named_model_provider(factory, name):
    def create(args):
        provider = factory(args)
        provider.cli_args.served_model_name = name
        provider._model_map[name] = args.model
        return provider

    return create


def _handle_named_models_request(handler):
    name = handler.response_generator.cli_args.served_model_name
    response = {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": handler.created,
                "owned_by": "local",
            }
        ],
    }
    handler._set_completion_headers(200)
    handler.end_headers()
    handler.wfile.write(json.dumps(response).encode())
    handler.wfile.flush()


def upstream_arguments(config):
    arguments = []
    for flag, value in (
        ("--max-tokens", config.max_tokens),
        ("--temp", config.temperature),
        ("--top-p", config.top_p),
        ("--top-k", config.top_k),
        ("--min-p", config.min_p),
        ("--chat-template-args", json.dumps(config.template_kwargs())),
    ):
        arguments.extend((flag, str(value)))
    return arguments


def _uncached_sharded_load(*args, **kwargs):
    raise ValueError("--no-model-cache requires single-process loading")


def main():
    parser = build_parser()
    args, remaining = parser.parse_known_args()
    help_requested = "--help" in remaining or "-h" in remaining
    try:
        dspark = _uses_dspark(args)
    except (OSError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    if dspark:
        from . import dspark_server

        return dspark_server.main(sys.argv[1:], prog=sys.argv[0])
    if args.served_model_name is not None and not help_requested:
        if not args.served_model_name.strip():
            parser.error("--served-model-name must not be empty")
        if args.model is None:
            parser.error("--served-model-name requires --model")
    inference = GenerationConfig.from_args(parser, args, include_kv=False)
    register_model()
    args.model_cache = model_cache_enabled(args.model, args.model_cache)
    if args.model is not None:
        remaining.extend(
            (
                "--model",
                args.model
                if help_requested or not args.model_cache
                else prepare_local_cli_model(
                    args.model, cache_dir=args.model_cache_dir
                ),
            )
        )
    original_argv = sys.argv
    original_load, original_sharded_load = mlx_server.load, mlx_server.sharded_load
    original_provider = mlx_server.ModelProvider
    original_models_request = mlx_server.APIHandler.handle_models_request
    try:
        if args.served_model_name is not None:
            mlx_server.ModelProvider = _named_model_provider(
                original_provider, args.served_model_name
            )
            mlx_server.APIHandler.handle_models_request = _handle_named_models_request
        if not args.model_cache and not help_requested:
            mlx_server.load = load_without_model_cache
            mlx_server.sharded_load = _uncached_sharded_load
        sys.argv = [original_argv[0], *upstream_arguments(inference), *remaining]
        return mlx_server.main()
    except SystemExit as error:
        if help_requested and error.code == 0:
            print()
            parser.print_help()
        raise
    finally:
        mlx_server.load, mlx_server.sharded_load = original_load, original_sharded_load
        mlx_server.ModelProvider = original_provider
        mlx_server.APIHandler.handle_models_request = original_models_request
        sys.argv = original_argv


if __name__ == "__main__":
    main()
