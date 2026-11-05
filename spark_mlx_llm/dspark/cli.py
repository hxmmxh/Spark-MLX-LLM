"""Inspect checkpoints, generate text, and run paired DSpark benchmarks.

Argument parsing and checkpoint inspection require only the standard library.
Model execution dependencies are imported only by generate/benchmark/export.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from spark_mlx_llm.model_source import add_model_cache_arguments, model_cache_enabled

from .checkpoint import inspect_checkpoint, validate_draft_shapes
from .config import DSparkConfig
from .metrics import GenerationResult
from .tokenization import eos_token_ids


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect local checkpoint metadata")
    inspect.add_argument(
        "model", help="Local target/merged directory or Spark target GGUF"
    )
    inspect.add_argument(
        "--draft-model", help="Standalone Spark DSpark GGUF or Safetensors directory"
    )
    for command in ("generate", "benchmark", "export-draft"):
        runtime = commands.add_parser(command, help=f"Run DSpark {command}")
        runtime.add_argument(
            "--model",
            required=True,
            help="BF16/INT4/MLX directory or Spark target GGUF",
        )
        add_model_cache_arguments(runtime)
        runtime.add_argument(
            "--draft-quantization",
            choices=("checkpoint", "bf16", "int8", "int4"),
            default=None,
            help="Default: int8 for floating drafts, preserve stored quantization; checkpoint keeps floating weights",
        )
        runtime.add_argument(
            "--draft-readout", choices=("checkpoint", "embedding"), default="checkpoint"
        )
        runtime.add_argument(
            "--draft-model",
            help="Standalone Spark DSpark GGUF or Safetensors directory",
        )
        if command == "export-draft":
            runtime.add_argument(
                "--output", required=True, help="New standalone MLX draft directory"
            )
            runtime.add_argument(
                "--device", choices=("auto", "cpu", "gpu"), default="auto"
            )
            continue
        runtime.add_argument("--prompt", "-p", default="你好")
        runtime.add_argument("--max-new-tokens", type=int, default=512)
        runtime.add_argument("--mode", choices=("baseline", "dspark"), default="dspark")
        runtime.add_argument("--max-draft-tokens", type=int)
        runtime.add_argument("--confidence-threshold", type=float, default=0.0)
        runtime.add_argument(
            "--greedy-verification",
            choices=("sequential", "batched"),
            default="batched",
            help="Batched verification (default); sequential is for correctness tests",
        )
        runtime.add_argument("--temperature", type=float, default=0.0)
        runtime.add_argument("--top-p", type=float, default=1.0)
        runtime.add_argument("--top-k", type=int, default=0)
        runtime.add_argument(
            "--draft-temperature",
            type=float,
            help="Draft temperature (default: inherit --temperature)",
        )
        runtime.add_argument(
            "--draft-top-p",
            type=float,
            help="Draft top-p (default: inherit --top-p)",
        )
        runtime.add_argument(
            "--draft-top-k",
            type=int,
            help="Draft top-k (default: inherit --top-k)",
        )
        runtime.add_argument("--seed", type=int, default=0)
        runtime.add_argument("--prefill-chunk-size", type=int, default=512)
        runtime.add_argument("--raw-prompt", action="store_true")
        runtime.add_argument(
            "--thinking", action=argparse.BooleanOptionalAction, default=True
        )
        runtime.add_argument(
            "--reasoning-effort", choices=("low", "high"), default="low"
        )
        runtime.add_argument("--dtype", choices=("bfloat16", "float32"))
        runtime.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
        runtime.add_argument(
            "--json", action="store_true", help="Print the result as JSON"
        )
        runtime.add_argument(
            "--stats", action="store_true", help="Print generation statistics to stderr"
        )
        runtime.add_argument("--trials", type=int, default=3)
        runtime.add_argument("--warmup-tokens", type=int, default=8)
    return parser


def _run_inspect(args):
    path = Path(args.model).expanduser().resolve()
    vocab = None
    if path.is_file() and path.suffix.lower() == ".gguf":
        from ..gguf_target import read_target_gguf

        raw, _, assets = read_target_gguf(path)
        vocab = json.loads(assets["tokenizer.json"])["model"]["vocab"]
        target_checkpoint = None
        if not args.draft_model:
            raise ValueError(
                "A standalone target GGUF requires --draft-model for DSpark inspection"
            )
    else:
        raw = json.loads((path / "config.json").read_text())
        target_checkpoint = inspect_checkpoint(path)
    if args.draft_model:
        from .draft_source import resolve_draft_source

        if target_checkpoint is not None and target_checkpoint.draft_tensors:
            raise ValueError(
                "Target already contains draft weights; omit --draft-model"
            )
        source = resolve_draft_source(
            args.draft_model,
            raw,
            target_directory=path if vocab is None else None,
            target_vocabulary=vocab,
        )
        raw = source.config
        config = DSparkConfig.from_spark_dict(source.config)
        checkpoint = source.checkpoint()
        shared_embedding = source.config.get("dspark_shared_embedding", False)
    else:
        config = DSparkConfig.from_spark_dict(raw)
        checkpoint = target_checkpoint
        shared_embedding = raw.get("dspark_shared_embedding", False)
    validate_draft_shapes(
        config,
        checkpoint,
        shared_embedding=shared_embedding,
        quantization=raw.get("draft_quantization"),
    )
    print(
        json.dumps(
            {
                "config": asdict(config),
                "checkpoint": checkpoint.summary(),
                "draft_shapes_valid": True,
                "prediction_layout_resolved": config.logits_start is not None,
                "draft_quantization": raw.get("draft_quantization"),
            },
            indent=2,
        )
    )
    return 0


def _run_runtime(args):
    import mlx.core as mx

    from .benchmark import benchmark
    from .generation import stream_generate
    from .loader import load_pair
    from .sampling import SamplingOptions

    if args.device != "auto":
        mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    sampling = SamplingOptions(args.temperature, args.top_p, args.top_k)
    draft_sampling = sampling.with_overrides(
        temperature=args.draft_temperature,
        top_p=args.draft_top_p,
        top_k=args.draft_top_k,
    )
    pair = load_pair(
        args.model,
        model_cache_dir=args.model_cache_dir,
        model_cache=model_cache_enabled(args.model, args.model_cache),
        draft_path=args.draft_model,
        dtype=args.dtype,
        draft_quantization=args.draft_quantization,
        draft_readout=args.draft_readout,
    )
    options = {
        "max_new_tokens": args.max_new_tokens,
        "max_draft_tokens": args.max_draft_tokens,
        "confidence_threshold": args.confidence_threshold,
        "greedy_verification": args.greedy_verification,
        "sampling": sampling,
        "draft_sampling": draft_sampling,
        "seed": args.seed,
        "prefill_chunk_size": args.prefill_chunk_size,
        "apply_chat_template": not args.raw_prompt,
        "enable_thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
    }
    if args.command == "benchmark":
        report = benchmark(
            pair,
            args.prompt,
            trials=args.trials,
            warmup_tokens=args.warmup_tokens,
            **options,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["greedy_match"] is False else 0
    detokenizer = pair.tokenizer.detokenizer
    detokenizer.reset()
    eos = eos_token_ids(pair)
    for event in stream_generate(pair, args.prompt, mode=args.mode, **options):
        if isinstance(event, GenerationResult):
            result = event
        elif not args.json and event.token_id not in eos:
            detokenizer.add_token(event.token_id)
            print(detokenizer.last_segment, end="", flush=True)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        detokenizer.finalize()
        print(detokenizer.last_segment, flush=True)
        if args.stats:
            print(json.dumps(result.to_dict(), ensure_ascii=False), file=sys.stderr)
    return 0


def _run_export(args):
    import mlx.core as mx

    from .draft_storage import save_draft
    from .loader import load_pair

    output = Path(args.output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Draft output already exists: {output}")
    if args.device != "auto":
        mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    pair = load_pair(
        args.model,
        draft_path=args.draft_model,
        draft_quantization=args.draft_quantization,
        draft_readout=args.draft_readout,
        model_cache_dir=args.model_cache_dir,
        model_cache=model_cache_enabled(args.model, args.model_cache),
    )
    print(json.dumps(save_draft(pair, output), ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original `python -m dspark /path/to/checkpoint` shorthand.
    if (
        argv
        and argv[0] not in ("inspect", "generate", "benchmark", "export-draft")
        and not argv[0].startswith("-")
    ):
        argv.insert(0, "inspect")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export-draft":
            return _run_export(args)
        return _run_inspect(args) if args.command == "inspect" else _run_runtime(args)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
