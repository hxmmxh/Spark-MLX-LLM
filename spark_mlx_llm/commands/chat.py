# Chat loop adapted from MLX LM (Copyright © 2023-2024 Apple Inc.).
#
# MIT License
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Terminal chat with Spark's shared settings and optional DSpark decoding."""

import argparse
from contextlib import closing
from pathlib import Path

import mlx.core as mx
from mlx_lm.chat import setup_arg_parser
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load, sharded_load

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
        parents=[setup_arg_parser()],
        add_help=False,
        conflict_handler="resolve",
        allow_abbrev=False,
        description="Chat with Spark using shared inference settings",
    )
    GenerationConfig.add_arguments(parser, include_kv=False)
    add_model_cache_arguments(parser)
    draft = parser.add_argument_group("draft model")
    draft.add_argument(
        "--draft-model",
        help="Enable DSpark with a matching Spark draft GGUF or Safetensors directory",
    )
    draft.add_argument(
        "--draft-quantization",
        choices=("checkpoint", "bf16", "int8", "int4"),
        help="Default: int8 for floating drafts, preserve stored quantization",
    )
    return parser


def _uses_dspark(args):
    if args.draft_model:
        return True
    path = Path(args.model).expanduser()
    if path.is_dir() and (
        (path / "model.safetensors.index.json").is_file()
        or any(path.glob("model*.safetensors"))
    ):
        from ..dspark.checkpoint import inspect_checkpoint

        return bool(inspect_checkpoint(path).draft_tensors)
    return False


def _run_dspark_chat(parser, args, inference, group):
    from ..dspark.generation import stream_generate as dspark_stream_generate
    from ..dspark.loader import load_pair
    from ..dspark.metrics import GenerationResult
    from ..dspark.sampling import SamplingOptions
    from ..dspark.tokenization import eos_token_ids

    unsupported = []
    if group.size() > 1 or args.pipeline:
        unsupported.append("distributed/pipeline execution")
    if args.adapter_path:
        unsupported.append("--adapter-path")
    if args.max_kv_size is not None:
        unsupported.append("--max-kv-size")
    if inference.min_p:
        unsupported.append("--min-p")
    if args.xtc_probability:
        unsupported.append("--xtc-probability")
    if unsupported:
        parser.error("DSpark chat does not support " + ", ".join(unsupported))
    pair = load_pair(
        args.model,
        draft_path=args.draft_model,
        draft_quantization=args.draft_quantization,
        model_cache_dir=args.model_cache_dir,
        model_cache=args.model_cache,
    )
    sampling = SamplingOptions(inference.temperature, inference.top_p, inference.top_k)
    system = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt is not None
        else []
    )
    messages = list(system)
    detokenizer = pair.tokenizer.detokenizer
    eos = eos_token_ids(pair)
    help_text = "The command list: 'q' to exit, 'r' to reset, 'h' for help"
    print(f"[INFO] Starting chat session with {args.model} (DSpark).")
    print(help_text)
    while True:
        try:
            query = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query == "q":
            return 0
        if query == "r":
            messages = list(system)
            continue
        if query == "h":
            print(help_text)
            continue
        if not query.strip():
            continue
        pending = [*messages, {"role": "user", "content": query}]
        prompt = pair.tokenizer.apply_chat_template(
            pending,
            tokenize=True,
            add_generation_prompt=True,
            **inference.template_kwargs(),
        )
        detokenizer.reset()
        result = None
        events = dspark_stream_generate(
            pair,
            prompt,
            mode="dspark",
            max_new_tokens=inference.max_tokens,
            sampling=sampling,
            seed=None,
            apply_chat_template=False,
            asynchronous_draft=True,
            async_context=True,
        )
        try:
            with closing(events):
                for event in events:
                    if isinstance(event, GenerationResult):
                        result = event
                    elif event.token_id not in eos:
                        detokenizer.add_token(event.token_id)
                        print(detokenizer.last_segment, end="", flush=True)
        except KeyboardInterrupt:
            print()
            return 0
        detokenizer.finalize()
        print(detokenizer.last_segment, flush=True)
        if result is None:
            raise RuntimeError("DSpark generation ended without a result")
        messages = [*pending, {"role": "assistant", "content": result.text}]


def main():
    parser = build_parser()
    args = parser.parse_args()
    inference = GenerationConfig.from_args(parser, args, include_kv=False)
    register_model()
    args.model_cache = model_cache_enabled(args.model, args.model_cache)
    try:
        dspark = _uses_dspark(args)
    except (OSError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    if args.draft_quantization is not None and not dspark:
        parser.error("--draft-quantization requires a draft model or merged checkpoint")
    if dspark:
        group = mx.distributed.init()
        mx.random.seed(args.seed)
        try:
            return _run_dspark_chat(parser, args, inference, group)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            parser.exit(1, f"{error}\n")
    if args.model_cache:
        args.model = prepare_local_cli_model(args.model, cache_dir=args.model_cache_dir)
    group = mx.distributed.init()
    rank = group.rank()
    mx.random.seed(args.seed)

    if group.size() > 1:
        if not args.model_cache:
            parser.error("--no-model-cache requires single-process loading")
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(
            args.model,
            group if args.pipeline else None,
            None if args.pipeline else group,
        )
    else:
        model_loader = load if args.model_cache else load_without_model_cache
        model, tokenizer = model_loader(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={
                "trust_remote_code": True if args.trust_remote_code else None
            },
        )

    def rprint(*values, **kwargs):
        if rank == 0:
            print(*values, **kwargs)

    def print_help():
        rprint("The command list: 'q' to exit, 'r' to reset, 'h' for help")

    rprint(f"[INFO] Starting chat session with {args.model}.")
    print_help()
    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    while True:
        try:
            query = input(">> " if rank == 0 else "")
        except EOFError:
            break
        if query == "q":
            break
        if query == "r":
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            continue
        if query == "h":
            print_help()
            continue
        messages = []
        if args.system_prompt is not None:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **inference.template_kwargs()
        )
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=inference.max_tokens,
            sampler=make_sampler(
                **inference.sampler_kwargs(),
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=tokenizer.encode("\n")
                + list(tokenizer.eos_token_ids),
            ),
            prompt_cache=prompt_cache,
        ):
            rprint(response.text, flush=True, end="")
        rprint()


if __name__ == "__main__":
    main()
