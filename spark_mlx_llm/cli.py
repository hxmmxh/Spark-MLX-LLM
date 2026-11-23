import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from .loader import load


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate text with Spark2_5 or Spark3"
    )
    parser.add_argument("--model", required=True, help="Local path or HF repo")
    parser.add_argument("--prompt", "-p", default="你好")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Still-image path; repeat for multiple images",
    )
    parser.add_argument(
        "--tokenizer-path", help="Separate matching tokenizer directory or HF repo"
    )
    parser.add_argument("--image-min-pixels", type=int)
    parser.add_argument("--image-max-pixels", type=int)
    parser.add_argument("--max-tokens", "-m", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="MLX execution device (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=("checkpoint", "float32", "bfloat16"),
        default="checkpoint",
        help="Weight dtype override (default: keep checkpoint dtype)",
    )
    parser.add_argument("--ignore-chat-template", action="store_true")
    parser.add_argument(
        "--chat-template",
        help="Path to the checkpoint's training chat template (Jinja)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.ignore_chat_template and args.chat_template:
        raise ValueError(
            "--chat-template cannot be combined with --ignore-chat-template"
        )
    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    elif args.device == "gpu":
        mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        dtype=None if args.dtype == "checkpoint" else args.dtype,
        tokenizer_path=args.tokenizer_path,
    )
    prompt = args.prompt
    processor = None
    if args.image:
        from .processing_spark3 import Spark3Processor

        if model.model_type != "spark3" or model.args.vision_config is None:
            raise ValueError("--image requires a Spark3 checkpoint with vision_config")
        from .loader import _resolve_model_path

        processor = Spark3Processor.from_pretrained(
            _resolve_model_path(args.model, None), tokenizer, model.args
        )
        if processor.image_token not in prompt:
            prompt = (
                "\n".join([processor.image_placeholder] * len(args.image))
                + "\n"
                + prompt
            )
    if not args.ignore_chat_template and (
        args.chat_template or tokenizer.has_chat_template
    ):
        template_options = {}
        if args.chat_template:
            template_options["chat_template"] = Path(args.chat_template).read_text()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **template_options,
        )

    sampler = make_sampler(
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    generation = {
        "max_tokens": args.max_tokens,
        "sampler": sampler,
        "verbose": True,
    }
    if processor is not None:
        from .multimodal import generate_multimodal

        image_options = {
            key: value
            for key, value in (
                ("min_pixels", args.image_min_pixels),
                ("max_pixels", args.image_max_pixels),
            )
            if value is not None
        }
        generate_multimodal(
            model,
            processor,
            prompt,
            args.image,
            images_kwargs=image_options,
            **generation,
        )
        return
    generate(
        model,
        tokenizer,
        prompt=prompt,
        **generation,
    )


if __name__ == "__main__":
    main()
