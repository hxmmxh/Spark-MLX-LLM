"""Benchmark the exact public generation path, preserving every repetition."""

from dataclasses import replace
import statistics


def benchmark(model, tokenizer, prompt, *, config=None, trials=3, warmup_tokens=128, tools=None):
    from ..config import GenerationConfig
    from ..generation import generate

    if type(trials) is not int or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if type(warmup_tokens) is not int or warmup_tokens < 0:
        raise ValueError("warmup_tokens must be a nonnegative integer")
    config = config or GenerationConfig()
    if isinstance(config, dict):
        config = GenerationConfig.from_dict(config)
    if warmup_tokens:
        length = min(warmup_tokens, config.max_tokens) if config.max_tokens is not None else warmup_tokens
        generate(model, tokenizer, prompt, config=replace(config, max_tokens=length), tools=tools)
    results = [generate(model, tokenizer, prompt, config=config, tools=tools) for _ in range(trials)]
    rates = [result.tokens_per_second for result in results]
    seconds = sum(result.generation_seconds for result in results)
    return {
        "trials": trials, "warmup_tokens": warmup_tokens,
        "pooled_tokens_per_second": sum(result.decode_tokens for result in results) / seconds if seconds else 0,
        "min_tokens_per_second": min(rates), "max_tokens_per_second": max(rates),
        "median_tokens_per_second": statistics.median(rates),
        "outputs_repeat_exact": all(result.token_ids == results[0].token_ids for result in results),
        "runs": [result.to_dict() for result in results],
    }
