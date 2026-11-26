"""Read and compute the HTTP service's fixed, already rendered prefix."""

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .dspark.cache import cache_arrays

logger = logging.getLogger(__name__)


@dataclass
class SparkPromptCache:
    """An exact checkpoint; sliding target/draft history cannot be rewound."""

    target: object
    draft: list

    @property
    def nbytes(self):
        return sum(cache.nbytes for cache in (*self.target.layers, *self.draft))

    def is_trimmable(self):
        return False


def spark_cache_key(model_key):
    # Ordinary decoding must never receive speculative cache objects.
    return ("spark-target-and-draft", model_key)


def save_spark_prefix(generator, tokens, state, caches, *, cache_type="user"):
    if not tokens or generator.prompt_cache.max_size < 1:
        return
    if state.offset != len(tokens) or any(
        cache.offset != len(tokens) or cache.pending for cache in (*state.layers, *caches)
    ):
        raise ValueError("Cached tokens must match committed target and draft offsets")
    checkpoint = SparkPromptCache(state, caches)
    if checkpoint.nbytes > generator.prompt_cache.max_bytes:
        return
    generator.prompt_cache.insert_cache(
        spark_cache_key(generator.model_provider.model_key), list(tokens),
        [deepcopy(checkpoint)], cache_type=cache_type,
    )


def read_prefix(config):
    if config.skip_prefill:
        return None
    directory = Path(__file__).with_name("astudio_template")
    path = Path(config.prefill_file).expanduser() if config.prefill_file else directory / "template.txt"
    raw = path.read_bytes()
    if config.prefill_file is None:
        info = json.loads((directory / "template-info.json").read_text(encoding="utf-8"))
        if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError("Bundled prefill template does not match template-info.json")
    text = raw.decode("utf-8")
    if config.prefill_newlines == "lf":
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("Prefill file must contain a nonempty rendered prefix")
    return text


def prefill(generator, text):
    """Run on the generation thread, without sampling or formatting a request."""
    provider = generator.model_provider
    if provider.model is None:
        raise ValueError("Startup prefill requires a default model")
    tokens = provider.tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError("Prefill file produced no tokens")
    limit = provider.spark_config.max_context_length
    if limit is not None and len(tokens) >= limit:
        raise ValueError(f"Prefill prefix must leave space in max_context_length={limit}")
    if generator.prompt_cache.max_size < 1:
        raise ValueError("Startup prefill requires --prompt-cache-size >= 1 or -1")
    # Leave the last token for the request so even an identical prompt has logits.
    prefix = tokens[:-1]
    pair = getattr(provider.model, "_spark_runtime", None)
    spark = pair is not None and pair.drafter is not None and provider.spark_config.max_draft_tokens > 0
    if spark:
        state = pair.target.make_state()
        caches = pair.drafter.make_context_cache()
    else:
        caches = make_prompt_cache(provider.model)
        if provider.draft_model is not None:
            caches += make_prompt_cache(provider.draft_model)
    step = provider.cli_args.prefill_step_size
    budget = provider.cli_args.prompt_cache_bytes
    for first in range(0, len(prefix), step):
        if generator._stop:
            raise RuntimeError("Startup prefill cancelled")
        chunk = mx.array([prefix[first:first + step]])
        if spark:
            output = pair.target.run(chunk, state, last_only=True)
            state.finish()
            pair.drafter.update_context(output.taps, caches)
            mx.eval(state.arrays(), cache_arrays(caches))
            size = sum(cache.nbytes for cache in (*state.layers, *caches))
        else:
            target_count = len(provider.model.layers)
            provider.model(chunk, cache=caches[:target_count])
            if provider.draft_model is not None:
                provider.draft_model(chunk, cache=caches[target_count:])
            mx.eval(cache_arrays(caches))
            size = sum(cache.nbytes for cache in caches)
        if budget is not None and size > budget:
            raise ValueError("Prefill prefix exceeds --prompt-cache-bytes")
        processed = first + chunk.shape[1]
        if processed == len(prefix) or processed // 8192 != first // 8192:
            logger.info("Startup prefill progress: %d/%d cached tokens", processed, len(prefix))
    if prefix:
        if spark:
            save_spark_prefix(generator, prefix, state, caches, cache_type="user")
            generator._spark_startup_prefix = (provider.model_key, prefix)
        else:
            generator.prompt_cache.insert_cache(provider.model_key, prefix, caches, cache_type="system")
    logger.info("Prefill complete: %d input tokens, %d cached tokens", len(tokens), len(prefix))
    logger.debug("Prefill memory: active=%d bytes, allocator cache=%d bytes",
                 mx.get_active_memory(), mx.get_cache_memory())
