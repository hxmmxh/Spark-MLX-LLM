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


@dataclass(frozen=True)
class PrefillStage:
    name: str
    text: str


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


def read_prefixes(config):
    if config.skip_prefill:
        return ()
    directory = Path(__file__).with_name("astudio_template")
    path = Path(config.prefill_file).expanduser() if config.prefill_file else directory / "template.txt"
    raw = path.read_bytes()
    if config.prefill_file is None:
        info = json.loads((directory / "template-info.json").read_text(encoding="utf-8"))
        if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError("Bundled prefill template does not match template-info.json")
    def decode(data):
        text = data.decode("utf-8")
        if config.prefill_newlines == "lf":
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise ValueError("Prefill file must contain a nonempty rendered prefix")
        return text

    text = decode(raw)
    if config.prefill_file is not None or config.single_prefill:
        return (PrefillStage("main", text),)
    stages = info["main_stages"]
    cuts = [stage["bytes"] for stage in stages]
    if len(cuts) != 3 or any(type(cut) is not int for cut in cuts) or not 0 < cuts[0] < cuts[1] < cuts[2] == len(raw):
        raise ValueError("Expected three increasing main template boundaries")
    if (
        [stage["name"] for stage in stages] != ["system1", "system1_system2", "system1_system2_user1_fixed"]
        or not raw[cuts[0]:].startswith("<｜start▁of▁sentence｜><|System|>".encode())
        or not raw[cuts[1]:].startswith("<｜start▁of▁sentence｜><|User|>".encode())
        or not text.endswith(info["prefix_boundary"])
    ):
        raise ValueError("Main template stage boundary mismatch")
    title_info = info["title"]
    title = (directory / "title.txt").read_bytes()
    if len(title) != title_info["bytes"] or hashlib.sha256(title).hexdigest() != title_info["sha256"]:
        raise ValueError("Bundled title template does not match template-info.json")
    title_text = decode(title)
    if not title_text.endswith(title_info["prefix_boundary"]):
        raise ValueError("Title template boundary mismatch")
    return (PrefillStage("title", title_text), *(
        PrefillStage(stage["name"], decode(raw[:stage["bytes"]])) for stage in stages
    ))


def read_prefix(config):
    stages = read_prefixes(config)
    return stages[-1].text if stages else None


def prefill(generator, text, *, stages=()):
    """Run on the generation thread, without sampling or formatting a request."""
    provider = generator.model_provider
    if provider.model is None:
        raise ValueError("Startup prefill requires a default model")
    limit = provider.spark_config.max_context_length
    if generator.prompt_cache.max_size < 1:
        raise ValueError("Startup prefill requires --prompt-cache-size >= 1 or -1")
    encoded = []
    previous = None
    for stage in stages or (PrefillStage("main", text),):
        tokens = list(provider.tokenizer.encode(stage.text, add_special_tokens=False))
        if not tokens:
            raise ValueError("Prefill file produced no tokens")
        if limit is not None and len(tokens) >= limit:
            raise ValueError(f"Prefill prefix must leave space in max_context_length={limit}")
        if previous is not None and (len(tokens) <= len(previous) or tokens[:len(previous)] != previous):
            raise ValueError("Main prefill stages must be nested token prefixes")
        encoded.append((stage.name, tokens))
        previous = None if stage.name == "title" else tokens
    pair = getattr(provider.model, "_spark_runtime", None)
    spark = pair is not None and pair.drafter is not None and provider.spark_config.max_draft_tokens > 0
    step = provider.cli_args.prefill_step_size
    budget = provider.cli_args.prompt_cache_bytes
    reset = True
    saved_prefixes = []
    for index, (name, tokens) in enumerate(encoded):
        if reset:
            if spark:
                state = pair.target.make_state()
                caches = pair.drafter.make_context_cache()
            else:
                caches = make_prompt_cache(provider.model)
                if provider.draft_model is not None:
                    caches += make_prompt_cache(provider.draft_model)
            reused = 0
        else:
            reused = len(encoded[index - 1][1]) - 1
        # Each checkpoint leaves one input token for the next request's logits.
        prefix = tokens[:-1]
        for first in range(reused, len(prefix), step):
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
                logger.info("Startup prefill %s progress: %d/%d cached tokens", name, processed, len(prefix))
        if prefix:
            cache_type = "user" if index == len(encoded) - 1 else "system"
            if spark:
                save_spark_prefix(generator, prefix, state, caches, cache_type=cache_type)
                saved_prefixes.append(prefix)
                if name != "title":
                    generator._spark_startup_prefix = (provider.model_key, prefix)
            else:
                generator.prompt_cache.insert_cache(provider.model_key, prefix, deepcopy(caches), cache_type="system")
        logger.info(
            "Startup prefill stage %s: input=%d cached=%d reused=%d computed=%d entries=%d bytes=%d",
            name, len(tokens), len(prefix), reused, len(prefix) - reused,
            len(generator.prompt_cache), generator.prompt_cache.nbytes,
        )
        reset = name == "title"
    if spark:
        key = spark_cache_key(provider.model_key)
        retained = sum(generator.prompt_cache._trie.search(key, prefix).exact is not None for prefix in saved_prefixes)
        if retained < len(saved_prefixes):
            logger.warning(
                "Startup prefill retained %d/%d checkpoints; increase --prompt-cache-bytes/--prompt-cache-size "
                "or use --single-prefill to warm only the full main prefix",
                retained, len(saved_prefixes),
            )
    logger.info("Prefill complete: %d input tokens, %d cached tokens", len(tokens), len(prefix))
    logger.debug("Prefill memory: active=%d bytes, allocator cache=%d bytes",
                 mx.get_active_memory(), mx.get_cache_memory())
