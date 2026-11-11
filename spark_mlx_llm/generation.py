"""The single public generation API, shared by every inference frontend."""

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, replace

from .config import GenerationConfig
from .dspark.metrics import GenerationResult

__all__ = ["GenerationConfig", "GenerationResult", "GenerationChunk", "generate"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationChunk:
    text: str = ""
    token_id: int | None = None
    from_draft: bool = False
    result: GenerationResult | None = None

    def to_dict(self):
        return {
            "text": self.text, "token_id": self.token_id, "from_draft": self.from_draft,
            "result": self.result.to_dict() if self.result is not None else None,
        }


def _messages(prompt):
    messages = []
    for message in prompt:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("Each message must be an object with a role")
        message = deepcopy(message)
        content = message.get("content", "")
        if content is None and message.get("tool_calls"):
            content = ""
        if isinstance(content, list):
            if any(not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str) for p in content):
                raise ValueError("Only text message content is supported")
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("Message content must be text or text parts")
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("tool_calls must contain function objects")
            function = call["function"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool_call.function.arguments must be a JSON object")
            function["arguments"] = arguments
        messages.append({**message, "content": content})
    return messages


def _encode_prompt(tokenizer, prompt, config, tools=None):
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools)):
        raise ValueError("tools must be a list of objects")
    is_messages = isinstance(prompt, (list, tuple)) and bool(prompt) and isinstance(prompt[0], dict)
    if is_messages or (isinstance(prompt, str) and config.apply_chat_template):
        if tokenizer is None or not getattr(tokenizer, "has_chat_template", False):
            if is_messages or tools:
                raise ValueError("Messages and tools require a chat template")
        else:
            if is_messages and not config.apply_chat_template:
                raise ValueError("Message inputs require apply_chat_template=True")
            messages = _messages(prompt) if is_messages else [{"role": "user", "content": prompt}]
            template = config.template_kwargs()
            if tools is not None:
                template["tools"] = tools
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template)
    elif tools:
        raise ValueError("tools require a chat-template input")
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("Text prompts require a tokenizer")
        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    else:
        ids = list(prompt.tolist() if hasattr(prompt, "tolist") else prompt)
    if not ids:
        raise ValueError("Prompt contains no tokens")
    return ids


def _context_limit(runtime, config):
    limits = [runtime.config.get("max_position_embeddings"), config.max_context_length]
    limits = [limit for limit in limits if limit is not None]
    return min(limits) if limits else None


def _prompt_ids(runtime, tokenizer, prompt, config, tools):
    ids = _encode_prompt(tokenizer, prompt, config, tools)
    vocab_size = runtime.config["vocab_size"]
    if not ids or any(type(t) is not int or not 0 <= t < vocab_size for t in ids):
        raise ValueError("Prompt must contain nonempty valid token IDs")
    limit = _context_limit(runtime, config)
    if limit is not None and (len(ids) >= limit or (config.max_tokens is not None and len(ids) + config.max_tokens > limit)):
        raise ValueError(
            f"Prompt plus max_tokens exceeds context limit={limit} "
            "(model capacity or max_context_length)"
        )
    return ids


def generate(model, tokenizer, prompt, *, config=None, stream=False, tools=None):
    """Return a GenerationResult, or GenerationChunk iterator when stream=True.

    Target-only and speculative decoding use this same API, configuration and
    timing convention. No sampler/keyword aliases or separate streaming API.
    """
    if config is None:
        config = GenerationConfig()
    elif isinstance(config, dict):
        config = GenerationConfig.from_dict(config)
    elif not isinstance(config, GenerationConfig):
        raise TypeError("config must be a GenerationConfig or a JSON object")
    if type(stream) is not bool:
        raise ValueError("stream must be a boolean")
    runtime = getattr(model, "_spark_runtime", None)
    if runtime is None:
        raise TypeError("model must come from spark_mlx_llm.load")
    if tokenizer is not runtime.tokenizer:
        raise ValueError("Use the tokenizer returned with this model")
    if not config.stop_at_eos:
        runtime = replace(runtime, generation_config={"eos_token_id": []})
    ids = _prompt_ids(runtime, tokenizer, prompt, config, tools)
    max_tokens = config.max_tokens
    context_limit = _context_limit(runtime, config)
    if config.max_context_length is not None and max_tokens is None:
        max_tokens = context_limit - len(ids)
    available = runtime.drafter.config.max_draft_tokens if runtime.drafter is not None else 0
    cap = min(config.max_draft_tokens, available)
    if cap and config.kv_bits is not None:
        raise ValueError("Quantized KV requires max_draft_tokens=0; speculative KV retains model precision")
    if not cap and config.confidence_threshold:
        raise ValueError("confidence_threshold requires an active draft model")
    import mlx.core as mx
    from .dspark.generation import _generate_events as engine
    from .dspark.sampling import SamplingOptions

    stochastic_gpu = bool(cap and config.temperature > 0 and mx.default_device() == mx.gpu)
    requested = config.to_dict()
    effective = {
        **requested, "requested": requested, "stream": stream,
        "device": str(mx.default_device()),
        "max_draft_tokens": cap,
        "max_tokens": max_tokens,
        "max_context_length": context_limit if config.max_context_length is not None else None,
        "asynchronous_draft": bool(config.asynchronous_draft and stochastic_gpu and not config.confidence_threshold),
        "async_context": bool(config.async_context and stochastic_gpu),
        "load": getattr(model, "_spark_load_options", {}),
    }
    encoded = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    effective["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    prompt_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    logger.info("Effective generation config: %s", json.dumps(effective, sort_keys=True))
    if config.cache_limit_mib is not None:
        mx.set_cache_limit(config.cache_limit_mib * 1024**2)
    if config.clear_cache:
        mx.synchronize()
        mx.clear_cache()
    events = engine(
        runtime, ids, mode="dspark" if cap else "baseline",
        max_new_tokens=max_tokens, max_draft_tokens=cap,
        sampling=SamplingOptions(config.temperature, config.top_p, config.top_k, config.min_p),
        draft_sampling=SamplingOptions(config.draft_temperature, config.draft_top_p, config.draft_top_k),
        seed=config.seed, prefill_chunk_size=config.prefill_step_size,
        confidence_threshold=config.confidence_threshold,
        asynchronous_draft=config.asynchronous_draft, async_context=config.async_context,
        compile_draft_chain=config.compile_draft_chain, apply_chat_template=False,
        kv_config=config,
    )
    if stream:
        return _chunks(events, runtime, effective, prompt_hash)
    try:
        for event in events:
            if isinstance(event, GenerationResult):
                return replace(event, effective_config=effective, prompt_sha256=prompt_hash)
    finally:
        events.close()
    raise RuntimeError("Generation ended without a result")


def _chunks(events, runtime, effective, prompt_hash):
    from .dspark.tokenization import eos_token_ids

    detokenizer = runtime.tokenizer.detokenizer
    detokenizer.reset()
    eos = eos_token_ids(runtime)
    try:
        for event in events:
            if isinstance(event, GenerationResult):
                detokenizer.finalize()
                tail = detokenizer.last_segment
                yield GenerationChunk(tail, result=replace(event, effective_config=effective, prompt_sha256=prompt_hash))
            else:
                text = ""
                if event.token_id not in eos:
                    detokenizer.add_token(event.token_id)
                    text = detokenizer.last_segment
                yield GenerationChunk(text, event.token_id, event.from_draft)
    finally:
        events.close()
