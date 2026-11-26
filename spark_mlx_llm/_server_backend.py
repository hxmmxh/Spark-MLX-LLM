"""Generation and loading adapters for the unchanged MLX-LM HTTP protocol."""

import json
import logging
from threading import Event
from dataclasses import replace
from unittest.mock import patch

import mlx.core as mx
from mlx_lm import server as upstream

from .config import GenerationConfig


logger = logging.getLogger(__name__)


class ModelProvider(upstream.ModelProvider):
    """Preserve upstream model/adapter selection and add Spark draft loading."""

    def __init__(self, cli_args, *, config=None, load_options=None, served_model_name=None, prefill_text=None, prefill_stages=()):
        super().__init__(cli_args)
        self.spark_config = config or GenerationConfig()
        self.spark_config.validate_http()
        self.load_options = dict(load_options or {})
        self.served_model_name = served_model_name
        self.preloaded = None
        self.prefill_text = prefill_text
        self.prefill_stages = prefill_stages
        if served_model_name:
            self._model_map[served_model_name] = cli_args.model

    def load_default(self):
        if self.preloaded is not None:
            self.model, self.tokenizer = self.preloaded
            self.model_key = (self.cli_args.model, None, None)
            self.draft_model = None
            self.is_batchable = False
        else:
            super().load_default()

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        if self.preloaded is not None and model_path in (
            "default_model", self.served_model_name, self.cli_args.model
        ) and adapter_path is None and draft_model_path in (None, "default_model"):
            self.load_default()
            return self.model, self.tokenizer
        return super().load(model_path, adapter_path, draft_model_path)

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        from ._target_loader import _resolve_model_path
        from .dspark.checkpoint import inspect_checkpoint
        from .loader import load

        use_draft = self.load_options.get("use_draft", True)
        path = _resolve_model_path(model_path, None)
        config_path = path / "config.json" if path.is_dir() else None
        raw = json.loads(config_path.read_text()) if config_path and config_path.is_file() else {}
        spark = raw.get("model_type") == "spark2_5" or path.suffix.lower() == ".gguf"
        merged = bool(
            use_draft and spark and path.is_dir() and any(path.glob("*.safetensors"))
            and inspect_checkpoint(path).draft_tensors
        )
        if not (use_draft and spark and (merged or draft_model_path)):
            super()._load(str(path), adapter_path, draft_model_path if use_draft else None)
            self.model_key = (model_path, adapter_path, draft_model_path)
            return
        if self.is_distributed:
            raise ValueError("Loading draft models is not supported in distributed mode")
        options = {name: value for name, value in self.load_options.items() if name != "use_draft"}
        options["tokenizer_config"] = {
            name: value for name, value in self._tokenizer_config.items() if value is not None
        }
        model, tokenizer = load(path, draft_model=draft_model_path, **options)
        if adapter_path is not None:
            from mlx_lm.utils import load_adapters
            model = load_adapters(model, adapter_path)
            model.eval()
        if self.cli_args.chat_template:
            tokenizer._tokenizer.chat_template = self.cli_args.chat_template
            tokenizer.has_chat_template = True
        if self.cli_args.use_default_chat_template and not tokenizer.has_chat_template:
            tokenizer._tokenizer.chat_template = tokenizer.default_chat_template
            tokenizer.has_chat_template = True
        self.model, self.tokenizer = model, tokenizer
        self.draft_model = None
        self.model_key = (model_path, adapter_path, draft_model_path)
        self.is_batchable = False


class ResponseGenerator(upstream.ResponseGenerator):
    """One generator interface; ordinary requests retain the upstream engine."""

    def __init__(self, model_provider, prompt_cache):
        self._ready = Event()
        self._startup_error = None
        self._spark_startup_prefix = None
        if model_provider.cli_args.prompt_cache_size == -1:
            # Preserve upstream's zero=disabled behavior; -1 removes only the
            # entry-count limit, while the byte budget remains effective.
            prompt_cache.max_size = float("inf")
        budget = model_provider.cli_args.prompt_cache_bytes
        if budget is not None:
            prompt_cache.max_bytes = min(prompt_cache.max_bytes, max(0, budget))
        logger.info(
            "Prompt cache limits: entries=%s bytes=%d",
            "unlimited" if prompt_cache.max_size == float("inf") else prompt_cache.max_size,
            prompt_cache.max_bytes,
        )
        super().__init__(model_provider, prompt_cache)

    def _generate(self):
        try:
            limit = self.model_provider.spark_config.cache_limit_mib
            if self.model_provider.prefill_text is not None and limit is not None:
                mx.set_cache_limit(limit * 1024**2)
            self.model_provider.load_default()
            if self.model_provider.prefill_text is not None:
                import time
                from ._server_prefill import prefill

                start = time.perf_counter()
                logger.info("Prefilling startup prefix...")
                prefill(self, self.model_provider.prefill_text, stages=self.model_provider.prefill_stages)
                logger.info("Startup prefill took %.3f seconds", time.perf_counter() - start)
        except Exception as error:
            self._startup_error = error
            return
        finally:
            self._ready.set()
        super()._generate()

    def wait_ready(self):
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def _tokenize(self, tokenizer, request, args):
        encoded = super()._tokenize(tokenizer, request, args)
        prompt = encoded[0]
        limits = [
            self.model_provider.spark_config.max_context_length,
            getattr(request, "spark_overrides", {}).get("max_context_length"),
        ]
        limits = [limit for limit in limits if limit is not None]
        if not limits:
            return encoded
        limit = min(limits)
        remaining = limit - len(prompt)
        message = None
        if not prompt or remaining <= 0:
            message = f"Prompt must leave space in max_context_length={limit}"
        elif args.max_tokens > remaining:
            if getattr(request, "spark_max_tokens_explicit", False):
                message = (
                    f"{len(prompt)} prompt + {args.max_tokens} output tokens "
                    f"exceeds max_context_length={limit}"
                )
            else:
                # Preserve the original service's distinction between its
                # default output cap and an explicitly requested output length.
                args.max_tokens = remaining
        if message is not None:
            request.spark_context_error = True
            raise ValueError(message)
        return encoded

    def _is_batchable(self, args):
        # 0.31.3 passes [eos, encode("\n")] into XTC, creating a nested index
        # list. Use its single-request path with the flattened sampler below.
        return not args.sampling.xtc_probability and super()._is_batchable(args)

    def _serve_single(self, item):
        pair = getattr(self.model_provider.model, "_spark_runtime", None)
        if pair is None or pair.drafter is None or min(
            item[2].num_draft_tokens, pair.drafter.config.max_draft_tokens
        ) == 0:
            if item[2].sampling.xtc_probability:
                from .dspark.sampling import TargetSamplingPolicy

                def sampler(args, tokenizer):
                    return TargetSamplingPolicy(
                        args.sampling, [],
                        [tokenizer.eos_token_id, *tokenizer.encode("\n")],
                    ).sampler

                with patch.object(upstream, "_make_sampler", sampler):
                    return super()._serve_single(item)
            return super()._serve_single(item)
        self._serve_spark(item, pair)

    def _serve_spark(self, item, pair):
        from ._server_prefill import save_spark_prefix, spark_cache_key
        from .dspark.generation import TokenEvent, _generate_events
        from .dspark.metrics import GenerationResult
        from .dspark.sampling import SamplingOptions, TargetSamplingPolicy

        queue, request, args = item
        events = None
        try:
            tokenizer = self.model_provider.tokenizer
            config = self.model_provider.spark_config.overlay(
                getattr(request, "spark_overrides", {})
            )
            prompt, segments, segment_types, initial = self._tokenize(tokenizer, request, args)
            model_key = self.model_provider.model_key
            # Leave one input token for logits, including identical/shorter prompts.
            cached, rest = self.prompt_cache.fetch_nearest_cache(
                spark_cache_key(model_key), prompt[:-1],
            )
            cached_count = len(prompt) - 1 - len(rest) if cached else 0
            boundaries = {len(prompt) - 1}
            offset = 0
            for segment, kind in zip(segments, segment_types):
                offset += len(segment)
                if kind == "system":
                    boundaries.add(offset)

            def common_length(left, right):
                return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                            min(len(left), len(right)))

            # Some chat templates render an empty user message as a literal
            # prefix, which upstream's mismatch-only segmentation misses.
            if request.request_type == "chat" and "system" not in segment_types:
                system_messages = []
                for message in request.messages:
                    if message["role"] != "system":
                        break
                    system_messages.append(message)
                if system_messages:
                    system_request = replace(request, messages=[
                        *system_messages, {"role": "user", "content": ""},
                    ])
                    system_tokens = super()._tokenize(tokenizer, system_request, args)[0]
                    boundaries.add(common_length(system_tokens, prompt))

            # Use the byte-bounded upstream index to discover common prefixes;
            # an unlimited entry count must not create an unbounded history list.
            match = self.prompt_cache._trie.search(spark_cache_key(model_key), prompt)
            boundaries.add(len(prompt) if match.exact is not None else match.common_prefix)
            startup = self._spark_startup_prefix
            startup_common = common_length(startup[1], prompt) if startup and startup[0] == model_key else 0
            logger.info(
                "dSpark prompt cache: input=%d cached=%d remaining=%d startup_common=%d/%d startup_expected=%s startup_actual=%s entries=%d bytes=%d",
                len(prompt), cached_count, len(prompt) - cached_count, startup_common,
                len(startup[1]) if startup else 0,
                startup[1][startup_common] if startup and startup_common < len(startup[1]) else None,
                prompt[startup_common] if startup and startup_common < len(prompt) else None,
                len(self.prompt_cache), self.prompt_cache.nbytes,
            )

            def save_checkpoint(tokens, state, caches):
                save_spark_prefix(
                    self, tokens, state, caches,
                    cache_type="user" if len(tokens) == len(prompt) - 1 else "system",
                )

            def save_generated_checkpoint(tokens, state, caches):
                save_spark_prefix(self, tokens, state, caches, cache_type="user")
                logger.info(
                    "dSpark generated checkpoint: prompt=%d prefix=%d generated_kv=%d entries=%d bytes=%d",
                    len(prompt), len(tokens), len(tokens) - len(prompt),
                    len(self.prompt_cache), self.prompt_cache.nbytes,
                )

            machine, sequences = self._make_state_machine(
                self.model_provider.model_key, tokenizer, args.stop_words, initial
            )
            state = machine.make_state()
            context = upstream.GenerationContext(
                tokenizer.has_tool_calling, tokenizer.has_thinking,
                tokenizer.tool_parser, sequences, prompt,
                prompt_cache_count=cached_count,
            )
            queue.put(context)
            policy = TargetSamplingPolicy(
                args.sampling, upstream._make_logits_processors(args),
                [tokenizer.eos_token_id, *tokenizer.encode("\n")],
            )
            cap = min(args.num_draft_tokens, pair.drafter.config.max_draft_tokens)
            pair = replace(pair, generation_config={
                **pair.generation_config, "eos_token_id": list(tokenizer.eos_token_ids),
            })
            events = _generate_events(
                pair, prompt, max_new_tokens=args.max_tokens, max_draft_tokens=cap,
                sampling=SamplingOptions(temperature=args.sampling.temperature),
                draft_sampling=SamplingOptions(
                    config.draft_temperature, config.draft_top_p, config.draft_top_k
                ),
                target_policy=policy,
                stop_sequences=tuple(
                    tuple(tokenizer.encode(word, add_special_tokens=False))
                    for word in args.stop_words if word
                ),
                seed=args.seed, prefill_chunk_size=self.cli_args.prefill_step_size,
                asynchronous_draft=config.asynchronous_draft,
                async_context=config.async_context,
                compile_draft_chain=config.compile_draft_chain,
                confidence_threshold=config.confidence_threshold,
                apply_chat_template=False,
                prompt_progress_callback=lambda processed, total: queue.put((processed, total)),
                _prompt_cache=(prompt[:cached_count], cached[0].target, cached[0].draft) if cached else None,
                _prefill_cache_boundaries=boundaries,
                _prefill_cache_callback=save_checkpoint,
                _generation_cache_callback=save_generated_checkpoint,
            )
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            pending = None
            for event in events:
                if isinstance(event, TokenEvent):
                    state, match, current = machine.match(state, event.token_id)
                    if event.token_id not in tokenizer.eos_token_ids:
                        detokenizer.add_token(event.token_id)
                    response = upstream.Response(
                        detokenizer.last_segment, event.token_id, current, match,
                        float(event.logprobs[event.token_id].item()), None,
                        upstream._format_top_logprobs(event.logprobs, args.top_logprobs, tokenizer),
                    )
                    if pending is not None:
                        queue.put(pending)
                    pending = response
                elif isinstance(event, GenerationResult):
                    if pending is not None:
                        pending = replace(pending, finish_reason=event.finish_reason)
                if context._should_stop:
                    break
            detokenizer.finalize()
            if pending is not None:
                pending = replace(pending, text=pending.text + detokenizer.last_segment)
                queue.put(pending)
            queue.put(None)
        except Exception as error:
            queue.put(error)
        finally:
            if events is not None:
                events.close()
                # A disconnected client can close before the engine's final
                # context flush. Finish submitted work on the owning thread.
                mx.synchronize()
