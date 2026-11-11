"""Orchestrate request prefill, drafting, target verification, and cache commits."""

import time
from dataclasses import dataclass
from functools import cache, partial

import mlx.core as mx

from .acceptance import AcceptanceResult, greedy_accept, sampled_accept
from .cache import cache_arrays
from .config import DEFAULT_MAX_DRAFT_TOKENS
from .draft_policy import verification_length
from .metrics import DraftLengthStats, GenerationResult
from .sampling import SamplingOptions, sample_draft, sample_logits
from .target import TargetOutput
from .tokenization import encode_prompt, eos_token_ids


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    from_draft: bool
    logprobs: object = None


def _stopped(tokens, eos, stop_sequences):
    return bool(tokens and (
        tokens[-1] in eos or any(
            len(tokens) >= len(sequence) and tuple(tokens[-len(sequence):]) == sequence
            for sequence in stop_sequences
        )
    ))


def _prompt_chunks(ids, chunk_size):
    # Match MLX-LM's prefill split: process the final prompt token by itself.
    for first in range(0, len(ids) - 1, chunk_size):
        yield ids[first : min(first + chunk_size, len(ids) - 1)]
    yield ids[-1:]


def _verify_sequential(target, pending, draft, state, capture, eos):
    """Run a sequential round, committing each forward and stopping at rejection/EOS."""
    outputs, committed = [], []
    accepted = 0
    for i in range(len(draft) + 1):
        token = pending if i == 0 else draft[i - 1]
        output = target.run(mx.array([[token]]), state, capture=capture)
        mx.eval(output.logits, state.arrays())
        state.finish()
        outputs.append(output)
        predicted = int(mx.argmax(output.logits[0, 0]).item())
        committed.append(predicted)
        if i < len(draft) and predicted == draft[i]:
            accepted += 1
        else:
            break
        if predicted in eos:
            break
    taps = tuple(
        mx.concatenate([output.taps[i] for output in outputs], axis=1)
        for i in range(len(outputs[0].taps))
    )
    combined = TargetOutput(
        mx.concatenate([output.logits for output in outputs], axis=1), taps
    )
    return (
        AcceptanceResult(tuple(committed), len(draft), accepted),
        combined,
        len(outputs),
    )


@cache
def _baseline_sampler(options):
    # Keep the target's sampling policy on raw logits, including applying
    # temperature before top-p. No draft acceptance packet is needed here.
    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def sample(logits):
        return sample_logits(logits, options)

    return sample


def _stream_baseline(pair, ids, options, max_new_tokens, chunk_size, mode, kv_config=None):
    """Use ordinary caches and pipeline target sampling without host token IDs."""
    model = pair.target.model
    if kv_config is not None and kv_config.kv_bits is not None:
        from ..kv_cache import make_prompt_cache
        from mlx_lm.generate import maybe_quantize_kv_cache

        caches = make_prompt_cache(model, kv_config.kv_quantization_scope)
    else:
        caches = model.make_cache()

    def quantize():
        if kv_config is not None and kv_config.kv_bits is not None:
            maybe_quantize_kv_cache(
                caches, kv_config.quantized_kv_start,
                kv_config.kv_group_size, kv_config.kv_bits,
            )
    sample = _baseline_sampler(options)

    def step(tokens):
        logits = model(tokens.reshape(1, -1), cache=caches)
        quantize()
        return sample(logits[:, -1, :])

    start = time.perf_counter()
    for first in range(0, len(ids) - 1, chunk_size):
        chunk = ids[first : min(first + chunk_size, len(ids) - 1)]
        model(mx.array([chunk]), cache=caches)
        quantize()
        mx.eval(cache_arrays(caches))
    current = step(mx.array(ids[-1:]))
    first_token = int(current.item())
    prefill_seconds = time.perf_counter() - start
    eos = eos_token_ids(pair)
    result_ids = [first_token]
    rounds = target_steps = 0
    next_token = None
    generation_start = time.perf_counter()
    try:
        if first_token not in eos and (max_new_tokens is None or max_new_tokens > 1):
            next_token = step(current)
            mx.async_eval(next_token)
            target_steps += 1
        yield TokenEvent(first_token, False)
        while next_token is not None:
            current = next_token
            next_token = None
            if max_new_tokens is None or len(result_ids) + 1 < max_new_tokens:
                # Queue the successor while current remains a device array,
                # before its host read and the consumer's token processing.
                next_token = step(current)
                mx.async_eval(next_token)
                target_steps += 1
            token = int(current.item())
            result_ids.append(token)
            rounds += 1
            yield TokenEvent(token, False)
            if token in eos:
                break
    finally:
        # EOS may leave one prefetched token. Charge its execution to this
        # request, and also finish pending work when a consumer closes early.
        pending = [current, *cache_arrays(caches)]
        if next_token is not None:
            pending.append(next_token)
        mx.eval(pending)
    duration = time.perf_counter() - generation_start
    visible = result_ids[:-1] if result_ids[-1] in eos else result_ids
    text = pair.tokenizer.decode(visible) if pair.tokenizer is not None else ""
    yield GenerationResult(
        tuple(result_ids),
        text,
        len(ids),
        "stop" if result_ids[-1] in eos else "length",
        prefill_seconds,
        duration,
        draft_seconds=0.0,
        verify_seconds=0.0,
        context_seconds=0.0,
        target_steps=target_steps,
        drafted_tokens=0,
        accepted_tokens=0,
        peak_memory_bytes=mx.get_peak_memory(),
        rounds=rounds,
        draft_length_stats=(
            {0: DraftLengthStats(rounds, 0, 0)} if mode == "dspark" and rounds else {}
        ),
        # Forward and sampling overlap across tokens; reporting additive
        # per-phase wall times here would imply a split we did not measure.
        target_forward_seconds=None,
        acceptance_seconds=None,
    )


def _generate_events(
    pair,
    prompt,
    *,
    max_new_tokens=512,
    mode="dspark",
    max_draft_tokens=None,
    confidence_threshold=0.0,
    compile_draft_chain=True,
    asynchronous_draft=True,
    async_context=True,
    greedy_verification="batched",
    sampling=None,
    draft_sampling=None,
    seed=0,
    prefill_chunk_size=512,
    apply_chat_template=True,
    enable_thinking=True,
    reasoning_effort="low",
    kv_config=None,
    target_policy=None,
    stop_sequences=(),
    prompt_progress_callback=None,
):
    """Yield TokenEvent values followed by exactly one GenerationResult.

    Token IDs include EOS when emitted; result.text excludes the trailing EOS.
    Every call creates its own target and draft caches.
    max_new_tokens=None removes the output length limit and stops only at EOS.
    max_draft_tokens=None caps each round at 3 candidates, or fewer when the
    checkpoint has fewer prediction slots. Explicit values override this cap;
    zero disables drafting without changing the checkpoint's trained block size.
    Greedy verification defaults to batched; select sequential for correctness tests.
    sampling controls the target; draft_sampling controls draft proposals and defaults
    to a copy of the target policy. Both use SamplingOptions.
    asynchronous_draft overlaps target graph construction with draft execution
    when IDs stay on device. Its draft_seconds measures submission; target_forward
    includes waiting for any remaining draft work. Both remain host-wall phases.
    async_context submits committed draft-cache updates without blocking only
    for stochastic GPU speculation. Subsequent draft graphs retain their array
    dependencies. Its context_seconds includes submission and the final flush;
    intermediate waits can appear in later phases. The final update is flushed
    before generation_seconds is finalized.
    Baseline mode and a zero draft cap use the target's ordinary caches and a
    pipelined single-token loop. They retain the target sampling policy but do
    not run transactional verification. A prefetched token may finish after EOS;
    its work is included in generation_seconds and target_steps, not rounds.
    """
    if mode not in ("baseline", "dspark"):
        raise ValueError("mode must be baseline or dspark")
    if greedy_verification not in ("sequential", "batched"):
        raise ValueError("greedy_verification must be sequential or batched")
    if max_new_tokens is not None and (
        type(max_new_tokens) is not int or max_new_tokens < 0
    ):
        raise ValueError("max_new_tokens must be a nonnegative integer or None")
    if type(prefill_chunk_size) is not int or prefill_chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be a positive integer")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    available = pair.drafter.config.max_draft_tokens if pair.drafter is not None else 0
    cap = (
        min(DEFAULT_MAX_DRAFT_TOKENS, available)
        if max_draft_tokens is None
        else max_draft_tokens
    )
    if type(cap) is not int or not 0 <= cap <= available:
        raise ValueError("max_draft_tokens must fit the trained draft prediction slots")
    options = sampling or SamplingOptions()
    draft_options = draft_sampling or options.with_overrides()
    if confidence_threshold and available and pair.drafter.confidence_head is None:
        raise ValueError("Confidence scheduling requires a confidence head")
    if seed is not None:
        mx.random.seed(seed)
    ids = encode_prompt(
        pair,
        prompt,
        apply_chat_template=apply_chat_template,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    if max_new_tokens == 0:
        yield GenerationResult((), "", len(ids), "length", 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return
    speculative = mode == "dspark" and cap > 0
    if not speculative:
        yield from _stream_baseline(
            pair, ids, options, max_new_tokens, prefill_chunk_size, mode, kv_config
        )
        return
    asynchronous_context = bool(
        async_context and speculative and options.temperature > 0
        and mx.default_device() == mx.gpu
    )
    state = pair.target.make_state()
    caches = pair.drafter.make_context_cache() if speculative else []
    start = time.perf_counter()
    processed = 0
    if prompt_progress_callback is not None:
        prompt_progress_callback(0, len(ids))
    for chunk in _prompt_chunks(ids, prefill_chunk_size):
        output = pair.target.run(
            mx.array([chunk]),
            state,
            capture=speculative,
            last_only=True,
        )
        state.finish()
        if speculative:
            pair.drafter.update_context(output.taps, caches)
        mx.eval(output.logits, state.arrays(), cache_arrays(caches))
        processed += len(chunk)
        if prompt_progress_callback is not None:
            prompt_progress_callback(processed, len(ids))
    first_logprobs = (
        # Upstream generate_step builds processor history from the final
        # prefill token, then appends generated tokens on subsequent steps.
        target_policy.logprobs(output.logits[0, -1], ids[-1:])
        if target_policy is not None else None
    )
    pending = int((
        target_policy.sampler(first_logprobs[None, :])
        if target_policy is not None else sample_logits(output.logits[0, -1], options)
    ).item())
    prefill_seconds = time.perf_counter() - start
    eos = eos_token_ids(pair)
    result_ids = [pending]
    draft_seconds = verify_seconds = context_seconds = 0.0
    target_forward_seconds = acceptance_seconds = target_commit_seconds = 0.0
    sequential = options.temperature == 0 and greedy_verification == "sequential" and target_policy is None
    rounds = target_steps = drafted_tokens = accepted_tokens = 0
    draft_length_stats = {}
    generation_start = time.perf_counter()
    yield TokenEvent(pending, False, first_logprobs)
    while (
        max_new_tokens is None or len(result_ids) < max_new_tokens
    ) and not _stopped(result_ids, eos, stop_sequences):
        count = cap if speculative else 0
        if max_new_tokens is not None:
            count = min(count, max_new_tokens - len(result_ids) - 1)
        proposal = None
        draft = []
        if count:
            t0 = time.perf_counter()
            block = mx.array(
                [
                    [pending]
                    + [pair.drafter.config.mask_token_id]
                    * (pair.drafter.config.block_size - 1)
                ]
            )
            hidden = pair.drafter.forward_cached(block, caches)
            prediction = sample_draft(
                pair.drafter,
                hidden,
                block[:, 0],
                draft_options,
                count,
                require_proposal_probs=options.temperature > 0,
                require_confidence=confidence_threshold > 0,
                compile_chain=compile_draft_chain,
            )
            tokens = prediction.tokens
            confidence = prediction.confidence
            proposal = prediction.proposal_probs
            evaluate_draft = (
                mx.async_eval
                if asynchronous_draft
                and not confidence_threshold
                and options.temperature > 0
                and mx.default_device() == mx.gpu
                else mx.eval
            )
            evaluate_draft(
                tokens,
                *([confidence] if confidence is not None else []),
                *([proposal] if proposal is not None else []),
            )
            if confidence_threshold:
                count = verification_length(
                    confidence[0].tolist(), confidence_threshold, max_draft_tokens=count
                )
            # Probabilistic verification consumes device token IDs directly;
            # acceptance returns the final committed prefix in one host read.
            draft = tokens[0, :count]
            if options.temperature == 0:
                draft = draft.tolist()
            if proposal is not None:
                proposal = proposal[0, :count]
            draft_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        if sequential:
            accepted, output, calls = _verify_sequential(
                pair.target, pending, draft, state, speculative, eos
            )
        else:
            verify_tokens = (
                mx.concatenate(
                    [mx.array([[pending]], dtype=mx.int32), draft[None, :]], axis=1
                )
                if isinstance(draft, mx.array)
                else mx.array([[pending] + draft])
            )
            output = pair.target.run(verify_tokens, state, capture=speculative)
            mx.eval(output.logits, state.arrays())
            forward_end = time.perf_counter()
            target_forward_seconds += forward_end - t0
            calls = 1
            verified_logprobs = None
            target_probabilities = None
            if target_policy is not None:
                draft_ids = draft.tolist() if isinstance(draft, mx.array) else draft
                verified_logprobs = [
                    target_policy.logprobs(row, [*ids[-1:], *result_ids, *draft_ids[:i]])
                    for i, row in enumerate(output.logits[0])
                ]
                if options.temperature > 0:
                    target_probabilities = mx.stack([
                        target_policy.probabilities(row) for row in verified_logprobs
                    ])
            if options.temperature == 0:
                accepted = greedy_accept(
                    draft, mx.argmax(
                        mx.stack(verified_logprobs) if verified_logprobs is not None
                        else output.logits[0], axis=-1
                    ).tolist()
                )
            else:
                accepted = sampled_accept(
                    draft, proposal, output.logits[0], options,
                    **({"target_probs": target_probabilities} if target_policy is not None else {}),
                )
            acceptance_seconds += time.perf_counter() - forward_end
        verify_seconds += time.perf_counter() - t0
        committed = list(accepted.tokens)
        stop_at = next(
            (i + 1 for i in range(len(committed))
             if _stopped([*result_ids, *committed[:i + 1]], eos, stop_sequences)),
            len(committed)
        )
        committed = committed[:stop_at]
        emitted_accepted = min(accepted.num_accepted, len(committed))
        keep = min(1 + emitted_accepted, output.logits.shape[1])
        if not sequential:
            commit_start = time.perf_counter()
            state.finish(keep)
            target_commit_seconds += time.perf_counter() - commit_start
        if speculative:
            t0 = time.perf_counter()
            pair.drafter.update_context(
                tuple(tap[:, :keep] for tap in output.taps), caches
            )
            evaluate_context = mx.async_eval if asynchronous_context else mx.eval
            evaluate_context(cache_arrays(caches))
            if any(cache.offset != state.offset for cache in caches):
                raise RuntimeError("Draft and target committed contexts diverged")
            context_seconds += time.perf_counter() - t0
        rounds += 1
        target_steps += calls
        drafted_tokens += len(draft)
        accepted_tokens += emitted_accepted
        if mode == "dspark":
            stats = draft_length_stats.setdefault(len(draft), DraftLengthStats())
            stats.add(DraftLengthStats(1, len(draft), emitted_accepted))
        result_ids.extend(committed)
        pending = committed[-1]
        for i, token in enumerate(committed):
            yield TokenEvent(
                token, i < emitted_accepted,
                verified_logprobs[i] if target_policy is not None else None,
            )
    if asynchronous_context:
        # Include the last outstanding cache update in the same decode timing
        # scope as the synchronous path, including early stop and token caps.
        context_start = time.perf_counter()
        mx.eval(cache_arrays(caches))
        context_seconds += time.perf_counter() - context_start
    duration = time.perf_counter() - generation_start
    visible = result_ids[:-1] if result_ids[-1] in eos else result_ids
    for sequence in stop_sequences:
        if len(visible) >= len(sequence) and tuple(visible[-len(sequence):]) == sequence:
            visible = visible[:-len(sequence)]
            break
    text = pair.tokenizer.decode(visible) if pair.tokenizer is not None else ""
    yield GenerationResult(
        tuple(result_ids),
        text,
        len(ids),
        "stop" if _stopped(result_ids, eos, stop_sequences) else "length",
        prefill_seconds,
        duration,
        draft_seconds,
        verify_seconds,
        context_seconds,
        target_steps,
        drafted_tokens,
        accepted_tokens,
        mx.get_peak_memory(),
        rounds=rounds,
        draft_length_stats=draft_length_stats,
        target_forward_seconds=None if sequential else target_forward_seconds,
        acceptance_seconds=None if sequential else acceptance_seconds,
        target_commit_seconds=target_commit_seconds,
    )


def _collect(pair, prompt, *, on_token=None, **kwargs):
    result = None
    for event in _generate_events(pair, prompt, **kwargs):
        if isinstance(event, TokenEvent):
            if on_token is not None:
                on_token(event)
        else:
            result = event
    return result
