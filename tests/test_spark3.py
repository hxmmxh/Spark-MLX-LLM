"""Numerical and cache checks for Spark3, including an independent NumPy model."""

import math
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    load_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.utils import quantize_model

from spark_mlx_llm import spark3
from spark_mlx_llm.dspark.cache import cache_arrays
from spark_mlx_llm.spark3 import Model, ModelArgs


def make_args(**overrides):
    config = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 12,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "sliding_window": 4,
        "q_lora_rank": 8,
        "kv_source_layer_ids": 3,
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
            "full_attention": {"rope_theta": 10000, "partial_rotary_factor": 0.5},
        },
    }
    config.update(overrides)
    return ModelArgs(**config)


@pytest.fixture(params=["cpu", "metal"])
def device(request):
    if request.param == "metal" and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu if request.param == "cpu" else mx.gpu)
    yield request.param
    mx.set_default_device(previous)


def numpy_reference(model, tokens):
    """Explicit FP64 RMSNorm/RoPE/softmax/GEGLU, without MLX attention helpers."""
    args = model.args
    weights = {
        name: np.array(value.astype(mx.float32), copy=True).astype(np.float64)
        for name, value in tree_flatten(model.parameters())
    }

    def linear(x, name):
        return x @ weights[name + ".weight"].T

    def norm(x, name):
        return (
            x
            / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + args.rms_norm_eps)
            * weights[name + ".weight"]
        )

    def rope(x, params):
        dims = int(args.head_dim * params["partial_rotary_factor"])
        angles = np.arange(x.shape[2])[:, None] / (
            params["rope_theta"] ** (np.arange(0, dims, 2) / dims)
        )
        cos, sin = np.cos(angles), np.sin(angles)
        a, b = x[..., : dims // 2], x[..., dims // 2 : dims]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos, x[..., dims:]], -1)

    x = weights["model.embedding.weight"][np.asarray(tokens)]
    batch, length, _ = x.shape
    positions = np.arange(length)
    causal = positions[:, None] >= positions[None]
    shared = None
    for i, layer_type in enumerate(args.layer_types):
        prefix = f"model.layers.{i}"
        attn = prefix + ".self_attn"
        h = norm(x, prefix + ".input_layernorm")
        q = linear(
            norm(linear(h, attn + ".q_down_proj"), attn + ".q_down_layernorm"),
            attn + ".q_up_proj",
        )
        q = q.reshape(batch, length, args.num_attention_heads, args.head_dim)
        q = rope(q.transpose(0, 2, 1, 3), args.rope_parameters[layer_type])
        if args.is_ced_consumer(i):
            k, v = shared
        else:
            shape = (batch, length, args.num_key_value_heads, args.head_dim)
            k = linear(h, attn + ".k_proj").reshape(shape).transpose(0, 2, 1, 3)
            v = linear(h, attn + ".v_proj").reshape(shape).transpose(0, 2, 1, 3)
            k = rope(k, args.rope_parameters[layer_type])
        if i == args.kv_source_layer_ids:
            shared = k, v
        repeats = args.num_attention_heads // args.num_key_value_heads
        k, v = np.repeat(k, repeats, axis=1), np.repeat(v, repeats, axis=1)
        scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(args.head_dim)
        mask = causal
        if layer_type == "sliding_attention":
            mask = causal & (positions[:, None] - positions[None] < args.sliding_window)
        scores = np.where(mask, scores, -np.inf)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        out = (probs @ v).transpose(0, 2, 1, 3)
        if args.headwise_attn_output_gate:
            gates = 1 / (1 + np.exp(-linear(h, attn + ".g_proj")))
            out *= gates[..., None]
        x = x + linear(out.reshape(batch, length, -1), attn + ".out_proj")
        h = norm(x, prefix + ".post_attention_layernorm")
        gate = linear(h, prefix + ".mlp.gate_proj")
        gelu = gate * (1 + np.vectorize(math.erf)(gate / math.sqrt(2))) / 2
        x = x + linear(
            gelu * linear(h, prefix + ".mlp.up_proj"), prefix + ".mlp.down_proj"
        )
    h = norm(x, "model.norm")
    return linear(h, "model.embedding" if args.tie_word_embeddings else "lm_head")


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"kv_source_layer_ids": None},
        {"headwise_attn_output_gate": False},
        {"tie_word_embeddings": True, "num_key_value_heads": 2},
    ],
)
def test_forward_matches_independent_numpy_reference(overrides, device):
    mx.random.seed(17)
    model = Model(make_args(**overrides))
    tokens = mx.array([[1, 2, 3, 4, 5, 6], [3, 1, 6, 2, 4, 7]])
    actual = model(tokens)
    mx.eval(actual)
    actual, expected = np.array(actual), numpy_reference(model, tokens)
    if device == "cpu":
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    else:
        # This Metal runtime's FP32 matmul differs from CPU BLAS by ~1e-3
        # on these small shapes. Keep the strict FP64 check on CPU and bound
        # both normalized and worst-case error for the 12-layer GPU forward.
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.005
        assert np.max(np.abs(actual - expected)) < 0.05


@pytest.mark.parametrize("chunks", [[1] * 15, [2, 7, 1, 5], [9, 4, 2]])
def test_cached_chunks_match_full_forward_across_swa_wrap(chunks, device):
    mx.random.seed(42)
    model = Model(make_args())
    tokens = mx.array([[i % 31 for i in range(15)]])
    expected = model(tokens)
    caches = model.make_cache()
    outputs = []
    start = 0
    for length in chunks:
        output = model(tokens[:, start : start + length], cache=caches)
        mx.eval(output)
        outputs.append(output)
        start += length
    actual = mx.concatenate(outputs, axis=1)
    tolerance = 3e-5 if device == "cpu" else 1e-2
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=tolerance, rtol=tolerance
    )
    # Shared consumers do not advance the owner a second time.
    assert all(c.offset == 15 for c in caches)
    assert len({id(c) for c in caches}) == len(caches)


def test_real_layer_schedule_omits_only_five_consumer_projections_and_caches():
    model = Model(make_args(num_hidden_layers=40, kv_source_layer_ids=19))
    consumers = [
        i for i, layer in enumerate(model.layers) if layer.self_attn.is_ced_consumer
    ]
    assert consumers == [23, 27, 31, 35, 39]
    assert len(model.make_cache()) == 35
    params = dict(tree_flatten(model.parameters()))
    assert params["model.layers.19.self_attn.q_down_proj.weight"].shape == (8, 16)
    assert params["model.layers.19.self_attn.q_up_proj.weight"].shape == (32, 8)
    assert params["model.layers.19.self_attn.out_proj.weight"].shape == (16, 32)
    assert params["lm_head.weight"].shape == (32, 16)
    for i in range(40):
        assert (f"model.layers.{i}.self_attn.k_proj.weight" in params) == (
            i not in consumers
        )
        assert (f"model.layers.{i}.self_attn.v_proj.weight" in params) == (
            i not in consumers
        )


def test_ced_is_causal():
    model = Model(make_args())
    a = model(mx.array([[1, 2, 3, 4, 5, 6]]))
    b = model(mx.array([[1, 2, 3, 9, 8, 7]]))
    np.testing.assert_allclose(np.array(a[:, :3]), np.array(b[:, :3]), atol=1e-6)


def test_native_cache_save_and_reload_preserve_ced(tmp_path):
    model = Model(make_args())
    caches = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3, 4, 5, 6]]), cache=caches))
    path = str(tmp_path / "cache.safetensors")
    save_prompt_cache(path, caches)
    restored = load_prompt_cache(path)
    a = model(mx.array([[7, 8]]), cache=caches)
    b = model(mx.array([[7, 8]]), cache=restored)
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-6)


def test_sanitize_accepts_old_consumer_kv_but_keeps_owner_weights_strict():
    model = Model(make_args())
    weights = dict(tree_flatten(model.parameters()))
    weights["model.embed_tokens.weight"] = weights.pop("model.embedding.weight")
    weights["model.layers.7.self_attn.k_proj.weight"] = mx.zeros((8, 16))
    weights["model.layers.7.self_attn.v_proj.weight"] = mx.zeros((8, 16))
    weights["draft_model.fc.weight"] = mx.zeros((8, 8))
    weights["draft_model.layers.0.self_attn.q_k_v_proj.weight"] = mx.zeros((32, 16))
    weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = mx.zeros((4,))
    sanitized = model.sanitize(weights)
    model.load_weights(list(sanitized.items()), strict=True)
    sanitized.pop("model.layers.3.self_attn.k_proj.weight")
    with pytest.raises(ValueError, match="Missing"):
        model.load_weights(list(sanitized.items()), strict=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"q_lora_rank": None}, "q_lora_rank"),
        ({"q_lora_rank": True}, "q_lora_rank"),
        ({"q_lora_rank": 0}, "q_lora_rank"),
        ({"kv_source_layer_ids": 0}, "producer"),
        ({"kv_source_layer_ids": 12}, "zero-based"),
        ({"kv_source_layer_ids": True}, "zero-based"),
        ({"ced_prefill_keep_tokens": 3}, "SWA window"),
        ({"ced_prefill_keep_tokens": True}, "SWA window"),
        ({"ced_prefill_keep_tokens": 4, "kv_source_layer_ids": None}, "CED producer"),
        ({"layer_types": ["qwen_sparse_attention"] * 12}, "Unsupported"),
        ({"num_key_value_heads": 3}, "divisible"),
        ({"layer_types": ["full_attention"]}, "length"),
    ],
)
def test_invalid_or_unsupported_config_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_args(**overrides)


def test_tail_prefill_defaults_to_window_with_ced_consumers():
    assert make_args().ced_prefill_keep_tokens == 4
    assert make_args(kv_source_layer_ids=None).ced_prefill_keep_tokens is None
    assert make_args(kv_source_layer_ids=11).ced_prefill_keep_tokens is None
    assert make_args(ced_prefill_keep_tokens=None).ced_prefill_keep_tokens is None


def test_tail_prefill_matches_explicit_cropped_reference(device):
    mx.random.seed(17)
    # The explicit mask and additional RoPE in the reference can round
    # differently on Metal for these small FP32 shapes.
    tolerance = 1e-5 if device == "cpu" else 3e-3
    model = Model(make_args(ced_prefill_keep_tokens=4))
    original, tail = model.make_cache(), model.make_cache()
    prefix = mx.array([[1, 2, 3, 4, 5]])
    block = mx.array([[6, 7, 8, 9, 10, 11, 12]])
    mx.eval(model(prefix, cache=original), model(prefix, cache=tail))

    actual = model.model(block, cache=tail, prefill_keep_tokens=4)
    hidden = model.model.embedding(block)
    owners = iter(zip(original, tail, strict=True))
    for layer in model.layers[:4]:
        old, new = next(owners)
        hidden, donor = layer(hidden, cache=old)
        for expected, cropped in zip(old.state, new.state, strict=True):
            np.testing.assert_array_equal(np.array(cropped), np.array(expected))

    hidden = hidden[:, -4:]
    mask = mx.arange(12)[None, :] <= mx.arange(8, 12)[:, None]
    donor = donor._replace(offset=8, mask=mask)
    for layer in model.layers[4:]:
        if layer.is_sliding:
            hidden, state = layer(hidden)
            _, new = next(owners)
            assert new.offset == 12 and new.keys.shape[2] == 4
            # Both Q and K were rotated from zero in the reference layer.
            # Rotate the stored K once more to compare its absolute phase.
            rows = state.keys.transpose(0, 2, 1, 3).reshape(4, 1, 1, 8)
            keys = layer.self_attn.rope(rows, offset=8).reshape(1, 1, 4, 8)
            np.testing.assert_allclose(np.array(new.keys), np.array(keys), atol=tolerance)
            np.testing.assert_allclose(np.array(new.values), np.array(state.values), atol=tolerance)
        else:
            hidden, _ = layer(hidden, shared_kv=donor)
    np.testing.assert_allclose(np.array(actual), np.array(model.model.norm(hidden)), atol=tolerance)

    late_caches = tail[4:]
    prior_keys = [np.array(cache.keys[..., -2:, :]) for cache in late_caches]
    model.prefill_cache(mx.array([[13, 14]]), tail)
    for cache, expected in zip(late_caches, prior_keys, strict=True):
        np.testing.assert_array_equal(np.array(cache.keys[..., :2, :]), expected)

    mx.eval(model(mx.array([[15]]), cache=tail))
    assert all(cache.offset == 15 for cache in tail)
    assert all(cache.keys.shape[2] == 4 for cache in late_caches)


def test_tail_prefill_short_chunks_preserve_contiguous_swa_prefix(device):
    model = Model(make_args(ced_prefill_keep_tokens=4))
    original, tail = model.make_cache(), model.make_cache()
    for chunk in ([[1, 2]], [[3, 4]]):
        inputs = mx.array(chunk)
        assert model(inputs, cache=original).shape[1] == 2
        model.prefill_cache(inputs, tail)

    inputs = mx.array([[5]])
    expected, actual = model(inputs, cache=original), model(inputs, cache=tail)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)


@pytest.mark.parametrize(
    ("chunk_size", "prompt_length", "expected_start"),
    [(6, 19, 12), (6, 17, 12), (6, 16, 6), (6, 14, 6), (3, 16, 0)],
)
def test_deferring_tail_prefill_preserves_final_logits_and_kv(
    chunk_size, prompt_length, expected_start, device,
):
    model = Model(make_args())
    original, deferred = model.make_cache(), model.make_cache()
    inputs = mx.arange(1, prompt_length + 3)[None]
    first_tail = model.prefill_skip_tail_before(prompt_length, chunk_size)
    assert first_tail == expected_start
    for start in range(0, prompt_length - 1, chunk_size):
        block = inputs[:, start : min(start + chunk_size, prompt_length - 1)]
        model.prefill_cache(block, original)
        mx.eval(cache_arrays(original))
        active = model.prefill_cache(block, deferred, prefix_only=start < first_tail)
        if active is not None:
            assert len(active) == model.args.kv_source_layer_ids + 1
        mx.eval(cache_arrays(deferred if active is None else active))

    for i in range(prompt_length - 1, prompt_length + 2):
        token = inputs[:, i : i + 1]
        expected = model(token, cache=original)
        actual = model(token, cache=deferred)
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
        assert [cache.meta_state for cache in deferred] == [cache.meta_state for cache in original]
        for expected_kv, actual_kv in zip(cache_arrays(original), cache_arrays(deferred), strict=True):
            assert mx.array_equal(actual_kv, expected_kv).item()


def test_cache_length_is_validated():
    model = Model(make_args())
    with pytest.raises(ValueError, match="owner cache entries"):
        model(mx.array([[1]]), cache=[None] * 12)


def test_input_embeddings_follow_the_same_path():
    model = Model(make_args())
    tokens = mx.array([[1, 2, 3]])
    expected = model(tokens)
    actual = model(None, input_embeddings=model.model.embedding(tokens))
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)


def test_left_padded_batch_prefill_and_decode_match_individual_requests(device):
    model = Model(make_args())
    padding = [2, 0]
    caches = [
        BatchRotatingKVCache(model.args.sliding_window, padding)
        if model.layers[i].is_sliding
        else BatchKVCache(padding)
        for i in model.model.cache_layer_ids
    ]
    batch = model(mx.array([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]]), cache=caches)
    expected = mx.concatenate(
        [
            model(mx.array([[1, 2, 3]]))[:, -1:],
            model(mx.array([[4, 5, 6, 7, 8]]))[:, -1:],
        ],
        axis=0,
    )
    tolerance = 3e-5 if device == "cpu" else 1e-2
    np.testing.assert_allclose(
        np.array(batch[:, -1:]), np.array(expected), atol=tolerance, rtol=tolerance
    )
    decoded = model(mx.array([[9], [10]]), cache=caches)
    expected = mx.concatenate(
        [
            model(mx.array([[1, 2, 3, 9]]))[:, -1:],
            model(mx.array([[4, 5, 6, 7, 8, 10]]))[:, -1:],
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        np.array(decoded), np.array(expected), atol=tolerance, rtol=tolerance
    )


def test_forward_and_loading_do_not_call_spark2_5(monkeypatch):
    from spark_mlx_llm import spark2_5

    def forbidden(*args, **kwargs):
        raise AssertionError("Spark3 called a Spark2_5 model implementation")

    for name in ("Model", "TransformerBlock", "MLP", "Attention"):
        monkeypatch.setattr(getattr(spark2_5, name), "__call__", forbidden)
    monkeypatch.setattr(spark2_5.Model, "sanitize", forbidden)
    monkeypatch.setattr(spark2_5.Model, "load_weights", forbidden)
    model = Model(make_args())
    weights = model.sanitize(dict(tree_flatten(model.parameters())))
    model.load_weights(list(weights.items()))
    logits = model(mx.array([[1, 2, 3]]), cache=model.make_cache())
    assert logits.shape == (1, 3, 32)
    assert bool(mx.all(mx.isfinite(logits)))


def test_w8a8_configuration_falls_back_to_native_mlx_weights(monkeypatch, device):
    monkeypatch.setattr(spark3, "_w8a8_operation", lambda: None)
    args = make_args(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        head_dim=16,
        q_lora_rank=32,
        quantization_config={
            "model.layers.0.self_attn.out_proj": {
                "bits": 8,
                "mode": "affine",
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            },
        },
    )
    model = Model(args)
    assert type(model.layers[0].self_attn.out_proj) is spark3._W8A8OutputProjection
    model, _ = quantize_model(model, asdict(args), group_size=32, bits=8)
    assert type(model.layers[0].self_attn.out_proj) is spark3.nn.QuantizedLinear
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    full = model(tokens)
    caches = model.make_cache()
    cached = mx.concatenate(
        [
            model(tokens[:, :3], cache=caches),
            model(tokens[:, 3:], cache=caches),
        ],
        axis=1,
    )
    tolerance = 3e-5 if device == "cpu" else 1e-2
    np.testing.assert_allclose(
        np.array(full), np.array(cached), atol=tolerance, rtol=tolerance
    )
