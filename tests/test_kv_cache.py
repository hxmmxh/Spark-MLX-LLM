"""Checks for quantized KV in the Hadamard basis."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache
from test_merged_interfaces import save_and_load
from test_model import make_args as spark2_args
from test_spark3 import make_args as spark3_args

from spark_mlx_llm import GenerationConfig, generate
from spark_mlx_llm.benchmark.bench import kv_generation_options, run_generation
from spark_mlx_llm.cli import build_parser
from spark_mlx_llm.kv_cache import (
    HadamardQuantizedKVCache,
    make_prompt_cache,
    maybe_quantize_kv_cache,
)
from spark_mlx_llm.ops import sdpa
from spark_mlx_llm.ops.sdpa import SDPAConfig, dispatch_sdpa
from spark_mlx_llm.spark2_5 import Model as Spark2Model
from spark_mlx_llm.spark3 import Model as Spark3Model


@pytest.mark.parametrize(
    ("dim", "group_size"), [(96, 32), (256, 32), (256, 64), (256, 128)]
)
def test_hadamard_kv_and_gqa_match_an_independent_basis(dim, group_size):
    rng = np.random.default_rng(11)
    queries = rng.normal(size=(1, 4, 3, dim)).astype(np.float32)
    keys = rng.normal(size=(1, 2, 17, dim)).astype(np.float32)
    values = rng.normal(size=keys.shape).astype(np.float32)
    keys[..., 0] *= 5
    values[..., 0] *= 5
    block = 32 if dim == 96 else 256
    basis = np.ones((1, 1), dtype=np.float32)
    while basis.shape[0] < block:
        basis = np.block([[basis, basis], [basis, -basis]])
    basis = np.kron(np.eye(dim // block, dtype=np.float32), basis / np.sqrt(block))

    cache = HadamardQuantizedKVCache(group_size)
    packed_keys, packed_values = cache.update_and_fetch(mx.array(keys), mx.array(values))
    assert packed_keys[0].dtype == packed_values[0].dtype == mx.uint32
    assert packed_keys[0].shape[-1] == dim // 4

    def dequantize(packed):
        return mx.dequantize(*packed, group_size=group_size, bits=8)

    decoded_keys = np.array(dequantize(packed_keys))
    decoded_values = np.array(dequantize(packed_values))
    np.testing.assert_allclose(decoded_keys, keys @ basis, atol=0.025, rtol=0)
    np.testing.assert_allclose(decoded_values, values @ basis, atol=0.025, rtol=0)

    q = mx.array(queries)
    scale = dim**-0.5
    actual = dispatch_sdpa(
        q, packed_keys, packed_values, cache=cache, scale=scale, mask="causal",
        config=SDPAConfig(fused_quantized_sdpa=False),
    )
    expected = mx.fast.scaled_dot_product_attention(
        q,
        mx.array(decoded_keys @ basis, dtype=mx.float32),
        mx.array(decoded_values @ basis, dtype=mx.float32),
        scale=scale,
        mask="causal",
    )
    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.002)


def test_hadamard_kv_conversion_preserves_history_and_non_kv_state():
    full, sliding, positions = KVCache(), RotatingKVCache(max_size=8), ArraysCache(1)
    rng = np.random.default_rng(21)
    data = mx.array(rng.normal(size=(1, 1, 258, 32)).astype(np.float32))
    full.update_and_fetch(data[..., :8, :], data[..., :8, :])
    full.trim(1)
    sliding.update_and_fetch(data[..., :7, :], data[..., :7, :])
    positions[0] = mx.array([3])
    caches = [full, sliding, positions]
    make_prompt_cache(None, prompt_cache=caches, kv_hadamard=True)

    maybe_quantize_kv_cache(caches, 8, 32, 8, kv_hadamard=True)
    assert caches[0] is full
    maybe_quantize_kv_cache(caches, 7, 32, 8, kv_hadamard=True)
    rotated = caches[0]
    assert isinstance(rotated, HadamardQuantizedKVCache)
    assert rotated.offset == 7
    history = np.array(rotated.state[0][0], copy=True)
    rotated.update_and_fetch(data[..., 7:, :], data[..., 7:, :])
    maybe_quantize_kv_cache(caches, 7, 32, 8, kv_hadamard=True)
    assert caches[0] is rotated
    assert rotated.offset == 258
    np.testing.assert_array_equal(rotated.state[0][0][..., :7, :], history)
    assert caches[1].offset == 7
    assert caches[1].keys is sliding.keys
    assert caches[2] is positions and positions[0].item() == 3


@pytest.mark.parametrize(
    ("gqa", "batch", "kv_heads", "dtype"),
    [(16, 2, 2, mx.bfloat16), (32, 1, 1, mx.float16)],
)
@pytest.mark.parametrize("hadamard", [False, True])
def test_wide_gqa_kv8_tensorops_preserves_attention(
    gqa, batch, kv_heads, dtype, hadamard, monkeypatch,
):
    if not sdpa._wide_kv8_sdpa.available():
        pytest.skip("M5 TensorOps SDPA is unavailable")
    with mx.stream(mx.gpu):
        mx.random.seed(31)
        dim, length = 256, 1057
        q = mx.random.normal((batch, kv_heads * gqa, 1, dim), dtype=dtype)
        shape = (batch, kv_heads, length, dim)
        k, v = (mx.random.normal(shape, dtype=dtype) for _ in range(2))
        cache = (
            HadamardQuantizedKVCache(32)
            if hadamard
            else QuantizedKVCache(group_size=32, bits=8)
        )
        keys, values = cache.update_and_fetch(k, v)
        original_q = mx.array(q)
        options = {"cache": cache, "scale": dim**-0.5, "mask": "causal"}
        reference = dispatch_sdpa(
            mx.array(q), keys, values, **options,
            config=SDPAConfig(fused_quantized_sdpa=False),
        )
        calls = []
        decode = sdpa._wide_kv8_sdpa.decode

        def record(queries, keys, values, scale):
            calls.append(queries.shape)
            return decode(queries, keys, values, scale)

        monkeypatch.setattr(sdpa._wide_kv8_sdpa, "decode", record)
        actual = dispatch_sdpa(q, keys, values, **options)
        assert calls == [q.shape]
        assert actual.shape == q.shape and actual.dtype == q.dtype
        assert mx.array_equal(q, original_q).item()
        np.testing.assert_allclose(
            actual.astype(mx.float32), reference.astype(mx.float32), atol=0.008, rtol=0.01
        )

        mask = (mx.arange(length) % 2 == 0)[None]
        dispatch_sdpa(mx.array(q), keys, values, **{**options, "mask": mask})
        assert calls == [q.shape]


@pytest.mark.parametrize("architecture", ["spark2_5", "spark3"])
def test_hadamard_kv_through_cli_config_and_public_generate(
    architecture, tmp_path, monkeypatch,
):
    mx.random.seed(21)
    model = (
        Spark2Model(spark2_args(head_dim=32)) if architecture == "spark2_5"
        else Spark3Model(spark3_args(head_dim=32))
    )
    model, tokenizer = save_and_load(model, tmp_path, monkeypatch)
    prompt, suffix = mx.array([[1, 2, 3]]), mx.array([[4, 5, 6]])
    reference_cache = model.make_cache()
    rotated_cache = make_prompt_cache(model, kv_hadamard=True)
    model(prompt, cache=reference_cache)
    model(prompt, cache=rotated_cache)
    maybe_quantize_kv_cache(rotated_cache, 3, 32, 8, kv_hadamard=True)
    reference = model(suffix, cache=reference_cache)
    actual = model(suffix, cache=rotated_cache)
    np.testing.assert_allclose(actual, reference, atol=0.02, rtol=0.02)

    parser = build_parser()
    args = parser.parse_args([
        "generate", "--model", str(tmp_path), "--kv-bits", "8", "--kv-hadamard",
        "--quantized-kv-start", "3", "--prefill-step-size", "3", "--max-tokens", "3",
        "--temperature", "0",
    ])
    config = GenerationConfig.from_args(parser, args)
    result = generate(model, tokenizer, [1, 2, 3, 4, 5, 6, 7], config=config)
    assert len(result.token_ids) == 3
    assert result.effective_config["kv_hadamard"] is True
    assert result.token_ids == list(generate(
        model, tokenizer, [1, 2, 3, 4, 5, 6, 7], config=config, stream=True
    ))[-1].result.token_ids

    benchmark_result = run_generation(
        model, tokenizer, [1, 2, 3, 4, 5, 6, 7], 3, 3, **kv_generation_options(args)
    )
    assert benchmark_result.effective_config["kv_hadamard"] is True

    with pytest.raises(ValueError, match="requires kv_bits=8"):
        config.overlay({"kv_bits": 4})
