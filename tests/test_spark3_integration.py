"""Opt-in checks against the actual BF16 Spark3 checkpoint."""

import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from spark_mlx_llm.loader import load
from spark_mlx_llm.spark3 import Model, ModelArgs

MODEL_PATH = os.environ.get("SPARK3_MODEL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not MODEL_PATH, reason="SPARK3_MODEL is not set"),
]


def test_real_checkpoint_headers_match_model():
    path = Path(MODEL_PATH)
    args = ModelArgs.from_dict(json.loads((path / "config.json").read_text()))
    expected = {
        name: list(weight.shape)
        for name, weight in tree_flatten(Model(args).parameters())
    }
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    actual = {}
    for shard in set(index.values()):
        file = path / shard
        with file.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        tensors = {
            name: spec for name, spec in header.items() if name != "__metadata__"
        }
        assert file.stat().st_size == 8 + header_length + max(
            spec["data_offsets"][1] for spec in tensors.values()
        ), f"Incomplete checkpoint shard: {file.name}"
        actual.update({name: spec["shape"] for name, spec in tensors.items()})
    assert actual == expected
    assert set(index) == set(actual)


def test_real_checkpoint_strict_load_prefill_and_decode():
    mx.set_cache_limit(64 * 1024 * 1024)
    model, tokenizer, config = load(
        MODEL_PATH, strict=True, lazy=True, return_config=True
    )
    assert config["model_type"] == "spark3"
    tokens = tokenizer.encode("你好")
    inputs = mx.array([tokens])
    caches = model.make_cache()
    full = model(inputs, cache=caches)
    mx.eval(full)
    assert full.shape == (1, len(tokens), config["vocab_size"])
    assert bool(mx.all(mx.isfinite(full)))
    assert len(caches) == len(model.model.cache_layer_ids)
    assert all(c.offset == len(tokens) for c in caches)

    next_token = mx.argmax(full[:, -1], axis=-1)[:, None]
    decoded = model(next_token, cache=caches)
    mx.eval(decoded)
    assert bool(mx.all(mx.isfinite(decoded)))
    assert all(c.offset == len(tokens) + 1 for c in caches)
    # BF16 GEMV (decode) and GEMM (prefill) round differently over 40 layers.
    # Compare both the selected token and the entire output distribution;
    # pointwise relative tolerances are unstable for logits near zero.
    complete = mx.concatenate([inputs, next_token], axis=1)
    uncached = model(complete)[:, -1:]
    mx.eval(uncached)
    cached_logits = np.array(decoded.astype(mx.float32)).astype(np.float64).ravel()
    full_logits = np.array(uncached.astype(mx.float32)).astype(np.float64).ravel()
    relative_l2 = np.linalg.norm(cached_logits - full_logits) / np.linalg.norm(
        full_logits
    )

    def log_softmax(logits):
        shifted = logits - logits.max()
        return shifted - np.log(np.exp(shifted).sum())

    log_p, log_q = log_softmax(full_logits), log_softmax(cached_logits)
    kl = float(np.sum(np.exp(log_p) * (log_p - log_q)))
    greedy_cached, greedy_full = int(cached_logits.argmax()), int(full_logits.argmax())
    print(
        json.dumps(
            {
                "cached_greedy_token": greedy_cached,
                "full_greedy_token": greedy_full,
                "relative_logit_l2": float(relative_l2),
                "max_absolute_logit_difference": float(
                    np.max(np.abs(cached_logits - full_logits))
                ),
                "distribution_kl": kl,
                "owner_caches": len(caches),
            }
        )
    )
    assert greedy_cached == greedy_full
    assert relative_l2 < 0.02
    assert kl < 0.002
