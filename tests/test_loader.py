import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx_lm.utils import load_model, quantize_model

from spark_mlx_llm import _target_loader as loader
from spark_mlx_llm.spark2_5 import Model as Spark2_5Model
from spark_mlx_llm.spark3 import Model, ModelArgs


def test_tokenizer_loading_delegates_to_mlx_lm(monkeypatch):
    model_path = Path("spark2_5")
    tokenizer_config = {"trust_remote_code": True}
    expected = object()

    def fake_load_tokenizer(path, *, tokenizer_config_extra, eos_token_ids):
        assert path == model_path
        assert tokenizer_config_extra == tokenizer_config
        assert eos_token_ids == 1
        return expected

    monkeypatch.setattr(loader, "load_tokenizer", fake_load_tokenizer)

    tokenizer = loader._load_tokenizer(model_path, tokenizer_config, 1)

    assert tokenizer is expected


@pytest.mark.parametrize(
    "model_type, expected",
    [
        ("spark2_5", Spark2_5Model),
        ("spark3", Model),
    ],
)
def test_loader_dispatches_by_checkpoint_type(model_type, expected):
    model_class, _ = loader._get_model_classes({"model_type": model_type})
    assert model_class is expected


def test_loader_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="model_type"):
        loader._get_model_classes({"model_type": "other"})


@pytest.mark.parametrize("quantized", [False, True])
def test_spark3_strict_checkpoint_roundtrip(tmp_path, monkeypatch, quantized):
    args = ModelArgs(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=64,
        sliding_window=4,
        q_lora_rank=32,
        kv_source_layer_ids=3,
    )
    model = Model(args)
    config = asdict(args)
    if quantized:
        model, config = quantize_model(model, config, group_size=32, bits=4)
        assert isinstance(model.layers[0].self_attn.g_proj, nn.Linear)
        assert isinstance(model.layers[0].self_attn.q_down_proj, nn.QuantizedLinear)
    model.save_weights(str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config))

    tokenizer = object()
    monkeypatch.setattr(loader, "_load_tokenizer", lambda *args: tokenizer)
    loaded, loaded_tokenizer = loader.load(tmp_path, strict=True)
    assert loaded_tokenizer is tokenizer
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    expected = model(tokens)
    caches = loaded.make_cache()
    actual = mx.concatenate(
        [
            loaded(tokens[:, :3], cache=caches),
            loaded(tokens[:, 3:], cache=caches),
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=3e-5, rtol=3e-5
    )

    # MLX LM's native resolver uses the same model after plugin registration.
    from spark_mlx_llm.registration import register_model

    register_model()
    native, _ = load_model(tmp_path, strict=True)
    assert isinstance(native, Model)
