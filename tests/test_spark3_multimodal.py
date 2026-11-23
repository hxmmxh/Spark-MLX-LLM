"""Numerical vision/mRoPE checks and CED multimodal cache integration."""

import json
import os
import struct
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache, trim_prompt_cache

from spark_mlx_llm._target_loader import _validate_checkpoint_files
from spark_mlx_llm.loader import load
from spark_mlx_llm.mrope import MultimodalRoPE, build_mrope_positions
from spark_mlx_llm.multimodal import _PositionedModel
from spark_mlx_llm.processing_spark3 import Spark3Processor
from spark_mlx_llm.spark3 import Model, ModelArgs
from spark_mlx_llm.spark3_vision import VisionTransformer


def tiny_args(**overrides):
    fields = {
        "hidden_size": 24,
        "intermediate_size": 32,
        "num_hidden_layers": 8,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 12,
        "vocab_size": 64,
        "sliding_window": 4,
        "q_lora_rank": 8,
        "kv_source_layer_ids": 3,
        "tie_word_embeddings": True,
        "image_token_id": 60,
        "rope_parameters": {
            "sliding_attention": {
                "rope_theta": 10000,
                "mrope_section": [2, 2, 2],
                "mrope_interleaved": True,
            },
            "full_attention": {
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 1],
                "mrope_interleaved": True,
            },
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 28,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 8,
            "base_grid": [2, 4],
        },
    }
    fields.update(overrides)
    return ModelArgs(**fields)


@pytest.fixture(params=["cpu", "metal"])
def device(request):
    if request.param == "metal" and not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    old = mx.default_device()
    mx.set_default_device(mx.cpu if request.param == "cpu" else mx.gpu)
    yield request.param
    mx.set_default_device(old)


@pytest.mark.parametrize("interleaved", [False, True])
def test_mrope_matches_independent_numpy_and_preserves_tail(device, interleaved):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2, 3, 5, 16)).astype(np.float32)
    positions = rng.integers(0, 30, size=(3, 2, 5), dtype=np.int32)
    section, dims = [2, 1, 3], 12
    rope = MultimodalRoPE(dims, 12345, section, interleaved)
    axes = (
        [
            1
            if i % 3 == 1 and i // 3 < section[1]
            else 2
            if i % 3 == 2 and i // 3 < section[2]
            else 0
            for i in range(6)
        ]
        if interleaved
        else np.repeat(np.arange(3), section)
    )
    expected = x.astype(np.float64)
    for i, axis in enumerate(axes):
        angle = positions[axis] / 12345 ** (i / 6)
        a, b = x[..., i], x[..., i + 6]
        expected[..., i] = a * np.cos(angle)[:, None] - b * np.sin(angle)[:, None]
        expected[..., i + 6] = b * np.cos(angle)[:, None] + a * np.sin(angle)[:, None]
    actual = np.array(rope(mx.array(x), mx.array(positions)))
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(actual[..., dims:], x[..., dims:])


def test_nested_config_roundtrip_and_legacy_9b():
    args = tiny_args()
    text = asdict(args)
    vision = text.pop("vision_config")
    original = {"model_type": "spark3", "text_config": text, "vision_config": vision}
    loaded = ModelArgs.from_dict(original)
    assert asdict(loaded) == asdict(args)
    assert original["vision_config"] == vision
    assert ModelArgs.from_dict(original | {"hidden_size": 24}).hidden_size == 24
    source_list = deepcopy(original)
    source_list["text_config"]["kv_source_layer_ids"] = [3]
    assert ModelArgs.from_dict(source_list).kv_source_layer_ids == 3
    ordinary = ModelArgs(q_lora_rank=1024, kv_source_layer_ids=19)
    assert len(Model(ordinary).make_cache()) == 35
    assert not ordinary.uses_mrope


@pytest.mark.parametrize(
    "change, match",
    [
        ({"mrope_section": [1, 1, 1]}, "mrope_section"),
        ({"mrope_interleaved": "true"}, "boolean"),
        ({"rope_type": "yarn"}, "RoPE"),
        ({"rope_theta": float("nan")}, "rope_theta"),
    ],
)
def test_mrope_rejects_bad_geometry(change, match):
    args = asdict(tiny_args())
    args["rope_parameters"]["sliding_attention"].update(change)
    with pytest.raises(ValueError, match=match):
        ModelArgs.from_dict(args)


def test_image_positions_include_boundaries_and_decode_delta():
    positions = np.array(
        build_mrope_positions(11, [[1, 4, 4], [1, 2, 4]], [(2, 5), (7, 8)])
    )[:, 0]
    np.testing.assert_array_equal(positions[:, :2], [[0, 1]] * 3)
    np.testing.assert_array_equal(
        positions[:, 2:6], [[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]]
    )
    np.testing.assert_array_equal(positions[:, 7:9], [[5, 5], [5, 5], [5, 6]])
    np.testing.assert_array_equal(positions[:, 9:], [[7, 8]] * 3)
    with pytest.raises(ValueError, match="overlap"):
        build_mrope_positions(8, [[1, 2, 2], [1, 2, 2]], [(2, 2), (2, 2)])


def _numpy_vision(model, pixels, grids):
    """Independent float64 NumPy implementation of the SGLang ViT formulas."""
    from math import erf

    args = model.args
    weights = {
        k: np.array(v.astype(mx.float32)).astype(np.float64)
        for k, v in tree_flatten(model.parameters())
    }

    def linear(x, name):
        return x @ weights[name + ".weight"].T + weights[name + ".bias"]

    def norm(x, name):
        centered = x - x.mean(-1, keepdims=True)
        return (
            centered
            / np.sqrt((centered**2).mean(-1, keepdims=True) + args.layer_norm_eps)
            * weights[name + ".weight"]
            + weights[name + ".bias"]
        )

    def gelu(x, approximate=False):
        if approximate:
            return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        return x * 0.5 * (1 + np.vectorize(erf)(x / np.sqrt(2)))

    def grouped(x, h, w):
        return np.concatenate(
            [
                x[r : r + 2, c : c + 2].reshape(4, -1)
                for r in range(0, h, 2)
                for c in range(0, w, 2)
            ]
        )

    def position_table(h, w):
        table = weights["pos_embed.weight"].reshape(*args.base_grid, -1)

        # Direct separable cubic convolution, independently of the MLX gather path.
        def cubic(d):
            d = abs(d)
            return (
                (1.25 * d - 2.25) * d**2 + 1
                if d <= 1
                else ((-0.75 * d + 3.75) * d - 6) * d + 3
                if d < 2
                else 0
            )

        result = np.zeros((h, w, args.hidden_size))
        bh, bw = args.base_grid
        for r in range(h):
            y = (r + 0.5) * bh / h - 0.5
            for c in range(w):
                x = (c + 0.5) * bw / w - 0.5
                for i in range(math_floor(y) - 1, math_floor(y) + 3):
                    for j in range(math_floor(x) - 1, math_floor(x) + 3):
                        result[r, c] += (
                            table[min(max(i, 0), bh - 1), min(max(j, 0), bw - 1)]
                            * cubic(y - i)
                            * cubic(x - j)
                        )
        return grouped(result, h, w)

    from math import floor as math_floor

    hidden = linear(
        pixels.reshape(
            -1, 3, args.temporal_patch_size, args.patch_size, args.patch_size
        )
        .mean(2)
        .reshape(len(pixels), -1),
        "patch_embed.proj",
    )
    hidden += np.concatenate([position_table(h, w) for _, h, w in grids])
    coordinates = np.concatenate(
        [
            grouped(
                np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), -1),
                h,
                w,
            )
            for _, h, w in grids
        ]
    )
    d = args.hidden_size // args.num_heads
    angle = (
        coordinates[..., None]
        * args.rope_theta ** (-np.arange(0, d // 2, 2) / (d // 2))
        * args.rope_alpha
    ).reshape(len(pixels), 1, -1)
    for i in range(args.depth):
        prefix = f"blocks.{i}"
        qkv = linear(norm(hidden, prefix + ".norm1"), prefix + ".attn.qkv").reshape(
            len(pixels), 3, args.num_heads, d
        )
        q, k, v = (qkv[:, j] for j in range(3))

        def rotate(x):
            return np.stack(
                [
                    x[..., ::2] * np.cos(angle) - x[..., 1::2] * np.sin(angle),
                    x[..., 1::2] * np.cos(angle) + x[..., ::2] * np.sin(angle),
                ],
                -1,
            ).reshape(x.shape)

        q, k = rotate(q), rotate(k)
        output, start = [], 0
        for _, h, w in grids:
            end = start + h * w
            qs, ks, vs = (x[start:end].transpose(1, 0, 2) for x in (q, k, v))
            scores = qs @ ks.transpose(0, 2, 1) / np.sqrt(d)
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            output.append(
                ((scores / scores.sum(-1, keepdims=True)) @ vs)
                .transpose(1, 0, 2)
                .reshape(h * w, -1)
            )
            start = end
        hidden += linear(np.concatenate(output), prefix + ".attn.proj")
        hidden += linear(
            gelu(
                linear(norm(hidden, prefix + ".norm2"), prefix + ".mlp.linear_fc1"),
                True,
            ),
            prefix + ".mlp.linear_fc2",
        )
    hidden = norm(hidden, "merger.norm").reshape(-1, 4 * args.hidden_size)
    return linear(gelu(linear(hidden, "merger.linear_fc1")), "merger.linear_fc2")


@pytest.mark.parametrize("grids", [[[1, 4, 4], [1, 2, 4]], [[1, 2, 2], [1, 4, 2]]])
def test_vit_numpy_reference_resize_and_image_isolation(device, grids):
    mx.random.seed(12)
    model = VisionTransformer(tiny_args().vision_config)
    lengths = [h * w for _, h, w in grids]
    pixels = mx.random.normal((sum(lengths), 24))
    actual = np.array(model(pixels, grids))
    expected = _numpy_vision(model, np.array(pixels).astype(np.float64), grids)
    # Metal SDPA/matmuls use lower-precision accumulation than the CPU path.
    atol, rtol = (8e-6, 8e-5) if device == "cpu" else (2e-3, 3e-3)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    single = model(pixels[: lengths[0]], grids[:1])
    # A different GEMM batch shape can change Metal's accumulation strategy.
    isolation_atol = 2e-6 if device == "cpu" else atol
    isolation_rtol = 2e-6 if device == "cpu" else rtol
    np.testing.assert_allclose(
        np.array(single),
        actual[: lengths[0] // 4],
        atol=isolation_atol,
        rtol=isolation_rtol,
    )
    # Keep all GEMM shapes fixed: changing only image 2 must leave image 1 exact.
    changed = np.array(pixels)
    changed[lengths[0] :] = changed[lengths[0] :] * 31 + 17
    perturbed = np.array(model(mx.array(changed), grids))
    np.testing.assert_array_equal(
        perturbed[: lengths[0] // 4], actual[: lengths[0] // 4]
    )
    with pytest.raises(ValueError, match="still images"):
        model(pixels, [[2, 2, 6]])


def image_inputs():
    ids = mx.array([[1, 2, 60, 60, 60, 60, 3, 60, 60, 4, 5]])
    return ids, mx.random.normal((24, 24)), [[1, 4, 4], [1, 2, 4]], [(2, 5), (7, 8)]


def test_image_ced_chunked_prefill_decode_cache_roundtrip_and_isolation(
    device, tmp_path
):
    mx.random.seed(13)
    model = Model(tiny_args())
    ids, pixels, grids, offsets = image_inputs()
    prepared = model.prepare_inputs(ids, pixels, grids, offsets)
    direct = model(
        ids, pixel_values=pixels, image_grid_thw=grids, image_offsets=offsets
    )
    full = model(ids, **prepared)
    np.testing.assert_allclose(np.array(direct), np.array(full), atol=2e-5, rtol=2e-5)
    bound = _PositionedModel(model, prepared["positions"])
    caches = bound.make_cache()
    pieces = []
    # Split inside both images and wrap the SWA cache.
    for start, end in [(0, 3), (3, 5), (5, 8), (8, 11)]:
        pieces.append(
            bound(
                ids[:, start:end],
                cache=caches,
                input_embeddings=prepared["input_embeddings"][:, start:end],
            )
        )
    tolerance = 1e-4 if device == "cpu" else 3e-3
    np.testing.assert_allclose(
        np.array(mx.concatenate(pieces, 1)),
        np.array(full),
        atol=tolerance,
        rtol=tolerance,
    )
    assert caches[-1][0].tolist() == [-2]
    filename = str(tmp_path / "mm-cache.safetensors")
    save_prompt_cache(filename, caches)
    restored = load_prompt_cache(filename)
    assert restored[-1][0].tolist() == [-2]
    assert (
        trim_prompt_cache(restored, 1) == 0
    )  # Never trim into unknown image coordinates.
    # A second request must not overwrite the first request's position delta.
    model(mx.array([[7, 8]]), cache=model.make_cache())
    next_id = mx.array([[6]])
    decoded = model(next_id, cache=restored)
    complete_ids = mx.concatenate([ids, next_id], axis=1)
    complete_embeds = mx.concatenate(
        [prepared["input_embeddings"], model.model.embedding(next_id)], axis=1
    )
    positions = mx.concatenate(
        [prepared["positions"], mx.full((3, 1, 1), 9, dtype=mx.int32)], axis=-1
    )
    expected = model(
        complete_ids, input_embeddings=complete_embeds, positions=positions
    )[:, -1:]
    np.testing.assert_allclose(
        np.array(decoded), np.array(expected), atol=tolerance, rtol=tolerance
    )


def test_equal_text_axes_match_basic_rope(device):
    args = tiny_args(vision_config=None)
    ordinary = deepcopy(args)
    for params in ordinary.rope_parameters.values():
        params.pop("mrope_section")
        params.pop("mrope_interleaved")
    a, b = Model(args), Model(ordinary)
    b.load_weights(tree_flatten(a.parameters()), strict=True)
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    np.testing.assert_allclose(
        np.array(a(tokens)), np.array(b(tokens)), atol=8e-5, rtol=8e-5
    )


def test_multimodal_strict_loader_roundtrip(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from spark_mlx_llm import _target_loader as loader

    args = tiny_args()
    model = Model(args)
    model.save_weights(str(tmp_path / "model.safetensors"))
    text = asdict(args)
    vision = text.pop("vision_config")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "spark3", "text_config": text, "vision_config": vision}
        )
    )
    monkeypatch.setattr(
        loader, "_load_tokenizer", lambda *args: SimpleNamespace(eos_token_ids=set())
    )
    loaded, _ = load(tmp_path)
    ids, pixels, grids, offsets = image_inputs()
    kwargs = {"pixel_values": pixels, "image_grid_thw": grids, "image_offsets": offsets}
    np.testing.assert_allclose(
        np.array(loaded(ids, **kwargs)), np.array(model(ids, **kwargs)), atol=1e-6
    )
    source = dict(tree_flatten(model.parameters()))
    source["model.visual.patch_embed.proj.weight"] = mx.zeros((16, 3, 2, 2, 2))
    with pytest.raises(ValueError):
        model.load_weights(list(model.sanitize(source).items()), strict=True)


class TinyTokenizer:
    def convert_ids_to_tokens(self, token):
        return {60: "<image>", 61: "<start>", 62: "<end>"}.get(token)

    def convert_tokens_to_ids(self, token):
        return {"<image>": 60, "<start>": 61, "<end>": 62}.get(token)

    def encode(self, text, add_special_tokens=True):
        return ([0] if add_special_tokens else []) + [ord(c) % 50 + 1 for c in text]

    def build_inputs_with_special_tokens(self, ids):
        return [0] + ids


def test_processor_packing_normalization_expanded_adjacent_images_and_errors():
    Image = pytest.importorskip("PIL.Image")
    processor = Spark3Processor(TinyTokenizer(), tiny_args(), {"do_resize": False})
    values = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    image = Image.fromarray(values)
    result = processor("a<image><image>b", [image, image])
    assert result["image_offsets"] == [(2, 3), (4, 5)]
    first = values[:2, :2].transpose(2, 0, 1).astype(np.float32) / 255 * 2 - 1
    np.testing.assert_allclose(
        np.array(result["pixel_values"][0]).reshape(3, 2, 2, 2),
        np.repeat(first[:, None], 2, axis=1),
        atol=2e-7,
    )
    expanded = processor("a<image><image><image><image>b", [image, image])
    np.testing.assert_array_equal(
        np.array(result["input_ids"]), np.array(expanded["input_ids"])
    )
    assert result["image_offsets"] == expanded["image_offsets"]
    with pytest.raises(ValueError, match="count"):
        processor("a", [image])
    with pytest.raises(ValueError, match="require supplied"):
        processor("<image>")
    with pytest.raises(ValueError, match="still images"):
        processor("a", videos=[image])


def test_incomplete_safetensors_rejected_before_loading(tmp_path):
    header = json.dumps(
        {"x": {"data_offsets": [0, 100], "shape": [25], "dtype": "F32"}}
    ).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"short"
    )
    with pytest.raises(ValueError, match="Incomplete checkpoint shard"):
        _validate_checkpoint_files(tmp_path)


def test_native_generation_chunks_image_embeddings():
    Image = pytest.importorskip("PIL.Image")
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from spark_mlx_llm.multimodal import generate_multimodal

    vocab = {f"w{i}": i for i in range(60)} | {
        "<image>": 60,
        "<unk>": 61,
        "<eos>": 62,
        "<bos>": 63,
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
    model = Model(tiny_args())
    processor = Spark3Processor(
        TokenizerWrapper(tokenizer), model.args, {"do_resize": False}
    )
    text = generate_multimodal(
        model,
        processor,
        "w1<image>w2",
        [Image.new("RGB", (8, 8), "red")],
        max_tokens=3,
        prefill_step_size=2,
    )
    assert isinstance(text, str)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("SPARK3_MM_MODEL"), reason="SPARK3_MM_MODEL is not set"
)
def test_real_multimodal_checkpoint_parameter_names_and_shapes():
    path = Path(os.environ["SPARK3_MM_MODEL"])
    args = ModelArgs.from_dict(json.loads((path / "config.json").read_text()))
    model = Model(args)
    actual = {}
    for shard in path.glob("model*.safetensors"):
        with shard.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            specs = json.loads(stream.read(size))
        actual.update(
            {k: tuple(v["shape"]) for k, v in specs.items() if k != "__metadata__"}
        )
    # Apply only naming/unused-weight normalization; never load tensor contents.
    actual = model.sanitize(actual)
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    assert actual == expected


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("SPARK3_MM_MODEL"), reason="SPARK3_MM_MODEL is not set"
)
def test_real_multimodal_checkpoint_load_image_prefill_and_decode():
    Image = pytest.importorskip("PIL.Image")
    mx.set_cache_limit(64 * 1024 * 1024)
    path = Path(os.environ["SPARK3_MM_MODEL"])
    model, tokenizer = load(path, strict=True, lazy=True)
    processor = Spark3Processor.from_pretrained(path, tokenizer, model.args)
    prompt = processor.image_placeholder + "\n图片里是什么颜色？"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    data = processor(prompt, [Image.new("RGB", (64, 64), "red")])
    ids = data.pop("input_ids")
    prepared = model.prepare_inputs(ids, **data)
    cache = model.make_cache()
    output = model(ids, cache=cache, **prepared)
    mx.eval(output)
    assert bool(mx.all(mx.isfinite(output)))
    next_id = mx.argmax(output[:, -1], axis=-1)[:, None]
    decoded = model(next_id, cache=cache)
    embeddings = mx.concatenate(
        [prepared["input_embeddings"], model.model.embedding(next_id)], axis=1
    )
    next_position = prepared["positions"].max() + 1
    positions = mx.concatenate(
        [prepared["positions"], mx.broadcast_to(next_position, (3, 1, 1))], axis=-1
    )
    full = model(
        mx.concatenate([ids, next_id], axis=1),
        input_embeddings=embeddings,
        positions=positions,
    )[:, -1:]
    mx.eval(decoded, full)
    a, b = (
        np.array(x.astype(mx.float32)).astype(np.float64).ravel()
        for x in (decoded, full)
    )
    relative_l2 = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert bool(mx.all(mx.isfinite(decoded)))
    # BF16 full prefill can tie two maxima that GEMV decode separates. Require
    # the decode winner to be a true maximum of the full-prefill distribution.
    assert b[a.argmax()] == b.max()
    assert relative_l2 < 0.02

    def log_softmax(values):
        shifted = values - values.max()
        return shifted - np.log(np.exp(shifted).sum())

    log_p, log_q = log_softmax(b), log_softmax(a)
    kl = float(np.sum(np.exp(log_p) * (log_p - log_q)))
    assert kl < 0.002
    print(
        json.dumps(
            {
                "greedy_token": int(a.argmax()),
                "full_top_ties": int(np.count_nonzero(b == b.max())),
                "relative_logit_l2": float(relative_l2),
                "distribution_kl": kl,
                "first_token_text": tokenizer.decode(next_id[0].tolist()),
                "owner_kv_caches": len(model.model.cache_layer_ids),
                "mrope_delta": cache[-1][0].tolist(),
            }
        )
    )
