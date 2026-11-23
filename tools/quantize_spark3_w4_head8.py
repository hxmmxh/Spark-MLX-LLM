#!/usr/bin/env python3
"""Stream BF16 Spark3 checkpoints into MLX W4/W8 affine checkpoints.

Text matrices (including embeddings) use 4 bits; lm_head uses 8 bits.
Vision, head gates, norms and biases retain their original BF16 values.
Tied embeddings are exported as independent W4 embeddings and a W8 head.
No calibration or activation quantization is performed.

Run with this repository's MLX Python environment. By default, inputs are
read from ../models/test and outputs written to ../models/quantized_mlx,
relative to the repository root. Existing outputs are never overwritten.
"""

import argparse
import gc
import json
import shutil
import struct
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
GROUP_SIZE = 64
SHARD_BYTES = 1024**3


def headers(directory):
    tensors = {}
    for file in sorted(directory.glob("model*.safetensors")):
        with file.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        assert file.stat().st_size == 8 + length + max(
            v["data_offsets"][1] for v in entries.values()
        ), f"Incomplete shard: {file}"
        for name, spec in entries.items():
            assert name not in tensors, f"Duplicate tensor: {name}"
            tensors[name] = {**spec, "file": file.name, "data_start": 8 + length}
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {k: v["file"] for k, v in tensors.items()}
    return tensors


def weight_bits(name, spec):
    if not name.endswith(".weight") or len(spec["shape"]) != 2:
        return None
    if name.startswith(("model.visual.", "visual.")):
        return None
    if name.endswith("self_attn.g_proj.weight"):
        return None
    assert spec["shape"][-1] % GROUP_SIZE == 0, name
    return 8 if name == "lm_head.weight" else 4


def verify_preserved(source, output, original, actual, names):
    checked_bytes = 0
    for name in names:
        before, after = original[name], actual[name]
        remaining = before["data_offsets"][1] - before["data_offsets"][0]
        with (
            (source / before["file"]).open("rb") as left,
            (output / after["file"]).open("rb") as right,
        ):
            left.seek(before["data_start"] + before["data_offsets"][0])
            right.seek(after["data_start"] + after["data_offsets"][0])
            while remaining:
                size = min(remaining, 8 * 1024**2)
                assert left.read(size) == right.read(size), name
                remaining -= size
                checked_bytes += size
    return checked_bytes


def smoke_test(directory, expected_modules):
    from spark_mlx_llm import load

    model, tokenizer = load(directory, strict=True, lazy=True)
    modules = dict(model.named_modules())
    for path, bits in expected_modules.items():
        module = modules[path]
        expected_type = (
            nn.QuantizedEmbedding if path == "model.embedding" else nn.QuantizedLinear
        )
        assert isinstance(module, expected_type), (path, type(module))
        assert (module.bits, module.group_size, module.mode) == (
            bits,
            GROUP_SIZE,
            "affine",
        )
    assert not model.args.tie_word_embeddings
    tokens = tokenizer.encode("你好")
    assert tokens
    inputs = mx.array([tokens])
    cache = model.make_cache()
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, len(tokens), model.args.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)))
    token = mx.argmax(logits[:, -1], axis=-1)[:, None]
    decoded = model(token, cache=cache)
    mx.eval(decoded)
    assert decoded.shape == (1, 1, model.args.vocab_size)
    assert bool(mx.all(mx.isfinite(decoded)))
    assert all(
        c.offset == len(tokens) + 1 for c in cache[: len(model.model.cache_layer_ids)]
    )
    result = {
        "strict_load": True,
        "all_quantized_module_types_and_bits_checked": True,
        "text_prefill_shape": list(logits.shape),
        "cached_decode_shape": list(decoded.shape),
        "logits_finite": True,
        "owner_kv_caches": len(model.model.cache_layer_ids),
    }
    if model.args.vision_config is not None:
        args = model.model.visual.args
        pixels = mx.zeros(
            (4, 3 * args.temporal_patch_size * args.patch_size**2), dtype=mx.float32
        )
        image_inputs = mx.array([[tokens[0], model.args.image_token_id, tokens[-1]]])
        image_logits = model(
            image_inputs,
            pixel_values=pixels,
            image_grid_thw=[[1, 2, 2]],
            image_offsets=[(1, 1)],
        )
        mx.eval(image_logits)
        assert image_logits.shape == (1, 3, model.args.vocab_size)
        assert bool(mx.all(mx.isfinite(image_logits)))
        result["synthetic_image_forward_shape"] = list(image_logits.shape)
        result["synthetic_image_logits_finite"] = True
    del model, modules, cache, logits, decoded
    gc.collect()
    mx.clear_cache()
    return result


def convert(source, destination):
    started = time.monotonic()
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError(
            f"Output or staging directory already exists: {destination}"
        )
    config = json.loads((source / "config.json").read_text())
    assert "quantization" not in config and "quantization_config" not in config
    original = headers(source)
    assert all(v["dtype"] == "BF16" for v in original.values())
    source_stats = {
        f.name: (f.stat().st_size, f.stat().st_mtime_ns)
        for f in source.iterdir()
        if f.is_file()
    }
    tied = config.get(
        "tie_word_embeddings",
        config.get("text_config", {}).get("tie_word_embeddings", False),
    )
    if tied:
        assert "lm_head.weight" not in original
        assert "model.embedding.weight" in original
        config["tie_word_embeddings"] = False
        if "text_config" in config:
            config["text_config"]["tie_word_embeddings"] = False
    else:
        assert "lm_head.weight" in original
    quant_config = {"bits": 4, "group_size": GROUP_SIZE, "mode": "affine"}
    expected, quantized_modules, preserved = {}, {}, []
    pending, pending_bytes, total_bytes = {}, 0, 0
    temporary_files, index_map = [], {}
    staging.mkdir(parents=True)

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        file = staging / f"model-part-{len(temporary_files) + 1:05d}.safetensors"
        mx.save_safetensors(str(file), pending, metadata={"format": "mlx"})
        temporary_files.append(file)
        index_map.update({name: file.name for name in pending})
        pending = {}
        pending_bytes = 0
        mx.clear_cache()

    def add(name, tensor):
        nonlocal pending_bytes, total_bytes
        mx.eval(tensor)
        if pending_bytes + tensor.nbytes > SHARD_BYTES:
            flush()
        assert name not in expected
        expected[name] = {
            "dtype": "U32" if tensor.dtype == mx.uint32 else "BF16",
            "shape": list(tensor.shape),
        }
        pending[name] = tensor
        pending_bytes += tensor.nbytes
        total_bytes += tensor.nbytes

    def process(name, tensor, spec):
        bits = weight_bits(name, spec)
        if bits is None:
            add(name, tensor)
            preserved.append(name)
            return
        path = name.removesuffix(".weight")
        quantized_modules[path] = bits
        quant_config[path] = {"bits": bits, "group_size": GROUP_SIZE, "mode": "affine"}
        packed, scales, biases = mx.quantize(
            tensor, group_size=GROUP_SIZE, bits=bits, mode="affine"
        )
        mx.eval(packed, scales, biases)
        assert packed.dtype == mx.uint32 and scales.dtype == biases.dtype == mx.bfloat16
        rows, columns = spec["shape"]
        assert packed.shape == (rows, columns * bits // 32)
        assert scales.shape == biases.shape == (rows, columns // GROUP_SIZE)
        assert bool(mx.all(mx.isfinite(scales) & mx.isfinite(biases)))
        add(name, packed)
        add(path + ".scales", scales)
        add(path + ".biases", biases)

    print(
        json.dumps(
            {
                "model": source.name,
                "phase": "quantizing",
                "source_tensors": len(original),
                "untie_embeddings": tied,
            }
        ),
        flush=True,
    )
    for file in sorted(source.glob("model*.safetensors")):
        weights = mx.load(str(file))
        for name in list(weights):
            tensor = weights.pop(name)
            process(name, tensor, original[name])
            if tied and name == "model.embedding.weight":
                process("lm_head.weight", tensor, original[name])
            del tensor
        del weights
        gc.collect()
        mx.clear_cache()
        print(
            json.dumps(
                {
                    "model": source.name,
                    "source_shard_done": file.name,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )
    flush()
    for i, file in enumerate(temporary_files, 1):
        new_name = f"model-{i:05d}-of-{len(temporary_files):05d}.safetensors"
        file.rename(staging / new_name)
        index_map = {k: new_name if v == file.name else v for k, v in index_map.items()}
    config["quantization"] = quant_config
    config["quantization_config"] = quant_config
    config["quantize_activations"] = False
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": dict(sorted(index_map.items())),
    }
    (staging / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    for pattern in (
        "tokenizer*",
        "vocab*",
        "merges.txt",
        "*.tiktoken",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template*",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        for file in source.glob(pattern):
            if file.is_file():
                shutil.copy2(file, staging / file.name)
    actual = headers(staging)
    assert {
        k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in actual.items()
    } == expected
    assert total_bytes == sum(
        v["data_offsets"][1] - v["data_offsets"][0] for v in actual.values()
    )
    preserved_bytes_checked = verify_preserved(
        source, staging, original, actual, preserved
    )
    print(
        json.dumps(
            {
                "model": source.name,
                "phase": "strict_load_and_forward_checks",
                "tensor_bytes": total_bytes,
            }
        ),
        flush=True,
    )
    checks = smoke_test(staging, quantized_modules)
    assert source_stats == {
        f.name: (f.stat().st_size, f.stat().st_mtime_ns)
        for f in source.iterdir()
        if f.is_file()
    }
    report = {
        "source": str(source),
        "output": str(destination),
        "method": "MLX affine weight-only quantization; no calibration",
        "versions": {"mlx": version("mlx"), "mlx-lm": version("mlx-lm")},
        "device": mx.device_info()["device_name"],
        "group_size": GROUP_SIZE,
        "text_weight_bits": 4,
        "lm_head_weight_bits": 8,
        "activation_quantization": False,
        "retained_precision": "BF16",
        "retained_bf16": "vision tower, attention head gates, normalization weights, biases",
        "untied_word_embeddings": tied,
        "source_dtype_counts": dict(Counter(v["dtype"] for v in original.values())),
        "source_tensor_bytes": sum(
            v["data_offsets"][1] - v["data_offsets"][0] for v in original.values()
        ),
        "output_tensor_bytes": total_bytes,
        "output_shards": len(temporary_files),
        "quantized_module_counts": dict(Counter(quantized_modules.values())),
        "quantized_modules": quantized_modules,
        "preserved_bf16_tensors": preserved,
        "verification": {
            "complete_headers_and_index": True,
            "all_output_dtypes_and_shapes": True,
            "preserved_bf16_bytes_exactly_equal": preserved_bytes_checked,
            "source_file_sizes_and_mtimes_unchanged": True,
            **checks,
        },
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (staging / "quantization_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    staging.rename(destination)
    print(
        json.dumps(
            {
                "model": source.name,
                "phase": "complete",
                "output": str(destination),
                "tensor_bytes": total_bytes,
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=WORKSPACE_ROOT / "models/test",
        help="Directory containing the BF16 model directories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["9b_ced_20000", "hf_latest", "second_hf"],
        help="Model directory names beneath --input-root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "models/quantized_mlx",
        help="Parent of the new <model>-w4-head8-g64 output directories",
    )
    args = parser.parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    mx.set_cache_limit(64 * 1024**2)
    mx.set_memory_limit(12 * 1024**3)
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.models:
        reports.append(
            convert(args.input_root / name, args.output_root / f"{name}-w4-head8-g64")
        )
    (args.output_root / "quantization_summary.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
