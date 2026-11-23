"""Strict loading of merged or separate Spark target and DSpark checkpoints."""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map_with_path
from mlx_lm.tokenizer_utils import load as load_tokenizer

from .checkpoint import DRAFT_PREFIX, inspect_checkpoint, validate_draft_shapes
from .drafter import DSparkDrafter
from .quantization_format import draft_quantization_config
from .target import SparkTarget


@dataclass(frozen=True)
class ModelPair:
    target: SparkTarget
    drafter: DSparkDrafter | None
    tokenizer: object
    config: dict
    generation_config: dict
    draft_quantization: dict | None = None


def _quantize_structure(model, weights, quantization):
    def predicate(path, module):
        if path + ".scales" not in weights:
            return False
        return quantization.get(path, True)

    nn.quantize(
        model,
        group_size=quantization.get("group_size", 64),
        bits=quantization.get("bits", 4),
        mode=quantization.get("mode", "affine"),
        class_predicate=predicate,
    )


def _source_weights(source):
    import numpy as np

    from .draft_source import read_tensor

    result = {}
    for name, desc in source.tensors.items():
        numpy_dtype = {
            "BF16": "<u2",
            "F16": "<f2",
            "F32": "<f4",
            "I8": "i1",
            "U32": "<u4",
        }[desc["dtype"]]
        value = mx.array(
            np.frombuffer(read_tensor(desc), dtype=numpy_dtype).reshape(desc["shape"])
        )
        result[name] = value.view(mx.bfloat16) if desc["dtype"] == "BF16" else value
    return result


def _target_model_config(checkpoint_config, model_config=None):
    """Apply dSpark defaults below checkpoint and explicit caller choices.

    Wide W4 reuses the existing guarded M8 path and needs no packed copies.
    W8 retains this loader's existing policy; ordinary model defaults are
    unaffected. Unsupported W4 descriptors fall back in operator dispatch.
    """
    from ..ops.linear import automatic_w8_backend

    return {
        "w4_qmv_backend": "multirow",
        **checkpoint_config,
        **(model_config or {}),
        "w8_qmv_backend": automatic_w8_backend(),
    }


def _load_pair(
    path,
    *,
    draft_path=None,
    draft_quantization=None,
    draft_readout="checkpoint",
    dtype=None,
    lazy=False,
    tokenizer=True,
    model_config=None,
    model_cache_dir=None,
    model_cache=None,
):
    """Load a merged directory, or a target plus a standalone draft source.

    draft_path accepts a floating-point Spark DSpark GGUF or a Safetensors
    directory. The target may be BF16/MLX, original compressed-tensors INT4,
    or a Spark GGUF file. Original INT4 is prepared in memory by default;
    target GGUF uses the MLX cache. model_cache=True enables INT4 disk caching.
    draft_quantization defaults to int8 for floating drafts; already quantized
    drafts retain their stored precision. checkpoint preserves floating weights.
    Floating inputs can select bf16 (unchanged), per-channel int8, or QKV/FFN
    int4/G32 with int8 on the other major draft projections. It does not modify
    checkpoint files. draft_readout may select the target embedding projection.
    model_cache_dir overrides SPARK_MLX_MODEL_CACHE for generated MLX files.
    model_cache=False prepares original compressed-tensors INT4 in memory;
    no converted model or temporary model file is written.
    Eligible M8 W4 projections default to the wide multirow path. Checkpoint
    settings and explicit model_config values override loader defaults;
    model_config={"w4_qmv_backend": "native"} disables this W4 selection.
    """
    from spark_mlx_llm.model import Model, ModelArgs
    from spark_mlx_llm.model_source import (
        load_uncached_model_source,
        model_cache_enabled,
        prepare_model_source,
    )
    from spark_mlx_llm.ops.linear import QuantizedProjections

    if draft_quantization not in (None, "checkpoint", "bf16", "int8", "int4"):
        raise ValueError("draft_quantization must be checkpoint, bf16, int8 or int4")
    if draft_readout not in ("checkpoint", "embedding"):
        raise ValueError("draft_readout must be checkpoint or embedding")
    path = Path(path).expanduser().resolve()
    model_cache = model_cache_enabled(path, model_cache)
    prepared = load_uncached_model_source(path) if not model_cache else None
    if model_cache:
        path = prepare_model_source(path, cache_dir=model_cache_dir)
    raw = (
        prepared[0]
        if prepared is not None
        else json.loads((path / "config.json").read_text())
    )
    checkpoint = inspect_checkpoint(path)
    source = None
    if draft_path is not None:
        from .draft_source import resolve_draft_source

        if checkpoint.draft_tensors:
            raise ValueError(
                "Target already contains draft weights; omit draft_path or use a target-only checkpoint"
            )
        source = resolve_draft_source(draft_path, raw, target_directory=path)
        raw = source.config
    schema = draft_quantization_config(raw.get("draft_quantization"))
    stored_quantized = schema["policy"] != "bf16"
    if draft_quantization is None:
        draft_quantization = "checkpoint" if stored_quantized else "int8"
    if stored_quantized and draft_quantization not in ("checkpoint", schema["policy"]):
        raise ValueError(
            f"Draft is stored as {schema['policy']}; use checkpoint or the same policy. "
            "Changing precision requires the original floating draft."
        )
    shared_embedding = raw.get("dspark_shared_embedding", False)
    if type(shared_embedding) is not bool:
        raise ValueError("dspark_shared_embedding must be a boolean")
    target_config = _target_model_config(raw, model_config)
    target_model = Model(ModelArgs.from_dict(target_config))
    draft_checkpoint = source.checkpoint() if source else checkpoint
    drafter = DSparkDrafter.from_spark_config(
        raw,
        readout_mode="linear"
        if draft_checkpoint.summary()["draft_has_lm_head"]
        else "external",
    )
    validate_draft_shapes(
        drafter.config,
        draft_checkpoint,
        shared_embedding=shared_embedding,
        quantization=raw.get("draft_quantization"),
    )
    if checkpoint.summary()["unclassified_tensors"]:
        raise ValueError("Merged checkpoint contains unclassified tensors")
    target_weights, draft_weights = {}, {}
    shards = (
        [prepared[1]]
        if prepared is not None
        else (
            mx.load(str(shard))
            for shard in sorted({t.shard for t in checkpoint.tensors})
        )
    )
    for shard_weights in shards:
        for name, array in shard_weights.items():
            if name.startswith(DRAFT_PREFIX):
                draft_weights[name.removeprefix(DRAFT_PREFIX)] = array
            else:
                target_weights[name] = array
    target_weights = target_model.sanitize(target_weights)
    if raw.get("quantization"):
        _quantize_structure(target_model, target_weights, raw["quantization"])
    target_model.load_weights(list(target_weights.items()), strict=True)
    if source is not None:
        draft_weights = _source_weights(source)
    if dtype is not None:
        dtypes = {"bfloat16": mx.bfloat16, "float32": mx.float32}
        if dtype not in dtypes:
            raise ValueError("dtype must be bfloat16 or float32")
        scale_paths = QuantizedProjections.scale_paths(target_model)
        target_model.update(
            tree_map_with_path(
                lambda name, value: (
                    value.astype(dtypes[dtype])
                    if name not in scale_paths
                    and mx.issubdtype(value.dtype, mx.floating)
                    else value
                ),
                target_model.parameters(),
            )
        )
        coefficients = {
            name + suffix
            for name in schema["modules"]
            for suffix in (".scales", ".biases")
        }
        draft_weights = {
            name: value.astype(dtypes[dtype])
            if name not in coefficients and mx.issubdtype(value.dtype, mx.floating)
            else value
            for name, value in draft_weights.items()
        }
    if shared_embedding:
        # Keep the shared array identical even after an explicit dtype override.
        draft_weights["embed_tokens.weight"] = target_model.model.embedding.weight
    from .draft_quantization import quantize_runtime_draft, restore_runtime_draft

    quantization_report = (
        restore_runtime_draft(drafter, draft_weights, schema)
        if stored_quantized
        else None
    )
    drafter.load_weights(list(draft_weights.items()), strict=True)
    target = SparkTarget(target_model, drafter.config.target_layer_ids)
    if drafter.lm_head is None:
        readout = (
            "embedding"
            if draft_readout == "embedding"
            else raw.get("dspark_readout", "target")
        )
        if readout not in ("target", "embedding"):
            raise ValueError("dspark_readout must be target or embedding")
        drafter.bind_lm_head(
            target_model.model.embedding.as_linear
            if readout == "embedding"
            else target.project
        )
        raw = {**raw, "dspark_readout": readout}
    elif draft_readout != "checkpoint":
        raise ValueError(
            "Embedding readout cannot replace an independent draft lm_head"
        )
    if quantization_report is None:
        quantization_report = quantize_runtime_draft(
            drafter,
            "bf16" if draft_quantization == "checkpoint" else draft_quantization,
        )
    if not lazy:
        mx.eval(target_model.parameters(), drafter.parameters())
    tok = (
        load_tokenizer(
            path,
            tokenizer_config_extra={"trust_remote_code": False},
            eos_token_ids=raw.get("eos_token_id"),
        )
        if tokenizer
        else None
    )
    generation_path = path / "generation_config.json"
    generation_config = (
        json.loads(generation_path.read_text()) if generation_path.exists() else {}
    )
    return ModelPair(target, drafter, tok, raw, generation_config, quantization_report)
