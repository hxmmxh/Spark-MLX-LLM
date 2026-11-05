"""Export deployable draft parameters for direct, precision-preserving reload."""

import copy
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .draft_quantization import RuntimeDraftW4Linear, RuntimeDraftW8Linear
from .draft_source import resolve_draft_source
from .quantization_format import module_spec, vocabulary_hash


def save_draft(pair, output):
    """Atomically save a standalone draft, including packed INT4/INT8 parameters.

    The main model and its external readout are referenced by config rather than
    copied. Existing outputs are never overwritten. Quantized drafts reload
    directly; use draft_quantization='checkpoint' to preserve floating drafts,
    or merge using --draft-source.
    """
    from ..conversion import publish

    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Draft output already exists: {output}")
    drafter = pair.drafter
    config = copy.deepcopy(pair.config)
    if drafter.lm_head is None:
        readout = config.get("dspark_readout", "target")
        expected = (
            pair.target.model.model.embedding.as_linear
            if readout == "embedding"
            else pair.target.project
        )
        if drafter._external_lm_head != expected:
            raise ValueError("Cannot export a draft with a custom external readout")
    modules = {}
    for name, module in drafter.named_modules():
        if type(module) is RuntimeDraftW4Linear:
            modules[name] = module_spec(name, "int4")
        elif type(module) is RuntimeDraftW8Linear:
            modules[name] = module_spec(name, "int8")
        elif isinstance(module, nn.QuantizedLinear) or (
            "weight" in module and not mx.issubdtype(module.weight.dtype, mx.floating)
        ):
            raise ValueError(f"Cannot export unsupported draft quantization: {name}")
    policy = (
        "int4"
        if any(spec and spec["bits"] == 4 for spec in modules.values())
        else "int8"
        if modules
        else "bf16"
    )
    config["draft_quantization"] = {
        "format": "spark-mlx-draft",
        "version": 1,
        "policy": policy,
        "modules": modules,
    }
    for key in (
        "quantization",
        "quantization_config",
        "conversion",
        "gguf_import",
        "w4_qmv_backend",
        "w8_qmv_backend",
    ):
        config.pop(key, None)
    if pair.tokenizer is not None:
        config["dspark_target_vocab_sha256"] = vocabulary_hash(
            pair.tokenizer.get_vocab()
        )
    weights = dict(tree_flatten(drafter.parameters()))
    if config.get("dspark_shared_embedding", False):
        if drafter.embed_tokens.weight is not pair.target.model.model.embedding.weight:
            raise ValueError("Shared draft embedding no longer matches the main model")
        weights.pop("embed_tokens.weight")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        mx.save_safetensors(str(partial / "model.safetensors"), weights)
        (partial / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        # Inspect the actual serialized names, shapes and dtypes before publishing.
        source = resolve_draft_source(partial, pair.config)
        summary = {
            "output": str(output),
            "policy": policy,
            "weight_bytes": sum(desc["bytes"] for desc in source.tensors.values()),
            "quantized_modules": len(modules),
        }
        publish(partial, output)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return summary
