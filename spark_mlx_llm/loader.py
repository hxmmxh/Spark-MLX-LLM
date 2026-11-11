"""One public loader for target-only and speculative inference."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from ._target_loader import _get_model_classes, _resolve_model_path

__all__ = ["load"]


def load(
    path_or_repo,
    *,
    draft_model=None,
    use_draft=True,
    draft_quantization=None,
    draft_readout="checkpoint",
    tokenizer_config=None,
    model_config=None,
    lazy=False,
    strict=True,
    revision=None,
    dtype=None,
    w4_qmv_backend=None,
    model_cache_dir=None,
    model_cache=None,
):
    """Return ``(model, tokenizer)``; an optional draft uses the same generate API.

    Merged draft weights are detected automatically. ``use_draft=False`` loads
    only the target; ``GenerationConfig(max_draft_tokens=0)`` disables drafting
    for an individual request without reloading the model.
    """
    if type(use_draft) is not bool:
        raise ValueError("use_draft must be a boolean")
    from .ops.linear import LinearConfig, automatic_w8_backend

    LinearConfig.validate_overrides(model_config or {})
    tokenizer_config = {"trust_remote_code": False, **(tokenizer_config or {})}
    if not use_draft and draft_model is not None:
        raise ValueError("draft_model cannot be supplied with use_draft=False")
    path = _resolve_model_path(path_or_repo, revision)
    from .model_source import model_cache_enabled

    model_cache = model_cache_enabled(path, model_cache)
    raw_path = path / "config.json" if path.is_dir() else None
    raw = json.loads(raw_path.read_text()) if raw_path and raw_path.is_file() else {}
    overrides = {
        "w4_qmv_backend": raw.get("w4_qmv_backend", "multirow"),
        **(model_config or {}),
    }
    for name, value in (("w4_qmv_backend", w4_qmv_backend),):
        if value is not None:
            overrides[name] = value
    merged = False
    if use_draft and path.is_dir() and any(path.glob("*.safetensors")):
        from .dspark.checkpoint import inspect_checkpoint

        merged = bool(inspect_checkpoint(path).draft_tensors)
    if use_draft and (draft_model is not None or merged):
        from .dspark.loader import _load_pair

        if not strict:
            raise ValueError("Draft loading requires strict=True")
        runtime = _load_pair(
            path, draft_path=draft_model, draft_quantization=draft_quantization,
            draft_readout=draft_readout, dtype=dtype, lazy=lazy,
            model_config=overrides, model_cache_dir=model_cache_dir, model_cache=model_cache,
        )
        if tokenizer_config != {"trust_remote_code": False}:
            from ._target_loader import _load_tokenizer
            from .model_source import prepare_model_source

            tokenizer_path = (
                path if path.is_dir()
                else prepare_model_source(path, cache_dir=model_cache_dir)
            )
            runtime = replace(runtime, tokenizer=_load_tokenizer(
                tokenizer_path, tokenizer_config, runtime.config.get("eos_token_id")
            ))
        model, tokenizer = runtime.target.model, runtime.tokenizer
    else:
        from ._target_loader import load as load_target
        from .dspark.loader import ModelPair

        if draft_quantization is not None or draft_readout != "checkpoint":
            raise ValueError("Draft loading options require a draft model")
        model, tokenizer, config = load_target(
            path, tokenizer_config=tokenizer_config, model_config=overrides,
            lazy=lazy, strict=strict, dtype=dtype, return_config=True,
            model_cache_dir=model_cache_dir, model_cache=model_cache,
        )
        generation_path = path / "generation_config.json" if path.is_dir() else None
        generation_config = (
            json.loads(generation_path.read_text())
            if generation_path and generation_path.is_file()
            else {"eos_token_id": list(tokenizer.eos_token_ids)}
        )
        runtime = ModelPair(SimpleNamespace(model=model), None, tokenizer, config, generation_config)
    # Store metadata outside nn.Module's parameter tree: runtime points back to
    # its target and must not become a recursively traversed model parameter.
    object.__setattr__(model, "_spark_runtime", runtime)
    object.__setattr__(model, "_spark_load_options", {
        "model": str(path.resolve()), "draft_model": str(Path(draft_model).expanduser().resolve()) if draft_model is not None else None,
        "draft_loaded": runtime.drafter is not None,
        "draft_quantization": (runtime.draft_quantization or {}).get("policy"),
        "dtype": dtype, "model_cache": model_cache,
        "model_config": overrides,
        "w4_qmv_backend": runtime.config.get("w4_qmv_backend", overrides["w4_qmv_backend"]),
        "w8_qmv_backend": automatic_w8_backend(),
    })
    return model, tokenizer
