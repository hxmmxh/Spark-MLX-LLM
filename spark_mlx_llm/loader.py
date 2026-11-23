import json
import struct
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_map
from mlx_lm.tokenizer_utils import load as load_tokenizer
from mlx_lm.utils import load_model

MODEL_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]


def _resolve_model_path(path_or_repo: str | Path, revision: str | None):
    model_path = Path(path_or_repo)
    if model_path.exists():
        return model_path

    return Path(
        snapshot_download(
            str(path_or_repo),
            revision=revision,
            allow_patterns=MODEL_ALLOW_PATTERNS,
        )
    )


def _get_model_classes(config: dict):
    model_type = config.get("model_type")
    if model_type == "spark2_5":
        from .spark2_5 import Model, ModelArgs
    elif model_type == "spark3":
        from .spark3 import Model, ModelArgs
    else:
        raise ValueError(
            f"Expected model_type 'spark2_5' or 'spark3', got {model_type!r}"
        )
    return Model, ModelArgs


def _load_tokenizer(model_path, tokenizer_config, eos_token_ids):
    return load_tokenizer(
        model_path,
        tokenizer_config_extra=tokenizer_config,
        eos_token_ids=eos_token_ids,
    )


def _validate_checkpoint_files(model_path):
    # A downloading safetensors shard can otherwise fail after allocating a model.
    for path in sorted(model_path.glob("model*.safetensors")):
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Incomplete checkpoint shard: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > 100_000_000:
                raise ValueError(f"Invalid safetensors header: {path}")
            try:
                header = json.loads(stream.read(header_size))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Incomplete safetensors header: {path}") from exc
        expected = (
            8
            + header_size
            + max(
                (
                    v["data_offsets"][1]
                    for k, v in header.items()
                    if k != "__metadata__"
                ),
                default=0,
            )
        )
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"Incomplete checkpoint shard: {path} ({actual}/{expected} bytes)"
            )


def load(
    path_or_repo: str | Path,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    lazy: bool = False,
    strict: bool = True,
    revision: str | None = None,
    dtype: str | None = None,
    return_config: bool = False,
    tokenizer_path: str | Path | None = None,
):
    model_path = _resolve_model_path(path_or_repo, revision)
    _validate_checkpoint_files(model_path)
    model, config = load_model(
        model_path,
        lazy=lazy,
        strict=strict,
        model_config=model_config,
        get_model_classes=_get_model_classes,
    )
    if dtype is not None:
        dtypes = {"float32": mx.float32, "bfloat16": mx.bfloat16}
        if dtype not in dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype!r}; choose from {sorted(dtypes)}"
            )
        target_dtype = dtypes[dtype]
        model.update(
            tree_map(
                lambda value: (
                    value.astype(target_dtype)
                    if mx.issubdtype(value.dtype, mx.floating)
                    else value
                ),
                model.parameters(),
            )
        )
        if not lazy:
            mx.eval(model.parameters())
    tokenizer = _load_tokenizer(
        _resolve_model_path(tokenizer_path, revision) if tokenizer_path else model_path,
        tokenizer_config or {},
        config.get("eos_token_id"),
    )

    if return_config:
        return model, tokenizer, config
    return model, tokenizer
