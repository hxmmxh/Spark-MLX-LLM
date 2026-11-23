"""Prepare original compressed-tensors Spark weights entirely in memory."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .conversion import _mlx_config, _plan, _rows, inventory, read, require

_NUMPY_DTYPES = {
    "BF16": "<u2",
    "F16": "<f2",
    "F32": "<f4",
    "I8": "i1",
    "I32": "<i4",
    "U32": "<u4",
    "I64": "<i8",
}


def _array(values, dtype):
    value = mx.array(values)
    return value.view(mx.bfloat16) if dtype == "BF16" else value


def load_compressed_source(path, *, chunk_rows=256):
    """Return normalized config and weights without writing any model files.

    Uses the disk converter's validation, integer repacking and coefficient
    rounding rules. Only one projection's final arrays and a row chunk need
    temporary space; there is no full-model BF16 expansion. Repacking is eager
    even when the caller later loads the model with lazy=True.
    """
    require(type(chunk_rows) is int and chunk_rows > 0, "chunk_rows must be positive")
    path = Path(path).expanduser().resolve()
    config_path = path / "config.json"
    config_signature = (config_path.stat().st_size, config_path.stat().st_mtime_ns)
    config = json.loads(config_path.read_text())
    require(
        (config.get("quantization_config") or {}).get("quant_method")
        == "compressed-tensors"
        and not config.get("quantization"),
        "Expected original compressed-tensors source for in-memory preparation",
    )
    tensors, files = inventory(path, include_draft_shards=True)
    signatures = {
        path / name: ((path / name).stat().st_size, (path / name).stat().st_mtime_ns)
        for name in files
    }
    signatures[config_path] = config_signature
    effective, target, modules, planned, _, _, _ = _plan(config, tensors, "mixed")
    effective = _mlx_config(effective, modules)
    weights = {}
    converted_keys = {
        name + "." + suffix
        for name in modules
        for suffix in ("weight", "scales", "biases")
    }
    sources = {**tensors, **target}
    for name in sorted(planned.keys() - converted_keys):
        desc = sources[name]
        values = np.frombuffer(read(desc), dtype=_NUMPY_DTYPES[desc["dtype"]]).reshape(
            desc["shape"]
        )
        weights[name] = _array(values, desc["dtype"])
        del values
    for name, scheme in modules.items():
        arrays = {
            suffix: np.empty(
                planned[name + "." + suffix]["shape"],
                dtype=_NUMPY_DTYPES[planned[name + "." + suffix]["dtype"]],
            )
            for suffix in ("weight", "scales", "biases")
        }
        for first in range(0, scheme["shape"][0], chunk_rows):
            stop = min(first + chunk_rows, scheme["shape"][0])
            payload, _ = _rows(name, scheme, target, first, stop)
            for suffix, data in payload.items():
                array = arrays[suffix]
                array[first:stop] = np.frombuffer(data, dtype=array.dtype).reshape(
                    stop - first, *array.shape[1:]
                )
        for suffix, array in arrays.items():
            weights[name + "." + suffix] = _array(
                array, planned[name + "." + suffix]["dtype"]
            )
        del arrays, array
    require(
        all(
            p.exists() and (p.stat().st_size, p.stat().st_mtime_ns) == signature
            for p, signature in signatures.items()
        ),
        "Source changed during in-memory model preparation",
    )
    return effective, weights
