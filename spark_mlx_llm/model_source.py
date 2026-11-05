"""Prepare local source checkpoints automatically and reuse atomic MLX caches."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path


def add_model_cache_arguments(parser):
    """Use in-memory INT4 preparation by default; target GGUF still needs a cache."""
    parser.add_argument(
        "--model-cache-dir",
        help="Generated MLX cache root (overrides SPARK_MLX_MODEL_CACHE)",
    )
    parser.add_argument(
        "--model-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache converted model files (default: off for INT4 directories, on for target GGUF)",
    )


def model_cache_enabled(path, model_cache):
    """Resolve automatic cache policy without writing files or importing MLX."""
    if model_cache is not None:
        if type(model_cache) is not bool:
            raise ValueError("model_cache must be a boolean or None")
        return model_cache
    return (
        path is not None
        and Path(path).suffix.lower() == ".gguf"
        and Path(path).expanduser().is_file()
    )


def prepare_model_source(path, *, cache_dir=None):
    """Pass through BF16/MLX directories; cache compressed-tensors/GGUF conversion.

    Keys include source file paths, sizes, nanosecond mtimes, and converter
    identities. A cooperative file lock prevents duplicate first conversions.
    The converter publishes a complete directory atomically and leaves sources
    untouched. SPARK_MLX_MODEL_CACHE controls the default cache directory.
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if not (path / "config.json").is_file():
            # Preserve the model loader's original missing-config behavior.
            return path
        config = json.loads((path / "config.json").read_text())
        quant = config.get("quantization_config") or {}
        if quant.get("quant_method") != "compressed-tensors" or config.get(
            "quantization"
        ):
            return path
        files = sorted(
            p
            for p in path.iterdir()
            if p.is_file()
            and p.suffix
            in (
                ".json",
                ".safetensors",
                ".jinja",
                ".model",
                ".txt",
            )
        )
        source_format = "compressed-tensors"
    elif path.is_file() and path.suffix.lower() == ".gguf":
        files, source_format = [path], "GGUF"
    else:
        raise FileNotFoundError(
            f"Expected a local model directory or target GGUF: {path}"
        )
    from . import conversion

    identities = [Path(__file__), Path(conversion.__file__)]
    if source_format == "GGUF":
        from importlib.metadata import version

        from . import gguf_target

        identities.append(Path(gguf_target.__file__))
    identity = {
        "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
        "code": [hashlib.sha256(p.read_bytes()).hexdigest() for p in identities],
        "policy": "mixed-w4g32-w8-native-v1",
        "gguf_version": version("gguf") if source_format == "GGUF" else None,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(
        cache_dir
        or os.environ.get(
            "SPARK_MLX_MODEL_CACHE", str(Path.home() / ".cache/spark-mlx-llm/models")
        )
    ).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    output = root / key
    with (root / (key + ".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            manifest = output / "conversion_manifest.json"
            if (
                not manifest.is_file()
                or json.loads(manifest.read_text()).get("status") != "complete"
            ):
                raise ValueError(f"Incomplete model cache; remove and retry: {output}")
            return output
        print(
            f"[INFO] Preparing {source_format} model in MLX cache: {output}",
            file=sys.stderr,
            flush=True,
        )
        conversion.convert_to_mlx(path, output, w8_backend="native")
    return output


def prepare_local_cli_model(value, *, cache_dir=None):
    """Normalize local CLI paths while leaving upstream HF identifiers alone."""
    if value is not None and Path(value).expanduser().exists():
        return str(prepare_model_source(value, cache_dir=cache_dir))
    return value


def load_uncached_model_source(path):
    """Return in-memory compressed weights, or None for directly loadable models."""
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".gguf":
        raise ValueError(
            "--no-model-cache currently supports compressed-tensors INT4 and BF16/MLX directories; "
            "a target GGUF requires disk caching or an explicitly converted MLX model."
        )
    config_path = path / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    if (config.get("quantization_config") or {}).get(
        "quant_method"
    ) != "compressed-tensors" or config.get("quantization"):
        return None
    from .compressed_source import load_compressed_source

    print(
        "[INFO] Preparing compressed-tensors weights in memory (model cache disabled)",
        file=sys.stderr,
        flush=True,
    )
    return load_compressed_source(path)
