"""Load the optional, prebuilt Metal extension only on compatible runtimes.

Keep this module MLX-free: an incompatible native binary must never be imported
just to discover its requirements. These requirements describe spark-mlx-ext
0.2.0's full Metal 4 build, not the minimum requirements of native MLX inference.
"""

import json
import os
import platform
import sys
from functools import cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

MIN_MACOS = (26, 2)
REQUIRED_MLX = "0.32.4"


@cache
def _compatibility_error():
    """Check immutable process/runtime requirements before importing any MLX."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "spark-mlx-ext requires native Apple Silicon macOS (arm64)"
    if sys.implementation.name != "cpython" or sys.version_info < (3, 11):
        return "spark-mlx-ext requires CPython 3.11 or newer"
    release = platform.mac_ver()[0]
    try:
        parts = release.split(".")
        macos = tuple(int(part) for part in parts)
    except ValueError:
        macos = ()
    if not macos:
        return "Cannot determine the macOS version; skipping spark-mlx-ext"
    macos = (*macos, 0)[:2]
    if macos < MIN_MACOS:
        return f"spark-mlx-ext requires macOS 26.2 or newer (found {release})"
    try:
        mlx_version = version("mlx")
    except PackageNotFoundError:
        return "MLX is not installed"
    if mlx_version != REQUIRED_MLX:
        return f"spark-mlx-ext requires mlx=={REQUIRED_MLX} (found {mlx_version})"
    return None


def _load_extension():
    if os.environ.get("SPARK_MLX_DISABLE_EXT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None, "Disabled by SPARK_MLX_DISABLE_EXT"
    reason = _compatibility_error()
    if reason is not None:
        return None, reason
    try:
        return import_module("spark_mlx_ext"), None
    except (ImportError, OSError) as error:
        # Missing packages, Python/MLX ABI mismatches, and dyld failures all
        # leave native MLX available. Errors while executing kernels propagate.
        return None, f"Cannot load spark-mlx-ext: {error}"


def get_extension():
    """Return the compatible extension, or None for native MLX fallbacks.

    Set SPARK_MLX_DISABLE_EXT=1 before starting Python to test a release's
    fallback paths even when the bundled extension is compatible. Platform and
    ABI checks are cached; successful imports use Python's normal module cache.
    """
    return _load_extension()[0]


def extension_status():
    """Report availability without touching incompatible native libraries."""
    extension, reason = _load_extension()
    return {
        "available": extension is not None,
        "reason": reason,
        "required_macos": ".".join(map(str, MIN_MACOS)),
        "required_mlx": REQUIRED_MLX,
    }


if __name__ == "__main__":
    print(json.dumps(extension_status(), ensure_ascii=False, indent=2))
