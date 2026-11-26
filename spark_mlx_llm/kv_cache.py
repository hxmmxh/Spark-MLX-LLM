"""Mixed KV caches used by target-only generation."""

import mlx.core as mx
from mlx_lm.generate import maybe_quantize_kv_cache as mlx_maybe_quantize_kv_cache
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache
from mlx_lm.models.cache import make_prompt_cache as mlx_make_prompt_cache


def rotate_hadamard(x):
    """Apply a self-inverse, normalized Hadamard transform to each head."""
    dim = x.shape[-1]
    # For non-power-of-two heads, use the largest power-of-two block that tiles
    # the head. Each block is orthonormal and uses the same transform on read.
    block_size = dim & -dim
    if block_size == dim:
        return mx.hadamard_transform(x)
    return mx.hadamard_transform(x.reshape(-1, block_size)).reshape(x.shape)


class HadamardQuantizedKVCache(QuantizedKVCache):
    """MLX affine INT8 storage in the Hadamard basis."""

    def __init__(self, group_size=32):
        super().__init__(group_size=group_size, bits=8)

    def update_and_fetch(self, keys, values):
        if keys.shape[-1] % self.group_size or values.shape[-1] % self.group_size:
            raise ValueError(
                "Hadamard KV head dimensions must be divisible by kv_group_size"
            )
        return super().update_and_fetch(rotate_hadamard(keys), rotate_hadamard(values))


class _UnquantizedRotatingKVCache(RotatingKVCache):
    """Preserve sliding-window state during upstream KV quantization checks."""

    def to_quantized(self, group_size=32, bits=8):
        return self


def make_prompt_cache(
    model,
    kv_quantization_scope="full-attention",
    *,
    kv_hadamard=False,
    prompt_cache=None,
    max_kv_size=None,
):
    """Prepare mixed caches for conversion at generation step boundaries.

    A supplied cache list is updated in place so repeated generation continues
    the same cache. Newly created caches preserve all rotating-cache state.
    """
    if kv_quantization_scope not in ("all", "full-attention"):
        raise ValueError(f"Unknown KV quantization scope: {kv_quantization_scope}")
    caches = (
        mlx_make_prompt_cache(model, max_kv_size)
        if prompt_cache is None
        else prompt_cache
    )
    replacements = []
    for index, cache in enumerate(caches):
        if isinstance(cache, RotatingKVCache):
            if kv_quantization_scope == "all":
                raise ValueError(
                    "KV quantization scope 'all' includes unsupported RotatingKVCache; "
                    "use --kv-quantization-scope full-attention"
                )
            if type(cache) is _UnquantizedRotatingKVCache:
                continue
            if type(cache) is not RotatingKVCache:
                raise ValueError(
                    "KV quantization scope 'full-attention' cannot preserve custom "
                    f"RotatingKVCache subclass {type(cache).__name__}"
                )
            preserved = _UnquantizedRotatingKVCache.__new__(_UnquantizedRotatingKVCache)
            preserved.__dict__.update(cache.__dict__)
            replacements.append((index, preserved))
        elif isinstance(cache, ArraysCache):
            # Spark3 keeps the mRoPE position delta in a separate cache entry.
            continue
        elif kv_hadamard and not isinstance(cache, (KVCache, HadamardQuantizedKVCache)):
            raise ValueError(
                f"KV cache layer {index} ({type(cache).__name__}) does not support "
                "Hadamard KV quantization"
            )
        elif not hasattr(cache, "to_quantized") and not isinstance(
            cache, QuantizedKVCache
        ):
            raise ValueError(
                f"KV cache layer {index} ({type(cache).__name__}) does not support KV quantization"
            )
    for index, cache in replacements:
        caches[index] = cache
    return caches


def maybe_quantize_kv_cache(caches, start, group_size, bits, *, kv_hadamard=False):
    """Convert eligible full-attention caches once, after a prefill or decode step."""
    if not kv_hadamard:
        mlx_maybe_quantize_kv_cache(caches, start, group_size, bits)
        return
    if bits != 8:
        raise ValueError("kv_hadamard requires kv_bits=8")

    start = max(start, 1)
    replacements = []
    for index, cache in enumerate(caches):
        if isinstance(cache, KVCache) and cache.offset >= start:
            quantized = HadamardQuantizedKVCache(group_size)
            # Rotate only active history, excluding spare capacity and trimmed tokens.
            quantized.update_and_fetch(*cache.state)
            replacements.append((index, quantized))
    for index, cache in replacements:
        caches[index] = cache
