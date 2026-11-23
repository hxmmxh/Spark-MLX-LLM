"""Mixed ordinary KV caches, used by the common autoregressive engine."""

from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache
from mlx_lm.models.cache import make_prompt_cache as mlx_make_prompt_cache

class _UnquantizedRotatingKVCache(RotatingKVCache):
    """Preserve sliding-window state during upstream KV quantization checks."""

    def to_quantized(self, group_size=32, bits=8):
        return self


def make_prompt_cache(
    model,
    kv_quantization_scope="full-attention",
    *,
    prompt_cache=None,
    max_kv_size=None,
):
    """Prepare mixed caches; conversion still happens at MLX-LM's step boundaries.

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
        elif not hasattr(cache, "to_quantized") and not isinstance(
            cache, QuantizedKVCache
        ):
            raise ValueError(
                f"KV cache layer {index} ({type(cache).__name__}) does not support KV quantization"
            )
    for index, cache in replacements:
        caches[index] = cache
    return caches
