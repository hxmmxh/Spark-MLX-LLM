"""Direct signed-INT8 projection with FP32 activations, reduction and output."""

from functools import lru_cache
from math import prod

import mlx.core as mx

from ._types import StreamOrDevice
from ._validation import gpu_stream
from ._w8a32_per_channel_sources import FP32_PER_CHANNEL

__all__ = ["w8a32_per_channel"]


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="spark_w8a32_per_channel",
        input_names=["x", "weight", "scales"],
        output_names=["y"],
        source=FP32_PER_CHANNEL,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def w8a32_per_channel(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    *,
    bias: mx.array | None = None,
    stream: StreamOrDevice = None,
) -> mx.array:
    """Return lazy FP32 ``(x @ weight.T) * scales + bias`` for 1..8 rows.

    ``x`` is FP32 ``[...,K]``, ``weight`` signed INT8 ``[N,K]``, and
    ``scales`` FP32 ``[N]`` or ``[N,1]``. N/K may be any positive size;
    the product of x's leading dimensions must be 1..8. A vector counts as
    one row. Output preserves those leading dimensions and ends in N.
    Optional bias is FP16, BF16 or FP32 ``[N]`` and is added separately in
    FP32, after the dot product and channel scale.

    No FP32 weight matrix or quantized activation is materialized. Each
    signed-weight quartet is reused across input rows. MLX lazily copies
    noncontiguous inputs as needed; unaligned offsets and N/K tails are
    supported. Inputs are never modified, evaluated or synchronized here.
    Unsupported descriptors and CPU execution raise; callers own fallback.

    This inference-only operator has no autodiff rule. FP32 reduction order
    can differ from native MLX GEMM, which may also use TF32 by default.
    The stored integer codes and channel scales are preserved; bitwise
    output or generation-quality equivalence is not implied.
    """
    if any(not isinstance(value, mx.array) for value in (x, weight, scales)):
        raise TypeError("x, weight and scales must be MLX arrays")
    if x.dtype != mx.float32 or weight.dtype != mx.int8 or scales.dtype != mx.float32:
        raise ValueError("W8A32 requires FP32 x, signed INT8 weight and FP32 scales")
    if x.ndim < 1 or weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError("W8A32 requires x[...,K] and nonempty weight[N,K]")
    n, k = weight.shape
    m = prod(x.shape[:-1])
    if x.shape[-1] != k or not 1 <= m <= 8:
        raise ValueError("W8A32 requires matching positive K and 1..8 input rows")
    if scales.shape not in ((n,), (n, 1)):
        raise ValueError("W8A32 requires FP32 scales [N] or [N,1]")
    if bias is not None and (
        not isinstance(bias, mx.array)
        or bias.shape != (n,)
        or bias.dtype not in (mx.float16, mx.bfloat16, mx.float32)
    ):
        raise ValueError("W8A32 bias must be FP16, BF16 or FP32 [N]")
    stream = gpu_stream(stream, operator="W8A32 per-channel")
    rows, groups = (4, 2) if m == 1 else (2, 4)
    value = _kernel()(
        inputs=[x, weight, scales.reshape(-1)],
        template=[("M", m), ("N", n), ("K", k), ("ROWS", rows), ("GROUPS", groups)],
        grid=(((n + rows * groups - 1) // (rows * groups)) * groups * 32, 1, 1),
        threadgroup=(groups * 32, 1, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[mx.float32],
        stream=stream,
    )[0]
    return mx.add(value, bias, stream=stream) if bias is not None else value
