"""两个直接分块的 Metal kernel：FC1+GELU、FC2+残差。"""

import mlx.core as mx

from ._nax_gemm import (
    BIAS_GELU,
    BIAS_RESIDUAL,
    _mlp_FC1_TILE,
    _mlp_FC2_TILE,
    _project,
)


def apply_mlp_with_residual(inputs):
    """依次执行 FC1 与近似 GELU、FC2 与残差相加。

    FC1 使用 ``BIAS_GELU``，加 bias 后计算近似 GELU，生成
    ``[N, 4304]`` 的中间激活。FC2 使用 ``BIAS_RESIDUAL``，加 bias 后加残差。

    Args:
        inputs (dict[str, mx.array]): BF16 输入字典。包含 norm2 输出
            ``x[N, 1152]``、参数 ``w1[4304, 1152]``、``b1[4304]``、
            ``w2[1152, 4304]`` 和 ``b2[1152]``，以及投影段产生的
            ``residual[N, 1152]``。N 为单张图像的 patch 数，须大于 64。

    Returns:
        dict[str, mx.array]: ``y`` 是 FC2 输出与残差之和，形状为
            ``[N, 1152]``，数据类型为 BF16。

    Raises:
        KeyError: inputs 缺少必需字段。
        ValueError: GPU 架构、输入形状、数据类型或 patch 数不受支持。
    """
    args = tuple(inputs[k] for k in ("x", "w1", "b1", "w2", "b2", "residual"))
    x, w1, b1, w2, b2, residual = args
    if (
        mx.default_device() != mx.gpu
        or mx.device_info(mx.gpu).get("architecture") != "applegpu_g17s"
    ):
        raise ValueError("ViT fusion requires Apple M5 Pro / applegpu_g17s")
    if any(a.dtype != mx.bfloat16 for a in args):
        raise ValueError("ViT fusion requires BF16 inputs and weights")
    if (
        x.ndim != 2
        or x.shape[0] <= 64
        or x.shape[1] != 1152
        or w1.shape != (4304, 1152)
        or b1.shape != (4304,)
        or w2.shape != (1152, 4304)
        or b2.shape != (1152,)
        or residual.shape != x.shape
    ):
        raise ValueError(
            "Expected x/residual[N,1152], w1[4304,1152], w2[1152,4304], matching biases; N>64"
        )
    hidden = _project(x, w1, b1, residual, BIAS_GELU, _mlp_FC1_TILE)
    output = _project(hidden, w2, b2, residual, BIAS_RESIDUAL, _mlp_FC2_TILE)
    return {"y": output}
