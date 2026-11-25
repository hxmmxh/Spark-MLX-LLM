"""直接分块投影加残差，再调用原生 LayerNorm。"""

import mlx.core as mx

from ._nax_gemm import BIAS_RESIDUAL, _project

_TILE = (16, 32, 2, 4, 128)


def project_attn_output_and_normalize(inputs):
    """执行 Attention 输出投影和残差相加，再执行 Block 的第二次 LayerNorm。

    输出投影使用 ``BIAS_RESIDUAL``，在同一个 GEMM kernel 中先加 bias，
    再加 Block 输入。LayerNorm 对该 kernel 的输出做归一化。

    Args:
        inputs (dict[str, mx.array]): BF16 输入字典。包含注意力特征
            ``x[N, 1152]``、输出投影参数 ``weight[1152, 1152]`` 和
            ``bias[1152]``、Block 输入 ``residual[N, 1152]``，
            以及 LayerNorm 的缩放 ``gamma[1152]`` 和偏置 ``beta[1152]``。
            N 为单张图像的 patch 数，须大于 64。

    Returns:
        dict[str, mx.array]: ``h`` 为投影结果与残差之和，``z`` 为
            LayerNorm 的输出，epsilon 取 1e-6。两者均为 ``[N, 1152]`` 的
            BF16 数组，分别用作 MLP 的残差和 FC1 输入。

    Raises:
        KeyError: inputs 缺少必需字段。
        ValueError: GPU 架构、输入形状、数据类型或 patch 数不受支持。
    """
    x, weight, bias, residual, gamma, beta = (
        inputs[k] for k in ("x", "weight", "bias", "residual", "gamma", "beta")
    )
    if (
        mx.default_device() != mx.gpu
        or mx.device_info(mx.gpu).get("architecture") != "applegpu_g17s"
    ):
        raise ValueError("ViT fusion requires Apple M5 Pro / applegpu_g17s")
    if any(a.dtype != mx.bfloat16 for a in (x, weight, bias, residual, gamma, beta)):
        raise ValueError("ViT fusion requires BF16 inputs and weights")
    if (
        x.ndim != 2
        or x.shape[0] <= 64
        or x.shape[1] != 1152
        or weight.shape != (1152, 1152)
        or bias.shape != (1152,)
        or residual.shape != x.shape
        or gamma.shape != (1152,)
        or beta.shape != (1152,)
    ):
        raise ValueError(
            "Expected x/residual[N,1152], weight[1152,1152], bias/gamma/beta[1152]; N>64"
        )
    hidden = _project(x, weight, bias, residual, BIAS_RESIDUAL, _TILE)
    normalized = mx.fast.layer_norm(hidden, gamma, beta, 1e-6)
    return {"h": hidden, "z": normalized}
