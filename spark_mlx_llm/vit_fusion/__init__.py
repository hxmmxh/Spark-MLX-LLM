"""Spark-X3 ViT 的 QKV、Attention、投影归一化与 MLP 分段融合。"""

import importlib.metadata
from functools import cache

import mlx.core as mx

__all__ = ["forward_block", "supports"]


@cache
def _available():
    """检查当前环境是否具备分段融合所需的 MLX 版本和 Metal 设备。

    Returns:
        bool: MLX 版本为 0.32.4、Metal 可用且 GPU 架构为
            ``applegpu_g17s`` 时返回 True。
    """
    return (
        importlib.metadata.version("mlx") == "0.32.4"
        and mx.metal.is_available()
        and mx.device_info(mx.gpu).get("architecture") == "applegpu_g17s"
    )


def supports(block, hidden, cos, sin, lengths):
    """判断一个视觉 Block 和当前输入能否使用分段融合。

    Args:
        block (VisionBlock): 提供注意力、MLP、两次 LayerNorm 及其参数。
        hidden (mx.array): 视觉特征，形状为 ``[N, 1152]``，N 为 patch 数。
        cos (mx.array): 二维 RoPE 的余弦，形状为 ``[N, 1, 36]``。
        sin (mx.array): 二维 RoPE 的正弦，形状与 cos 相同。
        lengths (Sequence[int]): 每张图像的 patch 数，融合路径要求为 ``[N]``。

    Returns:
        bool: 当前默认设备、单图边界、参数形状、BF16 数据类型及
            LayerNorm 的 epsilon 均满足融合条件时返回 True。
    """
    if not _available() or mx.default_device() != mx.gpu:
        return False
    if hidden.ndim != 2 or hidden.shape[1] != 1152 or hidden.shape[0] <= 64:
        return False
    tokens = hidden.shape[0]
    if list(lengths) != [tokens] or block.attn.heads != 16:
        return False
    if cos.shape != (tokens, 1, 36) or sin.shape != cos.shape:
        return False
    if block.norm1.eps != 1e-6 or block.norm2.eps != 1e-6:
        return False
    arrays = [
        (hidden, (tokens, 1152)),
        (cos, (tokens, 1, 36)),
        (sin, (tokens, 1, 36)),
        (block.attn.qkv.weight, (3456, 1152)),
        (block.attn.qkv.bias, (3456,)),
        (block.attn.proj.weight, (1152, 1152)),
        (block.attn.proj.bias, (1152,)),
        (block.mlp.linear_fc1.weight, (4304, 1152)),
        (block.mlp.linear_fc1.bias, (4304,)),
        (block.mlp.linear_fc2.weight, (1152, 4304)),
        (block.mlp.linear_fc2.bias, (1152,)),
        (block.norm1.weight, (1152,)),
        (block.norm1.bias, (1152,)),
        (block.norm2.weight, (1152,)),
        (block.norm2.bias, (1152,)),
    ]
    return all(
        isinstance(a, mx.array) and a.shape == shape and a.dtype == mx.bfloat16
        for a, shape in arrays
    )


def forward_block(block, hidden, cos, sin, lengths):
    """执行 Spark-X3 ViT 的一个分段融合 Block。

    先对输入做 LayerNorm，再依次执行 QKV/RoPE、自注意力、输出投影与
    归一化、MLP。各段通过 MLX 数组依赖连接，保留惰性求值。

    Args:
        block (VisionBlock): 当前 Block，提供计算所需的权重、偏置和归一化参数。
        hidden (mx.array): BF16 输入特征，形状为 ``[N, 1152]``。
        cos (mx.array): BF16 RoPE 余弦，形状为 ``[N, 1, 36]``。
        sin (mx.array): BF16 RoPE 正弦，形状与 cos 相同。
        lengths (Sequence[int]): 每张图像的 patch 数，当前输入为 ``[N]``。

    Returns:
        mx.array: MLP 与残差相加后的 BF16 特征，形状为 ``[N, 1152]``。

    Raises:
        ValueError: 设备、数据类型、图像边界或参数布局不满足融合条件。
    """
    if not supports(block, hidden, cos, sin, lengths):
        raise ValueError("Unsupported device, dtype, image boundary or ViT shape")
    from .attn_output_proj_norm import project_attn_output_and_normalize
    from .mlp_residual import apply_mlp_with_residual
    from .qkv_rope import project_qkv_with_rope
    from .self_attention import compute_self_attention

    packed = project_qkv_with_rope(
        {
            "x": block.norm1(hidden),
            "weight": block.attn.qkv.weight,
            "bias": block.attn.qkv.bias,
            "cos": cos,
            "sin": sin,
        }
    )
    context = compute_self_attention(packed)["y"]
    projected = project_attn_output_and_normalize(
        {
            "x": context,
            "weight": block.attn.proj.weight,
            "bias": block.attn.proj.bias,
            "residual": hidden,
            "gamma": block.norm2.weight,
            "beta": block.norm2.bias,
        }
    )
    # FC1 接收 norm2 的输出 z，FC2 将投影段保留的 h 加到输出上。
    result = apply_mlp_with_residual(
        {
            "x": projected["z"],
            "residual": projected["h"],
            "w1": block.mlp.linear_fc1.weight,
            "b1": block.mlp.linear_fc1.bias,
            "w2": block.mlp.linear_fc2.weight,
            "b2": block.mlp.linear_fc2.bias,
        }
    )
    return result["y"]
