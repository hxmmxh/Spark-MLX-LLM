"""QKV GEMM、bias、二维 RoPE 和 D72 重排。"""

from functools import cache

import mlx.core as mx

from ._nax_gemm import _mlp__steel_header

# GEMM 将 QKV 分块累加到寄存器，写回阶段直接处理每对偶数与奇数通道。
_GEMM = r"""
using namespace mlx::steel;
const uint sg = simdgroup_index_in_threadgroup;
const uint3 group = threadgroup_position_in_grid;
const int row0 = int(group.y) * (SM * WM) + int(sg / WN) * SM;
const int col0 = int(group.x) * (SN * WN) + int(sg % WN) * SN;
const short rows = short(max(0, min(SM, M - row0)));
const short cols = short(max(0, min(SN, N - col0)));
// All SIMD groups enter GEMM's barriers, including empty edge groups.
NAXTile<float, SM / 16, SN / 16> acc;
dispatch_bool(rows == SM, [&](auto aligned_m) {
    dispatch_bool(cols == SN, [&](auto aligned_n) {
        acc = gemm_loop<T, SM, SN, 32, BK, false, true,
            aligned_m.value, aligned_n.value, (D % BK == 0), float>(
                x + size_t(row0) * D, weight + size_t(col0) * D,
                D, D, D, D / BK, rows, cols);
    });
});
"""


_FUSED = r"""
STEEL_PRAGMA_UNROLL
for (short fm = 0; fm < SM / 16; ++fm) {
    STEEL_PRAGMA_UNROLL
    for (short fn = 0; fn < SN / 16; ++fn) {
        thread auto& values = acc.frag_at(fm, fn);
        // Adjacent even/odd channels are owned by the same NAX lane.
        // Head width72 and every output-tile origin are even.
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i += 2) {
            const short2 c = BaseNAXFrag::get_coord(i);
            const int row = row0 + fm * 16 + c.y;
            const int col = col0 + fn * 16 + c.x;
            if (row < M && col + 1 < N) {
                const int component = col / (H * 72);
                const int within = col % (H * 72);
                const int head = within / 72;
                const int channel = within % 72;
                const size_t dst = (size_t(head) * M + row) * P + channel;
                const float de = M == 1 ? float(T(values[i])) : values[i];
                const float do_ = M == 1 ? float(T(values[i + 1])) : values[i + 1];
                const T even = T(de + float(bias[col]));
                const T odd = T(do_ + float(bias[col + 1]));
                device T* output = component == 0 ? q : (component == 1 ? k : v);
                if (component < 2) {
                    const size_t pos = size_t(row) * 36 + channel / 2;
                    const float cs = float(cos[pos]), ss = float(sin[pos]);
                    // Preserve the input-dtype products and final rounding.
                    const T ec = T(float(even) * cs), os = T(float(odd) * ss);
                    const T es = T(float(even) * ss), oc = T(float(odd) * cs);
                    output[dst] = T(float(ec) - float(os));
                    output[dst + 1] = T(float(es) + float(oc));
                } else {
                    output[dst] = even;
                    output[dst + 1] = odd;
                }
                if constexpr (P == 96) {
                    // The first24 real lanes own their head's24 padding lanes.
                    if (channel < 24) {
                        output[dst + 72] = T(0);
                        output[dst + 73] = T(0);
                    }
                }
            }
        }
    }
}
"""


@cache
def _kernel():
    """创建并缓存带 bias、RoPE 和布局转换的 QKV kernel。

    Returns:
        Callable: 接收 Metal 模板参数和调度配置，生成 q、k、v 三个数组。
    """
    return mx.fast.metal_kernel(
        name="spark_vision_qkv_project_rope_pack",
        input_names=["x", "weight", "bias", "cos", "sin"],
        output_names=["q", "k", "v"],
        source=_GEMM + _FUSED,
        header=_mlp__steel_header(),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def project_qkv_with_rope(inputs):
    """计算 QKV 投影，对 Q 和 K 应用二维 RoPE，并按注意力头排布结果。

    Args:
        inputs (dict[str, mx.array]): BF16 输入字典。包含特征
            ``x[N, 1152]``、权重 ``weight[3456, 1152]``、偏置
            ``bias[3456]`` 和旋转系数 ``cos/sin[N, 1, 36]``。
            N 为单张图像的 patch 数，须大于 64。

    Returns:
        dict[str, mx.array]: ``q``、``k``、``v`` 均为连续的 BF16 数组，
            形状为 ``[1, 16, N, 72]``。

    Raises:
        KeyError: inputs 缺少必需字段。
        ValueError: 输入形状、数据类型或 patch 数不匹配。
    """
    x, weight, bias, cos, sin = (
        inputs[k] for k in ("x", "weight", "bias", "cos", "sin")
    )
    m = x.shape[0]
    if x.shape != (m, 1152) or weight.shape != (3456, 1152) or bias.shape != (3456,):
        raise ValueError("Expected x[N,1152], weight[3456,1152], bias[3456]")
    if cos.shape != (m, 1, 36) or sin.shape != cos.shape:
        raise ValueError("Expected cos/sin[N,1,36]")
    if m <= 64 or any(a.dtype != mx.bfloat16 for a in (x, weight, bias, cos, sin)):
        raise ValueError("ViT fusion requires BF16 and N>64")
    shape = (1, 16, m, 72)
    values = _kernel()(
        inputs=[x, weight, bias, cos, sin],
        template=[
            ("T", x.dtype),
            ("M", m),
            ("D", 1152),
            ("N", 3456),
            ("H", 16),
            ("P", 72),
            ("SM", 16),
            ("SN", 32),
            ("WM", 2),
            ("WN", 2),
            ("BK", 64),
        ],
        output_shapes=[shape] * 3,
        output_dtypes=[x.dtype] * 3,
        grid=(54 * 32, ((m + 31) // 32) * 4, 1),
        threadgroup=(32, 4, 1),
    )
    return dict(zip(("q", "k", "v"), values))
