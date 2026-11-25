"""共享 NAX GEMM、GELU 舍入和 MLX 私有头文件加载。"""

import re as _mlp_re
from functools import cache as _mlp_cache
from pathlib import Path as _mlp_Path

import mlx.core as _mlp_mx

BIAS_GELU = 1
BIAS_RESIDUAL = 2

# 分块参数依次为 SIMD 组输出块的高宽、线程组内 SIMD 组的行列数和 K 轴块宽。
_mlp_FC1_TILE = (32, 64, 2, 2, 64)


_mlp_FC2_TILE = (32, 32, 2, 2, 128)


_mlp__HEADER = r"""
using namespace mlx::steel;

template <typename T>
METAL_FUNC T spark_gelu(T x) {
    // Match the dtype-rounded arithmetic of nn.gelu_approx's compiled graph.
    T cube = T(metal::pow(float(x), 3.0f));
    T cubic = T(float(T(0.044715f)) * float(cube));
    T argument = T(float(T(0.7978845608028654f)) *
        float(T(float(x) + float(cubic))));
    T activation = T(metal::precise::tanh(float(argument)));
    T half_x = T(float(T(0.5f)) * float(x));
    return T(float(half_x) * float(T(1.0f + float(activation))));
}
"""


_mlp__SOURCE = r"""
const uint sg = simdgroup_index_in_threadgroup;
const uint3 group = threadgroup_position_in_grid;
const int row0 = int(group.y) * (SM * WM) + int(sg / WN) * SM;
const int col0 = int(group.x) * (SN * WN) + int(sg % WN) * SN;
const short rows = short(min(SM, M - row0));
const short cols = short(min(SN, N - col0));

// All SIMD groups enter gemm_loop, including empty edge groups, because its
// aligned K loop contains a threadgroup barrier. Never return early here.
// MLX 0.32.4/g17s selects two split-K partitions (4096 + 208)
// for these ViT FC2 shapes. Preserve their FP32 accumulation boundaries
// inside the same worker; retain the fused bias and residual epilogue.
constexpr bool SPLIT_FC2 = K == 4304 && N == 1152 && M > 64 && K >= 3 * (M > N ? M : N);
constexpr int FIRST_K = SPLIT_FC2 ? 4096 : K;
NAXTile<float, SM / 16, SN / 16> acc;
dispatch_bool(rows == SM, [&](auto aligned_m) {
    dispatch_bool(cols == SN, [&](auto aligned_n) {
        acc = gemm_loop<T, SM, SN, 32, BK, false, true,
            aligned_m.value, aligned_n.value, (FIRST_K % BK == 0), float>(
                x + size_t(row0) * K, weight + size_t(col0) * K,
                K, K, FIRST_K, FIRST_K / BK, rows, cols);
        if constexpr (SPLIT_FC2) {
            auto tail = gemm_loop<T, SM, SN, 32, BK, false, true,
                aligned_m.value, aligned_n.value, (208 % BK == 0), float>(
                    x + size_t(row0) * K + 4096, weight + size_t(col0) * K + 4096,
                    K, K, 208, 208 / BK, rows, cols);
            STEEL_PRAGMA_UNROLL
            for (short fm = 0; fm < SM / 16; ++fm) {
                STEEL_PRAGMA_UNROLL
                for (short fn = 0; fn < SN / 16; ++fn) {
                    thread auto& first = acc.frag_at(fm, fn);
                    thread auto& last = tail.frag_at(fm, fn);
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; ++i)
                        first[i] += last[i];
                }
            }
        }
    });
});

STEEL_PRAGMA_UNROLL
for (short fm = 0; fm < SM / 16; ++fm) {
    STEEL_PRAGMA_UNROLL
    for (short fn = 0; fn < SN / 16; ++fn) {
        thread auto& values = acc.frag_at(fm, fn);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; ++i) {
            const short2 coordinate = BaseNAXFrag::get_coord(i);
            const int row = row0 + fm * 16 + coordinate.y;
            const int col = col0 + fn * 16 + coordinate.x;
            if (row < M && col < N) {
                // Native Linear/addmm adds bias to the FP32 accumulator, then
                // rounds to T before either GELU or residual addition.
                // MLX0.32.4's M=1 GEMV fallback rounds its dot product before
                // adding bias; GEMM's addmm epilogue adds bias before rounding.
                float dot = M == 1 ? float(T(values[i])) : values[i];
                T value = T(dot + float(bias[col]));
                if constexpr (EPILOGUE == BIAS_GELU) value = spark_gelu(value);
                if constexpr (EPILOGUE == BIAS_RESIDUAL)
                    value = T(float(value) + float(residual[size_t(row) * N + col]));
                out[size_t(row) * N + col] = value;
            }
        }
    }
}
"""


@_mlp_cache
def _mlp__steel_header():
    """展开并缓存 MLX 的 NAX GEMM 头文件。

    自定义 Metal kernel 通过源码字符串接收 MLX 头文件。递归展开时使用
    路径集合处理重复引用，并移除文件级的 ``#pragma once``。

    Returns:
        str: 展开后的 NAX GEMM 头文件源码。

    Raises:
        OSError: MLX 安装目录中的头文件无法读取。
    """
    root = _mlp_Path(_mlp_mx.__file__).parent / "include"
    seen = set()

    def expand(name):
        """按依赖顺序展开一个头文件及其中的 MLX 引用。

        Args:
            name (str): 相对于 MLX include 目录的头文件路径。

        Returns:
            str: 展开后的源码。路径已处理时返回空字符串。
        """
        if name in seen:
            return ""
        seen.add(name)
        text = (root / name).read_text().replace("#pragma once", "")
        return _mlp_re.sub(
            '^#include "(mlx/[^\\"]+)"',
            lambda match: expand(match.group(1)),
            text,
            flags=_mlp_re.MULTILINE,
        )

    return expand("mlx/backend/metal/kernels/steel/gemm/gemm_nax.h")


@_mlp_cache
def _kernel(epilogue):
    """按 GEMM 写回模式创建并缓存 Metal kernel。

    Args:
        epilogue (int): ``BIAS_GELU`` 用于加 bias 后计算近似 GELU。
            ``BIAS_RESIDUAL`` 用于加 bias 后加残差。

    Returns:
        Callable: 接收矩阵、分块参数和调度配置，生成投影结果。
    """
    epilogue_constants = (
        f"constexpr int BIAS_GELU = {BIAS_GELU};\n"
        f"constexpr int BIAS_RESIDUAL = {BIAS_RESIDUAL};\n"
    )
    return _mlp_mx.fast.metal_kernel(
        name=f"spark_vision_mlp_direct_{epilogue}",
        input_names=["x", "weight", "bias", "residual"],
        output_names=["out"],
        header=_mlp__steel_header() + _mlp__HEADER,
        source=epilogue_constants + _mlp__SOURCE,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def _project(x, weight, bias, residual, epilogue, tile):
    """调度 NAX 矩阵乘法，并在写回时按指定模式处理输出。

    Args:
        x (mx.array): 输入矩阵，形状为 ``[M, K]``。
        weight (mx.array): 线性层权重，形状为 ``[N, K]``。计算使用其转置。
        bias (mx.array): 每个输出通道的偏置，形状为 ``[N]``。
        residual (mx.array): ``BIAS_RESIDUAL`` 将形状为 ``[M, N]`` 的残差
            加到输出。``BIAS_GELU`` 仅传入该缓冲以复用 kernel 的参数布局。
        epilogue (int): 写回模式。GEMM 先加 bias 并按输入数据类型舍入。
            ``BIAS_GELU`` 接着计算近似 GELU，``BIAS_RESIDUAL`` 接着加残差。
        tile (tuple[int, int, int, int, int]): ``(SM, SN, WM, WN, BK)``。
            SM 和 SN 为单个 SIMD 组的输出块高宽。WM 和 WN 为线程组内
            的 SIMD 组排列。BK 为 K 轴分块大小。

    Returns:
        mx.array: 形状为 ``[M, N]`` 的投影结果，数据类型与 x 相同。
    """
    m, k = x.shape
    n = weight.shape[0]
    sm, sn, wm, wn, bk = tile
    return _kernel(epilogue)(
        inputs=[x, weight, bias, residual],
        template=[
            ("T", x.dtype),
            ("M", m),
            ("N", n),
            ("K", k),
            ("SM", sm),
            ("SN", sn),
            ("WM", wm),
            ("WN", wn),
            ("BK", bk),
            ("EPILOGUE", epilogue),
        ],
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
        grid=(
            ((n + sn * wn - 1) // (sn * wn)) * 32,
            ((m + sm * wm - 1) // (sm * wm)) * (wm * wn),
            1,
        ),
        threadgroup=(32, wm * wn, 1),
    )[0]
