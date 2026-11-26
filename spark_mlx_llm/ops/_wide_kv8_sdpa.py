"""Metal TensorOps decode over affine INT8/G32 KV for GQA 16 and 32."""

from functools import cache
from platform import mac_ver

import mlx.core as mx

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// The fragment layout follows MLX steel/gemm/nax.h (Apple Inc., MIT).
// A 16x16 matrix fragment maps two groups of four elements to each lane.
template <typename T> using Frag = metal::vec<T, 8>;

template <typename T, bool Transpose>
inline void wide_mma(
    thread Frag<float>& c0, thread Frag<float>& c1, const thread Frag<T>& a,
    const thread Frag<T>& b0, const thread Frag<T>& b1
) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, Transpose, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto left = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto right = op.template get_right_input_cooperative_tensor<T, T, float>();
    auto dst = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        left[i] = a[i];
        right[i] = b0[i];
        right[8 + i] = b1[i];
        dst[i] = c0[i];
        dst[8 + i] = c1[i];
    }
    op.run(left, right, dst);
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        c0[i] = dst[i];
        c1[i] = dst[8 + i];
    }
}

template <typename T>
inline Frag<T> load_kv(
    const device uint* data, const device T* scales, const device T* biases,
    uint token, uint dim, uint length, bool aligned
) {
    Frag<T> tile;
    #pragma unroll
    for (uint row = 0; row < 2; ++row) {
        uint t = token + row * 8;
        float4 val(0.0f);
        if (aligned || t < length) {
            uint p = data[t * 64 + dim / 4];
            float s = float(scales[t * 8 + dim / 32]);
            float b = float(biases[t * 8 + dim / 32]);
            val = fma(float4(as_type<uchar4>(p)), float4(s), float4(b));
        }
        #pragma unroll
        for (uint j = 0; j < 4; ++j) tile[row * 4 + j] = T(val[j]);
    }
    return tile;
}
"""

_DECODE = r"""
constexpr uint TILE = 32;
const uint lane = thread_index_in_simdgroup;
const uint head_tile = threadgroup_position_in_grid.y;
const uint split = threadgroup_position_in_grid.x;
const uint kh = head_tile / (G / 16);
const uint head0 = head_tile * 16;
const uint N = keys_shape[2];
const uint fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
const uint fn = (((lane >> 2) & 2) | (lane & 1)) * 4;

const ulong kv_offset = ulong(kh) * N;
const device T* Q = queries + ulong(head0) * 256;
const device uint* K = keys + kv_offset * 64;
const device uint* V = values + kv_offset * 64;
const device T* ks = key_scales + kv_offset * 8;
const device T* kb = key_biases + kv_offset * 8;
const device T* vs = value_scales + kv_offset * 8;
const device T* vb = value_biases + kv_offset * 8;

auto q_frag = [&](uint d) {
    Frag<T> q;
    #pragma unroll
    for (uint r = 0; r < 2; ++r)
        #pragma unroll
        for (uint c = 0; c < 4; ++c)
            q[r * 4 + c] = Q[(fm + 8 * r) * 256 + d * 16 + fn + c];
    return q;
};
Frag<float> acc[16];
#pragma unroll
for (uint d = 0; d < 16; ++d) {
    acc[d] = Frag<float>(0.0f);
}
float2 maxs_local(-MAXFLOAT), sums_local(0.0f);
const uint tiles = (N + TILE - 1) / TILE;
const uint common = tiles / SPLITS, remainder = tiles % SPLITS;
const uint begin = split * common + min(split, remainder);
const uint count = common + uint(split < remainder);
constexpr uint KFRAGS = TILE / 16;
const float q_scale = scale * 1.4426950408889634f;

// All 16 query heads in this tile share the same decoded K and V fragments.
for (uint off = 0; off < count; ++off) {
    const uint first = (begin + off) * TILE;
    Frag<float> scores[KFRAGS];
    #pragma unroll
    for (uint t = 0; t < KFRAGS; ++t)
        scores[t] = Frag<float>(0.0f);
    #pragma clang loop unroll_count(4)
    for (uint d = 0; d < 16; ++d) {
        Frag<T> q = q_frag(d);
        #pragma unroll
        for (uint t = 0; t < KFRAGS; t += 2) {
            const uint token = first + t * 16 + fm, dim = d * 16 + fn;
            auto k0 = load_kv(K, ks, kb, token, dim, N, ALIGNED);
            auto k1 = load_kv(K, ks, kb, token + 16, dim, N, ALIGNED);
            wide_mma<T, true>(scores[t], scores[t + 1], q, k0, k1);
        }
    }

    // Online softmax stays in FP32. Normalize across splits in the merge.
    float2 new_max = maxs_local;
    #pragma unroll
    for (uint t = 0; t < KFRAGS; ++t) {
        #pragma unroll
        for (uint e = 0; e < 8; ++e) {
            scores[t][e] = (ALIGNED || first + t * 16 + fn + e % 4 < N) ?
                scores[t][e] * q_scale : -MAXFLOAT;
            new_max[e / 4] = max(new_max[e / 4], scores[t][e]);
        }
    }
    #pragma unroll
    for (uint r = 0; r < 2; ++r) {
        new_max[r] = max(new_max[r], simd_shuffle_xor(new_max[r], 1));
        new_max[r] = max(new_max[r], simd_shuffle_xor(new_max[r], 8));
    }
    const float2 factor = fast::exp2(maxs_local - new_max);
    maxs_local = new_max;
    float2 tile_sum(0.0f);
    Frag<T> probability[KFRAGS];
    #pragma unroll
    for (uint t = 0; t < KFRAGS; ++t) {
        #pragma unroll
        for (uint e = 0; e < 8; ++e) {
            float p = fast::exp2(scores[t][e] - new_max[e / 4]);
            tile_sum[e / 4] += p;
            probability[t][e] = T(p);
        }
    }
    #pragma unroll
    for (uint r = 0; r < 2; ++r) {
        tile_sum[r] += simd_shuffle_xor(tile_sum[r], 1);
        tile_sum[r] += simd_shuffle_xor(tile_sum[r], 8);
    }
    sums_local = fma(sums_local, factor, tile_sum);

    #pragma unroll
    for (uint d = 0; d < 16; d += 2) {
        #pragma unroll
        for (uint e = 0; e < 8; ++e) {
            acc[d][e] *= factor[e / 4];
            acc[d + 1][e] *= factor[e / 4];
        }
        #pragma unroll
        for (uint t = 0; t < KFRAGS; ++t) {
            const uint token = first + t * 16 + fm, dim = d * 16 + fn;
            auto v0 = load_kv(V, vs, vb, token, dim, N, ALIGNED);
            auto v1 = load_kv(V, vs, vb, token, dim + 16, N, ALIGNED);
            wide_mma<T, false>(acc[d], acc[d + 1], probability[t], v0, v1);
        }
    }
}

#pragma unroll
for (uint r = 0; r < 2; ++r) {
    const uint head = head0 + fm + 8 * r;
    const ulong out_row = ulong(head) * SPLITS + split;
    if (fn == 0) {
        sums[out_row] = sums_local[r];
        maxs[out_row] = maxs_local[r];
    }
    #pragma unroll
    for (uint d = 0; d < 16; ++d)
        #pragma unroll
        for (uint c = 0; c < 4; ++c)
            partials[out_row * 256 + d * 16 + fn + c] = acc[d][r * 4 + c];
}
"""

_MERGE = r"""
const uint dim = thread_position_in_threadgroup.x;
const uint head = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const ulong row = ulong(head) * SPLITS;
threadgroup float warp_max[8], warp_sum[8], weights[SPLITS];
const float m = dim < SPLITS ? maxs[row + dim] : -MAXFLOAT;
const float group_max = simd_max(m);
if (lane == 0) warp_max[sg] = group_max;
threadgroup_barrier(mem_flags::mem_threadgroup);

const float global_max = simd_max(lane < 8 ? warp_max[lane] : -MAXFLOAT);
// Compute one weight per split for the whole head.
float weighted_sum = 0.0f;
if (dim < SPLITS) {
    const float w = fast::exp2(m - global_max);
    weights[dim] = w;
    weighted_sum = w * sums[row + dim];
}
const float group_sum = simd_sum(weighted_sum);
if (lane == 0) warp_sum[sg] = group_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

const float denominator = simd_sum(lane < 8 ? warp_sum[lane] : 0.0f);
const device float* src = partials + row * 256 + dim;
float acc = 0.0f;
#pragma clang loop unroll_count(4)
for (uint s = 0; s < SPLITS; ++s) acc = fma(weights[s], src[s * 256], acc);
out[head * 256 + dim] = T(acc / denominator);
"""


@cache
def available():
    """TensorOps launch policy tuned for the Apple M5 GPU family."""
    if not mx.metal.is_available():
        return False
    try:
        release = tuple(map(int, mac_ver()[0].split(".")[:2]))
        mlx_version = tuple(map(int, mx.__version__.split(".")[:3]))
    except ValueError:
        return False
    return (
        release >= (26, 2)
        and mlx_version >= (0, 32, 4)
        and mx.device_info(mx.gpu).get("architecture", "").startswith("applegpu_g17")
    )


@cache
def _kernels():
    decode = mx.fast.metal_kernel(
        name="spark_kv8_wide_gqa_decode",
        input_names=[
            "queries",
            "keys",
            "key_scales",
            "key_biases",
            "values",
            "value_scales",
            "value_biases",
            "scale",
        ],
        output_names=["partials", "sums", "maxs"],
        header=_HEADER,
        source=_DECODE,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "fast"},
    )
    merge = mx.fast.metal_kernel(
        name="spark_kv8_wide_gqa_merge",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=_MERGE,
        compile_options={"math_mode": "fast"},
    )
    return decode, merge


def _decode_attention(queries, keys, values, scale, splits):
    batch, heads, _, dim = queries.shape
    kv_heads, length = keys[0].shape[1:3]
    decode, merge = _kernels()
    partials = decode(
        inputs=[queries, *keys, *values, mx.array(scale, mx.float32)],
        template=[
            ("T", queries.dtype),
            ("G", heads // kv_heads),
            ("SPLITS", splits),
            ("ALIGNED", length % 32 == 0),
        ],
        grid=(splits * 32, batch * heads // 16, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch * heads, splits, dim),
            (batch * heads, splits),
            (batch * heads, splits),
        ],
        output_dtypes=[mx.float32] * 3,
    )
    return merge(
        inputs=partials,
        template=[("T", queries.dtype), ("SPLITS", splits)],
        grid=(dim, batch * heads, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]


def decode(queries, keys, values, scale):
    kv_heads, length = keys[0].shape[1:3]
    gqa = queries.shape[1] // kv_heads
    # Limit partitions to the number of 32-token tiles. Different head counts
    # have different partial-output costs on M5.
    splits = min((length + 31) // 32, 192 if gqa == 16 else 160)
    return _decode_attention(queries, keys, values, scale, splits)
