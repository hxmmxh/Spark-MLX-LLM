"""Signed-byte / FP32 Metal source for the public per-channel projection.

Preserves the independently validated M1..8 dot/postscale arithmetic, including
odd-K tails and offset INT8 byte addressing. Specialization is by templates.
"""

FP32_PER_CHANNEL = r"""
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint n0 = (threadgroup_position_in_grid.x * GROUPS + sg) * ROWS;
float sums[M][ROWS];
for (uint m = 0; m < M; ++m)
    for (uint r = 0; r < ROWS; ++r) sums[m][r] = 0.0f;
for (uint k = lane * 4; k < K; k += 128) {
    float4 a[M];
    for (uint m = 0; m < M; ++m) {
        a[m] = float4(x[m * K + k],
                      k + 1 < K ? x[m * K + k + 1] : 0.0f,
                      k + 2 < K ? x[m * K + k + 2] : 0.0f,
                      k + 3 < K ? x[m * K + k + 3] : 0.0f);
    }
    for (uint r = 0; r < ROWS; ++r) {
        if (n0 + r < N) {
            const size_t p = size_t(n0 + r) * K + k;
            // Scalar byte addressing is valid for odd K and nonzero offsets.
            const float4 w = float4(float(weight[p]),
                k + 1 < K ? float(weight[p + 1]) : 0.0f,
                k + 2 < K ? float(weight[p + 2]) : 0.0f,
                k + 3 < K ? float(weight[p + 3]) : 0.0f);
            for (uint m = 0; m < M; ++m) sums[m][r] += dot(a[m], w);
        }
    }
}
for (uint m = 0; m < M; ++m) {
    for (uint r = 0; r < ROWS; ++r) {
        const float sum = simd_sum(sums[m][r]);
        if (lane == 0 && n0 + r < N)
            y[m * N + n0 + r] = sum * scales[n0 + r];
    }
}
"""
