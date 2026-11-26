"""Tile bounds for the X3 512-token sliding-window prefill on M5 Pro."""

import re
from functools import cache
from pathlib import Path

import mlx.core as mx

_HEADER = "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

_PARAMS = r"""
using T = bfloat;
using AccumType = float;
using MaskType = bool;
constexpr int BD = 256, BQ = 64, BK = 32, WM = 4, WN = 2, HEADS = 16;
constexpr bool align_Q = QL % BQ == 0, align_K = KL % BK == 0;
constexpr bool has_mask = false, do_causal = true;
constexpr bool has_sinks = false, save_lse = false;
constexpr AttnParams param_data{
    1, HEADS, BD, QL, KL, HEADS, 0.0625f,
    (QL + BQ - 1) / BQ, (KL + BK - 1) / BK,
    QL / BQ, KL / BK, QL % BQ, KL % BK, KL - QL,
    {HEADS * QL * BD, QL * BD, BD}, {KL * BD, KL * BD, BD},
    {KL * BD, KL * BD, BD}, {QL * HEADS * BD, BD, HEADS * BD}
};
const thread AttnParams* params = &param_data;
const thread AttnMaskParams* mask_params = nullptr;
const device MaskType* mask = nullptr;
const device T* Q = q;
const device T* K = k;
const device T* V = v;
const device T* sinks = nullptr;
device float* LSE = nullptr;
device T* O = out;
const uint simd_lane_id = thread_index_in_simdgroup;
const uint simd_group_id = simdgroup_index_in_threadgroup;
const uint3 tid = threadgroup_position_in_grid;
const uint3 lid = thread_position_in_threadgroup;
"""

_SKIP = r"""
  // The original causal upper bound excludes tiles after the last query.
  // Advance to the tile containing the first query's left window edge.
  const int q_start = int(tid.x) * BQ + params->qL_off;
  const int first_blk = max(0, q_start - 511) / BK;
  const int last_window_blk = (q_start + BQ - 1 - 511) / BK;
  K += first_blk * BK * int(params->K_strides[2]);
  V += first_blk * BK * int(params->V_strides[2]);
  for (int kb = first_blk; kb < kb_lim; kb++) {
"""

_WINDOW_MASK = r"""
    if (kb <= last_window_blk) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      const int base_row = int(tid.x) * BQ + params->qL_off + tm;
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ++ik) {
        thread auto& fg = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ++ii) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; ++jj) {
            const int r = base_row + ii * stile_t::kFragRowsJump + sm;
            const int c = base_col + ik * kU + jj + sn;
            if (c < r - 511) fg[ii * stile_t::kFragThrCols + jj] = neg_inf;
          }
        }
      }
    }
"""


def _source():
    # MLX supplies the NAX fragment layout and online softmax through its
    # installed Metal headers. Retain its Q/K tail loads and causal masking.
    root = Path(mx.__file__).parent / "include"
    seen = set()

    def expand(name):
        if name in seen:
            return ""
        seen.add(name)
        text = (root / name).read_text().replace("#pragma once", "")
        return re.sub(
            r'^#include "(mlx/[^"]+)"',
            lambda match: expand(match[1]),
            text,
            flags=re.MULTILINE,
        )

    text = expand(_HEADER)
    first = text.index("// clang-format off\ntemplate <")
    header = re.sub(
        r"^constant bool \w+ \[\[function_constant\(\d+\)\]\];$",
        "",
        text[:first],
        flags=re.MULTILINE,
    )
    dsplit = text.split("void attention_nax_dsplit(", 1)[1]
    body = dsplit.split("{ // clang-format on", 1)[1].strip().removesuffix("}")
    old_loop = "  for (int kb = 0; kb < kb_lim; kb++) {"
    old_mask = "    // Other masking as needed"
    if body.count(old_loop) != 1 or body.count(old_mask) != 1:
        raise ValueError("Unexpected MLX D256 attention source")
    body = body.replace(old_loop, _SKIP).replace(old_mask, _WINDOW_MASK + "\n" + old_mask)
    return header, _PARAMS + body


@cache
def _kernel():
    try:
        header, source = _source()
    except (OSError, ValueError, IndexError):
        return None
    return mx.fast.metal_kernel(
        name="spark3_swa_prefill_skip_kv",
        input_names=["q", "k", "v"],
        output_names=["out"],
        header=header,
        source=source,
        compile_options={"math_mode": "fast"},
    )


@cache
def _supported_gpu():
    return (
        mx.__version__ == "0.32.4"
        and mx.metal.is_available()
        and mx.device_info(mx.gpu).get("architecture") == "applegpu_g17s"
    )


def try_prefill(q, k, v):
    """Use the specialized kernel for chronological BF16 X3 Q/K/V."""
    batch, heads, ql, dim = q.shape
    kl = k.shape[2]
    if (
        batch != 1 or heads != 16 or dim != 256 or ql < 1024
        or not 0 <= kl - ql <= 511
        or k.shape != (1, 1, kl, dim) or v.shape != k.shape
        or q.dtype != mx.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
        or mx.default_device() != mx.gpu or not _supported_gpu()
    ):
        return None
    kernel = _kernel()
    if kernel is None:
        return None
    return kernel(
        inputs=[q, k, v],
        template=[("QL", ql), ("KL", kl)],
        grid=(((ql + 63) // 64) * 256, heads, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, ql, heads, dim)],
        output_dtypes=[q.dtype],
    )[0].transpose(0, 2, 1, 3)
