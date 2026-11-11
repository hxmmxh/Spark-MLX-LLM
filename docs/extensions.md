# Apple Silicon extension and attention operators

[Project overview](../README.md)

### Spark C++ extension on Apple Silicon

**`spark-mlx-ext`** provides the C++ QKV projection and RoPE path for supported single-token,
batch-size-one affine W4 decoding. The model detects it automatically after
restarting Python; no command-line flag is needed. It follows the
[MLX custom extension mechanism](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
and leaves the official `mlx.core` intact.

The extension is packaged separately and declared as a conditional dependency
of `spark-mlx-llm`. On macOS `arm64` with Python >= 3.11, `uv sync` automatically
builds and installs it from
[`extensions/spark-mlx-ext`](../extensions/spark-mlx-ext). Linux and Python 3.10
skip it. No separate MLX source checkout is needed.

**Install from the checkout.** Prepare an Xcode toolchain with the Metal 4
compiler and MetalPerformancePrimitives SDK headers, then run from the repository root:

```sh
uv sync --locked
```

See the [environment guide](en/environment.md) for the complete setup.
The current build and distributed wheel have these requirements:

| Component | Requirement |
| --- | --- |
| Hardware | Apple Silicon (`arm64`) |
| Python | Standard CPython 3.12 for the `cp312-cp312` wheel |
| MLX | Official `mlx==0.32.4` and `mlx-metal==0.32.4` |
| macOS | 26.2 or newer for the current binaries; tested on 26.5.1 |

The wheel's `macosx_26_0_arm64` filename does not express the backend's
26.2 minimum. Use the requirements in the release notes as well as the filename.

**Build a distribution wheel.** Follow the
[extension build instructions](../extensions/spark-mlx-ext/README.md#build-from-source)
when producing a wheel for release or a different supported Python ABI. This
compiles only the extension. Version 0.2.0 also builds Metal 4 W8A8 kernels
and requires the corresponding Xcode Metal compiler and SDK headers.
The current MLX backend's macOS requirement still applies.

For single-token FP16/BF16 decode, the extension provides
[`w8a16_per_channel`](../extensions/spark-mlx-ext/docs/operators.md#single-token-per-channel-w8a16).
It reads signed INT8 weights with one FP32 scale per output channel, uses
FP32 accumulation, and has no activation quantization or bias. Spark's
out_proj and lm_head shapes have dedicated kernels. This is an explicit API;
installing the extension does not select it in the model automatically.

The extension also exposes an explicit
[W8A8 linear API](../extensions/spark-mlx-ext/docs/operators.md#w8a8-api) for Apple M5+.
Installing it does not automatically change the model's quantization or
replace its output projections.

Production loading automatically selects its W8 routes and ignores historical
checkpoint backend preferences. W8A8 remains an explicit operator API for numerical
diagnostics; its activation scheme can be described by the following metadata:

```json
"model.layers.0.self_attn.out_proj": {
  "group_size": 32,
  "bits": 8,
  "mode": "affine",
  "input_activations": {
    "num_bits": 8,
    "type": "int",
    "strategy": "token",
    "symmetric": true,
    "dynamic": true
  }
}
```

Apply this to all output projections that should use W8A8. The separate
`quantization` map continues to describe the existing packed weights, and
the global `quantize_activations` setting stays false. Entries without
`input_activations` retain their existing W8A16 behavior. This routing is
implemented in the Spark model, using `spark-mlx-ext >= 0.2.0` on Apple M5+.
If the extension cannot be imported, lacks the W8A8 API, or reports that the
device is unsupported, the model loads these projections as ordinary MLX
W8A16 `QuantizedLinear` layers. This fallback uses the existing packed weights
and compiled attention output path without preparing an INT8 weight cache.
With W8A8 support available, the configured layers continue to use W8A8.

The W8A8 adapter accepts UINT32-packed 8-bit weights with repeated per-channel
scales and affine offsets `biases = -128 * scales`. Loading validates this
layout and caches an exact INT8 unpacking plus FP32 channel scales outside
the checkpoint parameter tree. Existing BF16 scale values remain rounded;
converting them to FP32 does not restore their original precision. Both
prefill and single-token decode use INT8 activations. The packed checkpoint
files and `sanitize()` behavior stay the same.

**Verify the installation**, then run your usual generation command:

```sh
.venv/bin/python - <<'PY'
from importlib.metadata import version
import mlx.core as mx
import spark_mlx_ext as ext
from spark_mlx_llm.ops import qkv

assert version("mlx") == version("mlx-metal") == "0.32.4"
assert qkv.spark_qkv_rope is ext.spark_qkv_rope
assert mx.metal.is_available()
print("Extension ready:", version("spark-mlx-ext"))
PY
```

With the automatic native target W8 policy and `fused_quantized_sdpa=False`, generation
uses the existing compiled or standard paths without requiring extension operators.
An installed but incompatible extension raises an import error. On a supported
Apple Silicon host, rebuild the configured version with
`uv sync --locked --reinstall-package spark-mlx-ext`, then restart
Python. Linux CPU and CUDA installations skip this Apple Silicon dependency.

The three Python `mx.compile` optimizations remain available independently of
this extension on supported Metal paths. Installing the extension alone does
not establish a performance improvement; measure your own workload.

### Experimental fused INT8 KV attention

The extension also provides fused attention over affine INT8/G32 KV caches.
Its core is maintained independently in
[`extensions/spark-mlx-ext/spark_mlx_ext/int8_quantized_sdpa.py`](../extensions/spark-mlx-ext/spark_mlx_ext/int8_quantized_sdpa.py).
It uses MLX's Metal JIT; no SDPA-specific C++ binding or MLX rebuild is required.
Use the current extension source through the uv installation workflow to include this operator.
Decode fusion is enabled by default. It can also be selected explicitly through the Python loader:

```python
from spark_mlx_llm import load

model, tokenizer = load(
    "/path/to/spark2_5",
    model_config={"fused_quantized_sdpa": True},
)
```

This setting selects the new operator for Metal GPU attention with
256-dimensional query/key/value heads, GQA ratios 1/2/4/8, and no mask or a causal
mask. Multi-token calls additionally require `fused_quantized_sdpa_prefill=True`
in `model_config`; this separate switch defaults to false. They use a tiled
prefill kernel; causal masking includes
the historical prefix of each chunk. See the
[INT8 prefill implementation notes](../extensions/spark-mlx-ext/docs/sdpa_int8_prefill.md).
It requires an INT8 KV cache with `group_size=32`, configured separately
through MLX LM's KV cache options; the setting does not quantize the cache.
Eligible decode calls use one pass for KV lengths below 1024 and two passes
from 1024 onward, including full-attention layers. The model uses the public
operator's default policy; 1024 itself belongs to the two-pass path.
Explicit array masks, other head dimensions/quantization formats, and CPU/CUDA
execution retain the existing MLX LM path. A missing extension API raises an
error when a supported GPU attention call reaches the operator. An existing model can opt in per
layer with `layer.self_attn.sdpa_config.fused_quantized_sdpa = True` for `layer in model.layers`.
Decode fusion is enabled by default; prefill fusion remains disabled. This path
has no end-to-end generation quality guarantee yet. See the [operator notes](../extensions/spark-mlx-ext/README.md)
for the kernel API and measurements.

### Experimental tuned BF16 attention

The extension provides `bf16_sdpa`, a two-pass decode operator with a
length-dependent KV split policy. Enable it through the Python loader:

```python
from spark_mlx_llm import load

model, tokenizer = load(
    "/path/to/spark2_5",
    dtype="bfloat16",
    model_config={"tuned_bf16_sdpa": True},
)
```

This opt-in applies only to full/global attention layers on Metal with BF16
Q/K/V, batch 1, 16 query heads, 4 KV heads, head dimension 256, a single query
token, and KV lengths from 2048 through 131072. No mask or the singleton
causal mask is supported. Prefill, sliding layers, explicit array masks,
other shapes/dtypes/devices and lengths outside this range use the existing
MLX LM path. Quantized KV caches retain their existing path; this setting
does not dequantize or change cache storage. It can coexist with
`fused_quantized_sdpa` because the two operators accept different KV formats.

The split counts are 64 below 32768 tokens, 128 below 131072, and 256 at
131072. Q/K/V, partial outputs and the final output use BF16 storage;
multiplication, accumulation and softmax use FP32. Different reduction
grouping can change rounding relative to native SDPA. The operator remains
lazy and does not add an evaluation or synchronization point to generation.

The setting defaults to false. An existing model can opt in with
`layer.self_attn.sdpa_config.tuned_bf16_sdpa = True` for each `layer in model.layers`;
sliding layers still retain their existing path. A missing `bf16_sdpa` API
raises an actionable error only when an eligible enabled call reaches it.
Update the extension package and restart Python to expose the new Python/JIT
Metal API; no MLX rebuild or C++ ABI change is required. See the
[operator API](../extensions/spark-mlx-ext/docs/operators.md#tuned-bf16-decode-attention)
and original measurements (local artifact: `experiments/sdpa-bf16-optimization/REPORT.md`).
Those isolated measurements do not establish end-to-end generation speed
or quality for this integration.
