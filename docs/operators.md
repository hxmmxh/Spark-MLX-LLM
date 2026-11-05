# Operator selection and implementation

[Project overview](../README.md)

### Select optimized W8 QMV operators

The production extension includes the final per-channel W8A16 kernels, W8A8
v9 small-projection dispatch, and the single-token W8A8 `lm_head` specialization.
Select them when loading a converted MLX checkpoint:

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/converted-mlx-checkpoint \
    --w8-qmv-backend w8a16 \
    --prompt '你好' --max-tokens 128
```

Use `--w8-qmv-backend w8a8` for the W8A8 experiment. Both choices cover affine W8
attention `out_proj` layers and an untied W8 `lm_head`. W4 projections are
controlled separately by `w4_qmv_backend`; embeddings are unchanged.

| `w8_qmv_backend` | Behavior |
|---|---|
| `auto` | Preserve checkpoint activation metadata and existing routing. |
| `native` | Force native MLX affine W8A16, including checkpoints with W8A8 metadata. |
| `w8a16` (default) | Optimized per-channel W8A16 for single-row FP16/BF16 GPU decode; native prefill/CPU fallback. |
| `w8a8` | Optimized W8A8 for GPU prefill and decode, including per-token activation quantization; native CPU fallback. Requires the M5+ extension. |

```python
from spark_mlx_llm import load

model, tokenizer = load(
    "/path/to/converted-mlx-checkpoint",
    w8_qmv_backend="w8a16",  # or "w8a8"
)
```

Both optimized W8 backends derive their parameters from the loaded MLX affine
checkpoint. INT8 codes are unpacked exactly and existing channel scales are
widened to FP32; original FP32 scale precision cannot be recovered from BF16
metadata. Groupwise or asymmetric W8 metadata is rejected. An explicitly
requested backend requires its extension API; it does not silently substitute
another backend during loading.

This setting also works in `model_config={"w8_qmv_backend": "w8a16"}` or
`config.json` for MLX LM loaders using this model implementation. `--dtype`
preserves the optimized operators' FP32 channel scales. Adapters are installed
after loading, retain both weight representations, and change the in-memory
parameter tree; save the original affine checkpoint before installing them.
W8A8 activation quantization and W8A16 post-dot FP32 scaling have different
numerical semantics from native affine W8A16. Historical microbenchmark gains
do not establish end-to-end generation throughput for this loader.

### Select the W4A16 decode implementation

`w4_qmv_backend` selects the W4 implementation when constructing/loading a
model. Its default is `native`, preserving the existing MLX path. Set it to
`splitk` to use the extension's eight-way split-K kernel for supported QKV and
FFN down projections, including the compiled model paths:

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/Spark-X2.6-4B-INT4-MLX \
    --w4-qmv-backend splitk \
    --prompt '你好' --max-tokens 128
```

Use `--w4-qmv-backend native` to select the original implementation.

```python
from spark_mlx_llm import load

model, tokenizer = load(
    "/path/to/Spark-X2.6-4B-INT4-MLX",
    w4_qmv_backend="splitk",  # "native" is the default
)
```

The same setting works in `model_config={"w4_qmv_backend": "splitk"}` or
checkpoint `config.json`; an explicit `load()` argument takes precedence.
The W4 and W8 switches are independent and can be combined. Changing only
`model.args` after construction does not reconfigure existing layers.

The model selects split-K only for one-row Metal GPU decode with BF16
activations/scales/biases, affine W4 group size 32, and logical weight shape
`[N,K]` of `[6144,2560]` (QKV) or `[2560,10240]` (down). Prefill, batched
decode, CPU execution, other dtypes/group sizes/shapes, and gate/up projections
retain the original path. A supported optimized call requires the local
`spark-mlx-ext` package exposing `w4a16_qmv`; otherwise it raises a clear error.
Weights keep their native packing and checkpoint parameter names, so no
conversion or extra persistent weight copy is needed.

Split-K changes FP32 summation order and can change BF16 rounding. Earlier
isolated M5 Pro measurements showed approximately 1.04x for QKV and 1.06x for
down; those are operator measurements, not generation-throughput claims.
The kernel is inference-only. Its standalone public API and opt-in benchmark
are documented in [spark-mlx-ext](../extensions/spark-mlx-ext/README.md).

### Explicit per-channel W8A16 decode adapter

After loading an affine W8 model, an individual projection can opt into the
signed INT8 / FP32-scale decode primitive:

```python
from spark_mlx_llm.ops.linear import W8A16PerChannelLinear

attention.out_proj = W8A16PerChannelLinear(
    attention.out_proj, int8_weight, fp32_channel_scales
)
```

Supply signed INT8 `[N,K]` weights and FP32 `[N]` or `[N,1]` scales matching
the loaded affine layer. Model loading prepares these from the current packed
weights and scales; widening BF16 metadata does not recover original FP32 precision. It accepts a bias-free, exact `nn.QuantizedLinear` affine W8
fallback. Install it after checkpoint loading; the nested adapter parameters
do not follow the original checkpoint serialization schema.

For supported single-row GPU inputs, attention preserves a compiled graph
covering output layout, the head-wise sigmoid gate, and the per-channel
projection. Weight and scale arrays are explicit compiled-function inputs,
so replacing same-shaped parameters does not reuse captured old parameters.
Multi-token prefill retains the compiled native affine path; CPU and other
activation dtypes use the native fallback. The same adapter can wrap `lm_head`.

Model loading now enables this adapter by default for existing W8 output/head
projections through `w8_qmv_backend="w8a16"`. W4 and unquantized layers retain
their original weight precision. Both parameter representations remain resident. Prefill retains the
fallback's BF16 metadata, while decode uses the supplied FP32 scales, so this
is not a claim of numerical equivalence across phases or to the original
W8A8 model. Performance and validation records are in
`qmv-w8a16-per-channel-optimization` (local artifact: `experiments/qmv-w8a16-per-channel-optimization/REPORT.md`).


### Operator boundaries

The model calls each operator's named dispatch function directly. Configuration, eligibility
checks, implementation selection, fallback and arithmetic stay in the same file;
there is no central `ops.dispatch` module. Implementation helpers prefixed with
`_` are local to their operator. Cross-operator reuse uses explicit public types.

| Model entry | Operator entry | Selected implementations |
| --- | --- | --- |
| `Attention.__call__` | `qkv.dispatch_qkv` | Module projection, compiled affine/split-K projection, fused QKV/RoPE |
| `Attention.__call__` | `sdpa.dispatch_sdpa` | MLX-LM, fused INT8 KV, tuned BF16 attention |
| `Attention.__call__` | `attention_output.dispatch_attention_output` | Module output, compiled affine output, compiled W8A16 output |
| `MLP.__call__` | `linear.dispatch_linear` | Original projection module or eligible W4 split-K |
| `TransformerBlock.__call__` | `ffn.dispatch_ffn` | Module composition or compiled residual/norm/FFN |
| `Model.load_weights` | `linear.QuantizedProjections` | Adapter unwrapping, parameter preparation and W8 backend installation |

`qkv`, `attention_output` and `ffn` export only their respective
`dispatch_qkv`, `dispatch_attention_output` and `dispatch_ffn` entry in `__all__`.
`sdpa` also exports `SDPAConfig`. `linear` exports one execution function plus
configuration, affine-parameter and adapter types needed by compiled graphs
and checkpoint loading. It does not expose a list of standalone setup helpers.

`AffineParameters` carries current affine tensors and metadata. Its pure kernels
are shared by compiled QKV, gated output and FFN; weights remain explicit graph
arguments. Cached factories are private to their composite operator and keyed
by geometry/selected implementation. FFN checks the promoted intermediate dtype
before choosing BF16 split-K. Custom projection, norm and MLP modules retain
their own calls.

### Operator configuration

| Owner | Configuration fields | Consumers |
| --- | --- | --- |
| `linear.LinearConfig` | `w8_qmv_backend`, `w4_qmv_backend` | Linear adapters, QKV and FFN |
| `sdpa.SDPAConfig` | `fused_quantized_sdpa`, `fused_quantized_sdpa_prefill`, `tuned_bf16_sdpa` | SDPA |
| Automatic composite operators | No additional switches | Gated output and eligible compiled graphs |

Both config dataclasses inherit `base.OperatorConfig`, which provides
`from_dict()`, `choices()`, `validate_values()` and `validate()`. Backend choices
are declared once in field metadata; the CLI uses those same choices.
Partial loader overrides are checked without assuming missing checkpoint values.

`ModelArgs` inherits the two keyword-only config mixins alongside MLX-LM's
`BaseModelArgs`. Existing flat checkpoint fields and loader/CLI option names
remain flat; the W8 selector is now named `w8_qmv_backend`. Operator fields are not duplicated in
`model.py`. Operator functions keep their own typed signatures rather than
inheriting a generic dispatcher with opaque positional arguments.

At construction, attention stores independent `linear_config` and `sdpa_config`
objects, and MLP stores its own `linear_config`. To change an already-created
layer's SDPA policy:

```python
model.layers[0].self_attn.sdpa_config.fused_quantized_sdpa = True
model.layers[0].self_attn.sdpa_config.fused_quantized_sdpa_prefill = True
```

For W4 selection, update the relevant QKV/MLP config:

```python
model.layers[0].self_attn.linear_config.w4_qmv_backend = "splitk"
model.layers[0].mlp.linear_config.w4_qmv_backend = "splitk"
```

Changing `model.args` alone does not update these runtime copies. W8 backend
selection still belongs to model construction/loading because it determines
which projection modules and prepared parameters are installed.

To add an implementation, update its operator file's implementation and dispatch
policy, define any new options beside it, and test supported inputs and fallback
behavior. Add a configuration subclass only when there are options to own;
automatically selected composite operators do not need empty config classes.
