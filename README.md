# Spark-MLX-LLM

Spark2_5 and Spark3 (Spark X3) inference support for
[MLX](https://github.com/ml-explore/mlx) and
[MLX LM](https://github.com/ml-explore/mlx-lm).

The package directly loads unquantized Hugging Face `safetensors` checkpoints
and checkpoints already converted to MLX format. It does not require GGUF
conversion and does not modify the installed `mlx-lm` package.

**Compressed-tensors INT4/INT8 checkpoints, including
`4b-head-only-int8-mse`, must be converted to MLX affine format before running
generation, chat, or the server.** Follow the
[conversion instructions](#convert-compressed-tensors-mixed-int4int8-checkpoints)
and pass the **converted output directory** to `--model`.

## Features

- Direct loading of unquantized Hugging Face or MLX-format checkpoints from a
  local directory or repository ID
- Strict checkpoint weight-name and shape validation
- CPU, Apple Silicon GPU, and optional Linux CUDA execution
- BF16 and FP32 weight overrides
- Sliding-window and full-attention KV caches
- Spark3 factorized queries and CED cross-layer KV sharing
- Spark3 still-image ViT, three-axis mRoPE, and multimodal generation
- Automatic Spark2_5 function-call parsing through MLX LM
- Spark2_5 wrappers for MLX LM conversion, chat, and server commands
- Command-line and Python inference interfaces

## Supported models

### Spark2_5

The implementation supports `Spark2_5ForCausalLM` checkpoints with
`model_type: spark2_5`. The reference 1.7B checkpoint uses:

- 28 decoder layers
- 8 query heads and 2 key/value heads
- Fused QKV projection
- Head-wise sigmoid attention output gates
- Three sliding-window layers followed by one full-attention layer
- A 512-token sliding window
- 256 rotary dimensions with theta 10,000 in sliding layers
- 64 rotary dimensions with theta 1,000,000 in full-attention layers
- Parallel GELU feed-forward layers
- Tied input and output embeddings

The implementation lives in `spark_mlx_llm/spark2_5.py` (previously `model.py`).
Package-level `Model` and `ModelArgs` continue to refer to Spark2_5.

### Spark3 / Spark X3

`spark_mlx_llm/spark3.py` supports `Spark3ForCausalLM` checkpoints with
`model_type: spark3`. The SWA/FULL model matches the Spark3 vLLM plugin.

The Spark3 model, MLP, attention, RoPE, residual blocks and quantization adapters
are defined independently in this module; it does not import or subclass the
Spark2_5 implementation.

- Factorized query projection: `q_down_proj -> RMSNorm -> q_up_proj -> RoPE`.
  A positive `q_lora_rank` matching the checkpoint is required.
- Separate K/V projections, MQA/GQA, partial non-interleaved RoPE, optional
  head-wise sigmoid output gates, and GELU-gated MLPs.
- Per-layer sliding/full attention and tied or independent output embeddings.
- Optional FULL CED: `kv_source_layer_ids` is one zero-based FULL producer
  index. Every later FULL layer reads its KV, computes its own Q, and omits
  its K/V projections. SWA layers retain independent KV.

The current 9B CED configuration uses 40 layers, hidden size 4096, intermediate
size 12288, 32 Q heads, one KV head, head dimension 256, Q rank 1024, a 512-token
window, and an independent LM head over 131072 tokens. The schedule is
`[SWA, SWA, SWA, FULL] * 10`. Producer layer 19 supplies layers 23, 27, 31, 35
and 39. Checkpoint RoPE settings override the configuration defaults.

`model.make_cache()` returns **35 cache objects** for this model, ordered by
the owning layer indices in `model.model.cache_layer_ids`. Consumers allocate
no cache. This permits native MLX LM prompt-cache save/load without duplicate
storage or updates. Both full-sequence inference and cached chunked prefill /
decode preserve causal attention and absolute RoPE positions across SWA wraps.
Legacy exports containing unused consumer K/V weights are accepted; missing
owner weights still fail strict loading.

Load an unquantized BF16 or already MLX-quantized checkpoint through the same
Python `load()` API and CLI. Model selection follows `config.json`:

```sh
spark-mlx-generate --model /path/to/spark3 --prompt '你好' --max-tokens 32
```

For explicit construction, use `from spark_mlx_llm.spark3 import Model,
ModelArgs`, or the package aliases `Spark3Model` and `Spark3ModelArgs`.
The `spark-mlx-convert`, `spark-mlx-chat` and `spark-mlx-server` wrappers register
both architectures. Native MLX affine weight quantization keeps head-gate
projections at source precision. The standalone compressed-tensors conversion
script under `tools/` remains Spark2_5-only.

This implementation does not support QSA, DSpark, or approximate CED prefill
truncation. Unsupported layer types and a non-null `ced_prefill_keep_tokens`
are rejected. It does not use the Spark2_5 fused-QKV extension for Spark3's
factorized query path.

### Spark3 images and mRoPE

Spark3 also loads nested `text_config` plus optional `vision_config`, including
the local `models/test/hf_latest` checkpoint. Explicit root text fields override
nested fields. A single-element CED source list such as `[19]` is accepted.
The existing flat 9B CED checkpoint continues to use its original text path,
ordinary RoPE and 35 owner KV caches.

The MLX vision tower follows `SglangSparkPlugin`: biased Linear patch embedding
after temporal averaging, interpolated absolute positions, adjacent-pair 2D
RoPE, pre-norm Transformer blocks, and a 2x2 spatial merger. It accepts the
checkpoint's `model.visual.*` names and the shorter `visual.*` prefix. Vision
weight shapes are strictly checked. Video and audio are not supported.

The text model reads `mrope_section` and `mrope_interleaved` independently for
SWA/FULL layers, including the interleaved layout used by `hf_latest`. Image
tokens receive T/H/W coordinates; later text and decode retain the resulting
position offset. Each mRoPE request has one additional native `ArraysCache`
holding that offset, so save/load retains it and concurrent requests do not
share position state. These caches cannot be trimmed into an image prefix;
the ordinary 9B cache behavior is unchanged. Approximate prefill truncation is
not enabled.

Install Pillow for image loading and resizing (no PyTorch is required):

```sh
python -m pip install -e '.[vision]'
spark-mlx-generate --model ../models/test/hf_latest \
  --image /absolute/path/to/image.png --prompt '请描述这张图片。' --max-tokens 128
```

Repeat `--image` for multiple images. The CLI inserts one image placeholder per
image before applying the checkpoint's chat template, unless the prompt already
contains image placeholders. Both one-placeholder-per-image and already expanded
contiguous placeholders are accepted. `--image-min-pixels` and
`--image-max-pixels` control the resize bounds. Supply `--tokenizer-path` if the
checkpoint's matching tokenizer is stored separately.

The supplied `hf_latest` tokenizer vocabulary contains `<|System|>`, `<|User|>`
and `<|Bot|>` (and older `<System>`, `<User>`, `<Bot>` tokens). Its bundled chat
template instead uses `<｜User｜>` / `<｜Assistant｜>`, which are not registered
role tokens in that vocabulary. This can produce repeated role text. The
package preserves the checkpoint template; pass the actual training template
with `--chat-template /path/to/template.jinja`, or supply the exact serialized
training prompt with `--ignore-chat-template`. A vocabulary alone does not
establish the training prompt's newline, end-of-turn or system-message rules.

Preprocessing reads `preprocessor_config.json` when present; otherwise it uses
Qwen-style patch grouping, Pillow bicubic resize, repeated still frames and RGB
mean/std 0.5, following the SGLang plugin's fallback convention. The processor
validates the patch/temporal/merge geometry against the vision config. Use the
checkpoint's training preprocessing settings for quality evaluation.

The Python image API uses MLX LM's generation/sampling machinery and encodes
images once, including when prefill chunks split an image:

```python
from spark_mlx_llm import load, Spark3Processor, generate_multimodal

model_path = "../models/test/hf_latest"
model, tokenizer = load(model_path)
processor = Spark3Processor.from_pretrained(model_path, tokenizer, model.args)
prompt = processor.image_placeholder + "\n请描述这张图片。"
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True,
)
text = generate_multimodal(
    model, processor, prompt, ["/absolute/path/to/image.png"], max_tokens=128,
)
```

`generate_multimodal` starts a fresh prompt. For explicit cache continuation,
call the model with processor-produced `pixel_values`, `image_grid_thw` and
`image_offsets`, or with prepared embeddings and explicit `[3,batch,length]`
positions. Text-only calls work for both ordinary RoPE and mRoPE checkpoints.
Image requests currently accept one prompt per call; multiple images within it
are supported. The native MLX LM chat/server wrappers remain text interfaces;
their HTTP image routing is not implemented by this package.

Run the numerical/cache tests and opt in to real multimodal header checks:

```sh
python -m pytest tests/ -v
SPARK3_MM_MODEL=../models/test/hf_latest \
  python -m pytest tests/test_spark3_multimodal.py -v
```

Run unit tests and opt in to real-checkpoint integration checks:

```sh
python -m pytest tests/ -v
SPARK3_MODEL=/path/to/spark3 python -m pytest tests/test_spark3_integration.py -v
```

The integration checks inspect every shard's completeness and parameter shapes,
strictly load the checkpoint, and compare cached decode with a full forward.
For BF16, they require equal greedy tokens, relative logit L2 error below 2%,
and output-distribution KL divergence below 0.002. The tiny-model unit tests
separately use a strict FP32-to-FP64 NumPy comparison on CPU, precision-aware
Metal checks, chunked prefill, SWA wraps and cache serialization.

The checkpoint directory must contain at least:

```text
config.json
model*.safetensors
tokenizer.json
tokenizer_config.json
```

If the tokenizer is stored in a nested directory, copy the tokenizer files to
the checkpoint root before loading the model.

## Installation

Clone the repository and create an isolated environment:

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
python3 -m venv .venv
```

Install the backend required by the host.

Apple Silicon:

```sh
.venv/bin/python -m pip install -e '.[test]'
```

Linux CPU:

```sh
.venv/bin/python -m pip install -e '.[cpu,test]'
```

Linux CUDA 12 or CUDA 13:

```sh
.venv/bin/python -m pip install -e '.[cuda12,test]'
# or
.venv/bin/python -m pip install -e '.[cuda13,test]'
```

Confirm the installation:

```sh
.venv/bin/python -c 'import mlx.core as mx; print(mx.__version__)'
```

### Optional Spark C++ extension on Apple Silicon

Install **`mlx-spark-ext`** in the same environment as `spark-mlx-generate`
to enable the C++ QKV projection and RoPE path for supported single-token,
batch-size-one affine W4 decoding. The model detects it automatically after
restarting Python; no command-line flag is needed. It follows the
[MLX custom extension mechanism](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
and leaves the official `mlx.core` intact.

The extension is a separate, optional package. Installing `spark-mlx-llm`
alone does not install it. Its source is included in
[`extensions/mlx-spark-ext`](extensions/mlx-spark-ext), so no separate MLX
source checkout is needed.

**Install a prebuilt wheel.** When available, download a matching
`mlx_spark_ext-*.whl` asset from this repository's
[GitHub Releases](https://github.com/XHToken/Spark-MLX-LLM/releases).
If no matching asset has been published, use the source-build instructions below.
The current build has these requirements:

| Component | Requirement |
| --- | --- |
| Hardware | Apple Silicon (`arm64`) |
| Python | Standard CPython 3.12 for the `cp312-cp312` wheel |
| MLX | Official `mlx==0.32.4` and `mlx-metal==0.32.4` |
| macOS | 26.2 or newer for the current binaries; tested on 26.5.1 |

The wheel's `macosx_26_0_arm64` filename does not express the backend's
26.2 minimum. Use the requirements in the release notes as well as the filename.

From the repository root, replace the path below with the downloaded wheel:

```sh
.venv/bin/python -m pip install \
  'mlx==0.32.4' 'mlx-metal==0.32.4' \
  /path/to/mlx_spark_ext-0.2.0-cp312-cp312-macosx_26_0_arm64.whl
```

**Build from source.** Follow the
[extension build instructions](extensions/mlx-spark-ext/README.md#build-from-source)
for another supported Python version or when a wheel is unavailable. This
compiles only the extension. Version 0.2.0 also builds Metal 4 W8A8 kernels
and requires the corresponding Xcode Metal compiler and SDK headers.
The current MLX backend's macOS requirement still applies.

The extension also exposes an explicit
[W8A8 linear API](extensions/mlx-spark-ext/README.md#w8a8-api) for Apple M5+.
Installing it does not automatically change the model's quantization or
replace its output projections.

For an already converted MLX checkpoint, enable W8A8 by adding
`input_activations` to each desired output projection's entry under
`config.json` → `quantization_config`:

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
implemented in the Spark model, using `mlx-spark-ext >= 0.2.0` on Apple M5+.
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
import mlx_spark_ext as ext
from spark_mlx_llm import spark2_5

assert version("mlx") == version("mlx-metal") == "0.32.4"
assert spark2_5._native_qkv_rope is ext.spark_qkv_rope
assert mx.metal.is_available()
print("Extension ready:", version("mlx-spark-ext"))
PY
```

If the extension is absent, generation uses the existing compiled or standard
paths. An installed but incompatible extension raises an import error; install
the matching build or remove it with
`.venv/bin/python -m pip uninstall mlx-spark-ext`. Restart Python after either
operation. CPU and CUDA users should omit this Apple Silicon build.

The three Python `mx.compile` optimizations remain available independently of
this extension on supported Metal paths. Installing the extension alone does
not establish a performance improvement; measure your own workload.

## Quick start

**For compressed-tensors quantized models, convert first, then run inference.**
Use the [conversion tool](#convert-compressed-tensors-mixed-int4int8-checkpoints)
below and replace the `--model` path in these examples with its output directory.

Run deterministic CPU generation from a local checkpoint:

```sh
.venv/bin/spark-mlx-generate \
    --device cpu \
    --dtype bfloat16 \
    --model /path/to/spark2_5 \
    --prompt '只输出53乘以42的结果，不要解释。' \
    --max-tokens 64 \
    --temp 0 \
    --seed 1
```

Expected response:

```text
53乘以42的结果是2226。
```

Run on a selected CUDA GPU:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/spark-mlx-generate \
    --device gpu \
    --dtype bfloat16 \
    --model /path/to/spark2_5 \
    --prompt '只输出53乘以42的结果，不要解释。' \
    --max-tokens 64 \
    --temp 0 \
    --seed 1
```

Confirm that MLX sees the CUDA device before inference:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \
    'import mlx.core as mx; mx.set_default_device(mx.gpu); print(mx.default_device()); print(mx.device_info())'
```

Hugging Face repository IDs are also accepted:

```sh
.venv/bin/spark-mlx-generate \
    --model XHToken/your-spark2_5-repository \
    --prompt '你好' \
    --max-tokens 128
```

The repository must expose the model and tokenizer files required above.

## MLX LM tool wrappers

The package provides wrappers for MLX LM tools that need to resolve the
Spark2_5 model architecture. Registration is limited to the command process;
the wrappers do not modify the installed `mlx_lm` package.

Convert or quantize a Hugging Face checkpoint:

```sh
.venv/bin/spark-mlx-convert \
    --hf-path XHToken/Spark-X2.5-1.7B \
    --mlx-path ./Spark-X2.5-1.7B-8bit \
    --quantize \
    --q-bits 8
```

The converter keeps the head-wise attention `g_proj` gates in BF16. These
small sigmoid gates scale every attention head and are sensitive to low-bit
quantization. Use 8-bit or BF16 weights for tool calling; 4-bit weights use
less memory but can reduce the accuracy of schema-constrained arguments.

### Quantize BF16 Spark3 to W4 with an 8-bit LM head

[`tools/quantize_spark3_w4_head8.py`](tools/quantize_spark3_w4_head8.py)
converts local BF16 Spark3 checkpoints directly to MLX affine weights without
calibration. Text matrices and input embeddings use 4 bits, while `lm_head`
uses 8 bits; both use group size 64. Vision weights, attention head gates,
normalization weights and biases remain BF16. Activations are not quantized.
For tied checkpoints, the converter exports independent 4-bit input embeddings
and an 8-bit output head and sets `tie_word_embeddings` to false.

Run from the repository root with the installed MLX environment:

```sh
.venv/bin/python tools/quantize_spark3_w4_head8.py \
    --input-root ../models/test \
    --models 9b_ced_20000 hf_latest second_hf \
    --output-root ../models/quantized_mlx
```

These are the defaults; default paths are resolved relative to the repository,
so the script can also be called from another working directory. Outputs are
named `<model>-w4-head8-g64`. Existing output or `.incomplete` directories are
never overwritten; use a new `--output-root` to repeat a conversion.

The converter processes tensors in batches and writes shards of approximately
1 GiB. Before publishing each output directory, it verifies the saved tensor
shapes and dtypes, byte equality of preserved BF16 tensors, strict model loading,
text prefill and cached decode, and a synthetic image forward for vision models.
Each model includes `quantization_report.json`; the output root receives
`quantization_summary.json` after all requested models finish. These checks
verify loading and execution, without evaluating generation quality.

### Convert compressed-tensors mixed INT4/INT8 checkpoints

**This conversion is required before using a compressed-tensors checkpoint
with `spark-mlx-generate`, `spark-mlx-chat`, or `spark-mlx-server`.**
For example, convert `4b-head-only-int8-mse` first and point `--model` at the
resulting MLX directory.

[`tools/convert_compressed_tensors_to_mlx.py`](tools/convert_compressed_tensors_to_mlx.py)
converts Spark2_5 compressed-tensors checkpoints to native MLX affine
`model.safetensors`, with configuration and tokenizer files. Run it with a
Python environment containing NumPy; the conversion itself runs on CPU and
does not require PyTorch, CUDA, or the Spark Metal extension.

**All W8A8 layers become W8A16 with BF16 activations, including every attention
`out_proj` and `lm_head`.** Input activation quantization is removed. The output
uses ordinary MLX quantized layers and does not select the W8A8 extension path.
This change is recorded in `config.json`, `conversion_manifest.json`, and
`CONVERSION_NOTES.txt`, and printed by the command.

The script retains the existing INT4/G32, G64, or G128 weight codes and the
symmetric per-channel INT8 codes. Per-channel INT8 scales are repeated across
MLX G32 groups. BF16 embedding, attention `g_proj`, and Norm weights are copied
unchanged; `tie_word_embeddings` is preserved. **All affine scales and biases
are stored in BF16**, so original FP32 scales and computed offsets may be
rounded. The manifest records the maximum coefficient rounding error per
module. Integer weights are not re-quantized, but the resulting inference is
not numerically equivalent to the source W8A8 model.

From the repository root, convert a local checkpoint into a new output directory:

```sh
.venv/bin/python tools/convert_compressed_tensors_to_mlx.py \
    --source /path/to/compressed-tensors-checkpoint \
    --output /path/to/spark-mlx-w8a16
```

The script reads local paths. Replace `--output ...` with `--inspect` to
validate configuration and tensor shapes without writing a model.
Existing output directories are never overwritten. Single-file and sharded
Safetensors are supported; unhandled group-index/permutation tensors,
activation schemes, weight transforms, and sparsity are rejected.

Run the converted model through this package's Spark loader:

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/spark-mlx-w8a16 \
    --device gpu --dtype bfloat16 \
    --prompt '你是谁，你能做什么？' --max-tokens 128 --temp 0
```

Start an interactive chat session:

```sh
.venv/bin/spark-mlx-chat \
    --model ./Spark-X2.5-1.7B-8bit
```

Start the OpenAI-compatible server:

```sh
.venv/bin/spark-mlx-server \
    --model ./Spark-X2.5-1.7B-8bit \
    --host 127.0.0.1 \
    --port 8080
```

Use these `spark-mlx-*` commands to register the bundled Spark architectures.
If the installed MLX LM already provides a native module for either model type,
the wrappers use it instead of the corresponding bundled implementation.

## Python API

```python
from mlx_lm import generate

from spark_mlx_llm import load


model, tokenizer = load(
    "/path/to/spark2_5",
    dtype="bfloat16",
)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=False,
    add_generation_prompt=True,
)
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=128,
)
print(response)
```

When the checkpoint provides the Spark2_5 tool-aware chat template, the returned
tokenizer automatically exposes MLX LM's tool parser:

```python
assert tokenizer.has_tool_calling
tool_call = tokenizer.tool_parser(
    "set_state"
    "<arg_key>name</arg_key><arg_value>上海</arg_value>"
    "<arg_key>count</arg_key><arg_value>42</arg_value>",
    [
        {
            "type": "function",
            "function": {
                "name": "set_state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            },
        }
    ],
)
```

Select an execution device before loading the model when using the Python API:

```python
import mlx.core as mx

mx.set_default_device(mx.gpu)  # or mx.cpu
```

## CUDA implementation note

The checkpoint stores Q, K, and V in one fused projection. After the projection,
the implementation splits, reshapes, and transposes the result into attention
heads. These tensors are non-contiguous views into the fused projection.

Passing those views directly to the CUDA RoPE path caused the K view to be
evaluated with incorrect storage aliasing. The error first appeared at layer 0
K-RoPE and later propagated through attention and the remaining layers. In the
full lazy graph it could produce all-NaN logits or an illegal CUDA memory access.

The model materializes Q, K, and V before RoPE and attention:

```python
queries = mx.contiguous(queries)
keys = mx.contiguous(keys)
values = mx.contiguous(values)
```

This is a compatibility workaround for the affected non-contiguous CUDA path;
it does not change the Spark2_5 computation.

## Verification

Run the unit tests and lint checks:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

Run the real-checkpoint integration test on CPU:

```sh
SPARK25_MODEL=/path/to/spark2_5 \
SPARK25_DTYPE=bfloat16 \
    .venv/bin/python -m pytest tests/test_integration.py -q -m integration
```

Run the complete test suite on CUDA:

```sh
CUDA_VISIBLE_DEVICES=0 \
SPARK25_MODEL=/path/to/spark2_5 \
SPARK25_DTYPE=bfloat16 \
    .venv/bin/python -m pytest -q
```

The integration test performs strict weight loading, validates complex
function-call parsing, and runs a real checkpoint forward pass. Always run an
end-to-end generation command as well, because a successful shape test alone
does not establish generation correctness.

The CUDA fix was validated with MLX 0.32.2, MLX LM 0.31.3, BF16 weights, and an
NVIDIA H100 PCIe GPU:

- CPU and CUDA prefill logits were finite
- CPU and CUDA selected the same first token
- Full-logit relative RMSE was approximately 0.528%
- The CUDA test suite reported `6 passed`
- Three consecutive deterministic generations produced the expected answer
- No NaN, illegal memory access, or non-deterministic text was observed

Other CUDA architectures should be validated with the same integration and
end-to-end generation commands before deployment.

## Project layout

```text
spark_mlx_llm/spark2_5.py Spark2_5 computation graph and cache selection
spark_mlx_llm/spark3.py   Independent Spark3 computation graph and CED caches
spark_mlx_llm/loader.py   Local and Hugging Face checkpoint loading
spark_mlx_llm/cli.py      Text generation command
extensions/mlx-spark-ext/ Optional C++ extension package
tests/                   Unit and real-checkpoint tests
```

## License

See [LICENSE](LICENSE).
