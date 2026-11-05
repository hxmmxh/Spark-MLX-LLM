# Model conversion and quantization

[Project overview](../../README.md) · [简体中文](../zh/conversion.md)

`spark-mlx-convert` converts local Spark BF16 or compressed-tensors INT4
checkpoints and Spark target GGUF files to MLX format. Conversion runs on CPU
in chunks without loading the model for inference. Safetensors conversion
requires NumPy only; GGUF decoding uses the `gguf` package.

## 1. Supported inputs

Prepare a local model directory containing `config.json`, single-file or sharded
Safetensors weights, and tokenizer files. The converter reads layer counts,
hidden dimensions, attention heads, and vocabulary size from the configuration.
Download the published Spark-X2.6-4B BF16 or INT4 model using the links in the
[README](../../README.md#1-download-a-model).

**Ordinary and DSpark inference accept local original checkpoints directly.**
BF16/MLX directories load directly; compressed-tensors INT4 defaults to in-memory
preparation, while target GGUFs are converted and cached on first loading.
Manual conversion is optional. Use `--model-cache` or Python's `model_cache=True`
to enable disk caching for INT4 directories. When enabled, caches default to
`~/.cache/spark-mlx-llm/models`. Configure the root using
`--model-cache-dir /path/to/cache`, the Python loading argument
`model_cache_dir`, or `SPARK_MLX_MODEL_CACHE`. Explicit CLI/Python arguments
override the environment variable, which overrides the default.

If the input contains floating-point or this project's quantized `draft_model.*` tensors, the converter
preserves their names, data, and draft configuration without quantizing them.
Alternatively, `--draft-source` attaches a standalone Safetensors draft directory
or the [public BF16 draft GGUF](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-DSpark-GGUF/files).
Draft GGUF inputs must be floating point; target GGUF support is described below.

## 2. Command-line conversion

Follow the [environment guide](environment.md) first. Run commands from the
repository root and replace `/path/to/...` with your local paths.

Inspect configuration and tensor metadata without writing output:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --inspect
```

`--inspect` checks structure and metadata; it does not scan all weight values
for NaNs or precision errors.

Convert the compressed INT4 model with the default mixed-precision scheme:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-MLX
```

Quantize a BF16 model with the same scheme:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B \
    --output /path/to/Spark-X2.6-4B-from-BF16-MLX
```

To retain BF16 main-model weights, use `--quantization none` with a BF16 source:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B \
    --output /path/to/Spark-X2.6-4B-BF16-MLX \
    --quantization none
```

`--output` must be a new directory, must not be a symlink, and must neither
contain nor be contained by the source directory. Conversion never overwrites
the source or an existing output directory. `--chunk-rows` defaults to `256`;
smaller values reduce temporary memory use during CPU conversion.

### Load original INT4 without disk preparation

Inference entry points default to preparing original compressed-tensors INT4 in
memory. Integer codes are repacked in row chunks into in-memory MLX affine arrays;
no converted cache or temporary model file is written. Python's `load(path)` and
`load_pair(path)` use the same default for INT4 directories.
Validation, scale/zero-point rounding and tied BF16 head handling match disk
conversion. Every startup repeats preparation and needs temporary memory.
Cache directory options and environment settings are ignored in this mode.
Ordinary chat/server require a single process; target GGUF is not yet supported.
Explicit convert/export-draft commands still save their requested outputs.

### Target GGUF import

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-Q4_K_M.gguf \
    --output /path/to/Spark-X2.6-4B-from-GGUF-MLX
```

Import requires the Spark `spark2_5` architecture and BPE tokenizer with an
embedded chat template. Supported tensor types are FP32/FP16/BF16, Q4_0/Q4_1,
Q4_K/Q5_K/Q6_K, and Q8_0, covering the published mixed Q4_K_M checkpoint.
Rows are decoded to BF16 and processed by the existing MLX conversion rules;
no complete intermediate BF16 checkpoint is written. Vocabulary, merges,
special tokens, and the template are exported automatically.

The default output uses this project's mixed W4/G32 + W8 policy. Explicit
`--quantization none` retains decoded BF16 weights instead. GGUF import
requantizes and does not preserve GGUF integer codes or promise bitwise
equivalence with llama.cpp. Automatic inference caches use the default mixed
policy with the `native` backend configuration; use explicit conversion for
other export policies.

### Merge a separate target and draft

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --draft-source /path/to/Spark-X2.6-4B-DSpark.gguf \
    --output /path/to/Spark-X2.6-4B-DSpark-MLX
```

`--draft-source` can also point to a standalone Safetensors directory containing
`config.json` and floating-point weights, or INT8/INT4 weights exported by this
project. Draft integer codes and coefficients are preserved when merging.
If `--source` is already MLX affine
quantized, target weights, scales, and biases are copied without requantization;
the manifest records `quantization="preserved"`. For inputs already containing
a draft, omit `--draft-source` and convert the merged checkpoint directly.
Both draft formats validate geometry, taps, tensor shapes, and available token
ID mappings. GGUF BF16/FP32 precision is preserved, Q/K/V are concatenated, and
shared embedding is recorded in the configuration. `--inspect` accepts separate
drafts too; the output must be disjoint from both sources.

To load without exporting a merged directory, generation and the DSpark HTTP
service accept `--draft-model`. Python uses
`load_pair(target_path, draft_path=draft_path)`; see the
[DSpark examples](../../README.md#dspark-models).

Conversion preserves the draft's stored precision. Floating drafts default to
INT8 quantization during loading from the same BF16 GGUF, standalone Safetensors,
or merged checkpoint. Use `--draft-quantization int4` to apply INT4/G32 to
QKV/FFN and per-channel INT8 to other major projections. This is a generation/
serving option, not a convert option, and preserves target quantization.

To save a quantized draft ahead of time, run
`python -m spark_mlx_llm.dspark export-draft --model TARGET --draft-model DRAFT --draft-quantization int8 --output NEW_DIRECTORY`
(`int4` also works). Later use `--draft-model NEW_DIRECTORY` or
`--draft-source NEW_DIRECTORY`. Quantized drafts retain their saved format by
default, restoring integer codes, FP32 W8 scales, and
BF16 W4 scales/offsets without startup requantization. The versioned format and
module rules are recorded in `config.json` under `draft_quantization`; unrelated
quantization layouts are not assumed compatible.

To preserve a floating draft's precision, explicitly pass
`--draft-quantization checkpoint` or `bf16`.

## 3. Default mixed-precision scheme

`--quantization mixed` is the default. The main model uses the following weight
formats. The compute column assumes W8A8 is available and enabled; see the
backend choices below for floating-point activation paths.

| Module | Weight quantization | Scale / affine bias dtype | Compute |
| --- | --- | --- | --- |
| QKV and MLP gate/up/down | Asymmetric INT4, group size 32 | BF16 | W4A16 |
| Attention `out_proj` | Symmetric INT8, one scale per output channel | BF16 | W8A8, dynamic per-token INT8 activations |
| `lm_head` | Symmetric INT8, one scale per vocabulary row | FP32 | W8A8, dynamic per-token INT8 activations |
| Embedding, `g_proj`, and RMSNorm | BF16, unquantized | — | Floating point |

Compressed input retains its existing integer weight codes without requantizing
them. MLX stores affine weights as UINT32-packed `weight`, `scales`, and `biases`.
W8 per-channel scales repeat across G32 groups, with `biases = -128 * scales`.
The `lm_head` scales remain FP32.

Scale and affine-bias conversion can introduce rounding; the conversion manifest
records coefficient rounding errors. Preserving integer codes does not imply
bit-for-bit identical inference results.

BF16 input uses per-group or per-channel min/max quantization with
round-to-nearest, without calibration data. This differs from the original
compressed model's GPTQ/MSE procedure, so evaluate quality on your target tasks.
When the source ties embedding and `lm_head`, mixed conversion creates a separate
INT8 `lm_head` and keeps the embedding in BF16. `--quantization none` requires
BF16 input; it does not dequantize an INT4 checkpoint.

## 4. W8 execution backend

`--w8-backend` selects the inference policy recorded in `config.json` and
`conversion_manifest.json`. It applies to W8 `out_proj` and `lm_head` under
mixed quantization and does not change their stored weight codes or scales.

| Value | Behavior |
| --- | --- |
| `w8a8` (default) | Dynamic per-token INT8 activations on the supported Metal path; requires a compatible `spark-mlx-ext`, Apple M5 or newer, and macOS >= 26.2 |
| `w8a16` | Floating-point activations; uses the optimized per-channel W8A16 path for eligible single-row GPU inputs |
| `native` | Floating-point activations using ordinary MLX affine operators |

Unavailable optimized backends fall back to MLX affine operators with
floating-point activations. W8A8 and floating-point activation paths can produce
different numerical results. All conversions still execute on CPU.

To explicitly select the native path:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-native-MLX \
    --w8-backend native
```

The output sets `quantize_activations=false` to disable MLX LM's generic
QQLinear path. When selected and available, this project's W8A8 backend
performs activation quantization separately.

## 5. Python API and arguments

The CLI and Python API use the same conversion implementation:

| CLI option | Python argument | Default and meaning |
| --- | --- | --- |
| `--source` | `source` | Required local BF16 or compressed-tensors model directory |
| `--output` | `output` | Required for conversion; a new directory disjoint from the source |
| `--inspect` | `inspect_only` | `False`; inspect metadata without writing output or scanning all weight values |
| `--quantization` | `quantization` | `"mixed"`; `"none"` retains BF16 main-model weights and requires BF16 input |
| `--w8-backend` | `w8_backend` | `"w8a8"`; also accepts `"w8a16"` or `"native"` |
| `--chunk-rows` | `chunk_rows` | `256`; a positive number of rows processed per CPU chunk |

```python
from spark_mlx_llm import convert_to_mlx

report = convert_to_mlx(
    source="/path/to/Spark-X2.6-4B-INT4",
    output="/path/to/Spark-X2.6-4B-python-MLX",
    quantization="mixed",
    w8_backend="native",
    chunk_rows=256,
)
print(report["quantized_module_counts"])
```

`source` and `output` accept strings or `pathlib.Path`. For inspection, call
`convert_to_mlx(source, inspect_only=True)` without an output directory.
The return value is a JSON-serializable report containing `status`,
`source_format`, `quantized_module_counts`, `lm_head_untied`, and draft metadata.
Completed conversions also include `coefficient_rounding` and save the full
report as `conversion_manifest.json`.

Invalid or unsupported values raise `ValueError`; missing files raise
`FileNotFoundError`; an existing output directory raises `FileExistsError`.
Unsupported group-index/permutation tensors, activation schemes, weight
transforms, and sparsity are rejected.

The module entry point is `python -m spark_mlx_llm.commands.convert`;
`python -m spark_mlx_llm.conversion` remains a compatibility entry point.
Use `--source` and `--output`; upstream options such as `--hf-path`, `--mlx-path`,
and `--q-bits` are not supported. See the
[Python conversion example](examples.md#1-model-conversion-in-python) for a runnable script.

## 6. Output and inference

The output contains `model.safetensors`, `model.safetensors.index.json`,
`config.json`, tokenizer files, and `conversion_manifest.json`. If draft tensors
are present, they are stored separately in `model-draft.safetensors` and included
in the index. Output is prepared in a temporary directory and published after
validation; a failed conversion cleans up the temporary directory.

Run the converted main model:

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/Spark-X2.6-4B-MLX \
    --prompt '你好，请介绍一下自己。' \
    --max-tokens 256 \
    --no-thinking
```

KV precision is controlled separately during generation. All generation entry
points default to unquantized KV caches, so BF16 models use BF16 KV regardless
of the INT4/INT8 weight scheme. See the [interface guide](examples.md) for
interactive chat, Python generation, and HTTP serving.
