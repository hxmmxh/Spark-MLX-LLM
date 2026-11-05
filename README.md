<p align="center">
  <img src="docs/assets/spark-mlx-logo.png" alt="Spark and MLX" width="480">
</p>

<h1 align="center">Spark-MLX-LLM</h1>

<p align="center">
  <strong>English</strong> | <a href="README.zh.md">简体中文</a>
</p>

Spark-MLX-LLM is a local inference project for **Spark 2.6 and Spark 2.5**, built on
[MLX](https://github.com/ml-explore/mlx) and [MLX LM](https://github.com/ml-explore/mlx-lm).
It adapts Spark's model architecture and mixed-precision quantization to MLX,
with dedicated inference optimizations for Apple Silicon. The project provides
model conversion, text generation, interactive chat, and HTTP serving to make
Spark models easier to deploy and use locally.

## Features

- **Spark model adaptation and conversion:** convert BF16 and compressed-tensors
  INT4 checkpoints to MLX format.
- **Mixed-precision quantization:** use INT4, INT8, and BF16 weights for different
  modules, with configurable W8 execution backends.
- **DSpark speculative decoding:** generate with a merged target/draft checkpoint,
  with independent target and draft sampling settings.
- **Apple Silicon optimizations:** dedicated QKV/RoPE, linear, and INT8 KV
  attention operators, selected according to configuration, device, and input shape.
- **Multiple interfaces:** command-line generation and chat, Python generation
  and streaming APIs, and an HTTP service.

## Environment setup

The Apple Silicon setup below requires:

| Component | Requirement |
| --- | --- |
| Hardware and OS | Apple Silicon; native MLX compatibility targets macOS >= 14 |
| Python | Native `arm64` CPython; 3.12 recommended |
| Optional extension | macOS >= 26.2; source builds require Xcode |

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python
and dependencies. After installing uv:

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
uv python install 3.12
uv sync --locked
```

This creates `.venv` and installs native MLX inference dependencies without the
extension or an Xcode requirement. On macOS >= 26.2 with the above Xcode toolchain,
run `uv sync --locked --extra metal` to build and install `spark-mlx-ext`.
The program automatically selects available acceleration for the runtime environment.
Run subsequent commands from the repository root.
Inference requires a Metal or CUDA GPU; model conversion runs on CPU.
For Linux/CUDA installation, environment checks, and troubleshooting, see the
[environment guide](docs/en/environment.md).

## Quick start

### 1. Download a model

**Spark-X2.6-4B-INT4 is recommended.** Download either format to a local
directory, or download a target GGUF file, and pass it directly to `--model`.

| Model | Format | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| Spark-X2.6-4B | BF16 | [Download](https://huggingface.co/XHToken/Spark-X2.6-4B) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B) |
| Spark-X2.6-4B-INT4 | INT4 | [Download](https://huggingface.co/XHToken/Spark-X2.6-4B-INT4) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-INT4) |
| Spark-X2.6-4B-GGUF | Q4_K_M and others | [Download](https://huggingface.co/XHToken/Spark-X2.6-4B-GGUF) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-GGUF/files) |

### 2. Quantize and convert (optional)

Manual conversion is optional: BF16 and MLX Safetensors load directly; original
compressed-tensors INT4 directories are prepared in memory by default, while
Spark target GGUF files are converted and cached on first load. Use
`--model-cache` to enable disk caching for INT4 directories. To export ahead of
time or select an output directory and quantization options, convert explicitly:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-MLX
```

`--source` is the local directory from step 1; `--output` must be a new directory.
The default conversion uses mixed INT4/INT8/BF16 weights; see the
[quantization scheme below](#quantization-and-compatibility) and the
[conversion guide](docs/en/conversion.md) for details.

### 3. Run inference

**Pass the original model directory, target GGUF file, or a converted directory**
to `--model`. Paths below can be replaced with a downloaded
`Spark-X2.6-4B-INT4` directory or `Spark-X2.6-4B-Q4_K_M.gguf` file.

Generate a response:

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/Spark-X2.6-4B-MLX \
    --prompt '你好，请介绍一下自己。' \
    --max-tokens 256
```

Start an interactive chat:

```sh
.venv/bin/spark-mlx-chat \
    --model /path/to/Spark-X2.6-4B-MLX
```

Generation, chat, and HTTP services default to low-effort thinking. Original
INT4 directories are prepared in memory without writing converted model files.
`--max-tokens` limits the combined thinking and final-answer length; increase it
as needed. See [thinking modes](docs/en/examples.md#thinking-modes).

### 4. Start an HTTP service

Serve the converted model on localhost:

```sh
.venv/bin/spark-mlx-server \
    --model /path/to/Spark-X2.6-4B-MLX \
    --served-model-name spark-x2.6-4b \
    --host 127.0.0.1 \
    --port 8080
```

Keep this terminal running and send requests to `http://127.0.0.1:8080` from
another terminal or application. Stop the service with Ctrl-C. See the
[HTTP examples](docs/en/examples.md#4-http-chat-streaming-chat-and-text-completion)
for chat, streaming responses, and text completion.

See the [complete interface guide](docs/en/examples.md) for `generate`, `chat`,
`server`, Python generation and streaming APIs, and HTTP requests.

## DSpark models

DSpark supports **separate target and draft loading** and **merged checkpoints**.
The target accepts original BF16, compressed-tensors INT4, or MLX Safetensors
directories, and Spark target GGUF files. Draft inputs can be floating-point
checkpoints or this project's exported INT8/INT4 MLX Safetensors. Floating drafts
can also be quantized at load time. They can be standalone Safetensors directories or the public
`Spark-X2.6-4B-DSpark.gguf` from the
[Spark-X2.6-4B-DSpark-GGUF ModelScope repository](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-DSpark-GGUF/files).
Its main weights are BF16, with some small tensors such as norms stored in FP32.
This draft is paired with Spark-X2.6-4B; downloading only the target does not enable DSpark.

**Option 1: load separately.** Use the downloaded target and draft directly:

```sh
.venv/bin/python -m spark_mlx_llm.dspark generate \
    --model /path/to/Spark-X2.6-4B-Q4_K_M.gguf \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
    --prompt 'Explain speculative decoding in one paragraph.' \
    --temperature 1 --top-p 0.95 --top-k 0 \
    --draft-temperature 1 --draft-top-p 1 --draft-top-k 0 \
    --max-new-tokens 512 --stats
```

**Option 2: export a merged directory.** Add `--draft-source` to conversion to
produce a self-contained target, draft, and DSpark configuration:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --draft-source /path/to/Spark-X2.6-4B-DSpark.gguf \
    --output /path/to/Spark-X2.6-4B-DSpark-MLX
```

A BF16 target is also supported; add `--quantization none` to retain BF16.
When attaching a draft to an already MLX affine-quantized target, its weights
and quantization parameters are preserved. The output may contain multiple
Safetensors shards. Both paths share configuration, tensor-shape, and available
vocabulary checks, and adapt the GGUF's separate Q/K/V tensors, tap indices,
stacked norms, and shared embedding. Quantized MLX drafts load directly, either
separately or merged; quantized draft GGUFs are currently unsupported.

### Draft quantization at load time

Generation, benchmark, export, and the DSpark HTTP service accept
`--draft-quantization checkpoint|bf16|int8|int4`. Floating drafts default to INT8
quantization, while quantized MLX drafts retain their saved format. Use
`checkpoint` or `bf16` to preserve floating weights. Use the same public BF16 GGUF for floating inputs;
quantization runs once during loading, with no separate quantized file needed.

| Draft module | `int8` | `int4` |
| --- | --- | --- |
| Attention QKV and FFN gate/up/down | Symmetric per-channel INT8, FP32 scales | Asymmetric INT4/G32, BF16 scales/offsets, matching the target conversion rule |
| Feature `fc`, attention `out_proj`, Markov output projection | Symmetric per-channel INT8, FP32 scales | Symmetric per-channel INT8, FP32 scales |
| Independent draft `lm_head`, if present | Symmetric per-channel INT8 | Symmetric per-channel INT8 |
| Token/Markov embeddings, attention gates, norms, confidence | Floating point | Floating point |
| Shared target output head | Existing readout policy | Existing readout policy |

Thus `int4` is a mixed W4/W8 policy. Target weights and source files are
preserved, and acceptance/rejection uses the quantized draft's actual proposal
probabilities. Unsupported W8 devices/shapes execute floating-point operations
on the quantized weights without retaining the original BF16 linear modules.
Python's `load_pair()` uses the same default; `draft_quantization="int4"` selects INT4.
`pair.draft_quantization` lists the quantized modules. Merged floating-point
draft checkpoints support the same option.

### Export a quantized draft once

To skip draft quantization on subsequent startups, export it once. Add
`--draft-quantization int4` to use the mixed W4/W8 policy above:

```sh
.venv/bin/python -m spark_mlx_llm.dspark export-draft \
    --model /path/to/Spark-X2.6-4B-MLX \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
    --output /path/to/Spark-X2.6-4B-DSpark-MLX-INT8
```

Point `--draft-model` at the output directory on later generate, benchmark, or
HTTP service runs; omit `--draft-quantization` to restore integer codes and
scales directly. The directory also works with conversion's `--draft-source`,
which preserves the draft's integer codes when merging. Explicitly requesting
the saved `int8`/`int4` policy also loads directly; changing precision requires
the original floating draft. The output directory must not exist.
Python exposes `save_draft(pair, output_path)` from `spark_mlx_llm.dspark`.
Shared embedding and external readout still come from the paired target.

### Merged models and serving

Merged directories also work with the existing commands from the repository root:

```sh
# Check checkpoint metadata and draft tensor shapes without loading weights.
.venv/bin/python -m spark_mlx_llm.dspark inspect /path/to/merged-checkpoint

# Generate with DSpark; greedy sampling and batched verification are the defaults.
.venv/bin/python -m spark_mlx_llm.dspark generate \
    --model /path/to/merged-checkpoint \
    --prompt 'Explain speculative decoding in one paragraph.' \
    --max-new-tokens 512 --stats

# Sample from the target while using a greedy draft.
.venv/bin/python -m spark_mlx_llm.dspark generate \
    --model /path/to/merged-checkpoint \
    --prompt 'Write a short story about a robot.' \
    --temperature 1 --top-p 0.95 --top-k 0 \
    --draft-temperature 0 --draft-top-p 1 --draft-top-k 0 \
    --max-new-tokens 512 --stats
```

Add `--mode baseline` to use only the target for generation from the same merged
checkpoint. Each omitted draft sampling option inherits its target counterpart.
`--max-draft-tokens N` caps candidates per round; zero disables drafting.
`--stats` writes statistics to stderr; `--json` returns text and metrics as JSON.
The model's chat template is applied by default; `--raw-prompt` skips it.

Python exposes the same controls:

```python
from spark_mlx_llm.dspark import SamplingOptions, generate, load_pair

pair = load_pair(
    "/path/to/Spark-X2.6-4B-MLX",
    draft_path="/path/to/Spark-X2.6-4B-DSpark.gguf",
)
# For a merged directory: load_pair("/path/to/merged-checkpoint").
result = generate(
    pair,
    "Explain speculative decoding.",
    max_new_tokens=512,
    sampling=SamplingOptions(temperature=1.0, top_p=0.95),
    draft_sampling=SamplingOptions(temperature=0.0),
)
print(result.text)
print(result.draft_acceptance_rate, result.mean_acceptance_length)
```

Start the DSpark HTTP service with low thinking, target T=1/P=0.95, draft
T=1/P=1, and a 256K context limit:

```sh
.venv/bin/spark-mlx-server \
    --model /path/to/Spark-X2.6-4B-Q4_K_M.gguf \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
    --served-model-name spark-x2.6-4b \
    --host 127.0.0.1 --port 8080 \
    --model-cache-dir /path/to/model-cache \
    --temperature 1 --top-p 0.95 --draft-temperature 1 --draft-top-p 1 \
    --max-context-length 262144 --max-tokens 4096
```

Use `http://127.0.0.1:8080/v1/chat/completions` with model name `spark-x2.6-4b`,
matching `--served-model-name`;
SSE streaming is supported. The 256K limit includes input and output tokens.
A request's `max_tokens` can override the default 4096-token output limit.
For a merged directory, pass only `--model`.

DSpark currently handles one request at a time. Run
`.venv/bin/python -m spark_mlx_llm.dspark generate --help` for all CLI options.

## Quantization and compatibility

The default conversion applies the following scheme to the main model. Decoder
modules use the same scheme in every layer; embedding and `lm_head` are model-wide.
The activation column describes execution with the W8A8 backend available and
enabled, while the weight and scale columns describe the converted checkpoint.

| Layer / module | Weight quantization | Activations (W8A8 enabled) | Weight scale type |
| --- | --- | --- | --- |
| `self_attn.q_k_v_proj` | Asymmetric INT4, group size 32 | BF16 | BF16 |
| `mlp.gate_proj` | Asymmetric INT4, group size 32 | BF16 | BF16 |
| `mlp.up_proj` | Asymmetric INT4, group size 32 | BF16 | BF16 |
| `mlp.down_proj` | Asymmetric INT4, group size 32 | BF16 | BF16 |
| `self_attn.out_proj` | Symmetric INT8, per output channel | Dynamic per-token INT8 | BF16 |
| `lm_head` | Symmetric INT8, per vocabulary row | Dynamic per-token INT8 | FP32 |
| `self_attn.g_proj` | BF16, unquantized | BF16 | — |
| `model.embedding` | BF16, unquantized | — | — |
| RMSNorm | BF16, unquantized | Unquantized | — |

All generation entry points default to unquantized KV caches that retain model
precision: **BF16 models use BF16 KV**, including full-attention and sliding-window
layers. To opt into INT8/G32 KV for full-attention layers, use `--kv-bits 8` with
`spark-mlx-generate` or `kv_bits=8` with this package's Python generation functions.
Generate and the Python wrappers use 512-token prompt chunks; chat and server
retain upstream MLX LM chunking behavior. These settings are separate from weight
quantization; see
[KV caching and prompt chunking](docs/en/examples.md#kv-caching-and-prompt-chunking).

## Documentation

See the [documentation index](docs/README.md) for all guides.

| Topic | Guide |
| --- | --- |
| Installation, platform requirements, and troubleshooting | [Environment setup](docs/en/environment.md) |
| Extracting and using the prebuilt release | [Release package guide](docs/en/release-package.md) |
| CLI, Python, HTTP, and thinking settings | [Interface guide](docs/en/examples.md) |
| Model conversion and quantization | [Conversion guide](docs/en/conversion.md) |
| Benchmark commands and metric definitions | [Benchmark guide](docs/en/benchmark.md) |
| DSpark generation, sampling, and Python APIs | [DSpark usage](#dspark-models) |

## License

See [LICENSE](LICENSE).
