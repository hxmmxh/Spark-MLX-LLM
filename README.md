<p align="center">
  <img src="docs/assets/spark-mlx-logo.png" alt="Spark and MLX" width="480">
</p>

<h1 align="center">Spark-MLX-LLM</h1>

<p align="center">
  <strong>English</strong> | <a href="README.zh.md">简体中文</a>
</p>

Spark-MLX-LLM is a local inference project for **Spark 2.6 and Spark 2.5**, built on
[MLX](https://github.com/ml-explore/mlx) and [MLX LM](https://github.com/ml-explore/mlx-lm).
It adapts and optimizes Spark's model architecture and mixed-precision weights for
Apple Silicon, loads official INT4 models directly, and provides text generation,
interactive chat, Python APIs, and HTTP serving.

[Environment setup](#environment-setup) · [Generation and chat](#2-generation-and-chat) ·
[HTTP serving](#3-start-an-http-service) · [Quantization and compatibility](#quantization-and-compatibility) ·
[Documentation](#documentation)

## Features

- **Load official models directly:** use Spark-X2.6-4B-INT4 and Spark-X2.6-1.7B-INT4
  without prior conversion.
- **Mixed-precision support:** preserve the official models' INT4, INT8, and BF16
  weight scheme, with configurable W8 execution backends.
- **DSpark speculative decoding:** load the target and its matching draft
  separately or use a merged checkpoint, with independent sampling settings for each.
- **Apple Silicon optimizations:** dedicated QKV/RoPE, linear, and INT8 KV
  attention operators, selected according to configuration, device, and input shape.
- **Multiple interfaces:** command-line generation and chat, Python generation
  and streaming APIs, and an HTTP service.

## Environment setup

The setup below is for Apple Silicon. For Linux/CUDA installation and
troubleshooting, see the [environment guide](docs/en/environment.md).

| Component | Requirement |
| --- | --- |
| Hardware and OS | Apple Silicon; native MLX compatibility targets macOS >= 14 |
| Python | Native `arm64` CPython; 3.12 recommended |
| Optional extension | macOS >= 26.2; source builds require Xcode |

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone
the repository and install dependencies:

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
uv python install 3.12
uv sync --locked
```

This creates `.venv` and uses native MLX by default, without installing the
extension or requiring Xcode. If your environment meets the extension requirements,
you can also run:

```sh
uv sync --locked --extra metal
```

The program automatically selects available acceleration for the runtime environment.

Run subsequent commands from the repository root; activating the virtual environment
is unnecessary. Inference requires a Metal or CUDA GPU; model conversion runs on CPU.

## Quick start

### 1. Download models

Download files from either platform below. The target requires a complete directory
containing its configuration, tokenizer, and Safetensors weights; for the DSpark
draft, download only the matching BF16 GGUF file.

| Purpose | Model repository | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| 4B target | Spark-X2.6-4B-INT4 | [Download](https://huggingface.co/XHToken/Spark-X2.6-4B-INT4) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-INT4) |
| 4B draft | Spark-X2.6-4B-DSpark-GGUF | [Download](https://huggingface.co/XHToken/Spark-X2.6-4B-DSpark-GGUF) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-DSpark-GGUF/files) |
| 1.7B target | Spark-X2.6-1.7B-INT4 | [Download](https://huggingface.co/XHToken/Spark-X2.6-1.7B-INT4) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-1.7B-INT4) |
| 1.7B draft | Spark-X2.6-1.7B-DSpark-GGUF | [Download](https://huggingface.co/XHToken/Spark-X2.6-1.7B-DSpark-GGUF) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-1.7B-DSpark-GGUF/files) |

**DSpark speculative decoding is recommended.** Compared with running only the
target, DSpark provides approximately **1.6×** speedup for the 4B model and
**1.2×** for the 1.7B model.
Download the target together with its matching `Spark-X2.6-4B-DSpark.gguf` or
`Spark-X2.6-1.7B-DSpark.gguf`. The target and draft must match in model size;
generation and serving examples follow below.

### 2. Generation and chat

Load **compressed-tensors INT4 Safetensors targets** and **BF16 GGUF DSpark
drafts** directly, without converting them to MLX format first.

Command-line generation and chat use the following defaults unless explicitly overridden:

| Setting | Default and behavior |
| --- | --- |
| Thinking mode | Enabled, with `low` reasoning effort |
| Target `temperature` | `0`: greedy sampling selects the highest-probability token at each step |
| Target `top_p` | `1`: no cumulative-probability truncation |
| Target `top_k` | `0`: no limit on the number of candidate tokens |
| DSpark draft sampling | Temperature, `top_p`, and `top_k` each inherit the target setting; greedy by default |
| DSpark draft precision | Floating-point drafts are quantized to INT8 at load time; saved INT8/INT4 drafts retain their format |
| Output limit | `512` tokens, including thinking and the final answer |

When temperature is greater than `0`, tokens are sampled from the probability
distribution after `top_p` and `top_k` filtering. Explicit options in the examples
override these defaults.

The examples use 4B models. Replace `/path/to/...` with local paths; for 1.7B,
replace both the target and draft paths.

**DSpark generation (recommended):**

```sh
.venv/bin/python -m spark_mlx_llm.dspark generate \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
    --prompt 'Explain speculative decoding in one paragraph.' \
    --temperature 1 --top-p 0.95 --top-k 0 \
    --draft-temperature 1 --draft-top-p 1 --draft-top-k 0 \
    --max-new-tokens 512 --stats
```

Specify the target with `--model` and its matching draft with `--draft-model`.
The target also accepts converted MLX directories, and the draft also accepts
exported MLX Safetensors directories. This example uses random sampling: target
temperature `1` with `top_p=0.95`, draft temperature `1` with `top_p=1`, and no
`top_k` limit for either.

**DSpark interactive chat:**

```sh
.venv/bin/spark-mlx-chat \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf
```

This loads the target and its matching draft for multi-turn conversations with DSpark.

`--max-tokens` limits the combined thinking and final-answer length; increase it
as needed. See [thinking modes](docs/en/examples.md#thinking-modes) for other settings.

<details>
<summary>Python generation example</summary>

```python
from spark_mlx_llm.dspark import SamplingOptions, generate, load_pair

pair = load_pair(
    "/path/to/Spark-X2.6-4B-INT4",
    draft_path="/path/to/Spark-X2.6-4B-DSpark.gguf",
)
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

See the [interface guide](docs/en/examples.md) for generation and streaming APIs.

</details>

<details>
<summary>More options: sampling parameters and draft precision</summary>

#### Sampling and generation parameters

Common DSpark `generate` options are listed below. Run
`.venv/bin/python -m spark_mlx_llm.dspark generate --help` for all options.

| Option | Purpose |
| --- | --- |
| `--mode baseline` | Run only the target for comparison with DSpark |
| `--temperature` / `--top-p` / `--top-k` | Target sampling; temperature defaults to `0` for greedy sampling |
| `--draft-temperature` / `--draft-top-p` / `--draft-top-k` | Draft sampling; each omitted field inherits its target counterpart |
| `--max-draft-tokens N` | Cap candidates per round; `0` disables drafting |
| `--max-new-tokens N` | Limit output length; the standard generation entry point uses `--max-tokens` |
| `--stats` / `--json` | Write statistics to stderr / return text and metrics as JSON |
| `--raw-prompt` | Skip the chat template applied by default |

#### Draft precision options

Floating-point drafts are quantized to INT8 at load time by default; saved
INT8/INT4 drafts are restored in their stored format. Use `--draft-quantization`
to preserve floating-point precision or select INT4 explicitly.

| `--draft-quantization` | Behavior |
| --- | --- |
| Omitted (default) | Quantize floating-point drafts to INT8; restore this project's quantized MLX drafts in their saved format |
| `checkpoint` | Preserve the checkpoint precision; load floating-point or this project's quantized MLX drafts directly |
| `bf16` | Preserve floating-point draft weights; existing quantized drafts require `checkpoint` or the matching quantization policy |
| `int8` | Use symmetric per-output-channel INT8 with FP32 scales for the main linear projections |
| `int4` | Use asymmetric INT4/G32 for QKV and FFN gate/up/down, and INT8 for the other main projections |

| Draft module | `int8` | `int4` |
| --- | --- | --- |
| Attention QKV and FFN gate/up/down | Symmetric per-channel INT8, FP32 scales | Asymmetric INT4/G32, BF16 scales/offsets |
| Feature fusion `fc`, attention `out_proj`, Markov output projection | Symmetric per-channel INT8, FP32 scales | Symmetric per-channel INT8, FP32 scales |
| Independent draft `lm_head`, if present | Symmetric per-channel INT8 | Symmetric per-channel INT8 |
| Token/Markov embeddings, attention gates, norms, confidence head | Floating point | Floating point |
| Shared target output head | Selected readout policy | Selected readout policy |

Quantization preserves the target and source files. Acceptance/rejection correction
uses the quantized draft's actual proposal probabilities. When dedicated W8 kernels
are unavailable, execution falls back to floating-point operations on quantized
weights without retaining copies of the original BF16 linear modules.
Python's `load_pair()` uses the same defaults; inspect `pair.draft_quantization`
for the actual quantized modules. These rules also apply to floating-point drafts
in merged directories.

</details>

### 3. Start an HTTP service

**DSpark service (recommended):** pass the target directory and its matching draft file directly.

```sh
.venv/bin/spark-mlx-server \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
    --served-model-name spark-x2.6-4b \
    --host 127.0.0.1 --port 8080 \
    --thinking --reasoning-effort low \
    --draft-quantization int8 \
    --temperature 1 --top-p 0.95 --top-k 0 \
    --draft-temperature 1 --draft-top-p 1 --draft-top-k 0 \
    --max-context-length 262144 \
    --max-tokens 262144
```

The DSpark service currently handles one request at a time.

**Target-only service:**

```sh
.venv/bin/spark-mlx-server \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --served-model-name spark-x2.6-4b \
    --host 127.0.0.1 --port 8080 \
    --thinking --reasoning-effort low \
    --temperature 0 --top-p 1 --top-k 0 \
    --max-tokens 262144
```

Start either service and keep its terminal running. Send a chat request from another terminal:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "spark-x2.6-4b",
      "messages": [{"role": "user", "content": "Hello, please introduce yourself."}],
      "stream": false
    }'
```

The request's `model` must match `--served-model-name`, here `spark-x2.6-4b`.
Requests inherit the service's output limit when `max_tokens` is omitted; they
can also set it explicitly within the context limit. Set `stream` to `true` and
add `-N` to curl to receive SSE responses incrementally. Stop the service with Ctrl-C.
See the [interface guide](docs/en/examples.md) for HTTP text completion and all options.

## Quantization and compatibility

The two official INT4 targets above use the following mixed-precision scheme.
Loading or conversion preserves existing integer weight codes and adapts
scales/offsets to MLX's storage representation; type conversion and rounding may
affect numerical results. The activation column describes execution when the
W8A8 backend is available and enabled.

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
precision: **BF16 models use BF16 KV**. To enable INT8/G32 KV for full-attention
layers, pass `--kv-bits 8` to `spark-mlx-generate` or `kv_bits=8` to this package's
Python generation functions. Generate and the Python wrappers use 512-token prompt
chunks by default; chat and server retain upstream MLX LM chunking behavior.
These settings are separate from weight quantization; see
[KV caching and prompt chunking](docs/en/examples.md#kv-caching-and-prompt-chunking).

## Documentation

See the [documentation index](docs/README.md) for all guides.

| Topic | Guide |
| --- | --- |
| Installation, platform requirements, and troubleshooting | [Environment setup](docs/en/environment.md) |
| Extracting, diagnosing, and deploying models with the prebuilt release | [Release package guide](docs/en/release-package.md) |
| CLI, Python, HTTP, thinking modes, and KV caching | [Interface guide](docs/en/examples.md) |
| Model format conversion and precision | [Model conversion](docs/en/conversion.md) |
| Benchmark commands and metric definitions | [Benchmark guide](docs/en/benchmark.md) |
| Standard and DSpark inference | [Generation and chat](#2-generation-and-chat) · [HTTP serving](#3-start-an-http-service) |
| Apple Silicon extension and operators | [Extension guide](extensions/spark-mlx-ext/README.md) |

## License

See [LICENSE](LICENSE).
