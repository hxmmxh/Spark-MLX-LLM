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
  weight scheme.
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

Ordinary and DSpark decoding share one loader, generation API and command.
Python/generation CLI use built-in `GenerationConfig` defaults; no JSON file is required.
See the [inference guide](docs/inference.md)
for the complete contract and reproducible measurement settings.

```sh
.venv/bin/spark-mlx generate --model /path/to/TARGET --draft-model /path/to/DRAFT --prompt 'Explain speculative decoding.' --json
.venv/bin/spark-mlx chat --model /path/to/TARGET --draft-model /path/to/DRAFT --stream
```

Omit `--draft-model` for a target-only checkpoint. Merged checkpoints detect the draft
automatically; `--no-use-draft` loads only their target.
Defaults are target temperature 1 / top-p 0.95, draft temperature 1 / top-p 1,
seed 0, output limit 512, at most 3 draft tokens, thinking low, and both async flags enabled.

```python
from spark_mlx_llm import GenerationConfig, load, generate

model, tokenizer = load("/path/to/TARGET", draft_model="/path/to/DRAFT")
config = GenerationConfig()
result = generate(model, tokenizer, "Explain speculative decoding.", config=config)
print(result.text)
for chunk in generate(model, tokenizer, "Explain speculative decoding.", config=config, stream=True):
    print(chunk.text, end="", flush=True)
```

### 3. Start an HTTP service

#### Start the server

Recommended settings for the 4B target and its matching draft are listed explicitly below.
Replace the two model paths with local paths, then keep this terminal running.

```sh
.venv/bin/spark-mlx server \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
    --served-model-name spark-x2.6-4b \
    --host 127.0.0.1 --port 8080 \
    --enable-thinking --reasoning-effort low \
    --draft-quantization int8 \
    --temperature 1 --top-p 0.95 --top-k 0 --min-p 0 \
    --draft-temperature 1 --draft-top-p 1 --draft-top-k 0 \
    --max-draft-tokens 3 \
    --asynchronous-draft --async-context --compile-draft-chain \
    --seed 0 --prefill-step-size 512 \
    --max-context-length 262144 \
    --max-tokens 262144
```

For 1.7B, replace both model paths and use a matching service name.
For ordinary decoding, omit the draft-related options.
To disable thinking, replace `--enable-thinking` with `--no-enable-thinking`.
`--max-context-length` limits input plus output tokens; `--max-tokens` limits output,
including reasoning and the final answer. Requests inherit these sampling settings unless overridden.

#### Test HTTP requests

Open another terminal and send a request. Its `model` must match `--served-model-name`:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "spark-x2.6-4b",
      "messages": [{"role": "user", "content": "Hello, please introduce yourself."}],
      "stream": false
    }'
```

For SSE, add `-N` to curl and change `"stream"` to `true`.

You can also run the provided test scripts from the repository root:

```sh
.venv/bin/python example/http_chat.py
.venv/bin/python example/http_completions.py
```

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

KV caches retain model precision by default. Python/generation CLI use 512-token
prefill chunks; HTTP preserves the original 2048-token default unless overridden.
For target-only decoding, `GenerationConfig(kv_bits=8)` or `--kv-bits 8`
enables quantized KV; see the [unified guide](docs/inference.md).

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
