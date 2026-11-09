# Using the macOS release package

[Project overview](../../README.md) · [简体中文](../zh/release-package.md) · [Documentation index](../README.md)

This guide is for users who download the prebuilt release. The package includes
Python, MLX, MLX LM, and the project's commands. You do not need to install
Python, uv, Homebrew, or Xcode, clone the repository, or activate a virtual
environment. Obtain model weights separately; they are not included in the runtime archive.

Known limitations in 0.1.1: DSpark HTTP streaming can return empty text chunks
despite reporting generated tokens. This endpoint also rejects text-part lists
and tool definitions. The bundled Python generation API can process the full
prompt after rendering messages and tools with the tokenizer's chat template.

## 1. Check system requirements

The examples use the currently generated `spark-mlx-0.1.1-macos26-arm64.tar.gz`:

| Component | Requirement |
| --- | --- |
| Chip | Apple Silicon with native ARM64 execution, such as M1 and later M-series chips; Intel Macs are unsupported |
| Operating system | **macOS 26.2 or newer** |
| Memory | **More than 8 GB of unified memory**; a quantized 4B model is recommended |

Check the chip, OS version, and memory under ** → About This Mac**.

## 2. Verify and extract

Download the archive and its matching `.sha256` file into the same directory.
Open a terminal there and run:

```sh
shasum -a 256 -c spark-mlx-0.1.1-macos26-arm64.tar.gz.sha256
tar -xzf spark-mlx-0.1.1-macos26-arm64.tar.gz
cd spark-mlx-0.1.1-macos26-arm64
```

The checksum should report `OK`. Substitute the filenames if using another
release. Run the following `./bin/...` commands from the extracted release root.
You can move the directory, including to a path with spaces or non-ASCII
characters. Keep the entire directory together rather than copying `bin/` alone.

## 3. Check the runtime

```sh
./bin/spark-mlx-doctor
```

The first line shows a Chinese verdict: “能跑” (ready) or “不能跑” (not ready),
followed by OS and Metal GPU check results. Failed checks include the cause.

```sh
./bin/spark-mlx-doctor --gpu   # Run GPU computation checks
./bin/spark-mlx-doctor --json  # Output JSON for scripts
```

## 4. Prepare a model

Download links are listed in the [project overview](../../README.md#1-download-models).
Use an **original compressed-tensors INT4 directory** or a **supported Spark target GGUF file** directly.

Pass the directory or GGUF file path to `--model`. The examples below use
`/path/to/Spark-X2.6-4B-INT4`; replace it with the corresponding `.gguf` file path when using GGUF.

## 5. Generate text and chat

The following generation and chat examples use DSpark speculative decoding.
Alongside the target model, prepare a matching draft. Download links
are in the [project overview](../../README.md#1-download-models).

Generate one response:

```sh
./bin/spark-mlx-dspark generate \
  --model "/path/to/Spark-X2.6-4B-INT4" \
  --draft-model "/path/to/Spark-X2.6-4B-DSpark.gguf" \
  --prompt 'Hello. Introduce yourself in one sentence.' \
  --max-draft-tokens 3 --max-new-tokens 128 --stats
```

Start an interactive chat:

```sh
./bin/spark-mlx-chat \
  --model "/path/to/Spark-X2.6-4B-INT4" \
  --draft-model "/path/to/Spark-X2.6-4B-DSpark.gguf" \
  --max-tokens 512
```

Enter `q` to exit, `r` to reset the conversation, or `h` for help. Low-effort
thinking is enabled by default, and floating-point drafts use INT8 by default.
The output token limit includes both reasoning and the final answer.

DSpark generation, chat, and HTTP serving default to at most 3 draft tokens per
round, capped by the checkpoint's available prediction slots. Generation and
server commands accept `--max-draft-tokens N` to override this limit; `0` disables drafting.

## 6. Start the HTTP chat service

```sh
./bin/spark-mlx-dspark-server \
  --model "/path/to/Spark-X2.6-4B-INT4" \
  --draft-model "/path/to/Spark-X2.6-4B-DSpark.gguf" \
  --served-model-name spark-x2.6-4b \
  --max-draft-tokens 3 \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 512
```

The service uses DSpark and low-effort thinking by default. Leave the terminal
running and send a request from another terminal:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "spark-x2.6-4b",
    "messages": [{"role": "user", "content": "Hello. Introduce yourself."}],
    "max_tokens": 128,
    "stream": false
  }'
```

The request's `model` matches `--served-model-name`. For multi-turn conversations,
include previous `user` and `assistant` messages in order in `messages`.
For an SSE stream, set `"stream": true` and use `curl -N` to display data as it arrives:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "spark-x2.6-4b",
    "messages": [{"role": "user", "content": "Explain speculative decoding."}],
    "max_tokens": 128,
    "stream": true
  }'
```

Check the service with `curl http://127.0.0.1:8080/health`.
`127.0.0.1` listens on the local machine. Press Ctrl-C in the server terminal to stop it.

## 7. Commands and directories

| Command | Purpose |
| --- | --- |
| `spark-mlx-doctor` | Runtime and GPU checks |
| `spark-mlx-generate` | Ordinary text generation |
| `spark-mlx-chat` | Terminal chat; uses DSpark when a matching draft is specified |
| `spark-mlx-server` | Ordinary HTTP service |
| `spark-mlx-convert` | Model inspection and conversion |
| `spark-mlx-dspark` | Recommended DSpark text generation and other subcommands |
| `spark-mlx-dspark-server` | DSpark HTTP service |
| `spark-mlx-python` | Bundled, isolated Python interpreter |

Use `--help` for command options and `./bin/spark-mlx-dspark generate --help`
for DSpark generation options. Bundled Python ignores system site-packages and
`PYTHONPATH`, and does not use dependencies from other virtual environments.

To call the commands from other working directories, run this from the release root:

```sh
export PATH="$PWD/bin:$PATH"
```

This affects the current terminal only. To upgrade, extract the new release into
a new directory and keep using your existing model paths with `--model`.
Keep each release's bundled runtime intact.

`README.txt` contains a quick guide, `BUILD-INFO.json` records versions and
validation, `NATIVE-AUDIT.json` records native requirements, and `FILES.sha256`
lists file checksums. To build a release yourself, see the
[build guide (Chinese)](../zh/release.md).

## 8. Troubleshooting

| Symptom | Action |
| --- | --- |
| `./bin/...` cannot be found | Enter the extracted release root and check that the archive was fully extracted |
| OS version is below the requirement | Upgrade to macOS 26.2 or newer |
| Model or tokenizer files are missing | Check `--model`, retain the complete model directory, and quote paths containing spaces |
| Slow initial loading | Original models need preparation on first use; reuse the cache on subsequent runs or use a converted MLX model |
| Insufficient memory | Use a smaller or more heavily quantized model, shorten the context, and reduce simultaneously loaded models; DSpark also needs draft memory |
| Port is already in use | Change `--port` and update the client's URL to match |
| `doctor --gpu` times out on the device lock | Wait for other shared GPU work to finish, or use `./bin/spark-mlx-doctor --gpu --lock-timeout 30` to wait up to 30 seconds |
| SHA256 verification fails | Obtain the archive and checksum file for the same release again, then verify before use |
