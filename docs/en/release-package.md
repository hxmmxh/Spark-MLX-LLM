# Using the macOS release package

[Project overview](../../README.md) · [简体中文](../zh/release-package.md) · [Documentation index](../README.md)

This guide is for users who download the prebuilt release. The package includes
Python, MLX, MLX LM, and the project's commands. You do not need to install
Python, uv, Homebrew, or Xcode, clone the repository, or activate a virtual
environment. Obtain model weights separately; they are not included in the runtime archive.

Check `BUILD-INFO.json` for the package version and OS requirements.

## 1. Check system requirements

The examples use the currently generated `spark-mlx-0.1.2-macos26-arm64.tar.gz`:

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
shasum -a 256 -c spark-mlx-0.1.2-macos26-arm64.tar.gz.sha256
tar -xzf spark-mlx-0.1.2-macos26-arm64.tar.gz
cd spark-mlx-0.1.2-macos26-arm64
```

The checksum should report `OK`. Substitute the filenames if using another
release. Run the following `./bin/...` commands from the extracted release root.
You can move the directory, including to a path with spaces or non-ASCII
characters. Keep the entire directory together rather than copying `bin/` alone.

## 3. Check and run

```sh
./bin/spark-mlx doctor --json
./bin/spark-mlx doctor --gpu
./bin/spark-mlx generate --model TARGET --draft-model DRAFT --prompt 'Hello' --max-tokens 512 --json
./bin/spark-mlx chat --model TARGET --draft-model DRAFT --stream
./bin/spark-mlx benchmark --model TARGET --draft-model DRAFT --input-file gen.txt --trials 5
```

Omit the draft option for ordinary generation. Chat commands are `/quit` and `/reset`.
Defaults are built in; no configuration file is required. Use `--config FILE` for optional overrides.
See the [inference guide](../inference.md) for all parameters and Python usage.

### Start the server

Recommended settings for the 4B target and its matching draft are listed explicitly below.
Replace the two model paths with local paths, then keep this terminal running.

```sh
./bin/spark-mlx server \
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

### Test HTTP requests

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

## 4. Commands and directories

`bin/spark-mlx` is the single command, with `generate`, `chat`, `server`, `benchmark`,
`doctor`, `inspect` and `convert` subcommands. `bin/spark-mlx-python` is the isolated interpreter.
Use each subcommand's `--help`. Legacy entry points and parameter aliases were removed.

`BUILD-INFO.json` records build/validation details and `FILES.sha256` records file hashes.
Move the entire directory; the interpreter does not read system packages or PYTHONPATH.
Check native fallback with `SPARK_MLX_DISABLE_EXT=1 ./bin/spark-mlx doctor --gpu`.
Model downloads are linked from the project overview; weights are not bundled.
