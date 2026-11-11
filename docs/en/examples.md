# Inference examples

The [unified inference guide](../inference.md) defines the Python, CLI and HTTP contract.
Ordinary and DSpark inference use `load()`, `generate()` and `spark-mlx`.
The examples use built-in defaults; edit model paths before running. No configuration file is required.
Use `--config FILE` or `GenerationConfig.from_file(FILE)` only for explicit JSON overrides;
[`inference.json`](../../example/inference.json) is an optional configuration example.

| Example | API |
| --- | --- |
| [python_generate.py](../../example/python_generate.py) | `generate(..., config=config)` → `GenerationResult` |
| [python_stream.py](../../example/python_stream.py) | `generate(..., config=config, stream=True)` → chunks |
| [benchmark.py](../../example/benchmark.py) | Public `benchmark()`; all repeats and effective configs |
| [cli_generate.sh](../../example/cli_generate.sh) | `spark-mlx generate` |
| [cli_chat.sh](../../example/cli_chat.sh) | `spark-mlx chat` |
| [cli_server.sh](../../example/cli_server.sh) | `spark-mlx server` |
| [http_chat.py](../../example/http_chat.py) | `/v1/chat/completions`, messages, full/SSE |
| [http_completions.py](../../example/http_completions.py) | `/v1/completions`, raw prompt, `choices[].text` |
| [python_convert.py](../../example/python_convert.py) | Offline conversion |
| [cli_convert.sh](../../example/cli_convert.sh) | `spark-mlx convert` |

```sh
SPARK_MODEL=/path/to/TARGET bash example/cli_generate.sh --draft-model /path/to/DRAFT
SPARK_MODEL=/path/to/TARGET bash example/cli_server.sh --draft-model /path/to/DRAFT
.venv/bin/python example/http_chat.py
```

Python/generation CLI thinking options are top-level `enable_thinking` and `reasoning_effort`.
Python/generation CLI enable low-effort thinking by default; use `--no-enable-thinking` to disable it.
HTTP preserves MLX-LM 0.31.3 defaults and its full request/response protocol for both ordinary and DSpark models.
Sampling and KV settings belong to `GenerationConfig`; weight/operator/cache options belong to `load()`.
Python content preserves reasoning/tool markers. HTTP uses upstream `reasoning` and `tool_calls` formatting;
the application executes tools.

Legacy Python/CLI entry points were removed; HTTP retains the upstream protocol.
Rebuild bundles to use the current API; existing archives retain their released interface.
