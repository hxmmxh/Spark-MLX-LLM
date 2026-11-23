# 推理示例

所有推理入口及配置定义见[统一推理接口](../inference.md)。
普通解码和 dSpark 共用 `load()`、`generate()` 和 `spark-mlx`。
下列示例直接使用代码内置默认值，修改模型路径即可运行，不需要配置文件。
只有显式传入 `--config FILE` 或调用 `GenerationConfig.from_file(FILE)` 才会加载 JSON；
[`inference.json`](../../example/inference.json) 仅作为可选配置样例。

| Example | API |
| --- | --- |
| [python_generate.py](../../example/python_generate.py) | `generate(..., config=config)` → `GenerationResult` |
| [python_stream.py](../../example/python_stream.py) | `generate(..., config=config, stream=True)` → chunks |
| [benchmark.py](../../example/benchmark.py) | Public `benchmark()`; all repeats and effective configs |
| [cli_generate.sh](../../example/cli_generate.sh) | `spark-mlx generate` |
| [cli_chat.sh](../../example/cli_chat.sh) | `spark-mlx chat` |
| [cli_server.sh](../../example/cli_server.sh) | `spark-mlx server` |
| [http_chat.py](../../example/http_chat.py) | `/v1/chat/completions`, messages, full/SSE |
| [http_completions.py](../../example/http_completions.py) | `/v1/completions`，原始 prompt，返回 `choices[].text` |
| [python_convert.py](../../example/python_convert.py) | Offline conversion |
| [cli_convert.sh](../../example/cli_convert.sh) | `spark-mlx convert` |

```sh
SPARK_MODEL=/path/to/TARGET bash example/cli_generate.sh --draft-model /path/to/DRAFT
SPARK_MODEL=/path/to/TARGET bash example/cli_server.sh --draft-model /path/to/DRAFT
.venv/bin/python example/http_chat.py
```

Python/生成 CLI 的思考参数使用顶层 `enable_thinking` 和 `reasoning_effort`。
Python/生成 CLI 默认开启 low 思考。`--no-enable-thinking` 关闭思考。
HTTP 对普通模型和 dSpark 保留 MLX-LM 0.31.3 的原有默认值及完整请求/响应协议。
KV 和采样参数均放在 `GenerationConfig`，加载器选项用于权重、算子后端及模型缓存。
Python 返回内容保留思考及工具标记；HTTP 沿用上游的 `reasoning`、`tool_calls` 响应结构。
工具执行由应用负责。

旧 Python/CLI 入口已移除；HTTP 保留上游协议。发布包需重新构建，既有压缩包保留发布时的接口。
