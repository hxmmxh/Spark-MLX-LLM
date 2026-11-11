<p align="center">
  <img src="docs/assets/spark-mlx-logo.png" alt="星火与 MLX" width="480">
</p>

<h1 align="center">Spark-MLX-LLM</h1>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

Spark-MLX-LLM 是面向 **Spark 2.6 和 Spark 2.5** 的本地推理项目，基于
[MLX](https://github.com/ml-explore/mlx) 和 [MLX LM](https://github.com/ml-explore/mlx-lm)。
项目针对星火模型结构、混合精度权重和 Apple Silicon 进行适配与优化，
支持直接加载官方 INT4 模型，提供文本生成、交互式聊天、Python API 和 HTTP 服务。

[环境配置](#环境配置) · [生成与聊天](#2-生成与聊天) · [HTTP 服务](#3-启动-http-服务) ·
[量化与兼容性](#量化与兼容性说明) · [文档导航](#文档导航)

## 项目特色

- **官方模型直接加载**：支持 Spark-X2.6-4B-INT4 和 Spark-X2.6-1.7B-INT4，无需提前转换。
- **混合精度适配**：保留官方模型已有的 INT4、INT8、BF16 权重方案。
- **DSpark 推测解码**：支持分别加载主模型与配套草稿，或加载合并模型，并独立配置两侧采样参数。
- **Apple Silicon 优化**：提供专用 QKV/RoPE、线性层和 INT8 KV 注意力算子，按配置、设备与输入形状选择执行路径。
- **多种使用方式**：提供命令行生成与聊天、Python 普通与流式生成 API，以及 HTTP 服务。

## 环境配置

以下以 Apple Silicon 为例。Linux/CUDA 安装与常见问题见[环境配置说明](docs/zh/environment.md)。

| 项目 | 要求 |
| --- | --- |
| 硬件与系统 | Apple Silicon；原生 MLX 兼容目标为 macOS ≥ 14 |
| Python | 原生 `arm64` CPython，推荐 3.12 |
| 可选加速扩展 | macOS ≥ 26.2；源码构建需要 Xcode |

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再获取源码并安装依赖：

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
uv python install 3.12
uv sync --locked
```

安装会自动创建 `.venv`，默认使用原生 MLX，不安装扩展，也不需要 Xcode。
满足扩展要求时，可额外执行：

```sh
uv sync --locked --extra metal
```

程序会根据运行环境自动选择可用的加速方式。

后续命令均在仓库根目录执行，无需激活虚拟环境。推理需要 Metal 或 CUDA GPU，模型转换在 CPU 上执行。

## 快速开始

### 1. 模型下载

从以下任一平台下载文件。主模型需要包含配置、分词器和 Safetensors 权重的完整目录；
DSpark 草稿只需下载对应的 BF16 GGUF 文件。

| 用途 | 模型仓库 | Hugging Face | 魔搭 |
| --- | --- | --- | --- |
| 4B 主模型 | Spark-X2.6-4B-INT4 | [下载](https://huggingface.co/XHToken/Spark-X2.6-4B-INT4) | [下载](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-INT4) |
| 4B 草稿 | Spark-X2.6-4B-DSpark-GGUF | [下载](https://huggingface.co/XHToken/Spark-X2.6-4B-DSpark-GGUF) | [下载](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-DSpark-GGUF/files) |
| 1.7B 主模型 | Spark-X2.6-1.7B-INT4 | [下载](https://huggingface.co/XHToken/Spark-X2.6-1.7B-INT4) | [下载](https://www.modelscope.cn/models/XHToken/Spark-X2.6-1.7B-INT4) |
| 1.7B 草稿 | Spark-X2.6-1.7B-DSpark-GGUF | [下载](https://huggingface.co/XHToken/Spark-X2.6-1.7B-DSpark-GGUF) | [下载](https://www.modelscope.cn/models/XHToken/Spark-X2.6-1.7B-DSpark-GGUF/files) |

**推荐使用 DSpark 推测解码。** 相比仅使用主模型，DSpark 在 4B 模型上约有 **1.6×** 的加速，
在 1.7B 模型上约有 **1.2×** 的加速。
请配套下载主模型与对应的 `Spark-X2.6-4B-DSpark.gguf` 或 `Spark-X2.6-1.7B-DSpark.gguf`，
主模型与草稿必须按尺寸匹配，生成与服务示例见下文。

### 2. 生成与聊天

普通解码和 dSpark 共用 `load()`、`generate()` 及 `spark-mlx` 命令。
Python 和生成/聊天 CLI 使用 `GenerationConfig`，
完整约定见[推理接口说明](docs/inference.md)。
默认值已内置在代码中，无需配置文件；只有需要 JSON 覆盖时才传 `--config FILE`。

```sh
.venv/bin/spark-mlx generate --model /path/to/TARGET --draft-model /path/to/DRAFT --prompt '请介绍推测解码。' --json
.venv/bin/spark-mlx chat --model /path/to/TARGET --draft-model /path/to/DRAFT --stream
```

普通检查点省略 `--draft-model`；合并检查点自动识别草稿，`--no-use-draft` 只加载主模型。
统一默认值：主模型温度 1 / top-p 0.95，草稿温度 1 / top-p 1，seed 0，
输出上限 512，每轮最多 3 个草稿 token，low 思考，两项异步均开启。

```python
from spark_mlx_llm import GenerationConfig, load, generate

model, tokenizer = load("/path/to/TARGET", draft_model="/path/to/DRAFT")
config = GenerationConfig()
result = generate(model, tokenizer, "请介绍推测解码。", config=config)
print(result.text)
for chunk in generate(model, tokenizer, "请介绍推测解码。", config=config, stream=True):
    print(chunk.text, end="", flush=True)
```

### 3. 启动 HTTP 服务

#### 启动服务

下面列出 4B 主模型及配套草稿的建议参数。替换两个模型路径后运行，并保持当前终端开启：

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

使用 1.7B 时，同时替换主模型、草稿路径及服务模型名。普通解码省略草稿相关参数。
关闭思考时，将 `--enable-thinking` 替换为 `--no-enable-thinking`。
`--max-context-length` 限制输入与输出的总 token 数；`--max-tokens` 只限制输出，
包含思考和最终回答。请求未指定采样参数时，使用上述服务配置。

#### 测试 HTTP 请求

在另一个终端发送请求，`model` 与启动时的 `--served-model-name` 保持一致：

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "spark-x2.6-4b",
      "messages": [{"role": "user", "content": "你好，请介绍一下自己。"}],
      "stream": false
    }'
```

流式测试时，为 curl 添加 `-N`，并将 `"stream"` 改为 `true`。

也可以在项目根目录运行提供的测试脚本：

```sh
.venv/bin/python example/http_chat.py
.venv/bin/python example/http_completions.py
```

## 量化与兼容性说明

上述两款官方 INT4 主模型采用以下混合精度方案。加载或转换时保留已有整数权重编码，
将 scale/offset 适配为 MLX 存储表达；类型转换与舍入可能影响数值结果。
激活列描述 W8A8 后端可用且启用时的计算方式。

| 层 / 模块 | 权重量化方式 | 激活（启用 W8A8 时） | 权重 scale 类型 |
| --- | --- | --- | --- |
| `self_attn.q_k_v_proj` | 非对称 INT4，group size 32 | BF16 | BF16 |
| `mlp.gate_proj` | 非对称 INT4，group size 32 | BF16 | BF16 |
| `mlp.up_proj` | 非对称 INT4，group size 32 | BF16 | BF16 |
| `mlp.down_proj` | 非对称 INT4，group size 32 | BF16 | BF16 |
| `self_attn.out_proj` | 对称 INT8，逐输出通道 | 按 token 动态 INT8 | BF16 |
| `lm_head` | 对称 INT8，逐词表行 | 按 token 动态 INT8 | FP32 |
| `self_attn.g_proj` | 保留 BF16，不量化 | BF16 | — |
| `model.embedding` | 保留 BF16，不量化 | — | — |
| RMSNorm | 保留 BF16，不量化 | 不量化 | — |

KV 默认保留模型精度。Python/生成 CLI 默认输入分块为 512 tokens；
HTTP 保留原有的 2048-token 默认值，可显式覆盖。
普通解码可通过 `GenerationConfig(kv_bits=8)` 或 `--kv-bits 8` 启用量化 KV，
详见[统一推理接口](docs/inference.md)。

## 文档导航

全部文档见[文档目录](docs/README.md)。

| 主题 | 文档 |
| --- | --- |
| 安装、平台要求与常见问题 | [环境配置](docs/zh/environment.md) |
| 预编译发布包的解压、诊断与模型部署 | [发布包使用指南](docs/zh/release-package.md) |
| CLI、Python、HTTP、思考模式与 KV 缓存 | [接口使用说明](docs/zh/examples.md) |
| 模型格式转换与精度说明 | [模型转换](docs/zh/conversion.md) |
| 性能测试命令与指标口径 | [Benchmark 指南](docs/zh/benchmark.md) |
| 普通与 DSpark 推理 | [生成与聊天](#2-生成与聊天) · [HTTP 服务](#3-启动-http-服务) |
| Apple Silicon 扩展与算子 | [扩展说明](extensions/spark-mlx-ext/README.zh.md) |

## 许可证

参见 [LICENSE](LICENSE)。
