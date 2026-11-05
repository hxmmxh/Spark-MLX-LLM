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
- **混合精度适配**：保留官方模型已有的 INT4、INT8、BF16 权重方案，支持配置 W8 执行后端。
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

支持直接加载 **compressed-tensors INT4 Safetensors 主模型**和 **BF16 GGUF 格式的 DSpark 草稿**，
无需提前转换为 MLX 模型。

未显式指定参数时，命令行生成与聊天使用以下默认配置：

| 配置 | 默认值与行为 |
| --- | --- |
| 思考模式 | 开启，强度为 `low` |
| 主模型温度 `temperature` | `0`，使用贪心采样，每步选择概率最高的 token |
| 主模型 `top_p` | `1`，不进行累计概率截断 |
| 主模型 `top_k` | `0`，不限制候选 token 数 |
| DSpark 草稿采样 | 温度、`top_p`、`top_k` 逐项继承主模型；默认同样使用贪心采样 |
| DSpark 草稿精度 | 浮点草稿加载时量化为 INT8；已有 INT8/INT4 草稿按保存格式加载 |
| 输出长度上限 | `512` tokens，包含思考与最终回答 |

温度大于 `0` 时，按 `top_p`、`top_k` 筛选后的概率分布随机采样。
示例命令中的显式参数会覆盖默认值。

以下以 4B 为例，请将 `/path/to/...` 替换为本地路径；使用 1.7B 时，同时替换主模型与草稿路径。

**DSpark 生成（推荐）：**

```sh
.venv/bin/python -m spark_mlx_llm.dspark generate \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
    --prompt '请用一段话解释推测解码。' \
    --temperature 1 --top-p 0.95 --top-k 0 \
    --draft-temperature 1 --draft-top-p 1 --draft-top-k 0 \
    --max-new-tokens 512 --stats
```

主模型通过 `--model` 指定，配套草稿通过 `--draft-model` 指定。
主模型也支持已转换的 MLX 目录，草稿也支持已导出的 MLX Safetensors 目录。
上例采用随机采样：主模型温度为 `1`、`top_p=0.95`，草稿温度为 `1`、`top_p=1`，两侧均不限制 `top_k`。

**DSpark 交互式聊天：**

```sh
.venv/bin/spark-mlx-chat \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf
```

该命令同时加载主模型与配套草稿，使用 DSpark 进行多轮对话。

`--max-tokens` 包含思考与最终回答的总长度，可按需调大。
其他思考模式见[思考模式说明](docs/zh/examples.md#思考模式)。

<details>
<summary>Python 生成示例</summary>

```python
from spark_mlx_llm.dspark import SamplingOptions, generate, load_pair

pair = load_pair(
    "/path/to/Spark-X2.6-4B-INT4",
    draft_path="/path/to/Spark-X2.6-4B-DSpark.gguf",
)
result = generate(
    pair,
    "请解释推测解码。",
    max_new_tokens=512,
    sampling=SamplingOptions(temperature=1.0, top_p=0.95),
    draft_sampling=SamplingOptions(temperature=0.0),
)
print(result.text)
print(result.draft_acceptance_rate, result.mean_acceptance_length)
```

普通生成与流式生成 API 见[接口使用说明](docs/zh/examples.md)。

</details>

<details>
<summary>更多选项：采样参数与草稿精度</summary>

#### 采样与生成参数

DSpark `generate` 的常用参数如下，完整参数可运行
`.venv/bin/python -m spark_mlx_llm.dspark generate --help` 查看。

| 参数 | 作用 |
| --- | --- |
| `--mode baseline` | 仅运行主模型，便于与 DSpark 对比 |
| `--temperature` / `--top-p` / `--top-k` | 主模型采样；默认温度为 `0`，使用贪心采样 |
| `--draft-temperature` / `--draft-top-p` / `--draft-top-k` | 草稿采样；未指定的字段逐项继承主模型设置 |
| `--max-draft-tokens N` | 限制每轮候选数；`0` 禁用草稿生成 |
| `--max-new-tokens N` | 限制生成长度；普通生成入口对应参数为 `--max-tokens` |
| `--stats` / `--json` | 将统计写入 stderr / 以 JSON 返回文本和指标 |
| `--raw-prompt` | 跳过默认应用的聊天模板 |

#### 草稿精度选项

浮点草稿默认在加载时量化为 INT8，已有 INT8/INT4 草稿直接按文件中的格式恢复。
如需保留浮点精度或改用 INT4，可通过 `--draft-quantization` 显式选择。

| `--draft-quantization` | 行为 |
| --- | --- |
| 不指定（默认） | 浮点草稿量化为 INT8；本项目已量化的 MLX 草稿按原格式恢复 |
| `checkpoint` | 保留文件记录的精度，直接加载浮点或本项目已量化的 MLX 草稿 |
| `bf16` | 保留浮点草稿权重；已有量化草稿需使用 `checkpoint` 或相同量化方案 |
| `int8` | 主要线性投影使用对称、逐输出通道 INT8，scale 为 FP32 |
| `int4` | QKV、FFN gate/up/down 使用非对称 INT4/G32，其余主要投影使用 INT8 |

| 草稿模块 | `int8` | `int4` |
| --- | --- | --- |
| Attention QKV、FFN gate/up/down | 对称 per-channel INT8，FP32 scale | 非对称 INT4/G32，BF16 scale/offset |
| 特征融合 `fc`、attention `out_proj`、Markov 输出投影 | 对称 per-channel INT8，FP32 scale | 对称 per-channel INT8，FP32 scale |
| 独立草稿 `lm_head`（如存在） | 对称 per-channel INT8 | 对称 per-channel INT8 |
| Token/Markov embedding、attention gate、归一化、置信度头 | 保留浮点 | 保留浮点 |
| 共享主模型输出头 | 沿用所选 readout 策略 | 沿用所选 readout 策略 |

量化不改写主模型或源文件，接受/拒绝校正使用量化后草稿的实际提议概率。
专用 W8 内核不可用时，使用量化权重的浮点计算回退，不保留原 BF16 线性层副本。
Python 的 `load_pair()` 使用相同默认值，通过 `pair.draft_quantization` 查看实际量化模块。
这些规则也适用于合并目录中的浮点草稿。

</details>

### 3. 启动 HTTP 服务

**DSpark 服务（推荐）：** 直接传入主模型目录与配套草稿文件。

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

当前 DSpark 服务一次处理一个请求。

**主模型服务：**

```sh
.venv/bin/spark-mlx-server \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --served-model-name spark-x2.6-4b \
    --host 127.0.0.1 --port 8080 \
    --thinking --reasoning-effort low \
    --temperature 0 --top-p 1 --top-k 0 \
    --max-tokens 262144
```

启动其中一种服务并保持终端运行，在另一个终端发送聊天请求：

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "spark-x2.6-4b",
      "messages": [{"role": "user", "content": "你好，请介绍一下自己。"}],
      "stream": false
    }'
```

请求中的 `model` 与启动时的 `--served-model-name` 保持一致，这里均为 `spark-x2.6-4b`。
请求未指定 `max_tokens` 时继承服务配置，也可按请求显式设置，但需满足上下文限制。
将 `stream` 改为 `true`，并为 curl 添加 `-N`，即可逐段接收 SSE 响应。使用 Ctrl-C 停止服务。
HTTP 文本续写和完整参数见[接口使用说明](docs/zh/examples.md)。

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

所有生成入口默认保留模型精度，不量化 KV cache：**BF16 模型使用 BF16 KV**。
如需对全局注意力层启用 INT8/G32 KV，
可向 `spark-mlx-generate` 显式传入 `--kv-bits 8`，或向本包 Python 生成函数传入
`kv_bits=8`。generate 与 Python 包装默认按 512 tokens 分块处理输入，chat/server
保留上游 MLX LM 的分块行为。这些设置与权重量化分别控制，详见
[KV 缓存与输入分块](docs/zh/examples.md#kv-缓存与输入分块)。

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
