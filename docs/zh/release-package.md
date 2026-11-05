# macOS 发布包使用指南

[项目首页](../../README.zh.md) · [English](../en/release-package.md) · [文档目录](../README.md)

本文面向下载预编译发布包的用户。发布包自带 Python、MLX、MLX LM 和项目命令，
无需安装 Python、uv、Homebrew 或 Xcode，也无需克隆源码、激活虚拟环境。
模型权重单独准备，不包含在运行时发布包中。

## 1. 确认系统要求

本文以当前生成的 `spark-mlx-0.1.0-macos26-arm64.tar.gz` 为例：

| 项目 | 要求 |
| --- | --- |
| 芯片 | Apple Silicon，原生 ARM64，例如 M1 及后续 M 系列；不支持 Intel Mac |
| 系统 | **macOS 26.2 或更新版本** |
| 内存 | **大于 8 GB 统一内存**，推荐运行 4B 量化模型 |

在左上角 ** → 关于本机** 查看芯片、系统版本和内存。

## 2. 校验并解压

将发布包和同名 `.sha256` 文件下载到同一目录，在该目录打开终端执行：

```sh
shasum -a 256 -c spark-mlx-0.1.0-macos26-arm64.tar.gz.sha256
tar -xzf spark-mlx-0.1.0-macos26-arm64.tar.gz
cd spark-mlx-0.1.0-macos26-arm64
```

校验应显示 `OK`。若下载的是其他版本，请相应替换文件名。
以下 `./bin/...` 命令均在解压后的发布包根目录执行。
目录可以搬移，也可以包含中文和空格；搬移时保留整个目录，不要单独复制 `bin/`。

## 3. 检查运行环境

```sh
./bin/spark-mlx-doctor
```

首行显示“能跑 / 不能跑”的结论，随后列出系统和 Metal GPU 检查结果。
检查失败时会说明原因。

```sh
./bin/spark-mlx-doctor --gpu   # 实测 GPU 运算
./bin/spark-mlx-doctor --json  # 输出 JSON，供脚本读取
```

## 4. 准备模型

模型下载地址见[项目首页](../../README.zh.md#1-模型下载)。支持直接使用
**原始 compressed-tensors INT4 目录**或**受支持的 Spark 主模型 GGUF 文件**。

将目录或 GGUF 文件路径传给 `--model` 即可。以下示例使用
`/path/to/Spark-X2.6-4B-INT4`；使用 GGUF 时替换为对应的 `.gguf` 文件路径。

## 5. 生成与聊天

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
./bin/spark-mlx-dspark generate \
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
./bin/spark-mlx-chat \
    --model /path/to/Spark-X2.6-4B-INT4 \
    --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf
```

该命令同时加载主模型与配套草稿，使用 DSpark 进行多轮对话。

`--max-tokens` 包含思考与最终回答的总长度，可按需调大。
其他思考模式见[思考模式说明](examples.md#思考模式)。

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

普通生成与流式生成 API 见[接口使用说明](examples.md)。

</details>

<details>
<summary>更多选项：采样参数与草稿精度</summary>

#### 采样与生成参数

DSpark `generate` 的常用参数如下，完整参数可运行
`./bin/spark-mlx-dspark generate --help` 查看。

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

## 6. 启动 HTTP 服务

**DSpark 服务（推荐）：** 直接传入主模型目录与配套草稿文件。

```sh
./bin/spark-mlx-server \
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
./bin/spark-mlx-server \
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
HTTP 文本续写和完整参数见[接口使用说明](examples.md)。

## 7. 命令与目录

| 命令 | 用途 |
| --- | --- |
| `spark-mlx-doctor` | 运行环境与 GPU 检查 |
| `spark-mlx-generate` | 普通文本生成 |
| `spark-mlx-chat` | 终端交互聊天；指定配套草稿时启用 DSpark |
| `spark-mlx-server` | HTTP 服务；指定配套草稿时启用 DSpark |
| `spark-mlx-convert` | 模型检查与转换 |
| `spark-mlx-dspark` | 默认推荐的 DSpark 文本生成及其他子命令 |
| `spark-mlx-dspark-server` | DSpark HTTP 服务 |
| `spark-mlx-python` | 包内隔离的 Python 解释器 |

各命令均可用 `--help` 查看参数；DSpark 生成参数用
`./bin/spark-mlx-dspark generate --help` 查看。包内 Python 不读取系统
site-packages 或 `PYTHONPATH`，也不会使用其他虚拟环境的依赖。

如需从其他工作目录调用命令，在发布包根目录执行：

```sh
export PATH="$PWD/bin:$PATH"
```

该设置只影响当前终端。更新时将新包解压到新目录，继续通过 `--model` 引用已有模型，
保留每个发布包自带的完整运行时。

包内 `README.txt` 提供简要说明，`BUILD-INFO.json` 记录版本与验证结果，
`NATIVE-AUDIT.json` 记录二进制要求，`FILES.sha256` 列出文件校验值。
重新构建发布包见[构建指南](release.md)。

## 8. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `./bin/...` | 进入解压后的根目录，并确认完整解压了发布包 |
| 系统版本不满足要求 | 升级到 macOS 26.2 或更新版本 |
| 找不到模型或 tokenizer 文件 | 检查 `--model` 的路径，保留完整模型目录；路径含空格时加引号 |
| 首次加载较慢 | 原始模型首次加载需要准备；后续可复用缓存，或直接使用已转换的 MLX 模型 |
| 内存不足 | 使用更小或更低精度的模型、缩短上下文，并减少同时加载的模型；DSpark 还需草稿内存 |
| 端口被占用 | 修改 `--port`，并将客户端 URL 改为相同端口 |
| `doctor --gpu` 提示设备锁超时 | 等待其他共享 GPU 工作结束；可用 `./bin/spark-mlx-doctor --gpu --lock-timeout 30` 等待最多 30 秒 |
| SHA256 校验失败 | 重新获取同一版本的压缩包和校验文件，校验通过后再使用 |
