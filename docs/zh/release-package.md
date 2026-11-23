# macOS 发布包使用指南

[项目首页](../../README.zh.md) · [English](../en/release-package.md) · [文档目录](../README.md)

本文面向下载预编译发布包的用户。发布包自带 Python、MLX、MLX LM 和项目命令，
无需安装 Python、uv、Homebrew 或 Xcode，也无需克隆源码、激活虚拟环境。
模型权重单独准备，不包含在运行时发布包中。

版本和系统要求以发布包内的 `BUILD-INFO.json` 为准。

## 1. 确认系统要求

本文以当前生成的 `spark-mlx-0.1.2-macos26-arm64.tar.gz` 为例：

| 项目 | 要求 |
| --- | --- |
| 芯片 | Apple Silicon，原生 ARM64，例如 M1 及后续 M 系列；不支持 Intel Mac |
| 系统 | **macOS 26.2 或更新版本** |
| 内存 | **大于 8 GB 统一内存**，推荐运行 4B 量化模型 |

在左上角 ** → 关于本机** 查看芯片、系统版本和内存。

## 2. 校验并解压

将发布包和同名 `.sha256` 文件下载到同一目录，在该目录打开终端执行：

```sh
shasum -a 256 -c spark-mlx-0.1.2-macos26-arm64.tar.gz.sha256
tar -xzf spark-mlx-0.1.2-macos26-arm64.tar.gz
cd spark-mlx-0.1.2-macos26-arm64
```

校验应显示 `OK`。若下载的是其他版本，请相应替换文件名。
以下 `./bin/...` 命令均在解压后的发布包根目录执行。
目录可以搬移，也可以包含中文和空格；搬移时保留整个目录，不要单独复制 `bin/`。

## 3. 检查与运行

```sh
./bin/spark-mlx doctor --json
./bin/spark-mlx doctor --gpu
./bin/spark-mlx generate --model TARGET --draft-model DRAFT --prompt '你好' --max-tokens 512 --json
./bin/spark-mlx chat --model TARGET --draft-model DRAFT --stream
./bin/spark-mlx benchmark --model TARGET --draft-model DRAFT --input-file gen.txt --trials 5
```

普通生成省略草稿参数。聊天使用 `/quit` 退出、`/reset` 清空历史。
默认值内置在代码中，无需配置文件；`--config FILE` 仅用于显式覆盖。
完整参数和 Python 用法见[推理接口说明](../inference.md)。

### 启动服务

下面列出 4B 主模型及配套草稿的建议参数。替换两个模型路径后运行，并保持当前终端开启：

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

使用 1.7B 时，同时替换主模型、草稿路径及服务模型名。普通解码省略草稿相关参数。
关闭思考时，将 `--enable-thinking` 替换为 `--no-enable-thinking`。
`--max-context-length` 限制输入与输出的总 token 数；`--max-tokens` 只限制输出，
包含思考和最终回答。请求未指定采样参数时，使用上述服务配置。

### 测试 HTTP 请求

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

## 4. 命令与目录

`bin/spark-mlx` 是唯一命令入口，包含 `generate`、`chat`、`server`、`benchmark`、
`doctor`、`inspect` 和 `convert` 子命令。`bin/spark-mlx-python` 是包内隔离解释器。
通过 `--help` 查看各子命令选项。旧入口和参数别名已移除。

`BUILD-INFO.json` 保存构建与验证信息，`FILES.sha256` 保存文件校验值。
保留完整目录搬移，不要仅复制 bin；隔离 Python 不读取系统包或 PYTHONPATH。
原生回退检查可用 `SPARK_MLX_DISABLE_EXT=1 ./bin/spark-mlx doctor --gpu`。
模型下载见项目首页，模型权重不包含在发布包内。
