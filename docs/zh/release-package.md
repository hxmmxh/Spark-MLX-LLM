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

指定模型时，服务默认先预热标题，再依次保存 System 1、System 1 + System 2、
System 1 + System 2 + User 1 固定部分三个累计主前缀，完成后再接收 HTTP 请求。
默认模板随发布包安装，路径不依赖当前工作目录，无需另行指定文件。
当前模板与 2026-11-26 Vulkan `d17ffbbfb` 的捕获资源一致，已用 MLX 模板重新渲染核对。
4B 模型的主前缀输入为 26,320 / 34,795 / 36,451 token，标题为 2,388 token；
每档留一个 token 供请求计算 logits。后两档接续前一档，工具、技能或固定工作区说明
变化后可匹配较短档位；配置变化时仍应重新捕获最新模板。
终端依次显示 `Prefill complete: ... input tokens, ... cached tokens` 和
`Starting httpd ...` 后，再连接客户端。
客户端继续发送完整请求；实际命中数见 `usage.prompt_tokens_details.cached_tokens`。
完整行为见[启动预热说明](../inference.md#启动时预先-prefill-固定前缀)。

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
    --prefill-newlines lf --cache-limit-mib 256 \
    --prompt-cache-size -1 --prompt-cache-bytes 8589934592 \
    --max-context-length 262144 \
    --max-tokens 262144
```

需要自定义预热时，在上面的启动命令末尾追加相应参数：

| 参数 | 用法 |
|---|---|
| `--prefill-file /path/to/prefix.txt` | 使用自己的 UTF-8 已渲染前缀，保留特殊消息标记和空白，不重复套聊天模板。省略时使用包内默认模板。 |
| `--prefill-newlines lf` | 将 CRLF/CR 规范为 LF，默认开启；改为 `--prefill-newlines preserve` 可保留文件原换行。 |
| `--skip-prefill` | 跳过启动预热，不能与 `--prefill-file` 同时使用。 |
| `--single-prefill` | 只预热完整主前缀，不预热标题和保存两个中间档位。自定义 `--prefill-file` 也使用单段方式。 |
| `--cache-limit-mib 256` | 启动预热时设置 MLX 分配器的临时缓存上限，单位 MiB；KV 前缀容量由 `--prompt-cache-bytes` 控制。 |

例如，自定义前缀时追加 `--prefill-file /path/to/prefix.txt`；跳过预热时追加
`--skip-prefill`。上述预热参数仅在服务启动时设置。
普通解码和 dSpark 均支持前缀复用；dSpark 同时预热 target KV 与 draft 上下文。
dSpark 也会缓存真实请求的前缀，供新窗口复用；动态窗口信息之后的内容仍需计算。
若启动模板不匹配，相同输入可从下次请求复用；新发现的共同前缀需先保存一次。
日志中的 `dSpark prompt cache: ... cached=... remaining=...` 可确认实际命中。
生成后也会缓存已提交的回答 KV 和 draft 上下文，后续发送完整历史可复用多轮前缀；
输入与生成后状态均占用缓存容量，未执行前向的末 token 仍需在下一轮计算。
标题和三个启动快照同样计入容量；预算不足时可能淘汰较早的档位。
按当前 4B 浮点 KV 布局估算，启动快照共约 3.73 GiB；再保留一组完整请求及回答
快照约需 6.4 GiB，因此示例使用 8 GiB。实际占用以服务日志为准。
`Startup prefill stage ... reused=... computed=...` 显示每档计算量；若启动结束时
未保留全部 dSpark 档位，日志会给出提示，可增加缓存预算或使用 `--single-prefill`。

上面的命令只限制 KV 总容量为 8 GiB，不限制条目数。容量参数的含义如下：

| 参数 | 用法 |
|---|---|
| `--prompt-cache-size -1` | 不限制条目数；正数限制条目数，省略时沿用上游默认的 10 条。一个对话可以占用多个条目。 |
| `--prompt-cache-bytes 8589934592` | KV 总容量上限为 8 GiB，包含输入前缀、已提交的回答 KV 和 draft 上下文；超出后按缓存策略淘汰。 |
| `--prompt-cache-size 0 --skip-prefill` | 禁用请求缓存，同时跳过需要缓存条目的启动预热。 |

`--prompt-cache-bytes` 控制 KV 缓存池容量；`--cache-limit-mib` 控制 MLX 分配器
保留的临时缓存。两者均不包含模型权重的容量。

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
