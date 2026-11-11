# 统一推理接口 / Unified inference API

普通解码和 dSpark 使用同一个 `load()`、`generate()`、CLI 和 HTTP 服务。
本次接口变更不保留旧 Python 函数和命令；HTTP 保留原普通服务的方法、路由、参数及响应协议。
发布包必须从此版本源码重新构建；
已有压缩包不会随源码更新。

## Python

```python
from contextlib import closing
from spark_mlx_llm import GenerationConfig, load, generate

model, tokenizer = load(
    "/path/to/Spark-X2.6-4B-INT4",
    draft_model="/path/to/Spark-X2.6-4B-DSpark.gguf",  # 可省略
)
config = GenerationConfig()
result = generate(model, tokenizer, "请介绍推测解码。", config=config)
print(result.text)
print(result.metrics())

with closing(generate(model, tokenizer, "请介绍推测解码。", config=config, stream=True)) as chunks:
    for chunk in chunks:
        print(chunk.text, end="", flush=True)
        if chunk.result is not None:
            final_result = chunk.result
```

`generate()` 返回 `GenerationResult`，其中 `text` 是完整文本，`token_ids` 包含 EOS。
`stream=True` 返回 `GenerationChunk` 迭代器，`text` 是本次增量，最后一项的
`result` 保存完整结果。提前终止消费时应关闭迭代器。

输入可以是字符串、消息列表或原始 token ID 列表。字符串默认套聊天模板；
原始文本续写设 `apply_chat_template=False`，token ID 始终直接使用。
消息支持文本和 `[{"type":"text","text":"..."}]` 分段；工具定义通过 `tools=[...]`
传给模板。返回内容保留模型的思考和工具标记，工具执行由调用方负责。

省略 `draft_model` 加载普通目标；合并检查点会自动识别草稿。
`load(..., use_draft=False)` 只加载主模型，单次请求 `max_draft_tokens=0` 禁用草稿。
浮点草稿默认 INT8，已有量化草稿保留检查点格式；可用 `draft_quantization`
选择 `checkpoint`、`bf16`、`int8` 或 `int4`。W4 默认 `multirow`，可由检查点或
显式加载选项覆盖。W8 路由自动选择：主模型保持 native，草稿单行使用 W8A16，
M5+ 多行保留 NAX，兼容旧 GPU（本轮验证为 M4）使用 matrix。
不再接受 `w8_qmv_backend`、`--w8-qmv-backend` 或转换参数 `--w8-backend`；
旧检查点的路径偏好忽略，量化权重及其系数精度保持。实际选择写入结果的加载元数据。

## 唯一配置

Python、生成/聊天 CLI 和 benchmark 使用 `GenerationConfig`；HTTP 保留 MLX-LM 0.31.3
的原有参数、校验和默认值，同时接受草稿设置。共有字段同名，CLI 将下划线写作连字符。
以下默认值适用于 Python、生成/聊天 CLI 和 benchmark。完整字段及校验位于
[`config.py`](../spark_mlx_llm/config.py)，`spark-mlx generate --help` 列出全部选项。

| 参数 | 统一默认值 |
| --- | --- |
| 主模型 temperature / top_p / top_k / min_p | 1 / 0.95 / 0 / 0 |
| 草稿 draft_temperature / draft_top_p / draft_top_k | 1 / 1 / 0 |
| seed | 0；null 不重置随机状态 |
| max_tokens | 512；null 等待自然 EOS |
| max_context_length | null；可选的输入与输出总 token 预算 |
| max_draft_tokens | 3；实际值不超过检查点槽位数，无草稿时为 0 |
| asynchronous_draft / async_context / compile_draft_chain | true / true / true |
| enable_thinking / reasoning_effort | true / low |
| apply_chat_template | true |
| prefill_step_size | 512 |
| cache_limit_mib / clear_cache | 256 / false |
| kv_bits / kv_group_size / quantized_kv_start / kv_quantization_scope | null / 32 / 0 / full-attention |
| stop_at_eos / confidence_threshold | true / 0 |

异步草稿和上下文仅在 GPU、随机采样及实际启用草稿时生效；草稿异步还要求未启用
置信度调度。结果同时记录 requested 和 effective，不能只看传入的 true。
KV 默认保留模型精度；量化 KV 需禁用草稿。`stop_at_eos=False` 用于固定长度诊断，
应显式指定有限 `max_tokens`。

默认值直接定义在 `GenerationConfig` 中，不需要配置文件；不传 `--config` 时不会读取 JSON。
`example/inference.json` 仅保留为可选配置样例，示例脚本默认使用代码内置值。
需要自定义覆盖时，显式传入 `--config /path/to/custom.json`。
配置优先级为代码默认值 → 可选 `--config` JSON → 显式 CLI 参数 → 单次请求 JSON。
Python 可用 `GenerationConfig.from_file()` 和 `.overlay({...})` 应用相同层次。
Python 和生成 CLI 的未知字段、旧别名和错误类型立即报错；思考参数统一放顶层，
`chat_template_kwargs` 仅用于额外模板参数。

## 单一 CLI

安装后使用 `spark-mlx`，或 `.venv/bin/python -m spark_mlx_llm`：

```sh
spark-mlx generate --model TARGET --draft-model DRAFT --prompt '你好' --json
spark-mlx generate --model TARGET --prompt '你好' --stream --stats
spark-mlx chat --model TARGET --draft-model DRAFT --stream
spark-mlx server --model TARGET --draft-model DRAFT --served-model-name spark
spark-mlx benchmark --model TARGET --draft-model DRAFT --input-file gen.txt --trials 5
spark-mlx generate --model TARGET --print-config
spark-mlx doctor --gpu
```

普通推理只省略草稿参数，其余命令和配置相同。聊天使用 `/quit` 退出、`/reset` 清空历史。
布尔参数如 `--asynchronous-draft` / `--no-asynchronous-draft`；可空整数用 `none`，
例如 `--max-tokens none`。`--print-config` 在加载模型前输出请求配置，
结果中的 `effective_config` 才包含实际设备和生效条件。
模型工具使用 `spark-mlx inspect --model MODEL`、`spark-mlx convert --help`。

## HTTP：保留原服务协议

普通模型和包含 dSpark 的模型共用一个 `mlx-lm==0.31.3` 服务。上游负责 HTTP 方法、
路由、参数解析、模型/适配器选择、停止序列、工具及思考内容处理和响应格式；
草稿模型的差异仅在加载与生成适配器内，不增加另一套服务或请求协议。
普通生成继续使用上游引擎，dSpark 使用同一目标采样及 logits 处理规则验证草稿。

| 方法及路由 | 行为 |
| --- | --- |
| `POST /v1/completions` | 原始文本续写，返回 `choices[].text` |
| `POST /v1/chat/completions` | 聊天生成，返回 `message` 或流式 `delta` |
| `POST /chat/completions` | 原有聊天路由别名 |
| `GET /v1/models` | 上游模型列表；指定 `--served-model-name` 时返回该服务名 |
| `GET /v1/models/<model_id>` | 上游模型查询子路径；命名服务沿用原固定列表行为 |
| `GET /health` | 原有 `{"status":"ok"}` 健康响应 |
| `OPTIONS /*` | CORS 预检，返回 204 |

原有 HTTP 参数继续可用：

| 参数 | 用途 |
| --- | --- |
| `model` / `draft_model` / `adapters` | 模型、草稿和适配器选择，保留按请求加载能力 |
| `max_tokens` / `max_completion_tokens` | 输出上限，后者沿用上游优先级 |
| `max_context_length` | 单次输入与输出总预算；请求只能收紧服务启动时的上限 |
| `temperature` / `top_p` / `top_k` / `min_p` | 目标模型采样 |
| `stop` | 单个停止字符串或字符串列表，支持跨多个 token 的停止序列 |
| `logprobs` / `top_logprobs` | 非流式响应的 token 概率和候选概率，保留上游格式 |
| `logit_bias` | 按 token ID 调整 logits |
| `repetition_penalty` / `repetition_context_size` | 重复惩罚及历史窗口 |
| `presence_penalty` / `presence_context_size` | 出现惩罚及历史窗口 |
| `frequency_penalty` / `frequency_context_size` | 频率惩罚及历史窗口 |
| `xtc_probability` / `xtc_threshold` | XTC 采样；共用适配层修正上游特殊 token 索引列表的嵌套问题 |
| `seed` | 单次请求的随机种子 |
| `chat_template_kwargs` / `role_mapping` / `tools` | 聊天模板、角色映射和工具定义 |
| `stream` / `stream_options.include_usage` | SSE 及独立 usage 事件 |
| `num_draft_tokens` | 原有草稿长度字段；dSpark 的实际长度不超过训练槽位数 |

dSpark 额外支持 `max_draft_tokens`、`draft_temperature`、`draft_top_p`、`draft_top_k`、
`asynchronous_draft`、`async_context`、`compile_draft_chain` 和 `confidence_threshold`。
`num_draft_tokens=0` 可在同一个服务中执行普通解码。

HTTP 同时接受顶层 `enable_thinking`、`reasoning_effort` 和原有的
`chat_template_kwargs` 思考参数；同一个请求中顶层值优先，均可覆盖服务默认值。
HTTP 保留模型精度的 KV，不接受 `kv_bits`、`kv_group_size`、`quantized_kv_start`
或 `kv_quantization_scope`：请求含这些字段时返回 400，服务 CLI 及 `--config` JSON
显式提供这些字段时在加载模型前报错。Python API 和生成/聊天/benchmark CLI
仍支持 KV 量化；`--prefill-step-size` 继续作为服务启动选项使用。

HTTP accepts top-level `enable_thinking` and `reasoning_effort` as well as the
original `chat_template_kwargs`; top-level request values take precedence over
nested values and server defaults. HTTP retains model-precision KV caches and
rejects `kv_bits`, `kv_group_size`, `quantized_kv_start`, and
`kv_quantization_scope` in requests (400), server CLI options, and config files.
The Python API and generation/chat/benchmark CLI retain KV quantization support.
`--prefill-step-size` remains a server startup option.

未显式配置时，HTTP 保留原普通服务默认值：temperature=0、top_p=1、max_tokens=512、
prefill_step_size=2048；请求省略 seed 时不重置随机状态；思考行为沿用模型模板。
`--config` 及显式 CLI 参数可以覆盖这些默认值。HTTP 的旧参数校验和未知字段处理沿用
上游，不再经过只接受 `GenerationConfig` 字段的请求白名单。

原有服务 CLI 选项继续转交上游，例如 `--temp`、`--adapter-path`、`--chat-template`、
`--chat-template-args`、`--trust-remote-code`、`--decode-concurrency`、
`--prompt-concurrency`、`--prompt-cache-size`、`--prompt-cache-bytes`、`--pipeline`
和 `--log-level`。省略 `--model` 时可以按请求指定模型；
`--served-model-name` 仅在需要固定服务名时设置。默认 CORS Origin 为 `*`，
可用 `--allowed-origins https://one.example,https://two.example` 配置。

`--max-context-length` 与 `--max-tokens` 是独立设置：前者限制聊天模板、历史消息、
工具定义等完整输入与输出的总 token 数，后者只限制本次输出。两种解码模式共用该预算。
未设置总预算时，不额外增加服务级上限；设置该值不会扩展模型本身的上下文能力。

```sh
spark-mlx server --model TARGET --draft-model DRAFT \
  --max-context-length 262144 --max-tokens 4096
```

请求省略输出上限时，实际上限为 `min(服务默认 max_tokens, 总预算 - 输入 token 数)`；
显式传入的 `max_tokens` 或 `max_completion_tokens` 超出剩余空间时返回 400，
不截断输入或聊天历史。输入已占满总预算时同样返回 400。
Python/生成 CLI 也接受 `GenerationConfig(max_context_length=...)` 或同名命令行选项；
指定的有限输出上限必须满足预算，`max_tokens=None` 时最多生成至预算用尽。

```sh
spark-mlx server --model TARGET --draft-model DRAFT --served-model-name spark
curl http://127.0.0.1:8080/v1/completions -H 'Content-Type: application/json' -d '{"model":"spark","prompt":"春天来了，","max_completion_tokens":128,"stop":["结束"],"logprobs":true,"top_logprobs":3}'
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"spark","messages":[{"role":"user","content":"你好"}],"stream":true,"stream_options":{"include_usage":true}}'
```

非流式和流式的输出结构、`reasoning`、`tool_calls`、`finish_reason` 和 `usage` 由上游
统一组装；流式以 `[DONE]` 结束。HTTP 不保证返回 Python 结果中的额外 metrics。
工具执行仍由调用方负责。

`generate --request-file` 继续支持 Python/CLI 的共有配置字段；HTTP 的专用扩展参数
（如 `logprobs`、惩罚参数、模型/适配器选择）直接用于 HTTP，不宣称可原样回放到 Python API。
正式性能测量使用下述 Python/benchmark 结果协议。

## 可比较的测量

Python `generate()` 和 benchmark 的每次结果保存 `effective_config`（含 requested、设备、加载选项、SHA256）、
`prompt_sha256`、输出 token、终止原因及时间。配置哈希不同就先核对差异，
相同哈希也还需核对源码、模型/扩展版本及设备环境。

decode tok/s 统一为 `(输出 token 数 - 1) / generation_seconds`，首 token 归入 prefill，
EOS 计入输出。`prefill_seconds` 包含输入前向及首 token，不含加载和模板构造。
多次测量用 `sum(输出 token 数 - 1) / sum(generation_seconds)` 汇总，
保留每一次结果。`benchmark` 在相同配置下预热（只缩短输出上限），再完整执行所有重复。

性能比较保持输入、seed、模板、输出上限、流式模式、KV、缓存和异步策略一致。
固定 seed 在随机采样下也不能保证普通解码与推测解码的 token 轨迹一致；
两者的随机数消费不同。记录输出并在同一模式内比较重复性。
正式 GPU 基线遵循 [`experiments/README.md`](../experiments/README.md)，先冻结当前源码和二进制。
历史实验快照保留其原接口，仅用于明确指定的历史回放。
