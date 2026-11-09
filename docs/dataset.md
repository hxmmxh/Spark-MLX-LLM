# 数据集顺序推理

在项目根目录运行 `python -m spark_mlx_llm.dspark.tests.run_dataset`，接受目录或单文件。
目录只读取第一层的
`.json` / `.jsonl` 文件，按文件名排序；每个文件内严格按原记录顺序执行和输出，
不按 ID 排序、不打乱。文件内容支持 JSON 数组、单个 JSON 对象、JSONL；
`dspark_test` 中扩展名为 `.json` 的 JSONL 文件也可直接输入。

## 普通推理

```sh
cd /Users/xmhuang6/workspace/spark_mlx_llm_my

.venv/bin/python -m spark_mlx_llm.dspark.tests.run_dataset \
  /Users/xmhuang6/workspace/datasets/dspark_test \
  --model /Users/xmhuang6/workspace/models/spark_2_6_opensource/Spark-X2.6-1.7B-INT4-MLX \
  --max-tokens 512 \
  --output-dir .results/dataset-baseline
```

单文件只需替换第一个参数，例如：

```sh
.venv/bin/python -m spark_mlx_llm.dspark.tests.run_dataset \
  /Users/xmhuang6/workspace/datasets/dspark_test/alpaca_format_80.json \
  --model /Users/xmhuang6/workspace/models/spark_2_6_opensource/Spark-X2.6-1.7B-INT4-MLX \
  --max-tokens 128 --limit 3
```

`--limit N` 表示每个文件只处理前 N 条；不传则处理全部记录。
模型只加载一次，每条请求创建独立缓存。默认不预热，首次推理的冷启动开销计入该条耗时。

## DSpark 推理及接受率

`--dspark`（等价于 `--mode dspark`）需要包含主模型和草稿权重的合并检查点：

```sh
.venv/bin/python -m spark_mlx_llm.dspark.tests.run_dataset \
  /Users/xmhuang6/workspace/datasets/dspark_test/alpaca_format_80.json \
  --model /Users/xmhuang6/workspace/models/dspark_test_model/OPD_v1024_iter30001 \
  --dspark --max-tokens 512 \
  --output-dir .results/dataset-dspark
```

同一合并检查点去掉 `--dspark` 即使用主模型 baseline，保持相同的运行时和 KV 精度。
默认 `--greedy-verification batched` 使用整块验证；仅在正确性测试时显式指定
`--greedy-verification sequential`。两者的数值差异和用途见
[DSpark 文档](../spark_mlx_llm/dspark/README.md)。
默认每轮最多生成 3 个草稿 token，受检查点可用预测槽位数限制。
可用 `--max-draft-tokens N`、`--confidence-threshold 0.3` 控制草稿验证。

## 输出及统计口径

每个输入文件对应 `<原文件名>.results.jsonl`，例如
`alpaca_format_80.json.results.jsonl`。每行包括：

- `index`：从 0 开始的原记录序号；`id`：原始 ID。
- `input`：完整原始记录，包含 messages、tools 及其他字段。
- `output`：生成文本；`finish_reason`：`stop` 或 `length`。
- `metrics.input_tokens`：模型实际输入的 token 数，含聊天模板、历史消息和工具定义。
- `metrics.output_tokens`：实际生成的 token 数，包含生成出的 EOS；输出文本去掉末尾 EOS。
- `metrics.inference_seconds`：分词后开始计时，包含 prefill、解码、输出解码及同步；
  不包含模型加载、输入读取、模板处理、分词、结果写盘。
- `metrics.prefill_seconds` / `generation_seconds`：底层运行时报告的 prefill / 解码时间；
  首个输出 token 在 prefill 中产生，二者之和不必等于外层测得的总耗时。
- `metrics.output_tokens_per_second`：输出 token 数 / 完整推理耗时，包含 prefill。
- `metrics.tokens_per_second`：输出 token 数 / 解码耗时，不包含 prefill；
  与 DSpark `GenerationResult.tokens_per_second` 的口径一致。
- DSpark 下另外记录 `drafted_tokens`、`accepted_tokens`、`draft_acceptance_rate`。
  接受率为接受的草稿 token 数 / 送入验证的草稿 token 数（0–1）；无草稿时为 0。
  被置信度调度提前截断、未送入验证的预测位置不计入分母。
- `metrics.rounds`：解码循环的实际轮次数。
- `metrics.committed_tokens`：解码循环提交的 token 数，等于每条输出 token 数减去
  prefill 产生的首个 token，包含主模型补充/修正 token 及 EOS。
- `metrics.mean_acceptance_length`：`committed_tokens / rounds`，无解码轮次时为 0。
  这是每轮实际提交的平均长度，不仅包含被接受的草稿 token。
- `metrics.mean_draft_length`：`drafted_tokens / rounds`，每轮平均提交校验的草稿长度，
  包含长度为 0 的轮次，不包含主模型的 `pending` token；无解码轮次时为 0。
- `metrics.draft_length_stats`：按实际提交长度分组，JSON 键为长度字符串。
  每组记录 `rounds`、`drafted_tokens`、`accepted_tokens`、`draft_acceptance_rate`。
  接受率为组内实际接受并输出的草稿 token 总数 / 提交的草稿 token 总数，
  无草稿 token 时为 0，不是整段草稿全部通过的比例。

`summary.json` 包含每个文件和全局的上述统计，另外记录成功/失败条数、参数、
模型加载耗时 `model_load_seconds` 和加载后运行总耗时 `run_seconds`。
汇总接受率使用总接受数 / 总验证数，不对逐条接受率做算术平均。
汇总吞吐同样使用总输出 token 数 / 总推理耗时。
`tokens_per_second` 使用总输出 token 数 / 总解码耗时；`mean_acceptance_length`
使用总 `committed_tokens` / 总 `rounds`。三项均由累计计数计算，不直接平均逐条比率。
同目录的 `summary.csv` 提供每个文件及 `TOTAL` 的三项指标和原始计数，可直接用表格工具打开。
平均提交草稿长度和各长度组同样按累计计数汇总；`summary.csv` 中的
`draft_length_stats` 单元格使用 JSON 保存分组统计。
汇总会在每条记录完成后原子更新，运行中也可读取当前每个文件的接受率；
`status` 区分 `running`、`completed`、`completed_with_errors` 和 `interrupted`。
每个文件结束后，stderr 另外输出该文件的汇总。

进度写入 stderr，最终汇总 JSON 也输出到 stdout。每条结果写入后立即 flush。
单条格式、模板或推理错误会在原位置写入 `error`，`output` / `metrics` 为 null，
继续后续记录；失败条目不计入 token 和推理耗时总计，进程最终返回 1。
JSON 文件解析错误在加载模型前终止，并报告文件与行号。
Ctrl-C 保存已完成的记录及部分汇总，状态标为 `interrupted`，返回 130。
相同命令追加 `--resume --output-dir 原输出目录` 可继续已停止的任务，跳过已保存的行；
会校验参数和已保存行对应的原始输入，保留错误行，不重复推理已完成记录。
运行中状态的目录不能直接续跑，需要先结束原任务。旧版本缺少 `rounds` 的 DSpark 结果
需先精确补齐轮次后再续跑；无法确定轮次时不会用估算值替代。
有解码轮次但缺少 `draft_length_stats` 的旧结果也不能直接续跑，需在新输出目录重跑
以收集完整的分组统计，避免把旧结果和新结果混合后给出不完整的分组接受率。

默认输出目录为 `.results/dataset-<mode>-<UTC时间戳>`；已有结果文件不会覆盖。

## 输入与参数

每条记录需要非空 `messages` 数组，或字符串 `prompt`。
使用模型自带聊天模板，保留完整多轮对话、system、tools、tool_calls 和 reasoning_content；
工具调用的 JSON 字符串参数会转换成模板所需字典，输出中的原始记录不变。
脚本只生成文本，不执行对话中的工具调用。

默认沿用模板的 thinking 设置；`--no-enable-thinking` 可关闭思考。
`--enable-thinking --reasoning-effort high` 显式开启高档思考，
`--reasoning-effort low` 选择低档；本地 Spark 模板默认档位为 `high`。
`--raw-prompt` 跳过模板，仅支持单条 user 字符串消息或 `prompt`，
遇到多轮对话或工具定义会明确报错。

采样参数：`--temp 0.8 --top-p 0.95 --top-k 40 --seed 0`。

`--seed` 默认是 `0`，每条请求都会重置随机状态。使用 `--seed none` 可禁用重置，
让进程的随机状态在请求之间继续推进；多次运行的生成结果可能不同。
主模型和 draft 可分别设置；例如在上述参数后追加
`--draft-temp 1 --draft-top-p 1 --draft-top-k 0`，让草稿使用温度 1 且不截断概率分布。
`--draft-temperature` 是 `--draft-temp` 的别名；每个未指定的 draft 参数继承主模型
对应值。`--draft-temp 0` 可让草稿贪心生成，同时主模型继续随机采样。
这些参数同样写入 `summary.json` 的 `options`，续跑时会校验一致性。
默认贪心解码；每条请求重置随机种子，单文件和目录运行中的相同记录使用相同种子。

脚本默认 `--kv-bits none`、W8/W4 后端为 `native`，以便普通推理和 DSpark 使用一致的
基准配置。普通检查点可显式设置 `--kv-bits 8` 和算子后端；合并 DSpark 检查点
当前只支持 `--kv-bits none`。其他参数见
`.venv/bin/python -m spark_mlx_llm.dspark.tests.run_dataset --help`。
