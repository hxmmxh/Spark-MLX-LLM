# DSpark：模块职责与采样方式

DSpark 先由草稿模型提出多个候选 token，再由主模型验证并提交可接受的前缀。
`greedy`（贪心）和 `probabilistic`（概率采样）决定如何选择 token，以及如何接受草稿。
当前实现使用 `SamplingOptions.temperature` 切换模式，没有单独的
`--sampling-mode greedy/probabilistic` 参数。

## 模块职责

按“模型计算 → token 选择 → 接受判定 → 流程编排”划分职责。贪心与概率模式共用
同一组模块，在各自负责的步骤内选择算法。

| 文件 | 主要职责 | 主要接口 |
| --- | --- | --- |
| `config.py` | 解析、校验草稿几何、Spark 执行约定和层结构 | `DSparkConfig.from_dict/from_spark_dict`、`spark_layer_specs` |
| `checkpoint.py` | 检查权重文件头、索引、名称与形状 | `inspect_checkpoint` |
| `loader.py` | 加载权重与分词器，组装主模型和草稿模型，绑定共享输出头 | `_load_pair`、`ModelPair` |
| `drafter.py` | 实现草稿网络、特征融合、Markov 与置信度计算、上下文投影 | `DSparkDrafter` |
| `draft_ops.py` | 草稿 FFN 融合、线性层补齐、Markov 缓存和上下文 K/V-only 投影 | `install_draft_linear_padding`、`install_markov_bias_cache`、`install_draft_context_kv` |
| `draft_quantization.py` | 从已加载权重派生可选 W8 草稿后端，支持安装与恢复 | `install_derived_draft_w8` |
| `target.py` | 适配主模型前向，返回 logits 和指定层特征，提供输出投影 | `SparkTarget`、`TargetOutput` |
| `cache.py` | 保存 K/V，执行缓存提交、回滚与滑动窗口维护 | `TargetState`、`TransactionalKVCache`、`DraftKVCache` |
| `sampling.py` | 变换概率分布，选择主模型或草稿 token | `SamplingOptions`、`sample_logits`、`sample_draft`、`DraftPrediction` |
| `draft_policy.py` | 在验证前，根据置信度和预算确定提交长度 | `prefix_survival_probabilities`、`verification_length` |
| `acceptance.py` | 根据验证结果判定实际接受前缀，产生修正或额外 token | `greedy_accept`、`sampled_accept`、`AcceptanceResult` |
| `tokenization.py` | 编码提示词、应用聊天模板、校验输入 ID、统一解析 EOS 配置 | `encode_prompt`、`eos_token_ids` |
| `generation.py` | 调度 prefill、草稿和验证，处理 EOS、长度限制、缓存提交与流式事件，记录计数和耗时 | `_collect`、`_generate_events`、`TokenEvent` |
| `metrics.py` | 定义生成结果、统计口径、按草稿长度分组的计数及序列化 | `GenerationResult`、`DraftLengthStats` |
| `../benchmark/__init__.py` | 通过公共 generate 测量，保存每次结果 | 基准测试函数 |
| `../cli.py` | 统一解析检查、生成和基准测试命令，调用功能并展示结果；运行时依赖按需导入 | `main` |
| `../__main__.py` | 调用统一 CLI 入口 | `python -m spark_mlx_llm` |
| `__init__.py` | 汇总公开接口，延迟导入运行时依赖 | 包级导出 |

`drafter.py` 明确表示草稿网络；主模型的网络实现位于父目录的
`spark_mlx_llm/model.py`。`target.py` 复用该实现，通过适配器补充推测解码需要的
特征采集与事务缓存支持。

### target.py 的边界

[SparkTarget](../spark_mlx_llm/dspark/target.py) 提供三个主要接口：

- `make_state()`：根据主模型各层的全局或滑动注意力配置，创建缓存状态。
- `run(tokens, state)`：执行前向计算，返回 `TargetOutput(logits, taps)`。
  本次 K/V 暂存于事务缓存，后续由调用方决定提交长度；前向异常时撤销本次缓存写入。
- `project(hidden)`：把隐藏状态投影到词表空间，也可绑定给草稿模型共享输出头。

`generation.py` 依据接受结果、EOS 和长度限制决定提交多少，再调用 `cache.py` 的
`state.finish(keep)` 执行提交与回滚。`target.py` 只需了解网络与缓存接口，不依赖
采样策略、接受算法或统计结果。主模型适配器与原生前向的数值对比放在测试中。

### 一轮生成如何协作

1. `drafter.py` 执行草稿网络，得到隐藏状态；`sampling.sample_draft` 选出候选 token。
2. `generation.py` 根据输出预算和 `draft_policy.verification_length` 确定待验证前缀。
3. `target.run` 计算主模型 logits；整块路径交给 `acceptance.greedy_accept` 或
   `acceptance.sampled_accept` 判定接受前缀及修正/额外 token。
4. `generation.py` 处理 EOS，提交主模型缓存，更新草稿上下文缓存，输出 token 事件。
5. 生成循环记录计数与耗时，最终构造 `metrics.GenerationResult`；指标公式和序列化
   由 `metrics.py` 统一定义。

逐 token 验证辅助函数 `_verify_sequential` 仍属于 `generation.py`：它交替执行
主模型前向、提交缓存、检查 EOS，并在首次不匹配时结束，承担的是执行流程。
`acceptance.py` 的接受函数处理已经得到的预测或分布，不发起模型前向，也不操作缓存。

`GenerationResult` 推荐从包级接口或 `metrics.py` 导入；原来的
`from dspark.generation import GenerationResult` 仍可使用。

### 新增函数的归属原则

- 网络参数与张量计算属于 `drafter.py` 或 `target.py`。例如草稿上下文的 K/V **投影**
  使用网络权重，放在草稿网络；投影结果的**存储、窗口维护**放在 `cache.py`。
- 从 logits 选择候选属于 `sampling.py`；根据置信度决定送多少候选属于
  `draft_policy.py`；读取主模型验证结果、决定实际接受多少属于 `acceptance.py`。
- 需要调用模型、推进缓存、检查 EOS 或输出事件的过程属于 `generation.py`。
  prefill 分块也属于执行调度；字符串和 token ID 的转换属于 `tokenization.py`。
- 配置解释归 `config.py`，文件结构与权重形状检查归 `checkpoint.py`，权重加载与
  模型组装归 `loader.py`。`metrics.py` 只处理结果数据、计数和指标计算。
- 命令行参数和终端展示统一放在 `cli.py`；`__main__.py` 只调用入口。模型执行层
  不反向依赖 CLI、benchmark 或测试模块。

### 导入路径迁移

直接使用旧内部模块路径的脚本，需要改为以下路径。包级的生成、采样、接受与统计
接口仍可使用；草稿网络也增加了包级 `DSparkDrafter` 导出。

| 原路径 | 当前路径 |
| --- | --- |
| `dspark.model.DSparkDrafter` | `dspark.drafter.DSparkDrafter` 或 `dspark.DSparkDrafter` |
| `dspark.decoding.greedy_accept / sampled_accept / AcceptanceResult` | `dspark.acceptance` 中的同名接口 |
| `dspark.decoding.verification_length / prefix_survival_probabilities` | `dspark.draft_policy` 中的同名接口 |
| `dspark.generation.encode_prompt` | 推荐 `dspark.tokenization.encode_prompt`；原导入仍可使用 |

## 两种模式的区别

下表按主模型采样方式区分，并以草稿继承主模型采样参数为例。
主模型和草稿可以分别设置，具体见下方参数说明。

| 对比项 | greedy：贪心 | probabilistic：概率采样 |
| --- | --- | --- |
| 启用方式 | `temperature=0`，默认值 | `temperature>0` |
| 草稿如何选择 token | 取修正后 logits 最大的 token，即 argmax | 按 temperature、top-k、top-p 处理后的概率分布抽样 |
| 主模型如何接受草稿 | 草稿 token 必须等于主模型在该位置的 argmax | 以 `min(1, p(token) / q(token))` 的概率接受 |
| 首次拒绝后 | 输出主模型的 argmax 修正 token，丢弃剩余草稿 | 从归一化的 `max(p-q, 0)` 残差分布抽取修正 token，丢弃剩余草稿 |
| 所有草稿都接受后 | 主模型再给出一个 argmax token | 从主模型下一位置的分布再抽取一个 token |
| 返回的 `proposal_probs` | `None` | 保存草稿概率分布，供接受判定和拒绝修正使用 |
| 常见用途 | 检查贪心输出一致性、做固定策略的性能对照 | 需要按概率生成多样化回答的场景 |

两种模式都只接受从头开始连续通过的草稿前缀。首次拒绝后，即使后面的草稿 token
碰巧符合主模型，也不会跳过拒绝位置继续接受。EOS 和生成长度限制可能使一轮提前结束。

## greedy：每次选择 argmax

草稿模型先计算基础 logits，再根据前一个实际选出的 token 加入 Markov 修正，
最后取 argmax。下一位置继续使用这个已选 token 作为条件。

例如，草稿候选为 `A B C`，主模型依次认可 `A`，但在第二个位置认为 `D` 才是 argmax，
则本轮提交 `A D`，丢弃草稿 `B C`。若三个候选全部通过，主模型还能产生一个额外 token。

当前贪心路径直接使用 logits 的 argmax，不执行 softmax、top-k/top-p 筛选或随机抽样，
也不保存草稿概率分布。因此 `temperature=0` 时，设置 `top_p`、`top_k` 不会改变选择结果。

在相同 logits 下，argmax 的选择是确定的。不过，逐 token 和整块计算的浮点误差仍可能
改变非常接近的候选排名；当前 `batched` 路径不保证与逐 token baseline 完全一致，
具体限制见 [DSpark README](../spark_mlx_llm/dspark/README.md)。

## probabilistic：按分布选择，并进行拒绝修正

设 `q` 是草稿模型的提议分布，`p` 是主模型的目标分布。这里的分布均指经过实际配置的
temperature、top-k、top-p 处理并归一化后的结果。

对每个候选位置，执行以下步骤：

1. 从草稿分布 `q` 中抽取候选 token `x`。
2. 主模型计算相同前缀下的 `p(x)`，以 `min(1, p(x)/q(x))` 的概率接受 `x`。
3. 若接受，继续检查下一个候选；若拒绝，从 `max(p-q, 0)` 归一化后的分布抽取修正 token，
   并结束这一轮的前缀接受。
4. 若草稿全部通过，从主模型最后一个位置的分布抽取额外 token。

例如，在某个位置，两种分布为：

| token | 草稿概率 `q` | 主模型概率 `p` | 该 token 被抽中后的接受概率 |
| --- | ---: | ---: | ---: |
| A | 0.6 | 0.3 | 0.5 |
| B | 0.3 | 0.5 | 1 |
| C | 0.1 | 0.2 | 1 |

如果草稿抽中了 A，就以 50% 的概率接受；拒绝 A 后，残差为 `[0, 0.2, 0.1]`，
归一化后从 B、C 中分别以 `2/3`、`1/3` 的概率抽取修正 token。
按这个例子计算，最终 A、B、C 的概率分别为 `0.3`、`0.5`、`0.2`，与目标分布一致。

因此，概率模式不会要求草稿 token 等于主模型的 argmax，也不会先让主模型独立抽一个
token，再用“两个 token 是否相同”决定接受。拒绝修正的目的是保留主模型目标分布。
实际 BF16 和整块计算的数值误差仍可能影响该分布，不能据此承诺与逐 token 实现数值完全相同。

固定随机种子有助于在相同执行路径下复现实验，但概率推测解码与 baseline 的随机数
使用顺序不同，即使 seed 相同，也不要求生成完全相同的 token 序列。

## 参数怎么配置

| 参数 | 当前默认值 | 作用 |
| --- | --- | --- |
| `temperature` | `1.0` | 零表示贪心；正值表示按概率采样。正温度下，在截断前，较低温度使分布更集中，较高温度使分布更平缓 |
| `top_k` | `0` | 概率采样时按最高 k 个 logits 的阈值筛选；零表示不启用 |
| `top_p` | `0.95` | 概率采样时保留累计概率达到阈值的高概率前缀，再归一化；1 表示不启用 |
| `seed` | `0` | 控制概率抽样及接受判定使用的随机数 |

概率路径的处理顺序是：logits 转为 FP32、除以 temperature、应用 top-k、softmax、
应用 top-p 并重新归一化，再进行抽样。当筛选后只剩一个候选时，概率采样也会得到
确定的 token；`temperature>0` 本身不意味着每次输出一定不同。

主模型和草稿分别使用 `SamplingOptions`。CLI 的 `--draft-temperature`、
`--draft-top-p`、`--draft-top-k` 单独控制草稿；默认草稿温度 1、top-p 1、top-k 0，全部由 GenerationConfig 定义。
例如，主模型 `--temperature 1 --top-p 0.95` 搭配 `--draft-temperature 0`，即可使用
“草稿贪心、主模型概率采样”。运行时保留贪心草稿的 one-hot 提议分布，供概率接受和
拒绝修正使用。Python 中统一传入 `GenerationConfig`。

`confidence_threshold` 则控制提交验证的草稿长度：它使用置信度头的预测，
不是这里 `min(1, p/q)` 的接受概率。

在仓库根目录运行以下命令。`--model` 需要指向包含主模型和草稿权重的合并检查点。

```sh
# 贪心生成，使用整块验证
.venv/bin/python -m spark_mlx_llm generate \
  --model /path/to/checkpoint \
  --prompt "请介绍一下推测解码。" \
  --max-tokens 64 --temperature 0 \
  --stats

# 概率生成，验证路径自动使用 batched
.venv/bin/python -m spark_mlx_llm generate \
  --model /path/to/checkpoint \
  --prompt "请写一个简短的故事。" \
  --max-tokens 64 --temperature 0.8 \
  --top-p 0.95 --top-k 40 --seed 0 --stats
```

将 `generate` 换成 `benchmark` 可重复测量同一配置；使用 `--max-draft-tokens 0`
可测普通解码，分开保存并核对实际配置。完整调用与统计见[统一推理接口](inference.md)。

## 内部正确性测试中的 sequential / batched

这两组名称控制不同的事情：

- `greedy / probabilistic`：如何选择 token、如何接受草稿。
- `sequential / batched`：主模型逐 token 验证，还是一次验证同一请求的整段草稿。

| 组合 | 当前行为 |
| --- | --- |
| greedy + sequential | 支持；仅在逐 token 正确性测试时显式指定 |
| greedy + batched | 支持，也是默认设置；用于评估整块验证的加速效果 |
| probabilistic + batched | 当前概率生成使用的路径 |
| probabilistic + sequential | 未实现；内部正确性测试选择 sequential 时，正温度下仍走 batched |

`batched` 在这里表示一次处理多个候选 token，运行时仍是单请求。
接受率或速度不能仅凭采样方式判断，还取决于模型、提示词、采样参数和草稿长度；
概率模式还需要计算并保留概率分布，应以相同条件下的实测结果为准。

## 代码入口

两种草稿生成模式已统一到 [sampling.py](../spark_mlx_llm/dspark/sampling.py) 的
`sample_draft(drafter, block_hidden, anchor_ids, options, count=None)`：

- `SamplingOptions` 定义温度及 top-k/top-p 配置。
- `sample_draft` 共用 logits、Markov 修正和置信度计算，仅在 token 选择处区分两种策略。
- `DraftPrediction` 统一返回 `tokens`、`confidence`、`proposal_probs`；
  贪心模式默认 `proposal_probs=None`；主模型随机采样时，运行时显式要求保留其
  one-hot 提议分布。概率草稿保留每个位置的完整提议分布。
- [acceptance.py](../spark_mlx_llm/dspark/acceptance.py) 集中提供 `greedy_accept` 与
  `sampled_accept`，分别执行贪心接受、概率接受与残差修正。
- [generation.py](../spark_mlx_llm/dspark/generation.py) 调用上述接口，并处理缓存提交、
  EOS、长度限制和统计记录；[metrics.py](../spark_mlx_llm/dspark/metrics.py) 定义
  返回结果及指标计算。

Markov 修正和置信度计算始终使用前一个实际选出的 token：贪心模式使用 argmax 结果，
概率模式使用抽样结果。主模型采样和草稿采样通过 `SamplingOptions` 配置即可，无需
分别修改模型类或生成循环。
