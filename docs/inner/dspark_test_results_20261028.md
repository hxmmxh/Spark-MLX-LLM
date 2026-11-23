# DSpark 测试结果：2026-10-28 快照

保存时间：2026-10-28 10:05:56（Asia/Shanghai）。成功 **1291/1399**，失败 **0**；已完成 **12/13** 个数据集。保存时任务状态为 `running`。

本文件及同名数据目录是固定快照，不自动跟随后台任务更新。跨框架对比使用的 11 个数据集已经全部完成，共 999 条；两个 Agent 数据集不纳入该对比。

## 本轮配置

| 参数 | 值 |
|---|---|
| 模型 | `/Users/xmhuang6/workspace/models/dspark_test_model/OPD_v1024_iter30001` |
| 输入目录 | `/Users/xmhuang6/workspace/datasets/dspark_test` |
| 主模型采样 | temperature=1，top_p=0.95，top_k=0 |
| Draft 采样 | temperature=1，top_p=1，top_k=0 |
| 推测配置 | dspark_block_size=7，logits_start=0，最多 7 个 draft token + 1 个 target token |
| Draft 注意力 | 5 层全部为 SWA，窗口 512 |
| 生成上限 / 思考 | max_tokens=4096；enable_thinking=true，reasoning_effort=high |
| 随机种子 | seed=0，每条请求重置 |
| 验证 / 调度 | batched；confidence_threshold=0；max_draft_tokens 自动取模型上限 7 |
| 设备 / 精度 | GPU；dtype=checkpoint；KV 不量化 |
| Prefill / 算子 | prefill_step_size=512；W8/W4 后端均为 native |

## 与其他框架对比：mean_acceptance_length

llama.cpp 全 SWA 和 vLLM 数值来自用户提供的截图；用户已确认采样参数、块长、生成上限和思考设置一致。没有对其他框架重新执行测试，也没有取得其原始逐条结果或总轮数。

本节统一按截图定义计算：`1 + 接受的 draft token 总数 / 解码总轮数`。我们原始指标按实际提交 token 计数；在 draft 提前输出 EOS 的轮次，不会再输出 target token。这里按每轮固定加 1 对齐，因此与原始值有极小差别，单个数据集调整最多约 0.0057。

| 数据集 | 样本数 | 我们 MLX | llama.cpp 全 SWA | vLLM | 我们 − llama.cpp | 我们 − vLLM |
|---|---:|---:|---:|---:|---:|---:|
| AIME24 | 30 | 4.4268 | 4.3039 | 4.1581 | +0.1229 | +0.2687 |
| AIME25 | 30 | 4.4068 | 4.0764 | 4.1833 | +0.3304 | +0.2235 |
| Alpaca | 80 | 2.8532 | 2.9733 | 2.8850 | -0.1201 | -0.0318 |
| Arena-Hard-v2 | 79 | 3.2813 | 3.3204 | 3.2745 | -0.0391 | +0.0068 |
| GSM8K | 80 | 4.4389 | 4.3626 | 4.4056 | +0.0763 | +0.0333 |
| HumanEval | 80 | 3.9297 | 3.8569 | 3.8743 | +0.0728 | +0.0554 |
| LBPP | 80 | 3.7195 | 3.5784 | 3.5331 | +0.1411 | +0.1864 |
| LiveCodeBench | 80 | 3.5503 | 3.9123 | 3.8870 | -0.3620 | -0.3367 |
| Math | 80 | 4.7649 | 4.6480 | 4.5942 | +0.1169 | +0.1707 |
| Math System | 300 | 3.8699 | 3.9078 | 3.8743 | -0.0379 | -0.0044 |
| MBPP | 80 | 4.1660 | 3.9276 | 3.8671 | +0.2384 | +0.2989 |
| 整体加权 | 999 | 3.7806 | 3.7693 | 3.7387 | +0.0113 | +0.0419 |

整体上，我们比 llama.cpp 高 0.30%，比 vLLM 高 1.12%。最大落后项是 LiveCodeBench，分别低 9.25% 和 8.66%。Alpaca 也低于两者；AIME25、MBPP、LBPP 的值较高。

整体行中，我们按总轮次加权，其他两列使用截图中的整体值。这是单轮观测，尚不能将全部差异归因于框架实现；也没有进行代码执行判题或准确率评估。

## 本轮各数据集结果

下表保留脚本原始 mean_acceptance_length 定义；尚未完成的数据集只统计已保存的成功记录。

| 数据集 | 完成数 | 状态 | tokens_per_second | draft_acceptance_rate | mean_acceptance_length |
|---|---:|---|---:|---:|---:|
| AIME24 | 30/30 | 已完成 | 83.63 | 49.00% | 4.4265 |
| AIME25 | 30/30 | 已完成 | 82.57 | 48.70% | 4.4065 |
| Alpaca | 80/80 | 已完成 | 54.30 | 26.49% | 2.8517 |
| Arena-Hard-v2 | 79/79 | 已完成 | 61.33 | 32.62% | 3.2812 |
| GSM8K | 80/80 | 已完成 | 85.64 | 49.13% | 4.4332 |
| HumanEval | 80/80 | 已完成 | 74.41 | 41.87% | 3.9283 |
| LBPP | 80/80 | 已完成 | 70.03 | 38.88% | 3.7192 |
| LiveCodeBench | 80/80 | 已完成 | 66.55 | 36.46% | 3.5499 |
| Math | 80/80 | 已完成 | 90.68 | 53.81% | 4.7626 |
| Math System | 300/300 | 已完成 | 73.10 | 41.03% | 3.8692 |
| MBPP | 80/80 | 已完成 | 79.65 | 45.25% | 4.1646 |
| Spark Agent Log 100 | 100/100 | 已完成 | 48.55 | 32.83% | 3.2933 |
| Spark Agent Log 300 | 192/300 | 进行中 | 46.40 | 31.78% | 3.2206 |
| 本轮整体 | 1291/1399 | 进行中 | 68.28 | 38.99% | 3.7264 |

各数据集的输入/输出 token 总数、推理耗时、生成耗时、接受/提交计数保存在 [datasets.csv](dspark_test_results_20261028/datasets.csv)。

## 与旧参数对比

旧轮为主模型 top_p=1、块长 8；本轮为主模型 top_p=0.95、块长 7。其他主要参数相同，draft 层均为全 SWA。只对比两轮共同成功的 310 条相同输入；HumanEval 只有前 11 条可配对。由于两项参数同时改变，这里反映合并影响。

| 数据集 | 配对样本 | tokens/s：旧 → 新 | 接受率：旧 → 新 | 平均接受长度：旧 → 新 |
|---|---:|---:|---:|---:|
| AIME24 | 30 | 80.05 → 83.63 | 41.65% → 49.00% | 4.3286 → 4.4265 |
| AIME25 | 30 | 79.04 → 82.57 | 41.42% → 48.70% | 4.3095 → 4.4065 |
| Alpaca | 80 | 50.19 → 54.30 | 20.97% → 26.49% | 2.6751 → 2.8517 |
| Arena-Hard-v2 | 79 | 54.00 → 61.33 | 24.34% → 32.62% | 2.9446 → 3.2812 |
| GSM8K | 80 | 81.76 → 85.64 | 41.34% → 49.13% | 4.3006 → 4.4332 |
| HumanEval | 11 | 57.91 → 71.98 | 27.13% → 40.18% | 3.1666 → 3.8093 |
| 配对整体 | 310 | 60.23 → 67.15 | 28.26% → 36.69% | 3.2579 → 3.5657 |

## 统计口径

- `tokens_per_second`：输出 token 总数 / 生成阶段总耗时，不包含 prefill 耗时。
- `inference_seconds`：分词后开始计时，包括 prefill、生成、输出解码及同步。
- `draft_acceptance_rate`：接受的 draft token 总数 / 提交验证的 draft token 总数。
- 原始 `mean_acceptance_length`：实际提交的输出 token 总数（不含每条请求的初始 anchor）/ 解码总轮数。
- 跨框架对齐值：`1 + accepted_tokens / rounds`；只在跨框架表中使用，原始结果没有改写。
- 输出 token 包含实际发出的 EOS；本轮开启 think high，思考过程 token 也参与统计。
- 汇总采用累计计数与耗时计算比率，不直接平均逐条记录或逐数据集的比率。

## 文件与复现

- [主测试脚本](../../spark_mlx_llm/dspark/tests/run_dataset.py)
- [本轮 worker](../../.results/dspark-swa-block7-target-p095-draft-p1-4096-high-20261027_210542/worker.py)；[实时 summary.json](../../.results/dspark-swa-block7-target-p095-draft-p1-4096-high-20261027_210542/summary.json)
- [旧参数结果](../../.results/dspark-all-swa-4096-think-high-temp1-20261027_172807/summary.json)
- [本轮结果快照](dspark_test_results_20261028/summary.json)；[数据集统计 CSV](dspark_test_results_20261028/datasets.csv)
- [跨框架对比 CSV](dspark_test_results_20261028/framework_comparison.csv)；[原始数值与口径说明](dspark_test_results_20261028/framework_comparison.json)
- [旧参数配对对比 CSV](dspark_test_results_20261028/paired_parameter_comparison.csv)；[配对原始计数](dspark_test_results_20261028/paired_parameter_comparison.json)
- [运行参数、模型配置及源码哈希](dspark_test_results_20261028/provenance.json)

在项目根目录可用以下参数复现，输出目录需选择尚不存在的路径：

```bash
.venv/bin/python -m spark_mlx_llm.dspark.tests.run_dataset \
  /Users/xmhuang6/workspace/datasets/dspark_test \
  --model /Users/xmhuang6/workspace/models/dspark_test_model/OPD_v1024_iter30001 \
  --dspark --max-tokens 4096 \
  --temp 1 --top-p 0.95 --top-k 0 \
  --draft-temperature 1 --draft-top-p 1 --draft-top-k 0 \
  --enable-thinking --reasoning-effort high --seed 0 \
  --greedy-verification batched --device gpu --kv-bits none \
  --output-dir .results/dspark-reproduction-new-run
```

复现前须保持模型配置中的 dspark_block_size=7。本轮已加载的实际配置与模型配置快照记录在 provenance.json 中的 effective_config、job_manifest 字段；原始完整模型配置位于运行目录的 config.snapshot.json。
