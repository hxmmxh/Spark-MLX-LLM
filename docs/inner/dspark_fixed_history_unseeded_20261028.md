# DSpark 固定历史窗口与无固定种子实验：2026-10-28

更新时间：2026-10-28T23:31:58.322201+08:00。此文件随实验更新；已成功 1166/1399，失败 0，完成 12/13 个数据集。

Draft 已修复为所有块内位置共享 anchor 前最多 512 个历史 token，包含左边界；块内保持双向可见。
Target temperature=1、top_p=0.95、top_k=0；draft temperature=1、top_p=1、top_k=0；block_size=7；max_tokens=4096；thinking=true/high；GPU、checkpoint BF16、KV 不量化。
无固定种子组使用 `--seed none`，不调用 `mx.random.seed`，同一进程内随机状态跨请求继续推进。LiveCodeBench 与其余数据集分两个新进程运行，均记录启动随机状态。

平均接受长度统一为 `1 + accepted_tokens / rounds`，吞吐为输出 token 总数 / 生成阶段总耗时。原始指标和计数仍保留。未完成行只统计已保存样本，不能直接与外部完整数据集比较。

| 数据集 | 成功/目标 | 平均接受长度 | tokens/s | 接受率 | llama.cpp 全 SWA | vLLM |
|---|---:|---:|---:|---:|---:|---:|
| aime24_format.json | 30/30 | 4.2263 | 79.24 | 46.13% | 4.3039 | 4.1581 |
| aime25_format_30.json | 30/30 | 4.1648 | 77.80 | 45.25% | 4.0764 | 4.1833 |
| alpaca_format_80.json | 80/80 | 2.9256 | 55.12 | 27.52% | 2.9733 | 2.8850 |
| arena-hard-v2-format-80.json | 79/79 | 3.2722 | 58.78 | 32.49% | 3.3204 | 3.2745 |
| gsm8k_format_80.json | 80/80 | 4.2759 | 77.23 | 46.81% | 4.3626 | 4.4056 |
| humaneval_format_80.json | 80/80 | 3.8919 | 69.75 | 41.34% | 3.8569 | 3.8743 |
| lbpp_format_80.json | 80/80 | 3.5093 | 64.70 | 35.88% | 3.5784 | 3.5331 |
| livecodebench_format_80.json | 80/80 | 3.8363 | 71.53 | 40.55% | 3.9123 | 3.8870 |
| math_format_80.json | 80/80 | 4.5349 | 84.80 | 50.52% | 4.6480 | 4.5942 |
| math_system.jsonl | 300/300 | 3.8810 | 72.57 | 41.18% | 3.9078 | 3.8743 |
| mbpp_format_80.json | 80/80 | 3.8616 | 73.20 | 40.90% | 3.9276 | 3.8671 |
| spark_agent_log_format_100.json | 100/100 | 3.2878 | 46.85 | 32.70% | — | — |
| spark_agent_log_format_300.json | 67/300 | 3.1042 | 41.12 | 30.08% | — | — |
| 跨框架对比的 11 集汇总 | 999/999 | 3.7330 | 69.15 | 39.07% | 3.7693 | 3.7387 |
| 全部 13 集汇总 | 1166/1399 | 3.6965 | 66.89 | 38.55% | — | — |

与 llama.cpp、vLLM 的当前差距：差值 = 本轮平均接受长度 − 参考值；百分比 =（本轮 / 参考值 − 1）× 100%。正值表示本轮较高，负值表示较低。
进行中数据集标为“阶段值”，仅反映当前部分样本与参考全量的差距；整体差距在 11 个对比数据集全部完成后计算。Agent 数据集没有外部参考，显示为 —。

| 数据集 | 成功/目标 | 本轮 | llama.cpp 全 SWA | 与 llama.cpp 差值（比例） | vLLM | 与 vLLM 差值（比例） |
|---|---:|---:|---:|---:|---:|---:|
| aime24_format.json | 30/30 | 4.2263 | 4.3039 | -0.0776（-1.80%） | 4.1581 | +0.0682（+1.64%） |
| aime25_format_30.json | 30/30 | 4.1648 | 4.0764 | +0.0884（+2.17%） | 4.1833 | -0.0185（-0.44%） |
| alpaca_format_80.json | 80/80 | 2.9256 | 2.9733 | -0.0477（-1.60%） | 2.8850 | +0.0406（+1.41%） |
| arena-hard-v2-format-80.json | 79/79 | 3.2722 | 3.3204 | -0.0482（-1.45%） | 3.2745 | -0.0023（-0.07%） |
| gsm8k_format_80.json | 80/80 | 4.2759 | 4.3626 | -0.0867（-1.99%） | 4.4056 | -0.1297（-2.94%） |
| humaneval_format_80.json | 80/80 | 3.8919 | 3.8569 | +0.0350（+0.91%） | 3.8743 | +0.0176（+0.45%） |
| lbpp_format_80.json | 80/80 | 3.5093 | 3.5784 | -0.0691（-1.93%） | 3.5331 | -0.0238（-0.67%） |
| livecodebench_format_80.json | 80/80 | 3.8363 | 3.9123 | -0.0760（-1.94%） | 3.8870 | -0.0507（-1.31%） |
| math_format_80.json | 80/80 | 4.5349 | 4.6480 | -0.1131（-2.43%） | 4.5942 | -0.0593（-1.29%） |
| math_system.jsonl | 300/300 | 3.8810 | 3.9078 | -0.0268（-0.69%） | 3.8743 | +0.0067（+0.17%） |
| mbpp_format_80.json | 80/80 | 3.8616 | 3.9276 | -0.0660（-1.68%） | 3.8671 | -0.0055（-0.14%） |
| spark_agent_log_format_100.json | 100/100 | 3.2878 | — | — | — | — |
| spark_agent_log_format_300.json | 67/300（阶段值） | 3.1042 | — | — | — | — |
| 跨框架对比的 11 集汇总 | 999/999 | 3.7330 | 3.7693 | -0.0363（-0.96%） | 3.7387 | -0.0057（-0.15%） |

与原版的比较仅使用双方均成功的相同输入，Agent Log 300 原版只保存了前 222 条。新旧随机轨迹可能不同。

| 数据集 | 配对数 | 原版平均接受长度 | 当前平均接受长度 | 差值 |
|---|---:|---:|---:|---:|
| aime24_format.json | 30 | 4.4268 | 4.2263 | -0.2004 |
| aime25_format_30.json | 30 | 4.4068 | 4.1648 | -0.2420 |
| alpaca_format_80.json | 80 | 2.8532 | 2.9256 | 0.0724 |
| arena-hard-v2-format-80.json | 79 | 3.2813 | 3.2722 | -0.0091 |
| gsm8k_format_80.json | 80 | 4.4389 | 4.2759 | -0.1630 |
| humaneval_format_80.json | 80 | 3.9297 | 3.8919 | -0.0378 |
| lbpp_format_80.json | 80 | 3.7195 | 3.5093 | -0.2102 |
| livecodebench_format_80.json | 80 | 3.5503 | 3.8363 | 0.2859 |
| math_format_80.json | 80 | 4.7649 | 4.5349 | -0.2299 |
| math_system.jsonl | 300 | 3.8699 | 3.8810 | 0.0111 |
| mbpp_format_80.json | 80 | 4.1660 | 3.8616 | -0.3044 |
| spark_agent_log_format_100.json | 100 | 3.2968 | 3.2878 | -0.0090 |
| spark_agent_log_format_300.json | 67 | 3.2637 | 3.1042 | -0.1595 |

固定历史窗口、seed=0 的 LiveCodeBench 组按用户要求在 25 条后停止，25 条均成功。对应原版平均接受长度 3.5310，修复后 3.5136（约 -0.49%）。
无固定种子组目前与该组可配对 25 条：seed=0 为 3.5136，不固定种子为 3.9608。

本次为单次无固定种子运行；固定种子子集没有显示窗口修复带来接受长度改善。随机策略的影响需结合完整结果判断，尚未做多次随机运行或代码执行判题。

llama.cpp 客户端没有传 seed，服务器文档默认 -1（随机）；draft 在请求开始时重置 sampler。MLX 本组则不重置全局随机状态，这两种随机策略并不完全相同。参考包缺少当时服务器日志，不能确认其启动参数。

验证：DSpark CPU 检查 76 项通过、1 项真实模型测试按默认设置跳过；draft GPU 检查 12 项通过；数据集脚本测试 25 项通过。新增窗口回归在旧 mask 上失败、在修复后通过；无固定种子 worker 对 seed() 调用设有失败检查。

结果与复现记录：

- [全部数据集汇总 JSON](../../.results/dspark-fixed-history-all-other-unseeded-20261028_110700/combined_summary.json)
- [数据集统计及配对 CSV](../../.results/dspark-fixed-history-all-other-unseeded-20261028_110700/combined_datasets.csv)
- [LiveCodeBench 对比 JSON](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/comparison.json)
- [LiveCodeBench 原始结果](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/livecodebench_format_80.json.results.jsonl)
- [其余数据集运行配置、输入与源码哈希](../../.results/dspark-fixed-history-all-other-unseeded-20261028_110700/job.json)
- [其余数据集运行日志](../../.results/dspark-fixed-history-all-other-unseeded-20261028_110700/run.log)
- [固定种子 25 条结果](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/fixed_seed_partial.results.jsonl)
