# LiveCodeBench：draft 窗口修复与随机种子实验

记录时间：2026-10-28T12:02:24.783474+08:00。无固定种子组已完成 **80/80**，失败 **0**；本文件是固定结果记录。

修复后的 draft 块中，所有位置共享 anchor 前最多 512 个历史 token，包含左边界；块内保持双向注意力。`--seed none` 禁用每条请求的 RNG 重置，使用新进程初始化的随机状态并跨请求持续推进。运行时检查确认本组 `mx.random.seed` 调用次数为 **0**。

其余配置与原版一致：target temperature=1、top_p=0.95、top_k=0；draft temperature=1、top_p=1、top_k=0；block_size=7；max_tokens=4096；thinking=true/high；GPU、checkpoint BF16、KV 不量化、prefill_step_size=512。80 条输入和 prompt token 数均逐条核对一致。

| 指标 | 原版窗口、每条 seed=0 | 修复窗口、不重置种子 | 变化 |
|---|---:|---:|---:|
| 平均接受长度 | 3.5503 | 3.8363 | +8.05% |
| Draft 接受率 | 36.46% | 40.55% | +4.10 个百分点 |
| 生成吞吐 tokens/s | 66.55 | 71.53 | +7.49% |
| 输出 token 总数 | 247155 | 268537 | — |
| 接受 draft token 总数 | 177504 | 198495 | — |
| 解码总轮数 | 69601 | 69985 | — |
| 生成耗时（秒） | 3714.01 | 3754.00 | — |

平均接受长度统一按 `1 + accepted_tokens / rounds` 计算。原始提交 token 定义的平均接受长度另保留在原始结果中。各比率均由累计计数计算，吞吐不包含 prefill；新组输出长度增加，因此吞吐提升并不表示本组总耗时减少。

| 外部参考 | 平均接受长度 | 原版 MLX 相对差距 | 本次 MLX 相对差距 |
|---|---:|---:|---:|
| llama.cpp 全 SWA | 3.9123 | -9.25% | -1.94% |
| vLLM | 3.8870 | -8.66% | -1.31% |

外部数字来自之前用户提供的截图，本次没有重跑其他框架。

为区分窗口与种子的影响，固定种子组按用户要求在 25 条后停止，25 条全部成功。以下仅比较相同的前 25 条：

| 配置 | 平均接受长度 |
|---|---:|
| 原版窗口、每条 seed=0 | 3.5310 |
| 修复窗口、每条 seed=0 | 3.5136 |
| 修复窗口、不重置种子 | 3.9608 |

仅修复窗口时，这 25 条变化约 -0.49%；在同一修复窗口下取消每条 seed=0 重置，变化约 +12.73%。结果支持随机策略是本次接受长度差距的重要来源。这里只做了一次无固定种子运行，不代表任意固定种子都会降低接受率；也未执行代码判题或准确率评估。

llama.cpp 测试客户端未传 seed，服务器文档默认 -1（随机）。它在新请求开始时重置 draft sampler，指定固定种子时 draft 使用 `seed ^ 0x85ebca6b`。本组 MLX 则跨请求延续全局 RNG，两种策略并不完全相同；参考包没有当时的服务器日志，不能核实是否有启动参数覆盖。

验证已完成：DSpark CPU 检查 76 项通过、1 项真实模型测试按默认设置跳过；draft GPU 检查 12 项通过；数据集脚本检查 25 项通过。新增窗口回归能在原 mask 上复现失败，修复后通过。源码快照、模型配置与输入哈希均已核对。

其余 12 个数据集共 1319 条已自动接续启动，沿用固定历史窗口与 `--seed none`，不重复 LiveCodeBench。[全数据集实时报告](dspark_fixed_history_unseeded_20261028.md)会继续自动更新。

- [本次逐条结果](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/livecodebench_format_80.json.results.jsonl)
- [完整统计与 25 条种子配对比较](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/comparison.json)
- [80 条逐样本配对 CSV](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/paired_samples.csv)
- [运行参数与源码哈希](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/job.json)
- [实际生效配置与初始随机状态](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/effective_config.json)
- [完成状态及种子重置检查](../../.results/dspark-fixed-history-livecodebench-unseeded-20261028_105100/worker_status.json)
- [窗口修复补丁](../../.results/dspark-fixed-history-livecodebench-20261028_103433/fixed_history.patch)
- [窗口回归验证记录](../../.results/dspark-fixed-history-livecodebench-20261028_103433/validation.json)
- [其余数据集运行目录](../../.results/dspark-fixed-history-all-other-unseeded-20261028_110700)
