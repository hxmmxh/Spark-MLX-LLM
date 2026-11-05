# 主模型混合量化、BF16 draft：DSpark 开关对照

更新时间：2026-10-29T00:30:39.730671+08:00。状态：interrupted。成功 57/1998，失败 0；完整文件对照 0/11。

量化模型：`/Users/xmhuang6/workspace/models/dspark_test_model/OPD_v1024_iter30001-MLX-Mixed-W4G32-W8`。主模型 QKV/MLP 为 W4/G32，out_proj/lm_head 为逐通道 W8；W4/W8 均使用 native。

Draft 53 个原始张量逐字节校验一致。源模型没有独立 draft lm_head，原来共用 target 的 BF16 embedding 输出投影；本轮 worker 显式继续绑定未量化的 BF16 embedding.as_linear，避免 draft 跟随 target 改用 W8 头。原始模型未修改；推理使用源码快照，计时接入版本单独归档。

顺序：每个文件先 DSpark 全量，再 baseline 全量，然后下一个文件。排除 spark_agent_log_format_100.json 和 spark_agent_log_format_300.json，共 999 条输入、1998 次生成。

参数：target temperature=1/top_p=0.95/top_k=0；draft temperature=1/top_p=1/top_k=0；block_size=7；max_tokens=4096；thinking=true/high；seed=none；固定历史窗口512；GPU；主干激活/KV为BF16，native主输出头为FP32；不量化KV；prefill_step_size=512。每个文件/模式为独立进程，不显式重置 RNG；不额外进行生成预热，延续旧 BF16 测试口径。

吞吐按输出 token 总数 / 生成总耗时计算；端到端吞吐包含 prefill。DSpark 收益 = 量化 DSpark 吞吐 / 量化 baseline 吞吐，只有完整相同文件两组都完成才计算。BF16 比较使用最新固定窗口、不固定种子的旧运行，并严格配对相同输入；阶段值只比较已完成的同一子集。

平均接受长度使用 1 + accepted_tokens / rounds；实际提交长度和原始计数保留在 JSON 中。接受率变化单位是百分点。这是单次随机采样的性能比较，输出长度可能不同，不是准确率评测。

| 文件 | DSpark/关闭完成数 | DSpark tok/s | 关闭 tok/s | DSpark 加速比 |
|---|---:|---:|---:|---:|
| aime24_format.json | 30/27（各 30） | 87.49 | 68.58 | — |
| aime25_format_30.json | 0/0（各 30） | — | — | — |
| alpaca_format_80.json | 0/0（各 80） | — | — | — |
| arena-hard-v2-format-80.json | 0/0（各 79） | — | — | — |
| gsm8k_format_80.json | 0/0（各 80） | — | — | — |
| humaneval_format_80.json | 0/0（各 80） | — | — | — |
| lbpp_format_80.json | 0/0（各 80） | — | — | — |
| livecodebench_format_80.json | 0/0（各 80） | — | — | — |
| math_format_80.json | 0/0（各 80） | — | — | — |
| math_system.jsonl | 0/0（各 300） | — | — | — |
| mbpp_format_80.json | 0/0（各 80） | — | — | — |
| 已完成文件合计 | 0/0 | — | — | — |

| 文件 | BF16→量化 DSpark tok/s | 量化速度比 | BF16→量化接受率 | 变化（百分点） | BF16→量化平均接受长度 |
|---|---:|---:|---:|---:|---:|
| aime24_format.json（30条） | 79.24 → 87.49 | 1.104× | 46.13% → 42.40% | -3.73 | 4.2263 → 3.9653 |
| aime25_format_30.json（0条） | — → — | — | — → — | — | — → — |
| alpaca_format_80.json（0条） | — → — | — | — → — | — | — → — |
| arena-hard-v2-format-80.json（0条） | — → — | — | — → — | — | — → — |
| gsm8k_format_80.json（0条） | — → — | — | — → — | — | — → — |
| humaneval_format_80.json（0条） | — → — | — | — → — | — | — → — |
| lbpp_format_80.json（0条） | — → — | — | — → — | — | — → — |
| livecodebench_format_80.json（0条） | — → — | — | — → — | — | — → — |
| math_format_80.json（0条） | — → — | — | — → — | — | — → — |
| math_system.jsonl（0条） | — → — | — | — → — | — | — → — |
| mbpp_format_80.json（0条） | — → — | — | — → — | — | — → — |


## 解码分阶段耗时

计时从接入记录后的请求开始；历史请求没有分阶段信息，覆盖数单独列出。占比、每轮/每token耗时只用有记录的同一批请求计算。原有吞吐表仍包含全部成功请求。

沿用现有 mx.eval/.item 同步点，无额外 GPU 同步；表中是主机墙钟耗时，包含构图、提交和等待，不是 GPU kernel 时间。Draft 包含网络前向、BF16 输出头、Markov 和采样；主模型前向含输出头；接受采样含 top-p 分布处理和拒绝修正。缓存提交列只测主机提交，延迟执行的工作可计入后续阶段；上下文更新包含 draft KV 更新。其他包括循环、事件产出及未单列的处理。

主模型输出头的量化系数是 FP32，native 路径 logits 为 FP32；其他主干激活和 KV 为 BF16。保持原有权重、后端和采样配置。

接受长度是每轮输出量，不等于速度倍数：decode 吞吐约为每轮输出 token 数 / 每轮耗时。BF16→量化表两边均开启 DSpark；DSpark 自身加速需对比同一量化模型的 baseline。

| 文件 / 模式 | 记录覆盖 | 每轮输出 | 每轮 ms | 每输出 token ms | Draft ms（占比） | 主模型前向 ms（占比） | 接受采样 ms（占比） | 缓存提交 ms（占比） | 上下文更新 ms（占比） | 其他 ms（占比） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aime24_format.json / dspark | 8/30 | 3.887 | 44.185 | 11.369 | 9.802（22.2%） | 30.731（69.6%） | 2.403（5.4%） | 0.049（0.1%） | 1.188（2.7%） | 0.011（0.0%） |
| aime24_format.json / baseline | 27/27 | 1.000 | 14.585 | 14.585 | 0.000（0.0%） | 13.938（95.6%） | 0.595（4.1%） | 0.047（0.3%） | 0.000（0.0%） | 0.004（0.0%） |

[分阶段 CSV](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/phase_timings.csv)。原始逐条记录和 summary.json 均包含 `metrics.phase_timings`，其中 requests 表示真实记录覆盖数。

运行目录：`/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300`。

- [实时 JSON](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/comparison.json)；[CSV](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/comparison.csv)；[状态](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/status.json)
- [运行配置](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/job.json)；[权重逐字节校验](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/weight_verification.json)；[量化记录](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/conversion_manifest.json)
- [顺序调度脚本](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/run.py)；[保留 BF16 draft 输出头的 worker](/Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/worker.py)

完整复现/续跑：在仓库根目录执行 `.venv/bin/python -u /Users/xmhuang6/workspace/spark_mlx_llm_my/.results/dspark-quantized-paired-20261028_233300/run.py`。已完成阶段跳过；中断阶段从已保存记录续跑，续跑随机状态会重新初始化。

2026-10-29优化补充：后续独立实验已完成AIME24全30条正式配对，最终committed decode吞吐为baseline68.489→优化后132.607 tok/s，**1.936172倍**，达到至少1.9倍的点估计目标；输入cluster bootstrap 95%区间[1.854029,2.022598]跨1.9，不能声称95%下界达标。最终使用256MiB allocator cache，33次整对尝试接受30对，3次环境拒绝后重试通过；接受请求task swapin及host swapout均为0，仍有有界host swapin、解压及内存压力边界。此前128MiB完整结果1.846809倍和512MiB失败尝试均独立保留。使用相同checkpoint及target采样配置，但候选启用运行时派生INT8 draft线性投影、draft temperature=.8及新的算子/流水路径，与本页原始全BF16 draft、无固定seed协议不同；不替换上述历史数据，也未评估答案准确率。

具体实现、失败记录、完整逐题数据及方法边界见 [DSpark 推理优化总报告](../../experiments/dspark-optimization/REPORT.md)。
