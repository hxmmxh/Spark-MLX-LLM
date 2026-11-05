# tuned_bf16_sdpa：4B 形状复测与 1.7B 适配决定

日期：2026-10-26（Asia/Shanghai）。状态：本机开发复测完成，未采用 1.7B 适配。

当前公共 `spark_mlx_ext.bf16_sdpa` 的 4B 形状路径在这次复测中比原生 MLX
更慢，没有复现历史冻结实验的收益。因此按“4B 有收益才继续优化 1.7B”的条件，
停止本次 1.7B 分块筛选和模型测速，撤回待验证的 Hq8/Hkv2 路由与算子扩展。
`tuned_bf16_sdpa` 默认仍为 `False`，1.7B 继续使用原生 BF16 attention。

## 测量条件

- Apple M5 Pro，24 GiB；MLX 0.32.4、MLX-LM 0.31.3、spark-mlx-ext 0.2.0。
- BF16 Q `[1,16,1,256]`、K/V `[1,4,N,256]`，scale=1/16，无 mask。
  这是 4B 的 attention 几何形状；本次输入为随机种子 42 的合成数组，不是整模型生成。
- 基线为 `mx.fast.scaled_dot_product_attention`，候选为实际安装包的公开
  `bf16_sdpa`，采用默认 split 策略。双方使用相同的三组已物化输入，均浅复制 Q
  描述符，避免原生接口对共享描述符的潜在修改。
- 每档六轮，轮换执行顺序；每个 block 先预热 12 次，再测 60 次新建结果并求值的调用。
  一个 block 内的调用不作为独立重复。数据准备与首次编译不计时。
- 导入 MLX 前取得共享设备锁；每轮开始前连续两次 CPU<15%、GPU<10%，内存和温度
  状态正常。任一实现发生换页或检测到竞争进程则拒绝整组，最多尝试三次。
  两次完整测量各有 18 组有效对照，本次均没有拒收组。

第一次记录 Metal command-buffer 时间与同步调用时间；第二次在新进程中关闭
GPU 计时钩子，独立复核同步调用时间。两次结果分别统计，没有混合样本。
GPU 指标是 command-buffer 时间区间和，不是单个 kernel 的时间。

## 结果

下面均为 block 均值的中位数，单位毫秒/attention 调用。GPU 列来自第一次测量；
wall 列来自关闭计时钩子的第二次测量。“wall 增加”计算为
`tuned wall 中位数 / native wall 中位数 - 1`。

| KV 长度 | 原生 GPU | tuned GPU | 原生 wall | tuned wall | wall 增加 |
|---|---:|---:|---:|---:|---:|
| 32K | 0.506 | 0.828 | 0.723 | 1.034 | +43.1% |
| 64K | 1.008 | 1.978 | 1.203 | 2.191 | +82.1% |
| 128K | 2.235 | 4.279 | 2.435 | 4.484 | +84.2% |

无钩子复核中，每档六轮 tuned 均更慢。逐轮耗时比的中位数分别为
1.4530、1.8078、1.8411；它们与上表的“中位数之比”是不同统计量。
候选与原生输出的最大 relative L2 约 0.322%，通过本次 1% 数值检查门槛；
这不构成生成质量评估。

## 与历史结果的关系

历史报告 (local artifact: `experiments/sdpa-bf16-optimization/REPORT.md`)中的冻结实验路径在
4B 真实 Q/K/V 上，64K、128K 的同步调用耗时分别下降约 3.42%、12.08%。
该报告同时明确：包接入支持实际 strides 和 64 位偏移，历史数字不是包接入后的新测速结果。
本次不使用旧实验的收益替代当前公共接口的结果，也不据此否定旧实验在其自身条件下的记录。

## 分析流程与边界

1. 两次新测量使用相同算子源码哈希；第二次在整个运行期间未检测到相关源码变化。
   GPU 和无钩子 wall 两条证据方向一致，因此计时钩子不能单独解释此次减速。
2. 每档六轮均慢，且全部通过本次环境检查，因此结论不依赖挑选最快值或删除慢样本。
3. 本轮 1.7B 原型没有改变 4B 的 Metal 两遍源码、split 数与 launch 几何。
   测量对应原型尚存在时的公共包装器之 4B 分支；随后仅撤回 1.7B 扩展与调参参数。
   不能把大幅 GPU 减速归因于未执行的 1.7B 分支。
4. 历史捕获数据、冻结内核与当前支持更多布局的包装器之间存在差异，具体减速原因尚未
   通过消融定位。本次按用户条件停止，不继续扩大为 4B 内核优化项目。
5. 本次没有完整 4B/1.7B 模型的 decode tokens/s 对照，不将算子耗时变化换算为整模型收益。
   未执行的 1.7B 分块筛选和模型测试脚本仅作为未采用原型保留。

撤回后另行运行算子正确性和模型分发测试，覆盖 1.7B 原生回退；结果见
验证记录 (local artifact: `.build/tuned-bf16-17b/final-validation.xml`)。

## 证据与复现

- GPU 及带钩子 wall 原始数据 (local artifact: `.build/tuned-bf16-17b/4b-current.json`)
- 无钩子 wall 原始数据、环境记录及源码哈希 (local artifact: `.build/tuned-bf16-17b/wall-confirm/result.json`)
- 独立 wall 复核脚本 (local artifact: `.build/tuned-bf16-17b/wall_confirm.py`)
- 未采用原型与决定 (local artifact: `.build/tuned-bf16-17b/unadopted-prototype/decision.json`)

当前公共 benchmark 的复现方式如下。正式运行前使用尚不存在的输出文件名：

```sh
.venv/bin/python extensions/spark-mlx-ext/tests/bench.py bf16_sdpa \
  --lock-timeout 1800 --idle-timeout 600 \
  --lengths 32768 65536 131072 --input-sets 3 \
  --warmup 12 --iterations 60 --repeats 6 --timing wall \
  --json .build/tuned-bf16-17b/repeat-wall.json
```

这些是带环境检查的本地开发测量，未作为正式实验管理器的归档运行。
