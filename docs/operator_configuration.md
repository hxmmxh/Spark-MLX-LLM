# 算子结构、配置与默认生成行为

更新日期：2026-11-11。

算子的配置、分发和实现现在放在同一个模块中。模型负责网络结构与调用顺序；
各模块用 `__all__` 声明公开接口，私有实现留在模块内部。

| 模块 | 执行入口 | 配置归属 |
|---|---|---|
| `ops/linear.py` | `dispatch_linear` | `LinearConfig`，另负责 W8 权重适配 |
| `ops/qkv.py` | `dispatch_qkv` | 复用 `LinearConfig` |
| `ops/attention_output.py` | `dispatch_attention_output` | 根据投影类型自动选择 |
| `ops/ffn.py` | `dispatch_ffn` | 复用 `LinearConfig` |
| `ops/sdpa.py` | `dispatch_sdpa` | `SDPAConfig` |

`LinearConfig` 和 `SDPAConfig` 继承 `ops/base.py` 的 `OperatorConfig`，共享
字段提取、可选值与校验。`ModelArgs` 继承这两个配置类，checkpoint 配置仍保持平铺。
不同算子入口保留各自的函数签名，不需要通用 dispatcher 基类。

## 默认算子配置

| 参数 | 默认值 | 行为与适用范围 |
|---|---|---|
| W8 路由（自动） | 主模型 native；草稿按设备选择 | 草稿单行 W8A16，多行在 M5+ 使用 NAX，在兼容旧GPU使用 matrix；CPU及不支持形状保留回退 |
| `w4_qmv_backend` | `"native"` | W4 层保持原生 affine W4A16 |
| `fused_quantized_sdpa` | `True` | 对满足条件的 INT8/G32 KV、D256、单 query GPU attention 使用融合算子 |
| `fused_quantized_sdpa_prefill` | `False` | 多 query attention 保留原有实现；开启时还需 `fused_quantized_sdpa=True` |
| `tuned_bf16_sdpa` | `False` | 默认关闭调优 BF16 decode SDPA |

W8 已移除 Python `w8_qmv_backend`、CLI `--w8-qmv-backend` 及转换命令的
`--w8-backend` 参数。主模型保持已验证的 native 算术；草稿为 INT8 权重、
BF16 激活，按实际设备能力选择单行内核和 NAX/matrix 多行内核。
旧 checkpoint 的路径偏好兼容读取并忽略；显式 model_config 手动覆盖会报错。
转换产物只记录量化存储格式，不再固定某台机器的 W8 路径，也不默认引入 A8 激活量化。
W4 后端仍可单独设置，不改变量化位宽。实际选择保存在结果加载元数据中。

`w4_qmv_backend="splitk"` 仅覆盖生产代码列出的 BF16/G32、单行 GPU 形状。
当前为 4B 的 QKV `[6144,2560]` 和 FFN down `[2560,10240]`，不覆盖 1.7B。
`tuned_bf16_sdpa=True` 限于全局层、B1/Hq16/Hkv4/D256、单 query、2K–128K
BF16 KV；1.7B 的 Hq8/Hkv2 不满足条件。

2026-10-26 对当前公共 BF16 算子的 4B 形状复测没有复现历史冻结实验收益；
因此保持关闭，并未采用本轮 1.7B 扩展。详见
[4B BF16 SDPA 复测记录](bf16_sdpa_4b_recheck.md)。

`attention_output.py` 无独立开关：W8A16 adapter 的合适单行输入选择 compiled
W8A16 输出路径，其他合适 affine 投影选择 compiled affine 路径。
后者由投影的 `bits` 决定精度，不等同于固定 W4A16。QKV 和 FFN 的 compiled
路径也按模块类型、形状和 dtype 自动选择。

## 默认生成配置

这些参数属于 `config.py` 的 `GenerationConfig`，由
`spark-mlx generate`、`spark_mlx_llm.generate(..., stream=True/False)` 和
benchmark/profiler 共用。

`spark-mlx generate` 的 `--config` JSON 文件也可以设置这些 KV/prefill 字段，
显式命令行参数优先于文件配置。Python 调用通过 `config.kv_kwargs()` 导出这些字段，
与采样和模板参数分别传递。

| 参数 | 默认值 | 行为 |
|---|---|---|
| `prefill_step_size` | `512` | prompt 按 512 token 分块 |
| `kv_bits` | `None` | 默认不量化，保留模型精度（BF16 模型使用 BF16 KV）；显式设为 `4/8` 可启用量化 |
| `kv_group_size` | `32` | 启用量化时，每 32 个元素共享量化 metadata |
| `quantized_kv_start` | `0` | 启用量化时，在最早的上游 chunk/step 检查点转换 KV；不是在第一次模型计算前量化 |
| `kv_quantization_scope` | `"full-attention"` | 启用量化时仅转换全局层；滑窗 cache 保持模型精度和旋转状态 |
| `dtype`（加载参数） | 不覆盖 | 保持 checkpoint 精度；此 1.7B checkpoint 为 BF16 |

此 1.7B 模型有 7 层全局 attention 和 21 层滑窗 attention。
打开融合 SDPA 本身不会创建 INT8 KV；需显式设置 `kv_bits=8` 才会启用 INT8 KV 路径。
每次未指定 cache 的生成创建新的 cache；传入 `prompt_cache` 则原地继续使用。

所有生成入口默认均不量化 KV。直接调用 `mlx_lm.generate`、MLX-LM chat/server
委托入口仍使用上游生成参数；分块大小等设置可以不同。
`load()` 仅控制模型加载，不设置上游生成函数的默认 KV 或分块大小。

## 使用与覆盖

默认命令无需逐项指定优化参数：

```sh
.venv/bin/spark-mlx generate \
  --model /path/to/Spark-X2.6-1.7B-INT4-MLX \
  --prompt '你好' --max-tokens 256
```

Python 使用本包的生成入口：

```python
from spark_mlx_llm import generate, load

model, tokenizer = load("/path/to/Spark-X2.6-1.7B-INT4-MLX")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=False,
    add_generation_prompt=True,
)
text = generate(model, tokenizer, prompt, max_tokens=256)
```

算子默认值可由 checkpoint 同名字段或显式 `model_config` 覆盖；
`load(..., w4_qmv_backend=...)` 的显式参数优先于
`model_config`。CLI 未传算子选项时保留 checkpoint 的显式设置。
生成参数直接传给生成函数或 CLI：

```sh
# 原生 W8 + 关闭融合 decode + 浮点 KV
.venv/bin/spark-mlx generate --model /path/to/checkpoint \
  --no-fused-quantized-sdpa --kv-bits none
```

CLI 支持三个 SDPA 参数的正反开关，例如 `--fused-quantized-sdpa-prefill` /
`--no-fused-quantized-sdpa-prefill`；其他生成参数使用连字符形式。
在 Python 中用 `generate(..., kv_bits=8)` 显式开启 INT8 KV 量化，CLI 对应 `--kv-bits 8`。
省略该参数或使用 Python `None` / CLI `none` 均保留模型 KV 精度。

## 本次迁移与实验依据

- W8 计算路径改为内部自动选择，旧 `qmv_backend` / `w8_qmv_backend`
  不再作为用户加载选项。历史实验快照保持原字节，不重写为当前接口结果。
- 删除 `qmv_source`。优化 W8 参数统一来自当前加载的 affine checkpoint：
  INT8 codes 精确解包，scale 转为 FP32，保留原有精度。
- 主模型 W8 保持 native；草稿多行按设备选择 NAX/matrix。
- 没有 W8 out_proj/head 的浮点或纯 W4 checkpoint 保持原层结构，不做 W8 适配。
  有 W8 投影时要求相应扩展 API 和兼容的对称 per-channel metadata；
  不支持的格式或设备保留原生回退。

推荐来自本机既有 1.7B 实验，偏重单请求 decode。历史 4K W8A16 约
176.1 token/s，新 W8A8 head 在另一会话约 173.2 token/s；尚无新同场配对结论。
当前融合 INT8 prefill 明显慢于原始实现，因此默认关闭。
512 分块沿用测量配置，不代表所有任务下的最优值。本次修改不新增性能结论。

参见 W8 head 实验 (local artifact: `experiments/spark17b-w8-head-optimization/REPORT.md`)
与 INT8 prefill 实验 (local artifact: `experiments/sdpa-int8-prefill-optimization/REPORT.md`)。
