# Spark-X3 ViT：计算流程、精简实现与验证

## 仓库入口

| 目录或文件 | 职责 |
| --- | --- |
| `spark_mlx_llm/spark3.py` | Spark-X3 语言模型和多模态模型入口 |
| `spark_mlx_llm/processing_spark3.py` | 图像缩放、归一化和 patch 排列 |
| `spark_mlx_llm/spark3_vision.py` | ViT 模型、权重布局、原始计算和融合分派 |
| `spark_mlx_llm/vit_fusion/` | 最终保留的视觉融合实现 |
| `spark_mlx_llm/ops/` | 语言模型算子和分派 |
| `extensions/spark-mlx-ext/` | 语言模型的可选原生扩展 |
| `extensions/gpu-clock/` | 独立的 GPU 测量工具 |
| `tests/test_vit_fusion.py` | 固定输入的融合正确性与回退测试 |
| `tests/test_spark3_multimodal.py` | 图像处理、原始 ViT、mRoPE 和多模态回归 |
| [vit-fusion-results.json](vit-fusion-results.json) | 精简版的耗时、数值核对和执行身份 |

旧的 `ops/vision_*.py`、ViT 原型扩展、配套测试和 `docs/x3_vit_*.md` 已移出运行仓库。最终实现集中保存在一个包中：

```text
spark_mlx_llm/vit_fusion/
├── __init__.py                 支持条件与完整 Block 调用
├── qkv_rope.py                 QKV 投影、bias、RoPE 和布局转换
├── self_attention.py           短序列 SDPA 与长序列 NAX Attention
├── attn_output_proj_norm.py    Attention 输出投影、bias、残差和 LayerNorm
├── mlp_residual.py             FC1 + GELU、FC2 + 残差
└── _nax_gemm.py                投影与 MLP 共用的 GEMM 内核和调用
```

每段计算使用一个实现文件。Attention 输出投影与 MLP 共用 `_nax_gemm.py` 的 GEMM 内核和调用函数，QKV 也从中导入 NAX 头文件加载函数。

算子版本与源码哈希记录在工作区的 `op_opt/fusion_research/docs/vit-fusion-origin.json`。内联 Metal 源码的 MLX 许可证位于仓库根目录，随安装包分发。 [MLX 许可证](../LICENSE-MLX)

## ViT 的原始计算

真实 Spark-X3 ViT 包含 27 个 Block，隐藏宽度为 1152，Attention 包含 16 个 D72 head，MLP 中间宽度为 4304。令 `N` 表示 patch 数。

图像处理器把图像排列成 RGB 16×16 patch。Patch Embedding 对时间轴求平均，把每个 patch 投影为 1152 维特征，再加入按网格插值的绝对位置编码。

每个 Block 依次计算：

```text
U = LayerNorm1(X)
Q, K, V = split(U @ Wqkv.T + bqkv)
Q, K = RoPE(Q, K)
C = concat_heads(softmax(Q @ K.T / sqrt(72)) @ V)
H = X + (C @ Wo.T + bo)
Z = LayerNorm2(H)
F = GELU_tanh(Z @ W1.T + b1)
Y = H + (F @ W2.T + b2)
```

`X/H/Z/Y` 的形状均为 `[N,1152]`，`F` 的形状为 `[N,4304]`。RoPE 编码 patch 的行列位置；Attention 按图像分别计算。

27 层完成后，Spatial Merger 对特征归一化，把相邻 2×2 patch 拼成 4608 维向量，再经过两层投影，生成送入语言模型的视觉 token。输出宽度由 checkpoint 配置决定，当前验证模型为 2560。

## 最终融合范围

| 阶段 | 实现 | 函数 | 计算范围 |
| --- | --- | --- | --- |
| QKV/RoPE | [qkv_rope.py](../spark_mlx_llm/vit_fusion/qkv_rope.py) | `project_qkv_with_rope` | 融合 QKV GEMM、bias、Q/K RoPE 和按 head 排列 |
| Self-Attention | [self_attention.py](../spark_mlx_llm/vit_fusion/self_attention.py) | `compute_self_attention` | 短序列调用原生 SDPA，长序列采用紧凑 NAX 路径 |
| Attention 输出投影与归一化 | [attn_output_proj_norm.py](../spark_mlx_llm/vit_fusion/attn_output_proj_norm.py) | `project_attn_output_and_normalize` | 一个 Metal kernel 生成 `H = X + output_proj(C)`，随后原生 LayerNorm 生成 `Z` |
| MLP 与残差 | [mlp_residual.py](../spark_mlx_llm/vit_fusion/mlp_residual.py) | `apply_mlp_with_residual` | 一个 kernel 生成 `F = GELU(FC1(Z))`，另一个生成 `Y = H + FC2(F)` |

投影阶段输出的 `Z` 是 FC1 的输入，`H` 是 MLP 最后一次相加的残差。FC2 保留原始实现的 split-K 累加边界。MLX 首次调用时编译对应的 Metal kernel，各段通过数组依赖保持执行顺序和惰性求值。

模型只保留一个后端选项：`block_backend="native"` 执行原始计算，`block_backend="segmented"` 调用这套分段实现。默认值为 `native`。

## 使用和测试

在仓库根目录安装：

```bash
python -m pip install -e '.[vision,vit-fusion,test]'
```

直接构造视觉模型时，将后端写入配置：

```python
from spark_mlx_llm.spark3_vision import VisionArgs, VisionTransformer

vision = VisionTransformer(VisionArgs(block_backend="segmented"))
# 按 checkpoint 配置构造模型并载入权重后，沿用原来的调用接口。
output = vision(pixel_values, image_grid_thw)
```

已经加载的 Spark-X3 多模态模型可以直接切换：

```python
for block in model.visual.blocks:
    block.block_backend = "segmented"
```

各段函数只接收 `inputs`。QKV 返回 `q/k/v`，Attention 和 MLP 返回 `y`，输出投影与归一化返回 `h/z`。完整 Block 可调用 `forward_block`。

```python
from spark_mlx_llm.vit_fusion import forward_block

output = forward_block(block, hidden, cos, sin, [len(hidden)])
```

固定输入回归只保留一个融合测试文件：

```bash
python -m pytest tests/test_vit_fusion.py -q
```

测试固定随机种子、权重和输入尺寸，覆盖 196、1040、2304、3844 patch，核对原始路径与融合路径的 BF16 位级一致性，同时检查不支持的输入和多图回退。原始 ViT 的独立 NumPy 参考测试为：

```bash
python -m pytest tests/test_spark3_multimodal.py -k vit_numpy_reference -q
```

## 已验证的效率与误差

2026-11-25，Apple M5 Pro、MLX 0.32.4。测试使用真实 BF16 视觉权重和两张固定图片，比较删除投影 C++ 桥接前后的完整 27 层视觉塔。两侧均采用精简后的 MLP。计时包含 Python 调用、最终输出求值和同步。

| 处理后尺寸 | Patch 数 | 精简前 ms | 精简后 ms | 中位耗时变化 |
| --- | ---: | ---: | ---: | ---: |
| 224×224 | 196 | 14.23 | 13.97 | −1.82% |
| 384×384 | 576 | 27.42 | 27.35 | −0.26% |
| 512×512 | 1024 | 47.26 | 47.11 | −0.31% |
| 640×416 | 1040 | 50.24 | 50.07 | −0.34% |
| 768×768 | 2304 | 119.47 | 119.22 | −0.21% |
| 896×896 | 3136 | 176.02 | 175.36 | −0.38% |
| 992×992 | 3844 | 220.24 | 221.19 | +0.43% |

两版耗时整体接近。投影 `H`、归一化 `Z` 和完整视觉塔最终输出共 70 项核对全部位级一致，相对 L2 与最大绝对误差均为 0。测试通过空闲准入，420 对采样全部通过活动检查。

[结果 JSON](vit-fusion-results.json)保存投影段和完整塔的全部汇总、逐项数值核对、协议及测量时的源码路径与哈希。工作区的 `op_opt/fusion_research/results/projection-python-20261125-idle/` 保存原始采样与冻结源码。`repository-cleanup-20261125/` 保存移出的旧原型归档和清理清单。

设备、输入范围和测量说明见 [支持与测量说明](defensive-notes.md)。
