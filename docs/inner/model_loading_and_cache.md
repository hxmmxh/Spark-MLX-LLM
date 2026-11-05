# 主模型自动准备与缓存

[中文首页](../../README.zh.md) · [English README](../../README.md) · [文档目录](../README.md)

本文记录模型加载与执行后端的内部实现和配置规则。

## 加载规则

普通生成、聊天、HTTP 服务与 DSpark 入口共用以下本地模型加载规则：

| `--model` 输入 | 加载行为 |
| --- | --- |
| 原始 BF16 Safetensors 目录 | 直接加载，保留 BF16 精度 |
| 官方 INT4 目录（默认） | 在内存中重排已有整数权重，不写转换缓存；每次启动重新准备 |
| 官方 INT4 目录，显式启用 `--model-cache` | 首次自动适配为 MLX 并保存，后续复用；首次需要转换时间和缓存磁盘空间 |
| 已转换的 MLX Safetensors 目录 | 直接加载，跳过主模型格式转换 |
| 主模型 GGUF 文件 | 自动转换并缓存；草稿 GGUF 可直接加载 |

启用缓存后，缓存默认位于 `~/.cache/spark-mlx-llm/models`，每个模型的子目录由程序自动创建。
源文件或转换实现变化后会生成新缓存。缓存根目录的配置优先级为：

**CLI `--model-cache-dir` / Python `model_cache_dir` ＞ 环境变量 `SPARK_MLX_MODEL_CACHE` ＞ 默认目录。**

Python 的 `load()` 和 `load_pair()` 使用相同规则；对 INT4 目录默认关闭缓存，
传入 `model_cache=True` 可启用。关闭缓存时忽略缓存目录设置。
内存准备保留原始 INT4/INT8 整数编码，按转换规则处理 scale/zero point，
不生成转换后的模型或临时模型文件，也无需将整个模型展开为 BF16；临时数组仍会占用额外内存。

上述默认行为适用于普通 generate/chat/server、DSpark generate/benchmark/export-draft 及 DSpark HTTP 服务；
普通 chat/server 在内存准备模式下仅支持单进程加载。主模型 GGUF 仍需转换缓存；
`export-draft` 显式指定的输出文件照常保存。已转换的 MLX 目录仍直接加载。

## 主模型 GGUF 适配

主模型 GGUF（包括 Q4_K_M）按行分块解码为 BF16，再量化为本项目的 W4/G32＋W8 方案并缓存。
该过程会重新量化，不能视为保留原 GGUF 整数编码，也不保证与 llama.cpp 的结果逐位一致。
词表、BPE 规则与聊天模板从 GGUF 中导入，无需单独下载分词器。

## W8 执行后端

原始 INT4 主模型的自动格式适配使用 `native` W8 路径；显式运行转换器时，默认记录
`--w8-backend w8a8`，也可传入 `--w8-backend native` 选择浮点激活的 MLX affine 路径。
W8A8 不可用时回退到 MLX affine 算子。W8 权重编码保持不变，浮点与 INT8 激活路径的数值结果可能不同。
转换选项与精度细节见[模型转换文档](../zh/conversion.md)。
