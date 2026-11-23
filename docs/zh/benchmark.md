# 统一吞吐测试

测试使用与 Python、生成 CLI 相同的 `generate()` 和 `GenerationConfig`。
默认值内置在代码中，无需配置文件；HTTP 保留原服务默认值。
完整配置、指标及协议要求见[统一推理接口](../inference.md)。

```sh
.venv/bin/spark-mlx benchmark --model TARGET --draft-model DRAFT \
    --input-file gen.txt --trials 5 --warmup-tokens 128 > result.json
```

普通模型省略草稿参数。自然 EOS 测试使用 `--max-tokens none`，固定长度诊断使用
`--max-tokens N --no-stop-at-eos`；这两种协议分开比较。

预热只缩短输出上限，保留采样、seed、缓存及异步配置。正式重复均完整保留，
不丢弃慢样本。输出包含每次 token、终止原因、时间、prompt 哈希和实际参数。
汇总吞吐为 `sum(输出 token 数 - 1) / sum(generation_seconds)`，
首 token 属于 prefill，EOS 计入输出。另输出 min/max/median 及重复输出是否完全相同。

CPU 一致性测试验证入口行为；正式 GPU 性能基线另按
[实验规范](../../experiments/README.md)冻结当前源码和二进制、检查环境并保留所有记录。
