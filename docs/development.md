# Development and validation

[Project overview](../README.md) · [Extension development](../extensions/spark-mlx-ext/docs/development.md)

## Supported model

The implementation supports `Spark2_5ForCausalLM` checkpoints with
`model_type: spark2_5`. The reference 1.7B checkpoint uses:

- 28 decoder layers
- 8 query heads and 2 key/value heads
- Fused QKV projection
- Head-wise sigmoid attention output gates
- Three sliding-window layers followed by one full-attention layer
- A 512-token sliding window
- 256 rotary dimensions with theta 10,000 in sliding layers
- 64 rotary dimensions with theta 1,000,000 in full-attention layers
- Parallel GELU feed-forward layers
- Tied input and output embeddings

The checkpoint directory must contain at least:

```text
config.json
model*.safetensors
tokenizer.json
tokenizer_config.json
```

If the tokenizer is stored in a nested directory, copy the tokenizer files to
the checkpoint root before loading the model.

## Test entry points

These checks require local test sources; ignored test directories may be absent
from a fresh clone. Run commands from the repository root after installing the
project with the backend described in the environment guide. Install pytest and
Ruff separately when doing development checks; `pyproject.toml` does not declare
a `test` extra.

```sh
uv pip install --python .venv/bin/python pytest ruff
.venv/bin/python tests/check.py --cpu -q
.venv/bin/python tests/check.py --gpu --lock-timeout 120 -q
.venv/bin/python tests/check.py --gpu --suite all --lock-timeout 120 -q
.venv/bin/python tests/check.py --cpu --suite extension -q
```

The root suite defaults to CPU execution. Tests requiring real GPU kernels carry
`gpu` markers and are skipped in CPU mode. `cpu` markers keep routing tests on
CPU even during a GPU run. The shared fixture restores the default device after
every test, including failures. Root and extension suites run in separate Python
processes; this avoids module-name collisions and keeps their configurations independent.

The root pytest hooks acquire `spark_mlx_llm/benchmark/runtime.py` before loading MLX
in GPU mode. The extension runner and benchmarks use the same lock. A process
snapshot identifies competing work; the lock alone does not establish idle GPU
conditions. Full checkpoint checks additionally require `SPARK25_MODEL`.

```sh
.venv/bin/python -m pytest tests/test_model.py --device cpu -q
.venv/bin/python -m pytest tests --device gpu --lock-timeout 120 -m gpu -q
SPARK25_MODEL=/path/to/spark2_5 SPARK25_DTYPE=bfloat16 \
  .venv/bin/python tests/check.py --cpu -m integration -q
CUDA_VISIBLE_DEVICES=0 SPARK25_MODEL=/path/to/spark2_5 SPARK25_DTYPE=bfloat16 \
  .venv/bin/python tests/check.py --gpu -q
```

`tests/` is the root suite; the independently developed DSpark subtree retains
its existing separate entry points and is outside these commands.

## Static checks

```sh
.venv/bin/ruff check . --extend-exclude spark_mlx_llm/dspark
.venv/bin/ruff format --check . --extend-exclude spark_mlx_llm/dspark
```

Use the root Ruff configuration for Python 3.10 code; the separately packaged
extension retains its Python 3.11 configuration. Historical experiment snapshots
and generated `.build/`/`.results/` records are excluded by the existing ignore rules.

## Local generation launcher

```sh
MODEL_PATH=/path/to/model ./run.sh --prompt '你好' --max-tokens 128
./run.sh --model /path/to/model --device gpu --dtype bfloat16 --temp 0 --seed 1
```

`MODEL_PATH` supplies the model default; `--model` overrides it. `SPARK_PYTHON`
selects the interpreter. Other options pass directly to the generation CLI.
The wrapper also works outside the repository directory after installation.

## CUDA implementation note

The checkpoint stores Q, K, and V in one fused projection. After the projection,
the implementation splits, reshapes, and transposes the result into attention
heads. These tensors are non-contiguous views into the fused projection.

Passing those views directly to the CUDA RoPE path caused the K view to be
evaluated with incorrect storage aliasing. The error first appeared at layer 0
K-RoPE and later propagated through attention and the remaining layers. In the
full lazy graph it could produce all-NaN logits or an illegal CUDA memory access.

The model materializes Q, K, and V before RoPE and attention:

```python
queries = mx.contiguous(queries)
keys = mx.contiguous(keys)
values = mx.contiguous(values)
```

This is a compatibility workaround for the affected non-contiguous CUDA path;
it does not change the Spark2_5 computation.

## Evidence and historical CUDA validation

Local experiment records and build results are not included in a fresh Git clone.
Documents label these as local artifact paths instead of presenting them as
repository links. They remain unchanged on the development machine.

The integration test performs strict weight loading, validates complex
function-call parsing, and runs a real checkpoint forward pass. Always run an
end-to-end generation command as well, because a successful shape test alone
does not establish generation correctness.

The CUDA fix was validated with MLX 0.32.2, MLX LM 0.31.3, BF16 weights, and an
NVIDIA H100 PCIe GPU:

- CPU and CUDA prefill logits were finite
- CPU and CUDA selected the same first token
- Full-logit relative RMSE was approximately 0.528%
- The CUDA test suite reported `6 passed`
- Three consecutive deterministic generations produced the expected answer
- No NaN, illegal memory access, or non-deterministic text was observed

Other CUDA architectures should be validated with the same integration and
end-to-end generation commands before deployment.

## Project layout

```text
spark_mlx_llm/model.py   ModelArgs, module tree, forward flow and KV cache lifecycle
spark_mlx_llm/ops/       Configuration, dispatch and implementations grouped by operator
  base.py               OperatorConfig: config extraction, choices and validation
  linear.py             LinearConfig, dispatch_linear, affine/W4/W8 kernels and loading adapters
  qkv.py                QKV/RoPE dispatch and module/compiled/fused implementations
  sdpa.py               SDPAConfig, dispatch_sdpa and native/INT8/BF16 implementations
  attention_output.py   Gated-output dispatch and module/compiled implementations
  ffn.py                Residual/RMSNorm/FFN dispatch and module/compiled implementations
spark_mlx_llm/loader.py  Local and Hugging Face checkpoint loading
spark_mlx_llm/generation.py Unified GenerationConfig and generation APIs
spark_mlx_llm/commands/  Independent command-line entry points
  generate.py           spark-mlx-generate: prompt, model loading and text generation
  chat.py               spark-mlx-chat: model registration and terminal chat loop
  server.py             spark-mlx-server: shared settings and MLX LM HTTP service
  convert.py            spark-mlx-convert: conversion arguments and report output
spark_mlx_llm/conversion.py BF16/INT4 conversion Python API
extensions/spark-mlx-ext/ Optional C++ extension package
tests/                   CPU/GPU unit and real-checkpoint tests
tests/check.py           Fresh-process correctness runner
spark_mlx_llm/benchmark/ Shared generation, profiling and device coordination
benchmark/kv_compare/   Paired KV comparison
example/                Reusable inference configuration
docs/                   Installation, conversion, operators and development
  en/                   English documentation
  zh/                   Chinese documentation
```

### Operator boundaries

Model structure remains in `model.py`; configuration, dispatch and implementations
live in `ops/`. See [operator boundaries](operators.md#operator-boundaries).

### Operator configuration

See [configuration APIs](operators.md#operator-configuration) and
[defaults and migration notes](operator_configuration.md).
