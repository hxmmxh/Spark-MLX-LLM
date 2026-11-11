#!/usr/bin/env bash
# 使用代码内置默认值；可追加 --draft-model PATH 或 --config FILE。
set -euo pipefail
SPARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_PYTHON="${SPARK_PYTHON:-$SPARK_ROOT/.venv/bin/python}"
SPARK_MODEL="${SPARK_MODEL:-/path/to/Spark-X2.6-4B-INT4}"
exec "$SPARK_PYTHON" -m spark_mlx_llm chat \
    --model "$SPARK_MODEL" \
    --system-prompt '请使用中文简洁回答。' \
    --stream \
    "$@"
