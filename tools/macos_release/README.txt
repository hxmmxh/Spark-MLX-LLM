Spark-MLX-LLM macOS ARM64 release / 发布包使用说明

发布包自带 Python、MLX、MLX LM；模型权重单独准备。系统要求以 BUILD-INFO.json 为准。
保持整个目录一起移动。

唯一命令入口：
  ./bin/spark-mlx doctor --json
  ./bin/spark-mlx doctor --gpu
  ./bin/spark-mlx generate --model TARGET --draft-model DRAFT --prompt '你好' --json
  ./bin/spark-mlx chat --model TARGET --draft-model DRAFT --stream
  ./bin/spark-mlx benchmark --model TARGET --draft-model DRAFT --input-file gen.txt --trials 5

普通解码省略草稿选项。默认值内置在代码中，不需要配置文件；--config FILE 仅用于显式覆盖。
Python/生成 CLI 默认：主模型温度1/top-p0.95，草稿温度1/top-p1，seed0，输出上限512，草稿长度3，
low 思考，两项异步均开启。结果包含实际参数及 prompt 哈希。
聊天用 /quit 退出、/reset 清空历史。

建议的 HTTP 服务启动命令（先替换模型路径）：
  ./bin/spark-mlx server \
      --model /path/to/Spark-X2.6-4B-INT4 \
      --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
      --served-model-name spark-x2.6-4b \
      --host 127.0.0.1 --port 8080 \
      --enable-thinking --reasoning-effort low \
      --draft-quantization int8 \
      --temperature 1 --top-p 0.95 --top-k 0 --min-p 0 \
      --draft-temperature 1 --draft-top-p 1 --draft-top-k 0 \
      --max-draft-tokens 3 \
      --asynchronous-draft --async-context --compile-draft-chain \
      --seed 0 --prefill-step-size 512 \
      --max-context-length 262144 --max-tokens 262144

在另一个终端测试 HTTP 请求：
  curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"spark-x2.6-4b","messages":[{"role":"user","content":"你好"}],"stream":false}'
流式测试时给 curl 添加 -N，并将 stream 改为 true。

Python 使用 bin/spark-mlx-python，同一个 load()/generate()，stream=True 开启流式。

完整说明：docs/inference.md、docs/zh/release-package.md、docs/en/release-package.md
BUILD-INFO.json 记录构建信息，FILES.sha256 提供文件校验值。
包内 Python 不读取系统包或 PYTHONPATH；许可证见 LICENSE 及第三方 dist-info/licenses。
