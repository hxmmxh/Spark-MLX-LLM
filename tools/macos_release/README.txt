Spark-MLX-LLM macOS ARM64 release / 发布包使用说明

发布包自带 Python、MLX、MLX LM 和项目命令，解压即可使用。
系统要求：Apple Silicon，macOS 14 或更新版本。
内存要求：大于 8 GB 统一内存，推荐运行 4B 量化模型。
模型权重单独准备。移动时保留整个目录，不要只复制 bin。

0.1.1 已知限制：DSpark HTTP 流式文本可能为空；接口暂不接受分段文本列表和工具定义。
完整消息与工具定义可通过包内 Python 接口应用 tokenizer 聊天模板后生成。

检查运行环境：

  ./bin/spark-mlx-doctor
  ./bin/spark-mlx-doctor --gpu

首行显示“能跑 / 不能跑”及原因；--gpu 实测 GPU 运算，--json 输出 JSON。

准备原始 compressed-tensors INT4 目录或受支持的 Spark 主模型 GGUF，
以及配套的草稿模型。以下命令使用 DSpark，浮点草稿默认按 INT8 加载。
生成、聊天和 HTTP 服务默认每轮最多生成 3 个草稿 token；预测槽位不足时使用实际可用槽位数。
生成与服务命令可用 --max-draft-tokens N 覆盖，设为 0 禁用草稿生成。

一次生成：

  ./bin/spark-mlx-dspark generate --model /path/to/Spark-X2.6-4B-INT4 --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf --prompt '你好，请介绍自己。' --max-new-tokens 128

交互聊天：

  ./bin/spark-mlx-chat --model /path/to/Spark-X2.6-4B-INT4 --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf --max-tokens 512

输入 q 退出、r 重置会话、h 查看帮助。默认使用 low 思考模式。

HTTP 聊天服务：

  ./bin/spark-mlx-dspark-server --model /path/to/Spark-X2.6-4B-INT4 --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf --host 127.0.0.1 --port 8080 --max-tokens 512

另一个终端中调用：

  curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dspark","messages":[{"role":"user","content":"你好"}],"max_tokens":128}'

合并的 DSpark 目录可直接传给 --model，无需 --draft-model。
各命令可使用 --help 查看参数。

完整使用文档 / Full usage guides：

  docs/zh/release-package.md
  docs/en/release-package.md

如需从任意工作目录调用命令，可在当前终端执行：

  export PATH="$PWD/bin:$PATH"

BUILD-INFO.json 记录版本与验证信息，FILES.sha256 提供文件校验值。
运行环境不读取系统 Python 的 site-packages 或 PYTHONPATH。
项目许可证见 LICENSE，第三方许可证保留在 runtime 中各包的 dist-info/licenses
以及 Python 的 LICENSE.txt 中。
