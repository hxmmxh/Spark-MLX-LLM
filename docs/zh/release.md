# macOS 发布包构建与使用

[项目首页](../../README.zh.md) · [环境配置](environment.md)

下载预编译包后的操作见[发布包使用指南](release-package.md)（[English](../en/release-package.md)）。
本文介绍构建工具；默认构建目标与已经生成的具体发布包可能不同，实际系统要求以包内
`BUILD-INFO.json` 为准。当前已生成的 `spark-mlx-0.1.0-macos26-arm64` 要求 macOS 26.2+。

## 构建发布包

在 Apple Silicon Mac 的仓库根目录执行：

```sh
uv run --no-project --python 3.12 python tools/build_macos_release.py
```

构建机需要 uv；默认构建扩展还需要支持 Metal 4 和
MetalPerformancePrimitives 的完整 Xcode 工具链。下载依赖时需要网络。
工具使用独立构建目录，不修改项目的 `.venv`，也不修改 uv 管理的原始 Python。

默认构建内容：

- 自带独立 ARM64 CPython 3.12.16。
- 根据 `uv.lock` 安装依赖，验证下载哈希，并按 macOS 14.0 选择二进制 wheel。
- 冻结当前生产源码后构建主包，以及要求 macOS 26.2 的 `spark-mlx-ext`。
- 生成所有命令启动脚本，保留原有命令并增加 DSpark 和环境诊断命令。
- 检查 ARM64 架构、二进制最低系统版本、动态库路径、符号链接和 editable 路径。
- 将目录移动至含中文及空格的新路径，从仓库之外验证所有命令的 `--help`。
- 分别验证自动加载扩展和禁用扩展时的 GPU 计算。
- 输出压缩包、SHA256 和构建信息。默认不包含模型权重。

默认输出：

```text
dist/
  spark-mlx-0.1.0-macos-arm64.tar.gz
  spark-mlx-0.1.0-macos-arm64.tar.gz.sha256
  spark-mlx-0.1.0-macos-arm64.BUILD-INFO.json
```

主包版本来自 `pyproject.toml`。已有同名压缩包时不会覆盖；可用 `--name`
指定新名称，或用 `--force` 明确覆盖。构建中间文件保留在 `.build/releases/`。
源码取当前工作目录内容，包含尚未提交的生产代码；构建信息记录 Git 版本、dirty
状态、源码 SHA256、Python 构建编号、wheel 哈希、依赖版本和验证结果。

### 构建时验证真实模型

```sh
uv run --no-project --python 3.12 python tools/build_macos_release.py \
  --smoke-model /path/to/target-only-mlx-model
```

该参数用本地模型分别验证自动加速和原生回退的生成，不会将权重打进包中。
请使用可供普通生成命令加载的主模型目录；DSpark 合并目录应使用 DSpark 命令验证。
验证 GPU 时会使用项目共享的设备锁；有其他工作占用设备锁时构建会明确失败。

### 复用已有扩展或只构建基础包

```sh
# 复用开发环境中已经编译好的扩展，无需重新编译 Metal
uv run --no-project --python 3.12 python tools/build_macos_release.py \
  --extension-python .venv/bin/python

# 使用匹配 CPython 3.12 / ARM64 / MLX 0.32.4 的预编译扩展 wheel
uv run --no-project --python 3.12 python tools/build_macos_release.py \
  --extension-wheel /path/to/spark_mlx_ext-0.2.0-cp312-cp312-macosx_26_0_arm64.whl

# 不带扩展，构建纯原生 MLX 运行时
uv run --no-project --python 3.12 python tools/build_macos_release.py \
  --without-extension --name spark-mlx-0.1.0-native-macos-arm64
```

`--extension-python` 会读取指定环境中的扩展目录和分发元数据，制作普通 wheel，
不带 editable 安装文件；构建信息记录来源为 `installed-precompiled` 及 wheel 哈希。
使用前应确保该环境中的原生扩展已经按所需源码编译完成。

基于本机缓存构建 macOS 26.2+ 包的示例：

```sh
UV_OFFLINE=1 uv run --no-project --python 3.12 python tools/build_macos_release.py \
  --extension-python .venv/bin/python \
  --min-macos 26.2 --name spark-mlx-0.1.0-macos26-arm64
```

`--min-macos` 会影响依赖 wheel 的选择、二进制审计和包内最低版本提示。
实际分发的系统要求以该包的 `BUILD-INFO.json` 中 `minimum_macos` 为准。
依赖源访问有问题时可用 `--index-url` 指定可用的 PyPI 镜像，不修改仓库索引配置。

完整选项：

```sh
uv run --no-project --python 3.12 python tools/build_macos_release.py --help
```

只有明确需要时才使用 `--skip-gpu-check`；构建信息会记录未执行 GPU 验证。
当前工具只在 Apple Silicon 构建 ARM64 macOS 包。

## 用户解压后使用

将压缩包和对应 `.sha256` 放在同一个目录：

```sh
shasum -a 256 -c spark-mlx-0.1.0-macos-arm64.tar.gz.sha256
tar -xzf spark-mlx-0.1.0-macos-arm64.tar.gz
cd spark-mlx-0.1.0-macos-arm64
./bin/spark-mlx-doctor --gpu
```

用户无需安装 Python、uv、Homebrew 或 Xcode。整个目录可以移动，路径可以包含
空格和中文；移动时需要保留完整目录结构。

生成与聊天：

```sh
./bin/spark-mlx-generate \
  --model /path/to/model \
  --prompt '你好，请介绍自己。' --max-tokens 128 --no-thinking

./bin/spark-mlx-chat --model /path/to/model --no-thinking
```

启动 HTTP 服务：

```sh
./bin/spark-mlx-server \
  --model /path/to/model \
  --host 127.0.0.1 --port 8080 --no-thinking
```

DSpark 主模型和草稿分开加载：

```sh
./bin/spark-mlx-dspark generate \
  --model /path/to/model \
  --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
  --draft-quantization int8 \
  --prompt '请解释推测解码。' --max-new-tokens 128 --stats

./bin/spark-mlx-dspark-server \
  --model /path/to/model \
  --draft-model /path/to/Spark-X2.6-4B-DSpark.gguf \
  --draft-quantization int8 \
  --host 127.0.0.1 --port 8080 --no-thinking
```

合并 DSpark 模型可省略 `--draft-model`。更多参数可用各命令的 `--help` 查看。
需要 Python 接口时使用 `./bin/spark-mlx-python`；该解释器不读取系统 Python 包或
`PYTHONPATH`。

## 系统兼容与扩展

基础运行时的构建目标为 Apple Silicon、macOS 14.0+；预编译扩展可以放在同一包里。
项目会在导入扩展前检查系统、架构、Python 和 MLX；不满足扩展要求时跳过扩展，
使用原生 MLX。M5 专用算子另外检查设备能力。

```sh
./bin/spark-mlx-doctor
SPARK_MLX_DISABLE_EXT=1 ./bin/spark-mlx-doctor --gpu
```

`NATIVE-AUDIT.json` 中基础二进制应不高于选定系统版本，扩展二进制单独允许
26.2。构建与运行环境记录见 `BUILD-INFO.json`。

模型权重应另行交付。已准备好本地 MLX 模型后可以离线运行；下载或转换源模型时
仍可能需要网络、额外磁盘空间，以及完整的 tokenizer 文件。

## 正式对外发布

该工具生成本地可测试的发布包，保留上游或编译产生的签名，不自动执行组织的
Developer ID 签名与 Apple 公证。面向公众发布时，将这些步骤接入组织发布流程。
模型的交付与运行时版本可以分别管理。
