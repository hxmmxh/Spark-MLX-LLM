"""Launch the single CLI using the relocatable bundle's interpreter."""
import json
import platform
import sys
from pathlib import Path

MODULES = {"spark-mlx": "spark_mlx_llm.cli"}


def minimum_macos():
    path = Path(__file__).resolve().parents[1] / "BUILD-INFO.json"
    return json.loads(path.read_text())["minimum_macos"]


def runtime_issues(minimum, release, architecture):
    issues = []
    if sys.platform != "darwin" or architecture != "arm64":
        issues.append("需要 Apple Silicon 和原生 ARM64 macOS 运行环境")
    try:
        found = tuple(map(int, release.split(".")))
        required = tuple(map(int, minimum.split(".")))
        if (*found, 0, 0)[:3] < (*required, 0, 0)[:3]:
            issues.append(f"此发布包需要 macOS {minimum} 或更新版本")
    except (AttributeError, ValueError):
        issues.append("无法识别 macOS 版本")
    return issues


def main():
    command, *arguments = sys.argv[1:]
    if command not in MODULES:
        raise ValueError("Use spark-mlx with generate, chat, server, benchmark or doctor")
    issues = runtime_issues(minimum_macos(), platform.mac_ver()[0], platform.machine())
    if issues:
        raise RuntimeError("; ".join(issues))
    from spark_mlx_llm.cli import main as run
    return run(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Spark-MLX: {error}", file=sys.stderr)
        raise SystemExit(1) from error
