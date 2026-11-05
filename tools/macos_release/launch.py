"""Entrypoint for the relocatable release; uses only its bundled interpreter."""

import argparse
import json
import platform
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

MODULES = {
    "spark-mlx-generate": "spark_mlx_llm.commands.generate",
    "spark-mlx-chat": "spark_mlx_llm.commands.chat",
    "spark-mlx-convert": "spark_mlx_llm.commands.convert",
    "spark-mlx-server": "spark_mlx_llm.commands.server",
    "spark-mlx-dspark": "spark_mlx_llm.dspark.cli",
    "spark-mlx-dspark-server": "spark_mlx_llm.commands.dspark_server",
}


def gpu_check():
    import mlx.core as mx

    from spark_mlx_llm.extension import get_extension

    if not mx.metal.is_available():
        raise RuntimeError("Metal GPU is unavailable")
    mx.set_default_device(mx.gpu)
    x = mx.ones((2, 64), dtype=mx.bfloat16)
    weight = mx.ones((32, 64), dtype=mx.bfloat16)
    packed, scales, biases = mx.quantize(weight, group_size=32, bits=4)
    dense = x @ weight.T
    quantized = mx.quantized_matmul(
        x, packed, scales, biases, transpose=True, group_size=32, bits=4
    )
    mx.eval(dense, quantized)
    if not bool(mx.all(dense == 64)) or not bool(mx.all(quantized == 64)):
        raise RuntimeError("Native BF16/INT4 GPU check failed")
    result = {"metal": True, "bf16_matmul": True, "int4_matmul": True}
    ext = get_extension()
    if ext is not None:
        value = ext.w8a16_per_channel(
            x[:1], weight.astype(mx.int8), mx.ones((32,), dtype=mx.float32)
        )
        mx.eval(value)
        if not bool(mx.all(value == 64)):
            raise RuntimeError("Extension W8A16 GPU check failed")
        result["extension_w8a16"] = True
        if ext.is_w8a8_available():
            value = ext.int8_matmul_int32(
                mx.ones((2, 64), dtype=mx.int8), weight.astype(mx.int8)
            )
            mx.eval(value)
            if not bool(mx.all(value == 64)):
                raise RuntimeError("Extension TensorOps GPU check failed")
            result["extension_tensorops"] = True
    return result


def model_check(path):
    from mlx_lm.sample_utils import make_sampler

    from spark_mlx_llm.generation import generate
    from spark_mlx_llm.loader import load

    model, tokenizer = load(path, tokenizer_config={"trust_remote_code": False})
    prompt = "你好，请用一句话介绍自己。"
    if tokenizer.has_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=16,
        sampler=make_sampler(temp=0),
        verbose=False,
    )
    if not text:
        raise RuntimeError("Model generation returned no text")
    return {"generation": True, "text": text}


def minimum_macos():
    bundle = Path(__file__).resolve().parents[1]
    info_path = bundle / "BUILD-INFO.json"
    return json.loads(info_path.read_text())["minimum_macos"]


def runtime_issues(minimum, release, architecture):
    issues = []
    if sys.platform != "darwin" or architecture != "arm64":
        issues.append("需要 Apple Silicon 和原生 ARM64 macOS 运行环境")
    try:
        found = tuple(map(int, release.split(".")))
        required = tuple(map(int, minimum.split(".")))
    except (ValueError, AttributeError):
        issues.append("无法识别 macOS 版本或发布包的最低系统版本")
    else:
        if (*found, 0, 0)[:3] < (*required, 0, 0)[:3]:
            issues.append(
                f"当前 macOS {release}，此发布包需要 macOS {minimum} 或更新版本"
            )
    return issues


def environment_check(report):
    """Check imports and GPU availability without evaluating any GPU arrays."""
    mx = import_module("mlx.core")
    report["metal_available"] = bool(mx.metal.is_available())
    if not report["metal_available"]:
        raise RuntimeError("Metal GPU 不可用，无法进行模型推理")
    import_module("mlx_lm")
    import_module("spark_mlx_llm.model")
    report["extension"] = import_module("spark_mlx_llm.extension").extension_status()
    report["checks"]["environment"] = "passed"


def print_doctor_summary(report):
    summary = report["summary"]
    print(f"结论：{summary['message']}")
    print(
        f"系统：macOS {report['macos'] or '未知'} / {report['architecture']}（最低要求 {report['minimum_macos'] or '未知'}）"
    )
    print(
        f"Python：{report['python']}；MLX：{report['packages']['mlx'] or '未安装'}；MLX LM：{report['packages']['mlx-lm'] or '未安装'}"
    )
    metal = report["metal_available"]
    print(
        "Metal GPU：" + ("可用" if metal else "不可用" if metal is False else "未检查")
    )
    if report["checks"]["environment"] == "passed":
        if report["extension"]["available"]:
            print("运行方式：扩展加速（按设备能力选择算子）")
        else:
            print("运行方式：原生 MLX（未启用可选扩展，不影响基础推理）")
            print(f"扩展说明：{report['extension']['reason']}")
    labels = {
        "passed": "通过",
        "failed": "失败",
        "not_requested": "未执行",
        "not_run": "未完成",
    }
    print(f"GPU 运算验证：{labels[report['checks']['gpu']]}")
    print(f"模型生成验证：{labels[report['checks']['model']]}")
    for error in report["errors"]:
        print(f"原因：{error}")
    if summary["status"] == "unverified":
        print("建议：等待其他 GPU 任务结束后重试，或添加 --lock-timeout 30。")
    elif summary["status"] == "not_ready":
        if report["checks"]["model"] == "failed":
            print("建议：检查模型路径、模型文件完整性和可用内存。")
        else:
            print(
                "建议：按上述原因检查系统和依赖，必要时重新解压适配本机的完整发布包。"
            )
    elif report["checks"]["gpu"] == "not_requested":
        print(
            "提示：尚未实测 GPU 运算或加载模型；可加 --gpu，或 --gpu --model /path/to/model 进一步验证。"
        )
    elif report["checks"]["model"] == "not_requested":
        print(
            "提示：GPU 运算已通过，尚未加载模型；可加 --model /path/to/model 验证具体模型。"
        )


def doctor(argv):
    parser = argparse.ArgumentParser(
        description="Inspect the bundled runtime and optional Metal extension"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Execute native and available extension GPU checks",
    )
    parser.add_argument(
        "--model",
        help="Additionally generate 16 tokens from a local model (requires --gpu)",
    )
    parser.add_argument("--lock-timeout", type=float, default=0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable report instead of the human summary",
    )
    args = parser.parse_args(argv)
    if args.model and not args.gpu:
        parser.error("--model requires --gpu")
    packages = {}
    for name in ("spark-mlx-llm", "mlx", "mlx-metal", "mlx-lm", "spark-mlx-ext"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "runtime": sys.prefix,
        "packages": packages,
        "minimum_macos": None,
        "metal_available": None,
        "extension": {"available": False, "reason": "未检查"},
        "checks": {
            "environment": "not_run",
            "gpu": "not_run" if args.gpu else "not_requested",
            "model": "not_run" if args.model else "not_requested",
        },
        "errors": [],
    }
    try:
        report["minimum_macos"] = minimum_macos()
        report["errors"].extend(
            runtime_issues(
                report["minimum_macos"], report["macos"], report["architecture"]
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(f"无法读取发布包系统要求：{error}")
    required = ("spark-mlx-llm", "mlx", "mlx-metal", "mlx-lm")
    report["errors"].extend(
        f"缺少必要依赖：{name}" for name in required if packages[name] is None
    )
    stage = "environment"
    pending = False
    if not report["errors"]:
        try:
            if args.gpu:
                # Acquire before any import that could load MLX or the extension.
                runtime = import_module("spark_mlx_llm.benchmark.runtime")
                try:
                    lock = runtime.device_lock(timeout=args.lock_timeout)
                except TimeoutError as error:
                    pending = True
                    report["errors"].append(
                        f"GPU 验证暂未完成，设备锁被其他任务占用：{error}"
                    )
                else:
                    with lock:
                        environment_check(report)
                        stage = "gpu"
                        report["gpu"] = gpu_check()
                        report["checks"]["gpu"] = "passed"
                        if args.model:
                            stage = "model"
                            report["model"] = model_check(args.model)
                            report["checks"]["model"] = "passed"
            else:
                environment_check(report)
        except Exception as error:  # noqa: BLE001 -- diagnostics must explain runtime failures
            report["checks"][stage] = "failed"
            report["errors"].append(f"{type(error).__name__}: {error}")
    else:
        report["checks"]["environment"] = "failed"
    if pending:
        status, can_run, message = "unverified", None, "暂不能确认（GPU 验证被占用）"
    elif report["errors"]:
        status, can_run = "not_ready", False
        message = (
            "不能跑（指定模型验证失败）"
            if stage == "model"
            else "不能跑（本次检查未通过）"
        )
    else:
        status, can_run = "ready", True
        message = (
            "能跑（指定模型生成验证通过）"
            if args.model
            else "能跑（GPU 运算验证通过）"
            if args.gpu
            else "能跑（基础环境检查通过）"
        )
    report["summary"] = {"status": status, "can_run": can_run, "message": message}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_doctor_summary(report)
    return 2 if pending else 1 if report["errors"] else 0


def main():
    command, *arguments = sys.argv[1:]
    if command == "spark-mlx-doctor":
        return doctor(arguments)
    issues = runtime_issues(minimum_macos(), platform.mac_ver()[0], platform.machine())
    if issues:
        raise RuntimeError("; ".join(issues))
    module = import_module(MODULES[command])
    sys.argv = [command, *arguments]
    return module.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Spark-MLX: {error}", file=sys.stderr)
        raise SystemExit(1) from error
