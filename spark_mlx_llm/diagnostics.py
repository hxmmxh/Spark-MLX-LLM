"""Environment and optional generation checks through the shared public API."""
import importlib.metadata
import json
import platform
import sys


def doctor(parser, args, config):
    if args.model and not args.gpu:
        parser.error("--model requires --gpu")
    report = {"python": sys.version, "platform": platform.platform(), "macos": platform.mac_ver()[0], "packages": {}, "checks": {}}
    for name in ("spark-mlx-llm", "mlx", "mlx-lm", "spark-mlx-ext"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import mlx.core as mx
        from .extension import extension_status
        report["device"] = str(mx.default_device())
        report["extension"] = extension_status()
        report["checks"]["environment"] = True
        if args.gpu:
            from .benchmark.runtime import device_lock
            with device_lock(timeout=args.lock_timeout):
                mx.set_default_device(mx.gpu)
                value = mx.ones((2, 32)) @ mx.ones((32, 4))
                mx.eval(value)
                if not bool(mx.all(value == 32)):
                    raise RuntimeError("GPU matrix check failed")
                report["checks"]["gpu"] = True
                if args.model:
                    from .cli import _load
                    from .generation import generate
                    model, tokenizer = _load(args)
                    result = generate(model, tokenizer, "你好，请介绍自己。", config=config)
                    report["checks"]["generation"] = bool(result.text)
                    report["text"] = result.text
                    report["metrics"] = result.metrics()
        report["status"] = "ok"
    except Exception as error:
        report.update(status="failed", error=str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1
