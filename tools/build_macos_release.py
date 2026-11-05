#!/usr/bin/env python3
"""Build a relocatable Apple Silicon CLI distribution with a bundled Python.

Run with: uv run --no-project --python 3.12 python tools/build_macos_release.py
Only uv and the native extension's build toolchain are needed on the builder.
The resulting archive needs neither uv nor a separately installed Python.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("macos_release")
COMMANDS = {
    "spark-mlx-generate": "spark_mlx_llm.commands.generate",
    "spark-mlx-chat": "spark_mlx_llm.commands.chat",
    "spark-mlx-convert": "spark_mlx_llm.commands.convert",
    "spark-mlx-server": "spark_mlx_llm.commands.server",
    "spark-mlx-dspark": "spark_mlx_llm.dspark.cli",
    "spark-mlx-dspark-server": "spark_mlx_llm.commands.dspark_server",
    "spark-mlx-doctor": None,
}
DOCUMENTATION_URL = "https://github.com/XHToken/Spark-MLX-LLM/blob/main"


def run(command, *, cwd=None, env=None, capture=False):
    print("+ " + " ".join(map(str, command)), flush=True)
    result = subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def os_version(value):
    if not re.fullmatch(r"\d+(?:\.\d+){0,2}", value):
        raise ValueError(f"Invalid macOS version: {value!r}")
    parts = tuple(map(int, value.split(".")))
    return (*parts, 0, 0)[:3]


def macho_info(path):
    """Read the ARM64 slice's deployment target and dyld paths, without Xcode."""
    data = Path(path).read_bytes()
    if data[:4] in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
        wide = data[:4] == b"\xca\xfe\xba\xbf"
        count = struct.unpack_from(">I", data, 4)[0]
        for index in range(count):
            offset = 8 + index * (32 if wide else 20)
            cpu = struct.unpack_from(">I", data, offset)[0]
            if cpu == 0x0100000C:
                start = struct.unpack_from(">Q" if wide else ">I", data, offset + 8)[0]
                data = data[start:]
                break
        else:
            raise ValueError(f"Mach-O has no ARM64 slice: {path}")
    if data[:4] not in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
        return None
    cpu, _, _, count = struct.unpack_from("<IIII", data, 4)
    if cpu != 0x0100000C:
        raise ValueError(f"Non-ARM64 Mach-O: {path}")
    offset = 32 if data[:4] == b"\xcf\xfa\xed\xfe" else 28
    minimum, paths = None, []
    for _ in range(count):
        command, size = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError(f"Invalid Mach-O load command: {path}")
        if command in (0x24, 0x32):
            position = offset + (12 if command == 0x32 else 8)
            value = struct.unpack_from("<I", data, position)[0]
            minimum = (value >> 16, value >> 8 & 255, value & 255)
        if command in (0xC, 0xD, 0x80000018, 0x8000001F, 0x20, 0x80000023, 0x8000001C):
            start = struct.unpack_from("<I", data, offset + 8)[0]
            value = data[offset + start : offset + size].split(b"\0", 1)[0].decode()
            paths.append(value)
        offset += size
    return {"minimum_macos": minimum, "dyld_paths": paths}


def audit_bundle(bundle, minimum):
    """Reject developer-machine links and binaries too new for the base runtime."""
    bundle = bundle.resolve()
    report = {}
    for path in sorted(bundle.rglob("*")):
        relative = path.relative_to(bundle).as_posix()
        if path.is_symlink():
            target = path.resolve()
            if Path(os.readlink(path)).is_absolute() or not target.is_relative_to(
                bundle
            ):
                raise ValueError(f"Non-relocatable symlink: {relative}")
            if not target.exists():
                raise ValueError(f"Broken symlink: {relative}")
            continue
        if not path.is_file():
            continue
        # Avoid loading shaders and large Python resources just to test their magic.
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic not in (
            b"\xcf\xfa\xed\xfe",
            b"\xce\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe",
            b"\xca\xfe\xba\xbf",
        ):
            continue
        info = macho_info(path)
        extension = "/site-packages/spark_mlx_ext/" in "/" + relative
        limit = (26, 2, 0) if extension else minimum
        if info["minimum_macos"] is None or tuple(info["minimum_macos"]) > limit:
            raise ValueError(
                f"{relative}: minimum macOS {info['minimum_macos']} exceeds {limit}"
            )
        for value in info["dyld_paths"]:
            if value.startswith("/") and not value.startswith(
                ("/usr/lib/", "/System/Library/")
            ):
                raise ValueError(f"{relative}: non-system absolute dyld path {value}")
        report[relative] = info
    if not report:
        raise ValueError("No native binaries found in release")
    for path in bundle.rglob("*.pth"):
        content = path.read_text()
        if "__editable__" in content or any(
            line.startswith("/") for line in content.splitlines()
        ):
            raise ValueError(f"Non-relocatable site initialization: {path}")
    return report


def snapshot_source(destination):
    destination.mkdir()
    for name in ("pyproject.toml", "uv.lock", "uv.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, destination / name)
    excluded = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.so", "*.metallib", "tests", "build", "*.egg-info"
    )
    shutil.copytree(
        ROOT / "spark_mlx_llm", destination / "spark_mlx_llm", ignore=excluded
    )
    ext_source = ROOT / "extensions/spark-mlx-ext"
    ext_target = destination / "extensions/spark-mlx-ext"
    ext_target.mkdir(parents=True)
    for name in (
        "pyproject.toml",
        "setup.py",
        "CMakeLists.txt",
        "MANIFEST.in",
        "README.md",
        "README.zh.md",
        "LICENSE",
        "THIRD_PARTY.md",
    ):
        shutil.copy2(ext_source / name, ext_target / name)
    for name in ("spark_mlx_ext", "src", "LICENSES"):
        shutil.copytree(ext_source / name, ext_target / name, ignore=excluded)
    # Freeze the release tools too; later working-tree edits cannot change this build.
    shutil.copytree(ASSETS, destination / "release-assets", ignore=excluded)
    shutil.copy2(__file__, destination / "release-assets/build_macos_release.py")
    for language in ("en", "zh"):
        guide = destination / "release-assets/docs" / language / "release-package.md"
        guide.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "docs" / language / "release-package.md", guide)
    return {
        path.relative_to(destination).as_posix(): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }


def standalone_python(uv, request):
    run([uv, "python", "install", request])
    executable = run(
        [uv, "python", "find", "--no-project", "--managed-python", "--system", request],
        capture=True,
    )
    info = json.loads(
        run(
            [
                executable,
                "-I",
                "-c",
                "import json,sys,platform; print(json.dumps({'prefix':sys.base_prefix,'version':platform.python_version(),'machine':platform.machine()}))",
            ],
            capture=True,
        )
    )
    if info["machine"] != "arm64" or not info["version"].startswith("3.12."):
        raise ValueError("This release tool requires a managed ARM64 CPython 3.12")
    prefix = Path(info["prefix"]).resolve()
    if not (prefix / "BUILD").is_file():
        raise ValueError("Expected a uv-managed python-build-standalone installation")
    return prefix, info["version"]


def copy_python(prefix, runtime):
    shutil.copytree(
        prefix,
        runtime,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "site-packages"),
    )
    (runtime / "lib/python3.12/site-packages").mkdir()
    # This copy belongs to the application. Leave the original managed Python
    # untouched and remove its package-manager marker only from our runtime.
    (runtime / "lib/python3.12/EXTERNALLY-MANAGED").unlink(missing_ok=True)
    library = runtime / "lib/libpython3.12.dylib"
    if library.is_file():
        # uv rewrites this identity to its managed install's absolute path.
        info = macho_info(library)
        if info and any(value.startswith(str(prefix)) for value in info["dyld_paths"]):
            run(["install_name_tool", "-id", "@rpath/libpython3.12.dylib", library])
            run(["codesign", "--force", "--sign", "-", library])


def shell_launcher(command):
    return (
        "#!/bin/sh\nset -eu\n"
        'SPARK_RELEASE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)\n'
        'exec "$SPARK_RELEASE_ROOT/runtime/bin/python3" -I -B '
        f'"$SPARK_RELEASE_ROOT/libexec/launch.py" {command} "$@"\n'
    )


def write_launchers(bundle, assets):
    (bundle / "bin").mkdir()
    (bundle / "libexec").mkdir()
    shutil.copy2(assets / "launch.py", bundle / "libexec/launch.py")
    for command in COMMANDS:
        path = bundle / "bin" / command
        path.write_text(shell_launcher(command))
        path.chmod(0o755)
    path = bundle / "bin/spark-mlx-python"
    path.write_text(
        "#!/bin/sh\nset -eu\n"
        'SPARK_RELEASE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)\n'
        'exec "$SPARK_RELEASE_ROOT/runtime/bin/python3" -I -B "$@"\n'
    )
    path.chmod(0o755)


def write_release_guides(bundle, assets):
    """Keep both user guides offline and link other project docs to the repository."""
    guides = {f"docs/{language}/release-package.md" for language in ("en", "zh")}
    for relative in sorted(guides):
        source = assets / relative
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        def link(match, guide=relative):
            value = match[1]
            if "://" in value or value.startswith("#"):
                return match[0]
            path, separator, fragment = value.partition("#")
            resolved = (ROOT / guide).parent.joinpath(path).resolve()
            repository_path = resolved.relative_to(ROOT).as_posix()
            if repository_path in guides:
                return match[0]
            return f"]({DOCUMENTATION_URL}/{repository_path}{separator}{fragment})"

        target.write_text(re.sub(r"\]\(([^)]+)\)", link, source.read_text()))


def clean_runtime(bundle):
    runtime = bundle / "runtime"
    # Installed console scripts have absolute shebangs. Expose only our launchers.
    for path in (runtime / "bin").iterdir():
        if path.name not in {"python", "python3", "python3.12"}:
            path.unlink()
    for path in runtime.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
    for name in ("direct_url.json", "uv_build.json", "uv_cache.json"):
        for path in runtime.rglob(name):
            if path.parent.name.endswith(".dist-info"):
                path.unlink()


def repack_installed_extension(executable, wheels):
    """Make a normal wheel from an installed, already compiled extension."""
    script = """
import importlib.util, json
from importlib.metadata import distribution
from pathlib import Path
d = distribution('spark-mlx-ext')
spec = importlib.util.find_spec('spark_mlx_ext')
metadata = next(p for p in d.files if p.name == 'METADATA' and '.dist-info' in str(p))
print(json.dumps({'root':str(Path(spec.origin).parent),
    'metadata':str(d.locate_file(metadata).parent), 'version':d.version}))
"""
    # Preserve a venv interpreter's path: resolving its symlink loses the venv.
    info = json.loads(
        run([os.path.abspath(executable), "-I", "-c", script], capture=True)
    )
    package = Path(info["root"])
    metadata = Path(info["metadata"])
    if not list(package.glob("_ext*.so")) or not list(package.glob("*.metallib")):
        raise ValueError(
            "Installed extension must already contain its native module and Metal libraries"
        )
    wheel_metadata = (metadata / "WHEEL").read_text()
    tag = next(
        line.removeprefix("Tag: ")
        for line in wheel_metadata.splitlines()
        if line.startswith("Tag: ")
    )
    if not tag.startswith("cp312-cp312-") or not tag.endswith("arm64"):
        raise ValueError(f"Installed extension wheel tag is incompatible: {tag}")
    dist_info = f"spark_mlx_ext-{info['version']}.dist-info"
    output = wheels / f"spark_mlx_ext-{info['version']}-{tag}.whl"
    entries = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            entries.append(
                (path, "spark_mlx_ext/" + path.relative_to(package).as_posix())
            )
    for path in sorted(metadata.rglob("*")):
        relative = path.relative_to(metadata)
        if path.is_file() and (
            relative.parts[0] == "licenses"
            or relative.as_posix() in {"METADATA", "WHEEL", "top_level.txt"}
        ):
            entries.append((path, dist_info + "/" + relative.as_posix()))
    records = io.StringIO(newline="")
    writer = csv.writer(records)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in entries:
            archive.write(path, name)
            digest = (
                base64.urlsafe_b64encode(bytes.fromhex(sha256(path)))
                .rstrip(b"=")
                .decode()
            )
            writer.writerow([name, "sha256=" + digest, path.stat().st_size])
        record = dist_info + "/RECORD"
        writer.writerow([record, "", ""])
        archive.writestr(record, records.getvalue())
    return output


def verify_release(bundle, *, gpu, model=None):
    """Validate relocated launchers from an unrelated cwd and hostile Python env."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "UV_", "DYLD_"))
    }
    env.update(
        PYTHONPATH="/nonexistent/developer/environment",
        PYTHONHOME="/nonexistent/python",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    env.pop("SPARK_MLX_DISABLE_EXT", None)
    report = {}
    with tempfile.TemporaryDirectory(prefix="spark-release-check-") as cwd:
        for name in COMMANDS:
            if name != "spark-mlx-doctor":
                run([bundle / "bin" / name, "--help"], cwd=cwd, env=env, capture=True)
        for disabled in (False, True):
            selected = dict(env)
            if disabled:
                selected["SPARK_MLX_DISABLE_EXT"] = "1"
            command = [
                bundle / "bin/spark-mlx-doctor",
                "--json",
                *(["--gpu"] if gpu else []),
            ]
            if model:
                command.extend(["--model", str(Path(model).resolve())])
            key = "native_fallback" if disabled else "automatic"
            report[key] = json.loads(run(command, cwd=cwd, env=selected, capture=True))
        if report["native_fallback"]["extension"]["available"]:
            raise ValueError("Extension disable check failed")
        automatic = report["automatic"]
        if (
            automatic["packages"].get("spark-mlx-ext")
            and os_version(automatic["macos"]) >= (26, 2, 0)
            and not automatic["extension"]["available"]
        ):
            raise ValueError(
                f"Bundled extension failed to load: {automatic['extension']['reason']}"
            )
    return report


def build(args):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError("Build this distribution on a native Apple Silicon Mac")
    minimum = os_version(args.min_macos)
    if minimum < (14, 0, 0):
        raise ValueError("The MLX base runtime requires macOS >= 14")
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("Install uv on the build machine first")
    if not any((args.without_extension, args.extension_wheel, args.extension_python)):
        # --find metal can locate only an Xcode shim with no compiler installed.
        run(["xcrun", "metal", "--version"], capture=True)
    registry = ["--default-index", args.index_url] if args.index_url else []
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    name = args.name or f"spark-mlx-{metadata['version']}-macos-arm64"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(
            "Release name must contain only letters, numbers, '.', '_' and '-'"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}.tar.gz"
    if archive.exists() and not args.force:
        raise FileExistsError(f"{archive} exists; choose --name or use --force")
    work_root = ROOT / ".build/releases"
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=work_root))
    print(f"Build workspace: {work}", flush=True)
    source = work / "source"
    source_hashes = snapshot_source(source)
    run([uv, "lock", "--check", "--offline"], cwd=source)
    prefix, python_version = standalone_python(uv, args.python)
    bundle = work / name
    bundle.mkdir()
    copy_python(prefix, bundle / "runtime")
    python = bundle / "runtime/bin/python3"
    wheels = work / "wheels"
    wheels.mkdir()
    requirements = work / "requirements.lock"
    run(
        [
            uv,
            "export",
            "--frozen",
            "--no-emit-project",
            "--no-editable",
            "--output-file",
            requirements,
        ],
        cwd=source,
        capture=True,
    )
    env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": args.min_macos}
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "--system",
            "--python-platform",
            "aarch64-apple-darwin",
            "--only-binary",
            ":all:",
            "--require-hashes",
            "--link-mode",
            "copy",
            "-r",
            requirements,
            *registry,
        ],
        cwd=source,
        env=env,
    )
    run(
        [uv, "build", "--wheel", "--python", python, "--out-dir", wheels, *registry],
        cwd=source,
    )
    main_wheel = next(wheels.glob("spark_mlx_llm-*.whl"))
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "--system",
            "--no-deps",
            "--link-mode",
            "copy",
            main_wheel,
        ],
        cwd=source,
    )
    if not args.without_extension:
        if args.extension_wheel:
            shutil.copy2(args.extension_wheel, wheels)
        elif args.extension_python:
            repack_installed_extension(args.extension_python, wheels)
        else:
            # Inherit the exact MLX binaries/headers selected for the base OS.
            # PEP 517's default isolation would choose the builder's newer MLX
            # wheel, which is not necessarily the same native build.
            build_env = work / "extension-build-env"
            run([uv, "venv", "--python", python, "--system-site-packages", build_env])
            build_python = build_env / "bin/python"
            build_requires = tomllib.loads(
                (source / "extensions/spark-mlx-ext/pyproject.toml").read_text()
            )["build-system"]["requires"]
            run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    build_python,
                    *[item for item in build_requires if not item.startswith("mlx")],
                    *registry,
                ],
                cwd=source,
            )
            native_env = {
                **os.environ,
                "MACOSX_DEPLOYMENT_TARGET": "26.2",
                "CMAKE_ARGS": "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 -DCMAKE_OSX_ARCHITECTURES=arm64",
                "SPARK_MLX_BUILD_ROOT": str(work / "native-build"),
                "DEBUG": "0",
            }
            run(
                [
                    uv,
                    "build",
                    source / "extensions/spark-mlx-ext",
                    "--wheel",
                    "--python",
                    build_python,
                    "--no-build-isolation",
                    "--out-dir",
                    wheels,
                    *registry,
                ],
                cwd=source,
                env=native_env,
            )
        ext_wheel = next(wheels.glob("spark_mlx_ext-*.whl"))
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--system",
                "--no-deps",
                "--link-mode",
                "copy",
                ext_wheel,
            ],
            cwd=source,
        )
        installed = bundle / "runtime/lib/python3.12/site-packages/spark_mlx_ext"
        if not list(installed.glob("_ext*.so")) or not list(
            installed.glob("*.metallib")
        ):
            raise ValueError(
                "Extension wheel must contain its native module and Metal libraries"
            )
    write_launchers(bundle, source / "release-assets")
    write_release_guides(bundle, source / "release-assets")
    readme = (source / "release-assets/README.txt").read_text()
    readme = readme.replace("macOS 14 或更新版本", f"macOS {args.min_macos} 或更新版本")
    (bundle / "README.txt").write_text(readme)
    shutil.copy2(source / "LICENSE", bundle / "LICENSE")
    shutil.copy2(requirements, bundle / "requirements.lock")
    clean_runtime(bundle)
    audit = audit_bundle(bundle, minimum)
    (bundle / "NATIVE-AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Move the directory, including spaces and Unicode, before any smoke checks.
    relocated = work / "relocation 空格验证" / name
    relocated.parent.mkdir()
    bundle.rename(relocated)
    # Doctor must check this release's actual requirement during smoke checks.
    (relocated / "BUILD-INFO.json").write_text(
        json.dumps({"minimum_macos": args.min_macos}) + "\n"
    )
    verification = verify_release(
        relocated, gpu=not args.skip_gpu_check, model=args.smoke_model
    )
    versions = json.loads(
        run(
            [
                relocated / "bin/spark-mlx-python",
                "-c",
                "import json; from importlib.metadata import distributions; print(json.dumps({d.metadata['Name']:d.version for d in distributions()}))",
            ],
            capture=True,
        )
    )
    git_revision = run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture=True)
    dirty = bool(run(["git", "status", "--porcelain"], cwd=ROOT, capture=True))
    info = {
        "name": name,
        "version": metadata["version"],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": git_revision,
        "source_dirty": dirty,
        "source_sha256": source_hashes,
        "python_version": python_version,
        "python_build": (prefix / "BUILD").read_text().strip(),
        "minimum_macos": args.min_macos,
        "architecture": "arm64",
        "extension_bundled": not args.without_extension,
        "extension_origin": "installed-precompiled"
        if args.extension_python
        else "provided-wheel"
        if args.extension_wheel
        else "source-build"
        if not args.without_extension
        else None,
        "extension_minimum_macos": "26.2" if not args.without_extension else None,
        "package_versions": versions,
        "wheels": {p.name: sha256(p) for p in sorted(wheels.glob("*.whl"))},
        "validation_host_macos": platform.mac_ver()[0],
        "validation": verification,
        "old_macos_end_to_end_verified": False,
        "distribution_signing": "upstream/ad-hoc signatures only; no release Developer ID signing or notarization",
    }
    (relocated / "BUILD-INFO.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    )
    checksums = [
        f"{sha256(path)}  {path.relative_to(relocated).as_posix()}"
        for path in sorted(relocated.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    (relocated / "FILES.sha256").write_text("\n".join(checksums) + "\n")
    temporary_archive = archive.with_suffix(".gz.tmp")
    with tarfile.open(temporary_archive, "w:gz", compresslevel=6) as tar:
        tar.add(relocated, arcname=name)
    temporary_archive.replace(archive)
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n")
    shutil.copy2(relocated / "BUILD-INFO.json", output / f"{name}.BUILD-INFO.json")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": str(checksum),
                "size_mib": round(archive.stat().st_size / 1024**2, 1),
                "verified_directory": str(relocated),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default="3.12.16",
        help="uv-managed CPython version (default: 3.12.16)",
    )
    parser.add_argument("--min-macos", default="14.0")
    parser.add_argument(
        "--index-url", help="Override the package index without changing uv.toml"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--name", help="Override the release directory/archive name")
    parser.add_argument(
        "--extension-wheel",
        type=Path,
        help="Reuse a prebuilt CPython 3.12 ARM64 extension wheel",
    )
    parser.add_argument("--without-extension", action="store_true")
    parser.add_argument(
        "--extension-python",
        type=Path,
        help="Repack the already compiled extension installed in this Python environment",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Explicitly skip real GPU validation",
    )
    parser.add_argument(
        "--smoke-model",
        type=Path,
        help="Optional local model for real generation during validation; not bundled",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing archive of the same name",
    )
    args = parser.parse_args(argv)
    if (
        sum(
            bool(value)
            for value in (
                args.extension_wheel,
                args.extension_python,
                args.without_extension,
            )
        )
        > 1
    ):
        parser.error(
            "Choose only one of --extension-wheel, --extension-python, --without-extension"
        )
    try:
        build(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release build failed: {error}\n")


if __name__ == "__main__":
    main()
