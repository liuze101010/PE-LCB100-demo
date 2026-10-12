#!/usr/bin/env python3
"""Install the paper-pinned PE-LCB science runtime in a user-owned directory.

The installer is intentionally bounded: download locations, versions and
SHA256 digests are compiled into this file.  It never invokes a system package
manager or sudo.  Site-owned prerequisites (Slurm, compiler, CMake, CUDA
driver/toolkit and outbound network access) are reported as administrator
actions instead of being modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "polymer_agent.science_runtime_install.v1"
GROMACS_VERSION = "2024.4"
GROMACS_URL = "https://ftp.gromacs.org/gromacs/gromacs-2024.4.tar.gz"
GROMACS_SHA256 = "ac618ece2e58afa86b536c5a2c4fcb937f0760318f12d18f10346b6bdebd86a8"
EMC_VERSION = "9.4.4"
EMC_RELEASE = "20260721"
EMC_DISTRIBUTIONS = {
    "x86_64": {
        "name": "emc-official-linux-x86_64",
        "version": f"{EMC_VERSION}+{EMC_RELEASE}",
        "filename": "emc_linux_x86_64_v9.4.4_20260701.tgz",
        "url": "https://downloads.sourceforge.net/project/montecarlo/emc_linux_x86_64_v9.4.4_20260701.tgz",
        "sha256": "acb825b947c8dc48b8cbf06e7694f55ecfc004a2a840c5960ccebbbf59f445d5",
    },
    "aarch64": {
        "name": "emc-official-linux-aarch64",
        "version": f"{EMC_VERSION}+{EMC_RELEASE}",
        "filename": "emc_linux_aarch64_v9.4.4_20260701.tgz",
        "url": "https://downloads.sourceforge.net/project/montecarlo/emc_linux_aarch64_v9.4.4_20260701.tgz",
        "sha256": "00d0aee109736054bed0f8c35a024947e562607e58fd1a0e08ce0a6f50ce9ed7",
    },
}
# Backward-compatible default used by deployment metadata and older callers.
EMC_DISTRIBUTION = EMC_DISTRIBUTIONS["x86_64"]
EMC_BINARIES = {
    "x86_64": "emc_linux_x86_64",
    "aarch64": "emc_linux_aarch64",
}
PYTHON_PACKAGES = (
    "numpy==1.26.4",
    "rdkit==2025.3.6",
    "pandas==2.2.3",
    "pydantic==2.11.7",
    "jsonschema==4.25.1",
    "PyYAML==6.0.2",
)
DOWNLOAD_RETRIES = 5
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Status:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "starting",
            "stage": "preflight",
            "percent": 1,
            "started_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            "admin_actions": [],
        }
        atomic_json(self.path, self.payload)

    def update(self, **values: Any) -> None:
        if "status" not in values and self.payload.get("status") == "starting":
            self.payload["status"] = "running"
        self.payload.update(values)
        self.payload["updated_at"] = utc_timestamp()
        atomic_json(self.path, self.payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_download(path: Path, expected_sha256: str, expected_size: int | None = None) -> None:
    if expected_size is not None:
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"incomplete download: expected {expected_size} bytes, got {actual_size}"
            )
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"download checksum mismatch: expected {expected_sha256}, got {actual}")


def _python_download_once(url: str, temporary: Path, expected_sha256: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "PE-LCB-science-runtime/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        header = response.headers.get("Content-Length")
        expected_size = int(header) if header and header.isdigit() else None
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
    _verified_download(temporary, expected_sha256, expected_size)


def _external_download_once(url: str, temporary: Path, program: str) -> None:
    if program == "curl":
        command = [
            "curl",
            "-L",
            "--fail",
            "--show-error",
            "--retry",
            "4",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--max-time",
            "1800",
            "-o",
            str(temporary),
            url,
        ]
    elif program == "wget":
        command = [
            "wget",
            "--tries=5",
            "--timeout=30",
            "--output-document",
            str(temporary),
            url,
        ]
    else:
        raise ValueError(f"unsupported downloader: {program}")
    result = subprocess.run(command, text=True, capture_output=True, timeout=1900, check=False)
    if result.returncode:
        tail = ((result.stdout or "") + "\n" + (result.stderr or ""))[-1200:]
        raise RuntimeError(f"{program} failed while downloading {url}: {tail}")


def download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    downloaders = ["python"]
    if shutil.which("curl"):
        downloaders.append("curl")
    if shutil.which("wget"):
        downloaders.append("wget")

    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        for downloader in downloaders:
            temporary.unlink(missing_ok=True)
            try:
                if downloader == "python":
                    _python_download_once(url, temporary, expected_sha256)
                else:
                    _external_download_once(url, temporary, downloader)
                    _verified_download(temporary, expected_sha256)
                temporary.replace(destination)
                return
            except Exception as exc:  # noqa: BLE001 - all methods are retried uniformly
                last_error = exc
                temporary.unlink(missing_ok=True)
        time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"download failed after automatic retries for {url}: {last_error}")


def safe_extract(bundle: tarfile.TarFile, destination: Path) -> None:
    """Extract an upstream archive without permitting path or link escape."""
    destination = destination.resolve()
    for member in bundle.getmembers():
        target = (destination / member.name).resolve()
        if os.path.commonpath([str(destination), str(target)]) != str(destination):
            raise RuntimeError(f"archive member escapes destination: {member.name}")
        if member.issym() or member.islnk():
            link = (target.parent / member.linkname).resolve()
            if os.path.commonpath([str(destination), str(link)]) != str(destination):
                raise RuntimeError(f"archive link escapes destination: {member.name}")
    bundle.extractall(destination)


def run(command: list[str], *, cwd: Path | None = None, timeout: int = 7200) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        tail = ((result.stdout or "") + "\n" + (result.stderr or ""))[-3000:]
        raise RuntimeError(f"command failed ({command[0]}): {tail}")


def prerequisite_report(acceleration: str) -> list[dict[str, str]]:
    required = {
        "cmake": "请由集群管理员提供 CMake 3.18 或更高版本，或加载相应 module。",
        "make": "请由集群管理员提供 GNU Make。",
    }
    actions = [
        {"component": name, "action": message}
        for name, message in required.items()
        if shutil.which(name) is None
    ]
    if not any(shutil.which(name) for name in ["c++", "g++", "clang++", "icpx", "nvc++"]):
        actions.append({
            "component": "cxx17_compiler",
            "action": "请加载 GROMACS 2024.4 支持的 C++17 编译器（例如 GCC/G++ 或 Clang）。",
        })
    if acceleration == "cuda" and shutil.which("nvcc") is None:
        actions.append({
            "component": "cuda_toolkit",
            "action": "已选择 CUDA 构建，但当前登录环境找不到 nvcc；请加载与计算节点驱动兼容的 CUDA module。",
        })
    return actions


def run(command: list[str], *, cwd: Path | None = None, timeout: int = 7200, env: dict[str, str] | None = None) -> None:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=full_env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        tail = ((result.stdout or "") + "\n" + (result.stderr or ""))[-3000:]
        raise RuntimeError(f"command failed ({command[0]}): {tail}")


def _version_tuple(text: str) -> tuple[int, int, int] | None:
    import re

    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def _conda_library_env(root: Path) -> dict[str, str]:
    library_dirs: list[str] = []
    for base in [root, *root.glob("envs/*"), *root.glob("pkgs/*")]:
        lib = base / "lib"
        if lib.is_dir():
            library_dirs.append(str(lib))
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        library_dirs.append(existing)
    return {"LD_LIBRARY_PATH": ":".join(library_dirs)}


def _cmake_version(command: str, env: dict[str, str] | None = None) -> tuple[int, int, int] | None:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        [command, "--version"],
        text=True,
        capture_output=True,
        timeout=30,
        env=full_env,
        check=False,
    )
    if result.returncode:
        return None
    return _version_tuple(result.stdout)


def find_cmake() -> tuple[str, dict[str, str], str]:
    candidates: list[tuple[Path, dict[str, str]]] = []
    if os.environ.get("CMAKE"):
        candidates.append((Path(os.environ["CMAKE"]), {}))
    if shutil.which("cmake"):
        candidates.append((Path(shutil.which("cmake") or "cmake"), {}))
    home = Path.home()
    for root in [home / "miniforge3", home / "anaconda3", home / ".conda"]:
        if not root.exists():
            continue
        env = _conda_library_env(root)
        for pattern in ["bin/cmake", "envs/*/bin/cmake", "pkgs/cmake-*/bin/cmake"]:
            for candidate in root.glob(pattern):
                candidates.append((candidate, env))
    seen: set[str] = set()
    for candidate, env in candidates:
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        version = _cmake_version(key, env)
        if version and version >= (3, 18, 0):
            return key, env, ".".join(str(part) for part in version)
    raise RuntimeError("CMake 3.18 or newer was not found in PATH or user conda directories")


def _compiler_version(command: str, env: dict[str, str] | None = None) -> tuple[int, int, int] | None:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        [command, "--version"],
        text=True,
        capture_output=True,
        timeout=30,
        env=full_env,
        check=False,
    )
    if result.returncode:
        return None
    return _version_tuple(result.stdout)


def find_cxx17_compilers() -> tuple[str, str, dict[str, str], str]:
    candidates: list[tuple[Path, Path, dict[str, str]]] = []
    for cc_name, cxx_name in [("gcc", "g++"), ("cc", "c++"), ("clang", "clang++")]:
        cc = shutil.which(cc_name)
        cxx = shutil.which(cxx_name)
        if cc and cxx:
            candidates.append((Path(cc), Path(cxx), {}))
    home = Path.home()
    for root in [home / "miniforge3", home / "anaconda3", home / ".conda"]:
        if not root.exists():
            continue
        env = _conda_library_env(root)
        for base in [root, *root.glob("envs/*"), *root.glob("pkgs/gcc_impl_linux-64-*")]:
            cc_matches = sorted(base.glob("bin/*gcc"))
            for cc in cc_matches:
                cxx = cc.with_name(cc.name.replace("gcc", "g++"))
                if cxx.exists():
                    candidates.append((cc, cxx, env))
        for base in root.glob("pkgs/gxx_impl_linux-64-*"):
            for cxx in sorted(base.glob("bin/*g++")):
                gcc_root = root / "pkgs"
                suffix = base.name.replace("gxx_impl_linux-64-", "")
                cc_candidates = sorted(gcc_root.glob(f"gcc_impl_linux-64-{suffix.split('-')[0]}*/bin/*gcc"))
                if cc_candidates:
                    candidates.append((cc_candidates[-1], cxx, env))
    seen: set[tuple[str, str]] = set()
    for cc, cxx, env in candidates:
        key = (str(cc), str(cxx))
        if key in seen:
            continue
        seen.add(key)
        cxx_version = _compiler_version(str(cxx), env)
        if cxx_version and cxx_version >= (9, 0, 0):
            return str(cc), str(cxx), env, ".".join(str(part) for part in cxx_version)
    raise RuntimeError("GCC/G++ 9 or newer was not found in PATH or user conda directories")


def _gromacs_version(command: str) -> str | None:
    result = subprocess.run(
        [command, "--version"],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("GROMACS version:"):
            return line.split(":", 1)[1].strip().split()[0]
    return None


def find_existing_gromacs(required_version: str) -> tuple[Path, str] | None:
    candidates: list[Path] = []
    if os.environ.get("POLYMER_AGENT_GROMACS"):
        candidates.append(Path(os.environ["POLYMER_AGENT_GROMACS"]))
    if shutil.which("gmx"):
        candidates.append(Path(shutil.which("gmx") or "gmx"))
    home = Path.home()
    for root in [
        home / "tools" / "build",
        home / ".local",
        Path("/public/software/apps/gromacs"),
        Path("/opt"),
        Path("/usr/local"),
    ]:
        if root.exists():
            candidates.extend(root.glob("**/bin/gmx"))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        version = _gromacs_version(key)
        if version == required_version:
            return candidate, version
    return None


def _python_runtime_version(command: str, env: dict[str, str] | None = None) -> tuple[int, int, int] | None:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        [command, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        text=True,
        capture_output=True,
        timeout=30,
        env=full_env,
        check=False,
    )
    if result.returncode:
        return None
    return _version_tuple(result.stdout.strip())


def find_science_python() -> tuple[str, dict[str, str], str]:
    candidates: list[tuple[Path, dict[str, str]]] = []
    for name in ["python3.12", "python3.11", "python3", "python"]:
        if shutil.which(name):
            candidates.append((Path(shutil.which(name) or name), {}))
    home = Path.home()
    for root in [home / "miniforge3", home / "anaconda3", home / ".conda"]:
        if not root.exists():
            continue
        env = _conda_library_env(root)
        for pattern in ["bin/python3.12", "bin/python3.11", "bin/python3.10", "bin/python3.9", "envs/*/bin/python3.12", "envs/*/bin/python3.11", "envs/*/bin/python3.10", "envs/*/bin/python3.9", "pkgs/python-3.12*/bin/python3.12", "pkgs/python-3.11*/bin/python3.11", "pkgs/python-3.10*/bin/python3.10", "pkgs/python-3.9*/bin/python3.9"]:
            for candidate in root.glob(pattern):
                candidates.append((candidate, env))
    seen: set[str] = set()
    for candidate, env in candidates:
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        version = _python_runtime_version(key, env)
        if version and (3, 11) <= version[:2] <= (3, 12):
            return key, env, ".".join(str(part) for part in version)
    raise RuntimeError("Python 3.11-3.12 was not found in PATH or user conda directories")


def prerequisite_report(acceleration: str) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    try:
        find_cmake()
    except Exception as exc:  # noqa: BLE001 - reported as a bounded prerequisite
        actions.append({
            "component": "cmake",
            "action": f"请提供 CMake 3.18 或更高版本，或允许平台在用户目录中安装新版 CMake。当前检测结果：{exc}",
        })
    if shutil.which("make") is None:
        actions.append({"component": "make", "action": "请提供 GNU Make。"})
    try:
        find_cxx17_compilers()
    except Exception as exc:  # noqa: BLE001 - reported as a bounded prerequisite
        actions.append({
            "component": "cxx17_compiler",
            "action": f"请提供 GCC/G++ 9 或更高版本，或允许平台在用户目录中安装新版编译器。当前检测结果：{exc}",
        })
    if acceleration == "cuda" and shutil.which("nvcc") is None:
        actions.append({
            "component": "cuda_toolkit",
            "action": "已选择 CUDA 构建，但当前登录环境找不到 nvcc；请加载与计算节点驱动兼容的 CUDA module。",
        })
    try:
        find_science_python()
    except Exception as exc:  # noqa: BLE001 - reported as a bounded prerequisite
        actions.append({
            "component": "python",
            "action": f"请提供 Python 3.9-3.12，或允许平台在用户目录中安装兼容 Python。当前检测结果：{exc}",
        })
    return actions


def install_emc(root: Path, cache: Path, status: Status) -> tuple[Path, dict[str, Any]]:
    machine = platform.machine().lower()
    machine = "x86_64" if machine in {"amd64", "x86_64"} else machine
    binary_name = EMC_BINARIES.get(machine)
    if binary_name is None:
        raise RuntimeError(f"unsupported EMC architecture: {machine}")
    distribution = EMC_DISTRIBUTIONS[machine]
    skipped_external_links: list[str] = []
    prefix = root / "emc" / EMC_VERSION
    binary = prefix / "bin" / binary_name
    if not binary.is_file():
        status.update(stage="emc_download", percent=10, message=f"正在下载 EMC {EMC_VERSION}")
        archive = cache / str(distribution["filename"])
        download(str(distribution["url"]), archive, str(distribution["sha256"]))
        status.update(stage="emc_install", percent=20, message=f"正在安装 EMC {EMC_VERSION}")
        with tempfile.TemporaryDirectory(dir=str(root)) as temporary:
            temporary_root = Path(temporary).resolve()
            with tarfile.open(archive, "r:gz") as bundle:
                safe_members: list[tarfile.TarInfo] = []
                for member in bundle.getmembers():
                    member_path = (temporary_root / member.name).resolve()
                    if os.path.commonpath([str(temporary_root), str(member_path)]) != str(temporary_root):
                        raise RuntimeError(f"EMC archive member escapes destination: {member.name}")
                    if member.issym():
                        link_path = (member_path.parent / member.linkname).resolve()
                        if os.path.commonpath([str(temporary_root), str(link_path)]) != str(temporary_root):
                            skipped_external_links.append(member.name)
                            continue
                    elif member.islnk():
                        link_path = (temporary_root / member.linkname).resolve()
                        if os.path.commonpath([str(temporary_root), str(link_path)]) != str(temporary_root):
                            skipped_external_links.append(member.name)
                            continue
                    safe_members.append(member)
                bundle.extractall(temporary_root, members=safe_members)
            extracted = temporary_root / f"v{EMC_VERSION}"
            if not extracted.is_dir():
                raise RuntimeError("EMC archive has an unexpected layout")
            prefix.parent.mkdir(parents=True, exist_ok=True)
            if prefix.exists():
                shutil.rmtree(prefix)
            shutil.move(str(extracted), str(prefix))
        binary.chmod(0o755)
    wrapper = root / "bin" / "emc"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"export EMC_ROOT={json.dumps(str(prefix))}\n"
        f"if [[ \"${{1:-}}\" == \"--version\" ]]; then echo \"EMC {EMC_VERSION}\"; exit 0; fi\n"
        f"if [[ \"${{1:-}}\" == \"--root\" ]]; then echo {json.dumps(str(prefix))}; exit 0; fi\n"
        f"exec {json.dumps(str(binary))} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper, {
        "version": EMC_VERSION,
        "distribution": f"{distribution['name']}=={distribution['version']}",
        "source_url": distribution["url"],
        "source_sha256": distribution["sha256"],
        "architecture": machine,
        "skipped_external_links": skipped_external_links,
    }


def install_gromacs(root: Path, cache: Path, status: Status, acceleration: str, jobs: int) -> tuple[Path, dict[str, Any]]:
    prefix = root / "gromacs" / GROMACS_VERSION
    binary = prefix / "bin" / "gmx"
    resolved_acceleration = acceleration
    if acceleration == "auto":
        resolved_acceleration = "cuda" if shutil.which("nvcc") else "cpu"
    if not binary.is_file():
        status.update(stage="gromacs_download", percent=28, message=f"正在下载 GROMACS {GROMACS_VERSION}")
        archive = cache / f"gromacs-{GROMACS_VERSION}.tar.gz"
        download(GROMACS_URL, archive, GROMACS_SHA256)
        source = root / "src" / f"gromacs-{GROMACS_VERSION}"
        build = root / "build" / f"gromacs-{GROMACS_VERSION}-{resolved_acceleration}"
        source_marker = source / "src" / "gromacs" / "hardware" / "cpuinfo.cpp"
        if not source_marker.is_file():
            if source.exists():
                shutil.rmtree(source)
            source.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as bundle:
                safe_extract(bundle, source.parent)
        build.mkdir(parents=True, exist_ok=True)
        status.update(stage="gromacs_configure", percent=40, message=f"正在配置 GROMACS {GROMACS_VERSION} ({resolved_acceleration})")
        configure = [
            "cmake",
            "-S", str(source),
            "-B", str(build),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DGMX_BUILD_OWN_FFTW=ON",
            "-DGMX_MPI=OFF",
            "-DGMX_OPENMP=ON",
            "-DGMX_GPU=" + ("CUDA" if resolved_acceleration == "cuda" else "OFF"),
            "-DREGRESSIONTEST_DOWNLOAD=ON",
        ]
        run(configure)
        status.update(stage="gromacs_build", percent=52, message=f"正在编译 GROMACS {GROMACS_VERSION}")
        run(["cmake", "--build", str(build), "--parallel", str(jobs)], timeout=14400)
        status.update(stage="gromacs_test", percent=72, message="正在运行 GROMACS 回归测试")
        run(["cmake", "--build", str(build), "--target", "check", "--parallel", str(jobs)], timeout=14400)
        status.update(stage="gromacs_install", percent=84, message=f"正在安装 GROMACS {GROMACS_VERSION}")
        run(["cmake", "--install", str(build)], timeout=1800)
    wrapper = root / "bin" / "gmx"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "set +u\n"
        f"source {json.dumps(str(prefix / 'bin' / 'GMXRC'))}\n"
        "set -u\n"
        f"exec {json.dumps(str(binary))} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    run([str(wrapper), "--version"], timeout=120)
    return wrapper, {
        "version": GROMACS_VERSION,
        "source_url": GROMACS_URL,
        "source_sha256": GROMACS_SHA256,
        "acceleration": resolved_acceleration,
    }


def install_gromacs(root: Path, cache: Path, status: Status, acceleration: str, jobs: int) -> tuple[Path, dict[str, Any]]:
    prefix = root / "gromacs" / GROMACS_VERSION
    binary = prefix / "bin" / "gmx"
    resolved_acceleration = acceleration
    if acceleration == "auto":
        resolved_acceleration = "cuda" if shutil.which("nvcc") else "cpu"
    cmake_version = None
    compiler_version = None
    existing = find_existing_gromacs(GROMACS_VERSION)
    if existing is not None:
        binary, existing_version = existing
        status.update(stage="gromacs_existing", percent=84, message=f"已发现可用 GROMACS {existing_version}")
        wrapper = root / "bin" / "gmx"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"exec {json.dumps(str(binary))} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        run([str(wrapper), "--version"], timeout=120)
        return wrapper, {
            "version": existing_version,
            "source": "existing_compatible_installation",
            "binary": str(binary),
            "acceleration": resolved_acceleration,
        }
    if not binary.is_file():
        cmake, cmake_env, cmake_version = find_cmake()
        cc, cxx, compiler_env, compiler_version = find_cxx17_compilers()
        build_env = dict(cmake_env)
        build_env.update(compiler_env)
        build_env["CC"] = cc
        build_env["CXX"] = cxx
        status.update(stage="gromacs_download", percent=28, message=f"正在准备 GROMACS {GROMACS_VERSION} 源码包")
        archive = cache / f"gromacs-{GROMACS_VERSION}.tar.gz"
        download(GROMACS_URL, archive, GROMACS_SHA256)
        source = root / "src" / f"gromacs-{GROMACS_VERSION}"
        build = root / "build" / f"gromacs-{GROMACS_VERSION}-{resolved_acceleration}"
        source_marker = source / "src" / "gromacs" / "hardware" / "cpuinfo.cpp"
        if not source_marker.is_file():
            if source.exists():
                shutil.rmtree(source)
            source.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as bundle:
                safe_extract(bundle, source.parent)
        if build.exists():
            shutil.rmtree(build)
        build.mkdir(parents=True, exist_ok=True)
        status.update(stage="gromacs_configure", percent=40, message=f"正在配置 GROMACS {GROMACS_VERSION} ({resolved_acceleration})")
        configure = [
            cmake,
            "-S", str(source),
            "-B", str(build),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DGMX_BUILD_OWN_FFTW=ON",
            "-DGMX_MPI=OFF",
            "-DGMX_OPENMP=ON",
            "-DGMX_GPU=" + ("CUDA" if resolved_acceleration == "cuda" else "OFF"),
            "-DREGRESSIONTEST_DOWNLOAD=OFF",
        ]
        run(configure, env=build_env)
        status.update(stage="gromacs_build", percent=52, message=f"正在编译 GROMACS {GROMACS_VERSION}")
        run([cmake, "--build", str(build), "--parallel", str(jobs)], timeout=14400, env=build_env)
        status.update(stage="gromacs_install", percent=84, message=f"正在安装 GROMACS {GROMACS_VERSION}")
        run([cmake, "--install", str(build)], timeout=1800, env=build_env)
    wrapper = root / "bin" / "gmx"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "set +u\n"
        f"source {json.dumps(str(prefix / 'bin' / 'GMXRC'))}\n"
        "set -u\n"
        f"exec {json.dumps(str(binary))} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    run([str(wrapper), "--version"], timeout=120)
    return wrapper, {
        "version": GROMACS_VERSION,
        "source_url": GROMACS_URL,
        "source_sha256": GROMACS_SHA256,
        "acceleration": resolved_acceleration,
        "cmake_version": cmake_version,
        "compiler_version": compiler_version,
        "regression_tests": "not_run_in_user_level_auto_install",
    }


def install_python_runtime(root: Path, status: Status) -> tuple[Path, dict[str, Any]]:
    source_python, python_env, python_version = find_science_python()
    environment = root / "python" / "venv"
    python = environment / "bin" / "python"
    if not python.is_file():
        status.update(stage="python_environment", percent=86, message="Creating the pinned scientific Python environment")
        run([source_python, "-m", "venv", str(environment)], timeout=600, env=python_env)
    status.update(stage="python_packages", percent=90, message="Installing pinned Python analysis packages")
    run([str(python), "-m", "pip", "install", "--disable-pip-version-check", *PYTHON_PACKAGES], timeout=1800, env=python_env)
    wrapper = root / "bin" / "python"
    exports = ""
    if python_env.get("LD_LIBRARY_PATH"):
        exports = f"export LD_LIBRARY_PATH={json.dumps(python_env['LD_LIBRARY_PATH'])}:\"${{LD_LIBRARY_PATH:-}}\"\n"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"{exports}"
        f"exec {json.dumps(str(python))} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    run([str(wrapper), "--version"], timeout=120)
    return wrapper, {"version": python_version, "packages": list(PYTHON_PACKAGES), "source_python": source_python}


def install(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    status = Status(Path(args.status).resolve())
    root.mkdir(parents=True, exist_ok=True)
    try:
        gromacs_present = (root / "gromacs" / GROMACS_VERSION / "bin" / "gmx").is_file()
        actions = [] if gromacs_present else prerequisite_report(args.acceleration)
        if actions:
            status.update(status="needs_admin", stage="preflight", percent=5, message="缺少编译依赖", admin_actions=actions)
            return 2
        cache = root / "cache"
        emc, emc_manifest = install_emc(root, cache, status)
        gmx, gromacs_manifest = install_gromacs(root, cache, status, args.acceleration, args.jobs)
        python, python_manifest = install_python_runtime(root, status)
        manifest = {
            "schema_version": "polymer_agent.science_runtime.v1",
            "profile": "pelcb-paper-2024.4",
            "installed_at": utc_timestamp(),
            "gromacs": gromacs_manifest,
            "emc": emc_manifest,
            "python": python_manifest,
            "commands": {"gromacs": str(gmx), "emc": str(emc), "python": str(python)},
        }
        manifest_path = root / "runtime_manifest.json"
        atomic_json(manifest_path, manifest)
        status.update(
            status="complete",
            stage="complete",
            percent=100,
            message="固定版本科学环境安装完成",
            admin_actions=[],
            runtime_manifest=str(manifest_path),
            commands=manifest["commands"],
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - converted to a bounded public status
        status.update(
            status="failed",
            stage="failed",
            message="科学环境安装失败",
            error_type=type(exc).__name__,
            error_summary=str(exc)[-1200:],
        )
        return 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--root", required=True)
    root.add_argument("--status", required=True)
    root.add_argument("--acceleration", choices=["auto", "cpu", "cuda"], default="auto")
    root.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 8))
    return root


if __name__ == "__main__":
    raise SystemExit(install(parser().parse_args()))
