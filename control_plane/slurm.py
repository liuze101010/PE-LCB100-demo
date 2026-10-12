from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import secrets
import shlex
import socket
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .security import decrypt_secret


def fingerprint_sha256(key: Any) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


class PinnedHostKeyPolicy:
    def __init__(self, expected: str) -> None:
        self.expected = expected.rstrip("=")

    def missing_host_key(self, client: Any, hostname: str, key: Any) -> None:
        actual = fingerprint_sha256(key)
        if not hmac_compare(actual, self.expected):
            raise RuntimeError(f"SSH host key mismatch for {hostname}")
        client.get_host_keys().add(hostname, key.get_name(), key)


def hmac_compare(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left.rstrip("="), right.rstrip("="))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_host_key_sha256(host: str, port: int = 22, timeout: int = 10) -> str:
    import paramiko

    sock = socket.create_connection((host, port), timeout=timeout)
    transport = paramiko.Transport(sock)
    try:
        transport.start_client(timeout=timeout)
        return fingerprint_sha256(transport.get_remote_server_key())
    finally:
        transport.close()


def install_public_key_with_password(
    *,
    host: str,
    port: int,
    username: str,
    host_key_sha256: str,
    password: str | None,
    public_key: str,
    timeout: int = 20,
) -> dict[str, Any]:
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedHostKeyPolicy(host_key_sha256))
    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            look_for_keys=password is None,
            allow_agent=password is None,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
        )
        key = shlex.quote(public_key.strip())
        command = (
            "set -eu; "
            "umask 077; "
            "mkdir -p ~/.ssh; "
            "touch ~/.ssh/authorized_keys; "
            f"grep -qxF {key} ~/.ssh/authorized_keys || printf '%s\\n' {key} >> ~/.ssh/authorized_keys; "
            "chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys"
        )
        raw = SSHSlurmBackend._run(client, command, timeout=timeout)
        combined = ((raw.get("stdout") or "") + (raw.get("stderr") or "")).encode()
        return {
            "ok": bool(raw["ok"]),
            "exit_code": raw["exit_code"],
            "output_sha256": hashlib.sha256(combined).hexdigest(),
        }
    finally:
        client.close()


def safe_remote_path(root: str, *parts: str) -> str:
    root_path = PurePosixPath(root)
    if not root_path.is_absolute() or ".." in root_path.parts:
        raise ValueError("remote_root must be an absolute normalized path")
    clean_parts: list[str] = []
    for part in parts:
        token = str(part)
        if not token or "/" in token or "\\" in token or token in {".", ".."}:
            raise ValueError("remote path component is unsafe")
        clean_parts.append(token)
    return str(root_path.joinpath(*clean_parts))


@dataclass(frozen=True)
class ClusterRuntime:
    host: str
    port: int
    username: str
    host_key_sha256: str
    private_key_pem: str
    partition: str
    account: str | None
    remote_root: str
    gromacs_command: str
    emc_command: str
    python_command: str
    runner_command: str
    module_init: list[str]
    limits: dict[str, Any]
    environment: dict[str, Any]


def runtime_from_model(cluster: Any, credential_key: str) -> ClusterRuntime:
    limits = json.loads(cluster.resource_limits_json)
    return ClusterRuntime(
        host=cluster.host,
        port=cluster.port,
        username=cluster.remote_username,
        host_key_sha256=cluster.host_key_sha256,
        private_key_pem=decrypt_secret(cluster.encrypted_private_key, credential_key),
        partition=cluster.partition,
        account=cluster.account,
        remote_root=cluster.remote_root,
        gromacs_command=cluster.gromacs_command,
        emc_command=cluster.emc_command,
        python_command=cluster.python_command,
        runner_command=cluster.runner_command,
        module_init=json.loads(cluster.module_init_json),
        limits=limits,
        environment=dict(limits.get("environment") or {"mode": "module", "pipeline_command": "pelcb-pipeline-core"}),
    )


class SSHSlurmBackend:
    def __init__(self, runtime: ClusterRuntime) -> None:
        self.runtime = runtime

    def _connect(self):
        import paramiko

        key = paramiko.Ed25519Key.from_private_key(io.StringIO(self.runtime.private_key_pem))
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(PinnedHostKeyPolicy(self.runtime.host_key_sha256))
        client.connect(
            hostname=self.runtime.host,
            port=self.runtime.port,
            username=self.runtime.username,
            pkey=key,
            look_for_keys=False,
            allow_agent=False,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
        )
        return client

    def _environment_command(self, command: list[str]) -> str:
        mode = str(self.runtime.environment.get("mode") or "module")
        if mode == "module":
            return " ".join(shlex.quote(item) for item in command)
        if mode != "container":
            raise ValueError("unsupported cluster environment mode")
        runtime = str(self.runtime.environment.get("container_runtime") or "")
        image = str(self.runtime.environment.get("container_image") or "")
        if runtime not in {"apptainer", "singularity"}:
            raise ValueError("container runtime must be apptainer or singularity")
        if not PurePosixPath(image).is_absolute() or ".." in PurePosixPath(image).parts:
            raise ValueError("container image must be an absolute normalized path")
        return " ".join(shlex.quote(item) for item in [runtime, "exec", image, *command])

    def detect_site_defaults(self) -> dict[str, Any]:
        """Select a writable user workspace and an available Slurm partition."""
        command = (
            "printf 'HOME=%s\\n' \"$HOME\"; "
            "if command -v sinfo >/dev/null 2>&1; then "
            "printf 'SCHEDULER=slurm\\n'; sinfo -h -o '%P|%G|%a' | sort -u; "
            "else printf 'SCHEDULER=manual\\n'; getent hosts; fi"
        )
        with self._connect() as client:
            raw = self._run(client, command)
        if not raw.get("ok"):
            raise RuntimeError("site defaults could not be detected")

        home = ""
        scheduler_mode = ""
        partitions: list[dict[str, Any]] = []
        manual_nodes: set[str] = set()
        for line in str(raw.get("stdout") or "").splitlines():
            if line.startswith("HOME="):
                home = line.removeprefix("HOME=").strip()
                continue
            if line.startswith("SCHEDULER="):
                scheduler_mode = line.removeprefix("SCHEDULER=").strip()
                continue
            host_match = re.fullmatch(r"\S+\s+(node-gpu\d+)", line.strip())
            if host_match:
                manual_nodes.add(host_match.group(1))
                continue
            fields = [item.strip() for item in line.split("|", 2)]
            if len(fields) != 3 or not fields[0]:
                continue
            raw_name, gres, availability = fields
            is_default = raw_name.endswith("*")
            name = raw_name.rstrip("*")
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", name):
                continue
            has_gpu = bool(gres and gres.lower() not in {"(null)", "n/a", "none"} and "gpu" in gres.lower())
            is_available = availability.lower() in {"up", "active"}
            partitions.append(
                {
                    "name": name,
                    "default": is_default,
                    "has_gpu": has_gpu,
                    "available": is_available,
                }
            )

        if not home or not PurePosixPath(home).is_absolute() or ".." in PurePosixPath(home).parts:
            raise RuntimeError("remote home directory could not be detected")
        if scheduler_mode == "manual":
            if not manual_nodes:
                raise RuntimeError("no manual GPU nodes could be discovered")
            candidates = sorted(manual_nodes, key=lambda name: int(re.search(r"\d+$", name).group()))
            probe_command = "for node in " + " ".join(shlex.quote(node) for node in candidates) + "; do " + (
                "(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=5 \"$node\" true "
                "&& printf 'NODE=%s\\n' \"$node\") & done; wait"
            )
            with self._connect() as client:
                node_probe = self._run(client, probe_command, timeout=15)
            reachable = {
                line.removeprefix("NODE=").strip()
                for line in str(node_probe.get("stdout") or "").splitlines()
                if line.startswith("NODE=")
            }
            candidates = [node for node in candidates if node in reachable]
            if not candidates:
                raise RuntimeError("no manual GPU node is reachable")
            return {
                "scheduler_mode": "manual",
                "partition": "manual",
                "remote_root": str(PurePosixPath(home)),
                "available_partitions": [],
                "manual_nodes": candidates,
            }

        available = [item for item in partitions if item["available"]] or partitions
        if not available:
            raise RuntimeError("no Slurm partition is available")
        selected = max(available, key=lambda item: (item["has_gpu"], item["default"], item["name"]))
        return {
            "scheduler_mode": "slurm",
            "partition": selected["name"],
            "remote_root": str(PurePosixPath(home)),
            "available_partitions": partitions,
            "manual_nodes": [],
        }

    def _scheduler_probe_command(self) -> str:
        if self.runtime.environment.get("scheduler_mode") != "manual":
            return "command -v sbatch && command -v squeue && command -v sacct && command -v scancel"
        nodes = self._manual_nodes()
        checks = [
            f"ssh -o BatchMode=yes -o ConnectTimeout=5 {shlex.quote(node)} true"
            for node in nodes
        ]
        return " && ".join(checks)

    def _manual_nodes(self) -> list[str]:
        nodes = list(self.runtime.environment.get("manual_nodes") or [])
        if not nodes or any(not re.fullmatch(r"node-gpu\d+", str(node)) for node in nodes):
            raise ValueError("manual scheduler requires validated node-gpu hosts")
        return [str(node) for node in nodes]

    @staticmethod
    def _public_probe(raw: dict[str, Any]) -> dict[str, Any]:
        combined = ((raw.get("stdout") or "") + (raw.get("stderr") or "")).encode()
        return {
            "ok": bool(raw.get("ok")),
            "exit_code": raw.get("exit_code"),
            "output_sha256": hashlib.sha256(combined).hexdigest(),
        }

    @staticmethod
    def _run(client: Any, command: str, timeout: int = 60) -> dict[str, Any]:
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        return {
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout.read().decode(errors="replace")[-4000:],
            "stderr": stderr.read().decode(errors="replace")[-4000:],
        }

    def preflight(self) -> dict[str, Any]:
        init = " && ".join(self.runtime.module_init)
        prefix = f"{init} && " if init else ""
        root = shlex.quote(self.runtime.remote_root)
        probes = {
            "slurm": self._scheduler_probe_command(),
            "gromacs": self._environment_command([self.runtime.gromacs_command, "--version"]),
            "emc": self._environment_command([self.runtime.emc_command, "--version"]),
            "python": f"{shlex.quote(self.runtime.python_command)} --version",
            "runner": f"{shlex.quote(self.runtime.runner_command)} self-test --json",
            "workspace": f"mkdir -p {root} && test -w {root}",
        }
        results: dict[str, Any] = {}
        with self._connect() as client:
            for name, command in probes.items():
                raw = self._run(client, "set -eu; " + prefix + command)
                combined = ((raw.get("stdout") or "") + (raw.get("stderr") or "")).encode()
                results[name] = {
                    "ok": raw["ok"],
                    "exit_code": raw["exit_code"],
                    "output_sha256": hashlib.sha256(combined).hexdigest(),
                }
        messages = {
            "slurm": "请由集群管理员提供可用的 Slurm 客户端命令和账户权限。",
            "gromacs": "GROMACS 命令不可用或版本自检失败。",
            "emc": "EMC 命令不可用或版本自检失败。",
            "python": "远程 Python 3 不可用。",
            "runner": "平台 runner 或完整 PE-LCB 生产 worker 自检失败；短程验收不能替代 100-case 生产管线。",
            "workspace": "远程工作根目录不可写。",
        }
        admin_actions = [
            {"component": name, "action": messages[name]}
            for name, result in results.items()
            if not result["ok"]
        ]
        return {"ok": not admin_actions, "checks": results, "admin_actions": admin_actions}

    def discover_environment(self) -> dict[str, Any]:
        init = " && ".join(self.runtime.module_init)
        prefix = f"{init} && " if init else ""
        root = shlex.quote(self.runtime.remote_root)
        pipeline = str(self.runtime.environment.get("pipeline_command") or "pelcb-pipeline-core")
        probes = {
            "slurm": self._scheduler_probe_command(),
            "python": f"{shlex.quote(self.runtime.python_command)} --version",
            "workspace": f"mkdir -p {root} && test -w {root}",
            "gromacs": self._environment_command([self.runtime.gromacs_command, "--version"]),
            "emc": self._environment_command([self.runtime.emc_command, "--version"]),
            "science_pipeline": self._environment_command([pipeline, "self-test", "--json"]),
        }
        if self.runtime.environment.get("mode") == "container":
            container_runtime = str(self.runtime.environment.get("container_runtime") or "")
            container_image = str(self.runtime.environment.get("container_image") or "")
            probes["container_runtime"] = f"command -v {shlex.quote(container_runtime)} && {shlex.quote(container_runtime)} --version"
            probes["container_image"] = f"test -r {shlex.quote(container_image)}"
        results: dict[str, Any] = {}
        with self._connect() as client:
            for name, command in probes.items():
                results[name] = self._public_probe(self._run(client, "set -eu; " + prefix + command))
            runner_raw = self._run(
                client,
                "set -eu; " + prefix + f"{shlex.quote(self.runtime.runner_command)} self-test --json",
            )
            results["platform_runner"] = self._public_probe(runner_raw)
        managed = self.runtime.environment.get("science_install_mode") == "managed"
        required_before_install = ["slurm", "python", "workspace"]
        if not managed:
            required_before_install.extend(["gromacs", "emc", "science_pipeline"])
        if self.runtime.environment.get("mode") == "container":
            required_before_install.extend(["container_runtime", "container_image"])
        admin_actions: list[dict[str, str]] = []
        messages = {
            "slurm": "请由集群管理员提供 sbatch、squeue、sacct 和 scancel。",
            "python": "请提供可运行 Python 3 的登录环境。",
            "workspace": "请为所选远程工作目录授予当前账号写权限。",
            "gromacs": "请加载或提供与项目 profile 兼容的 GROMACS。",
            "emc": "请加载 EMC 9.4.4，或改用平台固定版本的用户态自动安装。",
            "science_pipeline": "所选 module/container 中缺少 pelcb-pipeline-core 或其自检未通过。",
            "container_runtime": "请安装或加载 Apptainer/Singularity。",
            "container_image": "请将经过审核的计算镜像放到所填绝对路径并授予读取权限。",
        }
        for name in required_before_install:
            if not results.get(name, {}).get("ok"):
                admin_actions.append({"component": name, "action": messages[name]})
        installable = results["python"]["ok"] and results["workspace"]["ok"]
        ready_for_install = installable and not admin_actions
        platform_actions: list[str] = []
        if not results["platform_runner"]["ok"]:
            platform_actions.extend(["install_runner", "install_site_adapter"])
        if managed and (not results["gromacs"]["ok"] or not results["emc"]["ok"]):
            platform_actions.append("install_pinned_science_runtime")
        return {
            "schema_version": "polymer_agent.environment_discovery.v1",
            "ok": ready_for_install and results["platform_runner"]["ok"],
            "ready_for_platform_install": ready_for_install,
            "platform_installable": installable,
            "environment_mode": self.runtime.environment.get("mode", "module"),
            "science_install_mode": self.runtime.environment.get("science_install_mode", "existing"),
            "checks": results,
            "admin_actions": admin_actions,
            "platform_actions": platform_actions,
        }

    def install_platform_stack(self, *, files: dict[str, bytes], version: str, overwrite: bool = False) -> dict[str, Any]:
        root = PurePosixPath(self.runtime.remote_root)
        install_root = root / ".polymer-agent" / "runtime" / version
        bin_root = root / ".polymer-agent" / "bin"
        config_root = root / ".polymer-agent" / "config"
        runner_path = bin_root / "polymer-agent-remote"
        adapter_path = bin_root / "site-pelcb-pipeline"
        core_path = bin_root / "pelcb-pipeline-core"
        config_path = config_root / "site.json"
        if not overwrite:
            probe_path = str(install_root / "install_manifest.json")
        else:
            probe_path = ""
        environment = dict(self.runtime.environment)
        if environment.get("pipeline_command") in {None, "", "pelcb-pipeline-core"}:
            environment["pipeline_command"] = str(core_path)
        config = {
            "schema_version": "polymer_agent.site_config.v1",
            "environment": environment,
            "gromacs_command": self.runtime.gromacs_command,
            "emc_command": self.runtime.emc_command,
        }
        manifest = {
            "schema_version": "polymer_agent.remote_install.v1",
            "version": version,
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
            "environment_mode": environment.get("mode", "module"),
        }
        with self._connect() as client:
            if probe_path:
                existing = self._run(client, f"test -f {shlex.quote(probe_path)}")
                if existing["ok"]:
                    return {
                        "installed": False,
                        "already_present": True,
                        "version": version,
                        "runner_command": str(runner_path),
                        "install_root": str(install_root),
                    }
            mkdir = self._run(
                client,
                "mkdir -p " + " ".join(
                    shlex.quote(str(path))
                    for path in [
                        install_root,
                        install_root / "mdp",
                        install_root / "fixture",
                        bin_root,
                        config_root,
                    ]
                ),
            )
            if not mkdir["ok"]:
                raise RuntimeError("could not create remote platform directories")
            sftp = client.open_sftp()
            try:
                for name, content in files.items():
                    destination = install_root / name
                    with sftp.file(str(destination), "wb") as handle:
                        handle.write(content)
                    sftp.chmod(str(destination), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                extract = self._run(
                    client,
                    f"{shlex.quote(self.runtime.python_command)} -m zipfile -e "
                    f"{shlex.quote(str(install_root / 'science_core_bundle.zip'))} "
                    f"{shlex.quote(str(install_root))}",
                )
                if not extract["ok"]:
                    raise RuntimeError("could not extract the versioned science core bundle")
                verify_bundle = self._run(client, f"test -f {shlex.quote(str(install_root / 'science_core' / 'manifest.json'))}")
                if not verify_bundle["ok"]:
                    raise RuntimeError("science core manifest is missing after extraction")
                with sftp.file(str(config_path), "wb") as handle:
                    handle.write((json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode())
                runner_wrapper = (
                    "#!/usr/bin/env bash\nset -euo pipefail\n"
                    f"MANAGED_GMX={shlex.quote(str(root / '.polymer-agent' / 'science' / 'bin' / 'gmx'))}\n"
                    f"MANAGED_EMC={shlex.quote(str(root / '.polymer-agent' / 'science' / 'bin' / 'emc'))}\n"
                    f"MANAGED_PYTHON={shlex.quote(str(root / '.polymer-agent' / 'science' / 'bin' / 'python'))}\n"
                    f"if [ -x \"$MANAGED_GMX\" ]; then export POLYMER_AGENT_GROMACS=\"$MANAGED_GMX\"; else export POLYMER_AGENT_GROMACS={shlex.quote(self.runtime.gromacs_command)}; fi\n"
                    f"if [ -x \"$MANAGED_EMC\" ]; then export POLYMER_AGENT_EMC=\"$MANAGED_EMC\"; else export POLYMER_AGENT_EMC={shlex.quote(self.runtime.emc_command)}; fi\n"
                    f"export POLYMER_AGENT_SITE_PIPELINE={shlex.quote(str(adapter_path))}\n"
                    f"if [ -x \"$MANAGED_PYTHON\" ]; then PYTHON_BIN=\"$MANAGED_PYTHON\"; else PYTHON_BIN={shlex.quote(self.runtime.python_command)}; fi\n"
                    f"exec \"$PYTHON_BIN\" {shlex.quote(str(install_root / 'remote_runner.py'))} \"$@\"\n"
                ).encode()
                adapter_wrapper = (
                    "#!/usr/bin/env bash\nset -euo pipefail\n"
                    f"export POLYMER_AGENT_SITE_CONFIG={shlex.quote(str(config_path))}\n"
                    f"exec {shlex.quote(self.runtime.python_command)} {shlex.quote(str(install_root / 'site_pipeline_adapter.py'))} \"$@\"\n"
                ).encode()
                core_wrapper = (
                    "#!/usr/bin/env bash\nset -euo pipefail\n"
                    f"MANAGED_GMX={shlex.quote(str(root / '.polymer-agent' / 'science' / 'bin' / 'gmx'))}\n"
                    f"MANAGED_EMC={shlex.quote(str(root / '.polymer-agent' / 'science' / 'bin' / 'emc'))}\n"
                    f"if [ -x \"$MANAGED_GMX\" ]; then export POLYMER_AGENT_GROMACS=\"$MANAGED_GMX\"; else export POLYMER_AGENT_GROMACS={shlex.quote(self.runtime.gromacs_command)}; fi\n"
                    f"if [ -x \"$MANAGED_EMC\" ]; then export POLYMER_AGENT_EMC=\"$MANAGED_EMC\"; else export POLYMER_AGENT_EMC={shlex.quote(self.runtime.emc_command)}; fi\n"
                    f"export POLYMER_AGENT_SCIENCE_ROOT={shlex.quote(str(install_root / 'science_core'))}\n"
                    f"MANAGED_PYTHON={shlex.quote(str(root / '.polymer-agent' / 'science' / 'bin' / 'python'))}\n"
                    f"if [ -x \"$MANAGED_PYTHON\" ]; then PYTHON_BIN=\"$MANAGED_PYTHON\"; else PYTHON_BIN={shlex.quote(self.runtime.python_command)}; fi\n"
                    f"exec \"$PYTHON_BIN\" {shlex.quote(str(install_root / 'pelcb_pipeline_core.py'))} \"$@\"\n"
                ).encode()
                for path, content in [
                    (runner_path, runner_wrapper),
                    (adapter_path, adapter_wrapper),
                    (core_path, core_wrapper),
                ]:
                    with sftp.file(str(path), "wb") as handle:
                        handle.write(content)
                    sftp.chmod(str(path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                with sftp.file(str(install_root / "install_manifest.json"), "wb") as handle:
                    handle.write((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
            finally:
                sftp.close()
        return {
            "installed": True,
            "already_present": False,
            "version": version,
            "runner_command": str(runner_path),
            "site_adapter": str(adapter_path),
            "pipeline_core": str(core_path),
            "install_root": str(install_root),
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        }

    def seed_science_cache(self, *, archives: dict[str, Any]) -> dict[str, Any]:
        """Upload pinned upstream archives into the user-owned remote science cache."""
        science_root = PurePosixPath(self.runtime.remote_root) / ".polymer-agent" / "science"
        cache_root = science_root / "cache"
        uploaded: dict[str, dict[str, Any]] = {}
        with self._connect() as client:
            mkdir = self._run(client, f"mkdir -p {shlex.quote(str(cache_root))}")
            if not mkdir["ok"]:
                raise RuntimeError("could not create remote science cache")
            sftp = client.open_sftp()
            try:
                for name, local_path in archives.items():
                    if "/" in name or "\\" in name or name in {".", ".."}:
                        raise ValueError("unsafe science archive filename")
                    path = Path(local_path)
                    remote_path = cache_root / name
                    sftp.put(str(path), str(remote_path))
                    sftp.chmod(str(remote_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                    uploaded[name] = {
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                        "remote_path": str(remote_path),
                    }
            finally:
                sftp.close()
        return {
            "schema_version": "polymer_agent.science_cache_seed.v1",
            "uploaded": uploaded,
        }

    def science_runtime_status(self) -> dict[str, Any]:
        science_root = PurePosixPath(self.runtime.remote_root) / ".polymer-agent" / "science"
        status_path = science_root / "install_status.json"
        with self._connect() as client:
            result = self._run(client, f"test -f {shlex.quote(str(status_path))} && cat {shlex.quote(str(status_path))}")
        if not result["ok"]:
            return {
                "schema_version": "polymer_agent.science_runtime_install.v1",
                "status": "not_started",
                "percent": 0,
                "stage": "not_started",
            }
        try:
            payload = json.loads(result["stdout"])
        except json.JSONDecodeError as exc:
            raise RuntimeError("remote science-runtime status is not valid JSON") from exc
        if payload.get("schema_version") != "polymer_agent.science_runtime_install.v1":
            raise RuntimeError("remote science-runtime status schema is unsupported")
        return payload

    def start_managed_science_install(self, *, install_root: str) -> dict[str, Any]:
        if self.runtime.environment.get("science_install_mode") != "managed":
            return {"status": "not_requested", "percent": 100, "stage": "existing_environment"}
        if not self.runtime.environment.get("upstream_terms_accepted"):
            raise RuntimeError("upstream terms must be accepted before managed installation")
        runtime_root = PurePosixPath(install_root)
        installer = runtime_root / "install_science_runtime.py"
        science_root = PurePosixPath(self.runtime.remote_root) / ".polymer-agent" / "science"
        status_path = science_root / "install_status.json"
        log_path = science_root / "install.log"
        acceleration = str(self.runtime.environment.get("gromacs_acceleration") or "auto")
        if acceleration not in {"auto", "cpu", "cuda"}:
            raise ValueError("unsupported GROMACS acceleration mode")
        existing = self.science_runtime_status()
        if existing.get("status") == "complete":
            return existing
        if existing.get("status") in {"starting", "running"}:
            with self._connect() as client:
                live = self._run(client, "pgrep -f '[i]nstall_science_runtime.py' >/dev/null 2>&1", timeout=10)
            if live["ok"]:
                return existing
        jobs = max(1, min(int(self.runtime.limits.get("max_cpus", 8)), 8))
        init = " && ".join(self.runtime.module_init)
        prefix = f"{init} && " if init else ""
        install_args = [
            self.runtime.python_command,
            str(installer),
            "--root",
            str(science_root),
            "--status",
            str(status_path),
            "--acceleration",
            acceleration,
            "--jobs",
            str(jobs),
        ]
        spawn_code = (
            "import os, subprocess\n"
            f"os.makedirs({json.dumps(str(science_root))}, exist_ok=True)\n"
            f"log = open({json.dumps(str(log_path))}, 'ab', buffering=0)\n"
            "devnull = open(os.devnull, 'rb')\n"
            f"proc = subprocess.Popen({json.dumps(install_args)}, stdin=devnull, stdout=log, "
            "stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)\n"
            "print(proc.pid)\n"
        )
        command = f"{shlex.quote(self.runtime.python_command)} -c {shlex.quote(spawn_code)}"
        with self._connect() as client:
            result = self._run(client, "set -eu; " + prefix + command, timeout=30)
        if not result["ok"] or not result["stdout"].strip().splitlines()[-1].isdigit():
            raise RuntimeError("could not start managed science-runtime installation")
        return {
            "schema_version": "polymer_agent.science_runtime_install.v1",
            "status": "running",
            "stage": "starting",
            "percent": 1,
            "pid": result["stdout"].strip().splitlines()[-1],
            "commands": {
                "gromacs": str(science_root / "bin" / "gmx"),
                "emc": str(science_root / "bin" / "emc"),
            },
        }

    def render_script(self, *, run_id: str, command: list[str], cpus: int, gpus: int, walltime_hours: int) -> str:
        limits = self.runtime.limits
        if cpus > int(limits.get("max_cpus", 1)):
            raise ValueError("requested CPUs exceed cluster profile")
        if gpus > int(limits.get("max_gpus", 0)):
            raise ValueError("requested GPUs exceed cluster profile")
        if walltime_hours > int(limits.get("max_walltime_hours", 1)):
            raise ValueError("requested walltime exceeds cluster profile")
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"#SBATCH --job-name=pelcb-{run_id[:8]}",
            f"#SBATCH --partition={self.runtime.partition}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --time={walltime_hours}:00:00",
            "#SBATCH --output=slurm-%j.out",
        ]
        if self.runtime.account:
            lines.append(f"#SBATCH --account={self.runtime.account}")
        if gpus:
            lines.append(f"#SBATCH --gres=gpu:{gpus}")
        allowed_nodes = list(limits.get("allowed_nodes") or [])
        if allowed_nodes:
            lines.append("#SBATCH --nodelist=" + ",".join(allowed_nodes))
        lines.extend(self.runtime.module_init)
        lines.append(" ".join(shlex.quote(item) for item in command))
        return "\n".join(lines) + "\n"

    def render_array_script(
        self,
        *,
        run_id: str,
        stage: str,
        command: list[str],
        case_count: int,
        max_parallel: int,
        cpus: int,
        gpus: int,
        walltime_hours: int,
    ) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", stage):
            raise ValueError("unsafe stage name")
        if not 1 <= case_count <= 1000:
            raise ValueError("case_count must be between 1 and 1000")
        if not 1 <= max_parallel <= case_count:
            raise ValueError("max_parallel must be between 1 and case_count")
        script = self.render_script(
            run_id=run_id,
            command=command,
            cpus=cpus,
            gpus=gpus,
            walltime_hours=walltime_hours,
        )
        lines = script.splitlines()
        insertion = next(index for index, line in enumerate(lines) if line.startswith("#SBATCH --output="))
        lines[insertion] = "#SBATCH --output=slurm-%A_%a.out"
        lines.insert(insertion, f"#SBATCH --array=0-{case_count - 1}%{max_parallel}")
        lines[2] = f"#SBATCH --job-name=pelcb-{stage}-{run_id[:8]}"
        return "\n".join(lines) + "\n"

    def submit_text_bundle(self, *, run_id: str, files: dict[str, bytes], script: str) -> tuple[str, str]:
        remote_dir = safe_remote_path(self.runtime.remote_root, "polymer-agent", run_id)
        with self._connect() as client:
            mkdir = self._run(client, f"mkdir -p {shlex.quote(remote_dir)}")
            if not mkdir["ok"]:
                raise RuntimeError("could not create remote run directory")
            sftp = client.open_sftp()
            try:
                for name, content in {**files, "run.slurm": script.encode()}.items():
                    if "/" in name or "\\" in name or name in {".", ".."}:
                        raise ValueError("unsafe bundle filename")
                    with sftp.file(str(PurePosixPath(remote_dir) / name), "wb") as handle:
                        handle.write(content)
            finally:
                sftp.close()
            result = self._run(client, f"cd {shlex.quote(remote_dir)} && sbatch --parsable run.slurm")
            if not result["ok"]:
                raise RuntimeError("sbatch failed: " + result["stderr"][-500:])
            job_id = result["stdout"].strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError("Slurm returned an invalid job id")
        return job_id, remote_dir

    def submit_stage_bundle(
        self,
        *,
        run_id: str,
        stage: str,
        files: dict[str, bytes],
        script: str,
    ) -> tuple[str, str]:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", stage):
            raise ValueError("unsafe stage name")
        remote_dir = safe_remote_path(self.runtime.remote_root, "polymer-agent", run_id)
        script_name = f"stage-{stage}.slurm"
        if self.runtime.environment.get("scheduler_mode") == "manual":
            return self._submit_manual_stage_bundle(
                run_id=run_id,
                stage=stage,
                files=files,
                script=script,
                script_name=script_name,
            )
        with self._connect() as client:
            mkdir = self._run(client, f"mkdir -p {shlex.quote(remote_dir)}")
            if not mkdir["ok"]:
                raise RuntimeError("could not create remote run directory")
            sftp = client.open_sftp()
            try:
                for name, content in {**files, script_name: script.encode()}.items():
                    if "/" in name or "\\" in name or name in {".", ".."}:
                        raise ValueError("unsafe bundle filename")
                    with sftp.file(str(PurePosixPath(remote_dir) / name), "wb") as handle:
                        handle.write(content)
            finally:
                sftp.close()
            result = self._run(client, f"cd {shlex.quote(remote_dir)} && sbatch --parsable {shlex.quote(script_name)}")
            if not result["ok"]:
                raise RuntimeError("sbatch failed: " + result["stderr"][-500:])
            job_id = result["stdout"].strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError("Slurm returned an invalid job id")
        return job_id, remote_dir

    def _submit_manual_stage_bundle(
        self,
        *,
        run_id: str,
        stage: str,
        files: dict[str, bytes],
        script: str,
        script_name: str,
    ) -> tuple[str, str]:
        array = re.search(r"^#SBATCH --array=0-(\d+)%(\d+)$", script, re.MULTILINE)
        if array is None:
            raise ValueError("manual stage requires a bounded array declaration")
        case_count = int(array.group(1)) + 1
        requested_parallel = int(array.group(2))
        nodes = self._manual_nodes()
        worker_count = min(case_count, requested_parallel, len(nodes))
        remote_dir = safe_remote_path(self.runtime.remote_root, "polymer-agent", run_id)
        launcher_name = f"stage-{stage}.manual.sh"
        launcher_lines = [
            "#!/usr/bin/env bash",
            "set -uo pipefail",
            f"NODES=({' '.join(shlex.quote(node) for node in nodes[:worker_count])})",
            f"CASE_COUNT={case_count}",
            f"WORKER_COUNT={worker_count}",
            "failed=0",
            "for ((worker=0; worker<WORKER_COUNT; worker++)); do",
            "  (",
            "    node=${NODES[$worker]}",
            "    for ((idx=worker; idx<CASE_COUNT; idx+=WORKER_COUNT)); do",
            (
                "      ssh -o BatchMode=yes -o ConnectTimeout=10 \"$node\" "
                f"\"cd {shlex.quote(remote_dir)} && SLURM_ARRAY_TASK_ID=$idx bash {shlex.quote(script_name)}\" "
                f"> {shlex.quote(remote_dir)}/manual-{stage}-$idx.out 2>&1 || exit 1"
            ),
            "    done",
            "  ) &",
            "done",
            "for pid in $(jobs -p); do wait \"$pid\" || failed=1; done",
            "exit $failed",
            "",
        ]
        launcher = "\n".join(launcher_lines).encode()
        job_id = "manual-" + secrets.token_hex(8)
        jobs_root = safe_remote_path(self.runtime.remote_root, ".polymer-agent", "jobs")
        status_path = str(PurePosixPath(jobs_root) / f"{job_id}.status")
        pid_path = str(PurePosixPath(jobs_root) / f"{job_id}.pid")
        log_path = str(PurePosixPath(jobs_root) / f"{job_id}.log")
        launcher_path = str(PurePosixPath(remote_dir) / launcher_name)
        with self._connect() as client:
            mkdir = self._run(client, f"mkdir -p {shlex.quote(remote_dir)} {shlex.quote(jobs_root)}")
            if not mkdir["ok"]:
                raise RuntimeError("could not create remote run directory")
            sftp = client.open_sftp()
            try:
                for name, content in {**files, script_name: script.encode(), launcher_name: launcher}.items():
                    if "/" in name or "\\" in name or name in {".", ".."}:
                        raise ValueError("unsafe bundle filename")
                    with sftp.file(str(PurePosixPath(remote_dir) / name), "wb") as handle:
                        handle.write(content)
            finally:
                sftp.close()
            wrapped = (
                f"echo RUNNING > {shlex.quote(status_path)}; "
                f"if bash {shlex.quote(launcher_path)}; then echo COMPLETED > {shlex.quote(status_path)}; "
                f"else echo FAILED > {shlex.quote(status_path)}; fi"
            )
            launch = self._run(
                client,
                f"nohup setsid bash -c {shlex.quote(wrapped)} > {shlex.quote(log_path)} 2>&1 < /dev/null & "
                f"echo $! > {shlex.quote(pid_path)}",
            )
            if not launch["ok"]:
                raise RuntimeError("manual stage launch failed")
        return job_id, remote_dir

    def cancel(self, job_id: str) -> None:
        if str(job_id).startswith("manual-"):
            jobs_root = safe_remote_path(self.runtime.remote_root, ".polymer-agent", "jobs")
            pid_path = str(PurePosixPath(jobs_root) / f"{job_id}.pid")
            status_path = str(PurePosixPath(jobs_root) / f"{job_id}.status")
            with self._connect() as client:
                result = self._run(
                    client,
                    f"pid=$(cat {shlex.quote(pid_path)}); kill -- -$pid 2>/dev/null || true; "
                    f"echo CANCELLED > {shlex.quote(status_path)}",
                )
            if not result["ok"]:
                raise RuntimeError("manual job cancellation failed")
            return
        if not str(job_id).isdigit():
            raise ValueError("invalid Slurm job id")
        with self._connect() as client:
            result = self._run(client, f"scancel {job_id}")
        if not result["ok"]:
            raise RuntimeError("scancel failed")

    def status(self, job_id: str) -> str:
        if str(job_id).startswith("manual-"):
            jobs_root = safe_remote_path(self.runtime.remote_root, ".polymer-agent", "jobs")
            status_path = str(PurePosixPath(jobs_root) / f"{job_id}.status")
            with self._connect() as client:
                result = self._run(client, f"cat {shlex.quote(status_path)} 2>/dev/null || echo UNKNOWN")
            state = str(result.get("stdout") or "UNKNOWN").strip().upper()
            return state if state in {"RUNNING", "COMPLETED", "FAILED", "CANCELLED"} else "UNKNOWN"
        if not str(job_id).isdigit():
            raise ValueError("invalid Slurm job id")
        command = f"sacct -n -P -j {job_id} --format=State"
        with self._connect() as client:
            result = self._run(client, command)
        if not result["ok"]:
            raise RuntimeError("sacct failed")
        states = [line.split("|", 1)[0].strip().upper().split("+", 1)[0] for line in result["stdout"].splitlines() if line.strip()]
        if not states:
            return "UNKNOWN"
        failures = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
        active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED"}
        for state in ("OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "FAILED", "CANCELLED"):
            if state in states:
                return state
        for state in states:
            if state in active:
                return state
        if states and all(state == "COMPLETED" for state in states):
            return "COMPLETED"
        if any(state in failures for state in states):
            return "FAILED"
        return states[0]

    def read_remote_file(self, remote_path: str, max_bytes: int = 50 * 1024 * 1024) -> bytes:
        path = PurePosixPath(remote_path)
        root = PurePosixPath(self.runtime.remote_root)
        if not path.is_absolute() or path != root and root not in path.parents:
            raise ValueError("remote artifact is outside configured root")
        with self._connect() as client:
            sftp = client.open_sftp()
            try:
                stat = sftp.stat(str(path))
                if stat.st_size > max_bytes:
                    raise ValueError("remote artifact exceeds collection limit")
                with sftp.file(str(path), "rb") as handle:
                    return handle.read(max_bytes + 1)
            finally:
                sftp.close()

    def read_remote_files(self, remote_paths: list[str], max_each_bytes: int = 10 * 1024 * 1024) -> dict[str, bytes]:
        if not 1 <= len(remote_paths) <= 1000:
            raise ValueError("remote file batch must contain 1-1000 paths")
        root = PurePosixPath(self.runtime.remote_root)
        checked: list[PurePosixPath] = []
        for raw in remote_paths:
            path = PurePosixPath(raw)
            if not path.is_absolute() or path != root and root not in path.parents:
                raise ValueError("remote artifact is outside configured root")
            checked.append(path)
        output: dict[str, bytes] = {}
        with self._connect() as client:
            sftp = client.open_sftp()
            try:
                for path in checked:
                    info = sftp.stat(str(path))
                    if info.st_size > max_each_bytes:
                        raise ValueError("remote artifact exceeds collection limit")
                    with sftp.file(str(path), "rb") as handle:
                        output[str(path)] = handle.read(max_each_bytes + 1)
            finally:
                sftp.close()
        return output
