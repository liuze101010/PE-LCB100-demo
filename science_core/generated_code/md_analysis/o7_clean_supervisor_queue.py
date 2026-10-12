#!/usr/bin/env python3
"""O7 clean-engine sidecar queue for new-server GROMACS 2024.4 CUDA runs.

This is deliberately separate from the PE-LCB100 production supervisor.  It
reuses the supervisor's runtime dashboard, slot accounting, run-package
templates, and nohup launch pattern, but writes clean-engine reruns under a
distinct case-id namespace and remote root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_CODE_DIR = SCRIPT_DIR.parent
PIPELINE_DIR = GENERATED_CODE_DIR / "lcb_md_pipeline"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

import lcb300_supervisor as supervisor  # noqa: E402
import lcb_md_submit  # noqa: E402
from lcb_project_config import ClusterProfile, load_lcb_config  # noqa: E402


DEFAULT_AUDIT_DIR = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_engine_audit_20260522"
DEFAULT_SOURCE_CSV = DEFAULT_AUDIT_DIR / "source_dirs_combined_unique.csv"
DEFAULT_QUEUE_DIR = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_cuda2024_supervisor_queue"
DEFAULT_REMOTE_ROOT = "${PELCB100_REMOTE_HOME}/pelcb_clean_engine_runs"
DEFAULT_SUBSET_PLAN = (
    SCRIPT_DIR
    / "output"
    / "pelcb100_supervisor"
    / "clean_engine_robustness_20260525_stage1"
    / "o7_stage1_old2024_plan.csv"
)

NEW_HOST = "${REMOTE_USER}@private-4e2f23ff.invalid"
SOURCE_HOSTS = {
    "new": "${REMOTE_USER}@private-4e2f23ff.invalid",
    "newnew": "${REMOTE_USER}@private-169fa2c7.invalid",
}
GMX_2024_CUDA = "${PELCB100_REMOTE_HOME}/tools/gromacs/bin/gmx2024_4_cuda112"
MODULE_PRESET_2024 = "newserver-gromacs2024-cuda112"
ALLOWED_NODES = {"node-f40366db", "node-762c552a", "node-8accbfac", "node-d61b45bd"}
DISALLOWED_NODES = {"node-f168f4f6", "node-38956df6", "node-7f4add29", "node-3a4c4554"}
DEFAULT_CLUSTER_PROFILE = "o7_clean"
ENGINE_LABEL_2024 = "new_gromacs_2024.4_cuda112"
ENGINE_VERSION_2024 = "GROMACS 2024.4"


def legacy_o7_cluster_profile() -> ClusterProfile:
    return ClusterProfile(
        label=DEFAULT_CLUSTER_PROFILE,
        host_label="new",
        host=NEW_HOST,
        remote_root=DEFAULT_REMOTE_ROOT,
        engine_module="source ${PELCB100_REMOTE_HOME}/tools/gromacs/env_gromacs2024_4_cuda112.sh",
        module_preset=MODULE_PRESET_2024,
        gmx_bin=GMX_2024_CUDA,
        partition="manual",
        allowed_nodes=tuple(sorted(ALLOWED_NODES)),
        disallowed_nodes=tuple(sorted(DISALLOWED_NODES)),
        default_threads=2,
        threads_by_node={"node-d61b45bd": 6},
        node_cap_default=10,
        node_cap_by_node={"node-d61b45bd": 8},
        per_gpu_target_default=0,
        per_gpu_target_by_node={"node-d61b45bd": 2},
        launch_via_node=True,
        engine_label=ENGINE_LABEL_2024,
        engine_version_expected=ENGINE_VERSION_2024,
    )


def resolve_o7_cluster_profile(config: Any, label: str = DEFAULT_CLUSTER_PROFILE) -> ClusterProfile:
    if label in getattr(config, "cluster_profiles", {}):
        return config.cluster_profile(label)
    if label == DEFAULT_CLUSTER_PROFILE:
        return legacy_o7_cluster_profile()
    return config.cluster_profile(label)


def source_hosts_from_config(config: Any) -> dict[str, str]:
    return {
        "new": str(config.new_server),
        "newnew": str(config.newnew_server),
    }


def configure_windows_ssh_path() -> None:
    """Prefer a real OpenSSH binary over sandbox deny wrappers on Windows."""
    if os.name != "nt":
        return
    candidates = [
        Path("D:/app/OpenSSH-Win64"),
        Path("C:/Windows/System32/OpenSSH"),
    ]
    for candidate in candidates:
        if (candidate / "ssh.exe").exists() and (candidate / "scp.exe").exists():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            return


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(
    cmd: list[str],
    *,
    check: bool = True,
    timeout: int | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout, cwd=cwd)
    if check and result.returncode != 0:
        cwd_text = f"\nCWD: {cwd}" if cwd else ""
        raise RuntimeError(
            "command failed\n"
            f"CMD: {' '.join(cmd)}{cwd_text}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def ssh(host: str, command: str, *, check: bool = True, timeout: int | None = 120) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", host, command],
        check=check,
        timeout=timeout,
    )


def refresh_dashboard_for_queue(config_path: Path, timeout_sec: int) -> dict[str, Any]:
    command = [sys.executable, str(supervisor.DASHBOARD)]
    if config_path:
        command.extend(["--config", str(config_path)])
    try:
        proc = run_cmd(command, check=False, timeout=max(1, int(timeout_sec)))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "returncode": "",
            "stdout": str(stdout).strip(),
            "stderr": (str(stderr).strip() + f"\nTimeoutExpired: runtime dashboard exceeded {timeout_sec}s").strip(),
            "ok": False,
            "exception": repr(exc),
        }
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


def sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_case_id(case_id: str) -> str:
    if case_id.startswith("lcb_pe_"):
        return "lcb_pe24cuda_" + case_id.removeprefix("lcb_pe_")
    return "lcb_pe24cuda_" + case_id


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def runtime_snapshot_status(runtime_dir: Path, max_age_seconds: int = 1800) -> dict[str, Any]:
    """Return whether existing runtime CSVs are fresh enough for slot fallback."""
    required = [
        runtime_dir / "runtime_gpu_snapshot.csv",
        runtime_dir / "runtime_active_processes.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"ok": False, "reason": "missing_runtime_csv", "missing": missing}
    newest = max(path.stat().st_mtime for path in required)
    oldest = min(path.stat().st_mtime for path in required)
    age_seconds = max(0, int(datetime.now().timestamp() - oldest))
    if age_seconds > max_age_seconds:
        return {
            "ok": False,
            "reason": "runtime_snapshot_stale",
            "age_seconds": age_seconds,
            "max_age_seconds": max_age_seconds,
            "newest_mtime": datetime.fromtimestamp(newest).isoformat(timespec="seconds"),
            "oldest_mtime": datetime.fromtimestamp(oldest).isoformat(timespec="seconds"),
        }
    return {
        "ok": True,
        "reason": "runtime_snapshot_fresh",
        "age_seconds": age_seconds,
        "max_age_seconds": max_age_seconds,
        "newest_mtime": datetime.fromtimestamp(newest).isoformat(timespec="seconds"),
        "oldest_mtime": datetime.fromtimestamp(oldest).isoformat(timespec="seconds"),
    }


def mark_rows_blocked_no_slots(rows: list[dict[str, Any]], reason: str) -> None:
    """Mark retryable pending rows as waiting for capacity without losing queue intent."""
    blocked_at = iso_now()
    for row in rows:
        status = str(row.get("status", "")).upper()
        if status in {"SUBMITTED", "RUNNING", "DONE"}:
            continue
        row["status"] = "BLOCKED_NO_SLOTS"
        row["blocked_reason"] = reason
        row["blocked_at"] = blocked_at


def source_rows_by_case(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    return {row["case_id"]: row for row in rows if row.get("case_id")}


def default_case_ids(source_csv: Path, max_cases: int) -> list[str]:
    if DEFAULT_SUBSET_PLAN.exists():
        ids = [row["case_id"] for row in read_csv_rows(DEFAULT_SUBSET_PLAN) if row.get("case_id")]
        if ids:
            return ids[:max_cases]
    return [row["case_id"] for row in read_csv_rows(source_csv) if row.get("case_id")][:max_cases]


def requested_case_ids(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return args.cases
    if args.cases_file:
        rows = read_csv_rows(args.cases_file)
        if rows and "case_id" in rows[0]:
            return [row["case_id"] for row in rows if row.get("case_id")]
        return [
            line.strip()
            for line in args.cases_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return default_case_ids(args.source_csv, args.default_case_count)


def list_source_files(host: str, remote_dir: str) -> list[str]:
    command = (
        f"cd {sh_quote(remote_dir)} && "
        "find . -maxdepth 1 -type f "
        "\\( -name '*.gro' -o -name 'topol.top' -o -name '*.itp' "
        "-o -name 'steep.mdp' -o -name 'rlx.mdp' -o -name 'pr.mdp' "
        "-o -name 'nvt_prod.mdp' -o -name 'case.json' \\) "
        "-printf '%f\\n' | sort"
    )
    result = ssh(host, command)
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    required = {"topol.top", "steep.mdp", "rlx.mdp", "pr.mdp", "nvt_prod.mdp"}
    if not any(name.endswith(".gro") for name in files):
        raise RuntimeError(f"no .gro file found in {host}:{remote_dir}")
    if not any(name.endswith(".itp") for name in files):
        raise RuntimeError(f"no .itp file found in {host}:{remote_dir}")
    missing = sorted(required - set(files))
    if missing:
        raise RuntimeError(f"missing required files in {host}:{remote_dir}: {missing}")
    return files


def scp_from(host: str, remote_dir: str, filename: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = f"{host}:{remote_dir.rstrip('/')}/{filename}"
    # Keep the local destination argument short. Windows OpenSSH can fail on
    # long absolute output paths even when the parent directory exists.
    run_cmd(["scp", "-q", src, filename], timeout=600, cwd=dest_dir)


def merge_clean_case_metadata(
    source_metadata: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Merge the source MD-run case.json with the clean-engine provenance.

    Preserves all polymer/structural fields from the source so downstream
    analyses (e.g., compute_monomer_msd which requires backbone_repeat_units)
    keep working on clean-engine clones. The source's own source_case_id, if
    present (EMC builder reference), is preserved as ``emc_source_case_id``
    before the clean-engine provenance overrides occur.

    Returns a new dict; inputs are not mutated.
    """
    merged: dict[str, Any] = dict(source_metadata or {})
    # The source case.json's source_case_id (e.g., EMC builder code such as
    # 'lcb13_comb_b2_l20_c100') is one level deeper than the manifest case_id
    # and is preserved here before the clean-engine provenance overrides
    # source_case_id with the manifest case_id (which downstream tools key on).
    if "source_case_id" in merged and "emc_source_case_id" not in merged:
        merged["emc_source_case_id"] = merged["source_case_id"]
    merged.update(provenance)
    return merged


def stage_bridge_case(
    case_id: str,
    source_row: dict[str, str],
    bridge_root: Path,
    tag: str,
    source_hosts: dict[str, str] | None = None,
) -> tuple[str, Path]:
    host_label = source_row["host_label"]
    host = (source_hosts or SOURCE_HOSTS)[host_label]
    remote_dir = source_row["path"]
    clean_id = clean_case_id(case_id)
    dest = bridge_root / clean_id
    if not dest.exists():
        files = list_source_files(host, remote_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for filename in files:
            scp_from(host, remote_dir, filename, dest)
    # Read the source case.json that was just scp'd; it carries the polymer
    # metadata (backbone_repeat_units, branch_points, force_field, ...) that
    # downstream postprocess suites need. Earlier versions of this function
    # silently dropped these fields by overwriting case.json with provenance
    # only; the merge preserves them.
    source_case_json = dest / "case.json"
    source_metadata: dict[str, Any] = {}
    if source_case_json.exists():
        try:
            source_metadata = json.loads(source_case_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            source_metadata = {}
    provenance = {
        "case_id": clean_id,
        "source_case_id": case_id,
        "control_kind": "o7_clean_engine_full_new_cuda2024",
        "source_host_label": host_label,
        "source_host": host,
        "source_dir": remote_dir,
        "created_at": iso_now(),
        "run_tag": tag,
    }
    merged = merge_clean_case_metadata(source_metadata, provenance)
    (dest / "case.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return clean_id, dest


def lcb_submit_args(cpus: int, gpu_id: str, args: argparse.Namespace, slot: dict[str, Any] | None = None) -> SimpleNamespace:
    slot = slot or {}
    return SimpleNamespace(
        cpus=cpus,
        gmx_bin=str(slot.get("gmx_bin") or GMX_2024_CUDA),
        module_preset=str(slot.get("module_preset") or MODULE_PRESET_2024),
        partition=str(slot.get("partition") or "manual"),
        time_limit="30-00:00:00",
        maxwarn=0,
        gpu=True,
        gpu_id=gpu_id,
        script_name="run_manual.sh",
        template=PIPELINE_DIR / "manual_template.sh",
        steep_nsteps=args.steep_nsteps,
        rlx_ns=args.rlx_ns,
        pr_ns=args.pr_ns,
        nvt_ns=args.nvt_ns,
        overwrite=True,
    )


def build_local_run(
    clean_id: str,
    source_case_id: str,
    source_row: dict[str, str],
    bridge_case_dir: Path,
    local_run_base: Path,
    tag: str,
    slot: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    gro_files = sorted(bridge_case_dir.glob("*.gro"))
    itp_files = sorted(bridge_case_dir.glob("*.itp"))
    if not gro_files or not itp_files:
        raise RuntimeError(f"incomplete bridge dir: {bridge_case_dir}")
    metadata = json.loads((bridge_case_dir / "case.json").read_text(encoding="utf-8"))
    case = lcb_md_submit.BridgeCase(
        case_id=clean_id,
        case_dir=bridge_case_dir,
        gro_path=gro_files[0],
        top_path=bridge_case_dir / "topol.top",
        itp_paths=tuple(itp_files),
        case_json=bridge_case_dir / "case.json",
        metadata=metadata,
    )
    # The queue tag is already encoded in the parent workspace and remote root.
    # Keeping it out of the local run-directory name avoids Windows MAX_PATH
    # failures when copying long per-case topology filenames.
    run_dir = lcb_md_submit.build_run_dir(case, local_run_base, bridge_case_dir, "run", overwrite=True)
    submit_args = lcb_submit_args(cpus=int(slot["threads"]), gpu_id=str(slot["gpu"]), args=args, slot=slot)
    lcb_md_submit.apply_duration_overrides(run_dir, submit_args)
    lcb_md_submit.generate_slurm_script(case, run_dir, submit_args)

    mdp_hashes = {path.name: sha256_file(path) for path in sorted(run_dir.glob("*.mdp"))}
    input_hashes = {
        "gro": sha256_file(run_dir / gro_files[0].name),
        "topol.top": sha256_file(run_dir / "topol.top"),
        **{path.name: sha256_file(run_dir / path.name) for path in itp_files},
    }
    run_provenance = {
        **metadata,
        "engine": str(slot.get("engine") or ENGINE_LABEL_2024),
        "engine_version_expected": str(slot.get("engine_version_expected") or ENGINE_VERSION_2024),
        "engine_module": str(slot.get("engine_module", "")),
        "gmx_binary": str(slot.get("gmx_bin") or GMX_2024_CUDA),
        "host": str(slot.get("host") or NEW_HOST),
        "node": slot["node"],
        "gpu_id": slot["gpu"],
        "threads": slot["threads"],
        "module_preset": str(slot.get("module_preset") or MODULE_PRESET_2024),
        "source_case_id": source_case_id,
        "source_host_label": source_row["host_label"],
        "source_dir": source_row["path"],
        "run_tag": tag,
        "local_run_dir": str(run_dir),
        "mdp_sha256": mdp_hashes,
        "input_sha256": input_hashes,
        "nvt_ns_override": args.nvt_ns,
        "full_supervisor_style": args.nvt_ns is None,
        "stage2_blocked": False,
    }
    write_json(run_dir / "o7_provenance.json", run_provenance)
    return run_dir


def setup_supervisor_globals(config_path: Path, queue_dir: Path, cluster_profile: ClusterProfile | None = None) -> Any:
    config = load_lcb_config(config_path)
    profile = cluster_profile or resolve_o7_cluster_profile(config)
    supervisor.CONFIG_PATH = config_path
    supervisor.PROJECT_ID = "pelcb100"
    supervisor.OUTPUT_DIR = config.supervisor_output_dir
    supervisor.RUNTIME_DIR = config.runtime_output_dir
    supervisor.UPSTREAM_DIR = config.emc_output_dir
    supervisor.BRIDGE_DIRS = [config.default_emc_bridge_root, *config.bridge_roots]
    supervisor.NEW_SERVER = config.new_server
    supervisor.NEWNEW_SERVER = config.newnew_server
    supervisor.OLD_SERVER = config.old_server
    supervisor.PRODUCTION_CLUSTER_PROFILES = supervisor.production_cluster_profiles_from_config(config)
    # The shared supervisor computes slots by host_label ("new"/"newnew").
    # Clean-engine sidecars must therefore inject their selected cluster profile
    # under the host_label key before calling compute_slots; otherwise raw slots
    # fall back to the legacy production profile and are filtered out later.
    supervisor.PRODUCTION_CLUSTER_PROFILES[profile.host_label] = profile
    if profile.host_label == "new":
        supervisor.NEW_SERVER = profile.host
        supervisor.NEW_REMOTE_ROOT = profile.remote_root
        supervisor.NEW_GMX = profile.gmx_bin
    supervisor.DISABLED_MD_HOSTS = set(config.disabled_md_hosts)
    supervisor.DISABLED_NEW_NODES = set(config.disabled_new_nodes)
    # Keep queue-local decisions out of the production supervisor state.
    queue_dir.mkdir(parents=True, exist_ok=True)
    return config


def slot_key(slot: dict[str, Any]) -> str:
    return supervisor.slot_key("new", str(slot["node"]), str(slot["gpu"]))


def staging_roots(batch_dir: Path) -> tuple[Path, Path]:
    """Return short local staging directories for bridge inputs and run files.

    Daemon tags are intentionally descriptive and can become long.  Keep the
    intermediate directory names short so Windows local staging does not exceed
    path-length limits before files are copied to the remote cluster.
    """
    return batch_dir / "b", batch_dir / "r"


def active_sidecar_slot_keys(plan_rows: list[dict[str, str]]) -> set[str]:
    keys: set[str] = set()
    for row in plan_rows:
        if str(row.get("status", "")).upper() not in {"SUBMITTED", "RUNNING"}:
            continue
        node = str(row.get("node", "") or "").strip()
        gpu = str(row.get("gpu_id", "") or "").strip()
        if node and gpu:
            keys.add(supervisor.slot_key("new", node, gpu))
    return keys


def recent_queue_reservations(
    queue_dir: Path,
    *,
    max_age_seconds: int = 1800,
    now_iso: str | None = None,
) -> list[dict[str, str]]:
    """Return recent submitted queue rows that may not be visible in runtime CSVs yet."""
    now = datetime.fromisoformat(now_iso) if now_iso else datetime.now()
    reservations: dict[str, tuple[datetime, dict[str, str]]] = {}
    for status_path in queue_dir.glob("*/o7_new_cuda2024_status.csv"):
        fallback_time = datetime.fromtimestamp(status_path.stat().st_mtime)
        for row in read_csv_rows(status_path):
            if str(row.get("status", "")).upper() not in {"SUBMITTED", "RUNNING"}:
                continue
            node = str(row.get("node", "") or "").strip()
            gpu = str(row.get("gpu_id", "") or "").strip()
            if not node or not gpu:
                continue
            stamp_text = str(row.get("submitted_at") or row.get("created_at") or "").strip()
            try:
                stamp = datetime.fromisoformat(stamp_text) if stamp_text else fallback_time
            except ValueError:
                stamp = fallback_time
            age_seconds = (now - stamp).total_seconds()
            if age_seconds < 0 or age_seconds > max_age_seconds:
                continue
            remote_run_dir = str(row.get("remote_run_dir", "") or "").strip()
            key = remote_run_dir or f"{row.get('case_id', '')}|{node}|{gpu}"
            current = reservations.get(key)
            if current is None or stamp > current[0]:
                reservations[key] = (stamp, dict(row))
    return [item[1] for item in sorted(reservations.values(), key=lambda item: item[0])]


def merge_runtime_active_with_reservations(
    active: pd.DataFrame,
    reservations: list[dict[str, str]],
) -> pd.DataFrame:
    """Count queue reservations as active until the live dashboard sees their run directory."""
    merged = active.copy()
    active_cwds = {
        str(value).strip()
        for value in merged.get("cwd", pd.Series(dtype=str)).fillna("").astype(str)
        if str(value).strip()
    }
    active_case_ids = {
        str(value).strip()
        for value in merged.get("case_id", pd.Series(dtype=str)).fillna("").astype(str)
        if str(value).strip()
    }
    synthetic: list[dict[str, Any]] = []
    seen_reservations: set[str] = set()
    for row in reservations:
        if str(row.get("status", "")).upper() not in {"SUBMITTED", "RUNNING"}:
            continue
        node = str(row.get("node", "") or "").strip()
        gpu = str(row.get("gpu_id", "") or "").strip()
        case_id = str(row.get("case_id", "") or "").strip()
        remote_run_dir = str(row.get("remote_run_dir", "") or "").strip()
        reservation_key = remote_run_dir or f"{case_id}|{node}|{gpu}"
        if not node or not gpu or reservation_key in seen_reservations:
            continue
        seen_reservations.add(reservation_key)
        if case_id and case_id in active_case_ids:
            continue
        if remote_run_dir and any(cwd == remote_run_dir or cwd.startswith(remote_run_dir.rstrip("/") + "/") for cwd in active_cwds):
            continue
        synthetic.append(
            {
                "host_label": "new",
                "node": node,
                "gpu_id": gpu,
                "case_id": case_id,
                "cwd": remote_run_dir,
                "reservation_only": True,
            }
        )
    if synthetic:
        merged = pd.concat([merged, pd.DataFrame(synthetic)], ignore_index=True, sort=False)
    return merged


def choose_clean_slots(
    runtime: dict[str, pd.DataFrame],
    plan_rows: list[dict[str, str]],
    allow_second_per_gpu: bool,
    cluster_profile: ClusterProfile | None = None,
    reservation_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    profile = cluster_profile or legacy_o7_cluster_profile()
    all_reservations = [*plan_rows, *(reservation_rows or [])]
    active = merge_runtime_active_with_reservations(runtime["active"], all_reservations)
    blocked = set()
    if not allow_second_per_gpu:
        blocked |= active_sidecar_slot_keys(all_reservations)
    raw_slots = supervisor.compute_slots(runtime["gpus"], active, blocked_slots=blocked)
    slots: list[dict[str, Any]] = []
    seen_raw_slot_keys: set[str] = set()
    for raw in raw_slots:
        if str(raw.get("host_label")) != profile.host_label:
            continue
        node = str(raw.get("node", ""))
        if profile.allowed_nodes and node not in set(profile.allowed_nodes):
            continue
        if node in set(profile.disallowed_nodes):
            continue
        raw_slot_key = supervisor.slot_key("new", node, str(raw.get("gpu", "")))
        if not allow_second_per_gpu and raw_slot_key in seen_raw_slot_keys:
            continue
        seen_raw_slot_keys.add(raw_slot_key)
        slot = dict(raw)
        slot["host_label"] = profile.host_label
        slot["host"] = profile.host
        slot["remote_root"] = profile.remote_root
        slot["module_preset"] = profile.module_preset
        slot["gmx_bin"] = profile.gmx_bin
        slot["engine_module"] = profile.engine_module
        slot["engine"] = profile.engine_label or ENGINE_LABEL_2024
        slot["engine_version_expected"] = profile.engine_version_expected or ENGINE_VERSION_2024
        slot["partition"] = profile.partition or "manual"
        slot["threads"] = profile.threads_for_node(node)
        slot["launch_via_node"] = profile.launch_via_node
        slots.append(slot)
    return slots


def deploy_and_launch_o7(run_dir: Path, remote_base: str, slot: dict[str, Any]) -> dict[str, str]:
    remote_case_dir = f"{remote_base.rstrip('/')}/{run_dir.name}"
    host = str(slot.get("host") or NEW_HOST)
    ssh(host, f"mkdir -p {sh_quote(remote_base)}", timeout=60)
    run_cmd(["scp", "-q", "-r", str(run_dir), f"{host}:{remote_base.rstrip('/')}/"], timeout=900)
    launch_inner = supervisor.remote_launch_command(remote_case_dir)
    command = f"ssh -o BatchMode=yes -o ConnectTimeout=20 {sh_quote(str(slot['node']))} {sh_quote('bash -lc ' + sh_quote(launch_inner))}"
    proc = ssh(host, command, timeout=120)
    return {
        "remote_case_dir": remote_case_dir,
        "launch_stdout": proc.stdout.strip(),
        "launch_stderr": proc.stderr.strip(),
    }


def parse_pid(text: str) -> str:
    match = re.search(r"^\s*(\d+)\s+", text or "", re.MULTILINE)
    return match.group(1) if match else ""


def ensure_plan_rows(plan_path: Path, case_ids: list[str]) -> list[dict[str, str]]:
    rows = read_csv_rows(plan_path)
    existing = {row.get("case_id") for row in rows}
    for case_id in case_ids:
        if case_id in existing:
            continue
        rows.append({"case_id": case_id, "status": "PENDING"})
    return rows


def plan_fields(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "case_id",
        "clean_case_id",
        "status",
        "node",
        "gpu_id",
        "threads",
        "source_host_label",
        "source_dir",
        "local_run_dir",
        "remote_run_dir",
        "pid",
        "launch_stdout",
        "created_at",
        "submitted_at",
        "engine",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return preferred + extras


def update_status_outputs(
    batch_dir: Path,
    rows: list[dict[str, Any]],
    tag: str,
    execute: bool,
    slots: list[dict[str, Any]],
    runtime_status: dict[str, Any] | None = None,
    dashboard_status: dict[str, Any] | None = None,
) -> None:
    write_csv_rows(batch_dir / "o7_new_cuda2024_plan.csv", rows, plan_fields(rows))
    launch_rows = [row for row in rows if str(row.get("status", "")).upper() in {"SUBMITTED", "PREPARED"}]
    if launch_rows:
        write_csv_rows(batch_dir / "launch_manifest.csv", launch_rows, plan_fields(launch_rows))
    status_fields = [
        "case_id",
        "clean_case_id",
        "status",
        "node",
        "gpu_id",
        "threads",
        "remote_run_dir",
        "pid",
        "engine",
        "blocked_reason",
        "blocked_at",
    ]
    write_csv_rows(batch_dir / "o7_new_cuda2024_status.csv", rows, status_fields)
    # Backward-compatible alias for scripts written during the original
    # density-only pilot.  Full-suite runs should use o7_new_cuda2024_status.csv.
    write_csv_rows(batch_dir / "o7_new_cuda2024_stage1_status.csv", rows, status_fields)
    write_json(
        batch_dir / "o7_new_cuda2024_summary.json",
        {
            "tag": tag,
            "created_at": iso_now(),
            "execute": execute,
            "available_slots": slots,
            "runtime_status": runtime_status or {},
            "dashboard_status": dashboard_status or {},
            "rows": rows,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "pelcb100.json")
    parser.add_argument("--cluster-profile", default=DEFAULT_CLUSTER_PROFILE)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--remote-root", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--default-case-count", type=int, default=12)
    parser.add_argument("--max-submit", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-stale-runtime-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use existing runtime CSVs if live dashboard refresh times out or fails.",
    )
    parser.add_argument("--runtime-max-age-seconds", type=int, default=1800)
    parser.add_argument(
        "--dashboard-timeout-sec",
        type=int,
        default=45,
        help="Maximum seconds to wait for live runtime dashboard refresh before using cached runtime CSVs.",
    )
    parser.add_argument(
        "--allow-second-per-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow two clean-engine sidecar jobs to share one GPU. Use --no-allow-second-per-gpu for one case per GPU.",
    )
    parser.add_argument("--steep-nsteps", type=int, default=None)
    parser.add_argument("--rlx-ns", type=float, default=None)
    parser.add_argument("--pr-ns", type=float, default=None)
    parser.add_argument(
        "--nvt-ns",
        type=float,
        default=None,
        help="Optional nvt_prod duration override in ns. Omit for supervisor-style full MDP length.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use short duration overrides for a launch smoke. Does not change source mdp files.",
    )
    return parser.parse_args()


def main() -> int:
    configure_windows_ssh_path()
    args = parse_args()
    if args.smoke:
        args.steep_nsteps = args.steep_nsteps or 1000
        args.rlx_ns = args.rlx_ns if args.rlx_ns is not None else 0.001
        args.pr_ns = args.pr_ns if args.pr_ns is not None else 0.001
        args.nvt_ns = args.nvt_ns if args.nvt_ns is not None else 0.001

    tag = args.tag or "o7_new_cuda2024_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = args.queue_dir / tag
    bridge_root, local_run_base = staging_roots(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    config = load_lcb_config(args.config)
    cluster_profile = resolve_o7_cluster_profile(config, str(args.cluster_profile))
    remote_root = str(args.remote_root or cluster_profile.remote_root or DEFAULT_REMOTE_ROOT)
    remote_base = f"{remote_root.rstrip('/')}/{tag}"
    setup_supervisor_globals(args.config, args.queue_dir, cluster_profile=cluster_profile)
    source_hosts = source_hosts_from_config(config)

    source_rows = source_rows_by_case(args.source_csv)
    case_ids = requested_case_ids(args)
    missing = [case_id for case_id in case_ids if case_id not in source_rows]
    if missing:
        raise KeyError(f"case(s) missing from source csv: {missing}")

    plan_path = batch_dir / "o7_new_cuda2024_plan.csv"
    rows: list[dict[str, Any]] = ensure_plan_rows(plan_path, case_ids)
    dashboard = refresh_dashboard_for_queue(args.config, timeout_sec=int(args.dashboard_timeout_sec))
    runtime_status = runtime_snapshot_status(supervisor.RUNTIME_DIR, max_age_seconds=int(args.runtime_max_age_seconds))
    if not dashboard.get("ok"):
        if not args.allow_stale_runtime_fallback or not runtime_status.get("ok"):
            update_status_outputs(batch_dir, rows, tag, bool(args.execute), [], runtime_status, dashboard)
            raise RuntimeError(f"runtime dashboard failed and no usable fallback runtime exists: {dashboard}; {runtime_status}")
    runtime = supervisor.load_runtime()
    reservations = recent_queue_reservations(
        args.queue_dir,
        max_age_seconds=int(args.runtime_max_age_seconds),
    )
    runtime_status["queue_reservation_count"] = len(reservations)
    slots = choose_clean_slots(
        runtime,
        rows,
        allow_second_per_gpu=bool(args.allow_second_per_gpu),
        cluster_profile=cluster_profile,
        reservation_rows=reservations,
    )
    if not slots:
        mark_rows_blocked_no_slots(rows, "no eligible clean-engine GPU slots under current supervisor policy")
        update_status_outputs(batch_dir, rows, tag, bool(args.execute), slots, runtime_status, dashboard)
        print(json.dumps({"tag": tag, "action": "blocked_no_slots", "batch_dir": str(batch_dir)}, indent=2))
        return 0

    submitted = 0
    slot_index = 0
    for row in rows:
        if submitted >= max(0, args.max_submit):
            break
        status = str(row.get("status", "")).upper()
        if status in {"SUBMITTED", "RUNNING", "DONE"}:
            continue
        if slot_index >= len(slots):
            break
        case_id = str(row["case_id"])
        slot = slots[slot_index]
        slot_index += 1
        clean_id, bridge_dir = stage_bridge_case(case_id, source_rows[case_id], bridge_root, tag, source_hosts=source_hosts)
        run_dir = build_local_run(
            clean_id=clean_id,
            source_case_id=case_id,
            source_row=source_rows[case_id],
            bridge_case_dir=bridge_dir,
            local_run_base=local_run_base,
            tag=tag,
            slot=slot,
            args=args,
        )
        row.update(
            {
                "clean_case_id": clean_id,
                "status": "PREPARED",
                "node": slot["node"],
                "gpu_id": slot["gpu"],
                "threads": slot["threads"],
                "source_host_label": source_rows[case_id]["host_label"],
                "source_dir": source_rows[case_id]["path"],
                "local_run_dir": str(run_dir),
                "remote_run_dir": "",
                "pid": "",
                "launch_stdout": "",
                "created_at": iso_now(),
                "submitted_at": "",
                "engine": str(slot.get("engine") or ENGINE_LABEL_2024),
                "engine_module": str(slot.get("engine_module", "")),
                "full_supervisor_style": "true" if args.nvt_ns is None else "false",
                "nvt_nsteps": "200000000" if args.nvt_ns is None else "",
                "nvt_ns_override": "" if args.nvt_ns is None else args.nvt_ns,
            }
        )
        if args.execute:
            launch = deploy_and_launch_o7(run_dir, remote_base, slot)
            row["status"] = "SUBMITTED"
            row["remote_run_dir"] = launch["remote_case_dir"]
            row["launch_stdout"] = launch["launch_stdout"]
            row["launch_stderr"] = launch["launch_stderr"]
            row["pid"] = parse_pid(launch["launch_stdout"])
            row["submitted_at"] = iso_now()
            remote_prov = (
                f"cd {sh_quote(launch['remote_case_dir'])} && "
                "python3 - <<'PY'\n"
                "import json, pathlib\n"
                "p=pathlib.Path('o7_provenance.json')\n"
                "d=json.loads(p.read_text())\n"
                f"d['remote_run_dir']={json.dumps(launch['remote_case_dir'])}\n"
                f"d['launch_pid']={json.dumps(row['pid'])}\n"
                f"d['submitted_at']={json.dumps(row['submitted_at'])}\n"
                "p.write_text(json.dumps(d, indent=2, ensure_ascii=False)+'\\n')\n"
                "PY"
            )
            ssh(str(slot.get("host") or NEW_HOST), remote_prov, check=False, timeout=60)
        submitted += 1

    update_status_outputs(batch_dir, rows, tag, bool(args.execute), slots, runtime_status, dashboard)
    print(
        json.dumps(
            {
                "tag": tag,
                "execute": bool(args.execute),
                "submitted_or_prepared": submitted,
                "batch_dir": str(batch_dir),
                "remote_base": remote_base,
                "plan_csv": str(plan_path),
                "available_slots": len(slots),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
