#!/usr/bin/env python3
"""LCB300 local supervisor.

Default mode is dry-run.  In dry-run it refreshes the runtime dashboard and
writes auditable decisions.  With ``--execute`` it may:

* generate/deploy/launch new manual MD packages for backlog cases,
* restart only whitelisted owner processes missing ``-bonded gpu``,
* submit missing old-server postprocessing through ``lcb300_orchestrator.py``,
* collect completed postprocessing products through
  ``lcb300_postprocess_closure.py``.

The supervisor is intentionally conservative: duplicate runtime state blocks
all mutating actions, and any unavailable backend is reported as a decision
instead of being guessed around.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from lcb_project_config import ClusterProfile, load_lcb_config
from backfill_clean_intrachain_g3 import (
    discover_msd_roots as discover_clean_g3_msd_roots,
    run_backfill as run_clean_g3_backfill,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PIPELINE_DIR = SCRIPT_DIR.parent / "lcb_md_pipeline"
DASHBOARD = SCRIPT_DIR / "lcb300_runtime_dashboard.py"
ORCHESTRATOR = SCRIPT_DIR / "lcb300_orchestrator.py"
CLOSURE = SCRIPT_DIR / "lcb300_postprocess_closure.py"
CONTROL_QUEUE = SCRIPT_DIR / "pelcb100_control_queue.py"
OVERVIEW = SCRIPT_DIR / "build_lcb300_overview.py"
REPORT = SCRIPT_DIR / "lcb_report.py"
EMC_ORCHESTRATOR = SCRIPT_DIR / "lcb300_emc_orchestrator.py"
PUBLICATION_ASSETS = SCRIPT_DIR / "lcb_publication_assets.py"
HEALTH_MONITOR = SCRIPT_DIR / "lcb_health_monitor.py"
OUTPUT_DIR = SCRIPT_DIR / "output" / "lcb300_supervisor"
RUNTIME_DIR = SCRIPT_DIR / "output" / "lcb300_runtime_dashboard"
UPSTREAM_DIR = SCRIPT_DIR / "output" / "lcb300_emc_orchestrator"
ORCHESTRATOR_DIR = SCRIPT_DIR / "output" / "lcb300_orchestrator"
CLOSURE_DIR = SCRIPT_DIR / "output" / "lcb300_postprocess_closure"

BRIDGE_DIRS = [
    SCRIPT_DIR / "output" / "lcb300_supervisor" / "bridge_ready",
    SCRIPT_DIR.parent / "emc_alpha_olefin_builder" / "output" / "lcb300_bridge_lcb201_lcb300_20260507",
    SCRIPT_DIR.parent / "emc_alpha_olefin_builder" / "output" / "screen_lcb300",
]

NEW_SERVER = "${REMOTE_USER}@private-4e2f23ff.invalid"
NEWNEW_SERVER = "${REMOTE_USER}@private-169fa2c7.invalid"
OLD_SERVER = "${REMOTE_USER}@private-3f2a50db.invalid"
NEW_REMOTE_ROOT = "${PELCB100_REMOTE_HOME}/lcb_manual_runs"
NEWNEW_REMOTE_ROOT = "${PELCB100_REMOTE_ROOT}/lcb_manual_runs"
NEW_GMX = "${PELCB100_REMOTE_PATH}"
NEWNEW_GMX = "${PELCB100_REMOTE_ROOT}/miniforge3/envs/md/bin/gmx"
DISABLED_NEW_NODES: set[str] = set()
DISABLED_MD_HOSTS: set[str] = set()
PRODUCTION_CLUSTER_PROFILES: dict[str, ClusterProfile] = {}
DISK_WARNING_FREE_GB = int(os.environ.get("PELCB_DISK_WARNING_FREE_GB", "200"))
DISK_CRITICAL_FREE_GB = int(os.environ.get("PELCB_DISK_CRITICAL_FREE_GB", "50"))

CASE_RE = re.compile(r"(?:lcb(\d+)_|lcb_[a-z]+_(\d+)_)", re.IGNORECASE)
CONFIG_PATH: Path | None = None
PROJECT_ID = "lcb300"
MD_STAGE_ORDER = ["steep", "rlx", "pr", "nvt_prod"]


def legacy_production_cluster_profiles() -> dict[str, ClusterProfile]:
    return {
        "new": ClusterProfile(
            label="production_new_legacy",
            host_label="new",
            host=NEW_SERVER,
            remote_root=NEW_REMOTE_ROOT,
            engine_module="",
            module_preset="newserver-gromacs2021",
            gmx_bin=NEW_GMX,
            partition="manual",
            allowed_nodes=(),
            disallowed_nodes=(),
            default_threads=2,
            threads_by_node={"node-d61b45bd": 6},
            launch_via_node=True,
            node_cap_default=10,
            node_cap_by_node={"node-d61b45bd": 8},
            per_gpu_target_default=0,
            per_gpu_target_by_node={"node-d61b45bd": 2},
            module_preset_by_node={"node-d61b45bd": "newserver-gromacs2021-node8"},
            idle_external_memory_threshold_mib=700,
            project_memory_per_case_mib=414,
            external_memory_slack_mib=250,
            external_memory_per_case_mib=400,
        ),
        "newnew": ClusterProfile(
            label="production_newnew_legacy",
            host_label="newnew",
            host=NEWNEW_SERVER,
            remote_root=NEWNEW_REMOTE_ROOT,
            engine_module="",
            module_preset="none",
            gmx_bin=NEWNEW_GMX,
            partition="manual",
            allowed_nodes=(),
            disallowed_nodes=(),
            default_threads=4,
            threads_by_node={},
            launch_via_node=False,
            per_gpu_target_default=2,
            idle_skip_util_percent=10,
            idle_skip_memory_mib=500,
            project_memory_per_case_mib=500,
            active_extra_memory_skip_mib=1000,
        ),
    }


def production_profile_for(host_label: str) -> ClusterProfile:
    return PRODUCTION_CLUSTER_PROFILES.get(str(host_label)) or legacy_production_cluster_profiles()[str(host_label)]


def production_cluster_profiles_from_config(config: Any) -> dict[str, ClusterProfile]:
    profiles = legacy_production_cluster_profiles()
    for label in ("production_new", "production_newnew"):
        if label not in getattr(config, "cluster_profiles", {}):
            continue
        profile = config.cluster_profile(label)
        profiles[profile.host_label] = profile
    return profiles


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_heartbeat(output_dir: Path, phase: str, **extra: Any) -> None:
    write_json(
        output_dir / "supervisor_heartbeat.json",
        {
            "time": iso_now(),
            "pid": os.getpid(),
            "project_id": PROJECT_ID,
            "phase": phase,
            **extra,
        },
    )


def run(
    command: list[str],
    *,
    timeout: int | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        env=env,
    )


def project_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LCB_PROJECT_ID"] = PROJECT_ID
    return env


def sh_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


class SupervisorLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "SupervisorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"supervisor lock exists: {self.path}")
        os.write(self.fd, f"pid={os.getpid()} time={iso_now()}\n".encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def case_seq(case_id: str) -> int:
    match = CASE_RE.match(str(case_id))
    if not match:
        return 999999
    seq = next((group for group in match.groups() if group), None)
    return int(seq) if seq else 999999


def refresh_dashboard() -> dict[str, Any]:
    command = [sys.executable, str(DASHBOARD)]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    proc = run(command, timeout=180)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


def run_emc_orchestrator(*, execute: bool, max_cases: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(EMC_ORCHESTRATOR),
        "--max-cases",
        str(max_cases),
        "--json-only",
    ]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    if execute:
        command.append("--execute")
    proc = run(command, timeout=1800, cwd=ROOT)
    latest = read_json(UPSTREAM_DIR / "emc_last_decision.json", {})
    latest["returncode"] = proc.returncode
    latest["stdout"] = proc.stdout.strip()
    latest["stderr"] = proc.stderr.strip()
    latest["ok"] = proc.returncode == 0
    return latest


def load_runtime() -> dict[str, pd.DataFrame]:
    return {
        "cases": read_csv(RUNTIME_DIR / "runtime_cases.csv"),
        "active": read_csv(RUNTIME_DIR / "runtime_active_processes.csv"),
        "gpus": read_csv(RUNTIME_DIR / "runtime_gpu_snapshot.csv"),
        "duplicates": read_csv(RUNTIME_DIR / "runtime_duplicates.csv"),
        "backlog": read_csv(RUNTIME_DIR / "runtime_backlog_submit.csv"),
        "errors": read_csv(RUNTIME_DIR / "runtime_scan_errors.csv"),
    }


def is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def active_count(active: pd.DataFrame, host_label: str, node: str | None = None, gpu: str | int | None = None) -> int:
    if active.empty:
        return 0
    df = active[active["host_label"].astype(str) == host_label]
    if node is not None and "node" in df.columns:
        df = df[df["node"].astype(str) == str(node)]
    if gpu is not None and "gpu_id" in df.columns:
        df = df[df["gpu_id"].astype(str) == str(gpu)]
    return int(len(df))


def slot_key(host_label: str, node: str, gpu: str | int) -> str:
    return f"{host_label}|{node}|{gpu}"


def slot_key_for_plan(plan: dict[str, Any]) -> str:
    return slot_key(str(plan.get("host_label", "")), str(plan.get("node", "")), str(plan.get("gpu", "")))


def active_slot_quarantine(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = state.get("md_slot_quarantine", {})
    if not isinstance(raw, dict):
        return {}
    now = datetime.now()
    active: dict[str, dict[str, Any]] = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            continue
        until_text = str(row.get("until", "") or "")
        if until_text:
            try:
                if datetime.fromisoformat(until_text) <= now:
                    continue
            except ValueError:
                pass
        active[str(key)] = row
    return active


def compute_slots(
    gpus: pd.DataFrame,
    active: pd.DataFrame,
    blocked_hosts: set[str] | None = None,
    blocked_slots: set[str] | None = None,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    blocked_hosts = set(blocked_hosts or set())
    blocked_slots = set(blocked_slots or set())
    if gpus.empty:
        return slots
    active_by_node: dict[tuple[str, str], int] = {}
    active_by_gpu: dict[tuple[str, str, str], int] = {}
    for _, row in active.iterrows():
        key_node = (str(row.get("host_label", "")), str(row.get("node", "")))
        active_by_node[key_node] = active_by_node.get(key_node, 0) + 1
        key_gpu = (str(row.get("host_label", "")), str(row.get("node", "")), str(row.get("gpu_id", "")))
        active_by_gpu[key_gpu] = active_by_gpu.get(key_gpu, 0) + 1

    def estimate_external_cases(memory_used_mib: int, liuz_case_count: int, profile: ClusterProfile) -> int:
        if liuz_case_count <= 0:
            return 1 if memory_used_mib >= profile.idle_external_memory_threshold_mib else 0
        extra = memory_used_mib - liuz_case_count * profile.project_memory_per_case_mib
        return max(0, int(max(0, extra - profile.external_memory_slack_mib) / profile.external_memory_per_case_mib))

    gpu_rows = gpus.to_dict("records")
    # nvidia-smi process tables are not reliably available through the current
    # dashboard for all users, but memory usage is enough to avoid overfilling a
    # node.  A running GROMACS case on the 2080 nodes currently uses about
    # 160-190 MiB in the process table; use memory-derived occupancy only as a
    # placement hint and as an external-user estimate below, not as the node cap.
    estimated_by_gpu: dict[tuple[str, str, str], int] = {}
    for row in gpu_rows:
        key = (str(row.get("host_label", "")), str(row.get("node", "")), str(row.get("gpu", "")))
        try:
            mem = int(float(row.get("memory_used_mib", 0) or 0))
        except (TypeError, ValueError):
            mem = 0
        estimated_by_gpu[key] = max(active_by_gpu.get(key, 0), max(0, round(max(0, mem - 20) / 180.0)))
    external_by_gpu: dict[tuple[str, str, str], int] = {}
    gpu_free_by_gpu: dict[tuple[str, str, str], int] = {}

    for host_label in sorted({str(row.get("host_label", "")) for row in gpu_rows}):
        if host_label in DISABLED_MD_HOSTS:
            continue
        if host_label == "new":
            profile = production_profile_for("new")
            nodes = sorted({str(row.get("node", "")) for row in gpu_rows if str(row.get("host_label", "")) == "new"})
            for node in nodes:
                if node in DISABLED_NEW_NODES:
                    continue
                if profile.allowed_nodes and node not in set(profile.allowed_nodes):
                    continue
                if node in set(profile.disallowed_nodes):
                    continue
                node_gpus = [
                    row
                    for row in gpu_rows
                    if str(row.get("host_label", "")) == "new" and str(row.get("node", "")) == node
                ]
                cap = profile.node_cap_for_node(node)
                if cap <= 0:
                    continue
                used_liuz = active_by_node.get(("new", node), 0)
                policy_free = max(0, cap - used_liuz)
                if not node_gpus:
                    real_free = policy_free
                    per_gpu_target = cap
                else:
                    per_gpu_target = profile.per_gpu_target_for_node(node, node_cap=cap, gpu_count=len(node_gpus))
                    real_free = 0
                    for gpu_row in node_gpus:
                        gpu = str(gpu_row.get("gpu", "0"))
                        gpu_key = ("new", node, gpu)
                        liuz_count = active_by_gpu.get(gpu_key, 0)
                        try:
                            mem = int(float(gpu_row.get("memory_used_mib", 0) or 0))
                        except (TypeError, ValueError):
                            mem = 0
                        try:
                            visible_external = int(float(gpu_row.get("external_process_count", 0) or 0))
                        except (TypeError, ValueError):
                            visible_external = 0
                        external_est = max(visible_external, estimate_external_cases(mem, liuz_count, profile))
                        external_by_gpu[gpu_key] = external_est
                        gpu_free = max(0, per_gpu_target - liuz_count - external_est)
                        gpu_free_by_gpu[gpu_key] = gpu_free
                        real_free += gpu_free
                free = min(policy_free, real_free)
                if free <= 0:
                    continue
                node_gpus.sort(
                    key=lambda row: (
                        external_by_gpu.get(("new", node, str(row.get("gpu", ""))), 0),
                        active_by_gpu.get(("new", node, str(row.get("gpu", ""))), 0),
                        estimated_by_gpu.get(("new", node, str(row.get("gpu", ""))), 0),
                        int(float(row.get("util_percent", 999) or 999)),
                        str(row.get("gpu", "")),
                    )
                )
                placed = 0
                while placed < free:
                    candidate_rows = [
                        row
                        for row in node_gpus
                        if external_by_gpu.get(("new", node, str(row.get("gpu", "0"))), 0) == 0
                        and gpu_free_by_gpu.get(("new", node, str(row.get("gpu", "0"))), per_gpu_target) > 0
                    ]
                    if not candidate_rows:
                        break
                    candidate_rows.sort(
                        key=lambda row: (
                            active_by_gpu.get(("new", node, str(row.get("gpu", ""))), 0),
                            estimated_by_gpu.get(("new", node, str(row.get("gpu", ""))), 0),
                            int(float(row.get("util_percent", 999) or 999)),
                            str(row.get("gpu", "")),
                        )
                    )
                    gpu_row = candidate_rows[0]
                    gpu_key = ("new", node, str(gpu_row.get("gpu", "0")))
                    slots.append(
                        {
                            "host_label": "new",
                            "host": profile.host,
                            "node": node,
                            "gpu": str(gpu_row.get("gpu", "0")),
                            "threads": profile.threads_for_node(node),
                            "remote_root": profile.remote_root,
                            "module_preset": profile.module_preset_for_node(node),
                            "gmx_bin": profile.gmx_bin,
                            "launch_via_node": profile.launch_via_node,
                        }
                    )
                    estimated_by_gpu[gpu_key] = estimated_by_gpu.get(gpu_key, 0) + 1
                    active_by_gpu[gpu_key] = active_by_gpu.get(gpu_key, 0) + 1
                    gpu_free_by_gpu[gpu_key] = max(0, gpu_free_by_gpu.get(gpu_key, 0) - 1)
                    placed += 1
        elif host_label == "newnew":
            profile = production_profile_for("newnew")
            for row in gpu_rows:
                if str(row.get("host_label", "")) != "newnew":
                    continue
                gpu = str(row.get("gpu", ""))
                node = str(row.get("node", "aiserver")) or "aiserver"
                if profile.allowed_nodes and node not in set(profile.allowed_nodes):
                    continue
                if node in set(profile.disallowed_nodes):
                    continue
                util = int(float(row.get("util_percent", 0) or 0))
                mem = int(float(row.get("memory_used_mib", 0) or 0))
                gpu_key = ("newnew", node, gpu)
                active_count = active_by_gpu.get(gpu_key, 0)
                if active_count == 0 and (util >= profile.idle_skip_util_percent or mem >= profile.idle_skip_memory_mib):
                    continue
                if active_count > 0 and mem >= active_count * profile.project_memory_per_case_mib + profile.active_extra_memory_skip_mib:
                    continue
                per_gpu_target = profile.per_gpu_target_for_node(
                    node,
                    node_cap=profile.node_cap_for_node(node),
                    gpu_count=1,
                )
                free_on_gpu = max(0, per_gpu_target - active_count)
                for _ in range(free_on_gpu):
                    slots.append(
                        {
                            "host_label": "newnew",
                            "host": profile.host,
                            "node": node,
                            "gpu": gpu,
                            "threads": profile.threads_for_node(node),
                            "remote_root": profile.remote_root,
                            "module_preset": profile.module_preset_for_node(node),
                            "gmx_bin": profile.gmx_bin,
                            "launch_via_node": profile.launch_via_node,
                        }
                    )
                    active_by_gpu[gpu_key] = active_by_gpu.get(gpu_key, 0) + 1
                    active_count += 1
    def node_number(node: str) -> int:
        match = re.search(r"(\d+)$", str(node))
        return int(match.group(1)) if match else 999

    def gpu_number(gpu: str) -> int:
        try:
            return int(float(gpu))
        except (TypeError, ValueError):
            return 999

    def slot_sort_key(slot: dict[str, Any]) -> tuple[int, int, int, int]:
        host_label = str(slot.get("host_label", ""))
        node = str(slot.get("node", ""))
        gpu = str(slot.get("gpu", ""))
        # Prefer less-loaded nodes globally so newly idle node-8accbfac/8 and
        # newnew slots are used before adding more cases to a half-filled node.
        host_priority = 0 if host_label == "new" else 1
        return (
            active_by_node.get((host_label, node), 0),
            host_priority,
            node_number(node),
            gpu_number(gpu),
        )

    if blocked_hosts:
        slots = [slot for slot in slots if str(slot.get("host", "")) not in blocked_hosts]
    if blocked_slots:
        slots = [slot for slot in slots if slot_key_for_plan(slot) not in blocked_slots]
    return sorted(slots, key=slot_sort_key)


def bridge_dir_for_case(case_id: str) -> Path | None:
    for root in BRIDGE_DIRS:
        if (root / case_id).exists():
            return root
    return None


def upstream_cases_to_exclude(upstream: dict[str, Any]) -> set[str]:
    """Cases advanced upstream that should not be submitted in the same cycle."""

    decision = upstream.get("decision", {}) if isinstance(upstream, dict) else {}
    action = str(decision.get("action", "") or "")
    cases = {str(case) for case in (decision.get("cases", []) or []) if str(case)}
    if action == "rebuild_bridge":
        # A freshly rebuilt bridge is safe to submit in the same cycle once its
        # bridge directory is visible locally. Other upstream actions still need
        # a later cycle to collect/build concrete MD inputs.
        return {case for case in cases if bridge_dir_for_case(case) is None}
    return cases


def plan_md_submissions(
    backlog: pd.DataFrame,
    slots: list[dict[str, Any]],
    max_submit: int | None,
    state: dict[str, Any],
    extra_excluded: set[str] | None = None,
    allow_failed_retry: bool = False,
) -> list[dict[str, Any]]:
    if backlog.empty or not slots:
        return []
    submitted = set(state.get("submitted_md_cases", []))
    retry_counts = state.get("md_retry_counts", {})
    if not isinstance(retry_counts, dict):
        retry_counts = {}
    launch_fail_counts = state.get("md_launch_fail_counts", {})
    if not isinstance(launch_fail_counts, dict):
        launch_fail_counts = {}
    if extra_excluded:
        submitted |= set(extra_excluded)
    rows = backlog.sort_values("case_seq" if "case_seq" in backlog.columns else "case_id").to_dict("records")
    plans: list[dict[str, Any]] = []
    limit = min(len(slots), max_submit if max_submit is not None else len(slots))
    slot_index = 0
    for row in rows:
        if len(plans) >= limit:
            break
        case_id = str(row.get("case_id", ""))
        runtime_status = str(row.get("runtime_status", "")).strip().lower()
        retry_count = int(retry_counts.get(case_id, 0) or 0)
        is_failed_retry = runtime_status == "failed_stage"
        is_not_deployed = runtime_status == "not_deployed"
        if not case_id:
            continue
        if is_failed_retry and not allow_failed_retry:
            continue
        if case_id in submitted and not (is_failed_retry or is_not_deployed):
            continue
        if is_failed_retry and retry_count >= 1:
            continue
        if is_failed_retry and int(launch_fail_counts.get(case_id, 0) or 0) >= 2:
            continue
        bridge = bridge_dir_for_case(case_id)
        if bridge is None:
            continue
        slot = slots[slot_index]
        slot_index += 1
        plans.append(
            {
                "case_id": case_id,
                "case_seq": row.get("case_seq", case_seq(case_id)),
                "host_label": slot["host_label"],
                "host": slot["host"],
                "node": slot["node"],
                "gpu": slot["gpu"],
                "threads": slot["threads"],
                "remote_root": slot["remote_root"],
                "module_preset": slot["module_preset"],
                "gmx_bin": slot["gmx_bin"],
                "launch_via_node": slot["launch_via_node"],
                "bridge_dir": str(bridge),
                "action": "launch_manual_md",
                "retry_failed_stage": is_failed_retry,
                "retry_count": retry_count,
                "failure_reason": row.get("failure_reason", ""),
            }
        )
        if slot_index >= len(slots):
            break
    return plans


def split_list_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def nonempty_cell(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(str(value).strip())


def remote_case_dir_from_log_path(path: str) -> str:
    if not path:
        return ""
    base = posixpath.basename(path)
    parent = posixpath.dirname(path)
    if base in {"manual.out", "manual.err", "run_manual.sh", "manual.pid"}:
        return parent
    stage = posixpath.basename(parent)
    if stage in {"steep", "rlx", "pr", "nvt_prod"}:
        return posixpath.dirname(parent)
    return parent


def host_label_for_remote_path(path: str) -> str:
    if path.startswith("${PELCB100_REMOTE_ROOT}/"):
        return "newnew"
    if path.startswith("${PELCB100_REMOTE_HOME}/"):
        return "new"
    return ""


def host_for_label(host_label: str) -> str:
    return NEWNEW_SERVER if host_label == "newnew" else NEW_SERVER if host_label == "new" else ""


def latest_remote_case_dir(row: dict[str, Any], preferred_host_label: str | None = None) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for path in split_list_cell(row.get("latest_failure_logs", "")):
        case_dir = remote_case_dir_from_log_path(path)
        host_label = host_label_for_remote_path(case_dir)
        if case_dir:
            candidates.append((host_label, case_dir))
    for path in split_list_cell(row.get("deployed_paths", "")):
        host_label = host_label_for_remote_path(path)
        candidates.append((host_label, path))
    if preferred_host_label:
        preferred = [item for item in candidates if item[0] == preferred_host_label]
        if preferred:
            return preferred[-1]
    return candidates[-1] if candidates else ("", "")


def recovery_attempts(state: dict[str, Any]) -> dict[str, int]:
    raw = state.get("failed_recovery_attempts", {})
    if not isinstance(raw, dict):
        return {}
    attempts: dict[str, int] = {}
    for key, value in raw.items():
        try:
            attempts[str(key)] = int(value or 0)
        except (TypeError, ValueError):
            attempts[str(key)] = 0
    return attempts


def classify_failed_case(row: dict[str, Any], state: dict[str, Any], duplicate_ids: set[str]) -> dict[str, Any]:
    case_id = str(row.get("case_id", "") or "")
    text = "\n".join(
        [
            str(row.get("failure_reason", "")),
            str(row.get("failure_stage_name", "")),
            str(row.get("failure_tail", "")),
            str(row.get("latest_failure_logs", "")),
            str(row.get("deployed_hosts", "")),
        ]
    )
    lower = text.lower()
    attempts = recovery_attempts(state).get(case_id, 0)
    base = {
        "case_id": case_id,
        "classification": "manual_review",
        "recovery_kind": "",
        "decision_reason": "unknown_failed_stage",
        "failed_recovery_attempts": attempts,
        "remote_case_dir": "",
        "source_host_label": "",
        "source_host": "",
        "requires_slot": False,
        "target_host_label": "",
        "slot_key": "",
        "avoid_gpu": "",
        "planned_action": "none",
    }
    if not case_id:
        base["decision_reason"] = "missing_case_id"
        return base
    if case_id in duplicate_ids or nonempty_cell(row.get("duplicate_reason", "")):
        base["classification"] = "blocked_duplicate"
        base["decision_reason"] = "duplicate_or_conflict_runtime_state"
        return base
    if attempts >= 1:
        base["decision_reason"] = "recovery_attempt_limit_reached"
        return base

    stage_text = str(row.get("failure_stage_name", "") or "").lower()
    deployed_hosts = str(row.get("deployed_hosts", "") or "").lower()
    if (
        "newnew" in deployed_hosts
        and "rlx" in stage_text
        and "berendsen barostat" in lower
        and "too many warnings" in lower
        and "maxwarn" in lower
    ):
        host_label, case_dir = latest_remote_case_dir(row, preferred_host_label="newnew")
        base.update(
            {
                "classification": "recoverable",
                "recovery_kind": "newnew_rlx_maxwarn",
                "decision_reason": "GROMACS 2026 rejects known Berendsen rlx warning unless RLX_MAXWARN=1",
                "remote_case_dir": case_dir,
                "source_host_label": host_label,
                "source_host": host_for_label(host_label),
                "requires_slot": True,
                "target_host_label": "newnew",
                "planned_action": "await_slot",
            }
        )
        return base

    if "segmentation fault" in lower and ("steep" in stage_text or "/steep" in lower):
        host_label, case_dir = latest_remote_case_dir(row)
        base.update(
            {
                "classification": "recoverable",
                "recovery_kind": "steep_segfault_redeploy",
                "decision_reason": "steep mdrun segfault; recover by redeploying once to a non-quarantined slot",
                "remote_case_dir": case_dir,
                "source_host_label": host_label,
                "source_host": host_for_label(host_label),
                "requires_slot": True,
                "target_host_label": "",
                "planned_action": "await_slot",
            }
        )
        return base

    if "smalloc" in lower and ("nvt_prod" in stage_text or "nvt_prod" in lower):
        host_label, case_dir = latest_remote_case_dir(row)
        source_node = str(row.get("node", "") or "")
        if not source_node and host_label == "new":
            base.update(
                {
                    "classification": "manual_review",
                    "recovery_kind": "nvt_smalloc_notunepme",
                    "decision_reason": "nvt_prod smalloc is recoverable only after source compute node is known for in-place resume",
                    "remote_case_dir": case_dir,
                    "source_host_label": host_label,
                    "source_host": host_for_label(host_label),
                }
            )
            return base
        base.update(
            {
                "classification": "recoverable",
                "recovery_kind": "nvt_smalloc_notunepme",
                "decision_reason": "nvt_prod smalloc with old template; refresh -notunepme and resume in place",
                "remote_case_dir": case_dir,
                "source_host_label": host_label,
                "source_host": host_for_label(host_label),
                "requires_slot": False,
                "target_host_label": host_label,
                "planned_action": "resume_in_place",
            }
        )
        return base

    gpu_match = re.search(r"requested mdrun to use GPUs? with IDs?\s+([0-9, ]+)", text, re.IGNORECASE)
    if (
        "new" in deployed_hosts
        and ("nvt_prod" in stage_text or "nvt_prod" in lower)
        and "incompatible gpus" in lower
        and "cudaerrormemoryallocation" in lower
    ):
        host_label, case_dir = latest_remote_case_dir(row, preferred_host_label="new")
        avoid_gpu = ""
        if gpu_match:
            avoid_gpu = gpu_match.group(1).split(",")[0].strip()
        base.update(
            {
                "classification": "recoverable",
                "recovery_kind": "nvt_gpu_incompatible_resume_slot",
                "decision_reason": "nvt_prod failed before mdrun due GPU compatibility/CUDA memory sanity check; resume existing run dir on a different new-server slot",
                "remote_case_dir": case_dir,
                "source_host_label": host_label,
                "source_host": host_for_label(host_label),
                "requires_slot": True,
                "target_host_label": "new",
                "avoid_gpu": avoid_gpu,
                "planned_action": "await_slot",
            }
        )
        return base

    return base


def failed_case_recovery_rows(
    runtime_cases: pd.DataFrame,
    duplicates: pd.DataFrame,
    slots: list[dict[str, Any]],
    state: dict[str, Any],
    max_recovery: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if runtime_cases.empty or "runtime_status" not in runtime_cases.columns:
        return [], []
    duplicate_ids = duplicate_case_ids(duplicates)
    failed = runtime_cases[runtime_cases["runtime_status"].astype(str).str.lower() == "failed_stage"]
    rows = [classify_failed_case(row, state, duplicate_ids) for row in failed.to_dict("records")]
    slot_pool = list(slots)
    plans: list[dict[str, Any]] = []
    for row in rows:
        if len(plans) >= max(0, max_recovery):
            break
        if row.get("classification") != "recoverable":
            continue
        if row.get("requires_slot"):
            target = str(row.get("target_host_label", "") or "")
            avoid_gpu = str(row.get("avoid_gpu", "") or "")
            candidate_index = next(
                (
                    idx
                    for idx, slot in enumerate(slot_pool)
                    if not target or str(slot.get("host_label", "")) == target
                    if not avoid_gpu or str(slot.get("gpu", "")) != avoid_gpu
                ),
                None,
            )
            if candidate_index is None:
                row["planned_action"] = "blocked_no_slot"
                continue
            slot = slot_pool.pop(candidate_index)
            row["slot_key"] = slot_key_for_plan(slot)
            row["planned_action"] = (
                "resume_on_slot" if row.get("recovery_kind") == "nvt_gpu_incompatible_resume_slot" else "redeploy"
            )
            plan = {**row, **slot, "action": "recover_failed_stage", "bridge_dir": str(bridge_dir_for_case(str(row["case_id"])) or "")}
            if row.get("recovery_kind") != "nvt_gpu_incompatible_resume_slot" and not plan["bridge_dir"]:
                row["planned_action"] = "blocked_missing_bridge"
                continue
            plans.append(plan)
        else:
            plans.append({**row, "action": "recover_failed_stage"})
    return rows, plans


def write_recovery_outputs(output: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "classification",
        "recovery_kind",
        "decision_reason",
        "failed_recovery_attempts",
        "planned_action",
        "source_host_label",
        "source_host",
        "remote_case_dir",
        "requires_slot",
        "target_host_label",
        "avoid_gpu",
        "slot_key",
    ]
    path = output / "recovery_classification.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    lines = ["# Failed Case Recovery Classification", "", f"- Generated: `{iso_now()}`", ""]
    if not rows:
        lines.append("- No failed cases found.")
    for row in rows:
        lines.append(
            f"- `{row.get('case_id')}`: `{row.get('classification')}` / `{row.get('recovery_kind') or 'none'}` "
            f"action `{row.get('planned_action')}` reason `{row.get('decision_reason')}`"
        )
    write_text(output / "recovery_classification.md", "\n".join(lines) + "\n")


def fetch_recovery_evidence(plan: dict[str, Any], recovery_root: Path) -> Path:
    case_id = str(plan.get("case_id", "unknown_case"))
    case_dir = str(plan.get("remote_case_dir", "") or "")
    host = str(plan.get("source_host", "") or "")
    evidence_dir = recovery_root / case_id / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    write_json(evidence_dir / "recovery_plan.json", plan)
    if not case_dir or not host:
        write_text(evidence_dir / "evidence_error.txt", "missing remote case dir or source host\n")
        return evidence_dir
    files = [
        "run_manual.sh",
        "manual.pid",
        "manual.out",
        "manual.err",
        "steep/steep.log",
        "rlx/rlx.log",
        "pr/pr.log",
        "nvt_prod/nvt_prod.log",
    ]
    for rel in files:
        local = evidence_dir / rel.replace("/", "__")
        cmd = f"cd {sh_quote(case_dir)} && if [ -e {sh_quote(rel)} ]; then cat {sh_quote(rel)}; fi"
        proc = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, f"bash -lc {sh_quote(cmd)}"], timeout=120)
        if proc.returncode == 0 and proc.stdout:
            write_text(local, proc.stdout)
        elif proc.stderr:
            write_text(local.with_suffix(local.suffix + ".stderr"), proc.stderr)
    return evidence_dir


def backup_remote_run_manual(plan: dict[str, Any], timestamp: str) -> dict[str, Any]:
    case_dir = str(plan.get("remote_case_dir", "") or "")
    host = str(plan.get("source_host", "") or "")
    if not case_dir or not host:
        return {"returncode": 2, "stdout": "", "stderr": "missing remote case dir or source host"}
    cmd = (
        f"cd {sh_quote(case_dir)} && "
        "if [ -f run_manual.sh ]; then "
        f"cp -p run_manual.sh run_manual.sh.bak_recovery_{timestamp}; "
        f"echo run_manual.sh.bak_recovery_{timestamp}; "
        "else echo missing_run_manual; fi"
    )
    proc = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, f"bash -lc {sh_quote(cmd)}"], timeout=60)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def execute_in_place_notunepme_recovery(plan: dict[str, Any], timestamp: str) -> dict[str, Any]:
    case_dir = str(plan.get("remote_case_dir", "") or "")
    host = str(plan.get("source_host", "") or "")
    if not case_dir or not host:
        return {"returncode": 2, "stdout": "", "stderr": "missing remote case dir or source host"}
    patch = (
        "set -euo pipefail; "
        f"cd {sh_quote(case_dir)}; "
        "test -f run_manual.sh; "
        f"cp -p run_manual.sh run_manual.sh.bak_notunepme_recovery_{timestamp}; "
        "if ! grep -q -- '-notunepme' run_manual.sh; then "
        "perl -0pi -e 's/(nvt_prod\\)\\s+STAGE_GPU_EXTRA=\"[^\"]*-bonded gpu)([^\"]*\")/$1 -notunepme$2/s' run_manual.sh; "
        "fi; "
        "grep -q -- '-notunepme' run_manual.sh; "
        "test -s nvt_prod/nvt_prod.cpt || test -s pr/pr.cpt || test -s rlx/rlx.cpt; "
        "if [ -s manual.pid ] && kill -0 $(cat manual.pid) >/dev/null 2>&1; then echo already_running pid=$(cat manual.pid); exit 20; fi; "
        "ENABLE_GPU_EXTRA=1 MERGE_PARTS=1 nohup bash run_manual.sh > manual.out 2> manual.err < /dev/null & "
        "echo $! > manual.pid; sleep 2; ps -p $(cat manual.pid) -o pid,stat,etime,cmd"
    )
    proc = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, f"bash -lc {sh_quote(patch)}"], timeout=120)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def execute_existing_case_slot_recovery(plan: dict[str, Any], timestamp: str) -> dict[str, Any]:
    case_dir = str(plan.get("remote_case_dir", "") or "")
    host = str(plan.get("host", "") or "")
    if not case_dir or not host:
        return {"returncode": 2, "stdout": "", "stderr": "missing remote case dir or target host"}
    patch = (
        "set -euo pipefail; "
        f"cd {sh_quote(case_dir)}; "
        "test -f run_manual.sh; "
        f"cp -p run_manual.sh run_manual.sh.bak_gpu_recovery_{timestamp}; "
        f"perl -0pi -e 's/^MDRUN_GPU_ARGS=.*/MDRUN_GPU_ARGS=\"-gpu_id {str(plan.get('gpu', '0'))} -nb gpu\"/m' run_manual.sh; "
        "grep -q -- '-notunepme' run_manual.sh || "
        "perl -0pi -e 's/(nvt_prod\\)\\s+STAGE_GPU_EXTRA=\"[^\"]*-bonded gpu)([^\"]*\")/$1 -notunepme$2/s' run_manual.sh; "
        "grep -q -- '-notunepme' run_manual.sh; "
        f"grep -q '^MDRUN_GPU_ARGS=\"-gpu_id {str(plan.get('gpu', '0'))} -nb gpu\"' run_manual.sh; "
        "test -s nvt_prod/nvt_prod.tpr || test -s pr/pr.cpt || test -s pr/pr.gro; "
    )
    launch_inner = patch + remote_launch_command(case_dir)
    if plan.get("launch_via_node"):
        command = f"ssh -o BatchMode=yes -o ConnectTimeout=20 {sh_quote(str(plan['node']))} {sh_quote('bash -lc ' + sh_quote(launch_inner))}"
    else:
        command = f"bash -lc {sh_quote(launch_inner)}"
    proc = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command], timeout=120)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "remote_case_dir": case_dir}


def execute_recovery_plan(plan: dict[str, Any], package_root: Path, recovery_root: Path, timestamp: str) -> dict[str, Any]:
    evidence_dir = fetch_recovery_evidence(plan, recovery_root)
    backup = backup_remote_run_manual(plan, timestamp)
    kind = str(plan.get("recovery_kind", ""))
    if kind in {"newnew_rlx_maxwarn", "steep_segfault_redeploy"}:
        run_tag = f"recovery_{timestamp}_{plan['case_id']}"
        run_dir_case = generate_package(str(plan["case_id"]), plan, package_root, run_tag)
        launch = deploy_and_launch(plan, run_dir_case, f"recovery_{timestamp}")
        return {
            "returncode": 0,
            "mode": "redeploy",
            "evidence_dir": str(evidence_dir),
            "backup": backup,
            **launch,
        }
    if kind == "nvt_smalloc_notunepme":
        launch = execute_in_place_notunepme_recovery(plan, timestamp)
        return {"mode": "resume_in_place", "evidence_dir": str(evidence_dir), "backup": backup, **launch}
    if kind == "nvt_gpu_incompatible_resume_slot":
        launch = execute_existing_case_slot_recovery(plan, timestamp)
        return {"mode": "resume_existing_on_new_slot", "evidence_dir": str(evidence_dir), "backup": backup, **launch}
    return {"returncode": 2, "mode": "unsupported", "evidence_dir": str(evidence_dir), "backup": backup, "stderr": f"unsupported recovery kind {kind}"}


def generate_package(case_id: str, plan: dict[str, Any], run_base: Path, run_tag: str) -> Path:
    bridge = Path(str(plan["bridge_dir"]))
    if not bridge.exists():
        raise FileNotFoundError(f"bridge package root missing for {case_id}: {bridge}")
    command = [
        sys.executable,
        str(PIPELINE_DIR / "lcb_md_submit.py"),
        "--bridge-dir",
        str(bridge),
        "--run-base",
        str(run_base),
        "--mdp-dir",
        str(PIPELINE_DIR / "mdp"),
        "--template",
        str(PIPELINE_DIR / "manual_template.sh"),
        "--script-name",
        "run_manual.sh",
        "--module-preset",
        str(plan["module_preset"]),
        "--gmx-bin",
        str(plan["gmx_bin"]),
        "--cpus",
        str(plan["threads"]),
        "--gpu-id",
        str(plan["gpu"]),
        "--cases",
        case_id,
        "--run-tag",
        run_tag,
        "--dry-run",
        "--overwrite",
    ]
    proc = run(command, timeout=180, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"lcb_md_submit failed for {case_id}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    manifests = sorted(run_base.glob(f"lcb_md_pipeline_manifest_{run_tag}.json"))
    if not manifests:
        raise RuntimeError(f"no pipeline manifest generated for {case_id}")
    payload = json.loads(manifests[-1].read_text(encoding="utf-8"))
    rows = [row for row in payload if row.get("case_id") == case_id]
    if not rows:
        raise RuntimeError(f"generated manifest does not include {case_id}: {manifests[-1]}")
    run_dir = Path(rows[0]["run_dir"])
    if not (run_dir / "run_manual.sh").exists():
        raise RuntimeError(f"run_manual.sh missing in generated package: {run_dir}")
    return run_dir


def remote_launch_command(remote_case_dir: str) -> str:
    return (
        "set -euo pipefail; "
        f"cd {sh_quote(remote_case_dir)}; "
        "if [ -s manual.pid ] && kill -0 $(cat manual.pid) >/dev/null 2>&1; then "
        "echo already_running pid=$(cat manual.pid); exit 20; fi; "
        "nohup setsid env ENABLE_GPU_EXTRA=1 MERGE_PARTS=1 bash run_manual.sh "
        "> manual.out 2> manual.err < /dev/null & "
        "echo $! > manual.pid; disown $(cat manual.pid) >/dev/null 2>&1 || true; sleep 2; "
        "ps -p $(cat manual.pid) -o pid,stat,etime,cmd"
    )


def deploy_and_launch(plan: dict[str, Any], run_dir: Path, batch_tag: str) -> dict[str, Any]:
    remote_parent = f"{plan['remote_root']}/{PROJECT_ID}_supervisor_{batch_tag}"
    remote_case_dir = f"{remote_parent}/{run_dir.name}"
    host = str(plan["host"])
    mkdir = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, f"mkdir -p {sh_quote(remote_parent)}"], timeout=60)
    if mkdir.returncode != 0:
        raise RuntimeError(f"remote mkdir failed: {mkdir.stderr}")
    scp = run(["scp", "-r", str(run_dir), f"{host}:{remote_parent}/"], timeout=600)
    if scp.returncode != 0:
        raise RuntimeError(f"scp failed for {run_dir}\nSTDOUT:\n{scp.stdout}\nSTDERR:\n{scp.stderr}")
    launch_inner = remote_launch_command(remote_case_dir)
    if plan.get("launch_via_node"):
        command = f"ssh -o BatchMode=yes -o ConnectTimeout=20 {sh_quote(str(plan['node']))} {sh_quote('bash -lc ' + sh_quote(launch_inner))}"
    else:
        command = f"bash -lc {sh_quote(launch_inner)}"
    proc = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command], timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"launch failed for {plan['case_id']}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return {"remote_case_dir": remote_case_dir, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def duplicate_case_ids(duplicates: pd.DataFrame) -> set[str]:
    if duplicates.empty or "case_id" not in duplicates.columns:
        return set()
    return {str(item) for item in duplicates["case_id"].dropna().tolist()}


def plan_restarts(active: pd.DataFrame, duplicates: pd.DataFrame) -> list[dict[str, Any]]:
    if active.empty:
        return []
    duplicate_ids = duplicate_case_ids(duplicates)
    plans: list[dict[str, Any]] = []
    for row in active.to_dict("records"):
        case_id = str(row.get("case_id", ""))
        stage = str(row.get("stage", ""))
        cmd = str(row.get("cmd", ""))
        cwd = str(row.get("cwd", ""))
        if not case_id or case_id in duplicate_ids:
            continue
        if stage not in {"pr", "nvt_prod"}:
            continue
        if "-bonded gpu" in cmd:
            continue
        if "lcb_manual_runs" not in cwd:
            continue
        host_label = str(row.get("host_label", ""))
        if host_label not in {"new", "newnew"}:
            continue
        plans.append(
            {
                "case_id": case_id,
                "host_label": host_label,
                "host": row.get("host", NEW_SERVER if host_label == "new" else NEWNEW_SERVER),
                "node": row.get("node", ""),
                "gpu": row.get("gpu_id", ""),
                "pid": row.get("pid", ""),
                "stage": stage,
                "cwd": cwd,
                "reason": "missing -bonded gpu",
            }
        )
    return plans


def execute_restart(plan: dict[str, Any]) -> dict[str, Any]:
    cwd = str(plan["cwd"])
    stage = str(plan["stage"])
    pid = str(plan["pid"])
    case_dir = str(Path(cwd).parent)
    check_cmd = (
        f"cd {sh_quote(cwd)} && "
        f"test -s {sh_quote(stage + '.cpt')} && "
        f"! grep -q 'Finished mdrun' {sh_quote(stage + '.log')} 2>/dev/null"
    )
    host = str(plan["host"])
    if plan["host_label"] == "new":
        check = f"ssh -o BatchMode=yes -o ConnectTimeout=20 {sh_quote(str(plan['node']))} {sh_quote(check_cmd)}"
        restart_inner = (
            f"kill -INT {sh_quote(pid)} || true; sleep 30; "
            f"if kill -0 {sh_quote(pid)} >/dev/null 2>&1; then echo pid_still_running; exit 21; fi; "
            + remote_launch_command(case_dir)
        )
        restart = f"ssh -o BatchMode=yes -o ConnectTimeout=20 {sh_quote(str(plan['node']))} {sh_quote('bash -lc ' + sh_quote(restart_inner))}"
    else:
        check = check_cmd
        restart_inner = (
            f"kill -INT {sh_quote(pid)} || true; sleep 30; "
            f"if kill -0 {sh_quote(pid)} >/dev/null 2>&1; then echo pid_still_running; exit 21; fi; "
            + remote_launch_command(case_dir)
        )
        restart = f"bash -lc {sh_quote(restart_inner)}"
    check_proc = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, check], timeout=60)
    if check_proc.returncode != 0:
        raise RuntimeError(f"restart precheck failed for {plan['case_id']}: {check_proc.stderr.strip()}")
    proc = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, restart], timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"restart failed for {plan['case_id']}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return {"stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def find_active_postprocess(state: dict[str, Any]) -> list[str]:
    jobs = state.get("postprocess_jobs", [])
    if isinstance(jobs, list):
        return [str(job) for job in jobs if str(job).strip()]
    return []


def run_closure_status(jobs: list[str]) -> dict[str, Any]:
    command = [sys.executable, str(CLOSURE), "--status-only"]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    if jobs:
        command.extend(["--jobs", ",".join(jobs)])
    proc = run(command, timeout=180, cwd=ROOT)
    latest = latest_file(CLOSURE_DIR, "closure_status_*.json")
    payload = read_json(latest, {}) if latest else {}
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "json": str(latest) if latest else "", "payload": payload}


def latest_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    files = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def orchestrator_plan_needed(cases: pd.DataFrame) -> int:
    if cases.empty or "ready_for_analysis" not in cases.columns:
        return 0
    return int(cases["ready_for_analysis"].apply(is_truthy).sum())


def submit_postprocess(source_host: str = "all", execute: bool = False, max_cases: int | None = None) -> dict[str, Any]:
    command = [sys.executable, str(ORCHESTRATOR), "--source-host", source_host]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    if max_cases is not None:
        command.extend(["--max-cases", str(max_cases)])
    if execute:
        command.append("--execute")
    proc = run(command, timeout=900, cwd=ROOT)
    job_ids = re.findall(r"Submitted batch job\s+(\d+)", proc.stdout + "\n" + proc.stderr)
    latest_run = latest_orchestrator_run()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "job_ids": job_ids,
        "run_dir": str(latest_run) if latest_run else "",
    }


def latest_orchestrator_run_in(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir() and re.match(r"\d{8}_\d{6}$", path.name)]
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)[0] if dirs else None


def latest_orchestrator_run() -> Path | None:
    return latest_orchestrator_run_in(ORCHESTRATOR_DIR)


def submit_postprocess_single(
    case_id: str,
    suite: str,
    output: Path,
    execute: bool = False,
    source_host: str = "all",
) -> dict[str, Any]:
    """Submit one suite for one case by feeding orchestrator a scoped dashboard copy."""
    if suite != "baseline_density":
        raise ValueError(f"first postprocess gate only supports baseline_density, got {suite}")
    runtime_cases = read_csv(RUNTIME_DIR / "runtime_cases.csv")
    if runtime_cases.empty or "case_id" not in runtime_cases.columns:
        return {
            "returncode": 2,
            "stdout": "",
            "stderr": f"missing runtime_cases.csv in {RUNTIME_DIR}",
            "job_ids": [],
            "job_id": "",
            "run_dir": "",
        }
    one = runtime_cases[runtime_cases["case_id"].astype(str) == case_id].copy()
    if one.empty:
        return {
            "returncode": 2,
            "stdout": "",
            "stderr": f"case {case_id} not found in {RUNTIME_DIR / 'runtime_cases.csv'}",
            "job_ids": [],
            "job_id": "",
            "run_dir": "",
        }
    status_columns = {
        "baseline_density": "status_density",
        "baseline_rg": "status_Rg",
        "baseline_rdf": "status_RDF",
        "deep_rdf": "status_RDF",
        "msd_monomer_chain": "status_MSD",
        "deep_shape": "status_shape",
        "deep_dynamics": "status_dynamics",
        "deep_contact": "status_contact",
    }
    for col in status_columns.values():
        if col not in one.columns:
            one[col] = "OK"
        else:
            one[col] = "OK"
    one["status_density"] = "MISSING"
    one["missing_postprocess_observables"] = "density"
    one["ready_for_analysis"] = True
    one["strict_done"] = True

    trigger_root = output / "first_postprocess_gate" / case_id / f"trigger_{now_stamp()}"
    dashboard_dir = trigger_root / "dashboard"
    orchestrator_dir = trigger_root / "orchestrator"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    one.to_csv(dashboard_dir / "runtime_cases.csv", index=False, encoding="utf-8-sig")
    command = [
        sys.executable,
        str(ORCHESTRATOR),
        "--dashboard-dir",
        str(dashboard_dir),
        "--output-dir",
        str(orchestrator_dir),
        "--source-host",
        source_host,
        "--max-cases",
        "1",
    ]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    if execute:
        command.append("--execute")
    proc = run(command, timeout=900, cwd=ROOT)
    job_ids = re.findall(r"Submitted batch job\s+(\d+)", proc.stdout + "\n" + proc.stderr)
    latest_run = latest_orchestrator_run_in(orchestrator_dir)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "job_ids": job_ids,
        "job_id": job_ids[0] if job_ids else "",
        "run_dir": str(latest_run) if latest_run else "",
        "trigger_root": str(trigger_root),
        "dashboard_dir": str(dashboard_dir),
        "suite": suite,
        "case_id": case_id,
    }


def current_postprocess_plan_csv(state: dict[str, Any]) -> Path | None:
    run_dir = str(state.get("postprocess_orchestrator_run_dir", "") or "")
    if not run_dir:
        trigger = state.get("first_postprocess_gate_trigger")
        if isinstance(trigger, dict):
            run_dir = str(trigger.get("run_dir", "") or "")
    if not run_dir:
        return None
    path = Path(run_dir) / "full_postprocess_plan.csv"
    return path if path.exists() else None


def run_closure_all(jobs: list[str], execute: bool, plan_csv: Path | None = None) -> dict[str, Any]:
    if execute and plan_csv is None:
        return {
            "returncode": 2,
            "stdout": "",
            "stderr": "refusing closure apply without current full_postprocess_plan.csv",
            "merge_plan": "",
            "applied_merge_manifest": "",
            "plan_csv": "",
        }
    command = [sys.executable, str(CLOSURE), "--all"]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    if jobs:
        command.extend(["--jobs", ",".join(jobs)])
    if execute and plan_csv is not None:
        command.extend(["--apply-merge", "--plan-csv", str(plan_csv)])
    if not execute:
        command = [sys.executable, str(CLOSURE), "--status-only"]
        if CONFIG_PATH:
            command.extend(["--config", str(CONFIG_PATH)])
        if jobs:
            command.extend(["--jobs", ",".join(jobs)])
    proc = run(command, timeout=1800, cwd=ROOT)
    latest_merge = latest_file(CLOSURE_DIR, "merge_plan_*.json")
    latest_apply = latest_file(CLOSURE_DIR, "applied_merge_manifest_*.json")
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "merge_plan": str(latest_merge) if latest_merge else "",
        "applied_merge_manifest": str(latest_apply) if latest_apply else "",
        "plan_csv": str(plan_csv) if plan_csv else "",
    }


def rebuild_overview() -> dict[str, Any]:
    command = [sys.executable, str(OVERVIEW)]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    proc = run(command, timeout=600, cwd=ROOT)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}


def refresh_publication_assets() -> dict[str, Any]:
    command = [sys.executable, str(PUBLICATION_ASSETS)]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    proc = run(command, timeout=600, cwd=ROOT)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
    }


def run_report() -> dict[str, Any]:
    command = [sys.executable, str(REPORT)]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    proc = run(command, timeout=600, cwd=ROOT)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "ok": proc.returncode == 0}


def discover_project_batch_dates(observable: str) -> list[str]:
    prefix = f"{PROJECT_ID}_{observable}_"
    dates: list[str] = []
    for path in sorted((SCRIPT_DIR / "output").glob(f"{prefix}*")):
        if path.is_dir() and path.name.startswith(prefix):
            suffix = path.name[len(prefix):]
            if suffix:
                dates.append(suffix)
    return dates


def rebuild_postprocess_aggregates() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    rdf_migration_roots: list[str] = [
        str(SCRIPT_DIR / "output" / f"{PROJECT_ID}_integrated" / "RDF")
    ]
    for date_tag in discover_project_batch_dates("rdf"):
        rdf_migration_roots.append(str(SCRIPT_DIR / "output" / f"{PROJECT_ID}_rdf_{date_tag}"))
    if rdf_migration_roots:
        proc = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "tools" / "normalize_intrachain_pair_distribution.py"),
                *rdf_migration_roots,
            ],
            timeout=900,
            cwd=ROOT,
            env=project_subprocess_env(),
        )
        results.append(
            {
                "observable": "rdf_intrachain_pair_distribution_migration",
                "date_tag": "",
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
    for observable in ["rg", "msd", "rdf", "dynamics"]:
        for date_tag in discover_project_batch_dates(observable):
            command = [
                sys.executable,
                str(SCRIPT_DIR / "rebuild_canonical_aggregates.py"),
                "--output-root",
                str(SCRIPT_DIR / "output"),
                "--date-tag",
                date_tag,
                "--observables",
                observable,
            ]
            proc = run(command, timeout=600, cwd=ROOT, env=project_subprocess_env())
            results.append(
                {
                    "observable": observable,
                    "date_tag": date_tag,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )

    shape_raw = SCRIPT_DIR / "output" / f"{PROJECT_ID}_integrated" / "ShapeTensor" / "raw"
    shape_out = SCRIPT_DIR / "output" / f"{PROJECT_ID}_integrated" / "ShapeTensor"
    if shape_raw.exists():
        proc = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "aggregate_shape_tensor.py"),
                "--batch-dir",
                str(shape_raw),
                "--output-dir",
                str(shape_out),
            ],
            timeout=600,
            cwd=ROOT,
        )
        results.append({"observable": "shape", "date_tag": "", "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})

    intrachain = SCRIPT_DIR / "output" / f"{PROJECT_ID}_integrated" / "IntrachainDynamics"
    case_dirs = sorted((intrachain / "cases").glob("*")) if (intrachain / "cases").exists() else []
    cases = [path.name for path in case_dirs if path.is_dir()]
    if cases:
        proc = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "analyze_intrachain_dynamics.py"),
                "--batch-dir",
                str(intrachain),
                "--aggregate-only",
                "--cases",
                ",".join(cases),
            ],
            timeout=600,
            cwd=ROOT,
        )
        results.append({"observable": "analyze_intrachain_dynamics.py", "date_tag": "", "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})
        proc = run(
            [
                sys.executable,
                str(SCRIPT_DIR / "compute_monomer_msd.py"),
                "--batch-dir",
                str(intrachain),
                "--aggregate-only",
                "--cases",
                *cases,
            ],
            timeout=600,
            cwd=ROOT,
        )
        results.append({"observable": "compute_monomer_msd.py", "date_tag": "", "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})
        output_root = intrachain.parents[1]
        msd_roots = discover_clean_g3_msd_roots(output_root)
        if msd_roots:
            try:
                manifest = run_clean_g3_backfill(intrachain, msd_roots)
                results.append(
                    {
                        "observable": "backfill_clean_intrachain_g3.py",
                        "date_tag": "",
                        "returncode": 0,
                        "stdout": json.dumps(manifest, sort_keys=True),
                        "stderr": "",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "observable": "backfill_clean_intrachain_g3.py",
                        "date_tag": "",
                        "returncode": 1,
                        "stdout": "",
                        "stderr": str(exc),
                    }
                )
        else:
            results.append(
                {
                    "observable": "backfill_clean_intrachain_g3.py",
                    "date_tag": "",
                    "returncode": 0,
                    "stdout": "skipped: no o7clean_msd* roots discovered",
                    "stderr": "",
                }
            )

    dynamics_dates = discover_project_batch_dates("dynamics")
    for date_tag in dynamics_dates:
        dyn_root = SCRIPT_DIR / "output" / f"{PROJECT_ID}_dynamics_{date_tag}"
        case_dirs = sorted((dyn_root / "cases").glob("*")) if (dyn_root / "cases").exists() else []
        cases = [path.name for path in case_dirs if path.is_dir()]
        if not cases:
            continue
        msd_dates = discover_project_batch_dates("msd")
        rdf_dates = discover_project_batch_dates("rdf")
        msd_batch = f"{PROJECT_ID}_msd_{msd_dates[-1]}" if msd_dates else f"{PROJECT_ID}_msd_20260501"
        rdf_batch = f"{PROJECT_ID}_rdf_{rdf_dates[-1]}" if rdf_dates else f"{PROJECT_ID}_rdf_20260501"
        command = [
            sys.executable,
            str(SCRIPT_DIR / "analyze_dynamics.py"),
            "--output-root",
            str(SCRIPT_DIR / "output"),
            "--batch-date",
            date_tag,
            "--cases",
            *cases,
            "--skip-coordinates",
            "--msd-batch",
            msd_batch,
            "--rdf-batch",
            rdf_batch,
        ]
        proc = run(command, timeout=600, cwd=ROOT, env=project_subprocess_env())
        results.append({"observable": "dynamics", "date_tag": date_tag, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()})

    return {"ok": all(int(row.get("returncode", 1)) == 0 for row in results), "results": results}


def refresh_formal_outputs() -> dict[str, Any]:
    aggregates = rebuild_postprocess_aggregates()
    overview = rebuild_overview()
    publication = refresh_publication_assets()
    report = run_report()
    agg = (load_lcb_config(CONFIG_PATH).integrated_output_dir if CONFIG_PATH else SCRIPT_DIR / "output" / f"{PROJECT_ID}_integrated" / "aggregate")
    expected = {
        "summary_wide": agg / f"{PROJECT_ID}_summary_wide.csv",
        "ml_features": agg / "ml_features.csv",
        "ml_targets": agg / "ml_targets.csv",
        "ml_quality_mask": agg / "ml_quality_mask.csv",
        "ml_dataset_manifest": agg / "ml_dataset_manifest.json",
    }
    return {
        "aggregates": aggregates,
        "overview": overview,
        "publication_assets": publication,
        "report": report,
        "expected_outputs": {key: str(path) for key, path in expected.items()},
        "expected_outputs_present": {key: path.exists() for key, path in expected.items()},
        "ok": aggregates.get("ok") and overview.get("returncode") == 0 and publication.get("ok") and report.get("ok") and all(path.exists() for path in expected.values()),
    }


def refresh_health_monitor(remote_logs: bool = False) -> dict[str, Any]:
    command = [sys.executable, str(HEALTH_MONITOR)]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    if remote_logs:
        command.extend(["--remote-logs", "--max-remote-probes", "12"])
    proc = run(command, timeout=300, cwd=ROOT)
    output_root = OUTPUT_DIR.parent / f"{PROJECT_ID}_health_monitor"
    health_csv = output_root / "health_status.csv"
    status_counts: dict[str, int] = {}
    if health_csv.exists():
        health = read_csv(health_csv)
        if not health.empty and "status" in health.columns:
            status_counts = {str(key): int(value) for key, value in health["status"].value_counts().to_dict().items()}
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "ok": proc.returncode == 0,
        "remote_logs": remote_logs,
        "health_status_csv": str(health_csv) if health_csv.exists() else "",
        "status_counts": status_counts,
    }


def disk_targets() -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for host in [NEW_SERVER, NEWNEW_SERVER, OLD_SERVER]:
        user = host.split("@", 1)[0] if "@" in host else ""
        if not user:
            continue
        targets.append((host, f"${PELCB100_REMOTE_PATH}"))
    return targets


def parse_df_bg_line(line: str) -> dict[str, Any]:
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"unexpected df output: {line!r}")
    total_token = parts[1]
    free_token = parts[3]
    pct_token = parts[4]
    if not total_token.endswith("G") or not free_token.endswith("G"):
        raise ValueError(f"df output is not in -BG units: {line!r}")
    return {
        "total_gb": int(total_token[:-1]),
        "free_gb": int(free_token[:-1]),
        "used_pct": int(pct_token.rstrip("%")),
    }


def write_disk_status_md(path: Path, disk_status: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Supervisor Disk Status",
        "",
        f"Sampled at: `{iso_now()}`",
        "",
        "| Host | Home | Status | Free GB | Total GB | Used % | Sampled at |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for host, row in disk_status.items():
        lines.append(
            "| {host} | `{path}` | {status} | {free} | {total} | {pct} | {sampled} |".format(
                host=host,
                path=row.get("path", ""),
                status=row.get("status", ""),
                free=row.get("free_gb", ""),
                total=row.get("total_gb", ""),
                pct=row.get("used_pct", ""),
                sampled=row.get("sampled_at", ""),
            )
        )
    write_text(path, "\n".join(lines) + "\n")


def disk_sentinel(output: Path) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    sampled = iso_now()
    for host, home_path in disk_targets():
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            f"df -BG {sh_quote(home_path)} 2>&1 | tail -1",
        ]
        proc = run(command, timeout=20, cwd=ROOT)
        raw = (proc.stdout or proc.stderr or "").strip().splitlines()
        line = raw[-1] if raw else ""
        row: dict[str, Any] = {
            "path": home_path,
            "sampled_at": sampled,
            "returncode": proc.returncode,
            "raw": line,
        }
        if proc.returncode != 0:
            row.update({"status": "unreachable", "error": (proc.stderr or proc.stdout or "").strip()})
        else:
            try:
                row.update(parse_df_bg_line(line))
                free_gb = int(row["free_gb"])
                if free_gb < DISK_CRITICAL_FREE_GB:
                    row["status"] = "critical"
                elif free_gb < DISK_WARNING_FREE_GB:
                    row["status"] = "warning"
                else:
                    row["status"] = "ok"
            except Exception as exc:
                row.update({"status": "unreachable", "error": str(exc)})
        status[host] = row
    write_disk_status_md(output / "supervisor_disk_status.md", status)
    return status


def disk_critical_hosts(disk_status: dict[str, dict[str, Any]]) -> set[str]:
    return {host for host, row in disk_status.items() if row.get("status") == "critical"}


def strict_done_count(cases: pd.DataFrame) -> int:
    if cases.empty:
        return 0
    if "runtime_status" in cases.columns:
        return int(cases["runtime_status"].astype(str).str.lower().eq("strict_done").sum())
    if "strict_done" in cases.columns:
        return int(cases["strict_done"].apply(is_truthy).sum())
    return 0


def ready_for_analysis_rows(cases: pd.DataFrame) -> list[dict[str, Any]]:
    if cases.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in cases.iterrows():
        runtime_status = str(row.get("runtime_status", "")).lower()
        strict_done = is_truthy(row.get("strict_done", False))
        ready = is_truthy(row.get("ready_for_analysis", False))
        if ready or strict_done or runtime_status == "strict_done":
            rows.append({str(key): row.get(key, "") for key in cases.columns})

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        return (case_seq(str(item.get("case_id", ""))), str(item.get("case_id", "")))

    return sorted(rows, key=sort_key)


def parse_manual_date(value: str) -> datetime | None:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return None
    iso_candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        return parsed.replace(tzinfo=None)
    except ValueError:
        pass
    parts = text.split()
    if len(parts) >= 6 and parts[-1].isdigit() and parts[-2].isalpha():
        text_without_tz = " ".join(parts[:-2] + [parts[-1]])
    else:
        text_without_tz = text
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b %d %I:%M:%S %p %Y"):
        try:
            return datetime.strptime(text_without_tz, fmt)
        except ValueError:
            continue
    return None


def parse_manual_stage_timeline(text: str) -> dict[str, Any]:
    stage_start: dict[str, datetime] = {}
    stage_end: dict[str, datetime] = {}
    skipped: set[str] = set()
    started_at: datetime | None = None
    finished_at: datetime | None = None
    for line in str(text or "").splitlines():
        if line.startswith("Started:"):
            started_at = parse_manual_date(line.split("Started:", 1)[1])
            continue
        if line.startswith("Finished:"):
            finished_at = parse_manual_date(line.split("Finished:", 1)[1])
            continue
        skip_match = re.search(r"\[resume\]\s+Stage\s+(steep|rlx|pr|nvt_prod)\s+already completed; skipping at\s+(.+)$", line)
        if skip_match:
            skipped.add(skip_match.group(1))
            continue
        start_match = re.search(
            r"\[resume\]\s+(?:Starting fresh|Resuming)\s+mdrun for\s+(steep|rlx|pr|nvt_prod)\b.*\sat\s+(.+)$",
            line,
        )
        if start_match:
            parsed = parse_manual_date(start_match.group(2))
            if parsed is not None:
                stage_start[start_match.group(1)] = parsed
            continue
        done_match = re.search(r"Stage\s+(steep|rlx|pr|nvt_prod)\s+completed at\s+(.+)$", line)
        if done_match:
            parsed = parse_manual_date(done_match.group(2))
            if parsed is not None:
                stage_end[done_match.group(1)] = parsed

    durations: dict[str, int] = {}
    for stage in MD_STAGE_ORDER:
        if stage in skipped:
            continue
        if stage not in stage_start or stage not in stage_end:
            continue
        seconds = int((stage_end[stage] - stage_start[stage]).total_seconds())
        if seconds > 0:
            durations[stage] = seconds
    total_s = None
    if started_at is not None and finished_at is not None:
        seconds = int((finished_at - started_at).total_seconds())
        if seconds > 0:
            total_s = seconds
    status = "ok" if len(durations) == len(MD_STAGE_ORDER) else "partial" if durations else "no_stage_times"
    return {
        "parse_status": status,
        "started_at": started_at.isoformat(timespec="seconds") if started_at else "",
        "finished_at": finished_at.isoformat(timespec="seconds") if finished_at else "",
        "stage_durations_s": durations,
        "total_duration_s": total_s,
        "skipped_stages": sorted(skipped),
    }


def completed_runtime_rows(cases: pd.DataFrame) -> list[dict[str, Any]]:
    if cases.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in cases.iterrows():
        runtime_status = str(row.get("runtime_status", "")).lower()
        strict_done = is_truthy(row.get("strict_done", False))
        if runtime_status == "strict_done" or strict_done:
            rows.append({str(key): row.get(key, "") for key in cases.columns})
    return sorted(rows, key=lambda item: (case_seq(str(item.get("case_id", ""))), str(item.get("case_id", ""))))


def eta_fetch_manual_out(row: dict[str, Any]) -> tuple[str, str, str, str]:
    host_label, case_dir = latest_remote_case_dir(row)
    host = host_for_label(host_label)
    if not host or not case_dir:
        return host_label, case_dir, "", "missing_host_or_case_dir"
    remote_path = posixpath.join(case_dir, "manual.out")
    proc = run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, f"cat {sh_quote(remote_path)}"],
        timeout=45,
    )
    if proc.returncode != 0:
        return host_label, case_dir, "", (proc.stderr.strip() or proc.stdout.strip() or f"ssh_returncode={proc.returncode}")[:500]
    return host_label, case_dir, proc.stdout, ""


def eta_stage_stats(values: list[int]) -> dict[str, Any]:
    clean = sorted(int(value) for value in values if int(value) > 0)
    if not clean:
        return {"count": 0, "median_s": "", "p90_s": ""}
    series = pd.Series(clean, dtype=float)
    return {
        "count": int(len(clean)),
        "median_s": int(series.median()),
        "p90_s": int(series.quantile(0.9, interpolation="nearest")),
    }


def eta_record_has_stage_duration(record: dict[str, Any]) -> bool:
    return any(str(record.get(f"{stage}_duration_s", "")).strip() for stage in MD_STAGE_ORDER)


def eta_record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("case_id", "") or ""), str(record.get("remote_case_dir", "") or ""))


def eta_probe_attempts(record: dict[str, Any]) -> int:
    try:
        return int(float(record.get("eta_probe_attempts", 0) or 0))
    except (TypeError, ValueError):
        return 0


def calibrate_runtime_eta(
    output: Path,
    runtime: dict[str, pd.DataFrame],
    *,
    max_new_cases: int = 20,
    retry_unstable: bool = True,
) -> dict[str, Any]:
    cases = runtime.get("cases", pd.DataFrame())
    done_rows = completed_runtime_rows(cases)
    calibration_csv = output / "runtime_eta_calibration_cases.csv"
    calibration_json = output / "runtime_eta_calibration.json"
    calibration_md = output / "runtime_eta_calibration.md"
    existing = read_csv(calibration_csv)
    records: list[dict[str, Any]] = []
    if not existing.empty:
        for raw in existing.to_dict("records"):
            records.append({key: "" if pd.isna(value) else value for key, value in raw.items()})
    by_key: dict[tuple[str, str], dict[str, Any]] = {
        eta_record_key(row): row for row in records if eta_record_key(row)[0] and eta_record_key(row)[1]
    }
    probed = 0
    skipped_stable = 0
    retryable_unstable = 0
    for row in done_rows:
        case_id = str(row.get("case_id", ""))
        host_label, case_dir = latest_remote_case_dir(row)
        key = (case_id, case_dir)
        if not case_id or not case_dir:
            continue
        existing_record = by_key.get(key)
        if existing_record and eta_record_has_stage_duration(existing_record):
            skipped_stable += 1
            continue
        if existing_record and not retry_unstable:
            retryable_unstable += 1
            continue
        if probed >= max_new_cases:
            break
        probed += 1
        fetched_host, fetched_dir, manual_text, fetch_error = eta_fetch_manual_out(row)
        attempts = eta_probe_attempts(existing_record or {}) + 1
        last_probe_at = iso_now()
        if manual_text:
            parsed = parse_manual_stage_timeline(manual_text)
            manual_hash = hashlib.sha256(manual_text.encode("utf-8", errors="ignore")).hexdigest()
            durations = parsed.get("stage_durations_s", {})
            record = {
                "sampled_at": iso_now(),
                "case_id": case_id,
                "host_label": fetched_host or host_label,
                "remote_case_dir": fetched_dir or case_dir,
                "manual_out_sha256": manual_hash,
                "parse_status": parsed["parse_status"],
                "eta_probe_attempts": attempts,
                "last_probe_at": last_probe_at,
                "last_probe_error": "" if durations else f"no_parseable_stage_durations:{parsed['parse_status']}",
                "started_at": parsed["started_at"],
                "finished_at": parsed["finished_at"],
                "total_duration_s": parsed.get("total_duration_s") or "",
                "skipped_stages": ";".join(parsed.get("skipped_stages", [])),
            }
            for stage in MD_STAGE_ORDER:
                record[f"{stage}_duration_s"] = durations.get(stage, "")
        else:
            record = {
                "sampled_at": iso_now(),
                "case_id": case_id,
                "host_label": fetched_host or host_label,
                "remote_case_dir": fetched_dir or case_dir,
                "manual_out_sha256": "",
                "parse_status": "manual_out_unavailable",
                "eta_probe_attempts": attempts,
                "last_probe_at": last_probe_at,
                "last_probe_error": fetch_error,
                "started_at": "",
                "finished_at": "",
                "total_duration_s": "",
                "skipped_stages": "",
                **{f"{stage}_duration_s": "" for stage in MD_STAGE_ORDER},
            }
        by_key[key] = record
    records = sorted(
        by_key.values(),
        key=lambda item: (case_seq(str(item.get("case_id", ""))), str(item.get("case_id", "")), str(item.get("remote_case_dir", ""))),
    )

    fields = [
        "sampled_at",
        "case_id",
        "host_label",
        "remote_case_dir",
        "manual_out_sha256",
        "parse_status",
        "eta_probe_attempts",
        "last_probe_at",
        "last_probe_error",
        "started_at",
        "finished_at",
        "total_duration_s",
        "skipped_stages",
        *[f"{stage}_duration_s" for stage in MD_STAGE_ORDER],
    ]
    calibration_csv.parent.mkdir(parents=True, exist_ok=True)
    with calibration_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})

    stage_values: dict[str, list[int]] = {stage: [] for stage in MD_STAGE_ORDER}
    total_values: list[int] = []
    for record in records:
        for stage in MD_STAGE_ORDER:
            value = record.get(f"{stage}_duration_s", "")
            try:
                if str(value).strip():
                    stage_values[stage].append(int(float(value)))
            except (TypeError, ValueError):
                pass
        try:
            total_value = record.get("total_duration_s", "")
            if str(total_value).strip():
                total_values.append(int(float(total_value)))
        except (TypeError, ValueError):
            pass
    stage_stats = {stage: eta_stage_stats(values) for stage, values in stage_values.items()}
    usable_case_ids = sorted(
        {
            str(record.get("case_id", ""))
            for record in records
            if any(str(record.get(f"{stage}_duration_s", "")).strip() for stage in MD_STAGE_ORDER)
        }
    )
    baseline_status = (
        "no_completed_stage_baseline"
        if not done_rows
        else "calibrated"
        if any(row["count"] > 0 for row in stage_stats.values())
        else "completed_cases_unparsed"
    )
    summary = {
        "generated_at": iso_now(),
        "baseline_status": baseline_status,
        "strict_done_seen": len(done_rows),
        "new_cases_probed": probed,
        "retry_unstable": retry_unstable,
        "retryable_unstable_skipped": retryable_unstable,
        "stable_records_skipped": skipped_stable,
        "calibration_case_count": len(usable_case_ids),
        "usable_case_ids": usable_case_ids,
        "stage_stats": stage_stats,
        "total_stats": eta_stage_stats(total_values),
        "cases_csv": str(calibration_csv),
    }
    write_json(calibration_json, summary)
    lines = [
        f"# {PROJECT_ID.upper()} Runtime ETA Calibration",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Baseline status: `{baseline_status}`",
        f"- Strict done seen: `{len(done_rows)}`",
        f"- Calibrating cases: `{len(usable_case_ids)}`",
        "",
        "| Stage | n | p50 h | p90 h |",
        "|---|---:|---:|---:|",
    ]
    for stage in MD_STAGE_ORDER:
        stats = stage_stats[stage]
        median = stats.get("median_s")
        p90 = stats.get("p90_s")
        median_h = round(float(median) / 3600.0, 2) if str(median).strip() else ""
        p90_h = round(float(p90) / 3600.0, 2) if str(p90).strip() else ""
        lines.append(f"| `{stage}` | {stats.get('count', 0)} | {median_h} | {p90_h} |")
    write_text(calibration_md, "\n".join(lines) + "\n")
    return summary


def eta_stage_remaining(stage: str, elapsed_s: int, calibration: dict[str, Any], quantile_key: str) -> tuple[int | None, str]:
    stage_stats = calibration.get("stage_stats", {}) if isinstance(calibration, dict) else {}
    if stage not in MD_STAGE_ORDER:
        return None, "unknown_stage"
    missing: list[str] = []
    remaining = 0
    current_index = MD_STAGE_ORDER.index(stage)
    for index, item in enumerate(MD_STAGE_ORDER[current_index:]):
        value = stage_stats.get(item, {}).get(quantile_key, "")
        if not str(value).strip():
            missing.append(item)
            continue
        seconds = int(float(value))
        if index == 0:
            remaining += max(0, seconds - max(0, elapsed_s))
        else:
            remaining += seconds
    if missing:
        return None, "missing_stage_baseline:" + ";".join(missing)
    return remaining, "calibrated"


def write_runtime_eta_summary(
    output: Path,
    runtime: dict[str, pd.DataFrame],
    *,
    retry_unstable_calibration: bool = True,
) -> dict[str, Any]:
    active = runtime.get("active", pd.DataFrame())
    cases = runtime.get("cases", pd.DataFrame())
    now = iso_now()
    calibration = calibrate_runtime_eta(output, runtime, retry_unstable=retry_unstable_calibration)
    path_csv = output / "runtime_eta_summary.csv"
    path_md = output / "runtime_eta_summary.md"
    rows: list[dict[str, Any]] = []
    if not active.empty:
        for _, row in active.iterrows():
            try:
                etimes_s = int(float(row.get("etimes", 0) or 0))
            except (TypeError, ValueError):
                etimes_s = 0
            stage = str(row.get("stage", ""))
            p50_remaining, p50_status = eta_stage_remaining(stage, etimes_s, calibration, "median_s")
            p90_remaining, p90_status = eta_stage_remaining(stage, etimes_s, calibration, "p90_s")
            p50_finish = ""
            p90_finish = ""
            if p50_remaining is not None:
                p50_finish = (datetime.now() + timedelta(seconds=p50_remaining)).isoformat(timespec="seconds")
            if p90_remaining is not None:
                p90_finish = (datetime.now() + timedelta(seconds=p90_remaining)).isoformat(timespec="seconds")
            rows.append(
                {
                    "checked_at": now,
                    "case_id": row.get("case_id", ""),
                    "host_label": row.get("host_label", ""),
                    "host": row.get("host", ""),
                    "node": row.get("node", ""),
                    "gpu_id": row.get("gpu_id", ""),
                    "pid": row.get("pid", ""),
                    "stage": stage,
                    "etimes_s": etimes_s,
                    "etimes_h": round(etimes_s / 3600.0, 3),
                    "eta_status": p50_status if p50_status == p90_status else f"p50={p50_status};p90={p90_status}",
                    "eta_p50_remaining_s": p50_remaining if p50_remaining is not None else "",
                    "eta_p50_remaining_h": round(p50_remaining / 3600.0, 2) if p50_remaining is not None else "",
                    "eta_p50_finish_local": p50_finish,
                    "eta_p90_remaining_s": p90_remaining if p90_remaining is not None else "",
                    "eta_p90_remaining_h": round(p90_remaining / 3600.0, 2) if p90_remaining is not None else "",
                    "eta_p90_finish_local": p90_finish,
                    "calibration_case_count": calibration.get("calibration_case_count", 0),
                    "pcpu": row.get("pcpu", ""),
                    "cwd": row.get("cwd", ""),
                }
            )
    fields = [
        "checked_at",
        "case_id",
        "host_label",
        "host",
        "node",
        "gpu_id",
        "pid",
        "stage",
        "etimes_s",
        "etimes_h",
        "eta_status",
        "eta_p50_remaining_s",
        "eta_p50_remaining_h",
        "eta_p50_finish_local",
        "eta_p90_remaining_s",
        "eta_p90_remaining_h",
        "eta_p90_finish_local",
        "calibration_case_count",
        "pcpu",
        "cwd",
    ]
    if rows:
        path_csv.parent.mkdir(parents=True, exist_ok=True)
        with path_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        write_text(path_csv, ",".join(fields) + "\n")

    stage_counts = pd.Series([row["stage"] for row in rows], dtype=str).value_counts().to_dict() if rows else {}
    longest = max(rows, key=lambda item: int(item["etimes_s"])) if rows else {}
    done_count = strict_done_count(cases)
    baseline_status = str(calibration.get("baseline_status", "no_completed_stage_baseline"))
    eta_policy = {
        "no_completed_stage_baseline": "no completed stage baseline yet; reporting stage ages only",
        "completed_cases_unparsed": "strict_done exists, but completed manual.out timelines were not parseable yet",
        "calibrated": "completed manual.out timelines available; reporting calibrated p50/p90 remaining time",
    }.get(baseline_status, baseline_status)
    lines = [
        f"# {PROJECT_ID.upper()} Runtime ETA Summary",
        "",
        f"- Generated: `{now}`",
        f"- Active processes: `{len(rows)}`",
        f"- Strict done cases: `{done_count}`",
        f"- ETA policy: `{eta_policy}`",
        f"- Calibration cases: `{calibration.get('calibration_case_count', 0)}`",
        "",
        "## Stage Counts",
        "",
    ]
    if stage_counts:
        for stage, count in sorted(stage_counts.items()):
            lines.append(f"- `{stage}`: {count}")
    else:
        lines.append("- No active processes in the current dashboard snapshot.")
    lines.extend(["", "## Longest Active Stage", ""])
    if longest:
        lines.append(
            f"- `{longest.get('case_id')}` on `{longest.get('node')}` GPU `{longest.get('gpu_id')}` "
            f"stage `{longest.get('stage')}` age `{longest.get('etimes_h')}` h"
        )
    else:
        lines.append("- n/a")
    if rows and baseline_status == "calibrated":
        ready_eta = [row for row in rows if str(row.get("eta_p50_remaining_s", "")).strip()]
        if ready_eta:
            next_p50 = min(ready_eta, key=lambda item: int(item["eta_p50_remaining_s"]))
            next_p90 = min(ready_eta, key=lambda item: int(item["eta_p90_remaining_s"]))
            lines.extend(
                [
                    "",
                    "## Calibrated Completion Estimates",
                    "",
                    f"- Earliest p50 finish: `{next_p50.get('case_id')}` at `{next_p50.get('eta_p50_finish_local')}` "
                    f"({next_p50.get('eta_p50_remaining_h')} h remaining)",
                    f"- Earliest p90 finish: `{next_p90.get('case_id')}` at `{next_p90.get('eta_p90_finish_local')}` "
                    f"({next_p90.get('eta_p90_remaining_h')} h remaining)",
                ]
            )
    write_text(path_md, "\n".join(lines) + "\n")
    return {
        "eta_csv": str(path_csv),
        "eta_md": str(path_md),
        "eta_calibration_json": str(output / "runtime_eta_calibration.json"),
        "eta_baseline_status": baseline_status,
        "eta_calibration_case_count": calibration.get("calibration_case_count", 0),
        "active_count": len(rows),
        "strict_done_count": done_count,
        "stage_counts": {str(key): int(value) for key, value in stage_counts.items()},
        "longest_case_id": str(longest.get("case_id", "")) if longest else "",
        "longest_stage_h": longest.get("etimes_h", "") if longest else "",
    }


def paper_freeze_ready_status(runtime: dict[str, pd.DataFrame], state: dict[str, Any], ready_count: int, duplicate_count: int) -> dict[str, Any]:
    aggregate = SCRIPT_DIR / "output" / f"{PROJECT_ID}_integrated" / "aggregate"
    ml_outputs = {
        "ml_features.csv": aggregate / "ml_features.csv",
        "ml_targets.csv": aggregate / "ml_targets.csv",
        "ml_quality_mask.csv": aggregate / "ml_quality_mask.csv",
        "ml_dataset_manifest.json": aggregate / "ml_dataset_manifest.json",
    }
    ml_present = {name: path.exists() for name, path in ml_outputs.items()}
    summary = read_csv(aggregate / f"{PROJECT_ID}_summary_wide.csv")
    any_observable_ok = False
    if not summary.empty:
        if "status_ok_count" in summary.columns:
            ok_count = pd.to_numeric(summary["status_ok_count"], errors="coerce").fillna(0)
            any_observable_ok = bool((ok_count > 0).any())
        else:
            status_cols = [col for col in summary.columns if col.startswith("status_")]
            any_observable_ok = any(summary[col].astype(str).str.upper().eq("OK").any() for col in status_cols)
    postprocess_idle = not bool(state.get("postprocess_jobs"))
    strict_done = strict_done_count(runtime.get("cases", pd.DataFrame()))
    blockers: list[str] = []
    if duplicate_count:
        blockers.append("runtime_duplicates_nonempty")
    if strict_done <= 0:
        blockers.append("no_strict_done_cases")
    if ready_count:
        blockers.append("ready_for_analysis_pending")
    if not postprocess_idle:
        blockers.append("postprocess_jobs_active")
    if not all(ml_present.values()):
        blockers.append("missing_ml_outputs")
    if not any_observable_ok:
        blockers.append("no_integrated_observable_ok")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "strict_done_count": strict_done,
        "ready_for_analysis_count": ready_count,
        "duplicate_count": duplicate_count,
        "postprocess_idle": postprocess_idle,
        "ml_outputs_present": ml_present,
        "any_observable_ok": any_observable_ok,
        "policy": "publication_release freeze remains manual; run pelcb100_paper_release_freeze.py --release-level publication_release --refresh-assets when ready",
    }


def write_postprocess_failure_triage(output: Path, summary: dict[str, Any], jobs: list[str]) -> dict[str, str]:
    triage_dir = output / "postprocess_failure_triage"
    triage_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_stamp()
    rows = summary.get("failed_rows", []) if isinstance(summary, dict) else []
    csv_path = triage_dir / f"postprocess_failure_triage_{timestamp}.csv"
    md_path = triage_dir / f"postprocess_failure_triage_{timestamp}.md"
    fields = ["job_id", "job_name", "state", "elapsed", "exit_code", "source"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    lines = [
        "# Postprocess Failure Triage",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Jobs: `{';'.join(jobs)}`",
        f"- Failed rows: `{len(rows)}`",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('job_id', '')}` `{row.get('job_name', '')}` "
            f"`{row.get('state', '')}` `{row.get('exit_code', '')}`"
        )
    write_text(md_path, "\n".join(lines) + "\n")
    return {"csv": str(csv_path), "md": str(md_path)}


def failed_postprocess_job_ids(summary: dict[str, Any]) -> list[str]:
    rows = summary.get("failed_rows", []) if isinstance(summary, dict) else []
    ids = []
    for row in rows:
        job_id = str(row.get("job_id", "")).strip()
        if job_id:
            ids.append(job_id)
    return sorted(set(ids))


def mark_failed_postprocess_jobs_triaged(state: dict[str, Any], job_ids: list[str], triage: dict[str, str]) -> None:
    if not job_ids:
        return
    existing = {str(item) for item in state.get("ignored_failed_postprocess_jobs", []) if str(item).strip()}
    existing.update(str(item) for item in job_ids if str(item).strip())
    state["ignored_failed_postprocess_jobs"] = sorted(existing)
    state.setdefault("ignored_failed_postprocess_jobs_log", [])
    state["ignored_failed_postprocess_jobs_log"].append(
        {
            "ignored_at": iso_now(),
            "job_ids": sorted(set(job_ids)),
            "reason": "failed postprocess jobs triaged and preserved; allow next cycle to resubmit after code/template fix",
            "triage": triage,
        }
    )


def flatten_json_keys(payload: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            keys.append(new_prefix)
            keys.extend(flatten_json_keys(value, new_prefix))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload[:20]):
            keys.extend(flatten_json_keys(value, f"{prefix}[{idx}]"))
    return keys


def has_any_key(keys: list[str], terms: list[str]) -> bool:
    lowered = [key.lower() for key in keys]
    return any(any(term in key for term in terms) for key in lowered)


def nonempty_json_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def find_json_error_markers(payload: Any, prefix: str = "") -> list[str]:
    markers: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if any(term in lowered for term in ["fatal", "traceback", "exception", "parser_error", "error"]):
                if nonempty_json_value(value):
                    markers.append(dotted)
            markers.extend(find_json_error_markers(value, dotted))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            markers.extend(find_json_error_markers(item, f"{prefix}[{index}]"))
    return markers


def find_case_density_analysis(case_id: str, output: Path) -> Path | None:
    candidates = [
        output.parent / f"{PROJECT_ID}_density_energy" / case_id / "analysis.json",
        output.parent / f"{PROJECT_ID}_orchestrator" / f"{PROJECT_ID}_density_energy" / case_id / "analysis.json",
        SCRIPT_DIR / "output" / f"{PROJECT_ID}_density_energy" / case_id / "analysis.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    for root in [ORCHESTRATOR_DIR, CLOSURE_DIR, output.parent]:
        if not root.exists():
            continue
        try:
            matches = [
                path
                for path in root.rglob("analysis.json")
                if case_id in str(path) and ("density" in str(path).lower() or "energy" in str(path).lower())
            ]
        except OSError:
            matches = []
        if matches:
            return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    return None


def first_postprocess_smoke_gate(first_case: dict[str, Any], output: Path, state: dict[str, Any]) -> dict[str, Any]:
    case_id = str(first_case.get("case_id", "")).strip()
    gate_dir = output / "first_postprocess_gate" / (case_id or "unknown_case")
    gate_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "case_id": case_id,
        "checked_at": iso_now(),
        "status": "validate",
        "pass": False,
        "missing_fields": [],
        "analysis_json": "",
        "note": "baseline_density smoke gate validates existing single-case density output; bulk postprocess remains blocked until this gate passes",
    }
    if not case_id:
        result["missing_fields"].append("case_id")
        write_json(gate_dir / "gate_result.json", result)
        return result

    analysis_path = find_case_density_analysis(case_id, output)
    if not analysis_path:
        old_trigger = state.get("first_postprocess_gate_trigger")
        if isinstance(old_trigger, dict) and old_trigger:
            old_jobs_in_postprocess = {str(job) for job in state.get("postprocess_jobs", []) or [] if str(job).strip()}
            old_job_ids = {str(job) for job in old_trigger.get("job_ids", []) or [] if str(job).strip()}
            # The previously submitted baseline_density gate job has been
            # closed/removed from postprocess_jobs, but no analysis.json was
            # produced.  Count this as a real gate failure instead of launching
            # a new trigger every cycle.
            if old_job_ids and not (old_job_ids & old_jobs_in_postprocess):
                result["trigger_failed_previous"] = old_trigger
                state["first_postprocess_gate_failed_attempts"] = int(
                    state.get("first_postprocess_gate_failed_attempts", 0) or 0
                ) + 1
                state["first_postprocess_gate_trigger"] = None
                result["missing_fields"].extend(["analysis_json", "density_timeseries"])
                result["failed_attempts"] = state["first_postprocess_gate_failed_attempts"]
                result["status"] = (
                    "halt"
                    if state["first_postprocess_gate_failed_attempts"] >= 3
                    else "trigger_failed_previous"
                )
                write_json(gate_dir / "gate_result.json", result)
                return result
        result["status"] = "trigger_baseline_density"
        result["missing_fields"].extend(["analysis_json", "density_timeseries"])
        write_json(gate_dir / "gate_result.json", result)
        return result

    result["analysis_json"] = str(analysis_path)
    analysis = read_json(analysis_path, {})
    keys = flatten_json_keys(analysis)
    required_inputs = {
        "thermal": ["temp", "temperature"],
        "energetic": ["energy", "potential", "total"],
        "compaction": ["density", "volume", "box"],
    }
    optional_inputs = {
        "density_timeseries": ["density_time", "density_series", "density.timeseries", "density_ts"],
        "void": ["void", "voxel", "low_density"],
        "fit": ["msd", "slope", "diffusion", "fit"],
        "runtime": ["runtime", "fatal", "warning", "checkpoint", "stage"],
    }
    required_presence = {name: has_any_key(keys, terms) for name, terms in required_inputs.items()}
    optional_presence = {name: has_any_key(keys, terms) for name, terms in optional_inputs.items()}
    missing = [name for name, present in required_presence.items() if not present]
    analysis_case_id = str(analysis.get("case_id", "") or "").strip() if isinstance(analysis, dict) else ""
    if analysis_case_id and analysis_case_id != case_id:
        missing.append("case_id_mismatch")
    if not isinstance(analysis, dict) or not analysis:
        missing.append("analysis_json_empty_or_invalid")
    error_markers = find_json_error_markers(analysis)
    if error_markers:
        missing.append("parser_fatal_marker")
    result["missing_fields"] = missing
    result["required_presence"] = required_presence
    result["optional_presence"] = optional_presence
    result["analysis_case_id"] = analysis_case_id
    result["parser_error_markers"] = error_markers
    result["rule_input_presence"] = {**required_presence, **optional_presence}
    result["pass"] = not missing
    result["status"] = "pass" if result["pass"] else "schema_fail"
    write_json(gate_dir / "gate_result.json", result)
    return result


def run_pelcb100_control_queue(*, execute: bool) -> dict[str, Any]:
    """Run the optional PE-LCB100 validation-control queue once."""

    if PROJECT_ID != "pelcb100" or not CONTROL_QUEUE.exists():
        return {"ok": True, "skipped": True, "reason": "not pelcb100 or control queue missing"}
    command = [
        sys.executable,
        str(CONTROL_QUEUE),
        "--config",
        str(CONFIG_PATH or (SCRIPT_DIR / "configs" / "pelcb100.json")),
        "--max-submit",
        "2",
        "--max-concurrent",
        "6",
        "--allow-second-per-gpu",
        "--only-when-main-idle",
    ]
    if execute:
        command.append("--execute")
    proc = run(command, timeout=300, cwd=SCRIPT_DIR)
    decision_path = OUTPUT_DIR / "new_from_newnew_same_case_controls_20260518" / "control_queue_last_decision.json"
    latest = read_json(decision_path, {})
    latest["returncode"] = proc.returncode
    latest["stdout"] = proc.stdout.strip()
    latest["stderr"] = proc.stderr.strip()
    latest["ok"] = proc.returncode == 0
    latest["skipped"] = bool(latest.get("skipped", False))
    return latest


def write_decisions(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {PROJECT_ID.upper()} Supervisor Decisions",
        "",
        f"- Time: `{payload['timestamp']}`",
        f"- Mode: `{'execute' if payload['execute'] else 'dry-run'}`",
        f"- Dashboard ok: `{payload['dashboard'].get('ok')}`",
        f"- Upstream action: `{payload.get('upstream', {}).get('decision', {}).get('action', 'unknown')}`",
        f"- Duplicate rows: `{payload['duplicate_count']}`",
        f"- Free slots: `{len(payload['slots'])}`",
        f"- Planned MD submissions: `{len(payload['md_plans'])}`",
        f"- Planned restarts: `{len(payload['restart_plans'])}`",
        f"- Ready for analysis: `{payload['ready_for_analysis_count']}`",
        "",
        "## MD Plans",
        "",
    ]
    for plan in payload["md_plans"][:30]:
        lines.append(
            f"- `{plan['case_id']}` -> `{plan['host_label']}:{plan['node']}:gpu{plan['gpu']}` "
            f"threads `{plan['threads']}` action `{plan['action']}`"
        )
    upstream = payload.get("upstream", {})
    if upstream:
        lines.extend(["", "## Upstream EMC/Bridge", ""])
        decision = upstream.get("decision", {})
        lines.append(f"- action: `{decision.get('action', 'unknown')}`")
        lines.append(f"- cases: `{len(decision.get('cases', []) or [])}`")
        if upstream.get("stderr"):
            lines.append(f"- stderr: `{str(upstream.get('stderr'))[:500]}`")
    if payload["restart_plans"]:
        lines.extend(["", "## Restart Plans", ""])
        for plan in payload["restart_plans"]:
            lines.append(f"- `{plan['case_id']}` pid `{plan['pid']}` `{plan['reason']}`")
    if payload.get("failed_recovery"):
        recovery = payload["failed_recovery"]
        lines.extend(["", "## Failed Recovery", ""])
        lines.append(
            f"- classified: recoverable `{recovery.get('recoverable_count', 0)}`, "
            f"manual_review `{recovery.get('manual_review_count', 0)}`, "
            f"blocked_duplicate `{recovery.get('blocked_duplicate_count', 0)}`"
        )
        lines.append(f"- execute recovery: `{bool(recovery.get('execute_recovery'))}`")
        lines.append(f"- table: `{recovery.get('classification_csv', '')}`")
        if recovery.get("recoverable_count", 0) and not recovery.get("plans"):
            lines.append("- status: `recoverable cases are blocked only by slot availability; recovery remains prioritized before new backlog when a slot opens`")
        for plan in recovery.get("plans", [])[:20]:
            lines.append(
                f"- `{plan.get('case_id')}` recovery `{plan.get('recovery_kind')}` "
                f"action `{plan.get('planned_action')}` target `{plan.get('host_label', plan.get('source_host_label', ''))}:"
                f"{plan.get('node', '')}:gpu{plan.get('gpu', '')}`"
            )
    if payload.get("validation_control_queue"):
        control = payload["validation_control_queue"]
        lines.extend(["", "## Validation Control Queue", ""])
        lines.append(f"- action: `{control.get('action', 'unknown')}`")
        lines.append(f"- ok: `{control.get('ok')}`")
        if control.get("main_idle_reason"):
            lines.append(f"- main idle: `{control.get('main_idle_reason')}`")
        if control.get("active_controls") is not None:
            lines.append(f"- active controls: `{control.get('active_controls')}`")
        for row in control.get("submitted", [])[:10]:
            lines.append(
                f"- `{row.get('source_case_id', row.get('case_id', ''))}` -> "
                f"`{row.get('control_case_id', row.get('case_id', ''))}` "
                f"`{row.get('node', '')}:gpu{row.get('gpu', '')}`"
            )
        if control.get("stderr"):
            lines.append(f"- stderr: `{str(control.get('stderr'))[:500]}`")
    if payload.get("paper_freeze_ready"):
        freeze = payload["paper_freeze_ready"]
        lines.extend(["", "## Paper Release Freeze", ""])
        lines.append(f"- ready: `{freeze.get('ready')}`")
        lines.append(f"- blockers: `{';'.join(freeze.get('blockers', []))}`")
        lines.append(f"- policy: `{freeze.get('policy', '')}`")
        present = freeze.get("ml_outputs_present", {})
        if isinstance(present, dict):
            for name, value in present.items():
                lines.append(f"- `{name}` present: `{value}`")
    if payload.get("disk_status"):
        lines.extend(["", "## Disk Sentinel", ""])
        for host, row in payload["disk_status"].items():
            status = row.get("status", "")
            free = row.get("free_gb", "")
            total = row.get("total_gb", "")
            pct = row.get("used_pct", "")
            if status == "critical":
                lines.append(f"- disk_critical: `{host}` free `{free}` GB / total `{total}` GB used `{pct}%`; new deploys blocked for this host")
            elif status == "warning":
                lines.append(f"- disk_warning: `{host}` free `{free}` GB / total `{total}` GB used `{pct}%`; deploys still allowed")
            elif status == "unreachable":
                lines.append(f"- disk_unreachable: `{host}` `{row.get('error', row.get('raw', ''))}`")
            else:
                lines.append(f"- `{host}` ok free `{free}` GB / total `{total}` GB used `{pct}%`")
    if payload.get("postprocess_decision"):
        lines.extend(["", "## Postprocess", "", str(payload["postprocess_decision"])])
        decision = payload["postprocess_decision"]
        if isinstance(decision, dict) and (decision.get("closure_all") or decision.get("formal_outputs")):
            closure = decision.get("closure_all", {}) if isinstance(decision.get("closure_all"), dict) else {}
            formal = decision.get("formal_outputs", {}) if isinstance(decision.get("formal_outputs"), dict) else {}
            lines.extend(
                [
                    "",
                    "## Postprocess Closure",
                    "",
                    f"- merge plan: `{closure.get('merge_plan', '')}`",
                    f"- applied merge: `{closure.get('applied_merge_manifest', '')}`",
                    f"- plan csv: `{closure.get('plan_csv', '')}`",
                    f"- formal outputs ok: `{formal.get('ok', '')}`",
                ]
            )
            present = formal.get("expected_outputs_present", {})
            if isinstance(present, dict):
                for key, value in present.items():
                    lines.append(f"- `{key}` present: `{value}`")
    if payload.get("health_monitor"):
        health = payload["health_monitor"]
        lines.extend(["", "## Health Monitor", ""])
        lines.append(f"- refresh: `{'ok' if health.get('ok') else 'fail'}`")
        if health.get("status_counts"):
            pieces = ", ".join(f"{key}={value}" for key, value in sorted(health.get("status_counts", {}).items()))
            lines.append(f"- status counts: `{pieces}`")
        if health.get("health_status_csv"):
            lines.append(f"- table: `{health.get('health_status_csv')}`")
        if health.get("stderr"):
            lines.append(f"- stderr: `{str(health.get('stderr'))[:500]}`")
    if payload.get("eta_summary"):
        eta = payload["eta_summary"]
        lines.extend(["", "## Runtime ETA / Stage Age", ""])
        lines.append(f"- active: `{eta.get('active_count')}`")
        lines.append(f"- strict done: `{eta.get('strict_done_count')}`")
        lines.append(f"- longest active case: `{eta.get('longest_case_id')}` age `{eta.get('longest_stage_h')}` h")
        lines.append(f"- report: `{eta.get('eta_md')}`")
    if payload.get("publication_assets"):
        assets = payload["publication_assets"]
        lines.extend(["", "## Publication Assets", ""])
        if assets.get("skipped"):
            lines.append(f"- refresh: `skipped`")
            lines.append(f"- reason: `{assets.get('reason', '')}`")
        else:
            lines.append(f"- refresh: `{'ok' if assets.get('ok') else 'fail'}`")
        if assets.get("stdout"):
            lines.append(f"- stdout: `{str(assets.get('stdout'))[:500]}`")
        if assets.get("stderr"):
            lines.append(f"- stderr: `{str(assets.get('stderr'))[:500]}`")
    write_text(path, "\n".join(lines) + "\n")


def event(action: str, result: str, **kwargs: Any) -> dict[str, Any]:
    return {"time": iso_now(), "action": action, "result": result, **kwargs}


def event_fields(rows: list[dict[str, Any]]) -> list[str]:
    base = ["time", "action", "result", "case_id", "host_label", "host", "node", "gpu", "message"]
    extra = sorted({key for row in rows for key in row if key not in base})
    return base + extra


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    locks = output / "locks"
    with SupervisorLock(locks / "supervisor.lock"):
        timestamp = now_stamp()
        write_heartbeat(output, "cycle_start", run_id=timestamp, execute=bool(args.execute))
        state_path = output / "supervisor_state.json"
        state = read_json(state_path, {})
        events: list[dict[str, Any]] = []

        write_heartbeat(output, "refresh_dashboard_start", run_id=timestamp)
        dashboard = refresh_dashboard()
        write_heartbeat(output, "refresh_dashboard_done", run_id=timestamp, dashboard_ok=dashboard.get("ok"))
        runtime = load_runtime()
        duplicates = runtime["duplicates"]
        duplicate_count = int(len(duplicates))
        write_heartbeat(output, "disk_sentinel_start", run_id=timestamp)
        disk_status = disk_sentinel(output)
        blocked_hosts = disk_critical_hosts(disk_status)
        write_heartbeat(
            output,
            "disk_sentinel_done",
            run_id=timestamp,
            critical_hosts=",".join(sorted(blocked_hosts)),
        )
        events.append(
            event(
                "disk_sentinel",
                "critical" if blocked_hosts else "ok",
                message=",".join(sorted(blocked_hosts)) if blocked_hosts else "",
            )
        )
        write_heartbeat(output, "upstream_start", run_id=timestamp, duplicate_count=duplicate_count)
        upstream = run_emc_orchestrator(
            execute=bool(args.execute and not duplicate_count),
            max_cases=args.max_upstream_per_round,
        )
        write_heartbeat(
            output,
            "upstream_done",
            run_id=timestamp,
            upstream_ok=upstream.get("ok"),
            upstream_action=upstream.get("decision", {}).get("action", ""),
        )
        write_heartbeat(output, "planning_start", run_id=timestamp)
        slot_quarantine = active_slot_quarantine(state)
        state["md_slot_quarantine"] = slot_quarantine
        slots = (
            compute_slots(
                runtime["gpus"],
                runtime["active"],
                blocked_hosts=blocked_hosts,
                blocked_slots=set(slot_quarantine),
            )
            if dashboard["ok"]
            else []
        )
        recovery_rows, recovery_plans = failed_case_recovery_rows(
            runtime["cases"],
            duplicates,
            slots,
            state,
            args.max_recovery_per_round,
        )
        write_recovery_outputs(output, recovery_rows)
        reserved_recovery_slots = {
            str(plan.get("slot_key", ""))
            for plan in recovery_plans
            if args.execute_recovery and str(plan.get("slot_key", "")).strip()
        }
        md_slots = [
            slot for slot in slots if not reserved_recovery_slots or slot_key_for_plan(slot) not in reserved_recovery_slots
        ]
        dry_run_excluded = set(state.get("dry_run_planned_md_cases", [])) if not args.execute else set()
        upstream_cases = set(upstream.get("decision", {}).get("cases", []) or [])
        upstream_excluded = upstream_cases_to_exclude(upstream)
        md_plans = [] if duplicate_count else plan_md_submissions(
            runtime["backlog"],
            md_slots,
            args.max_submit_per_round,
            state,
            extra_excluded=dry_run_excluded | upstream_excluded,
            allow_failed_retry=False,
        )
        restart_plans = [] if duplicate_count or args.no_restart else plan_restarts(runtime["active"], duplicates)
        ready_cases = ready_for_analysis_rows(runtime["cases"])
        ready_count = len(ready_cases)
        done_count = strict_done_count(runtime["cases"])
        previous_ready_count = int(state.get("last_ready_for_analysis_count", -1) or -1)
        previous_done_count = int(state.get("last_strict_done_count", -1) or -1)
        cycle_count = int(state.get("cycle_count", 0) or 0) + 1
        state["disk_status"] = disk_status
        state["first_postprocess_gate_passed"] = bool(state.get("first_postprocess_gate_passed", False))
        state.setdefault("first_postprocess_gate_case", None)
        state["first_postprocess_gate_failed_attempts"] = int(
            state.get("first_postprocess_gate_failed_attempts", 0) or 0
        )

        run_dir = output / "runs" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        if not dashboard["ok"]:
            events.append(event("dashboard", "fail", message=dashboard["stderr"]))
        else:
            events.append(event("dashboard", "ok"))
        upstream_action = str(upstream.get("decision", {}).get("action", "unknown"))
        events.append(
            event(
                "upstream_emc",
                "ok" if upstream.get("ok") else "fail",
                message=f"{upstream_action}; cases={len(upstream_cases)}",
            )
        )
        if duplicate_count:
            events.append(event("duplicate_gate", "blocked", message=f"{duplicate_count} duplicate/conflict rows"))

        package_root = output / "md_packages" / timestamp
        batch_tag = timestamp
        submitted_cases = set(state.get("submitted_md_cases", []))
        retry_counts = state.get("md_retry_counts", {})
        if not isinstance(retry_counts, dict):
            retry_counts = {}
        launch_fail_counts = state.get("md_launch_fail_counts", {})
        if not isinstance(launch_fail_counts, dict):
            launch_fail_counts = {}
        failed_recovery_attempts = recovery_attempts(state)
        slot_quarantine = state.get("md_slot_quarantine", {})
        if not isinstance(slot_quarantine, dict):
            slot_quarantine = {}

        md_submit_ok_count = 0
        if args.execute and not duplicate_count:
            write_heartbeat(output, "md_submit_start", run_id=timestamp, planned=len(md_plans))
            for plan in md_plans:
                if plan["action"] != "launch_manual_md":
                    events.append(event("md_submit", "blocked", case_id=plan["case_id"], message=plan["action"]))
                    continue
                try:
                    run_tag = f"supervisor_{timestamp}_{plan['case_id']}"
                    run_dir_case = generate_package(plan["case_id"], plan, package_root, run_tag)
                    result = deploy_and_launch(plan, run_dir_case, batch_tag)
                    submitted_cases.add(plan["case_id"])
                    if plan.get("retry_failed_stage"):
                        retry_counts[plan["case_id"]] = int(retry_counts.get(plan["case_id"], 0) or 0) + 1
                    md_submit_ok_count += 1
                    events.append(
                        event(
                            "md_submit",
                            "ok",
                            case_id=plan["case_id"],
                            host_label=plan["host_label"],
                            host=plan["host"],
                            node=plan["node"],
                            gpu=plan["gpu"],
                            message=result["remote_case_dir"],
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    launch_fail_counts[plan["case_id"]] = int(launch_fail_counts.get(plan["case_id"], 0) or 0) + 1
                    key = slot_key_for_plan(plan)
                    slot_quarantine[key] = {
                        "case_id": plan["case_id"],
                        "host_label": plan.get("host_label", ""),
                        "host": plan.get("host", ""),
                        "node": plan.get("node", ""),
                        "gpu": plan.get("gpu", ""),
                        "quarantined_at": iso_now(),
                        "until": (datetime.now() + timedelta(hours=6)).isoformat(timespec="seconds"),
                        "reason": str(exc).splitlines()[0][:200],
                    }
                    events.append(event("md_submit", "fail", case_id=plan["case_id"], message=str(exc)))
        else:
            for plan in md_plans:
                events.append(
                    event(
                        "md_submit",
                        "dry-run",
                        case_id=plan["case_id"],
                        host_label=plan.get("host_label", ""),
                        host=plan.get("host", ""),
                        node=plan.get("node", ""),
                        gpu=plan.get("gpu", ""),
                        message=plan["action"],
                    )
                )

        recovery_ok_count = 0
        recovery_root = output / "failed_recovery" / timestamp
        recovery_package_root = output / "recovery_md_packages" / timestamp
        if args.execute and args.execute_recovery and not duplicate_count:
            for plan in recovery_plans:
                case_id = str(plan.get("case_id", ""))
                try:
                    failed_recovery_attempts[case_id] = int(failed_recovery_attempts.get(case_id, 0) or 0) + 1
                    result = execute_recovery_plan(plan, recovery_package_root, recovery_root, timestamp)
                    if int(result.get("returncode", 0) or 0) != 0:
                        raise RuntimeError(
                            f"recovery failed for {case_id}: {result.get('stderr', '') or result.get('stdout', '')}"
                        )
                    submitted_cases.add(case_id)
                    recovery_ok_count += 1
                    events.append(
                        event(
                            "failed_recovery",
                            "ok",
                            case_id=case_id,
                            host_label=plan.get("host_label", plan.get("source_host_label", "")),
                            host=plan.get("host", plan.get("source_host", "")),
                            node=plan.get("node", ""),
                            gpu=plan.get("gpu", ""),
                            message=f"{plan.get('recovery_kind')} {result.get('mode')} {result.get('remote_case_dir', '')}",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    events.append(event("failed_recovery", "fail", case_id=case_id, message=str(exc)))
        else:
            for plan in recovery_plans:
                events.append(
                    event(
                        "failed_recovery",
                        "dry-run",
                        case_id=plan.get("case_id", ""),
                        host_label=plan.get("host_label", plan.get("source_host_label", "")),
                        host=plan.get("host", plan.get("source_host", "")),
                        node=plan.get("node", ""),
                        gpu=plan.get("gpu", ""),
                        message=plan.get("recovery_kind", ""),
                    )
                )

        control_queue_decision: dict[str, Any] = {}
        if not duplicate_count and not md_plans and not recovery_plans:
            try:
                control_queue_decision = run_pelcb100_control_queue(execute=bool(args.execute))
                events.append(
                    event(
                        "validation_control_queue",
                        "ok" if control_queue_decision.get("ok") else "fail",
                        message=str(control_queue_decision.get("action", control_queue_decision.get("reason", ""))),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                control_queue_decision = {"ok": False, "action": "error", "stderr": str(exc)}
                events.append(event("validation_control_queue", "fail", message=str(exc)))
        else:
            control_queue_decision = {
                "ok": True,
                "skipped": True,
                "reason": "main md/recovery work has priority",
            }
            events.append(event("validation_control_queue", "skipped", message=control_queue_decision["reason"]))

        if args.execute and not duplicate_count and not args.no_restart:
            for plan in restart_plans:
                try:
                    result = execute_restart(plan)
                    events.append(
                        event(
                            "restart",
                            "ok",
                            case_id=plan["case_id"],
                            host_label=plan["host_label"],
                            host=plan["host"],
                            node=plan["node"],
                            gpu=plan["gpu"],
                            message=result["stdout"][:500],
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    events.append(event("restart", "fail", case_id=plan["case_id"], message=str(exc)))
        else:
            for plan in restart_plans:
                events.append(event("restart", "dry-run", case_id=plan["case_id"], message=plan["reason"]))

        postprocess_decision: dict[str, Any] = {}
        write_heartbeat(output, "postprocess_status_start", run_id=timestamp, ready_count=ready_count)
        post_jobs = find_active_postprocess(state)
        postprocess_safe_to_collect = bool(state.get("postprocess_safe_to_collect"))
        closure = run_closure_status(post_jobs)
        summary = closure.get("payload", {}).get("summary", {})
        discovered_jobs = [str(item) for item in summary.get("explicit_jobs", []) if str(item).strip()]
        # When no explicit jobs are supplied, the closure helper discovers
        # today's lcb300_* Slurm rows.  That is the important safety gate for
        # supervisor restarts after state loss.
        active_pp = int(summary.get("active_count") or 0)
        failed_pp = int(summary.get("failed_count") or 0)
        failed_ids = failed_postprocess_job_ids(summary)
        ignored_failed_ids = {
            str(item) for item in state.get("ignored_failed_postprocess_jobs", []) if str(item).strip()
        }
        ignored_stale_failed_pp = bool(
            failed_pp > 0 and not post_jobs and failed_ids and set(failed_ids).issubset(ignored_failed_ids)
        )
        if ignored_stale_failed_pp:
            failed_pp = 0
        has_pp_rows = int(summary.get("sacct_row_count") or 0) + int(summary.get("squeue_row_count") or 0) > 0
        if active_pp or failed_pp or (has_pp_rows and summary.get("complete_gate_ok")):
            jobs_for_closure = post_jobs or discovered_jobs
            postprocess_decision = {
                "action": "closure_status",
                "jobs": jobs_for_closure,
                "complete_gate_ok": summary.get("complete_gate_ok", False),
                "active_count": active_pp,
                "failed_count": failed_pp,
                "report": closure.get("json", ""),
                "safe_to_collect": postprocess_safe_to_collect,
            }
            if args.execute and active_pp == 0 and (summary.get("complete_gate_ok") or failed_pp > 0):
                if failed_pp > 0:
                    triage = write_postprocess_failure_triage(output, summary, jobs_for_closure)
                    postprocess_decision["failure_triage"] = triage
                    mark_failed_postprocess_jobs_triaged(state, failed_ids, triage)
                    postprocess_decision["ignored_after_triage"] = failed_ids
                    state["postprocess_jobs"] = []
                    state["postprocess_safe_to_collect"] = False
                    events.append(
                        event(
                            "postprocess_closure",
                            "fail",
                            message=f"postprocess jobs failed; triage={triage.get('md', '')}",
                        )
                    )
                elif not postprocess_safe_to_collect:
                    message = (
                        "collect/merge blocked: postprocess jobs were not submitted by the "
                        "current safe orchestrator generation"
                    )
                    postprocess_decision["closure_all_blocked"] = message
                    stale_jobs = jobs_for_closure
                    state.setdefault("discarded_unsafe_postprocess_jobs", [])
                    state["discarded_unsafe_postprocess_jobs"].append(
                        {
                            "discarded_at": iso_now(),
                            "jobs": stale_jobs,
                            "reason": message,
                            "failed_count": failed_pp,
                            "complete_gate_ok": bool(summary.get("complete_gate_ok")),
                        }
                    )
                    state["postprocess_jobs"] = []
                    state["postprocess_safe_to_collect"] = False
                    postprocess_decision["discarded_jobs"] = stale_jobs
                    events.append(event("postprocess_closure", "discarded", message=message))
                else:
                    plan_csv = current_postprocess_plan_csv(state)
                    closed = run_closure_all(jobs_for_closure, execute=True, plan_csv=plan_csv)
                    postprocess_decision["closure_all"] = closed
                    if closed["returncode"] == 0:
                        state["postprocess_jobs"] = []
                        state["postprocess_safe_to_collect"] = False
                        if state.get("postprocess_smoke_active") and not state.get("postprocess_smoke_passed"):
                            state["postprocess_smoke_passed"] = True
                            state["postprocess_smoke_active"] = False
                            state["postprocess_smoke_passed_at"] = iso_now()
                        formal = refresh_formal_outputs()
                        postprocess_decision["formal_outputs"] = formal
                        events.append(
                            event(
                                "postprocess_closure",
                                "ok" if formal.get("ok") else "warn",
                                message=closed.get("applied_merge_manifest") or closed.get("merge_plan", ""),
                            )
                        )
                    else:
                        events.append(event("postprocess_closure", "fail", message=closed["stderr"]))
            else:
                events.append(event("postprocess_closure", "wait", message=json.dumps(postprocess_decision, ensure_ascii=False)))
        elif ready_count and not duplicate_count:
            first_gate_passed = bool(state.get("first_postprocess_gate_passed"))
            if not first_gate_passed:
                first_case = ready_cases[0] if ready_cases else {}
                gate_result = first_postprocess_smoke_gate(first_case, output, state)
                postprocess_decision = {
                    "action": "first_postprocess_gate",
                    "case_id": gate_result.get("case_id", ""),
                    "status": gate_result.get("status", ""),
                    "pass": bool(gate_result.get("pass")),
                    "gate_result": str(output / "first_postprocess_gate" / str(gate_result.get("case_id", "unknown_case")) / "gate_result.json"),
                    "missing_fields": gate_result.get("missing_fields", []),
                }
                if gate_result.get("status") == "halt":
                    postprocess_decision["halt_postprocess"] = (
                        "first postprocess gate failed at least three times; bulk postprocess requires manual unblock"
                    )
                    events.append(
                        event(
                            "first_postprocess_gate",
                            "halt",
                            case_id=gate_result.get("case_id", ""),
                            message=";".join(str(item) for item in gate_result.get("missing_fields", [])),
                        )
                    )
                elif gate_result.get("status") == "trigger_failed_previous":
                    events.append(
                        event(
                            "first_postprocess_gate",
                            "trigger_failed_previous",
                            case_id=gate_result.get("case_id", ""),
                            message=";".join(str(item) for item in gate_result.get("missing_fields", [])),
                        )
                    )
                elif gate_result.get("status") == "trigger_baseline_density":
                    if args.execute:
                        trig = submit_postprocess_single(
                            case_id=str(gate_result.get("case_id", "")),
                            suite="baseline_density",
                            output=output,
                            execute=True,
                        )
                        postprocess_decision["trigger"] = trig
                        postprocess_decision["action"] = "first_postprocess_gate_trigger"
                        if trig.get("returncode") == 0 and trig.get("job_ids"):
                            state["first_postprocess_gate_trigger"] = {
                                "case_id": gate_result.get("case_id", ""),
                                "triggered_at": iso_now(),
                                "job_id": trig.get("job_id", ""),
                                "job_ids": trig.get("job_ids", []),
                                "run_dir": trig.get("run_dir", ""),
                            }
                            state["postprocess_jobs"] = trig.get("job_ids", [])
                            state["postprocess_safe_to_collect"] = True
                            state["postprocess_orchestrator_run_dir"] = trig.get("run_dir", "")
                            events.append(
                                event(
                                    "first_postprocess_gate",
                                    "triggered",
                                    case_id=gate_result.get("case_id", ""),
                                    message=str(trig.get("job_id", "")),
                                )
                            )
                        elif trig.get("returncode") == 0:
                            events.append(
                                event(
                                    "first_postprocess_gate",
                                    "trigger_warn",
                                    case_id=gate_result.get("case_id", ""),
                                    message="baseline_density plan produced no sbatch job id",
                                )
                            )
                        else:
                            events.append(
                                event(
                                    "first_postprocess_gate",
                                    "trigger_fail",
                                    case_id=gate_result.get("case_id", ""),
                                    message=str(trig.get("stderr", ""))[:500],
                                )
                            )
                    else:
                        trig = submit_postprocess_single(
                            case_id=str(gate_result.get("case_id", "")),
                            suite="baseline_density",
                            output=output,
                            execute=False,
                        )
                        postprocess_decision["trigger_plan"] = trig
                        postprocess_decision["action"] = "first_postprocess_gate_trigger_plan"
                        events.append(
                            event(
                                "first_postprocess_gate",
                                "trigger_dry_run",
                                case_id=gate_result.get("case_id", ""),
                                message=str(trig.get("run_dir", "")),
                            )
                        )
                elif gate_result.get("pass"):
                    state["first_postprocess_gate_passed"] = True
                    state["first_postprocess_gate_case"] = gate_result.get("case_id", "")
                    state["postprocess_smoke_passed"] = True
                    events.append(event("first_postprocess_gate", "ok", case_id=gate_result.get("case_id", "")))
                else:
                    state["first_postprocess_gate_failed_attempts"] = int(
                        state.get("first_postprocess_gate_failed_attempts", 0) or 0
                    ) + 1
                    if state["first_postprocess_gate_failed_attempts"] >= 3:
                        postprocess_decision["halt_postprocess"] = (
                            "first postprocess gate failed at least three times; bulk postprocess requires manual unblock"
                        )
                        events.append(
                            event(
                                "first_postprocess_gate",
                                "halt",
                                case_id=gate_result.get("case_id", ""),
                                message=";".join(str(item) for item in gate_result.get("missing_fields", [])),
                            )
                        )
                    else:
                        events.append(
                            event(
                                "first_postprocess_gate",
                                "fail",
                                case_id=gate_result.get("case_id", ""),
                                message=";".join(str(item) for item in gate_result.get("missing_fields", [])),
                            )
                        )
            elif args.execute:
                submitted = submit_postprocess(source_host="all", execute=True, max_cases=None)
                postprocess_decision = submitted
                postprocess_decision["mode"] = "bulk"
                if submitted["returncode"] == 0 and submitted["job_ids"]:
                    state["postprocess_jobs"] = submitted["job_ids"]
                    state["postprocess_safe_to_collect"] = True
                    state["postprocess_orchestrator_run_dir"] = submitted.get("run_dir", "")
                    events.append(event("postprocess_submit", "ok", message=";".join(submitted["job_ids"])))
                elif submitted["returncode"] == 0:
                    events.append(event("postprocess_submit", "warn", message="no sbatch job ids parsed"))
                else:
                    events.append(event("postprocess_submit", "fail", message=submitted["stderr"]))
            else:
                planned = submit_postprocess(source_host="all", execute=False, max_cases=None)
                postprocess_decision = planned
                postprocess_decision["mode"] = "bulk"
                events.append(event("postprocess_submit", "dry-run", message=planned.get("run_dir", "")))
        else:
            postprocess_decision = {"action": "none", "reason": "no ready cases or duplicate gate"}

        write_heartbeat(output, "health_monitor_start", run_id=timestamp)
        remote_health = cycle_count % 6 == 0
        health_monitor = refresh_health_monitor(remote_logs=remote_health)
        write_heartbeat(output, "health_monitor_done", run_id=timestamp, health_ok=health_monitor.get("ok"))
        events.append(
            event(
                "health_monitor",
                "ok" if health_monitor.get("ok") else "fail",
                message=json.dumps(health_monitor.get("status_counts", {}), ensure_ascii=False),
            )
        )
        eta_retry_unstable = bool((done_count != previous_done_count) or (cycle_count % 6 == 0))
        eta_summary = write_runtime_eta_summary(
            output,
            runtime,
            retry_unstable_calibration=eta_retry_unstable,
        )
        eta_summary["retry_unstable_calibration"] = eta_retry_unstable

        state["updated_at"] = iso_now()
        state["cycle_count"] = cycle_count
        state["last_ready_for_analysis_count"] = ready_count
        state["last_strict_done_count"] = done_count
        state["submitted_md_cases"] = sorted(submitted_cases)
        state["md_retry_counts"] = retry_counts
        state["md_launch_fail_counts"] = launch_fail_counts
        state["failed_recovery_attempts"] = failed_recovery_attempts
        state["md_slot_quarantine"] = slot_quarantine
        if args.execute:
            state["dry_run_planned_md_cases"] = []
        else:
            state["dry_run_planned_md_cases"] = sorted(
                set(state.get("dry_run_planned_md_cases", []))
                | {str(plan.get("case_id", "")) for plan in md_plans if plan.get("case_id")}
            )
        state.setdefault("postprocess_jobs", post_jobs)
        write_json(state_path, state)

        postprocess_action = str(postprocess_decision.get("action", ""))
        postprocess_changed = any(
            row.get("action", "").startswith("postprocess_") and row.get("result") not in {"wait", "dry-run"}
            for row in events
        )
        should_refresh_assets = (
            md_submit_ok_count > 0
            or recovery_ok_count > 0
            or ready_count != previous_ready_count
            or done_count != previous_done_count
            or postprocess_changed
            or postprocess_action not in {"", "none", "closure_status"}
            or cycle_count % 6 == 0
        )
        if should_refresh_assets:
            write_heartbeat(output, "publication_assets_start", run_id=timestamp)
            publication_assets = refresh_publication_assets()
            events.append(
                event(
                    "publication_assets",
                    "ok" if publication_assets.get("ok") else "fail",
                    message=publication_assets.get("stdout") or publication_assets.get("stderr", ""),
                )
            )
        else:
            publication_assets = {"ok": True, "skipped": True, "reason": "no MD/ready/postprocess state change this cycle"}
            events.append(event("publication_assets", "skipped", message=publication_assets["reason"]))

        paper_freeze_ready = paper_freeze_ready_status(runtime, state, ready_count, duplicate_count)
        payload = {
            "timestamp": iso_now(),
            "execute": bool(args.execute),
            "dashboard": dashboard,
            "upstream": upstream,
            "duplicate_count": duplicate_count,
            "slots": slots,
            "md_plans": md_plans,
            "failed_recovery": {
                "classification_csv": str(output / "recovery_classification.csv"),
                "classification_md": str(output / "recovery_classification.md"),
                "recoverable_count": sum(1 for row in recovery_rows if row.get("classification") == "recoverable"),
                "manual_review_count": sum(1 for row in recovery_rows if row.get("classification") == "manual_review"),
                "blocked_duplicate_count": sum(1 for row in recovery_rows if row.get("classification") == "blocked_duplicate"),
                "plans": recovery_plans,
                "execute_recovery": bool(args.execute_recovery),
            },
            "restart_plans": restart_plans,
            "ready_for_analysis_count": ready_count,
            "disk_status": disk_status,
            "health_monitor": health_monitor,
            "eta_summary": eta_summary,
            "postprocess_decision": postprocess_decision,
            "publication_assets": publication_assets,
            "paper_freeze_ready": paper_freeze_ready,
            "validation_control_queue": control_queue_decision,
            "state_path": str(state_path),
        }
        write_json(run_dir / "supervisor_decision.json", payload)
        write_json(output / "supervisor_last_decision.json", payload)
        write_decisions(output / "supervisor_decisions.md", payload)
        write_decisions(run_dir / "supervisor_decisions.md", payload)
        append_csv(output / "supervisor_events.csv", events, event_fields(events))
        write_heartbeat(output, "cycle_done", run_id=timestamp, md_plans=len(md_plans), ready_count=ready_count)
        return payload


def write_windows_launcher(output_dir: Path) -> None:
    config_part = f' --config "{CONFIG_PATH}"' if CONFIG_PATH else ""
    script = f"""# Run one LCB supervisor cycle. Schedule this with Windows Task Scheduler if needed.
Set-Location "{ROOT}"
python "{SCRIPT_DIR / 'lcb_supervisor.py'}"{config_part} --once --execute --max-submit-per-round 8 --max-upstream-per-round 10 --no-restart
"""
    write_text(output_dir / "run_lcb_supervisor.ps1", script)
    task_name = "PELCB100Supervisor"
    install = f"""$ErrorActionPreference = "Stop"
$TaskName = "{task_name}"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ScriptDir "run_lcb_supervisor.ps1"
if (-not (Test-Path $RunScript)) {{
    throw "Missing supervisor launcher: $RunScript"
}}
$Existing = schtasks /query /tn $TaskName 2>$null
if ($LASTEXITCODE -eq 0) {{
    Write-Host "Task $TaskName already exists. Not creating a duplicate."
    exit 0
}}
$Action = "powershell.exe -ExecutionPolicy Bypass -File `"$RunScript`""
schtasks /create /tn $TaskName /sc ONLOGON /tr $Action /rl HIGHEST
if ($LASTEXITCODE -ne 0) {{
    throw "Failed to create scheduled task $TaskName"
}}
Write-Host "Created scheduled task $TaskName"
"""
    uninstall = f"""$ErrorActionPreference = "Stop"
$TaskName = "{task_name}"
schtasks /query /tn $TaskName 2>$null
if ($LASTEXITCODE -ne 0) {{
    Write-Host "Task $TaskName does not exist."
    exit 0
}}
schtasks /delete /tn $TaskName /f
if ($LASTEXITCODE -ne 0) {{
    throw "Failed to delete scheduled task $TaskName"
}}
Write-Host "Deleted scheduled task $TaskName"
"""
    write_text(output_dir / "install_task_scheduler.ps1", install)
    write_text(output_dir / "uninstall_task_scheduler.ps1", uninstall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Allow remote submission/restart/postprocessing actions.")
    parser.add_argument("--once", action="store_true", help="Run one supervisor cycle and exit.")
    parser.add_argument("--daemon", action="store_true", help="Run continuously.")
    parser.add_argument("--interval-min", type=float, default=10.0)
    parser.add_argument("--no-restart", action="store_true", help="Disable automatic whitelisted restarts.")
    parser.add_argument(
        "--execute-recovery",
        action="store_true",
        help="Allow guarded failed-stage recovery. Requires --execute; default cycles only classify and plan recovery.",
    )
    parser.add_argument(
        "--max-recovery-per-round",
        type=int,
        default=1,
        help="Maximum failed-stage recovery actions per supervisor cycle.",
    )
    parser.add_argument("--max-submit-per-round", type=int, default=None)
    parser.add_argument(
        "--max-upstream-per-round",
        type=int,
        default=10,
        help="Maximum EMC/bridge cases to advance per supervisor cycle.",
    )
    return parser.parse_args()


def main() -> int:
    global CONFIG_PATH, PROJECT_ID, OUTPUT_DIR, RUNTIME_DIR, UPSTREAM_DIR, ORCHESTRATOR_DIR, CLOSURE_DIR, BRIDGE_DIRS
    global NEW_SERVER, NEWNEW_SERVER, OLD_SERVER, NEW_REMOTE_ROOT, NEWNEW_REMOTE_ROOT, NEW_GMX, NEWNEW_GMX
    global DISABLED_NEW_NODES, DISABLED_MD_HOSTS, PRODUCTION_CLUSTER_PROFILES
    args = parse_args()
    config = load_lcb_config(args.config)
    CONFIG_PATH = args.config or (SCRIPT_DIR / "configs" / "lcb300.json")
    PROJECT_ID = config.project_id
    if args.output_dir is None:
        args.output_dir = config.supervisor_output_dir
    OUTPUT_DIR = args.output_dir
    RUNTIME_DIR = config.runtime_output_dir
    UPSTREAM_DIR = config.emc_output_dir
    ORCHESTRATOR_DIR = SCRIPT_DIR / "output" / f"{config.project_id}_orchestrator"
    CLOSURE_DIR = SCRIPT_DIR / "output" / f"{config.project_id}_postprocess_closure"
    BRIDGE_DIRS = [config.default_emc_bridge_root, *config.bridge_roots]
    NEW_SERVER = config.new_server
    NEWNEW_SERVER = config.newnew_server
    OLD_SERVER = config.old_server
    NEW_REMOTE_ROOT = (config.remote_md_roots.get("new") or [NEW_REMOTE_ROOT])[0]
    NEWNEW_REMOTE_ROOT = (config.remote_md_roots.get("newnew") or [NEWNEW_REMOTE_ROOT])[0]
    NEW_GMX = config.new_gmx
    NEWNEW_GMX = config.newnew_gmx
    DISABLED_NEW_NODES = set(config.disabled_new_nodes)
    DISABLED_MD_HOSTS = set(config.disabled_md_hosts)
    PRODUCTION_CLUSTER_PROFILES = production_cluster_profiles_from_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_windows_launcher(args.output_dir)
    if not args.daemon:
        args.once = True
    while True:
        try:
            payload = run_once(args)
            print(f"Supervisor cycle complete: {payload['timestamp']}")
            print(f"Decisions: {args.output_dir / 'supervisor_decisions.md'}")
        except Exception as exc:  # noqa: BLE001
            error = event("supervisor", "fail", message=str(exc))
            append_csv(args.output_dir / "supervisor_events.csv", [error], event_fields([error]))
            print(f"Supervisor failed: {exc}", file=sys.stderr)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval_min * 60.0))


if __name__ == "__main__":
    raise SystemExit(main())
