#!/usr/bin/env python3
"""Bounded repair-routing controller for PE-LCB100 reliability states.

The controller turns the released reliability map into a deterministic repair
plan.  By default it is dry-run only: it writes route tables and ledgers but
does not submit MD, rerun postprocessing, change rules, or edit the canonical
release.  Mutating execution is deliberately narrow and sidecar-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from pelcb_audit_diagnosis import (
    diagnose_case,
    infer_diagnoses,
    route_from_diagnosis,
    select_primary_diagnosis,
)
from lcb_project_config import load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent


def _project_config_path() -> str | Path:
    return (
        os.environ.get("PELCB_PROJECT_CONFIG")
        or os.environ.get("LCB_PROJECT_CONFIG")
        or SCRIPT_DIR / "configs" / "pelcb100.json"
    )


PROJECT_CONFIG = load_lcb_config(_project_config_path())
AGGREGATE_DIR = PROJECT_CONFIG.integrated_output_dir
DEFAULT_OUTPUT_DIR = PROJECT_CONFIG.repair_loop_dir
DEFAULT_SOURCE_CSV = (
    PROJECT_CONFIG.supervisor_output_dir / "o7_clean_engine_audit_20260522" / "source_dirs_combined_unique.csv"
)
DEFAULT_O7_QUEUE_ROOT = PROJECT_CONFIG.o7_queue_dir
DEFAULT_O7_CLOSURE_STATUS = PROJECT_CONFIG.o7_closure_dir / "o7_clean_run_status.csv"
DEFAULT_O7_SAFE_RETRY_ROOT = PROJECT_CONFIG.supervisor_output_dir / "o7_clean_safe_retry"
DEFAULT_OVERLAY_DIR = PROJECT_CONFIG.overlay_dir
DEFAULT_RUNTIME_ACTIVE_CSV = PROJECT_CONFIG.runtime_output_dir / "runtime_active_processes.csv"
DEFAULT_CLEAN_REGATE_DIR = SCRIPT_DIR / "output" / "pelcb100_clean_overlay_regate"
DEFAULT_MSD_REFIT_CLASSIFICATION = DEFAULT_CLEAN_REGATE_DIR / "msd_extend_vs_refit_20260611.csv"
DEFAULT_WARN_AUDIT_ROOT = PROJECT_CONFIG.warn_audit_output_dir
SAFE_RETRY_SUBMISSION_GRACE_SECONDS = 30 * 60
O7_QUEUE = SCRIPT_DIR / "o7_clean_supervisor_queue.py"
O7_CLOSURE = SCRIPT_DIR / "o7_clean_closure.py"
O7_SAFE_RETRY = SCRIPT_DIR / "o7_clean_safe_retry.py"
NVT_EXTENSION = SCRIPT_DIR / "pelcb_clean_nvt_extension.py"
STRICT_MSD_REFIT = SCRIPT_DIR / "pelcb_strict_msd_refit.py"
MSD_WINDOW_REPAIR = SCRIPT_DIR / "pelcb_msd_window_repair.py"
MSD_WINDOW_OVERLAY = SCRIPT_DIR / "pelcb_build_msd_window_overlay.py"
CLEAN_REGATE = SCRIPT_DIR / "pelcb_clean_overlay_regate.py"
PUBLICATION_ASSETS = SCRIPT_DIR / "lcb_publication_assets.py"
RUNTIME_DASHBOARD = SCRIPT_DIR / "lcb_runtime_dashboard.py"

REPAIR_SCHEMA_VERSION = "pelcb_repair_loop.v1"
RUNTIME_ACTIVE_FIELDS = [
    "host_label",
    "host",
    "node",
    "gpu_id",
    "pid",
    "etimes",
    "pcpu",
    "pmem",
    "case_id",
    "stage",
    "ntomp",
    "deffnm",
    "cwd",
    "cmd",
]
OLD_CLEAN_HOST = os.environ.get("PELCB_OLD_SERVER") or "${REMOTE_USER}@private-3f2a50db.invalid"
OLD_CLEAN_INTEL_NODES = tuple(f"node{i}" for i in range(4, 8))

OLD_CLEAN_RUNTIME_SCAN = r"""
import getpass
import json
import os
import pwd
import re
import subprocess
from pathlib import Path

case_re = re.compile(r"(lcb_pe(?:24cuda|24)_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+|lcb_pe_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+)")
stages = {"steep", "rlx", "pr", "nvt_prod"}

def run(cmd):
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    except Exception:
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": ""})()

def parse_cmd_value(tokens, flag):
    for i, token in enumerate(tokens):
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return ""

def normalize_case_id(case_id):
    match = re.match(r"lcb_pe(?:24cuda|24)_(\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+)$", str(case_id or ""), flags=re.I)
    if match:
        return "lcb_pe_" + match.group(1)
    return str(case_id or "")

def parse_case(value):
    match = case_re.search(str(value))
    return normalize_case_id(match.group(1)) if match else ""

def parse_stage(cwd, deffnm):
    name = Path(cwd).name if cwd else ""
    if name in stages:
        return name
    if deffnm in stages:
        return deffnm
    return deffnm or name

def username_for_pid(pid):
    try:
        status = (Path("/proc") / str(pid) / "status").read_text(errors="ignore")
        uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
        if not uid_line:
            return ""
        return pwd.getpwuid(int(uid_line.split()[1])).pw_name
    except Exception:
        return ""

def pid_stats(pid):
    result = run(["ps", "-p", str(pid), "-o", "etimes=", "-o", "pcpu=", "-o", "pmem="])
    parts = result.stdout.split()
    return {
        "etimes": parts[0] if len(parts) > 0 else "",
        "pcpu": parts[1] if len(parts) > 1 else "",
        "pmem": parts[2] if len(parts) > 2 else "",
    }

uid = os.getuid()
payload = {"ok": True, "hostname": os.uname().nodename, "user": getpass.getuser(), "processes": []}
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    pid = proc.name
    try:
        status = (proc / "status").read_text(errors="ignore")
        uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
        if not uid_line or int(uid_line.split()[1]) != uid:
            continue
        raw = (proc / "cmdline").read_bytes()
        cmd = raw.replace(b"\0", b" ").decode(errors="replace").strip()
        if "mdrun" not in cmd or ("gmx" not in cmd and "gromacs" not in cmd):
            continue
        cwd = os.readlink(proc / "cwd")
    except Exception:
        continue
    tokens = cmd.split()
    deffnm = parse_cmd_value(tokens, "-deffnm")
    case_id = parse_case(cwd) or parse_case(cmd)
    if not case_id:
        continue
    stats = pid_stats(pid)
    payload["processes"].append({
        "pid": pid,
        "cwd": cwd,
        "cmd": cmd,
        "case_id": case_id,
        "stage": parse_stage(cwd, deffnm),
        "gpu_id": parse_cmd_value(tokens, "-gpu_id") or "slurm",
        "ntomp": parse_cmd_value(tokens, "-ntomp"),
        "deffnm": deffnm,
        **stats,
    })
print(json.dumps(payload, ensure_ascii=False))
"""


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def source_case_sequence(case_id: str) -> str:
    match = re.search(r"lcb_pe_(\d+)_", str(case_id or ""))
    return match.group(1).zfill(3) if match else ""


def case_ids_by_sequence(paths: Iterable[Path] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths or (DEFAULT_O7_CLOSURE_STATUS,):
        rows = read_csv(path)
        if rows.empty or "case_id" not in rows.columns:
            continue
        for case_id in rows["case_id"].dropna().astype(str):
            seq = source_case_sequence(case_id)
            if seq and case_id.startswith("lcb_pe_"):
                out.setdefault(seq, case_id)
    return out


def write_heartbeat(output_dir: Path, phase: str, **extra: Any) -> None:
    write_json(
        output_dir / "repair_loop_heartbeat.json",
        {
            "updated_at": iso_now(),
            "pid": os.getpid(),
            "phase": phase,
            **extra,
        },
    )


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_daemon_lock(output_dir: Path) -> Path:
    lock_dir = output_dir / "repair_loop.daemon.lock"
    owner_path = lock_dir / "owner.json"
    if lock_dir.exists():
        owner = {}
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        owner_pid = int(owner.get("pid", 0) or 0)
        if owner_pid and owner_pid != os.getpid() and pid_is_running(owner_pid):
            raise RuntimeError(f"repair-loop daemon already active with pid {owner_pid}")
        shutil.rmtree(lock_dir, ignore_errors=True)
    lock_dir.mkdir(parents=True, exist_ok=False)
    write_json(owner_path, {"pid": os.getpid(), "started_at": iso_now()})
    return lock_dir


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_automatic_review_packets(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    warn_audit_jsonl: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Materialize terminal REVIEW evidence without a human routing step."""

    packet_dir = output_dir / "automatic_review_packets"
    packets: list[dict[str, Any]] = []
    for row in rows:
        route = str(row.get("repair_route") or "")
        stage = str(row.get("closed_loop_stage") or "")
        status = str(row.get("case_status") or "")
        automation_level = str(row.get("automation_level") or "")
        is_review = (
            status.upper() == "REVIEW"
            or stage.lower() == "review"
            or "review" in route.lower()
            or automation_level == "automatic_review_packet"
        )
        if not is_review:
            continue
        case_id = str(row.get("case_id") or "unknown_case")
        safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
        packet_path = packet_dir / f"{safe_case_id}.json"
        packet = {
            "schema_version": "pelcb_automatic_review_packet.v1",
            "created_at": iso_now(),
            "case_id": case_id,
            "terminal_status": "REVIEW",
            "assignment_mode": "automatic",
            "human_action_required_for_status": False,
            "eligible_for_pass_dataset": False,
            "new_evidence_required_for_reentry": True,
            "review_trigger": {
                "repair_route": route,
                "primary_audit_diagnosis": row.get("primary_audit_diagnosis", ""),
                "remaining_warn_sources": row.get("remaining_warn_sources", ""),
                "repair_attempt_index": row.get("repair_attempt_index", ""),
                "max_repair_attempts": row.get("max_repair_attempts", ""),
            },
            "source_warn_audit_jsonl": str(warn_audit_jsonl or ""),
            "evidence_record": row,
        }
        write_json(packet_path, packet)
        packets.append({
            "case_id": case_id,
            "terminal_status": "REVIEW",
            "assignment_mode": "automatic",
            "packet_path": str(packet_path),
        })
    write_jsonl(output_dir / "automatic_review_packet_index.jsonl", packets)
    return packets


STATE_CHANGING_REPAIR_ACTIONS = {
    "submit_o7_clean_full_supervisor",
    "submit_o7_clean_density_validation",
    "retry_failed_clean_nvt_safe_offload",
    "extend_clean_nvt_prod_from_checkpoint",
}
NVT_EXTENSION_REPAIR_ACTIONS = {
    "extend_clean_nvt_prod_from_checkpoint",
}
MSD_EXTENSION_MAX_COMPLETED_ROUNDS = 1
MSD_EXTENSION_TARGET_TOTAL_PS = 800_000.0


def parse_iso_like_timestamp(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def submitted_case_ids_from_supervisor_result(row: dict[str, Any]) -> list[str]:
    result = row.get("result") or {}
    if not isinstance(result, dict):
        return []
    stdout = cell_text(result.get("stdout"))
    if not stdout:
        return []
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    if cell_text(payload.get("action")) == "blocked_no_slots":
        return []
    plan_csv = cell_text(payload.get("plan_csv"))
    if not plan_csv:
        return []
    plan = read_csv(Path(plan_csv))
    if plan.empty or "case_id" not in plan.columns:
        return []
    if "status" in plan.columns:
        plan = plan[plan["status"].map(lambda value: upper_text(value) == "SUBMITTED")]
    return [cell_text(case_id) for case_id in plan["case_id"].tolist() if cell_text(case_id)]


def successful_nvt_extension_case_ids(tag: str) -> list[str]:
    plan_path = (
        PROJECT_CONFIG.supervisor_output_dir
        / "o7_clean_nvt_extension"
        / tag
        / "pelcb_clean_nvt_extension_plan.csv"
    )
    plan = read_csv(plan_path)
    if plan.empty or "case_id" not in plan.columns or "result" not in plan.columns:
        return []
    rows = plan[
        plan["result"].map(
            lambda value: "'ok': True" in cell_text(value) or '"ok": true' in cell_text(value).lower()
        )
    ]
    return [cell_text(case_id) for case_id in rows["case_id"].tolist() if cell_text(case_id)]


def read_nvt_extension_attempt_times(
    output_dir: Path | None = None,
) -> dict[str, float]:
    """Return the latest confirmed clean-NVT extension launch time per case.

    Repair-loop ledgers count only actions initiated by this controller.  Some
    long MSD extensions can also be submitted through the narrow extension tool
    during migration or recovery.  The extension tool writes a sidecar plan for
    every confirmed launch, so this reader uses those plans as the authority
    for whether existing clean postprocess outputs pre-date the current
    extended trajectory.
    """

    root = output_dir or PROJECT_CONFIG.supervisor_output_dir / "o7_clean_nvt_extension"
    latest: dict[str, float] = {}
    if not root.exists():
        return latest
    for plan_path in sorted(root.glob("*/pelcb_clean_nvt_extension_plan.csv")):
        plan = read_csv(plan_path)
        if plan.empty or "case_id" not in plan.columns:
            continue
        for _, row in plan.iterrows():
            case_id = cell_text(row.get("case_id"))
            if not case_id:
                continue
            result_text = cell_text(row.get("result", "")).lower()
            if "'ok': true" not in result_text and '"ok": true' not in result_text:
                continue
            timestamp = parse_iso_like_timestamp(row.get("timestamp"))
            if timestamp is None:
                try:
                    timestamp = plan_path.stat().st_mtime
                except OSError:
                    continue
            latest[case_id] = max(latest.get(case_id, 0.0), float(timestamp))
    return latest


def selected_safe_retry_case_ids_from_result(row: dict[str, Any]) -> list[str]:
    """Return case ids that the safe-retry tool actually selected.

    A safe-retry command can exit successfully with ``selected_count == 0``
    when a requested case is blocked by live runtime state or has no eligible
    retry slot.  Such no-op invocations must not consume a bounded repair round.
    """
    result = row.get("result") or {}
    if not isinstance(result, dict):
        return []
    stdout = cell_text(result.get("stdout"))
    if not stdout:
        return []
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    selected_count = int(payload.get("selected_count", 0) or 0)
    if selected_count <= 0:
        return []
    output_dir = cell_text(payload.get("output_dir"))
    if output_dir:
        summary_path = Path(output_dir) / "o7_clean_safe_retry_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
        rows = summary.get("rows") if isinstance(summary, dict) else None
        if isinstance(rows, list):
            selected = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                case_id = cell_text(item.get("case_id"))
                action = cell_text(item.get("action"))
                if case_id and action:
                    selected.append(case_id)
            if selected:
                return selected
    requested = row.get("case_ids") or []
    if isinstance(requested, str):
        requested = source_list(requested)
    return [cell_text(case_id) for case_id in requested if cell_text(case_id)][:selected_count]


def read_repair_attempt_counts(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    since: float | None = None,
    *,
    actions: set[str] | None = None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    counted_actions = actions if actions is not None else STATE_CHANGING_REPAIR_ACTIONS
    if not output_dir.exists():
        return {}
    for ledger_path in sorted(output_dir.glob("*/repair_loop_execution_ledger.jsonl")):
        try:
            ledger_mtime = ledger_path.stat().st_mtime
        except OSError:
            ledger_mtime = None
        if since is not None and ledger_mtime is not None and ledger_mtime < since:
            continue
        try:
            lines = ledger_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is not None:
                row_time = parse_iso_like_timestamp(row.get("timestamp"))
                if row_time is None:
                    row_time = ledger_mtime
                if row_time is not None and row_time < since:
                    continue
            action = cell_text(row.get("action"))
            if action not in counted_actions:
                continue
            case_ids = row.get("actual_case_ids") or []
            if action in NVT_EXTENSION_REPAIR_ACTIONS and not case_ids:
                # Extension bounds count only confirmed launches/submissions.  Older
                # daemon ledgers often recorded requested case_ids even when the
                # extension tool returned BLOCKED/queued/no-slot; those are not
                # real nvt_prod extension rounds.
                continue
            if action == "retry_failed_clean_nvt_safe_offload":
                case_ids = selected_safe_retry_case_ids_from_result(row)
                if not case_ids:
                    continue
            if not case_ids and action in {"submit_o7_clean_full_supervisor", "submit_o7_clean_density_validation"}:
                case_ids = submitted_case_ids_from_supervisor_result(row)
                if not case_ids:
                    continue
            if not case_ids:
                case_ids = row.get("case_ids") or []
            if isinstance(case_ids, str):
                case_ids = source_list(case_ids)
            for case_id in case_ids:
                text = cell_text(case_id)
                if text:
                    counts[text] += 1
    return dict(counts)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def upper_text(value: Any) -> str:
    return cell_text(value).upper()


def source_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [cell_text(item) for item in value if cell_text(item)]
    text = cell_text(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text.replace("'", '"'))
            if isinstance(payload, list):
                return [cell_text(item) for item in payload if cell_text(item)]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def source_case_ids(source_csv: Path) -> set[str]:
    if not source_csv.exists():
        return set()
    rows = read_csv(source_csv)
    if rows.empty or "case_id" not in rows.columns:
        return set()
    return {str(case_id) for case_id in rows["case_id"].dropna().astype(str)}


def read_diagnosis_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    """Read optional tri-model WARN diagnosis records for repair routing.

    The repair router remains deterministic: even when the selected diagnosis
    comes from a tri-model audit, routing is still mapped through
    ``pelcb_audit_diagnosis.DIAGNOSIS_TO_ROUTE`` rather than through a model-
    proposed shell command.
    """
    if not path or not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = cell_text(row.get("case_id", ""))
            diagnosis = cell_text(row.get("primary_audit_diagnosis", ""))
            if not case_id or not diagnosis:
                continue
            route = route_from_diagnosis(row)
            row["recommended_reentry_module"] = route["module"]
            row["recommended_action"] = route["repair_action"]
            row["repair_tool"] = route["tool_plan"]["tool"]
            row["tool_parameters"] = dict(route["tool_plan"]["parameters"])
            records[case_id] = row
    return records


def latest_frozen_warn_audit_jsonl(
    audit_root: Path = DEFAULT_WARN_AUDIT_ROOT,
    project_root: Path = SCRIPT_DIR.parents[1],
) -> Path | None:
    """Resolve the canonical frozen full-cohort WARN audit JSONL."""
    pointer = audit_root / "warn_audit_latest.json"
    if pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8-sig"))
            diagnosed_cases = int(payload.get("diagnosed_cases", 0))
            is_legacy_full_cohort = diagnosed_cases == 97 and "full_cohort" not in payload
            is_declared_full_cohort = payload.get("full_cohort") is True
            complete_panels = int(payload.get("model_panel_complete_cases", 0) or 0)
            warn_cases = int(payload.get("warn_cases", 0) or 0)
            declared_cohort_complete = (
                is_declared_full_cohort
                and diagnosed_cases > 0
                and warn_cases == diagnosed_cases
                and complete_panels == diagnosed_cases
            )
            if (
                str(payload.get("mode", "")).lower() == "execute"
                and (is_legacy_full_cohort or declared_cohort_complete)
                and payload.get("jsonl")
            ):
                candidate = Path(str(payload["jsonl"]))
                if not candidate.is_absolute():
                    candidate = project_root / candidate
                if candidate.exists():
                    return candidate
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    candidates = sorted(
        audit_root.glob("*/warn_audit_diagnosis_merged.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def latest_msd_window_overlay_mask(overlay_dir: Path = DEFAULT_OVERLAY_DIR) -> Path | None:
    """Return the newest sidecar MSD-window overlay quality-mask CSV, if any."""
    if not overlay_dir.exists():
        return None
    candidates = sorted(
        overlay_dir.glob("*/ml_quality_mask_msd_window_overlay_v3.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def msd_window_overlay_masks(overlay_dir: Path = DEFAULT_OVERLAY_DIR) -> list[Path]:
    """Return all sidecar MSD-window overlay masks in creation order."""
    if not overlay_dir.exists():
        return []
    return sorted(
        overlay_dir.glob("*/ml_quality_mask_msd_window_overlay_v3.csv"),
        key=lambda path: path.stat().st_mtime,
    )


def read_msd_overlay_states(path: Path | None) -> dict[str, dict[str, Any]]:
    """Read case-level diagnostic status from a sidecar MSD-window overlay.

    The overlay is sidecar evidence, not a canonical table rewrite. Local
    window candidates are retained only as diagnostics under the strict
    publication-grade MSD policy; they cannot close a publication gap.
    """
    if not path or not path.exists():
        return {}
    rows = read_csv(path)
    if rows.empty or "case_id" not in rows.columns or "msd_window_overlay_status" not in rows.columns:
        return {}
    states: dict[str, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        case_id = cell_text(row.get("case_id", ""))
        status = cell_text(row.get("msd_window_overlay_status", ""))
        if not case_id or not status or status == "not_applicable":
            continue
        states[case_id] = {
            "msd_window_overlay_status": status,
            "msd_window_overlay_applied": truthy(row.get("msd_window_overlay_applied", "")),
            "msd_window_overlay_source": str(path),
            "msd_window_overlay_source_tag": cell_text(row.get("msd_window_overlay_source_tag", "")),
            "msd_window_overlay_fit_window_ns": cell_text(row.get("msd_window_overlay_fit_window_ns", "")),
            "msd_window_overlay_note": cell_text(row.get("msd_window_overlay_note", "")),
        }
    return states


def read_msd_overlay_history(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Merge case-level MSD overlay states without dropping earlier batches."""
    states: dict[str, dict[str, Any]] = {}
    for path in paths:
        states.update(read_msd_overlay_states(path))
    return states


def latest_clean_overlay_regate_csv(regate_dir: Path = DEFAULT_CLEAN_REGATE_DIR) -> Path | None:
    """Return the newest clean-engine overlay re-gate CSV, if any."""
    latest = regate_dir / "clean_overlay_regate_latest.json"
    if latest.exists():
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            csv_path = Path(payload.get("summary", {}).get("csv") or "")
            if csv_path.exists():
                return csv_path
        except (json.JSONDecodeError, OSError):
            pass
    candidates = sorted(
        regate_dir.glob("clean_overlay_regate_*/clean_overlay_regate.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_clean_regate_states(path: Path | None) -> dict[str, dict[str, Any]]:
    """Read case-level clean-engine overlay re-gate verdicts."""
    if not path or not path.exists():
        return {}
    rows = read_csv(path)
    if rows.empty or "case_id" not in rows.columns:
        return {}
    states: dict[str, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        case_id = cell_text(row.get("case_id", ""))
        if not case_id:
            continue
        states[case_id] = {
            "clean_regate_source": str(path),
            "clean_targeted_regate_verdict": cell_text(row.get("targeted_regate_verdict", "")),
            "clean_regate_verdict": cell_text(row.get("targeted_regate_verdict", "")),
            "clean_release_overlay_verdict": cell_text(row.get("release_overlay_verdict", "")),
            "clean_publication_grade": cell_text(row.get("release_overlay_verdict", "")).upper() == "PASS",
            "clean_resolved_warn_sources": cell_text(row.get("resolved_warn_sources", "")),
            "clean_remaining_target_warn_sources": cell_text(row.get("remaining_target_warn_sources", "")),
            "clean_warn_sources": cell_text(row.get("clean_warn_sources", "")),
            "clean_density_rule_verdict": cell_text(row.get("density_rule_verdict", "")),
            "clean_density_value_gcc": cell_text(row.get("density_value_gcc", "")),
            "clean_qc_rule_verdict": cell_text(row.get("qc_rule_verdict", "")),
            "clean_missing_outputs": cell_text(row.get("missing_clean_outputs", "")),
            "clean_status_MSD": cell_text(row.get("status_MSD", "")),
            "clean_msd_publication_grade": truthy(row.get("msd_publication_grade", "")),
            "clean_msd_publication_issues": cell_text(row.get("msd_publication_issues", "")),
            "clean_msd_clean_engine_source": cell_text(row.get("msd_clean_engine_source", "")),
            "clean_msd_tail_window_ok": cell_text(row.get("msd_tail_window_ok", "")),
            "clean_msd_fit_R2_publication": cell_text(row.get("msd_fit_R2_publication", "")),
            "clean_msd_fit_quality": cell_text(row.get("fit_quality", "")),
            "clean_msd_fit_begin_ps": cell_text(row.get("fit_begin_ps", "")),
            "clean_msd_fit_end_ps": cell_text(row.get("fit_end_ps", "")),
            "clean_msd_time_last_ps": cell_text(row.get("time_last_ps", "")),
            "clean_msd_loglog_slope": cell_text(row.get("loglog_slope", "")),
            "clean_msd_diffusive_regime_ok": cell_text(row.get("diffusive_regime_ok", "")),
            "clean_msd_regime_warning": cell_text(row.get("regime_warning", "")),
            "clean_msd_window_sensitive": cell_text(row.get("window_sensitive", "")),
            "clean_msd_window_D_cv_pct": cell_text(row.get("window_D_cv_pct", "")),
            "clean_msd_window_overlay_status": cell_text(row.get("clean_msd_window_overlay_status", "")),
            "clean_msd_window_overlay_source": cell_text(row.get("clean_msd_window_overlay_source", "")),
            "clean_msd_window_overlay_fit_window_ns": cell_text(row.get("clean_msd_window_overlay_fit_window_ns", "")),
            "clean_msd_window_overlay_loglog_slope": cell_text(row.get("clean_msd_window_overlay_loglog_slope", "")),
            "clean_msd_window_overlay_D_m2_s": cell_text(row.get("clean_msd_window_overlay_D_m2_s", "")),
            "clean_msd_window_overlay_local_D_cv_pct": cell_text(row.get("clean_msd_window_overlay_local_D_cv_pct", "")),
            "clean_qc_domain_verdicts_json": cell_text(row.get("qc_domain_verdicts_json", "")),
            "clean_status_intrachain_g3": cell_text(row.get("status_intrachain_g3", "")),
            "clean_intrachain_g3_alpha_100_300": cell_text(row.get("intrachain_g3_alpha_100_300", "")),
            "clean_intrachain_g3_time_last_ps": cell_text(row.get("intrachain_g3_time_last_ps", "")),
        }
    return states


def latest_clean_msd_root_for_case(clean_case_id: str, output_root: Path | None = None) -> Path | None:
    clean_case_id = cell_text(clean_case_id)
    if not clean_case_id:
        return None
    root = output_root or (SCRIPT_DIR / "output")
    candidates = sorted(
        root.glob(f"o7clean_msd_*/cases/{clean_case_id}/msd/msd_timeseries.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].parents[3]


def has_warn_source(warn_sources: Iterable[str], source: str) -> bool:
    return source in {str(item) for item in warn_sources}


def clean_publication_pass(clean_regate_state: dict[str, Any]) -> bool:
    return upper_text(clean_regate_state.get("clean_release_overlay_verdict")) == "PASS"


def clean_regate_available(clean_regate_state: dict[str, Any]) -> bool:
    return bool(cell_text(clean_regate_state.get("clean_release_overlay_verdict")))


SCHEDULER_STATE_ALIASES = {
    "active": "active_clean_queue",
    "md_clean_active": "active_clean_queue",
    "active_clean_queue": "active_clean_queue",
    "badconstraints": "scheduler_repair_required",
    "bad_constraints": "scheduler_repair_required",
    "dependencysatisfied": "scheduler_repair_required",
    "dependencyneversatisfied": "scheduler_repair_required",
    "scheduler_repair_required": "scheduler_repair_required",
    "stale_active": "scheduler_repair_required",
    "stale_tracking_gap": "scheduler_repair_required",
    "clean_nvt_done": "clean_nvt_done",
    "waiting_for_slots": "clean_queue_waiting_for_slots",
    "clean_queue_waiting_for_slots": "clean_queue_waiting_for_slots",
    "runtime_fatal": "clean_runtime_fatal",
    "clean_runtime_fatal": "clean_runtime_fatal",
    "checkpoint_pending": "clean_nvt_checkpoint_pending",
    "checkpoint_resume_required": "clean_nvt_checkpoint_pending",
    "clean_nvt_checkpoint_pending": "clean_nvt_checkpoint_pending",
}


def normalize_scheduler_state(value: Any) -> str:
    text = cell_text(value).lower().replace(" ", "").replace("-", "_")
    return SCHEDULER_STATE_ALIASES.get(text, cell_text(value))


def read_scheduler_status_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    """Read optional live/scheduler status overrides.

    The closure CSV is a file-state scan.  It can tell that PR completed and
    NVT is missing, but it cannot prove that a remote job is still alive.  This
    sidecar override lets a live scheduler probe mark stale active rows or bad
    Slurm constraints without editing canonical evidence.
    """
    if not path or not path.exists():
        return {}
    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        records = read_csv(path).to_dict("records")
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    overrides: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = cell_text(record.get("case_id", ""))
        if not case_id:
            continue
        raw_state = (
            record.get("clean_repair_state")
            or record.get("scheduler_repair_state")
            or record.get("scheduler_state")
            or record.get("state")
            or record.get("status")
        )
        clean_state = normalize_scheduler_state(raw_state)
        if not clean_state:
            continue
        scheduler_status = cell_text(record.get("scheduler_status") or record.get("status") or raw_state)
        overrides[case_id] = {
            "clean_repair_state": clean_state,
            "clean_status_source": str(path),
            "scheduler_status_source": str(path),
            "scheduler_status": scheduler_status,
            "scheduler_reason": cell_text(record.get("scheduler_reason") or record.get("reason") or ""),
            "scheduler_repair_action": cell_text(
                record.get("scheduler_repair_action") or record.get("repair_action") or ""
            ),
            "remote_run_dir": cell_text(record.get("remote_run_dir", "")),
            "clean_tag": cell_text(record.get("clean_tag") or record.get("tag") or ""),
        }
    return overrides


def merge_scheduler_overrides(
    clean_states: dict[str, dict[str, Any]],
    scheduler_overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply live scheduler/probe facts over file-state clean repair rows."""
    if not scheduler_overrides:
        return clean_states
    merged = {case_id: dict(state) for case_id, state in clean_states.items()}
    for case_id, override in scheduler_overrides.items():
        current = merged.get(case_id, {})
        merged[case_id] = {
            **current,
            **{key: value for key, value in override.items() if value != ""},
            "scheduler_previous_clean_repair_state": current.get("clean_repair_state", ""),
        }
    return merged


def closed_loop_state_for_route(route: str, action: str, executable_now: bool) -> tuple[str, str]:
    """Map a repair route to an explicit 100-case loop state."""
    if route == "no_action":
        return "released_or_no_repair_needed", "terminal"
    if route == "resolved_by_postprocess_overlay":
        return "resolved_by_postprocess_overlay", "regated"
    if route == "resolved_by_clean_engine_overlay":
        return "resolved_by_clean_engine_overlay", "regated"
    if route == "clean_engine_overlay_ready_for_regate":
        return "clean_engine_overlay_ready_for_regate", "regate"
    if route == "persistent_warn_review":
        return "persistent_warn_review", "review"
    if route == "persistent_msd_relaxed_window_review":
        return "persistent_msd_relaxed_window_review", "review"
    if route == "persistent_density_protocol_review":
        return "persistent_density_protocol_review", "review"
    if route == "persistent_clean_engine_overlay_warn":
        return "persistent_clean_engine_overlay_warn", "review"
    if route in {"review_audit_disagreement"}:
        return "audit_review_required", "review"
    if route in {"repair_md_clean_engine_ready_for_closure", "optional_clean_engine_validation_ready_for_closure"}:
        return "postprocess_closure_ready_or_submitted", "postprocess_closure"
    if route in {"repair_md_clean_engine_active", "optional_clean_engine_validation_active"}:
        return "md_repair_active", "md"
    if route in {"repair_md_clean_engine_waiting_for_slots", "optional_clean_engine_validation_waiting_for_slots"}:
        return "md_repair_waiting_for_capacity", "md"
    if route == "scheduler_repair_required":
        return "scheduler_repair_required", "scheduler"
    if route == "repair_md_clean_engine_safe_retry":
        return "md_safe_retry_ready" if executable_now else "md_safe_retry_planned", "md"
    if route == "repair_postprocess_msd_window":
        return "postprocess_repair_ready" if executable_now else "postprocess_repair_planned", "postprocess"
    if route == "repair_postprocess_clean_msd_window":
        return "clean_postprocess_repair_ready" if executable_now else "clean_postprocess_repair_planned", "postprocess"
    if route == "repair_postprocess_clean_msd":
        return "postprocess_repair_ready" if executable_now else "postprocess_repair_planned", "postprocess"
    if route == "repair_postprocess_fsqt":
        return "postprocess_repair_ready" if executable_now else "postprocess_repair_planned", "postprocess"
    if route == "repair_postprocess_intrachain":
        return "postprocess_repair_ready" if executable_now else "postprocess_repair_planned", "postprocess"
    if route == "repair_postprocess_packing":
        return "postprocess_repair_planned", "postprocess"
    if route in {"repair_md_clean_engine", "optional_clean_engine_validation"}:
        return "md_repair_ready" if executable_now else "md_repair_planned", "md"
    if route == "repair_emc_rebuild":
        return "emc_repair_planned", "emc"
    if action == "none":
        return "released_or_no_repair_needed", "terminal"
    return "review_required", "review"


def runtime_state_for_row(clean_state: dict[str, Any], route: str) -> str:
    """Separate live process state from broader sidecar/queue route labels."""
    if cell_text(clean_state.get("clean_runtime_pid")):
        return "runtime_active"
    clean_repair_state = cell_text(clean_state.get("clean_repair_state"))
    if clean_repair_state == "active_clean_queue":
        return "sidecar_active_no_live_pid"
    if clean_repair_state == "clean_queue_waiting_for_slots" or route.endswith("_waiting_for_slots"):
        return "waiting_for_slots"
    if clean_repair_state == "clean_nvt_done" or "ready_for_closure" in route:
        return "ready_for_closure"
    if clean_repair_state == "clean_nvt_checkpoint_pending":
        return "checkpoint_retry_needed"
    if clean_repair_state == "clean_postprocess_closed":
        return "clean_postprocess_closed"
    return clean_repair_state or "not_started_or_unknown"


def pass90_category_for_row(route: str, runtime_state: str, effective_publication_grade: bool) -> str:
    """High-level status used by the 90+ publication-grade closure audit."""
    if effective_publication_grade or route == "no_action":
        return "release_pass"
    if runtime_state == "runtime_active":
        return "runtime_active"
    if runtime_state == "sidecar_active_no_live_pid":
        return "sidecar_state_active_no_live_process"
    if runtime_state in {"waiting_for_slots", "ready_for_closure", "checkpoint_retry_needed"}:
        return runtime_state
    if route == "review_audit_disagreement":
        return "audit_review"
    if route in {
        "persistent_clean_engine_overlay_warn",
        "persistent_density_protocol_review",
        "bounded_attempts_exhausted_review",
        "persistent_msd_relaxed_window_review",
        "persistent_msd_window_diagnostic_review",
    }:
        return "persistent_warn_or_review"
    return "needs_routing_or_regate"


def bounded_attempt_guard_bypass(row: dict[str, Any]) -> bool:
    """Return whether a row is still finishing the current repair round.

    The two-round publication-grade policy is a limit on completed repair
    rounds that have had a chance to re-enter the rule layer. A checkpointed
    clean NVT run has not reached that rule-layer re-gate yet; resuming it is
    part of the same unfinished round, not a third scientific repair attempt.
    """
    return (
        row.get("repair_route") == "repair_md_clean_engine_safe_retry"
        and row.get("clean_repair_state") == "clean_nvt_checkpoint_pending"
        and row.get("repair_action") == "resume_incomplete_clean_nvt_from_checkpoint"
    )


def msd_extension_completed_round_exhausted(row: dict[str, Any], extension_attempts: int) -> bool:
    """Treat one bounded MSD extension plus failed re-gate as terminal.

    This is a framework-level MSD policy for any polymer/material routed through
    the clean COM publication gate.  Checkpoint resumes are still allowed while
    the first extension is unfinished; once the clean trajectory has closed,
    postprocessed, re-gated, and still has a tail slope below 0.95 after one
    bounded +400 ns extension (typically about 400 ns -> 800 ns total), more
    automatic NVT extension is converted to review instead of consuming another
    GPU round.  Some future adapters may not expose `clean_msd_time_last_ps` in
    the repair row; in that case a completed bounded extension round plus a
    closed clean postprocess ledger is treated as the bounded evidence. If
    `clean_msd_time_last_ps` is present, it remains authoritative and must
    reach the bounded target.
    """
    return bool(
        row.get("repair_route") == "repair_md_clean_nvt_extension"
        and int(extension_attempts) >= MSD_EXTENSION_MAX_COMPLETED_ROUNDS
        and cell_text(row.get("clean_repair_state")) == "clean_postprocess_closed"
        and (
            clean_msd_reached_bounded_extension_target(row)
            or float_or_none(row.get("clean_msd_time_last_ps", "")) is None
        )
        and clean_msd_tail_extension_required(row)
        and not truthy(row.get("effective_publication_grade"))
    )


def clean_msd_reached_bounded_extension_target(row: dict[str, Any]) -> bool:
    """Return whether clean-MSD evidence spans the bounded 800 ns target.

    A launched extension attempt is not enough to send a case to final review.
    The rule-layer evidence must itself contain a postprocessed clean MSD
    trajectory reaching the bounded target; otherwise the case is still waiting
    for collection or deterministic MSD recomputation.
    """

    time_last_ps = float_or_none(row.get("clean_msd_time_last_ps", ""))
    return bool(time_last_ps is not None and time_last_ps + 1.0 >= MSD_EXTENSION_TARGET_TOTAL_PS)


def clean_msd_at_bounded_cap_still_warn(row: dict[str, Any]) -> bool:
    """Return whether the clean MSD gate has reached the 800 ns cap and still blocks release.

    This evidence-based guard is intentionally independent of repair-attempt
    counters.  A case may have valid postprocessed 800 ns clean-MSD evidence
    even when the launch ledger that created the extension is absent or was
    recorded under a different route.  Once such evidence remains
    non-publication-grade, the framework should retain the case for review
    rather than scheduling another deterministic MSD postprocess loop.
    """

    if truthy(row.get("effective_publication_grade")):
        return False
    if truthy(row.get("clean_msd_publication_grade")):
        return False
    clean_warns = set(source_list(row.get("clean_warn_sources", "")))
    remaining_target_warns = set(source_list(row.get("clean_remaining_target_warn_sources", "")))
    has_msd_warn = "msd_warn" in clean_warns or "msd_warn" in remaining_target_warns
    return bool(has_msd_warn and clean_msd_reached_bounded_extension_target(row))


def clean_msd_postprocess_stale_after_extension(
    row: dict[str, Any],
    extension_attempt_times: Mapping[str, float],
) -> bool:
    """Return whether clean-MSD outputs pre-date a confirmed NVT extension.

    A clean sidecar can have complete postprocess files from the pre-extension
    trajectory.  Once an NVT extension has been launched, those files must not
    be used to send the case directly to review or to justify another
    extension; the extended trajectory must first be postprocessed and re-gated.
    """

    case_id = cell_text(row.get("case_id"))
    extension_time = float(extension_attempt_times.get(case_id, 0.0) or 0.0)
    if extension_time <= 0:
        return False
    if cell_text(row.get("clean_repair_state")) != "clean_postprocess_closed":
        return False

    msd_mtime = float_or_none(row.get("clean_msd_postprocess_mtime", ""))
    if msd_mtime is not None:
        return msd_mtime + 1.0 < extension_time

    try:
        mtimes = json.loads(cell_text(row.get("clean_postprocess_mtimes", "")) or "{}")
    except json.JSONDecodeError:
        mtimes = {}
    msd_mtime = float_or_none(mtimes.get("MSD", ""))
    return bool(msd_mtime is not None and msd_mtime + 1.0 < extension_time)


CLEAN_REPAIR_STATE_PRIORITY = {
    "clean_queue_waiting_for_slots": 10,
    "active_clean_queue": 20,
    "clean_runtime_fatal": 30,
    "clean_nvt_checkpoint_pending": 35,
    "clean_nvt_done": 40,
    "clean_postprocess_closed": 50,
}


def clean_postprocess_overlay_state(
    clean_case_id: str,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Return whether a clean-engine sidecar case has full postprocess outputs.

    This is intentionally sidecar-only.  It checks the O7 clean postprocess
    products that are collected under ``output/o7clean_*`` and does not infer a
    canonical release verdict.
    """

    clean_case_id = str(clean_case_id or "").strip()
    if not clean_case_id:
        return {"clean_postprocess_state": "", "clean_postprocess_missing": []}
    root = output_root or (SCRIPT_DIR / "output")
    required: dict[str, list[Path]] = {
        "density": [root / "o7clean_density_energy" / clean_case_id / "analysis.json"],
        "Rg": list(root.glob(f"o7clean_rg_*/cases/{clean_case_id}/rg/rg_summary.json")),
        "RDF": [root / "o7clean_integrated" / "RDF" / "raw" / "cases" / clean_case_id / "rdf" / "rdf_summary.json"],
        "shape": [
            root
            / "o7clean_integrated"
            / "ShapeTensor"
            / "raw"
            / "cases"
            / clean_case_id
            / "shape_tensor"
            / "shape_summary.json"
        ],
        "MSD": list(root.glob(f"o7clean_msd_*/cases/{clean_case_id}/msd/msd_summary.json")),
        "intrachain_g1": [
            root
            / "o7clean_integrated"
            / "IntrachainDynamics"
            / "cases"
            / clean_case_id
            / "intrachain"
            / "g1_monomer_summary.json"
        ],
        "contact": [
            root
            / "o7clean_integrated"
            / "IntrachainDynamics"
            / "cases"
            / clean_case_id
            / "intrachain"
            / "contact_probability_summary.json"
        ],
        "intrachain_g3": [
            root
            / "o7clean_integrated"
            / "IntrachainDynamics"
            / "cases"
            / clean_case_id
            / "intrachain"
            / "g1_g3_summary.json"
        ],
    }
    present: dict[str, str] = {}
    mtimes: dict[str, float] = {}
    missing: list[str] = []
    for observable, candidates in required.items():
        found = next((path for path in candidates if path.exists() and path.stat().st_size > 0), None)
        if found is None:
            missing.append(observable)
        else:
            present[observable] = str(found)
            try:
                mtimes[observable] = float(found.stat().st_mtime)
            except OSError:
                pass
    state = "clean_postprocess_closed" if not missing else "clean_postprocess_incomplete"
    msd_related_mtimes = [
        value
        for key, value in mtimes.items()
        if key in {"MSD", "intrachain_g3"}
    ]
    return {
        "clean_postprocess_state": state,
        "clean_postprocess_missing": missing,
        "clean_postprocess_paths": present,
        "clean_postprocess_mtimes": mtimes,
        "clean_postprocess_latest_mtime": max(mtimes.values()) if mtimes else "",
        "clean_msd_postprocess_mtime": max(msd_related_mtimes) if msd_related_mtimes else "",
    }


def normalize_clean_case_to_source_case(text: Any) -> str:
    """Map O7 clean-run identifiers/paths back to the canonical PE-LCB100 case ID."""
    raw = cell_text(text)
    if not raw:
        return ""
    source_match = re.search(r"(lcb_pe_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+)", raw, flags=re.I)
    if source_match:
        return source_match.group(1)
    clean_match = re.search(r"lcb_pe(?:24cuda|24)_(\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+)", raw, flags=re.I)
    if clean_match:
        return "lcb_pe_" + clean_match.group(1)
    return ""


def read_runtime_active_states(
    runtime_active_csv: Path | None = DEFAULT_RUNTIME_ACTIVE_CSV,
    *,
    max_age_seconds: int = 7200,
) -> dict[str, dict[str, Any]]:
    """Read live runtime-dashboard active clean MD processes as case-state overrides."""
    if not runtime_active_csv or not runtime_active_csv.exists():
        return {}
    age_seconds = time.time() - runtime_active_csv.stat().st_mtime
    if max_age_seconds > 0 and age_seconds > max_age_seconds:
        return {}
    active = read_csv(runtime_active_csv)
    if active.empty:
        return {}
    states: dict[str, dict[str, Any]] = {}
    for _, row in active.iterrows():
        case_id = (
            cell_text(row.get("case_id", ""))
            or normalize_clean_case_to_source_case(row.get("cwd", ""))
            or normalize_clean_case_to_source_case(row.get("cmd", ""))
        )
        if not case_id:
            continue
        cwd = cell_text(row.get("cwd", ""))
        states[case_id] = {
            "clean_repair_state": "active_clean_queue",
            "clean_status": "LIVE_RUNTIME_ACTIVE",
            "clean_status_source": str(runtime_active_csv),
            "clean_runtime_pid": cell_text(row.get("pid", "")),
            "clean_runtime_node": cell_text(row.get("node", "")),
            "clean_runtime_gpu_id": cell_text(row.get("gpu_id", "")),
            "clean_runtime_stage": cell_text(row.get("stage", "")),
            "clean_runtime_cwd": cwd,
            "remote_run_dir": cwd.removesuffix("/nvt_prod") if cwd else "",
        }
    return states


def merge_runtime_active_states(
    clean_states: dict[str, dict[str, Any]],
    runtime_active_states: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Live process evidence overrides closure-scan checkpoint/incomplete states."""
    if not runtime_active_states:
        return clean_states
    merged = {case_id: dict(state) for case_id, state in clean_states.items()}
    for case_id, runtime_state in runtime_active_states.items():
        current = merged.get(case_id, {})
        merged[case_id] = {
            **current,
            **runtime_state,
            "scheduler_previous_clean_repair_state": current.get("clean_repair_state", ""),
        }
    return merged


def prefer_clean_state(
    states: dict[str, dict[str, Any]],
    case_id: str,
    candidate: dict[str, Any],
) -> None:
    current = states.get(case_id)
    if current is None:
        states[case_id] = candidate
        return
    current_priority = CLEAN_REPAIR_STATE_PRIORITY.get(str(current.get("clean_repair_state", "")), 0)
    candidate_priority = CLEAN_REPAIR_STATE_PRIORITY.get(str(candidate.get("clean_repair_state", "")), 0)
    current_mtime = float(current.get("clean_status_mtime", 0) or 0)
    candidate_mtime = float(candidate.get("clean_status_mtime", 0) or 0)
    if candidate_priority > current_priority or (
        candidate_priority == current_priority and candidate_mtime >= current_mtime
    ):
        states[case_id] = candidate


def float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_msd_fit_window_exceeds_trajectory(clean_regate_state: dict[str, Any]) -> bool:
    """Return whether the requested clean-MSD fit extends beyond the trajectory.

    Re-running deterministic postprocessing cannot repair a window whose end
    time is absent from the trajectory.  Such a case must return to bounded MD
    extension (or review after the sampling cap) instead of looping through
    the same refit indefinitely.
    """

    fit_end_ps = float_or_none(clean_regate_state.get("clean_msd_fit_end_ps", ""))
    time_last_ps = float_or_none(clean_regate_state.get("clean_msd_time_last_ps", ""))
    return bool(
        fit_end_ps is not None
        and time_last_ps is not None
        and fit_end_ps > time_last_ps + 1.0
    )


def clean_msd_tail_extension_required(clean_regate_state: dict[str, Any]) -> bool:
    """Return whether a closed clean trajectory needs more NVT sampling.

    Publication-grade MSD re-enters clean NVT extension when the trajectory has
    not reached a stable late-time diffusive fit.  Window-policy, D-stability,
    and chain-scale displacement diagnostics are handled by postprocessing or
    review first and do not by themselves justify more production sampling.
    """
    issues = {
        item.strip()
        for item in cell_text(clean_regate_state.get("clean_msd_publication_issues", "")).split(";")
        if item.strip()
    }
    clean_warns = set(source_list(clean_regate_state.get("clean_warn_sources", "")))
    remaining_target_warns = set(source_list(clean_regate_state.get("clean_remaining_target_warn_sources", "")))
    slope = float_or_none(clean_regate_state.get("clean_msd_loglog_slope", ""))
    fit_window_exceeds_trajectory = clean_msd_fit_window_exceeds_trajectory(clean_regate_state)
    has_msd_warn = "msd_warn" in clean_warns or "msd_warn" in remaining_target_warns
    if fit_window_exceeds_trajectory:
        return bool(
            not truthy(clean_regate_state.get("clean_msd_publication_grade", ""))
            and has_msd_warn
        )
    postprocess_first_issues = {
        "missing_fit_or_trajectory_time",
        "msd_fit_R2_below_0.95",
        "D_mean_not_positive",
    }
    if issues & postprocess_first_issues or any(item.startswith("fit_quality=gmx_msd_failed") for item in issues):
        return False
    under_sampled_issues = {
        "msd_tail_slope_not_diffusive",
        "diffusive_regime_not_confirmed",
    }
    needs_more_sampling = bool(issues & under_sampled_issues)
    # Extension is reserved for trajectories that are still too slow to reach
    # the diffusive tail.  Slopes above the strict band point to window/refit
    # review rather than more sampling.
    slope_indicates_undersampling = slope is not None and slope < 0.95
    return bool(
        needs_more_sampling
        and not truthy(clean_regate_state.get("clean_msd_publication_grade", ""))
        and has_msd_warn
        and slope_indicates_undersampling
    )


def clean_fsqt_postprocess_required(clean_regate_state: dict[str, Any]) -> bool:
    """Return whether the remaining clean-overlay incompleteness is Fs(q,t).

    The clean postprocess stack has a deterministic dynamics suite for
    Fs(q,t).  A missing tau-alpha should re-enter that postprocess route before
    it is treated as a persistent release review item.
    """

    if "evidence_incomplete_warn" not in set(source_list(clean_regate_state.get("clean_warn_sources", ""))):
        return False
    try:
        domains = json.loads(cell_text(clean_regate_state.get("clean_qc_domain_verdicts_json", "")) or "{}")
    except json.JSONDecodeError:
        domains = {}
    return str(domains.get("fsqt", "")).upper() == "WARN"


def clean_msd_postprocess_required(clean_regate_state: dict[str, Any]) -> bool:
    """Return whether clean-MSD evidence should be recomputed before MD extension.

    Missing/failed MSD fits are postprocessing evidence gaps.  They should be
    re-entered through the deterministic clean postprocess closure before the
    case is interpreted as a long-trajectory sampling problem.
    """

    clean_warns = set(source_list(clean_regate_state.get("clean_warn_sources", ""))) | set(
        source_list(clean_regate_state.get("clean_remaining_target_warn_sources", ""))
    )
    if "msd_warn" not in clean_warns:
        return False
    if clean_msd_fit_window_exceeds_trajectory(clean_regate_state):
        return False
    issues = {
        item.strip()
        for item in cell_text(clean_regate_state.get("clean_msd_publication_issues", "")).split(";")
        if item.strip()
    }
    postprocess_first_issues = {
        "missing_fit_or_trajectory_time",
        "msd_fit_R2_below_0.95",
        "D_mean_not_positive",
    }
    return bool(
        issues & postprocess_first_issues
        or any(item.startswith("fit_quality=gmx_msd_failed") for item in issues)
        or str(clean_regate_state.get("clean_status_MSD", "")).strip().upper() == "MISSING"
    )


# The intrachain g3 chain-COM log-log slope (alpha) is fit over the fixed
# 100-300 ns window (`intrachain_g3_alpha_100_300`).  A clean trajectory shorter
# than this window cannot supply the alpha value through any amount of
# deterministic postprocessing.
INTRACHAIN_G3_ALPHA_FIT_END_PS = 300_000.0


def clean_intrachain_alpha_window_unreachable(clean_regate_state: dict[str, Any]) -> bool:
    """Return whether the intrachain g3 alpha_100_300 metric is unattainable.

    The publication gate fits the g3 chain-COM log-log slope over the fixed
    100-300 ns window.  When the clean-engine trajectory itself is shorter than
    that window, the alpha value cannot be produced by re-running the
    deterministic postprocess closure; only a longer trajectory (an NVT
    extension) could supply it.  This is a bounded sampling limit rather than a
    postprocess evidence gap, so the case must be retained for review instead of
    looping through the intrachain postprocess closure.  The guard is deliberately
    narrow: the g3 evidence must already be present and healthy (status ok), with
    only the fixed-window alpha missing because the trajectory ends before the
    window closes.
    """

    if "evidence_incomplete_warn" not in set(source_list(clean_regate_state.get("clean_warn_sources", ""))):
        return False
    try:
        domains = json.loads(cell_text(clean_regate_state.get("clean_qc_domain_verdicts_json", "")) or "{}")
    except json.JSONDecodeError:
        domains = {}
    if str(domains.get("intrachain_dynamics", "")).upper() != "WARN":
        return False
    if str(clean_regate_state.get("clean_status_intrachain_g3", "")).strip().lower() not in {"ok", "pass"}:
        return False
    if float_or_none(clean_regate_state.get("clean_intrachain_g3_alpha_100_300", "")) is not None:
        return False
    time_last_ps = float_or_none(clean_regate_state.get("clean_intrachain_g3_time_last_ps", ""))
    if time_last_ps is None or time_last_ps <= 0.0:
        return False
    return time_last_ps + 1.0 < INTRACHAIN_G3_ALPHA_FIT_END_PS


def clean_intrachain_postprocess_required(clean_regate_state: dict[str, Any]) -> bool:
    """Return whether clean-overlay incompleteness is native intrachain dynamics.

    A written summary with status=warn is not usable release evidence.  It must
    re-enter the deterministic postprocess closure before the case is routed to
    review.  A trajectory that is simply shorter than the fixed g3 alpha fit
    window is excluded: re-running postprocessing cannot lengthen the trajectory,
    so that case is retained for review rather than looped.
    """

    if "evidence_incomplete_warn" not in set(source_list(clean_regate_state.get("clean_warn_sources", ""))):
        return False
    try:
        domains = json.loads(cell_text(clean_regate_state.get("clean_qc_domain_verdicts_json", "")) or "{}")
    except json.JSONDecodeError:
        domains = {}
    if str(domains.get("intrachain_dynamics", "")).upper() != "WARN":
        return False
    if clean_intrachain_alpha_window_unreachable(clean_regate_state):
        return False
    return True


def clean_nvt_extension_eligible(
    clean_state: dict[str, Any],
    clean_regate_state: dict[str, Any] | None = None,
) -> bool:
    """Return whether a clean run is eligible for checkpoint-aware nvt extension."""
    remote_run_dir = cell_text(clean_state.get("remote_run_dir", ""))
    clean_case_id = cell_text(clean_state.get("clean_case_id", ""))
    host_label = cell_text(clean_state.get("clean_target_host_label", ""))
    host = cell_text(clean_state.get("clean_target_host", ""))
    repair_state = cell_text(clean_state.get("clean_repair_state", ""))
    clean_regate_state = clean_regate_state or {}
    if repair_state == "active_clean_queue":
        return False
    if repair_state == "clean_postprocess_closed" and not clean_msd_tail_extension_required(clean_regate_state):
        return False
    if repair_state and repair_state not in {"clean_nvt_checkpoint_pending", "clean_postprocess_closed"}:
        return False
    if not remote_run_dir:
        return False
    is_new_clean = remote_run_dir.startswith("${PELCB100_REMOTE_HOME}/") and not remote_run_dir.startswith("${PELCB100_REMOTE_ROOT}/")
    is_oldserver_clean = remote_run_dir.startswith("${PELCB100_REMOTE_ROOT}/")
    if clean_case_id and "cuda" not in clean_case_id and not is_oldserver_clean:
        return False
    if not (is_new_clean or is_oldserver_clean):
        return False
    if is_new_clean and host_label and host_label not in {"new", "newnew", "cuda2024"}:
        return False
    if is_new_clean and host and "private-4e2f23ff.invalid" not in host and "${REMOTE_USER}@" not in host:
        return False
    if is_oldserver_clean and host and "private-3f2a50db.invalid" not in host and "${REMOTE_USER}@" not in host:
        return False
    return True


def clean_repair_state(
    queue_root: Path,
    closure_status: Path,
    safe_retry_root: Path = DEFAULT_O7_SAFE_RETRY_ROOT,
) -> dict[str, dict[str, Any]]:
    """Return existing clean-sidecar coverage by source case ID.

    A repair loop must not resubmit a case that is already in a clean-engine
    sidecar run.  Submitted/running queue rows are active.  Closure scans with
    completed PR but no NVT completion marker are checkpoint-pending unless a
    fatal log is present; a file-state scan alone cannot prove that a remote job
    is still alive.  Dry-run PREPARED rows are planning artifacts and must not
    block later repair scheduling.  Completed nvt_prod rows are considered
    ready for postprocess closure/overlay review.
    """
    states: dict[str, dict[str, Any]] = {}

    for status_path in sorted(queue_root.glob("*/o7_new_cuda2024_status.csv")):
        status = read_csv(status_path)
        if status.empty or "case_id" not in status.columns:
            continue
        status_mtime = status_path.stat().st_mtime
        for _, status_row in status.iterrows():
            case_id = cell_text(status_row.get("case_id", ""))
            if not case_id:
                continue
            status_value = cell_text(status_row.get("status", "")).upper()
            if status_value == "BLOCKED_NO_SLOTS":
                prefer_clean_state(states, case_id, {
                    "clean_repair_state": "clean_queue_waiting_for_slots",
                    "clean_tag": status_path.parent.name,
                    "clean_status_source": str(status_path),
                    "clean_status_mtime": status_mtime,
                    "clean_status": status_value,
                    "clean_blocked_reason": status_row.get("blocked_reason", ""),
                    "remote_run_dir": status_row.get("remote_run_dir", ""),
                })
                continue
            if status_value not in {"SUBMITTED", "RUNNING"}:
                continue
            prefer_clean_state(states, case_id, {
                "clean_repair_state": "active_clean_queue",
                "clean_tag": status_path.parent.name,
                "clean_status_source": str(status_path),
                "clean_status_mtime": status_mtime,
                "clean_status": status_value,
                "remote_run_dir": status_row.get("remote_run_dir", ""),
            })

    for plan_path in sorted(queue_root.glob("*/o7_stage1_old2024_plan.csv")):
        status = read_csv(plan_path)
        if status.empty or "case_id" not in status.columns:
            continue
        status_mtime = plan_path.stat().st_mtime
        for _, status_row in status.iterrows():
            case_id = cell_text(status_row.get("case_id", ""))
            if not case_id:
                continue
            status_value = cell_text(status_row.get("status", "")).upper()
            if status_value not in {"SUBMITTED", "RUNNING"}:
                continue
            prefer_clean_state(states, case_id, {
                "clean_repair_state": "active_clean_queue",
                "clean_case_id": cell_text(status_row.get("clean_case_id", "")),
                "clean_tag": plan_path.parent.name,
                "clean_status_source": str(plan_path),
                "clean_status_mtime": status_mtime,
                "clean_status": status_value,
                "remote_run_dir": status_row.get("remote_run_dir", ""),
                "clean_target_host_label": "old",
                "clean_target_host": "${REMOTE_USER}@private-3f2a50db.invalid",
                "clean_engine": status_row.get("engine", "old_intel_gromacs_2024.4_cuda"),
                "clean_job_output": status_row.get("job_output", ""),
            })

    closure = read_csv(closure_status)
    closure_mtime = closure_status.stat().st_mtime if closure_status.exists() else 0.0
    if not closure.empty and "case_id" in closure.columns:
        for _, status_row in closure.iterrows():
            case_id = cell_text(status_row.get("case_id", ""))
            if not case_id:
                continue
            nvt_done = truthy(status_row.get("nvt_prod_done"))
            fatal_entries: list[tuple[str, str]] = []
            for stage in ("steep", "rlx", "pr", "nvt_prod"):
                fatal_log = cell_text(status_row.get(f"{stage}_fatal_log", ""))
                if fatal_log:
                    fatal_entries.append((stage, fatal_log))
            current = states.get(case_id, {})
            if nvt_done:
                clean_case_id = cell_text(status_row.get("clean_case_id", ""))
                postprocess_state = clean_postprocess_overlay_state(clean_case_id)
                clean_repair_state_value = (
                    "clean_postprocess_closed"
                    if postprocess_state.get("clean_postprocess_state") == "clean_postprocess_closed"
                    else "clean_nvt_done"
                )
                states[case_id] = {
                    **current,
                    "clean_repair_state": clean_repair_state_value,
                    "clean_case_id": clean_case_id,
                    "clean_tag": status_row.get("tag", current.get("clean_tag", "")),
                    "clean_status_source": str(closure_status),
                    "clean_status_mtime": closure_mtime,
                    "remote_run_dir": status_row.get("remote_run_dir", current.get("remote_run_dir", "")),
                    "clean_target_host_label": status_row.get("source_host_label", current.get("clean_target_host_label", "")),
                    "clean_target_host": status_row.get("source_host", current.get("clean_target_host", "")),
                    "clean_node": status_row.get("node", current.get("clean_node", "")),
                    "clean_gpu_id": status_row.get("gpu_id", current.get("clean_gpu_id", "")),
                    **postprocess_state,
                }
            elif fatal_entries:
                stage, fatal_log = fatal_entries[-1]
                states[case_id] = {
                    **current,
                    "clean_repair_state": "clean_runtime_fatal",
                    "clean_tag": status_row.get("tag", current.get("clean_tag", "")),
                    "clean_status_source": str(closure_status),
                    "clean_status_mtime": closure_mtime,
                    "remote_run_dir": status_row.get("remote_run_dir", current.get("remote_run_dir", "")),
                    "clean_target_host_label": status_row.get("source_host_label", current.get("clean_target_host_label", "")),
                    "clean_target_host": status_row.get("source_host", current.get("clean_target_host", "")),
                    "clean_node": status_row.get("node", current.get("clean_node", "")),
                    "clean_gpu_id": status_row.get("gpu_id", current.get("clean_gpu_id", "")),
                    "clean_fatal_stage": stage,
                    "clean_fatal_log": fatal_log,
                    "clean_fatal_reason": f"{stage} fatal log detected in clean-engine closure scan",
                }
            elif any(truthy(status_row.get(field)) for field in ("steep_done", "rlx_done", "pr_done")):
                states[case_id] = {
                    **current,
                    "clean_repair_state": "clean_nvt_checkpoint_pending",
                    "clean_case_id": cell_text(status_row.get("clean_case_id", current.get("clean_case_id", ""))),
                    "clean_tag": status_row.get("tag", current.get("clean_tag", "")),
                    "clean_status_source": str(closure_status),
                    "clean_status_mtime": closure_mtime,
                    "remote_run_dir": status_row.get("remote_run_dir", current.get("remote_run_dir", "")),
                    "clean_target_host_label": status_row.get("source_host_label", current.get("clean_target_host_label", "")),
                    "clean_target_host": status_row.get("source_host", current.get("clean_target_host", "")),
                    "clean_node": status_row.get("node", current.get("clean_node", "")),
                    "clean_gpu_id": status_row.get("gpu_id", current.get("clean_gpu_id", "")),
                    "clean_resume_stage": "nvt_prod",
                    "clean_status": "NVT_CHECKPOINT_PENDING",
                    "nvt_prod_has_tpr": status_row.get("nvt_prod_has_tpr", ""),
                    "nvt_prod_has_cpt": status_row.get("nvt_prod_has_cpt", ""),
                    "nvt_prod_has_edr": status_row.get("nvt_prod_has_edr", ""),
                }

    for summary_path in sorted(safe_retry_root.glob("*/o7_clean_safe_retry_summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not summary.get("execute"):
            continue
        for retry_row in summary.get("rows", []):
            case_id = cell_text(retry_row.get("case_id", ""))
            if not case_id:
                continue
            result = retry_row.get("result") or {}
            if not result.get("ok"):
                continue
            current = states.get(case_id, {})
            if current.get("clean_repair_state") in {"clean_nvt_done", "clean_postprocess_closed"}:
                continue
            # A safe-retry summary is a historical submission record.  It must
            # only override an older fatal/checkpoint scan. A newer closure scan
            # proves the submitted retry has since stopped and remains decisive.
            # Live processes are merged later from the runtime dashboard.
            summary_mtime = summary_path.stat().st_mtime
            summary_age = max(0.0, time.time() - summary_mtime)
            if summary_age > SAFE_RETRY_SUBMISSION_GRACE_SECONDS:
                # Historical submission proves an attempt occurred, not that
                # its remote process is still alive. Live runtime evidence is
                # merged after this file-state pass and can restore active.
                continue
            candidate = {
                **current,
                "clean_repair_state": "active_clean_queue",
                "clean_tag": current.get("clean_tag", ""),
                "clean_status_source": str(summary_path),
                "clean_status_mtime": summary_mtime,
                "clean_status": "SAFE_RETRY_SUBMITTED",
                "remote_run_dir": retry_row.get("remote_run_dir", current.get("remote_run_dir", "")),
                "clean_safe_retry_mode": retry_row.get("mode", ""),
                "clean_safe_retry_pid": result.get("stdout", ""),
            }
            current_mtime = float(current.get("clean_status_mtime", 0) or 0)
            if float(candidate["clean_status_mtime"]) >= current_mtime:
                states[case_id] = candidate
    return states


def classify_row(
    row: pd.Series,
    source_ids: set[str],
    clean_states: dict[str, dict[str, Any]],
    diagnosis_overrides: dict[str, dict[str, Any]] | None = None,
    msd_overlay_states: dict[str, dict[str, Any]] | None = None,
    clean_regate_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    case_id = str(row.get("case_id", ""))
    case_status = str(row.get("case_status", "") or "")
    warning_reason = str(row.get("warning_reason", "") or "")
    ml_usable = truthy(row.get("ml_usable"))
    audit_disagreement_count = int(float(row.get("audit_disagreement_count", 0) or 0))
    a3_member = truthy(row.get("a3_locked_eval_member"))
    source_available = case_id in source_ids
    clean_state = clean_states.get(case_id, {})
    msd_overlay_state = (msd_overlay_states or {}).get(case_id, {})
    clean_regate_state = (clean_regate_states or {}).get(case_id, {})
    publication_grade = truthy(row.get("publication_grade"))
    clean_release_pass = clean_publication_pass(clean_regate_state)
    effective_publication_grade = publication_grade or clean_release_pass
    publication_grade_source = (
        "canonical_v1_publication_grade"
        if publication_grade
        else ("clean_release_overlay" if clean_release_pass else "")
    )
    fallback_diagnosis = diagnose_case(row, clean_state)
    llm_diagnosis = (diagnosis_overrides or {}).get(case_id)
    diagnosis = {**fallback_diagnosis, **llm_diagnosis} if llm_diagnosis else fallback_diagnosis
    if llm_diagnosis and "cause_diagnosis" not in llm_diagnosis:
        diagnosis["cause_diagnosis"] = diagnosis.get("primary_audit_diagnosis", diagnosis.get("audit_diagnosis", ""))
    original_warn_sources = [str(item) for item in diagnosis.get("warn_sources", [])]
    diagnosis_primary = str(
        diagnosis.get("primary_audit_diagnosis")
        or diagnosis.get("audit_diagnosis")
        or diagnosis.get("cause_diagnosis")
        or ""
    )
    if effective_publication_grade:
        diagnosis.update(
            {
                "rule_verdict": "PASS",
                "warn_sources": [],
                "warn_source": "none",
                "primary_audit_diagnosis": "none",
                "cause_diagnosis": "none",
                "secondary_audit_diagnoses": [],
                "audit_diagnosis": "none",
                "diagnosis_basis": [],
                "recommended_reentry_module": "audit_review",
                "recommended_action": "none",
                "repair_tool": "none",
                "tool_parameters": {},
                "repair_instruction": {},
                "confidence": 1.0,
                "requires_freeze": False,
                "requires_profile_freeze": False,
                "rationale": (
                    "case is publication-grade under the canonical release"
                    if publication_grade
                    else "case passed the full clean release-overlay gate"
                ),
                "max_repair_attempts": 0,
            }
        )
        diagnosis_primary = "none"
        original_warn_sources = []
    if diagnosis_primary != "cross_model_disagreement":
        original_warn_sources = [source for source in original_warn_sources if source != "audit_disagreement_warn"]
        diagnosis["warn_sources"] = original_warn_sources
        diagnosis["warn_source"] = ",".join(original_warn_sources)
    legacy_msd_overlay_stale = (
        not effective_publication_grade
        and has_warn_source(original_warn_sources, "msd_warn")
        and bool(msd_overlay_state.get("msd_window_overlay_status"))
    )
    legacy_msd_overlay_stale_reason = ""
    if legacy_msd_overlay_stale:
        legacy_msd_overlay_stale_reason = (
            "legacy MSD-window overlay is stale for publication-grade closure; "
            "publication gaps require clean-engine trajectory evidence and full re-gate"
        )
        msd_overlay_state = {}
    resolved_warn_sources: list[str] = []
    remaining_warn_sources = list(original_warn_sources)
    if (
        msd_overlay_state.get("msd_window_overlay_status") == "strict_pass_window_candidate"
        and "msd_warn" in remaining_warn_sources
    ):
        resolved_warn_sources.append("msd_warn")
        remaining_warn_sources = [source for source in remaining_warn_sources if source != "msd_warn"]
        if remaining_warn_sources:
            candidates = infer_diagnoses(row, remaining_warn_sources, clean_state)
            primary = select_primary_diagnosis(candidates)
            secondary = [
                candidate["diagnosis"]
                for candidate in candidates
                if candidate["diagnosis"] != primary["diagnosis"]
            ]
            diagnosis = {
                **diagnosis,
                "diagnosis_source": "postprocess_overlay_regate",
                "rule_verdict": "WARN",
                "warn_sources": remaining_warn_sources,
                "warn_source": ",".join(remaining_warn_sources),
                "primary_audit_diagnosis": primary["diagnosis"],
                "cause_diagnosis": primary["diagnosis"],
                "secondary_audit_diagnoses": secondary,
                "audit_diagnosis": primary["diagnosis"],
                "diagnosis_basis": primary["diagnosis_basis"],
                "recommended_reentry_module": primary["recommended_reentry_module"],
                "recommended_action": primary["recommended_action"],
                "repair_tool": primary["repair_tool"],
                "tool_parameters": primary["tool_parameters"],
                "repair_instruction": {
                    "repair_tool": primary["repair_tool"],
                    "tool_parameters": primary["tool_parameters"],
                    "expected_success_condition": (
                        "run the selected deterministic module, write sidecar overlay evidence, "
                        "and re-enter the rule gate"
                    ),
                },
                "confidence": primary["confidence"],
                "requires_freeze": primary["requires_freeze"],
                "requires_profile_freeze": primary["requires_freeze"],
                "rationale": (
                    "MSD WARN was cleared by the strict sidecar overlay, but other WARN sources "
                    f"remain: {remaining_warn_sources}. Route the next unresolved warning."
                ),
            }
    route_meta = route_from_diagnosis(diagnosis)
    primary_diagnosis = str(diagnosis.get("primary_audit_diagnosis") or diagnosis.get("audit_diagnosis") or "none")

    route = route_meta["repair_route"]
    action = route_meta["repair_action"]
    automation_level = route_meta["automation_level"]
    executor = route_meta["executor"]
    executable_now = False
    rationale = diagnosis["rationale"]
    terminal_policy = "retain_current_release_status"
    regate_verdict = "WARN" if resolved_warn_sources and remaining_warn_sources else ""
    publication_state = "accepted/resolved" if effective_publication_grade else "publication_pending"
    publication_state_reason = (
        "canonical v1 publication_grade=True"
        if publication_grade
        else (
            "full clean release-overlay PASS"
            if clean_release_pass
            else "case has not passed the publication-grade release gate"
        )
    )
    msd_window_repair_scope = ""
    msd_window_case_id = case_id
    msd_window_msd_root = ""

    if primary_diagnosis == "fit_window_or_method_issue" and msd_overlay_state.get(
        "msd_window_overlay_status"
    ) == "strict_pass_window_candidate":
        route = "persistent_msd_window_diagnostic_review"
        action = "review_local_msd_window_diagnostic"
        automation_level = "sidecar_postprocess_diagnostic_only"
        executor = "pelcb_msd_window_repair.py"
        executable_now = False
        rationale = (
            "legacy MSD-window overlay is diagnostic only under the publication-grade MSD policy; "
            "a local fitted window cannot close a case without clean-engine tail-diffusion evidence"
        )
        terminal_policy = "publication_pending until clean nvt_prod reaches a true diffusive tail and the full rule gate passes"
        regate_verdict = "WARN"
    elif primary_diagnosis == "fit_window_or_method_issue" and msd_overlay_state.get(
        "msd_window_overlay_status"
    ) == "c100_relaxed_window_candidate":
        route = "persistent_warn_review"
        action = "review_relaxed_msd_window_overlay"
        automation_level = "sidecar_postprocess_overlay_review"
        executor = "pelcb_msd_window_repair.py"
        executable_now = False
        rationale = (
            "MSD WARN has a relaxed C100-aware window candidate only; keep as persistent WARN "
            "until a stricter rule/profile freeze approves it"
        )
        terminal_policy = "persistent WARN; review relaxed-window evidence instead of promoting automatically"
        regate_verdict = "WARN"
    elif primary_diagnosis in {"missing_md_outputs", "engine_or_protocol_sensitive_result"} and clean_state.get(
        "clean_repair_state"
    ) == "clean_postprocess_closed":
        clean_regate_verdict = clean_regate_state.get("clean_regate_verdict", "")
        clean_release_overlay_verdict = clean_regate_state.get("clean_release_overlay_verdict", "")
        if upper_text(clean_release_overlay_verdict) == "PASS":
            route = "resolved_by_clean_engine_overlay"
            action = "regate_pass_from_clean_engine_overlay"
            automation_level = "sidecar_clean_engine_overlay_resolved"
            executor = "pelcb_clean_overlay_regate.py"
            regate_verdict = "PASS"
            rationale = (
                "clean-engine MD and postprocessing overlay cleared the targeted WARN sources; "
                f"release-overlay verdict is {clean_regate_state.get('clean_release_overlay_verdict', 'unknown')}"
            )
            terminal_policy = (
                "sidecar clean overlay resolves targeted WARN sources; canonical v1 remains unchanged "
                "until an explicit release overlay is approved"
            )
        elif clean_regate_verdict == "WARN" or upper_text(clean_release_overlay_verdict) == "WARN":
            remaining_clean_warns = {
                source.strip()
                for source in clean_regate_state.get("clean_remaining_target_warn_sources", "").split(",")
                if source.strip()
            }
            if not remaining_clean_warns:
                remaining_clean_warns = {
                    source.strip()
                    for source in clean_regate_state.get("clean_warn_sources", "").split(",")
                    if source.strip()
                }
            all_clean_warns = set(remaining_clean_warns) | set(source_list(clean_regate_state.get("clean_warn_sources", "")))
            if "msd_warn" in all_clean_warns and clean_msd_postprocess_required(clean_regate_state):
                route = "repair_postprocess_clean_msd"
                action = "postprocess_clean_msd_recompute"
            elif "msd_warn" in all_clean_warns and clean_msd_tail_extension_required(clean_regate_state):
                route = "repair_md_clean_nvt_extension"
                action = "extend_clean_nvt_prod_from_checkpoint"
            elif "density_warn" in remaining_clean_warns:
                route = "persistent_density_protocol_review"
                action = "review_density_protocol_or_clean_profile_second_check"
            elif (
                "msd_warn" in remaining_clean_warns
                and clean_regate_state.get("clean_msd_window_overlay_status") == "c100_relaxed_window_candidate"
            ):
                route = "persistent_msd_relaxed_window_review"
                action = "review_relaxed_msd_window_after_clean_overlay"
            elif remaining_clean_warns == {"msd_warn"}:
                route = "persistent_msd_window_diagnostic_review"
                action = "review_msd_publication_gate_after_clean_overlay"
            elif clean_fsqt_postprocess_required(clean_regate_state):
                route = "repair_postprocess_fsqt"
                action = "postprocess_fsqt_recompute"
            elif clean_intrachain_postprocess_required(clean_regate_state):
                route = "repair_postprocess_intrachain"
                action = "postprocess_intrachain_recompute"
            else:
                route = "persistent_clean_engine_overlay_warn"
                action = "review_clean_engine_overlay_remaining_warn"
            regate_verdict = "WARN"
            if route == "repair_md_clean_nvt_extension":
                automation_level = "sidecar_md_execution_extension"
                executor = "pelcb_clean_nvt_extension.py"
                executable_now = clean_nvt_extension_eligible(clean_state, clean_regate_state)
                rationale = (
                    "clean-engine MD/postprocessing completed but the full release-overlay gate still has an MSD WARN; "
                    "extend clean nvt_prod from checkpoint and recompute MSD from the longer clean trajectory"
                )
                terminal_policy = "after the extended clean trajectory is collected, run full postprocessing and re-gate"
            elif route == "repair_postprocess_clean_msd":
                automation_level = "sidecar_postprocess_closure"
                executor = "o7_clean_closure.py"
                executable_now = True
                rationale = (
                    "clean-engine overlay contains failed or incomplete MSD postprocessing evidence; rerun the "
                    "deterministic clean MSD/postprocess closure before deciding whether more NVT sampling is needed"
                )
                terminal_policy = "after clean MSD postprocessing, re-enter the clean release-overlay gate"
            elif route == "repair_postprocess_fsqt":
                automation_level = "sidecar_postprocess_closure"
                executor = "o7_clean_closure.py"
                executable_now = True
                rationale = (
                    "clean-engine overlay is complete except for Fs(q,t) evidence; rerun the deterministic "
                    "clean dynamics postprocess suite and then re-gate"
                )
                terminal_policy = "after Fs(q,t) postprocessing, re-enter the clean release-overlay gate"
            elif route == "repair_postprocess_intrachain":
                automation_level = "sidecar_postprocess_closure"
                executor = "o7_clean_closure.py"
                executable_now = True
                rationale = (
                    "clean-engine overlay is complete except for native intrachain-dynamics evidence; rerun "
                    "the deterministic clean intrachain postprocess suite and then re-gate"
                )
                terminal_policy = "after intrachain postprocessing, re-enter the clean release-overlay gate"
            else:
                automation_level = "sidecar_clean_engine_overlay_review"
                executor = "pelcb_clean_overlay_regate.py"
                executable_now = False
                rationale = (
                    "clean-engine overlay was re-gated but targeted WARN sources remain: "
                    f"{clean_regate_state.get('clean_remaining_target_warn_sources', '')}; clean WARN sources: "
                    f"{clean_regate_state.get('clean_warn_sources', '')}"
                )
                terminal_policy = "persistent WARN after one clean-engine repair; retain sidecar evidence for audit/review"
        else:
            route = "clean_engine_overlay_ready_for_regate"
            action = "rerun_rule_gate_on_clean_engine_overlay"
            automation_level = "sidecar_overlay_regate_ready"
            executor = "pelcb_repair_loop.py"
            regate_verdict = "PENDING"
            rationale = (
                "clean-engine MD and full postprocessing outputs are already collected in the sidecar; "
                "do not resubmit MD or postprocess, re-enter the rule gate/overlay comparison next"
            )
            terminal_policy = (
                "sidecar overlay ready for re-gate; canonical v1 remains unchanged until an explicit release overlay is approved"
            )
        if route not in {
            "repair_md_clean_nvt_extension",
            "repair_postprocess_clean_msd",
            "repair_postprocess_fsqt",
            "repair_postprocess_intrachain",
        }:
            executable_now = False
    elif primary_diagnosis == "missing_md_outputs" and clean_state.get("clean_repair_state") == "clean_nvt_done":
        route = "repair_md_clean_engine_ready_for_closure"
        action = "run_o7_clean_postprocess_closure"
        automation_level = "sidecar_postprocess_closure"
        executor = "o7_clean_closure.py"
        executable_now = True
        rationale = "case has an existing clean-engine nvt_prod completion; route to clean postprocess closure and release-overlay review"
        terminal_policy = "compare clean closure outputs to current missing status before any release overlay"
    elif primary_diagnosis in {
        "missing_md_outputs",
        "engine_or_protocol_sensitive_result",
        "fatal_runtime_error",
    } and clean_state.get("clean_repair_state") == "scheduler_repair_required":
        route = "scheduler_repair_required"
        action = "regenerate_clean_engine_submit_without_bad_constraints"
        automation_level = "sidecar_scheduler_repair"
        executor = "o7_clean_supervisor_queue.py"
        executable_now = False
        scheduler_reason = clean_state.get("scheduler_reason") or "live scheduler/probe reported that the clean MD route is not active"
        rationale = (
            f"{scheduler_reason}; regenerate a sidecar clean-engine submit plan without bad scheduler "
            "constraints before re-entering the MD module"
        )
        terminal_policy = (
            "verify clean GROMACS binary/profile, regenerate a constraint-clean submit plan, then rerun MD sidecar; "
            "after nvt_prod completion, run postprocess closure and re-gate"
        )
    elif primary_diagnosis in {
        "missing_md_outputs",
        "engine_or_protocol_sensitive_result",
    } and clean_state.get("clean_repair_state") == "clean_nvt_checkpoint_pending":
        route = "repair_md_clean_engine_safe_retry"
        action = "resume_incomplete_clean_nvt_from_checkpoint"
        automation_level = "sidecar_md_execution_resume"
        executor = "o7_clean_safe_retry.py"
        executable_now = bool(clean_state.get("remote_run_dir"))
        rationale = (
            "clean-engine closure scan found completed upstream stages and incomplete nvt_prod evidence, "
            "but no completion marker; treat this as a checkpoint-resume requirement rather than a live active job"
        )
        terminal_policy = (
            "resume the existing clean run directory through the guarded clean retry tool; after nvt_prod "
            "completion, run clean postprocess closure and re-gate"
        )
    elif primary_diagnosis == "missing_md_outputs" and clean_state.get("clean_repair_state") == "active_clean_queue":
        route = "repair_md_clean_engine_active"
        action = "monitor_existing_o7_clean_full_supervisor"
        automation_level = "sidecar_md_execution_active"
        executor = "o7_clean_supervisor_queue.py"
        executable_now = False
        rationale = "case is already in a clean-engine sidecar run; monitor existing run instead of submitting a duplicate"
        terminal_policy = "after clean nvt_prod completion, run clean postprocess closure and compare new quality mask before any release overlay"
    elif primary_diagnosis == "missing_md_outputs" and clean_state.get("clean_repair_state") == "clean_queue_waiting_for_slots":
        route = "repair_md_clean_engine_waiting_for_slots"
        action = "retry_when_clean_engine_slot_available"
        automation_level = "sidecar_md_execution_queued"
        executor = "o7_clean_supervisor_queue.py"
        executable_now = source_available
        rationale = "case is queued for clean-engine sidecar execution; retry the supervisor queue so newly available GPU slots are filled"
        terminal_policy = "submit when supervisor capacity opens; after clean nvt_prod completion, run clean postprocess closure"
    elif primary_diagnosis == "missing_md_outputs":
        action = "submit_o7_clean_full_supervisor"
        executable_now = source_available
        terminal_policy = "after clean nvt_prod completion, run clean postprocess closure and compare new quality mask before any release overlay"
    elif primary_diagnosis == "fatal_runtime_error" and clean_state.get("clean_repair_state") == "clean_runtime_fatal":
        route = "repair_md_clean_engine_safe_retry"
        action = "retry_failed_clean_nvt_safe_offload"
        automation_level = "sidecar_md_execution_safe_retry"
        executor = "o7_clean_safe_retry.py"
        executable_now = bool(clean_state.get("remote_run_dir"))
        rationale = (
            "clean-engine sidecar reached a runtime fatal state; archive the failed stage and retry from the "
            "existing clean run directory with CPU-only mdrun offload, preserving the same mdp physics"
        )
        terminal_policy = "after safe retry nvt_prod completion, run clean postprocess closure and compare the overlay before any release change"
    elif primary_diagnosis == "fit_window_or_method_issue":
        if not effective_publication_grade:
            if clean_state.get("clean_repair_state") == "clean_postprocess_closed":
                if not clean_regate_available(clean_regate_state):
                    route = "clean_engine_overlay_ready_for_regate"
                    action = "rerun_rule_gate_on_clean_engine_overlay"
                    automation_level = "sidecar_overlay_regate_ready"
                    executor = "pelcb_clean_overlay_regate.py"
                    executable_now = False
                    regate_verdict = "PENDING"
                    rationale = (
                        "clean full postprocessing is available for a publication gap; re-enter the full rule gate "
                        "before any resolved state is allowed"
                    )
                    terminal_policy = "full release-overlay re-gate is required before closure"
                else:
                    clean_warns = set(source_list(clean_regate_state.get("clean_warn_sources", "")))
                    if "msd_warn" in clean_warns and clean_msd_postprocess_required(clean_regate_state):
                        route = "repair_postprocess_clean_msd"
                        action = "postprocess_clean_msd_recompute"
                        automation_level = "sidecar_postprocess_closure"
                        executor = "o7_clean_closure.py"
                        executable_now = True
                        regate_verdict = "WARN"
                        rationale = (
                            "clean MSD evidence failed during deterministic postprocessing; rerun clean MSD/"
                            "postprocess closure before deciding whether a longer clean trajectory is required"
                        )
                        terminal_policy = "after clean MSD postprocessing, re-enter the full release-overlay gate"
                    elif clean_fsqt_postprocess_required(clean_regate_state):
                        route = "repair_postprocess_fsqt"
                        action = "postprocess_fsqt_recompute"
                        automation_level = "sidecar_postprocess_closure"
                        executor = "o7_clean_closure.py"
                        executable_now = True
                        regate_verdict = "WARN"
                        rationale = (
                            "clean-engine overlay is missing Fs(q,t) evidence; rerun the deterministic clean "
                            "dynamics postprocess suite before review"
                        )
                        terminal_policy = "after Fs(q,t) postprocessing, re-enter the full release-overlay gate"
                    elif clean_intrachain_postprocess_required(clean_regate_state):
                        route = "repair_postprocess_intrachain"
                        action = "postprocess_intrachain_recompute"
                        automation_level = "sidecar_postprocess_closure"
                        executor = "o7_clean_closure.py"
                        executable_now = True
                        regate_verdict = "WARN"
                        rationale = (
                            "clean-engine overlay has incomplete native intrachain-dynamics evidence; rerun "
                            "the deterministic clean intrachain postprocess suite before review"
                        )
                        terminal_policy = "after intrachain postprocessing, re-enter the full release-overlay gate"
                    elif "msd_warn" in clean_warns and clean_msd_tail_extension_required(clean_regate_state):
                        route = "repair_md_clean_nvt_extension"
                        action = "extend_clean_nvt_prod_from_checkpoint"
                        automation_level = "sidecar_md_execution_extension"
                        executor = "pelcb_clean_nvt_extension.py"
                        executable_now = clean_nvt_extension_eligible(clean_state, clean_regate_state)
                        regate_verdict = "WARN"
                        rationale = (
                            "clean trajectory still fails the publication-grade MSD gate; extend nvt_prod rather than "
                            "accepting a local MSD-window stopgap"
                        )
                        terminal_policy = "extended clean trajectory must re-enter full postprocessing and the release-overlay gate"
                    else:
                        route = "persistent_clean_engine_overlay_warn"
                        action = "review_clean_engine_overlay_remaining_warn"
                        automation_level = "sidecar_clean_engine_overlay_review"
                        executor = "pelcb_clean_overlay_regate.py"
                        executable_now = False
                        regate_verdict = "WARN"
                        rationale = f"full clean release-overlay still has WARN sources: {sorted(clean_warns)}"
                        terminal_policy = "publication_pending until final audit/review explains persistent WARN"
            elif clean_state.get("clean_repair_state") == "clean_nvt_done":
                route = "repair_md_clean_engine_ready_for_closure"
                action = "run_o7_clean_postprocess_closure"
                automation_level = "sidecar_postprocess_closure"
                executor = "o7_clean_closure.py"
                executable_now = True
                rationale = "clean nvt_prod is complete; run full clean postprocessing and then full release re-gate"
                terminal_policy = "publication-grade closure requires clean postprocessing plus full re-gate"
            elif clean_state.get("clean_repair_state") == "clean_nvt_checkpoint_pending":
                route = "repair_md_clean_engine_safe_retry"
                action = "resume_incomplete_clean_nvt_from_checkpoint"
                automation_level = "sidecar_md_execution_resume"
                executor = "o7_clean_safe_retry.py"
                executable_now = bool(clean_state.get("remote_run_dir"))
                rationale = "publication-grade MSD closure requires the clean trajectory; resume incomplete clean nvt_prod"
                terminal_policy = "after clean nvt_prod completion, run full clean postprocessing and full re-gate"
            elif clean_state.get("clean_repair_state") == "active_clean_queue":
                route = "repair_md_clean_engine_active"
                action = "monitor_existing_o7_clean_full_supervisor"
                automation_level = "sidecar_md_execution_active"
                executor = "o7_clean_supervisor_queue.py"
                executable_now = False
                rationale = "clean-engine publication-grade repair is already running; monitor instead of resubmitting"
                terminal_policy = "after clean nvt_prod completion, run full clean postprocessing and full re-gate"
            elif clean_state.get("clean_repair_state") == "clean_queue_waiting_for_slots":
                route = "repair_md_clean_engine_waiting_for_slots"
                action = "retry_when_clean_engine_slot_available"
                automation_level = "sidecar_md_execution_queued"
                executor = "o7_clean_supervisor_queue.py"
                executable_now = source_available
                rationale = "publication-grade MSD closure is queued for clean-engine execution"
                terminal_policy = "submit when supervisor capacity opens; then full postprocess and re-gate"
            else:
                route = "repair_md_clean_engine"
                action = "submit_o7_clean_full_supervisor"
                automation_level = "sidecar_md_execution"
                executor = "o7_clean_supervisor_queue.py"
                executable_now = source_available
                rationale = (
                    "MSD-window stopgap cannot close a publication-grade gap; run same-input clean-engine MD "
                    "to full nvt_prod and re-gate the clean evidence"
                )
                terminal_policy = "publication-grade closure requires clean nvt_prod, full postprocessing, and full re-gate"
        else:
            route = "no_action"
            action = "none"
            automation_level = "none"
            executor = ""
            executable_now = False
            terminal_policy = "publication-grade case already passed; MSD-window diagnostics are not required"
    elif primary_diagnosis == "packing_void_or_local_density_issue":
        action = "run_packing_void_repair_dry_run"
        executable_now = False
        terminal_policy = "run sidecar packing/void postprocessing overlay; do not rerun MD unless structure or trajectory evidence is missing"
    elif primary_diagnosis == "engine_or_protocol_sensitive_result":
        if clean_state.get("clean_repair_state") == "clean_nvt_done":
            route = "optional_clean_engine_validation_ready_for_closure"
            action = "run_o7_clean_postprocess_closure"
            automation_level = "sidecar_postprocess_closure"
            executor = "o7_clean_closure.py"
            executable_now = True
            rationale = "density WARN already has a clean-engine nvt_prod completion; route to clean postprocess closure instead of submitting a duplicate"
            terminal_policy = "compare clean closure outputs to current density-qualified status before any release overlay"
        elif clean_state.get("clean_repair_state") == "active_clean_queue":
            route = "optional_clean_engine_validation_active"
            action = "monitor_existing_o7_clean_density_validation"
            automation_level = "sidecar_md_execution_active"
            executor = "o7_clean_supervisor_queue.py"
            executable_now = False
            rationale = "density WARN already has an active clean-engine sidecar run; monitor existing run instead of submitting a duplicate"
            terminal_policy = "after clean nvt_prod completion, run clean postprocess closure and compare density overlay before any release change"
        elif clean_state.get("clean_repair_state") == "clean_nvt_checkpoint_pending":
            route = "repair_md_clean_engine_safe_retry"
            action = "resume_incomplete_clean_nvt_from_checkpoint"
            automation_level = "sidecar_md_execution_resume"
            executor = "o7_clean_safe_retry.py"
            executable_now = bool(clean_state.get("remote_run_dir"))
            rationale = (
                "density clean-engine validation has upstream clean stages and incomplete nvt_prod evidence, "
                "but no completion marker; resume the existing sidecar run instead of counting it as active"
            )
            terminal_policy = "after resumed clean nvt_prod completion, run clean postprocess closure and compare density overlay"
        elif clean_state.get("clean_repair_state") == "clean_queue_waiting_for_slots":
            route = "optional_clean_engine_validation_waiting_for_slots"
            action = "retry_when_clean_engine_slot_available"
            automation_level = "sidecar_md_execution_queued"
            executor = "o7_clean_supervisor_queue.py"
            executable_now = source_available
            rationale = "density WARN clean-engine validation is queued; retry the supervisor queue so newly available GPU slots are filled"
            terminal_policy = "submit when supervisor capacity opens; after clean nvt_prod completion, run clean postprocess closure"
        else:
            action = "optional_o7_clean_density_validation"
            executable_now = source_available
            terminal_policy = "do not replace production row unless a separately frozen clean-engine overlay is approved"
    elif primary_diagnosis in {"builder_input_incomplete", "emc_builder_failed"}:
        executable_now = False
        terminal_policy = "rebuild only into a sidecar project, then compare new evidence before any release overlay"
    elif primary_diagnosis in {
        "cross_model_disagreement",
        "low_confidence_audit",
        "insufficient_evidence_for_diagnosis",
        "automatic_review_required",
        "manual_review_required",
        "evidence_conflict_between_observables",
    }:
        executable_now = False
        terminal_policy = (
            "automatically assign REVIEW, archive the evidence packet, and wait for "
            "new deterministic evidence or a registered capability"
        )
    elif primary_diagnosis in {"topology_unsupported", "force_field_assignment_unsupported", "unsupported_by_current_system"}:
        executable_now = False
        terminal_policy = "report unsupported candidate to user; do not submit EMC, MD, or postprocessing"
    elif primary_diagnosis == "none":
        route = "no_action"
        action = "none"
        automation_level = "none"
        executor = ""
        executable_now = False
        if case_status == "completed_quality_passed" and ml_usable:
            rationale = "publication-grade case"

    priority = {
        "repair_md_clean_engine": 1,
        "repair_md_clean_engine_ready_for_closure": 1,
        "repair_md_clean_engine_active": 1,
        "repair_md_clean_engine_waiting_for_slots": 1,
        "repair_md_clean_engine_safe_retry": 1,
        "scheduler_repair_required": 1,
        "repair_emc_rebuild": 1,
        "repair_postprocess_msd_window": 2,
        "repair_md_clean_nvt_extension": 1,
        "clean_engine_overlay_ready_for_regate": 2,
        "repair_postprocess_clean_msd": 2,
        "repair_postprocess_fsqt": 2,
        "persistent_clean_engine_overlay_warn": 6,
        "persistent_msd_relaxed_window_review": 6,
        "persistent_msd_window_diagnostic_review": 6,
        "persistent_density_protocol_review": 6,
        "resolved_by_clean_engine_overlay": 7,
        "resolved_by_postprocess_overlay": 7,
        "persistent_warn_review": 7,
        "repair_postprocess_packing": 2,
        "review_audit_disagreement": 3,
        "optional_clean_engine_validation": 4,
        "optional_clean_engine_validation_waiting_for_slots": 4,
        "no_action": 9,
    }.get(route, 8)
    module_name = str(diagnosis.get("recommended_reentry_module", ""))
    regate_required = module_name in {"emc", "md", "postprocess"} or action in {
        "run_o7_clean_postprocess_closure",
        "rerun_rule_gate_on_clean_engine_overlay",
        "retry_failed_clean_nvt_safe_offload",
    }
    regate_policy = (
        "write sidecar overlay evidence, rerun rule gate, and re-enter audit if WARN persists"
        if regate_required
        else "no automatic re-gate; retain review or unsupported terminal state"
    )
    closed_loop_state, closed_loop_stage = closed_loop_state_for_route(route, action, executable_now)
    runtime_state = runtime_state_for_row(clean_state, route)
    pass90_category = pass90_category_for_row(route, runtime_state, bool(effective_publication_grade))

    return {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "case_id": case_id,
        "case_status": case_status,
        "warning_reason": warning_reason,
        "ml_usable": ml_usable,
        "publication_grade": publication_grade,
        "effective_publication_grade": effective_publication_grade,
        "publication_grade_source": publication_grade_source,
        "publication_state": publication_state,
        "publication_state_reason": publication_state_reason,
        "density_tier_si_o": row.get("density_tier_si_o", ""),
        "rule_equilibration": row.get("rule_equilibration", ""),
        "rule_fit_quality": row.get("rule_fit_quality", ""),
        "rule_anomaly": row.get("rule_anomaly", ""),
        "rule_verdict": diagnosis.get("rule_verdict", ""),
        "diagnosis_source": diagnosis.get("diagnosis_source", "deterministic_fallback"),
        "warn_sources": json.dumps(original_warn_sources, ensure_ascii=False),
        "warn_source": ",".join(original_warn_sources) if original_warn_sources else "none",
        "resolved_warn_sources": json.dumps(resolved_warn_sources, ensure_ascii=False),
        "remaining_warn_sources": json.dumps(remaining_warn_sources, ensure_ascii=False),
        "cause_diagnosis": diagnosis.get("cause_diagnosis", diagnosis.get("primary_audit_diagnosis", "")),
        "primary_audit_diagnosis": diagnosis.get("primary_audit_diagnosis", ""),
        "secondary_audit_diagnoses": json.dumps(diagnosis.get("secondary_audit_diagnoses", []), ensure_ascii=False),
        "audit_diagnosis": diagnosis.get("audit_diagnosis", ""),
        "diagnosis_basis": json.dumps(diagnosis.get("diagnosis_basis", []), ensure_ascii=False),
        "evidence_sufficiency": json.dumps(diagnosis.get("evidence_sufficiency", {}), ensure_ascii=False, sort_keys=True),
        "recommended_reentry_module": diagnosis.get("recommended_reentry_module", ""),
        "recommended_action": diagnosis.get("recommended_action", ""),
        "repair_tool": diagnosis.get("repair_tool", ""),
        "tool_parameters": json.dumps(diagnosis.get("tool_parameters", {}), ensure_ascii=False, sort_keys=True),
        "repair_instruction": json.dumps(diagnosis.get("repair_instruction", {}), ensure_ascii=False, sort_keys=True),
        "modification_plan": json.dumps(diagnosis.get("modification_plan", {}), ensure_ascii=False, sort_keys=True),
        "raw_model_diagnosis": diagnosis.get("raw_model_diagnosis", ""),
        "raw_model_repair_suggestion": diagnosis.get("raw_model_repair_suggestion", ""),
        "open_cause_hypothesis": diagnosis.get("open_cause_hypothesis", ""),
        "llm_tool_call_proposal": json.dumps(diagnosis.get("llm_tool_call_proposal", {}), ensure_ascii=False, sort_keys=True),
        "guard_verdict": diagnosis.get("guard_verdict", ""),
        "guard_rejection_reason": diagnosis.get("guard_rejection_reason", ""),
        "tool_affordance_match": diagnosis.get("tool_affordance_match", ""),
        "max_repair_attempts": diagnosis.get("max_repair_attempts", ""),
        "repair_attempt_index": diagnosis.get("repair_attempt_index", 0),
        "repair_extension_attempt_index": diagnosis.get("repair_extension_attempt_index", 0),
        "regate_required": regate_required,
        "regate_policy": regate_policy,
        "regate_verdict": regate_verdict,
        "closed_loop_state": closed_loop_state,
        "closed_loop_stage": closed_loop_stage,
        "runtime_state": runtime_state,
        "pass90_category": pass90_category,
        "msd_window_overlay_status": msd_overlay_state.get("msd_window_overlay_status", ""),
        "msd_window_overlay_source": msd_overlay_state.get("msd_window_overlay_source", ""),
        "msd_window_overlay_source_tag": msd_overlay_state.get("msd_window_overlay_source_tag", ""),
        "msd_window_overlay_fit_window_ns": msd_overlay_state.get("msd_window_overlay_fit_window_ns", ""),
        "msd_window_overlay_note": msd_overlay_state.get("msd_window_overlay_note", ""),
        "legacy_msd_overlay_stale": legacy_msd_overlay_stale,
        "legacy_msd_overlay_stale_reason": legacy_msd_overlay_stale_reason,
        "msd_window_repair_scope": msd_window_repair_scope,
        "msd_window_case_id": msd_window_case_id,
        "msd_window_msd_root": msd_window_msd_root,
        "diagnosis_confidence": diagnosis.get("confidence", ""),
        "diagnosis_rationale": diagnosis.get("rationale", ""),
        "llm_model_count": diagnosis.get("llm_model_count", 0),
        "llm_selected_by": diagnosis.get("selected_by", ""),
        "llm_model_errors": len(diagnosis.get("model_errors", []) or []),
        "requires_profile_freeze": diagnosis.get(
            "requires_profile_freeze",
            diagnosis.get("requires_human_freeze", ""),
        ),
        "audit_disagreement_count": audit_disagreement_count,
        "a3_locked_eval_member": a3_member,
        "repair_route": route,
        "repair_action": action,
        "automation_level": automation_level,
        "executor": executor,
        "source_input_available": source_available,
        "executable_now": executable_now,
        "clean_repair_state": clean_state.get("clean_repair_state", ""),
        "clean_case_id": clean_state.get("clean_case_id", ""),
        "clean_postprocess_state": clean_state.get("clean_postprocess_state", ""),
        "clean_postprocess_missing": json.dumps(clean_state.get("clean_postprocess_missing", []), ensure_ascii=False),
        "clean_postprocess_paths": json.dumps(clean_state.get("clean_postprocess_paths", {}), ensure_ascii=False),
        "clean_postprocess_mtimes": json.dumps(clean_state.get("clean_postprocess_mtimes", {}), ensure_ascii=False),
        "clean_postprocess_latest_mtime": clean_state.get("clean_postprocess_latest_mtime", ""),
        "clean_msd_postprocess_mtime": clean_state.get("clean_msd_postprocess_mtime", ""),
        "clean_regate_source": clean_regate_state.get("clean_regate_source", ""),
        "clean_regate_verdict": clean_regate_state.get("clean_regate_verdict", ""),
        "clean_release_overlay_verdict": clean_regate_state.get("clean_release_overlay_verdict", ""),
        "clean_resolved_warn_sources": clean_regate_state.get("clean_resolved_warn_sources", ""),
        "clean_remaining_target_warn_sources": clean_regate_state.get("clean_remaining_target_warn_sources", ""),
        "clean_warn_sources": clean_regate_state.get("clean_warn_sources", ""),
        "clean_density_rule_verdict": clean_regate_state.get("clean_density_rule_verdict", ""),
        "clean_density_value_gcc": clean_regate_state.get("clean_density_value_gcc", ""),
        "clean_qc_rule_verdict": clean_regate_state.get("clean_qc_rule_verdict", ""),
        "clean_status_MSD": clean_regate_state.get("clean_status_MSD", ""),
        "clean_msd_publication_grade": clean_regate_state.get("clean_msd_publication_grade", ""),
        "clean_msd_publication_issues": clean_regate_state.get("clean_msd_publication_issues", ""),
        "clean_msd_clean_engine_source": clean_regate_state.get("clean_msd_clean_engine_source", ""),
        "clean_msd_tail_window_ok": clean_regate_state.get("clean_msd_tail_window_ok", ""),
        "clean_msd_fit_R2_publication": clean_regate_state.get("clean_msd_fit_R2_publication", ""),
        "clean_msd_fit_begin_ps": clean_regate_state.get("clean_msd_fit_begin_ps", ""),
        "clean_msd_fit_end_ps": clean_regate_state.get("clean_msd_fit_end_ps", ""),
        "clean_msd_time_last_ps": clean_regate_state.get("clean_msd_time_last_ps", ""),
        "clean_msd_loglog_slope": clean_regate_state.get("clean_msd_loglog_slope", ""),
        "clean_msd_diffusive_regime_ok": clean_regate_state.get("clean_msd_diffusive_regime_ok", ""),
        "clean_msd_regime_warning": clean_regate_state.get("clean_msd_regime_warning", ""),
        "clean_msd_window_sensitive": clean_regate_state.get("clean_msd_window_sensitive", ""),
        "clean_msd_window_D_cv_pct": clean_regate_state.get("clean_msd_window_D_cv_pct", ""),
        "clean_msd_window_overlay_status": clean_regate_state.get("clean_msd_window_overlay_status", ""),
        "clean_msd_window_overlay_source": clean_regate_state.get("clean_msd_window_overlay_source", ""),
        "clean_msd_window_overlay_fit_window_ns": clean_regate_state.get("clean_msd_window_overlay_fit_window_ns", ""),
        "clean_msd_window_overlay_loglog_slope": clean_regate_state.get("clean_msd_window_overlay_loglog_slope", ""),
        "clean_msd_window_overlay_D_m2_s": clean_regate_state.get("clean_msd_window_overlay_D_m2_s", ""),
        "clean_msd_window_overlay_local_D_cv_pct": clean_regate_state.get("clean_msd_window_overlay_local_D_cv_pct", ""),
        "clean_tag": clean_state.get("clean_tag", ""),
        "clean_target_host_label": clean_state.get("clean_target_host_label", ""),
        "clean_target_host": clean_state.get("clean_target_host", ""),
        "clean_status_source": clean_state.get("clean_status_source", ""),
        "clean_remote_run_dir": clean_state.get("remote_run_dir", ""),
        "clean_runtime_pid": clean_state.get("clean_runtime_pid", ""),
        "clean_runtime_node": clean_state.get("clean_runtime_node", ""),
        "clean_runtime_gpu_id": clean_state.get("clean_runtime_gpu_id", ""),
        "clean_runtime_stage": clean_state.get("clean_runtime_stage", ""),
        "clean_fatal_stage": clean_state.get("clean_fatal_stage", ""),
        "clean_fatal_log": clean_state.get("clean_fatal_log", ""),
        "clean_fatal_reason": clean_state.get("clean_fatal_reason", ""),
        "scheduler_status": clean_state.get("scheduler_status", ""),
        "scheduler_reason": clean_state.get("scheduler_reason", ""),
        "scheduler_status_source": clean_state.get("scheduler_status_source", ""),
        "scheduler_repair_action": clean_state.get("scheduler_repair_action", ""),
        "scheduler_previous_clean_repair_state": clean_state.get("scheduler_previous_clean_repair_state", ""),
        "priority": priority,
        "rationale": rationale,
        "terminal_policy": terminal_policy,
    }


def build_plan(
    mask_path: Path,
    source_csv: Path,
    include_optional_validation: bool = False,
    o7_queue_root: Path = DEFAULT_O7_QUEUE_ROOT,
    o7_closure_status: Path = DEFAULT_O7_CLOSURE_STATUS,
    warn_audit_jsonl: Path | None = None,
    msd_overlay_mask: Path | None = None,
    scheduler_status_jsonl: Path | None = None,
    runtime_active_csv: Path | None = DEFAULT_RUNTIME_ACTIVE_CSV,
    attempt_count_since: float | None = None,
) -> list[dict[str, Any]]:
    mask = read_csv(mask_path)
    if mask.empty:
        raise FileNotFoundError(f"quality mask missing or empty: {mask_path}")
    source_ids = source_case_ids(source_csv)
    clean_states = clean_repair_state(o7_queue_root, o7_closure_status)
    clean_states = merge_runtime_active_states(clean_states, read_runtime_active_states(runtime_active_csv))
    clean_states = merge_scheduler_overrides(clean_states, read_scheduler_status_overrides(scheduler_status_jsonl))
    audit_path = warn_audit_jsonl if warn_audit_jsonl is not None else latest_frozen_warn_audit_jsonl()
    diagnosis_overrides = read_diagnosis_overrides(audit_path)
    overlay_paths = [msd_overlay_mask] if msd_overlay_mask is not None else msd_window_overlay_masks()
    msd_overlay_states = read_msd_overlay_history(path for path in overlay_paths if path is not None)
    clean_regate_states = read_clean_regate_states(latest_clean_overlay_regate_csv())
    rows = [
        classify_row(row, source_ids, clean_states, diagnosis_overrides, msd_overlay_states, clean_regate_states)
        for _, row in mask.iterrows()
    ]
    attempt_counts = read_repair_attempt_counts(DEFAULT_OUTPUT_DIR, since=attempt_count_since)
    nvt_extension_attempt_counts = read_repair_attempt_counts(
        DEFAULT_OUTPUT_DIR,
        since=attempt_count_since,
        actions=NVT_EXTENSION_REPAIR_ACTIONS,
    )
    nvt_extension_attempt_times = read_nvt_extension_attempt_times()
    for row in rows:
        case_id = str(row.get("case_id", ""))
        total_attempts = int(attempt_counts.get(case_id, 0))
        extension_attempts = int(nvt_extension_attempt_counts.get(case_id, 0))
        attempts = extension_attempts if row.get("repair_route") == "repair_md_clean_nvt_extension" else total_attempts
        row["repair_attempt_index"] = attempts
        row["repair_extension_attempt_index"] = extension_attempts
        row["max_repair_attempts"] = (
            MSD_EXTENSION_MAX_COMPLETED_ROUNDS
            if row.get("repair_route") == "repair_md_clean_nvt_extension"
            else 2
        )
        closure_or_regate_action = row.get("repair_action") in {
            "run_o7_clean_postprocess_closure",
            "rerun_rule_gate_on_clean_engine_overlay",
        }
        if clean_msd_at_bounded_cap_still_warn(row):
            slope_text = cell_text(row.get("clean_msd_loglog_slope", ""))
            time_last_text = cell_text(row.get("clean_msd_time_last_ps", ""))
            row["repair_route"] = "bounded_attempts_exhausted_review"
            row["repair_action"] = "final_audit_review_after_800ns_msd_regate"
            row["automation_level"] = "automatic_review_packet"
            row["executor"] = "automatic_review_packet"
            row["executable_now"] = False
            row["regate_verdict"] = "WARN"
            row["closed_loop_state"] = "publication_pending_after_msd_extension"
            row["closed_loop_stage"] = "review"
            row["priority"] = 6
            row["publication_state"] = "publication_pending"
            row["publication_state_reason"] = (
                "clean MSD evidence reaches the bounded 800 ns target and remains "
                f"non-publication-grade (time_last_ps={time_last_text or 'missing'}, "
                f"loglog_slope={slope_text or 'missing'}); final diagnosis retained for review"
            )
            row["terminal_policy"] = (
                "once postprocessed clean-MSD evidence reaches about 800 ns and still fails "
                "the strict diffusion gate, the case is retained for review rather than "
                "scheduled for another automatic postprocess or NVT extension"
            )
            row["rationale"] = row["publication_state_reason"]
            row["runtime_state"] = runtime_state_for_row(row, row["repair_route"])
            row["pass90_category"] = pass90_category_for_row(
                row["repair_route"],
                row["runtime_state"],
                truthy(row.get("effective_publication_grade")),
            )
            continue
        if (
            row.get("repair_route") == "repair_md_clean_nvt_extension"
            and extension_attempts >= MSD_EXTENSION_MAX_COMPLETED_ROUNDS
            and clean_msd_postprocess_stale_after_extension(row, nvt_extension_attempt_times)
        ):
            row["repair_route"] = "repair_postprocess_clean_msd"
            row["repair_action"] = "postprocess_clean_msd_recompute"
            row["automation_level"] = "sidecar_postprocess_closure"
            row["executor"] = "o7_clean_closure.py"
            row["executable_now"] = True
            row["regate_verdict"] = "WARN"
            row["closed_loop_state"], row["closed_loop_stage"] = closed_loop_state_for_route(
                row["repair_route"],
                row["repair_action"],
                True,
            )
            row["publication_state"] = "publication_pending"
            row["runtime_state"] = runtime_state_for_row(row, row["repair_route"])
            row["pass90_category"] = pass90_category_for_row(
                row["repair_route"],
                row["runtime_state"],
                truthy(row.get("effective_publication_grade")),
            )
            row["priority"] = 3
            row["rationale"] = (
                "clean NVT extension was launched after the current clean-MSD postprocess output; "
                "recompute clean MSD and re-enter the release-overlay gate before review or another extension"
            )
            row["terminal_policy"] = (
                "after extended clean nvt_prod completion, rerun clean postprocessing and full re-gate; "
                "only a post-extension WARN can be retained as review"
            )
            continue
        if (
            row.get("repair_route") == "repair_md_clean_nvt_extension"
            and extension_attempts >= MSD_EXTENSION_MAX_COMPLETED_ROUNDS
            and cell_text(row.get("clean_repair_state")) == "clean_postprocess_closed"
            and not clean_msd_reached_bounded_extension_target(row)
        ):
            time_last_text = cell_text(row.get("clean_msd_time_last_ps", "")) or "missing"
            row["repair_route"] = "repair_postprocess_clean_msd"
            row["repair_action"] = "postprocess_clean_msd_recompute"
            row["automation_level"] = "sidecar_postprocess_closure"
            row["executor"] = "o7_clean_closure.py"
            row["executable_now"] = True
            row["regate_verdict"] = "WARN"
            row["closed_loop_state"], row["closed_loop_stage"] = closed_loop_state_for_route(
                row["repair_route"],
                row["repair_action"],
                True,
            )
            row["publication_state"] = "publication_pending"
            row["runtime_state"] = runtime_state_for_row(row, row["repair_route"])
            row["pass90_category"] = pass90_category_for_row(
                row["repair_route"],
                row["runtime_state"],
                truthy(row.get("effective_publication_grade")),
            )
            row["priority"] = 3
            row["rationale"] = (
                "an MSD NVT extension attempt exists, but the current clean-MSD re-gate evidence "
                f"only reaches time_last_ps={time_last_text}; collect or recompute clean MSD through "
                "the bounded 800 ns target before review"
            )
            row["terminal_policy"] = (
                "bounded MSD review is only allowed after postprocessed clean-MSD evidence reaches "
                "about 800 ns and still fails the strict diffusion gate"
            )
            continue
        if msd_extension_completed_round_exhausted(row, extension_attempts):
            slope_text = cell_text(row.get("clean_msd_loglog_slope", ""))
            row["repair_route"] = "bounded_attempts_exhausted_review"
            row["repair_action"] = "final_audit_review_after_one_failed_msd_extension"
            row["automation_level"] = "automatic_review_packet"
            row["executor"] = "automatic_review_packet"
            row["executable_now"] = False
            row["regate_verdict"] = "WARN"
            row["closed_loop_state"] = "publication_pending_after_msd_extension"
            row["closed_loop_stage"] = "review"
            row["priority"] = 6
            row["publication_state"] = "publication_pending"
            row["publication_state_reason"] = (
                "clean MSD NVT was extended by one 400 ns round and re-gated, "
                f"but clean_msd_loglog_slope={slope_text or 'missing'} remains below 0.95; "
                "final diagnosis retained for review"
            )
            row["terminal_policy"] = (
                "one bounded MSD extension, typically 400 ns -> 800 ns, without reaching the strict "
                "diffusive slope is treated as bounded closure failure for any material; no second "
                "automatic MSD extension"
            )
            row["rationale"] = row["publication_state_reason"]
            continue
        if (
            attempts >= 2
            and row.get("executable_now")
            and not closure_or_regate_action
            and not bounded_attempt_guard_bypass(row)
            and not truthy(row.get("effective_publication_grade"))
        ):
            row["repair_route"] = "bounded_attempts_exhausted_review"
            row["repair_action"] = "final_audit_review_after_two_failed_rounds"
            row["automation_level"] = "automatic_review_packet"
            row["executor"] = "automatic_review_packet"
            row["executable_now"] = False
            row["regate_verdict"] = "WARN"
            row["closed_loop_state"] = "publication_pending_after_two_rounds"
            row["closed_loop_stage"] = "review"
            row["priority"] = 6
            row["publication_state"] = "publication_pending"
            row["publication_state_reason"] = (
                "maximum two repair rounds reached without publication_grade=True; "
                f"final diagnosis retained: {row.get('primary_audit_diagnosis', '')}"
            )
            row["terminal_policy"] = "bounded two-round repair exhausted; no third automatic attempt"
            row["rationale"] = row["publication_state_reason"]
    if not include_optional_validation:
        rows = [row for row in rows if row["repair_route"] != "optional_clean_engine_validation"]
    rows.sort(key=lambda row: (int(row["priority"]), str(row["case_id"])))
    return rows


def _terminate_process_tree(pid: int) -> None:
    """Best-effort termination of a process and all its descendants.

    Critical for hang-proofing: a launch helper may spawn a detached remote
    ssh/mdrun; on timeout the whole local tree must die rather than leave the
    parent blocked.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        else:
            import signal

            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def run_command(command: list[str], timeout: int | None = 3600) -> dict[str, Any]:
    """Run an external command without ever hanging the daemon.

    Child stdout/stderr go to temp files, never to a PIPE.  A detached
    grandchild (e.g. a backgrounded remote launch that inherits the child's
    handles) therefore cannot keep a pipe open and block this process.
    ``proc.wait(timeout)`` only depends on the direct child; on timeout the
    whole process tree is terminated and the temp files are read (file reads
    never block).  This is the safety net that guarantees a single stuck remote
    launch can never freeze the closed-loop daemon.
    """
    timed_out = False
    popen_kwargs: dict[str, Any] = {
        "cwd": str(SCRIPT_DIR.parents[1]),
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    with tempfile.TemporaryFile(mode="w+b") as out_f, tempfile.TemporaryFile(mode="w+b") as err_f:
        proc = subprocess.Popen(command, stdout=out_f, stderr=err_f, **popen_kwargs)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(proc.pid)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().decode("utf-8", "replace")
        stderr = err_f.read().decode("utf-8", "replace")
    returncode = proc.returncode if proc.returncode is not None else -1
    if timed_out:
        stderr = (stderr + f"\n[run_command] TIMEOUT after {timeout}s; process tree terminated").strip()
    return {
        "command": command,
        "returncode": returncode,
        "ok": returncode == 0 and not timed_out,
        "timed_out": timed_out,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
    }


def scan_oldserver_intel_runtime(timeout_per_node: int = 45) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read-only scan of old-server intel nodes for active clean GROMACS mdrun processes."""
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for node in OLD_CLEAN_INTEL_NODES:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            OLD_CLEAN_HOST,
            f"ssh -o BatchMode=yes -o ConnectTimeout=8 {node} python3 -",
        ]
        try:
            proc = subprocess.run(
                command,
                input=OLD_CLEAN_RUNTIME_SCAN,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_per_node,
                check=False,
            )
        except Exception as exc:
            errors.append({"host_label": "oldserver_intelgpu", "host": OLD_CLEAN_HOST, "node": node, "error": str(exc)})
            continue
        if proc.returncode != 0:
            error_text = proc.stderr.strip() or proc.stdout.strip()
            if "pam_slurm_adopt" in error_text and "no active jobs" in error_text:
                continue
            errors.append(
                {
                    "host_label": "oldserver_intelgpu",
                    "host": OLD_CLEAN_HOST,
                    "node": node,
                    "error": error_text[-1200:],
                }
            )
            continue
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            errors.append(
                {
                    "host_label": "oldserver_intelgpu",
                    "host": OLD_CLEAN_HOST,
                    "node": node,
                    "error": f"invalid json: {exc}; stdout={proc.stdout[:500]!r}",
                }
            )
            continue
        for item in payload.get("processes", []):
            rows.append(
                {
                    "host_label": "oldserver_intelgpu",
                    "host": OLD_CLEAN_HOST,
                    "node": node,
                    "gpu_id": cell_text(item.get("gpu_id", "")),
                    "pid": cell_text(item.get("pid", "")),
                    "etimes": cell_text(item.get("etimes", "")),
                    "pcpu": cell_text(item.get("pcpu", "")),
                    "pmem": cell_text(item.get("pmem", "")),
                    "case_id": cell_text(item.get("case_id", "")),
                    "stage": cell_text(item.get("stage", "")),
                    "ntomp": cell_text(item.get("ntomp", "")),
                    "deffnm": cell_text(item.get("deffnm", "")),
                    "cwd": cell_text(item.get("cwd", "")),
                    "cmd": cell_text(item.get("cmd", "")),
                }
            )
    return rows, errors


def scan_oldserver_intel_slurm_queue(timeout_sec: int = 45) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read-only scan of old-server Slurm queue so pending clean extensions block duplicates."""
    command = (
        "u=$(whoami); "
        "squeue -h -p intel -u \"$u\" -o '%i|%T|%j|%R|%b' 2>/dev/null || true"
    )
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                OLD_CLEAN_HOST,
                f"bash -lc {sh_quote(command)}",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except Exception as exc:
        return [], [{"host_label": "oldserver_intelgpu", "host": OLD_CLEAN_HOST, "node": "slurm", "error": str(exc)}]
    if proc.returncode != 0:
        return [], [
            {
                "host_label": "oldserver_intelgpu",
                "host": OLD_CLEAN_HOST,
                "node": "slurm",
                "error": (proc.stderr.strip() or proc.stdout.strip())[-1200:],
            }
        ]
    seq_map = case_ids_by_sequence()
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, state, job_name, reason, gres = [part.strip() for part in parts[:5]]
        if not job_name.startswith("pelcb_ext_"):
            continue
        seq_match = re.search(r"pelcb_ext_(\d{3})(?:\b|_)", job_name)
        seq = seq_match.group(1) if seq_match else ""
        case_id = seq_map.get(seq, "")
        if not case_id:
            continue
        rows.append(
            {
                "host_label": "oldserver_intelgpu",
                "host": OLD_CLEAN_HOST,
                "node": reason if state.upper() == "RUNNING" and reason.startswith("node") else "slurm",
                "gpu_id": "slurm",
                "pid": job_id,
                "etimes": "",
                "pcpu": "",
                "pmem": "",
                "case_id": case_id,
                "stage": "nvt_prod",
                "ntomp": "",
                "deffnm": "nvt_prod",
                "cwd": "",
                "cmd": f"squeue:{state}:{job_name}:{reason}:{gres}",
            }
        )
    return rows, []


def refresh_runtime_active_processes(args: argparse.Namespace) -> dict[str, Any]:
    """Refresh runtime-active CSV for both 48.122 clean GPUs and 71.52 intel GPUs."""
    active_csv = Path(args.runtime_active_csv)
    dashboard_result = run_command(
        [
            sys.executable,
            str(RUNTIME_DASHBOARD),
            "--config",
            str(args.config),
            "--host",
            "new",
            "--json-only",
        ],
        timeout=240,
    )
    old_rows: list[dict[str, Any]] = []
    old_queue_rows: list[dict[str, Any]] = []
    old_errors: list[dict[str, Any]] = []
    if getattr(args, "runtime_oldserver_probe", True):
        old_rows, old_errors = scan_oldserver_intel_runtime()
        old_queue_rows, old_queue_errors = scan_oldserver_intel_slurm_queue()
        old_errors.extend(old_queue_errors)
    rows: list[dict[str, Any]] = []
    if dashboard_result.get("ok") and active_csv.exists():
        active_df = read_csv(active_csv)
        if not active_df.empty:
            rows.extend(active_df.to_dict("records"))
    rows.extend(old_rows)
    rows.extend(old_queue_rows)

    def runtime_row_priority(row: Mapping[str, Any]) -> int:
        cmd = cell_text(row.get("cmd", "")).lower()
        if not cmd.startswith("squeue:"):
            return 0
        if cmd.startswith("squeue:running"):
            return 1
        return 2

    best_by_case: dict[tuple[str, str, str], tuple[int, int, dict[str, Any]]] = {}
    fallback_rows: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        case_id = cell_text(row.get("case_id", ""))
        host_label = cell_text(row.get("host_label", ""))
        stage = cell_text(row.get("stage", ""))
        if not case_id:
            fallback_rows.append((index, row))
            continue
        key = (host_label, case_id, stage)
        priority = runtime_row_priority(row)
        current = best_by_case.get(key)
        if current is None or priority < current[0]:
            best_by_case[key] = (priority, index, row)

    deduped_with_index = [(index, row) for _, index, row in best_by_case.values()]
    deduped_with_index.extend(fallback_rows)
    deduped = [row for _, row in sorted(deduped_with_index, key=lambda item: item[0])]
    if dashboard_result.get("ok") and not old_errors:
        write_csv(active_csv, deduped, RUNTIME_ACTIVE_FIELDS)
    return {
        "ok": bool(dashboard_result.get("ok") and not old_errors and active_csv.exists()),
        "dashboard_ok": bool(dashboard_result.get("ok")),
        "dashboard_returncode": dashboard_result.get("returncode"),
        "dashboard_stderr": dashboard_result.get("stderr", ""),
        "oldserver_ok": not old_errors,
        "oldserver_active_rows": len(old_rows),
        "oldserver_queue_rows": len(old_queue_rows),
        "oldserver_errors": old_errors,
        "runtime_active_csv": str(active_csv),
        "active_rows_written": len(deduped) if dashboard_result.get("ok") and not old_errors else "",
    }


def preflight_strict_msd_refit(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    """Run clean-MSD refit sidecar before deciding which cases need extension."""
    if not getattr(args, "run_strict_msd_refit", False):
        return []
    executed: list[dict[str, Any]] = []
    refit_result = run_command(
        [
            sys.executable,
            str(STRICT_MSD_REFIT),
            "--classification-csv",
            str(args.msd_refit_classification_csv),
        ],
        timeout=3600,
    )
    executed.append(
        {
            "timestamp": iso_now(),
            "action": "strict_msd_refit_before_extend",
            "case_ids": [],
            "result": refit_result,
        }
    )
    if refit_result.get("ok"):
        executed.append(
            {
                "timestamp": iso_now(),
                "action": "rerun_clean_overlay_regate_after_strict_msd_refit",
                "case_ids": [],
                "result": run_command([sys.executable, str(CLEAN_REGATE)], timeout=1800),
            }
        )
        if getattr(args, "refresh_publication_assets", False):
            executed.append(
                {
                    "timestamp": iso_now(),
                    "action": "refresh_publication_assets_after_strict_msd_refit",
                    "case_ids": [],
                    "result": run_command(
                        [sys.executable, str(PUBLICATION_ASSETS), "--config", str(args.config)],
                        timeout=1200,
                    ),
                }
            )
    write_jsonl(output_dir / "repair_loop_preflight_ledger.jsonl", executed)
    return executed


def execute_routes(rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    if not args.execute:
        return executed
    if args.allowlist_tag != "PELCB_REPAIR_LOOP_EXECUTE":
        raise PermissionError("execution requires --allowlist-tag PELCB_REPAIR_LOOP_EXECUTE")
    if len(str(args.reason or "").strip()) < 12:
        raise ValueError("execution requires --reason with at least 12 characters")
    if not truthy(getattr(args, "runtime_refresh_ok", False)):
        raise PermissionError("execution blocked because runtime refresh did not complete successfully")

    md_cases = [
        row["case_id"]
        for row in rows
        if row["repair_route"] in {"repair_md_clean_engine", "repair_md_clean_engine_waiting_for_slots"}
        and row["executable_now"]
    ][: max(0, int(args.max_execute_md))]
    if md_cases:
        command = [
            sys.executable,
            str(O7_QUEUE),
            "--source-csv",
            str(args.source_csv),
            "--queue-dir",
            str(args.o7_queue_dir),
            "--tag",
            args.tag,
            "--max-submit",
            str(len(md_cases)),
            "--cases",
            *md_cases,
            "--execute",
            "--allow-second-per-gpu",
        ]
        result = run_command(command, timeout=7200)
        entry = {
            "timestamp": iso_now(),
            "action": "submit_o7_clean_full_supervisor",
            "case_ids": md_cases,
            "result": result,
        }
        entry["actual_case_ids"] = submitted_case_ids_from_supervisor_result(entry)
        executed.append(entry)

    safe_retry_cases = [
        row["case_id"]
        for row in rows
        if row["repair_route"] == "repair_md_clean_engine_safe_retry" and row["executable_now"]
    ][: max(0, int(getattr(args, "max_execute_safe_retry", 0)))]
    if safe_retry_cases:
        command = [
            sys.executable,
            str(O7_SAFE_RETRY),
            "--status-csv",
            str(args.o7_closure_status),
            "--tag",
            f"{args.tag}_safe_retry",
            "--cases",
            *safe_retry_cases,
            "--execute",
            "--allowlist-tag",
            "PELCB_CLEAN_SAFE_RETRY",
            "--reason",
            args.reason,
            "--preferred-node",
            args.safe_retry_preferred_node,
        ]
        executed.append(
            {
                "timestamp": iso_now(),
                "action": "retry_failed_clean_nvt_safe_offload",
                "case_ids": safe_retry_cases,
                "result": run_command(command, timeout=1200),
            }
        )

    nvt_extension_cases = [
        row["case_id"]
        for row in rows
        if row["repair_route"] == "repair_md_clean_nvt_extension" and row["executable_now"]
    ][: max(0, int(getattr(args, "max_execute_nvt_extension", 0)))]
    if nvt_extension_cases:
        command = [
            sys.executable,
            str(NVT_EXTENSION),
            "--status-csv",
            str(args.o7_closure_status),
            "--tag",
            f"{args.tag}_nvt_extend",
            "--cases",
            *nvt_extension_cases,
            "--execute",
            "--allowlist-tag",
            "PELCB_CLEAN_NVT_EXTENSION",
            "--reason",
            args.reason,
            "--extend-ns",
            "400.0",
            "--preferred-node",
            str(getattr(args, "nvt_extension_preferred_node", "") or ""),
        ]
        if getattr(args, "nvt_extension_allow_second_per_gpu", False):
            command.append("--allow-second-per-gpu")
        if getattr(args, "enable_oldserver_fallover", False):
            command.extend(
                [
                    "--enable-oldserver-fallover",
                    "--refit-classification-csv",
                    str(args.msd_refit_classification_csv),
                    "--oldserver-max-gpu-jobs",
                    str(args.oldserver_max_gpu_jobs),
                ]
            )
        result = run_command(command, timeout=1200)
        executed.append(
            {
                "timestamp": iso_now(),
                "action": "extend_clean_nvt_prod_from_checkpoint",
                "case_ids": nvt_extension_cases,
                "actual_case_ids": successful_nvt_extension_case_ids(f"{args.tag}_nvt_extend"),
                "result": result,
            }
        )

    density_cases = [
        row["case_id"]
        for row in rows
        if row["repair_route"]
        in {"optional_clean_engine_validation", "optional_clean_engine_validation_waiting_for_slots"}
        and row["executable_now"]
    ][: max(0, int(getattr(args, "max_execute_density", 0)))]
    if density_cases:
        command = [
            sys.executable,
            str(O7_QUEUE),
            "--source-csv",
            str(args.source_csv),
            "--queue-dir",
            str(args.o7_queue_dir),
            "--tag",
            f"{args.tag}_density",
            "--max-submit",
            str(len(density_cases)),
            "--cases",
            *density_cases,
            "--execute",
            "--allow-second-per-gpu",
        ]
        result = run_command(command, timeout=7200)
        entry = {
            "timestamp": iso_now(),
            "action": "submit_o7_clean_density_validation",
            "case_ids": density_cases,
            "result": result,
        }
        entry["actual_case_ids"] = submitted_case_ids_from_supervisor_result(entry)
        executed.append(entry)

    msd_window_cases = [
        row["case_id"]
        for row in rows
        if row["repair_route"] == "repair_postprocess_msd_window" and row["executable_now"]
    ][: max(0, int(getattr(args, "max_execute_msd_window", 0)))]
    if msd_window_cases:
        repair_tag = f"{args.tag}_msd_window"
        repair_command = [
            sys.executable,
            str(MSD_WINDOW_REPAIR),
            "--mask",
            str(args.mask),
            "--output-dir",
            str(args.output_dir),
            "--tag",
            repair_tag,
        ]
        for case_id in msd_window_cases:
            repair_command.extend(["--case-id", case_id])
        repair_result = run_command(repair_command, timeout=7200)
        executed.append(
            {
                "timestamp": iso_now(),
                "action": "run_msd_window_repair",
                "case_ids": msd_window_cases,
                "result": repair_result,
            }
        )
        if repair_result.get("ok") and getattr(args, "build_msd_window_overlay", True):
            candidates = args.output_dir / repair_tag / "msd_window_repair_candidates.csv"
            overlay_command = [
                sys.executable,
                str(MSD_WINDOW_OVERLAY),
                "--mask",
                str(args.mask),
                "--candidates",
                str(candidates),
                "--tag",
                f"{args.tag}_msd_window_overlay",
            ]
            executed.append(
                {
                    "timestamp": iso_now(),
                    "action": "build_msd_window_overlay",
                    "case_ids": msd_window_cases,
                    "result": run_command(overlay_command, timeout=1200),
                }
            )

    clean_msd_window_rows = [
        row
        for row in rows
        if row["repair_route"] == "repair_postprocess_clean_msd_window" and row["executable_now"]
    ][: max(0, int(getattr(args, "max_execute_msd_window", 0)))]
    clean_msd_by_root: dict[str, list[dict[str, Any]]] = {}
    for row in clean_msd_window_rows:
        root = cell_text(row.get("msd_window_msd_root", ""))
        if not root:
            continue
        clean_msd_by_root.setdefault(root, []).append(row)
    for root, group in sorted(clean_msd_by_root.items()):
        repair_tag = f"{args.tag}_clean_msd_window_{Path(root).name}"
        repair_command = [
            sys.executable,
            str(MSD_WINDOW_REPAIR),
            "--mask",
            str(args.mask),
            "--msd-root",
            root,
            "--output-dir",
            str(args.output_dir),
            "--tag",
            repair_tag,
        ]
        case_ids = [cell_text(row.get("msd_window_case_id", "")) for row in group if cell_text(row.get("msd_window_case_id", ""))]
        for clean_case_id in case_ids:
            repair_command.extend(["--case-id", clean_case_id])
        repair_result = run_command(repair_command, timeout=7200)
        executed.append(
            {
                "timestamp": iso_now(),
                "action": "run_clean_msd_window_repair",
                "case_ids": [row["case_id"] for row in group],
                "clean_case_ids": case_ids,
                "msd_root": root,
                "result": repair_result,
            }
        )
    if clean_msd_by_root:
        executed.append(
            {
                "timestamp": iso_now(),
                "action": "rerun_clean_overlay_regate_after_clean_msd_window",
                "case_ids": [row["case_id"] for row in clean_msd_window_rows],
                "result": run_command([sys.executable, str(CLEAN_REGATE)], timeout=1800),
            }
        )

    ready_closure_cases = [
        row["case_id"]
        for row in rows
        if row["repair_route"]
        in {
            "repair_md_clean_engine_ready_for_closure",
            "optional_clean_engine_validation_ready_for_closure",
            "repair_postprocess_clean_msd",
            "repair_postprocess_fsqt",
            "repair_postprocess_intrachain",
        }
        and row["executable_now"]
    ]
    force_postprocess_cases = [
        row["case_id"]
        for row in rows
        if row["repair_route"]
        in {
            "repair_postprocess_clean_msd",
            "repair_postprocess_fsqt",
            "repair_postprocess_intrachain",
        }
        and row["executable_now"]
    ]
    ready_closure_tags = sorted(
        {
            str(row.get("clean_tag", "")).strip()
            for row in rows
            if row["repair_route"]
            in {
            "repair_md_clean_engine_ready_for_closure",
            "optional_clean_engine_validation_ready_for_closure",
            "repair_postprocess_clean_msd",
            "repair_postprocess_fsqt",
            "repair_postprocess_intrachain",
            }
            and row["executable_now"]
            and str(row.get("clean_tag", "")).strip()
        }
    )
    should_run_closure = bool(args.run_closure or (getattr(args, "auto_run_ready_closure", True) and ready_closure_cases))
    if should_run_closure:
        command = [sys.executable, str(O7_CLOSURE), "--queue-dir", str(args.o7_queue_dir)]
        if ready_closure_tags:
            command.extend(["--tags", *ready_closure_tags])
        if ready_closure_cases:
            command.extend(["--cases", *ready_closure_cases])
        if force_postprocess_cases:
            command.extend(["--force-postprocess-cases", *force_postprocess_cases])
        if args.auto_submit_postprocess or ready_closure_cases:
            command.append("--auto-submit-postprocess")
        if getattr(args, "auto_finalize_postprocess", True):
            command.append("--auto-finalize-postprocess")
        if getattr(args, "closure_watch", False):
            command.extend(["--watch", "--interval-sec", str(args.closure_interval_sec)])
            if int(args.closure_max_cycles) > 0:
                command.extend(["--max-cycles", str(args.closure_max_cycles)])
        closure_timeout = None if getattr(args, "closure_watch", False) and int(args.closure_max_cycles) == 0 else 7200
        executed.append(
            {
                "timestamp": iso_now(),
                "action": "run_o7_clean_closure",
                "case_ids": ready_closure_cases,
                "result": run_command(command, timeout=closure_timeout),
            }
        )

    if args.refresh_publication_assets:
        command = [sys.executable, str(PUBLICATION_ASSETS), "--config", str(args.config)]
        executed.append(
            {
                "timestamp": iso_now(),
                "action": "refresh_publication_assets",
                "case_ids": [],
                "result": run_command(command, timeout=1200),
            }
        )

    write_jsonl(output_dir / "repair_loop_execution_ledger.jsonl", executed)
    return executed


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    executed: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    preflight_entries: list[dict[str, Any]] | None = None,
    blocked_entries: list[dict[str, Any]] | None = None,
) -> None:
    route_counts = Counter(row["repair_route"] for row in rows)
    actionable_counts = Counter(row["repair_route"] for row in rows if row["executable_now"])
    preflight_entries = preflight_entries or []
    blocked_entries = blocked_entries or []
    lines = [
        "# PE-LCB100 repair-loop plan",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Mode: `{'execute' if args.execute else 'dry-run'}`",
        f"- Rows in plan: `{len(rows)}`",
        f"- Preflight actions: `{len(preflight_entries)}`",
        f"- Executed state-changing actions: `{len(executed)}`",
        f"- Blocked actions: `{len(blocked_entries)}`",
        "",
        "## Route counts",
        "",
        "| Route | Count | Executable now |",
        "|---|---:|---:|",
    ]
    for route, count in sorted(route_counts.items()):
        lines.append(f"| `{route}` | {count} | {actionable_counts.get(route, 0)} |")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- `incomplete_md` routes to the O7 clean-engine supervisor sidecar when archived source inputs are available.",
            "- Rule gates flag PASS/WARN/FAIL; WARN is a trigger, not a diagnosis.",
            "- Audit diagnosis assigns warning sources, a primary cause class, concrete tool parameters, modification plan, and deterministic re-entry module.",
            "- When a tri-model WARN-diagnosis JSONL is supplied, the repair router uses the selected audit diagnosis but still maps execution through the registered deterministic route table.",
            "- Publication-grade closure is stricter than targeted WARN closure: `accepted/resolved` requires canonical `publication_grade=True` or a full clean release-overlay PASS.",
            "- Legacy MSD-window overlays are treated as stale for publication gaps; clean-engine trajectory evidence is required before closure.",
            "- If a clean release-overlay still contains MSD WARN, only strict clean-MSD refit or bounded clean NVT extension can close it; MSD-window overlays remain diagnostic.",
            "- MSD NVT extension is bounded to one +400 ns round for all materials: after the clean trajectory reaches about 800 ns, a re-gated tail slope below 0.95 is sent to publication_pending/review rather than another automatic extension.",
            "- In PE-LCB100, density WARN with engine/protocol sensitivity routes to the MD module for clean-engine validation without rebuilding EMC.",
            "- Each case is bounded to at most two automatic state-changing repair rounds; unresolved cases remain publication_pending/review with a retained diagnosis.",
            "- Audit disagreement routes to review packets unless a narrower deterministic repair cause is available.",
            "- Builder/topology failure is the only route that re-enters the EMC module.",
            "- No route edits `.mdp` physics, rule thresholds, locked labels, or the PE-LCB100 v1 canonical release in dry-run mode.",
            "",
            "## Top planned cases",
            "",
            "| Case | Route | Action | Executable | Rationale |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in rows[:30]:
        lines.append(
            f"| `{row['case_id']}` | `{row['repair_route']}` | `{row['repair_action']}` | "
            f"{'yes' if row['executable_now'] else 'no'} | {row['rationale']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "pelcb100.json")
    parser.add_argument("--mask", type=Path, default=AGGREGATE_DIR / "ml_quality_mask.csv")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--o7-queue-root", type=Path, default=DEFAULT_O7_QUEUE_ROOT)
    parser.add_argument("--o7-closure-status", type=Path, default=DEFAULT_O7_CLOSURE_STATUS)
    parser.add_argument(
        "--warn-audit-jsonl",
        type=Path,
        default=None,
        help="Optional tri-model WARN diagnosis JSONL from pelcb_warn_audit_llm.py; overrides deterministic fallback diagnosis.",
    )
    parser.add_argument(
        "--msd-overlay-mask",
        type=Path,
        default=None,
        help="Optional sidecar ml_quality_mask_msd_window_overlay_v3.csv. Defaults to the newest overlay under the project overlay directory.",
    )
    parser.add_argument(
        "--scheduler-status-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL/CSV live scheduler override for clean sidecar rows, e.g. BadConstraints or stale active jobs.",
    )
    parser.add_argument(
        "--runtime-active-csv",
        type=Path,
        default=DEFAULT_RUNTIME_ACTIVE_CSV,
        help="Runtime dashboard active-process CSV used to keep live clean MD runs marked active.",
    )
    parser.add_argument(
        "--refresh-runtime-dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh the read-only runtime dashboard before planning. Daemon mode refreshes by default.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="repair_loop_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--include-optional-validation", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--allowlist-tag", default="")
    parser.add_argument("--max-execute-md", type=int, default=0)
    parser.add_argument("--max-execute-safe-retry", type=int, default=0)
    parser.add_argument(
        "--safe-retry-preferred-node",
        default="node-d61b45bd",
        help="Prefer this supervisor-approved node for checkpoint safe retries.",
    )
    parser.add_argument(
        "--nvt-extension-preferred-node",
        default="node-d61b45bd",
        help="Prefer this supervisor-approved node for clean NVT extension launches.",
    )
    parser.add_argument(
        "--nvt-extension-allow-second-per-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the supervisor policy of up to two clean extension jobs on one GPU.",
    )
    parser.add_argument("--max-execute-nvt-extension", type=int, default=0)
    parser.add_argument("--max-execute-density", type=int, default=0)
    parser.add_argument("--max-execute-msd-window", type=int, default=0)
    parser.add_argument("--run-strict-msd-refit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--msd-refit-classification-csv", type=Path, default=DEFAULT_MSD_REFIT_CLASSIFICATION)
    parser.add_argument("--enable-oldserver-fallover", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--oldserver-max-gpu-jobs", type=int, default=32)
    parser.add_argument(
        "--runtime-oldserver-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe private-3f2a50db.invalid runtime/Slurm state during duplicate checks; disable only when oldserver SSH is unavailable and execution is restricted to newserver.",
    )
    parser.add_argument(
        "--attempt-count-since",
        default=os.environ.get("PELCB_REPAIR_ATTEMPT_SINCE", ""),
        help=(
            "Only count state-changing repair attempts at or after this ISO timestamp. "
            "Use this when a stricter closure policy supersedes legacy stopgap ledgers."
        ),
    )
    parser.add_argument(
        "--build-msd-window-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After MSD window repair, build a sidecar ML-table overlay from strict repair candidates.",
    )
    parser.add_argument("--o7-queue-dir", type=Path, default=SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_cuda2024_supervisor_queue")
    parser.add_argument("--run-closure", action="store_true")
    parser.add_argument(
        "--auto-run-ready-closure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically run clean postprocess closure when clean nvt_prod done rows are detected in execute mode.",
    )
    parser.add_argument("--auto-submit-postprocess", action="store_true")
    parser.add_argument(
        "--auto-finalize-postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect completed clean postprocessing, apply sidecar outputs, re-gate, and refresh the repair plan.",
    )
    parser.add_argument("--closure-watch", action="store_true", help="Run o7 clean closure in foreground watch mode.")
    parser.add_argument("--closure-interval-sec", type=int, default=600)
    parser.add_argument("--closure-max-cycles", type=int, default=0)
    parser.add_argument("--refresh-publication-assets", action="store_true")
    parser.add_argument("--daemon", action="store_true", help="Continuously rebuild the repair plan and refill newly available slots.")
    parser.add_argument("--interval-sec", type=int, default=600, help="Seconds between repair-loop daemon cycles.")
    parser.add_argument("--max-cycles", type=int, default=0, help="Maximum daemon cycles; 0 runs until stopped.")
    return parser.parse_args()


def run_cycle(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.output_dir / args.tag
    effective_warn_audit = args.warn_audit_jsonl or latest_frozen_warn_audit_jsonl()
    attempt_count_since = parse_iso_like_timestamp(args.attempt_count_since)
    if args.attempt_count_since and attempt_count_since is None:
        raise ValueError(f"invalid --attempt-count-since timestamp: {args.attempt_count_since}")
    runtime_refresh_result: dict[str, Any] | None = None
    if args.refresh_runtime_dashboard or args.daemon or args.execute:
        runtime_refresh_result = refresh_runtime_active_processes(args)
    runtime_refresh_ok = bool(runtime_refresh_result and runtime_refresh_result.get("ok"))
    setattr(args, "runtime_refresh_ok", runtime_refresh_ok)
    preflight_entries = preflight_strict_msd_refit(args, run_dir)
    rows = build_plan(
        args.mask,
        args.source_csv,
        include_optional_validation=args.include_optional_validation or args.max_execute_density > 0,
        o7_queue_root=args.o7_queue_root,
        o7_closure_status=args.o7_closure_status,
        warn_audit_jsonl=effective_warn_audit,
        msd_overlay_mask=args.msd_overlay_mask,
        scheduler_status_jsonl=args.scheduler_status_jsonl,
        runtime_active_csv=args.runtime_active_csv,
        attempt_count_since=attempt_count_since,
    )
    fields = [
        "schema_version",
        "case_id",
        "case_status",
        "warning_reason",
        "ml_usable",
        "publication_grade",
        "effective_publication_grade",
        "publication_grade_source",
        "publication_state",
        "publication_state_reason",
        "density_tier_si_o",
        "rule_equilibration",
        "rule_fit_quality",
        "rule_anomaly",
        "rule_verdict",
        "diagnosis_source",
        "warn_sources",
        "warn_source",
        "resolved_warn_sources",
        "remaining_warn_sources",
        "cause_diagnosis",
        "primary_audit_diagnosis",
        "secondary_audit_diagnoses",
        "audit_diagnosis",
        "diagnosis_basis",
        "evidence_sufficiency",
        "recommended_reentry_module",
        "recommended_action",
        "repair_tool",
        "tool_parameters",
        "repair_instruction",
        "modification_plan",
        "raw_model_diagnosis",
        "raw_model_repair_suggestion",
        "open_cause_hypothesis",
        "llm_tool_call_proposal",
        "guard_verdict",
        "guard_rejection_reason",
        "tool_affordance_match",
        "max_repair_attempts",
        "repair_attempt_index",
        "repair_extension_attempt_index",
        "regate_required",
        "regate_policy",
        "regate_verdict",
        "closed_loop_state",
        "closed_loop_stage",
        "runtime_state",
        "pass90_category",
        "msd_window_overlay_status",
        "msd_window_overlay_source",
        "msd_window_overlay_source_tag",
        "msd_window_overlay_fit_window_ns",
        "msd_window_overlay_note",
        "legacy_msd_overlay_stale",
        "legacy_msd_overlay_stale_reason",
        "msd_window_repair_scope",
        "msd_window_case_id",
        "msd_window_msd_root",
        "diagnosis_confidence",
        "diagnosis_rationale",
        "llm_model_count",
        "llm_selected_by",
        "llm_model_errors",
        "requires_profile_freeze",
        "audit_disagreement_count",
        "a3_locked_eval_member",
        "repair_route",
        "repair_action",
        "automation_level",
        "executor",
        "source_input_available",
        "executable_now",
        "clean_repair_state",
        "clean_case_id",
        "clean_postprocess_state",
        "clean_postprocess_missing",
        "clean_postprocess_paths",
        "clean_postprocess_mtimes",
        "clean_postprocess_latest_mtime",
        "clean_msd_postprocess_mtime",
        "clean_regate_source",
        "clean_regate_verdict",
        "clean_release_overlay_verdict",
        "clean_resolved_warn_sources",
        "clean_remaining_target_warn_sources",
        "clean_warn_sources",
        "clean_density_rule_verdict",
        "clean_density_value_gcc",
        "clean_qc_rule_verdict",
        "clean_status_MSD",
        "clean_msd_publication_grade",
        "clean_msd_publication_issues",
        "clean_msd_clean_engine_source",
        "clean_msd_tail_window_ok",
        "clean_msd_fit_R2_publication",
        "clean_msd_fit_begin_ps",
        "clean_msd_fit_end_ps",
        "clean_msd_time_last_ps",
        "clean_msd_loglog_slope",
        "clean_msd_diffusive_regime_ok",
        "clean_msd_regime_warning",
        "clean_msd_window_sensitive",
        "clean_msd_window_D_cv_pct",
        "clean_msd_window_overlay_status",
        "clean_msd_window_overlay_source",
        "clean_msd_window_overlay_fit_window_ns",
        "clean_msd_window_overlay_loglog_slope",
        "clean_msd_window_overlay_D_m2_s",
        "clean_msd_window_overlay_local_D_cv_pct",
        "clean_tag",
        "clean_target_host_label",
        "clean_target_host",
        "clean_status_source",
        "clean_remote_run_dir",
        "clean_runtime_pid",
        "clean_runtime_node",
        "clean_runtime_gpu_id",
        "clean_runtime_stage",
        "clean_fatal_stage",
        "clean_fatal_log",
        "clean_fatal_reason",
        "scheduler_status",
        "scheduler_reason",
        "scheduler_status_source",
        "scheduler_repair_action",
        "scheduler_previous_clean_repair_state",
        "priority",
        "rationale",
        "terminal_policy",
    ]
    write_csv(run_dir / "repair_loop_plan.csv", rows, fields)
    write_jsonl(run_dir / "repair_loop_plan.jsonl", rows)
    automatic_review_packets = write_automatic_review_packets(
        run_dir,
        rows,
        warn_audit_jsonl=effective_warn_audit,
    )
    execute_blocked_by_runtime = bool(args.execute and not runtime_refresh_ok)
    blocked_entries: list[dict[str, Any]] = []
    if execute_blocked_by_runtime:
        blocked_entries = [
            {
                "timestamp": iso_now(),
                "action": "execute_blocked_runtime_refresh_failed",
                "case_ids": [],
                "result": runtime_refresh_result or {"ok": False, "error": "runtime refresh was not attempted"},
            }
        ]
        write_jsonl(run_dir / "repair_loop_blocked_ledger.jsonl", blocked_entries)
        executed = []
    else:
        executed = execute_routes(rows, args, run_dir)
    execution_ledger = run_dir / "repair_loop_execution_ledger.jsonl"
    preflight_ledger = run_dir / "repair_loop_preflight_ledger.jsonl"
    blocked_ledger = run_dir / "repair_loop_blocked_ledger.jsonl"
    summary = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "created_at": iso_now(),
        "mode": "execute-blocked-runtime-refresh" if execute_blocked_by_runtime else ("execute" if args.execute else "dry-run"),
        "plan_csv": str(run_dir / "repair_loop_plan.csv"),
        "plan_jsonl": str(run_dir / "repair_loop_plan.jsonl"),
        "automatic_review_packet_index": str(run_dir / "automatic_review_packet_index.jsonl"),
        "automatic_review_packet_count": len(automatic_review_packets),
        "report_md": str(run_dir / "repair_loop_report.md"),
        "route_counts": dict(Counter(row["repair_route"] for row in rows)),
        "executable_counts": dict(Counter(row["repair_route"] for row in rows if row["executable_now"])),
        "preflight_actions": len(preflight_entries),
        "executed_actions": len(executed),
        "blocked_actions": len(blocked_entries),
        "preflight_ledger": str(preflight_ledger) if preflight_entries and preflight_ledger.exists() else "",
        "executed_ledger": str(execution_ledger) if executed and execution_ledger.exists() else "",
        "blocked_ledger": str(blocked_ledger) if blocked_entries and blocked_ledger.exists() else "",
        "warn_audit_jsonl": str(effective_warn_audit) if effective_warn_audit else "",
        "scheduler_status_jsonl": str(args.scheduler_status_jsonl) if args.scheduler_status_jsonl else "",
        "runtime_active_csv": str(args.runtime_active_csv),
        "runtime_refresh_ok": runtime_refresh_ok,
        "runtime_refresh_result": runtime_refresh_result or {},
        "attempt_count_since": args.attempt_count_since,
    }
    write_json(run_dir / "repair_loop_summary.json", summary)
    write_json(args.output_dir / "repair_loop_latest.json", summary)
    write_report(
        run_dir / "repair_loop_report.md",
        rows,
        executed,
        args,
        preflight_entries=preflight_entries,
        blocked_entries=blocked_entries,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def daemon_cycle_tag(base_tag: str, cycle_index: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_tag}_cycle{cycle_index:04d}_{stamp}"


def main() -> int:
    args = parse_args()
    if not getattr(args, "daemon", False):
        run_cycle(args)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = acquire_daemon_lock(args.output_dir)
    cycle_index = 1
    try:
        while True:
            cycle_args = copy.copy(args)
            cycle_args.tag = daemon_cycle_tag(str(args.tag), cycle_index)
            write_heartbeat(args.output_dir, "cycle_start", cycle=cycle_index, tag=cycle_args.tag)
            try:
                summary = run_cycle(cycle_args)
                write_heartbeat(
                    args.output_dir,
                    "cycle_done",
                    cycle=cycle_index,
                    tag=cycle_args.tag,
                    executed_actions=summary.get("executed_actions", 0),
                    route_counts=summary.get("route_counts", {}),
                )
            except Exception as exc:
                error = {
                    "timestamp": iso_now(),
                    "cycle": cycle_index,
                    "tag": cycle_args.tag,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(args.output_dir / "repair_loop_last_error.json", error)
                write_heartbeat(args.output_dir, "cycle_error", cycle=cycle_index, tag=cycle_args.tag, error=str(exc))
                print(json.dumps(error, indent=2, ensure_ascii=False), file=sys.stderr, flush=True)
            if int(getattr(args, "max_cycles", 0)) > 0 and cycle_index >= int(args.max_cycles):
                return 0
            cycle_index += 1
            time.sleep(max(1, int(getattr(args, "interval_sec", 600))))
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
