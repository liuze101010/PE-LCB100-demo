#!/usr/bin/env python3
"""Submit PE-LCB100 same-case validation controls on the new server.

This queue is intentionally separate from the main PE-LCB100 production
manifest.  It reruns selected cases that completed on ``newnew`` from scratch
on ``new`` using a distinct control case id (``lcb_pectrl_*``).  The distinct
id prevents the runtime dashboard from marking the original PE case as a
duplicate while the control is running.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import lcb300_supervisor as supervisor
from lcb_project_config import load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUEUE_DIR = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "new_from_newnew_same_case_controls_20260518"
DEFAULT_PLAN = DEFAULT_QUEUE_DIR / "new_from_newnew_same_case_control_plan.csv"
DEFAULT_REMOTE_ROOT = "${PELCB100_REMOTE_HOME}/pelcb100_validation/new_from_newnew_same_case_controls_20260518"


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
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


def control_case_id(source_case_id: str) -> str:
    if source_case_id.startswith("lcb_pe_"):
        return "lcb_pectrl_" + source_case_id.removeprefix("lcb_pe_")
    return "lcb_pectrl_" + source_case_id


def setup_supervisor_globals(config_path: Path) -> Any:
    config = load_lcb_config(config_path)
    supervisor.CONFIG_PATH = config_path
    supervisor.PROJECT_ID = config.project_id
    supervisor.OUTPUT_DIR = config.supervisor_output_dir
    supervisor.RUNTIME_DIR = config.runtime_output_dir
    supervisor.UPSTREAM_DIR = config.emc_output_dir
    supervisor.BRIDGE_DIRS = [config.default_emc_bridge_root, *config.bridge_roots]
    supervisor.NEW_SERVER = config.new_server
    supervisor.NEWNEW_SERVER = config.newnew_server
    supervisor.OLD_SERVER = config.old_server
    supervisor.NEW_REMOTE_ROOT = (config.remote_md_roots.get("new") or [supervisor.NEW_REMOTE_ROOT])[0]
    supervisor.NEWNEW_REMOTE_ROOT = (config.remote_md_roots.get("newnew") or [supervisor.NEWNEW_REMOTE_ROOT])[0]
    supervisor.NEW_GMX = config.new_gmx
    supervisor.NEWNEW_GMX = config.newnew_gmx
    supervisor.DISABLED_NEW_NODES = set(config.disabled_new_nodes)
    supervisor.DISABLED_MD_HOSTS = set(config.disabled_md_hosts)
    return config


def prepare_control_bridge(source_bridge_root: Path, queue_dir: Path, source_case_id: str) -> tuple[str, Path]:
    source_dir = source_bridge_root / source_case_id
    if not source_dir.exists():
        raise FileNotFoundError(f"source bridge case not found: {source_dir}")
    ctrl_id = control_case_id(source_case_id)
    bridge_root = queue_dir / "bridge_ready"
    dest_dir = bridge_root / ctrl_id
    if dest_dir.exists():
        return ctrl_id, bridge_root
    shutil.copytree(source_dir, dest_dir)
    case_json = dest_dir / "case.json"
    payload: dict[str, Any] = {}
    if case_json.exists():
        payload = json.loads(case_json.read_text(encoding="utf-8"))
    payload["case_id"] = ctrl_id
    payload["source_case_id"] = source_case_id
    payload["control_kind"] = "new_from_newnew_same_case_from_scratch"
    payload["control_created_at"] = iso_now()
    case_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ctrl_id, bridge_root


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def active_control_count(active: pd.DataFrame) -> int:
    if active.empty or "case_id" not in active.columns:
        return 0
    return int(active["case_id"].astype(str).str.startswith("lcb_pectrl_").sum())


def planned_active_control_count(rows: list[dict[str, Any]]) -> int:
    active_status = {"SUBMITTED", "RUNNING"}
    count = 0
    for row in rows:
        status = str(row.get("status", "")).upper()
        if status in active_status and str(row.get("control_case_id", "")).strip():
            count += 1
    return count


def main_idle(runtime: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    backlog = runtime.get("backlog", pd.DataFrame())
    if not backlog.empty:
        return False, f"main backlog has {len(backlog)} row(s)"
    duplicates = runtime.get("duplicates", pd.DataFrame())
    if not duplicates.empty:
        return False, f"duplicate gate has {len(duplicates)} row(s)"
    return True, "main backlog empty and duplicate gate clear"


def active_control_slot_keys(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        status = str(row.get("status", "")).upper()
        if status not in {"SUBMITTED", "RUNNING"}:
            continue
        node = str(row.get("control_node", "") or "").strip()
        gpu = str(row.get("control_gpu", "") or "").strip()
        if node and gpu:
            keys.add(supervisor.slot_key("new", node, gpu))
    return keys


def choose_new_slots(
    runtime: dict[str, pd.DataFrame],
    supervisor_state: dict[str, Any],
    plan_rows: list[dict[str, Any]],
    allow_second_per_gpu: bool = False,
) -> list[dict[str, Any]]:
    quarantine = supervisor.active_slot_quarantine(supervisor_state)
    blocked = set(quarantine)
    if not allow_second_per_gpu:
        blocked |= active_control_slot_keys(plan_rows)
    slots = supervisor.compute_slots(
        runtime["gpus"],
        runtime["active"],
        blocked_slots=blocked,
    )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slot in slots:
        if str(slot.get("host_label", "")) != "new":
            continue
        key = supervisor.slot_key("new", str(slot.get("node", "")), str(slot.get("gpu", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(slot)
    return unique


def update_plan_status(plan_path: Path, source_case_id: str, updates: dict[str, Any]) -> None:
    rows = read_csv_rows(plan_path)
    fields = list(rows[0].keys()) if rows else []
    for key in updates:
        if key not in fields:
            fields.append(key)
    for row in rows:
        if row.get("case_id") == source_case_id:
            row.update(updates)
    write_csv_rows(plan_path, rows, fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "pelcb100.json")
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-submit", type=int, default=2)
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument(
        "--allow-second-per-gpu",
        action="store_true",
        help="Allow validation controls to use the second policy slot on an already occupied new-server GPU.",
    )
    parser.add_argument("--only-when-main-idle", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    config = setup_supervisor_globals(args.config)
    decision_path = args.queue_dir / "control_queue_last_decision.json"
    state_path = args.queue_dir / "control_queue_state.json"
    state = load_state(state_path)
    state.setdefault("submitted", {})

    dashboard = supervisor.refresh_dashboard()
    runtime = supervisor.load_runtime()
    idle_ok, idle_reason = main_idle(runtime)
    rows = read_csv_rows(args.plan)
    controls_active_dashboard = active_control_count(runtime.get("active", pd.DataFrame()))
    controls_active_plan = planned_active_control_count(rows)
    controls_active = max(controls_active_dashboard, controls_active_plan)
    submitted_sources = set((state.get("submitted") or {}).keys())
    completed_or_running = {
        row.get("case_id")
        for row in rows
        if str(row.get("status", "")).upper() in {"SUBMITTED", "RUNNING", "DONE", "SKIPPED"}
    }

    decision: dict[str, Any] = {
        "timestamp": iso_now(),
        "execute": bool(args.execute),
        "dashboard_ok": dashboard.get("ok"),
        "main_idle": idle_ok,
        "main_idle_reason": idle_reason,
        "active_controls": controls_active,
        "active_controls_dashboard": controls_active_dashboard,
        "active_controls_plan": controls_active_plan,
        "max_concurrent": args.max_concurrent,
        "submitted": [],
        "skipped": [],
    }
    if not dashboard.get("ok"):
        decision["action"] = "blocked_dashboard"
        decision["stderr"] = dashboard.get("stderr", "")
        write_json(decision_path, decision)
        return 1
    if args.only_when_main_idle and not idle_ok:
        decision["action"] = "blocked_main_not_idle"
        write_json(decision_path, decision)
        return 0
    if controls_active >= args.max_concurrent:
        decision["action"] = "blocked_control_concurrency"
        write_json(decision_path, decision)
        return 0

    slots = choose_new_slots(
        runtime,
        load_state(supervisor.OUTPUT_DIR / "supervisor_state.json"),
        rows,
        allow_second_per_gpu=bool(args.allow_second_per_gpu),
    )
    decision["available_new_slots"] = slots
    if not slots:
        decision["action"] = "blocked_no_new_slot"
        write_json(decision_path, decision)
        return 0

    available_submit = min(args.max_submit, args.max_concurrent - controls_active, len(slots))
    if available_submit <= 0:
        decision["action"] = "blocked_no_submit_capacity"
        write_json(decision_path, decision)
        return 0

    package_root = args.queue_dir / "md_packages" / datetime.now().strftime("%Y%m%d_%H%M%S")
    submitted_count = 0
    for row in rows:
        source_case_id = str(row.get("case_id", "")).strip()
        if not source_case_id:
            continue
        if source_case_id in submitted_sources or source_case_id in completed_or_running:
            continue
        if submitted_count >= available_submit:
            break
        slot = slots[submitted_count]
        ctrl_id, bridge_root = prepare_control_bridge(config.default_emc_bridge_root, args.queue_dir, source_case_id)
        plan = {
            "case_id": ctrl_id,
            "source_case_id": source_case_id,
            "host_label": "new",
            "host": config.new_server,
            "node": slot["node"],
            "gpu": slot["gpu"],
            "threads": slot["threads"],
            "remote_root": DEFAULT_REMOTE_ROOT,
            "module_preset": slot["module_preset"],
            "gmx_bin": config.new_gmx,
            "launch_via_node": slot["launch_via_node"],
            "bridge_dir": str(bridge_root),
            "action": "launch_control_md",
        }
        if args.execute:
            # Keep the tag short.  The run directory already includes ctrl_id;
            # repeating it here can exceed the legacy Windows path limit during
            # local package assembly.
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_tag = f"newctrl_{stamp}_{submitted_count + 1:02d}"
            run_dir = supervisor.generate_package(ctrl_id, plan, package_root, run_tag)
            result = supervisor.deploy_and_launch(plan, run_dir, f"newctrl_{stamp}_{submitted_count + 1:02d}")
            state["submitted"][source_case_id] = {
                "control_case_id": ctrl_id,
                "submitted_at": iso_now(),
                "node": slot["node"],
                "gpu": slot["gpu"],
                "remote_case_dir": result.get("remote_case_dir", ""),
                "source_case_id": source_case_id,
            }
            update_plan_status(
                args.plan,
                source_case_id,
                {
                    "status": "SUBMITTED",
                    "control_case_id": ctrl_id,
                    "submitted_at": state["submitted"][source_case_id]["submitted_at"],
                    "control_node": slot["node"],
                    "control_gpu": slot["gpu"],
                    "control_remote_case_dir": result.get("remote_case_dir", ""),
                },
            )
            decision["submitted"].append(state["submitted"][source_case_id])
        else:
            decision["submitted"].append({"dry_run": True, **plan})
        submitted_count += 1

    if submitted_count:
        decision["action"] = "submitted" if args.execute else "dry_run_plan"
    else:
        decision["action"] = "nothing_to_submit"
    write_json(state_path, state)
    write_json(decision_path, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
