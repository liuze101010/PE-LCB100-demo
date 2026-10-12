#!/usr/bin/env python3
"""Stop redundant active clean-MD runs after publication-grade acceptance.

The release decision and the runtime process are deliberately joined only at
this guarded boundary.  A process is eligible to stop when the latest repair
plan reports all three of the following for the same case:

* effective publication-grade acceptance,
* a PASS release-overlay verdict, and
* no remaining repair action.

Execution verifies the live PID, working directory and mdrun command before
sending SIGINT.  The checkpoint and the evidence used for acceptance are then
recorded in a content-hashed freeze manifest.  Later evidence can replace this
snapshot only through an explicit re-gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from lcb_project_config import ClusterProfile, load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "pelcb100.json"
DEFAULT_RUNTIME_DIR = SCRIPT_DIR / "output" / "pelcb100_runtime_dashboard"
DEFAULT_REPAIR_DIR = SCRIPT_DIR / "output" / "pelcb100_repair_loop"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "pelcb100_pass_stop"
ALLOWLIST_TAG = "PELCB_STOP_PASSED_ACTIVE"


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sh_quote(text: str) -> str:
    return "'" + str(text).replace("'", "'\"'\"'") + "'"


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "pass"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)


def ssh_via_node(host: str, node: str, inner: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    node_command = f"bash -lc {sh_quote(inner)}"
    outer = (
        "ssh -o BatchMode=yes -o ConnectTimeout=20 "
        f"{sh_quote(node)} {sh_quote(node_command)}"
    )
    return run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, outer],
        timeout=timeout,
    )


def latest_repair_plan(root: Path) -> Path | None:
    candidates = [path for path in root.glob("*/repair_loop_plan.csv") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def is_release_stop_eligible(row: dict[str, str]) -> bool:
    return (
        truthy(row.get("effective_publication_grade"))
        and str(row.get("clean_release_overlay_verdict", "")).strip().upper() == "PASS"
        and str(row.get("repair_route", "")).strip() == "no_action"
    )


def runtime_cwd_matches_release_dir(runtime_cwd: str, release_dir: str) -> bool:
    runtime = runtime_cwd.rstrip("/")
    release = release_dir.rstrip("/")
    return runtime == release or runtime == release + "/nvt_prod"


def plan_pass_stops(
    repair_rows: list[dict[str, str]],
    active_rows: list[dict[str, str]],
    profile: ClusterProfile,
) -> list[dict[str, Any]]:
    """Join strict release decisions to unique live clean-MD processes."""
    eligible: dict[str, list[dict[str, str]]] = {}
    for row in repair_rows:
        if is_release_stop_eligible(row):
            eligible.setdefault(str(row.get("case_id", "")).strip(), []).append(row)

    active_by_case: dict[str, list[dict[str, str]]] = {}
    for row in active_rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id in eligible:
            active_by_case.setdefault(case_id, []).append(row)

    plans: list[dict[str, Any]] = []
    for case_id in sorted(active_by_case):
        decision_rows = eligible.get(case_id, [])
        runtime_rows = active_by_case[case_id]
        errors: list[str] = []
        if len(decision_rows) != 1:
            errors.append(f"expected_one_release_decision_found_{len(decision_rows)}")
        if len(runtime_rows) != 1:
            errors.append(f"expected_one_active_process_found_{len(runtime_rows)}")
        decision = decision_rows[0] if decision_rows else {"case_id": case_id}
        runtime = runtime_rows[0] if runtime_rows else {"case_id": case_id}
        pid = str(runtime.get("pid", "")).strip()
        cwd = str(runtime.get("cwd", "")).strip()
        command = str(runtime.get("cmd", "")).strip()
        if str(runtime.get("stage", "")).strip() != "nvt_prod":
            errors.append("active_stage_is_not_nvt_prod")
        if str(runtime.get("host_label", "")).strip() != profile.host_label:
            errors.append("active_host_profile_mismatch")
        if not pid.isdigit():
            errors.append("active_pid_is_not_numeric")
        if not cwd:
            errors.append("active_cwd_missing")
        if "mdrun" not in command:
            errors.append("active_command_is_not_mdrun")
        if profile.allowed_nodes and str(runtime.get("node", "")) not in profile.allowed_nodes:
            errors.append("active_node_not_allowed")
        if str(runtime.get("node", "")) in profile.disallowed_nodes:
            errors.append("active_node_disallowed")

        expected_pid = str(decision.get("clean_runtime_pid", "")).strip()
        expected_cwd = str(decision.get("clean_remote_run_dir", "")).strip()
        if expected_pid and expected_pid != pid:
            errors.append("release_plan_runtime_pid_mismatch")
        if expected_cwd and not runtime_cwd_matches_release_dir(cwd, expected_cwd):
            errors.append("release_plan_runtime_cwd_mismatch")

        plans.append(
            {
                "case_id": case_id,
                "node": runtime.get("node", ""),
                "gpu_id": runtime.get("gpu_id", ""),
                "pid": pid,
                "cwd": cwd,
                "command": command,
                "status": "READY" if not errors else "BLOCKED",
                "errors": errors,
                "release_decision": decision,
            }
        )
    return plans


def checkpoint_metadata(host: str, node: str, cwd: str) -> dict[str, Any]:
    inner = (
        f"cd {sh_quote(cwd)}; test -s nvt_prod.cpt; "
        "stat -c 'size=%s mtime=%Y' nvt_prod.cpt; "
        "latest=$(ls -t nvt_prod*.log 2>/dev/null | head -n 1); "
        "printf 'log=%s\\n' \"$latest\""
    )
    proc = ssh_via_node(host, node, inner, timeout=60)
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def stop_passed_process(
    host: str,
    plan: dict[str, Any],
    *,
    wait_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Stop one exact mdrun process and require a non-empty checkpoint."""
    pid = str(plan["pid"])
    if not pid.isdigit():
        raise ValueError("PID must be numeric")
    cwd = str(plan["cwd"])
    loops = max(1, int(wait_seconds / 2))
    inner = (
        "set -euo pipefail; "
        f"if [ ! -d /proc/{pid} ]; then cd {sh_quote(cwd)}; "
        "test -s nvt_prod.cpt; echo already_stopped_checkpoint_ready; exit 0; fi; "
        f"test \"$(readlink /proc/{pid}/cwd 2>/dev/null || true)\" = {sh_quote(cwd)}; "
        f"cmd=$(tr '\\0' ' ' < /proc/{pid}/cmdline 2>/dev/null || true); "
        "case \"$cmd\" in *mdrun*) ;; *) echo guarded_pid_is_not_mdrun; exit 23;; esac; "
        "count=0; for proc in /proc/[0-9]*; do "
        "[ -r \"$proc/cmdline\" ] || continue; "
        "pcwd=$(readlink \"$proc/cwd\" 2>/dev/null || true); "
        f"[ \"$pcwd\" = {sh_quote(cwd)} ] || continue; "
        "pcmd=$(tr '\\0' ' ' < \"$proc/cmdline\" 2>/dev/null || true); "
        "case \"$pcmd\" in *mdrun*) count=$((count+1));; esac; done; "
        "[ \"$count\" -eq 1 ] || { echo guarded_mdrun_count=$count; exit 25; }; "
        f"kill -INT {pid}; "
        f"for i in $(seq 1 {loops}); do "
        f"if ! kill -0 {pid} >/dev/null 2>&1; then cd {sh_quote(cwd)}; "
        "test -s nvt_prod.cpt || exit 24; "
        "stat -c 'stopped_checkpoint_size=%s stopped_checkpoint_mtime=%Y' nvt_prod.cpt; "
        "exit 0; fi; sleep 2; done; echo guarded_pid_still_running; exit 21"
    )
    return ssh_via_node(host, str(plan["node"]), inner, timeout=wait_seconds + 45)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_files(decision: dict[str, Any]) -> list[Path]:
    values: list[str] = []
    source = str(decision.get("clean_regate_source", "")).strip()
    if source:
        values.append(source)
    try:
        paths = json.loads(str(decision.get("clean_postprocess_paths", "") or "{}"))
    except json.JSONDecodeError:
        paths = {}
    if isinstance(paths, dict):
        values.extend(str(value) for value in paths.values() if str(value).strip())
    window_source = str(decision.get("clean_msd_window_overlay_source", "")).strip()
    if window_source:
        values.append(window_source)
    return [Path(value) for value in dict.fromkeys(values)]


def freeze_evidence(
    run_dir: Path,
    plan: dict[str, Any],
    checkpoint: dict[str, Any],
    repair_plan: Path,
) -> Path:
    decision = dict(plan["release_decision"])
    files: list[dict[str, Any]] = []
    for path in evidence_files(decision):
        record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            record.update(size_bytes=path.stat().st_size, sha256=sha256_file(path))
        files.append(record)
    manifest = {
        "schema_version": "pelcb_publication_evidence_freeze.v1",
        "frozen_at": iso_now(),
        "case_id": plan["case_id"],
        "decision_source": str(repair_plan),
        "decision": {
            "effective_publication_grade": decision.get("effective_publication_grade", ""),
            "clean_release_overlay_verdict": decision.get("clean_release_overlay_verdict", ""),
            "repair_route": decision.get("repair_route", ""),
            "publication_grade_source": decision.get("publication_grade_source", ""),
            "clean_regate_source": decision.get("clean_regate_source", ""),
        },
        "runtime": {
            "node": plan["node"],
            "gpu_id": plan["gpu_id"],
            "pid": plan["pid"],
            "cwd": plan["cwd"],
        },
        "checkpoint_after_stop": checkpoint,
        "evidence_files": files,
        "policy": (
            "This manifest fixes the evidence version supporting publication-grade PASS. "
            "Later trajectory or analysis files require an explicit re-gate before replacing it."
        ),
    }
    path = run_dir / "cases" / str(plan["case_id"]) / "publication_evidence_freeze.json"
    write_json(path, manifest)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--repair-plan", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="pass_stop_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--reason", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allowlist-tag", default="")
    parser.add_argument("--stop-wait-seconds", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repair_plan = args.repair_plan or latest_repair_plan(DEFAULT_REPAIR_DIR)
    if repair_plan is None or not repair_plan.exists():
        raise FileNotFoundError("repair_loop_plan.csv was not found")
    config = load_lcb_config(args.config)
    profile = config.cluster_profile("o7_clean")
    repair_rows = read_csv_rows(repair_plan)
    active_rows = read_csv_rows(args.runtime_dir / "runtime_active_processes.csv")
    plans = plan_pass_stops(repair_rows, active_rows, profile)

    run_dir = args.output_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = run_dir / "pass_stop_ledger.jsonl"
    summary: dict[str, Any] = {
        "schema_version": "pelcb_pass_stop.v1",
        "generated_at": iso_now(),
        "project_id": config.project_id,
        "reason": args.reason,
        "execute": bool(args.execute),
        "repair_plan": str(repair_plan),
        "runtime_source": str(args.runtime_dir / "runtime_active_processes.csv"),
        "plans": plans,
    }
    write_json(run_dir / "pass_stop_plan.json", summary)
    blocked = [plan for plan in plans if plan["status"] != "READY"]
    ready = [plan for plan in plans if plan["status"] == "READY"]
    if not args.execute:
        summary["status"] = "DRY_RUN_READY" if not blocked else "DRY_RUN_BLOCKED"
        write_json(run_dir / "pass_stop_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if not blocked else 2
    if args.allowlist_tag != ALLOWLIST_TAG:
        raise PermissionError(f"execution requires --allowlist-tag {ALLOWLIST_TAG}")

    results: list[dict[str, Any]] = []
    for plan in ready:
        stop = stop_passed_process(
            profile.host,
            plan,
            wait_seconds=max(30, args.stop_wait_seconds),
        )
        checkpoint = checkpoint_metadata(profile.host, str(plan["node"]), str(plan["cwd"]))
        result: dict[str, Any] = {
            "case_id": plan["case_id"],
            "stopped_at": iso_now(),
            "returncode": stop.returncode,
            "stdout": stop.stdout.strip(),
            "stderr": stop.stderr.strip(),
            "checkpoint": checkpoint,
        }
        if stop.returncode == 0 and checkpoint["ok"]:
            manifest = freeze_evidence(run_dir, plan, checkpoint, repair_plan)
            result["status"] = (
                "ALREADY_STOPPED_EVIDENCE_FROZEN"
                if "already_stopped_checkpoint_ready" in stop.stdout
                else "STOPPED_EVIDENCE_FROZEN"
            )
            result["evidence_manifest"] = str(manifest)
        else:
            result["status"] = "FAILED_STOP_OR_CHECKPOINT"
        results.append(result)
        append_jsonl(ledger, {"event": "pass_stop", "at": iso_now(), **result})

    summary["results"] = results
    if blocked:
        summary["status"] = "PARTIAL" if results else "BLOCKED"
    elif all(result["status"].endswith("EVIDENCE_FROZEN") for result in results):
        summary["status"] = "COMPLETE" if results else "NO_ACTION"
    else:
        summary["status"] = "PARTIAL"
    summary["completed_at"] = iso_now()
    write_json(run_dir / "pass_stop_summary.json", summary)
    write_json(args.output_dir / "pass_stop_latest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] in {"COMPLETE", "NO_ACTION"} else 3


if __name__ == "__main__":
    sys.exit(main())
