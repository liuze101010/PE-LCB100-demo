#!/usr/bin/env python3
"""LCB300 postprocess closure helper.

This script closes the loop after ``lcb300_orchestrator.py --execute``:

1. Check the old-server Slurm jobs.
2. Refuse to collect while jobs are still running/pending, unless explicitly
   allowed.
3. Collect remote outputs through the existing orchestrator boundary.
4. Build an auditable merge plan from the collected staging directory.

It intentionally does not merge into ``lcb300_integrated`` by default.  The
point is to make the closure step repeatable without letting partial Slurm
products silently pollute the formal integrated dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from lcb_project_config import load_lcb_config
from backfill_clean_intrachain_g3 import (
    discover_msd_roots as discover_clean_g3_msd_roots,
    run_backfill as run_clean_g3_backfill,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ID = "lcb300"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "lcb300_postprocess_closure"
ORCHESTRATOR_OUTPUT = SCRIPT_DIR / "output" / "lcb300_orchestrator"
ORCHESTRATOR_SCRIPT = SCRIPT_DIR / "lcb300_orchestrator.py"
OLD_SERVER = "${REMOTE_USER}@private-3f2a50db.invalid"
CONFIG_PATH: Path | None = None

# Current all-source batch submitted after the output-root fix.
CURRENT_BATCH_JOBS = ["134760", "134761", "134762", "134763", "134764", "134765", "134766"]

ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
}
OK_TERMINAL_STATES = {"COMPLETED"}
BAD_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
TERMINAL_STATES = OK_TERMINAL_STATES | BAD_TERMINAL_STATES


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def ssh(command: str) -> subprocess.CompletedProcess[str]:
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", OLD_SERVER, command])


def parse_jobs(text: str) -> list[str]:
    jobs: list[str] = []
    for part in re.split(r"[\s,;]+", text.strip()):
        if not part:
            continue
        if not re.match(r"^\d+(?:_\d+)?$", part):
            raise ValueError(f"invalid Slurm job id: {part}")
        jobs.append(part)
    return jobs


def normalize_state(state: str) -> str:
    # sacct can return COMPLETED+ or CANCELLED by 12345.  Keep the primary state.
    return re.split(r"[+\s]", str(state).strip().upper())[0]


def query_slurm(jobs: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    sacct_rows: list[dict[str, Any]] = []
    squeue_rows: list[dict[str, Any]] = []

    if jobs:
        job_csv = ",".join(jobs)
        sacct_cmd = (
            f"sacct -j {job_csv} -n -X "
            "--format=JobIDRaw,JobName%50,State,Elapsed,ExitCode -P 2>/dev/null || true"
        )
        squeue_cmd = f"squeue -j {job_csv} -h -o '%i|%j|%T|%M|%D|%R' 2>/dev/null || true"
    else:
        # Fallback discovery for project-specific jobs submitted today.  This is useful
        # for future runs when job ids were not copied into the command line.
        sacct_cmd = (
            "sacct -u ${REMOTE_USER} -S $(date +%F) -n -X "
            "--format=JobIDRaw,JobName%50,State,Elapsed,ExitCode -P 2>/dev/null "
            f"| grep '|{PROJECT_ID}_' || true"
        )
        squeue_cmd = f"squeue -u ${{REMOTE_USER}} -h -o '%i|%j|%T|%M|%D|%R' 2>/dev/null | grep '|{PROJECT_ID}_' || true"

    sacct = ssh(sacct_cmd)
    if sacct.returncode != 0:
        errors.append(f"sacct failed: {sacct.stderr.strip()}")
    for line in sacct.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id, name, state, elapsed, exit_code = parts[:5]
        if f"{PROJECT_ID}_" not in name and jobs:
            # For explicit job ids, keep all rows; for discovered rows grep already filtered.
            pass
        sacct_rows.append(
            {
                "job_id": job_id.strip(),
                "job_name": name.strip(),
                "state": normalize_state(state),
                "elapsed": elapsed.strip(),
                "exit_code": exit_code.strip(),
                "source": "sacct",
            }
        )

    squeue = ssh(squeue_cmd)
    if squeue.returncode != 0:
        errors.append(f"squeue failed: {squeue.stderr.strip()}")
    for line in squeue.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        job_id, name, state, elapsed, nodes, reason = parts[:6]
        squeue_rows.append(
            {
                "job_id": job_id.strip(),
                "job_name": name.strip(),
                "state": normalize_state(state),
                "elapsed": elapsed.strip(),
                "nodes": nodes.strip(),
                "reason": reason.strip(),
                "source": "squeue",
            }
        )
    return sacct_rows, squeue_rows, errors


def summarize_status(
    jobs: list[str],
    sacct_rows: list[dict[str, Any]],
    squeue_rows: list[dict[str, Any]],
    errors: list[str],
    *,
    overlay_available: bool = False,
    overlay_path: str = "",
) -> dict[str, Any]:
    states: dict[str, int] = {}
    for row in sacct_rows + squeue_rows:
        state = str(row.get("state", "UNKNOWN"))
        states[state] = states.get(state, 0) + 1
    active = [row for row in squeue_rows if str(row.get("state")) in ACTIVE_STATES]
    failed = [row for row in sacct_rows if str(row.get("state")) in BAD_TERMINAL_STATES]
    completed = [row for row in sacct_rows if str(row.get("state")) in OK_TERMINAL_STATES]
    ignored_discovered_failed = []
    if not jobs and failed:
        # Fallback discovery is used after supervisor restarts when explicit
        # job ids were not retained. Historical failed rows from earlier code
        # generations must be preserved for triage, but they should not block a
        # fixed supervisor from submitting a fresh postprocess wave forever.
        # Explicit job ids still fail closed through the normal path above.
        ignored_discovered_failed = failed
        failed = []
    explicit_job_count = len(jobs)
    # If explicit base array ids were supplied, sacct may report one parent row
    # per array plus squeue individual pending tasks.  The safe completion gate is
    # simply: nothing active in squeue and no failed terminal rows.
    complete = bool(jobs) and bool(sacct_rows) and not active and not failed
    closure_state = "waiting_for_remote_jobs"
    if errors:
        complete = False
        closure_state = "slurm_query_error"
    elif not jobs and overlay_available:
        complete = True
        closure_state = "overlay_available_no_remote_job_needed"
    elif not jobs:
        complete = False
        closure_state = "tracking_gap_warn"
    elif active:
        closure_state = "remote_jobs_active"
    elif failed:
        closure_state = "remote_jobs_failed"
    elif complete:
        closure_state = "remote_jobs_complete"
    elif jobs and not sacct_rows and not squeue_rows:
        closure_state = "tracking_gap_warn"
    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "old_server": OLD_SERVER,
        "closure_state": closure_state,
        "overlay_available": overlay_available,
        "overlay_path": overlay_path,
        "explicit_jobs": jobs,
        "explicit_job_count": explicit_job_count,
        "sacct_row_count": len(sacct_rows),
        "squeue_row_count": len(squeue_rows),
        "state_counts": dict(sorted(states.items())),
        "active_count": len(active),
        "failed_count": len(failed),
        "completed_row_count": len(completed),
        "complete_gate_ok": complete,
        "errors": errors,
        "active_rows": active[:50],
        "failed_rows": failed,
        "ignored_discovered_failed_rows": ignored_discovered_failed,
    }


def latest_collect_dir(base: Path | None = None) -> Path | None:
    if base is None:
        base = ORCHESTRATOR_OUTPUT
    if not base.exists():
        return None
    candidates = [path for path in base.iterdir() if path.is_dir() and path.name.startswith("remote_collect_")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def collect_outputs(output_root: Path) -> Path:
    command = [
        sys.executable,
        str(ORCHESTRATOR_SCRIPT),
        "--collect",
        "--output-dir",
        str(ORCHESTRATOR_OUTPUT),
    ]
    if CONFIG_PATH:
        command.extend(["--config", str(CONFIG_PATH)])
    proc = run(command)
    if proc.returncode != 0:
        raise RuntimeError(f"collect failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    collect_dir = latest_collect_dir(ORCHESTRATOR_OUTPUT)
    if collect_dir is None:
        raise RuntimeError("collect completed but no remote_collect_* directory was found")
    marker = output_root / f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(marker, {"collect_dir": str(collect_dir), "stdout": proc.stdout.strip()})
    return collect_dir


def logical_relpath(rel: Path) -> Path:
    """Return the analysis-root-relative path inside a collect tree.

    The collector currently extracts remote files beneath an ``output/`` prefix
    because older smoke runs wrote to ``$OLD_WORK/output``.  Newer runs write
    directly to ``$OLD_OUTPUT``.  The merge planner accepts both layouts.
    """

    if rel.parts and rel.parts[0] == "output":
        return Path(*rel.parts[1:])
    return rel


def infer_case_id(parts: tuple[str, ...]) -> str:
    for part in parts:
        match = re.match(r"((?:lcb\d+|lcb_[A-Za-z0-9]+)_\d+_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_c\d+)", part)
        if match:
            return match.group(1)
    return ""


def infer_observable(rel: Path) -> str:
    rel = logical_relpath(rel)
    parts = rel.parts
    joined = "/".join(parts).lower()
    integrated_dir = f"{PROJECT_ID}_integrated"
    if parts and parts[0] == f"{PROJECT_ID}_density_energy":
        return "density"
    if parts and parts[0].startswith(f"{PROJECT_ID}_rg"):
        return "Rg"
    if parts and parts[0].startswith(f"{PROJECT_ID}_rdf"):
        return "RDF"
    if parts and parts[0].startswith(f"{PROJECT_ID}_msd"):
        if "/ree/" in joined:
            return "Ree"
        return "MSD"
    if parts and parts[0].startswith(f"{PROJECT_ID}_trajectory"):
        return "trajectory_intermediate"
    if parts and parts[0].startswith(f"{PROJECT_ID}_dynamics"):
        return "dynamics"
    if len(parts) > 2 and parts[0] == integrated_dir and parts[1] == "IntrachainDynamics":
        filename = parts[-1].lower()
        if "g1_monomer" in filename or "monomer_mapping" in filename:
            return "g1_monomer"
        if "contact" in filename:
            return "contact"
        return "intrachain_dynamics"
    if len(parts) > 2 and parts[0] == integrated_dir and parts[1] == "ShapeTensor":
        return "shape"
    return "unknown"


def build_merge_plan(collect_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    if not collect_dir.exists():
        raise FileNotFoundError(f"collect dir not found: {collect_dir}")
    rows: list[dict[str, Any]] = []
    for path in collect_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "remote_outputs.tgz" or path.name == "collect_manifest.json":
            continue
        extracted_rel = path.relative_to(collect_dir)
        rel = logical_relpath(extracted_rel)
        observable = infer_observable(rel)
        case_id = infer_case_id(rel.parts)
        target_hint = ""
        merge_action = "review"
        if rel.parts and rel.parts[0] == f"{PROJECT_ID}_integrated":
            target_hint = str(SCRIPT_DIR / "output" / rel)
            merge_action = "copy_after_validation"
        else:
            target_hint = str(SCRIPT_DIR / "output" / f"{PROJECT_ID}_integrated" / "_remote_import_staging" / rel)
            merge_action = "stage_then_rebuild_aggregate"
        rows.append(
            {
                "observable": observable,
                "case_id": case_id,
                "source_file": str(path),
                "relative_file": str(rel).replace("\\", "/"),
                "target_hint": target_hint,
                "merge_action": merge_action,
                "size_bytes": path.stat().st_size,
            }
        )
    rows.sort(key=lambda row: (row["observable"], row["case_id"], row["relative_file"]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"merge_plan_{timestamp}.csv"
    json_path = output_dir / f"merge_plan_{timestamp}.json"
    fields = ["observable", "case_id", "source_file", "relative_file", "target_hint", "merge_action", "size_bytes"]
    write_csv(csv_path, rows, fields)
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "collect_dir": str(collect_dir),
        "file_count": len(rows),
        "observables": {},
        "cases_by_observable": {},
        "project_id": PROJECT_ID,
        "note": f"Plan only. Formal {PROJECT_ID}_integrated outputs were not modified.",
    }
    for row in rows:
        obs = row["observable"]
        summary["observables"][obs] = summary["observables"].get(obs, 0) + 1
        if row["case_id"]:
            summary["cases_by_observable"].setdefault(obs, set()).add(row["case_id"])
    summary["cases_by_observable"] = {
        obs: sorted(cases) for obs, cases in summary["cases_by_observable"].items()
    }
    write_json(json_path, summary)
    return csv_path, json_path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_planned_cases(plan_csv: Path | None) -> set[str]:
    if plan_csv is None:
        return set()
    if not plan_csv.exists():
        raise FileNotFoundError(f"postprocess plan CSV not found: {plan_csv}")
    cases: set[str] = set()
    with plan_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id", "") or "").strip()
            if case_id:
                cases.add(case_id)
    return cases


def allowed_project_relpath(rel: Path) -> bool:
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        return False
    root = rel.parts[0]
    allowed_exact = {
        f"{PROJECT_ID}_density_energy",
        f"{PROJECT_ID}_integrated",
    }
    allowed_prefixes = (
        f"{PROJECT_ID}_rg_",
        f"{PROJECT_ID}_rdf_",
        f"{PROJECT_ID}_msd_",
        f"{PROJECT_ID}_ree_",
        f"{PROJECT_ID}_dynamics_",
        f"{PROJECT_ID}_trajectory_",
    )
    return root in allowed_exact or any(root.startswith(prefix) for prefix in allowed_prefixes)


def apply_merge(
    collect_dir: Path,
    output_dir: Path,
    *,
    plan_csv: Path | None = None,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    if not collect_dir.exists():
        raise FileNotFoundError(f"collect dir not found: {collect_dir}")
    planned_cases = load_planned_cases(plan_csv)
    if not planned_cases:
        raise ValueError("apply-merge requires a non-empty full_postprocess_plan.csv case set")
    target_root = (SCRIPT_DIR / "output").resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(collect_dir.rglob("*")):
        if not path.is_file() or path.name in {"remote_outputs.tgz", "collect_manifest.json"}:
            continue
        extracted_rel = path.relative_to(collect_dir)
        rel = logical_relpath(extracted_rel)
        case_id = infer_case_id(rel.parts)
        observable = infer_observable(rel)
        status = "skipped"
        reason = ""
        target = target_root / rel
        if not allowed_project_relpath(rel):
            reason = "outside_allowed_project_prefix"
        elif not case_id:
            reason = "no_case_id_in_path"
        elif case_id not in planned_cases:
            reason = "case_not_in_current_postprocess_plan"
        elif not str(target.resolve()).startswith(str(target_root)):
            reason = "target_outside_output_root"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            status = "copied"
            reason = "ok"
        rows.append(
            {
                "status": status,
                "reason": reason,
                "case_id": case_id,
                "observable": observable,
                "source_file": str(path),
                "relative_file": str(rel).replace("\\", "/"),
                "target_file": str(target),
                "size_bytes": path.stat().st_size,
                "source_sha256": file_sha256(path),
                "target_sha256": file_sha256(target) if status == "copied" and target.exists() else "",
            }
        )
    backfill_summary: dict[str, Any] = {}
    intrachain = target_root / f"{PROJECT_ID}_integrated" / "IntrachainDynamics"
    msd_roots = discover_clean_g3_msd_roots(target_root)
    if intrachain.exists() and msd_roots:
        try:
            backfill_summary = run_clean_g3_backfill(intrachain, msd_roots)
        except Exception as exc:
            backfill_summary = {"status": "failed", "error": str(exc)}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"applied_merge_manifest_{timestamp}.csv"
    json_path = output_dir / f"applied_merge_manifest_{timestamp}.json"
    fields = [
        "status",
        "reason",
        "case_id",
        "observable",
        "source_file",
        "relative_file",
        "target_file",
        "size_bytes",
        "source_sha256",
        "target_sha256",
    ]
    write_csv(csv_path, rows, fields)
    write_json(
        json_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "project_id": PROJECT_ID,
            "collect_dir": str(collect_dir),
            "plan_csv": str(plan_csv) if plan_csv else "",
            "planned_case_count": len(planned_cases),
            "copied_file_count": sum(1 for row in rows if row["status"] == "copied"),
            "skipped_file_count": sum(1 for row in rows if row["status"] != "copied"),
            "clean_g3_backfill": backfill_summary,
            "rows": rows,
        },
    )
    return csv_path, json_path, rows


def write_status_markdown(path: Path, summary: dict[str, Any], sacct_rows: list[dict[str, Any]], squeue_rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {PROJECT_ID.upper()} Postprocess Closure Status",
        "",
        f"- Checked: `{summary['checked_at']}`",
        f"- Old server: `{summary['old_server']}`",
        f"- Closure state: `{summary.get('closure_state', 'unknown')}`",
        f"- Complete gate: `{'OK' if summary['complete_gate_ok'] else 'WAIT'}`",
        f"- Active rows in squeue: `{summary['active_count']}`",
        f"- Failed terminal rows: `{summary['failed_count']}`",
        "",
        "## State Counts",
        "",
    ]
    for state, count in summary["state_counts"].items():
        lines.append(f"- `{state}`: {count}")
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in summary["errors"]:
            lines.append(f"- {error}")
    if squeue_rows:
        lines.extend(["", "## Active/Pending Sample", ""])
        for row in squeue_rows[:30]:
            lines.append(
                f"- `{row.get('job_id')}` `{row.get('job_name')}` `{row.get('state')}` "
                f"`{row.get('elapsed')}` `{row.get('reason', '')}`"
            )
    if summary["failed_rows"]:
        lines.extend(["", "## Failed Rows", ""])
        for row in summary["failed_rows"]:
            lines.append(f"- `{row.get('job_id')}` `{row.get('job_name')}` `{row.get('state')}` `{row.get('exit_code')}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="LCB project config JSON. Defaults to configs/lcb300.json.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--jobs",
        default=None,
        help="Comma/space separated Slurm job IDs. Pass empty string to discover today's project-specific jobs.",
    )
    parser.add_argument("--status-only", action="store_true", help="Only query Slurm and write status files.")
    parser.add_argument("--collect", action="store_true", help="Collect old-server outputs if all jobs are complete.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow collect even with active/pending jobs.")
    parser.add_argument("--merge-plan", action="store_true", help="Build a merge plan from a collected staging directory.")
    parser.add_argument("--apply-merge", action="store_true", help="Copy validated collected outputs into local canonical output.")
    parser.add_argument("--collect-dir", type=Path, help="Existing remote_collect_* directory for --merge-plan.")
    parser.add_argument("--plan-csv", type=Path, help="full_postprocess_plan.csv from the orchestrator run being closed.")
    parser.add_argument("--all", action="store_true", help="Status, collect when complete, then merge-plan.")
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> None:
    global PROJECT_ID, OLD_SERVER, ORCHESTRATOR_OUTPUT, CONFIG_PATH

    config = load_lcb_config(args.config)
    CONFIG_PATH = Path(args.config) if args.config else None
    PROJECT_ID = config.project_id
    OLD_SERVER = config.old_server
    ORCHESTRATOR_OUTPUT = SCRIPT_DIR / "output" / f"{PROJECT_ID}_orchestrator"
    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "output" / f"{PROJECT_ID}_postprocess_closure"
    if args.jobs is None:
        args.jobs = ",".join(CURRENT_BATCH_JOBS) if PROJECT_ID == "lcb300" else ""


def main() -> None:
    args = parse_args()
    apply_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = parse_jobs(args.jobs) if args.jobs.strip() else []

    sacct_rows, squeue_rows, errors = query_slurm(jobs)
    status = summarize_status(jobs, sacct_rows, squeue_rows, errors)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_json = args.output_dir / f"closure_status_{timestamp}.json"
    status_md = args.output_dir / f"closure_status_{timestamp}.md"
    write_json(
        status_json,
        {
            "summary": status,
            "sacct_rows": sacct_rows,
            "squeue_rows": squeue_rows,
        },
    )
    write_status_markdown(status_md, status, sacct_rows, squeue_rows)

    collect_requested = args.collect or args.all
    merge_requested = args.merge_plan or args.apply_merge or args.all
    collect_dir = args.collect_dir
    if collect_requested:
        if not status["complete_gate_ok"] and not args.allow_partial:
            print(f"WAIT: jobs are not complete; wrote {status_md}")
            print("Use --allow-partial only for debugging partial remote products.")
            return
        collect_dir = collect_outputs(args.output_dir)
        print(f"Collected remote outputs into {collect_dir}")

    if merge_requested:
        if collect_dir is None:
            collect_dir = latest_collect_dir()
        if collect_dir is None:
            print(f"No collect dir available; wrote {status_md}")
            return
        csv_path, json_path = build_merge_plan(collect_dir, args.output_dir)
        print(f"Wrote merge plan {csv_path}")
        print(f"Wrote merge summary {json_path}")
        if args.apply_merge:
            applied_csv, applied_json, rows = apply_merge(collect_dir, args.output_dir, plan_csv=args.plan_csv)
            copied = sum(1 for row in rows if row["status"] == "copied")
            print(f"Applied merge {applied_csv}")
            print(f"Wrote applied merge summary {applied_json}")
            print(f"Copied files: {copied}")

    if args.status_only or not (collect_requested or merge_requested):
        print(f"Wrote status {status_md}")


if __name__ == "__main__":
    main()
