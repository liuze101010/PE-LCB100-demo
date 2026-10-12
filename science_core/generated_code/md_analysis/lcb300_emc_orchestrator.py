#!/usr/bin/env python3
"""LCB300 EMC and bridge automation helper.

This helper is intentionally narrower than the main supervisor.  It only moves
cases through the upstream part of the pipeline:

    screen_lcb300 case -> EMC package -> EMC Slurm array -> local EMC artifacts
    -> GROMACS bridge package

Default mode is dry-run.  With ``--execute`` it may prepare local EMC packages,
submit old-server Slurm arrays, collect completed EMC artifacts, or rebuild
bridge packages.  It never launches MD and never changes the LCB300 design
manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lcb_project_config import LCBProjectConfig, load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
EMC_DIR = SCRIPT_DIR.parent / "emc_alpha_olefin_builder"
PREPARE_EMC = EMC_DIR / "prepare_emc_production_build.py"
SUBMIT_EMC_ARRAY = EMC_DIR / "submit_emc_production_build_array.py"
REBUILD_BRIDGE = EMC_DIR / "batch_rebuild_bridge_fixed.py"
CASE_RE = re.compile(r"(?:lcb(\d+)_|lcb_[a-z]+_(\d+)_)", re.IGNORECASE)
CONFIG: LCBProjectConfig


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(command: list[str], *, timeout: int | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, cwd=str(cwd) if cwd else None)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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


def case_seq(case_id: str) -> int:
    match = CASE_RE.match(str(case_id))
    if not match:
        return 999999
    seq = next((group for group in match.groups() if group), None)
    return int(seq) if seq else 999999


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def read_manifest() -> pd.DataFrame:
    path = CONFIG.manifest_csv
    if not path.exists():
        raise FileNotFoundError(f"LCB manifest missing: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    if "case_id" not in df.columns:
        raise ValueError(f"manifest missing case_id column: {path}")
    df["case_seq"] = df["case_id"].map(case_seq)
    return df.sort_values("case_seq")


def source_run_label(case_id: str) -> str:
    root = source_case_root(case_id)
    for case_root in CONFIG.case_roots:
        if case_root.path == root:
            return case_root.label
    return root.name


def source_case_root(case_id: str) -> Path:
    matches = [case_root.path for case_root in CONFIG.case_roots if (case_root.path / case_id).exists()]
    if not matches:
        raise FileNotFoundError(f"Case {case_id} not found in configured case_roots")
    return matches[0]


def case_dir(case_id: str) -> Path:
    return source_case_root(case_id) / case_id


def package_dir(case_id: str) -> Path:
    return case_dir(case_id) / "emc_production_build"


def remote_build_dir(case_id: str) -> Path:
    return case_dir(case_id) / "remote_build" / source_run_label(case_id)


def has_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def read_exitcode(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def emc_packaged(case_id: str) -> bool:
    return has_file(package_dir(case_id) / "emc_package_manifest.json")


def emc_done(case_id: str) -> bool:
    root = remote_build_dir(case_id)
    return (
        read_exitcode(root / "setup.exitcode") == "0"
        and read_exitcode(root / "build.exitcode") == "0"
        and has_file(root / f"{case_id}.data")
        and has_file(root / f"{case_id}.params")
        and has_file(root / f"{case_id}.vmd")
    )


def bridge_ready(case_id: str) -> bool:
    for root in [*CONFIG.bridge_roots, CONFIG.default_emc_bridge_root]:
        cdir = root / case_id
        if has_file(cdir / f"{case_id}.gro") and has_file(cdir / "topol.top") and list(cdir.glob("*.itp")):
            return True
    return False


def active_emc_jobs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [job for job in state.get("emc_jobs", []) if str(job.get("status", "")) in {"submitted", "collect_failed"}]


def job_is_active(job_id: str) -> tuple[bool, str]:
    proc = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            CONFIG.old_server,
            f"squeue -h -j {shell_quote(job_id)} -o '%i|%T|%R' 2>/dev/null || true",
        ],
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"squeue failed for EMC job {job_id}: {proc.stderr.strip()}")
    output = proc.stdout.strip()
    return bool(output), output


def build_status_rows(manifest: pd.DataFrame, state: dict[str, Any]) -> list[dict[str, Any]]:
    submitted_cases = {
        case_id: job
        for job in active_emc_jobs(state)
        for case_id in job.get("cases", [])
    }
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        case_id = str(row["case_id"])
        rows.append(
            {
                "case_id": case_id,
                "case_seq": int(row["case_seq"]),
                "design_ready": True,
                "source_run_label": source_run_label(case_id),
                "case_dir": str(case_dir(case_id)),
                "emc_packaged": emc_packaged(case_id),
                "emc_submitted": case_id in submitted_cases,
                "emc_job_id": submitted_cases.get(case_id, {}).get("job_id", ""),
                "emc_done": emc_done(case_id),
                "bridge_ready": bridge_ready(case_id),
                "remote_build_dir": str(remote_build_dir(case_id)),
            }
        )
    return rows


def prepare_packages(cases: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case_id in cases:
        command = [
            sys.executable,
            str(PREPARE_EMC),
            "--case-dir",
            str(case_dir(case_id)),
            "--package-name",
            "emc_production_build",
            "--run-label",
            source_run_label(case_id),
            "--partition",
            "amd",
            "--cpus-per-task",
            "4",
            "--time-limit",
            "02:00:00",
            "--remote-emc-root",
            CONFIG.old_emc_root,
        ]
        proc = run(command, timeout=120, cwd=ROOT)
        results.append(
            {
                "case_id": case_id,
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "package_dir": str(package_dir(case_id)),
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(f"prepare EMC package failed for {case_id}: {proc.stderr.strip()}")
    return results


def submit_emc_array(cases: list[str], output_dir: Path, timestamp: str) -> dict[str, Any]:
    run_labels = {source_run_label(case_id) for case_id in cases}
    if len(run_labels) != 1:
        raise ValueError(f"EMC array cases must share one run label, got {sorted(run_labels)}")
    packages = [package_dir(case_id) for case_id in cases]
    missing = [str(path) for path in packages if not has_file(path / "emc_package_manifest.json")]
    if missing:
        raise FileNotFoundError("EMC package manifest missing: " + "; ".join(missing))
    run_root = output_dir / "emc_runs" / timestamp
    command = [
        sys.executable,
        str(SUBMIT_EMC_ARRAY),
        "--remote-host",
        CONFIG.old_server,
        "--remote-root",
        CONFIG.emc_remote_root,
        "--array-name",
        f"{CONFIG.project_id}_supervisor_emc_{timestamp}",
        "--partition",
        "amd",
        "--cpus-per-task",
        "4",
        "--time-limit",
        "04:00:00",
        "--array-concurrency",
        "10",
        "--remote-emc-root",
        CONFIG.old_emc_root,
        "--log-root",
        str(run_root),
        "--submit-only",
    ]
    for path in packages:
        command.extend(["--package-dir", str(path)])
    proc = run(command, timeout=600, cwd=ROOT)
    summary = run_root / "array_submit_summary.json"
    payload = read_json(summary, {})
    job_id = str(payload.get("array_job_id", ""))
    if proc.returncode != 0 or not job_id:
        raise RuntimeError(f"EMC array submit failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return {
        "job_id": job_id,
        "cases": cases,
        "summary_path": str(summary),
        "run_root": str(run_root),
        "submitted_at": iso_now(),
        "status": "submitted",
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def validate_canonical(case_id: str, root: Path) -> tuple[bool, dict[str, Any]]:
    details = {
        "setup_exitcode": read_exitcode(root / "setup.exitcode"),
        "build_exitcode": read_exitcode(root / "build.exitcode"),
        "has_data": has_file(root / f"{case_id}.data"),
        "has_params": has_file(root / f"{case_id}.params"),
        "has_vmd": has_file(root / f"{case_id}.vmd"),
    }
    ok = (
        details["setup_exitcode"] == "0"
        and details["build_exitcode"] == "0"
        and details["has_data"]
        and details["has_params"]
        and details["has_vmd"]
    )
    return ok, details


def collect_emc_job(job: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    summary_path = Path(str(job["summary_path"]))
    summary = read_json(summary_path, {})
    remote_dir = str(summary.get("remote_dir", ""))
    remote_host = str(summary.get("remote_host", CONFIG.old_server))
    if not remote_dir:
        raise RuntimeError(f"EMC submit summary lacks remote_dir: {summary_path}")
    artifact_root = output_dir / "emc_runs" / f"artifacts_{job['job_id']}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for record in summary.get("cases", []):
        case_id = str(record["case_id"])
        package_name = str(record["package_name"])
        run_label = str(record["run_label"])
        remote_run = f"{remote_dir}/packages/{case_id}/{package_name}/{run_label}"
        artifact_dir = artifact_root / case_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        proc = run(
            [
                "scp",
                "-r",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                f"{remote_host}:{remote_run}",
                str(artifact_dir),
            ],
            timeout=900,
        )
        if proc.returncode != 0:
            results.append({"case_id": case_id, "ok": False, "error": proc.stderr.strip()})
            continue
        source_dir = artifact_dir / run_label
        if not source_dir.exists():
            source_dir = artifact_dir
        canonical = Path(str(record["canonical_remote_build_dir"]))
        ok_existing, _details_existing = validate_canonical(case_id, canonical)
        if canonical.exists() and not ok_existing:
            shutil.rmtree(canonical)
        if not canonical.exists():
            canonical.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, canonical)
        ok, details = validate_canonical(case_id, canonical)
        results.append(
            {
                "case_id": case_id,
                "ok": ok,
                "canonical_dir": str(canonical),
                "artifact_dir": str(artifact_dir),
                **details,
            }
        )
    ok_count = sum(1 for row in results if row.get("ok"))
    payload = {
        "job_id": job["job_id"],
        "collected_at": iso_now(),
        "total_cases": len(results),
        "ok_cases": ok_count,
        "failed_cases": len(results) - ok_count,
        "results": results,
    }
    write_json(output_dir / "emc_runs" / f"collect_{job['job_id']}.json", payload)
    if ok_count != len(results):
        raise RuntimeError(f"EMC collect incomplete for job {job['job_id']}: {ok_count}/{len(results)} ok")
    return payload


def rebuild_bridge(cases: list[str]) -> dict[str, Any]:
    bridge_output_root = CONFIG.default_emc_bridge_root
    bridge_output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    grouped: dict[str, list[str]] = {}
    for case_id in cases:
        grouped.setdefault(source_run_label(case_id), []).append(case_id)
    for run_label, group_cases in sorted(grouped.items()):
        command = [
            sys.executable,
            str(REBUILD_BRIDGE),
            "--input-root",
            str(source_case_root(group_cases[0])),
            "--output-root",
            str(bridge_output_root),
            "--run-label",
            run_label,
        ]
        for case_id in group_cases:
            command.extend(["--case", case_id])
        proc = run(command, timeout=900, cwd=ROOT)
        outputs.append(
            {
                "run_label": run_label,
                "cases": group_cases,
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(f"bridge rebuild failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return {
        "cases": cases,
        "output_root": str(bridge_output_root),
        "runs": outputs,
    }


def choose_action(rows: list[dict[str, Any]], state: dict[str, Any], max_cases: int) -> dict[str, Any]:
    for job in active_emc_jobs(state):
        active, raw = job_is_active(str(job["job_id"]))
        if active:
            return {"action": "wait_emc_array", "job": job, "squeue": raw, "cases": job.get("cases", [])}
        return {"action": "collect_emc_array", "job": job, "cases": job.get("cases", [])}

    needs_bridge = [row["case_id"] for row in rows if row["emc_done"] and not row["bridge_ready"]]
    if needs_bridge:
        return {"action": "rebuild_bridge", "cases": needs_bridge[:max_cases]}

    needs_emc = [row["case_id"] for row in rows if not row["emc_done"] and row["emc_packaged"] and not row["emc_submitted"]]
    if needs_emc:
        first_label = source_run_label(needs_emc[0])
        same_label = [case_id for case_id in needs_emc if source_run_label(case_id) == first_label]
        return {"action": "submit_emc_array", "cases": same_label[:max_cases], "run_label": first_label}

    not_packaged = [row["case_id"] for row in rows if not row["emc_done"] and not row["emc_packaged"]]
    if not_packaged:
        return {"action": "prepare_emc_packages", "cases": not_packaged[:max_cases]}

    return {"action": "none", "cases": []}


def write_decisions(path: Path, payload: dict[str, Any]) -> None:
    action = payload["decision"]["action"]
    cases = payload["decision"].get("cases", [])
    lines = [
        "# LCB300 EMC Orchestrator Decisions",
        "",
        f"- Time: `{payload['timestamp']}`",
        f"- Mode: `{'execute' if payload['execute'] else 'dry-run'}`",
        f"- Action: `{action}`",
        f"- Cases: `{len(cases)}`",
        f"- Status rows: `{payload['status_count']}`",
        "",
    ]
    for case_id in cases[:50]:
        lines.append(f"- `{case_id}`")
    if action == "wait_emc_array":
        lines.extend(["", "## Active EMC job", "", f"- `{payload['decision']['job']['job_id']}`"])
    write_text(path, "\n".join(lines) + "\n")


def event(action: str, result: str, **kwargs: Any) -> dict[str, Any]:
    return {"time": iso_now(), "action": action, "result": result, **kwargs}


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "emc_state.json"
    state = read_json(state_path, {"emc_jobs": []})
    manifest = read_manifest()
    rows = build_status_rows(manifest, state)
    write_csv(
        output_dir / "emc_bridge_status.csv",
        rows,
        [
            "case_id",
            "case_seq",
            "design_ready",
            "source_run_label",
            "case_dir",
            "emc_packaged",
            "emc_submitted",
            "emc_job_id",
            "emc_done",
            "bridge_ready",
            "remote_build_dir",
        ],
    )
    decision = choose_action(rows, state, max(1, args.max_cases))
    events: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    timestamp = now_stamp()

    if args.execute:
        action = decision["action"]
        cases = list(decision.get("cases", []))
        try:
            if action == "prepare_emc_packages":
                result = {"prepared": prepare_packages(cases)}
                events.append(event(action, "ok", cases=";".join(cases)))
            elif action == "submit_emc_array":
                job = submit_emc_array(cases, output_dir, timestamp)
                state.setdefault("emc_jobs", []).append(job)
                result = job
                events.append(event(action, "ok", cases=";".join(cases), job_id=job["job_id"]))
            elif action == "collect_emc_array":
                job = decision["job"]
                result = collect_emc_job(job, output_dir)
                for item in state.get("emc_jobs", []):
                    if str(item.get("job_id")) == str(job.get("job_id")):
                        item["status"] = "collected"
                        item["collected_at"] = iso_now()
                events.append(event(action, "ok", cases=";".join(cases), job_id=job["job_id"]))
            elif action == "rebuild_bridge":
                result = rebuild_bridge(cases)
                events.append(event(action, "ok", cases=";".join(cases)))
            elif action == "wait_emc_array":
                result = {"squeue": decision.get("squeue", "")}
                events.append(event(action, "wait", cases=";".join(cases), job_id=decision["job"]["job_id"]))
            else:
                events.append(event(action, "none"))
        except Exception as exc:  # noqa: BLE001
            events.append(event(action, "fail", cases=";".join(cases), message=str(exc)))
            result = {"error": str(exc)}
            if action == "collect_emc_array":
                for item in state.get("emc_jobs", []):
                    if str(item.get("job_id")) == str(decision.get("job", {}).get("job_id")):
                        item["status"] = "collect_failed"
            write_json(state_path, state)
            raise
    else:
        events.append(event(decision["action"], "dry-run", cases=";".join(decision.get("cases", []))))

    state["updated_at"] = iso_now()
    write_json(state_path, state)
    append_csv(
        output_dir / "emc_events.csv",
        events,
        ["time", "action", "result", "cases", "job_id", "message"],
    )

    payload = {
        "timestamp": iso_now(),
        "execute": bool(args.execute),
        "status_count": len(rows),
        "decision": decision,
        "result": result,
        "state_path": str(state_path),
    }
    write_json(output_dir / "emc_last_decision.json", payload)
    write_json(output_dir / "runs" / timestamp / "emc_decision.json", payload)
    write_decisions(output_dir / "emc_decisions.md", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    global CONFIG
    args = parse_args()
    CONFIG = load_lcb_config(args.config)
    if args.output_dir is None:
        args.output_dir = CONFIG.emc_output_dir
    payload = run_once(args)
    if not args.json_only:
        print(f"LCB300 EMC orchestrator action: {payload['decision']['action']}")
        print(f"Decisions: {Path(args.output_dir) / 'emc_decisions.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
