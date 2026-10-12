#!/usr/bin/env python3
"""Read-only health triage for LCB automation runs.

The monitor classifies active MD tasks and post-processing closure state.  It
does not submit jobs, kill processes, migrate tasks, or modify remote state.
Remote log tailing is optional and read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lcb_project_config import LCBProjectConfig, load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent
STATUS_ORDER = [
    "running_ok",
    "low_gpu_expected",
    "stalled_no_log_growth",
    "failed_stage",
    "missing_checkpoint",
    "duplicate_or_unsafe",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def present(value: Any) -> bool:
    text = str(value if value is not None else "").strip()
    return text != "" and text.lower() not in {"nan", "none", "missing"}


def latest_json(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def ssh_target_for(row: pd.Series) -> str:
    host = str(row.get("host", "") or "").strip()
    node = str(row.get("node", "") or "").strip()
    host_label = str(row.get("host_label", "") or "").strip()
    if host_label == "new" and node:
        return node
    return host


def remote_stage_probe(row: pd.Series, timeout_s: int = 15) -> dict[str, Any]:
    target = ssh_target_for(row)
    cwd = str(row.get("cwd", "") or "").strip()
    stage = str(row.get("stage", "") or "").strip() or "nvt_prod"
    if not target or not cwd:
        return {"remote_probe_status": "skipped", "remote_probe_reason": "missing_target_or_cwd"}
    quoted_cwd = shlex.quote(cwd)
    quoted_stage = shlex.quote(stage)
    script = (
        f"cd {quoted_cwd} 2>/dev/null || exit 11; "
        f"latest=$(ls -1t {quoted_stage}.log {quoted_stage}.part*.log 2>/dev/null | head -1 || true); "
        f"cpt={quoted_stage}.cpt; "
        "now=$(date +%s); "
        "if [ -n \"$latest\" ]; then mtime=$(stat -c %Y \"$latest\" 2>/dev/null || echo 0); "
        "tailtxt=$(tail -n 20 \"$latest\" 2>/dev/null | tr '\\n\\t' '  ' | cut -c1-800); "
        "else mtime=0; tailtxt=''; fi; "
        "if [ -e \"$cpt\" ]; then cpt_present=1; cpt_mtime=$(stat -c %Y \"$cpt\" 2>/dev/null || echo 0); "
        "else cpt_present=0; cpt_mtime=0; fi; "
        "printf 'latest_log=%s\\nlatest_log_mtime=%s\\nlog_age_s=%s\\ncpt_present=%s\\ncpt_mtime=%s\\nlog_tail=%s\\n' "
        "\"$latest\" \"$mtime\" \"$((now-mtime))\" \"$cpt_present\" \"$cpt_mtime\" \"$tailtxt\""
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target, "bash", "-lc", script],
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    out: dict[str, Any] = {
        "remote_probe_status": "ok" if proc.returncode == 0 else "failed",
        "remote_probe_returncode": proc.returncode,
    }
    if proc.returncode != 0:
        out["remote_probe_reason"] = (proc.stderr or proc.stdout).strip()[:500]
        return out
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def classify_active(
    row: pd.Series,
    gpu_util: int | None,
    duplicate_cases: set[str],
    stalled_log_age_s: int,
    remote: dict[str, Any],
) -> tuple[str, str, bool]:
    case_id = str(row.get("case_id", "") or "").strip()
    stage = str(row.get("stage", "") or "").strip()
    etimes = as_int(row.get("etimes"), 0)
    cmd = str(row.get("cmd", "") or "")
    log_tail = str(remote.get("log_tail", "") or "")
    log_age_s = as_int(remote.get("log_age_s"), 0)
    cpt_present = str(remote.get("cpt_present", "")).strip() in {"1", "true", "True"}

    if not case_id or not stage or case_id in duplicate_cases:
        return "duplicate_or_unsafe", "missing case/stage or duplicate active case", False
    if any(token in log_tail.lower() for token in ["fatal error", "segmentation", "error in user input", "cannot open", "failed"]):
        return "failed_stage", "stage log tail contains failure text", False
    if remote.get("remote_probe_status") == "ok" and not cpt_present and etimes > 3600 and stage not in {"steep", "em"}:
        return "missing_checkpoint", "no stage checkpoint after more than 1 h", False
    if remote.get("remote_probe_status") == "ok" and log_age_s > stalled_log_age_s and "Finished mdrun" not in log_tail:
        return "stalled_no_log_growth", f"log has not grown for {log_age_s}s", cpt_present and "-bonded gpu" not in cmd
    if gpu_util is not None and gpu_util < 10:
        return "low_gpu_expected", f"GPU util {gpu_util}% is low; verify stage is CPU/update limited", cpt_present and "-bonded gpu" not in cmd
    return "running_ok", "active task has no local health blocker", False


def build_health(
    config: LCBProjectConfig,
    output_dir: Path,
    remote_logs: bool,
    stalled_log_age_s: int,
    max_remote_probes: int,
) -> dict[str, Path]:
    active = read_csv(config.runtime_output_dir / "runtime_active_processes.csv")
    runtime_cases = read_csv(config.runtime_output_dir / "runtime_cases.csv")
    gpus = read_csv(config.runtime_output_dir / "runtime_gpu_snapshot.csv")
    duplicates = read_csv(config.runtime_output_dir / "runtime_duplicates.csv")
    duplicate_cases = set(duplicates.get("case_id", pd.Series(dtype=str)).dropna().astype(str)) if not duplicates.empty else set()
    gpu_lookup: dict[tuple[str, str, str], int] = {}
    if not gpus.empty:
        for _, row in gpus.iterrows():
            gpu_lookup[
                (
                    str(row.get("host_label", "")),
                    str(row.get("node", "")),
                    str(row.get("gpu", "")),
                )
            ] = as_int(row.get("util_percent"), -1)

    rows: list[dict[str, Any]] = []
    remote_probe_count = 0
    for _, row in active.iterrows():
        gpu_key = (str(row.get("host_label", "")), str(row.get("node", "")), str(row.get("gpu_id", "")))
        gpu_util = gpu_lookup.get(gpu_key)
        if gpu_util is not None and gpu_util < 0:
            gpu_util = None
        if remote_logs and (max_remote_probes <= 0 or remote_probe_count < max_remote_probes):
            remote = remote_stage_probe(row)
            remote_probe_count += 1
        elif remote_logs:
            remote = {"remote_probe_status": "skipped", "remote_probe_reason": "remote_probe_limit"}
        else:
            remote = {"remote_probe_status": "skipped"}
        status, reason, restart_eligible = classify_active(row, gpu_util, duplicate_cases, stalled_log_age_s, remote)
        rows.append(
            {
                "checked_at": now_iso(),
                "case_id": row.get("case_id", ""),
                "host_label": row.get("host_label", ""),
                "host": row.get("host", ""),
                "node": row.get("node", ""),
                "gpu_id": row.get("gpu_id", ""),
                "pid": row.get("pid", ""),
                "stage": row.get("stage", ""),
                "etimes_s": row.get("etimes", ""),
                "gpu_util_percent": "" if gpu_util is None else gpu_util,
                "cwd": row.get("cwd", ""),
                "status": status,
                "reason": reason,
                "auto_restart_eligible": restart_eligible,
                "remote_probe_status": remote.get("remote_probe_status", ""),
                "latest_log": remote.get("latest_log", ""),
                "latest_log_mtime": remote.get("latest_log_mtime", ""),
                "log_age_s": remote.get("log_age_s", ""),
                "cpt_present": remote.get("cpt_present", ""),
                "cpt_mtime": remote.get("cpt_mtime", ""),
                "log_tail": remote.get("log_tail", ""),
            }
        )
    if not runtime_cases.empty and "runtime_status" in runtime_cases.columns:
        failed = runtime_cases[
            runtime_cases["runtime_status"].astype(str).str.lower().eq("failed_stage")
            & ~runtime_cases.get("active", pd.Series(False, index=runtime_cases.index)).astype(bool)
        ]
        for _, row in failed.iterrows():
            rows.append(
                {
                    "checked_at": now_iso(),
                    "case_id": row.get("case_id", ""),
                    "host_label": row.get("deployed_hosts", ""),
                    "host": "",
                    "node": "",
                    "gpu_id": "",
                    "pid": "",
                    "stage": row.get("failure_stage_name", "") or "unknown",
                    "etimes_s": "",
                    "gpu_util_percent": "",
                    "cwd": row.get("deployed_paths", ""),
                    "status": "failed_stage",
                    "reason": row.get("failure_reason", "runtime dashboard detected failed stage"),
                    "auto_restart_eligible": False,
                    "remote_probe_status": "runtime_dashboard",
                    "latest_log": row.get("latest_failure_logs", ""),
                    "latest_log_mtime": "",
                    "log_age_s": "",
                    "cpt_present": "",
                    "cpt_mtime": "",
                    "log_tail": row.get("failure_tail", ""),
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
        "gpu_util_percent",
        "cwd",
        "status",
        "reason",
        "auto_restart_eligible",
        "remote_probe_status",
        "latest_log",
        "latest_log_mtime",
        "log_age_s",
        "cpt_present",
        "cpt_mtime",
        "log_tail",
    ]
    status_csv = output_dir / "health_status.csv"
    write_csv(status_csv, rows, fields)

    counts = Counter(row["status"] for row in rows)
    lines = [
        f"# {config.project_id.upper()} Health Status",
        "",
        f"- Generated: `{now_iso()}`",
        f"- Remote log tails: `{remote_logs}`",
        f"- Remote probes performed: `{remote_probe_count}`",
        f"- Active processes inspected: `{len(rows)}`",
        "",
        "## Status Counts",
        "",
    ]
    for status in STATUS_ORDER:
        lines.append(f"- `{status}`: {counts.get(status, 0)}")
    unsafe = [row for row in rows if row["status"] in {"stalled_no_log_growth", "failed_stage", "missing_checkpoint", "duplicate_or_unsafe"}]
    lines.extend(["", "## Attention", ""])
    if unsafe:
        for row in unsafe[:80]:
            lines.append(f"- `{row['case_id']}` {row['host_label']}/{row['node']}/GPU{row['gpu_id']}: {row['status']} - {row['reason']}")
    else:
        lines.append("- No active task is classified as stalled/failed/unsafe by local health gates.")
    status_md = output_dir / "health_status.md"
    status_md.parent.mkdir(parents=True, exist_ok=True)
    status_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"health_status_csv": status_csv, "health_status_md": status_md}


def build_postprocess_triage(config: LCBProjectConfig, output_dir: Path) -> dict[str, Path]:
    closure_root = config.output_root / f"{config.project_id}_postprocess_closure"
    latest = latest_json(closure_root, "closure_status_*.json")
    rows: list[dict[str, Any]] = []
    if latest:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        for item in summary.get("failed_rows", []) or []:
            rows.append(
                {
                    "checked_at": now_iso(),
                    "source_report": str(latest),
                    "job_id": item.get("job_id", ""),
                    "job_name": item.get("job_name", ""),
                    "case_id": "",
                    "observable": item.get("job_name", ""),
                    "state": item.get("state", ""),
                    "reason": item.get("reason", ""),
                    "stderr_summary": "",
                    "auto_retry_eligible": "true",
                    "retry_policy": "max_once_after_manual_review_of_stderr",
                }
            )
        for error in summary.get("errors", []) or []:
            rows.append(
                {
                    "checked_at": now_iso(),
                    "source_report": str(latest),
                    "job_id": "",
                    "job_name": "",
                    "case_id": "",
                    "observable": "",
                    "state": "ERROR",
                    "reason": str(error),
                    "stderr_summary": "",
                    "auto_retry_eligible": "false",
                    "retry_policy": "closure_error_blocks_merge",
                }
            )
    fields = [
        "checked_at",
        "source_report",
        "job_id",
        "job_name",
        "case_id",
        "observable",
        "state",
        "reason",
        "stderr_summary",
        "auto_retry_eligible",
        "retry_policy",
    ]
    path = output_dir / "postprocess_failure_triage.csv"
    write_csv(path, rows, fields)
    return {"postprocess_failure_triage_csv": path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "lcb300.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--remote-logs", action="store_true", help="Read remote stage log tails through SSH.")
    parser.add_argument("--stalled-log-age-s", type=int, default=7200)
    parser.add_argument("--max-remote-probes", type=int, default=12, help="Maximum active cases to probe remotely when --remote-logs is enabled; <=0 means unlimited.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_lcb_config(args.config)
    output_dir = args.output_dir or (config.output_root / f"{config.project_id}_health_monitor")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    outputs.update(build_health(config, output_dir, args.remote_logs, args.stalled_log_age_s, args.max_remote_probes))
    outputs.update(build_postprocess_triage(config, output_dir))
    manifest = {
        "generated_at": now_iso(),
        "project_id": config.project_id,
        "remote_logs": args.remote_logs,
        "max_remote_probes": args.max_remote_probes,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "notes": [
            "Read-only monitor: no remote files are modified and no process is restarted.",
            "auto_restart_eligible is an advisory flag for the supervisor safety gate, not an action.",
        ],
    }
    manifest_path = output_dir / "health_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Wrote {outputs['health_status_csv']}")
    print(f"Wrote {outputs['health_status_md']}")
    print(f"Wrote {outputs['postprocess_failure_triage_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
