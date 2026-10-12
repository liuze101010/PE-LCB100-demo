#!/usr/bin/env python3
"""Closure watcher for O7 clean-engine sidecar runs.

The sidecar queue runs clean-engine MD under a separate remote root.  This
watcher keeps the downstream products separate from the production PE-LCB100
canonical table:

* remote stage status for every clean run,
* pressure-relaxation density from ``pr/pr.edr`` as soon as ``pr`` completes,
* a clean-only ready queue for full postprocessing after ``nvt_prod`` completes.

It intentionally does not merge clean outputs into the production canonical
aggregate.  Full postprocessing is queued under a clean validation namespace.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUEUE_DIR = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_cuda2024_supervisor_queue"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_cuda2024_closure"
NEW_HOST = "${REMOTE_USER}@private-4e2f23ff.invalid"
OLD_HOST = "${REMOTE_USER}@private-3f2a50db.invalid"
GMX_2024_CUDA = "${PELCB100_REMOTE_HOME}/tools/gromacs/bin/gmx2024_4_cuda112"
GMX_2024_ENV = "${PELCB100_REMOTE_HOME}/tools/gromacs/env_gromacs2024_4_cuda112.sh"
OLD_GMX_2024 = "${PELCB100_REMOTE_PATH}"
ORCHESTRATOR = SCRIPT_DIR / "lcb300_orchestrator.py"
POSTPROCESS_CLOSURE = SCRIPT_DIR / "lcb300_postprocess_closure.py"
CLEAN_REGATE = SCRIPT_DIR / "pelcb_clean_overlay_regate.py"
REPAIR_LOOP = SCRIPT_DIR / "pelcb_repair_loop.py"
PASS_STOPPER = SCRIPT_DIR / "pelcb_stop_passed_active.py"
REPAIR_LOOP_OUTPUT_DIR = SCRIPT_DIR / "output" / "pelcb100_repair_loop"
RUNTIME_ACTIVE_CSV = SCRIPT_DIR / "output" / "pelcb100_runtime_dashboard" / "runtime_active_processes.csv"
NVT_EXTENSION_OUTPUT_DIR = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_nvt_extension"
STAGES = ("steep", "rlx", "pr", "nvt_prod")
PLAN_FILES = {
    "o7_new_cuda2024_plan.csv": {"plan_kind": "new_cuda2024", "source_host_label": "new", "source_host": NEW_HOST},
    "o7_stage1_old2024_plan.csv": {"plan_kind": "old_intel_2024", "source_host_label": "old", "source_host": OLD_HOST},
}


def is_full_postprocess_eligible(row: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a clean run should enter full postprocessing.

    Smoke runs can legitimately reach nvt_prod_done with very short trajectories.
    They are useful for launch validation but must not be routed into full
    density/Rg/RDF/MSD/shape closure where 400 ns production evidence is
    expected.
    """
    tag = str(row.get("tag", "") or "").lower()
    remote_run_dir = str(row.get("remote_run_dir", "") or "").lower()
    if "smoke" in tag or "smoke" in remote_run_dir:
        return False, "smoke_run_not_full_postprocess"
    return True, ""


def configure_windows_ssh_path() -> None:
    if os.name != "nt":
        return
    if shutil.which("ssh") and shutil.which("scp"):
        return
    for candidate in (Path("D:/app/OpenSSH-Win64"), Path("C:/Windows/System32/OpenSSH")):
        if (candidate / "ssh.exe").exists() and (candidate / "scp.exe").exists():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            return


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_heartbeat(output_dir: Path, phase: str, **extra: Any) -> None:
    write_json(
        output_dir / "o7_clean_closure_heartbeat.json",
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


def acquire_watch_lock(output_dir: Path) -> Path:
    lock_dir = output_dir / "o7_clean_closure.watch.lock"
    owner_path = lock_dir / "owner.json"
    if lock_dir.exists():
        owner = load_json(owner_path, {})
        owner_pid = int(owner.get("pid", 0) or 0)
        if owner_pid and owner_pid != os.getpid() and pid_is_running(owner_pid):
            raise RuntimeError(f"closure watcher already active with pid {owner_pid}")
        shutil.rmtree(lock_dir, ignore_errors=True)
    lock_dir.mkdir(parents=True, exist_ok=False)
    write_json(owner_path, {"pid": os.getpid(), "started_at": iso_now()})
    return lock_dir


def release_watch_lock(lock_dir: Path | None) -> None:
    if lock_dir is not None:
        shutil.rmtree(lock_dir, ignore_errors=True)


def run_cmd(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def ssh(command: str, *, check: bool = True, timeout: int | None = 120) -> subprocess.CompletedProcess[str]:
    return ssh_host(NEW_HOST, command, check=check, timeout=timeout)


def ssh_host(
    host: str,
    command: str,
    *,
    check: bool = True,
    timeout: int | None = 120,
) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", host, command],
        check=check,
        timeout=timeout,
    )


def sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def row_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")).strip() for field in fields)


def merge_rows_by_key(
    existing: list[dict[str, Any]],
    current: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    merged = {row_key(row, key_fields): dict(row) for row in existing if all(row_key(row, key_fields))}
    for row in current:
        key = row_key(row, key_fields)
        if all(key):
            merged[key] = dict(row)
    return list(merged.values())


def truthy_bool(value: Any) -> bool:
    """Return True only for explicit truthy status values.

    CSV round-tripping turns booleans into strings.  A plain ``if
    row.get("nvt_prod_done")`` treats the string "False" as truthy and
    overstates closure counts, so status summaries must use this helper.
    """

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def case_sequence(case_id: str) -> str:
    match = re.search(r"_(\d{3})_", str(case_id or ""))
    return match.group(1) if match else ""


def case_has_excluded_sequence(case_id: str, excluded_sequences: set[str]) -> bool:
    seq = case_sequence(case_id)
    return bool(seq and seq in excluded_sequences)


def active_case_sequences_from_runtime_dashboard(path: Path = RUNTIME_ACTIVE_CSV) -> set[str]:
    sequences: set[str] = set()
    for row in read_csv(path):
        seq = case_sequence(str(row.get("case_id", "")))
        if seq:
            sequences.add(seq)
    return sequences


def load_plan_rows(
    queue_dir: Path,
    tags: list[str] | None = None,
    case_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    paths: list[Path] = []
    if tags:
        for tag in tags:
            for plan_name in PLAN_FILES:
                paths.append(queue_dir / tag / plan_name)
    else:
        for plan_name in PLAN_FILES:
            paths.extend(sorted(queue_dir.glob(f"*/{plan_name}")))
    allowed_cases = {case_id.strip() for case_id in (case_ids or []) if case_id.strip()}
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        plan_defaults = PLAN_FILES.get(path.name, {})
        for row in read_csv(path):
            remote = str(row.get("remote_run_dir", "")).strip()
            case_id = str(row.get("case_id", "")).strip()
            clean_case_id = str(row.get("clean_case_id", "")).strip()
            status = str(row.get("status", "")).upper()
            if not remote or status not in {"SUBMITTED", "RUNNING", "DONE", "PREPARED"}:
                continue
            if allowed_cases and case_id not in allowed_cases and clean_case_id not in allowed_cases:
                continue
            key = (case_id, remote)
            if key in seen:
                continue
            seen.add(key)
            row = dict(row)
            row["plan_csv"] = str(path)
            row["tag"] = path.parent.name
            row["plan_kind"] = str(row.get("plan_kind") or plan_defaults.get("plan_kind", ""))
            row["source_host_label"] = str(
                row.get("target_host_label")
                or row.get("source_host_label")
                or plan_defaults.get("source_host_label", "")
            )
            row["source_host"] = str(row.get("target_host") or row.get("source_host") or plan_defaults.get("source_host", ""))
            rows.append(row)
    return rows


def remote_status_script(remote_dirs: list[str]) -> str:
    script = f"""
import json, os, pathlib, re, sys
dirs = {json.dumps(remote_dirs, ensure_ascii=False)}
stages = {list(STAGES)!r}
def tail_contains(path, needle, max_bytes=1048576):
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, 2)
            data = handle.read()
        return needle.encode() in data
    except OSError:
        return False
def finished(stage_dir, stage):
    if not stage_dir.exists():
        return False
    logs = [stage_dir / f"{{stage}}.log"] + sorted(stage_dir.glob(f"{{stage}}.part*.log"))
    return any(p.exists() and tail_contains(p, "Finished mdrun") for p in logs)
def fatal(stage_dir, stage):
    if not stage_dir.exists():
        return ""
    logs = [stage_dir / f"{{stage}}.log"] + sorted(stage_dir.glob(f"{{stage}}.part*.log"))
    for p in logs:
        if p.exists() and tail_contains(p, "Fatal error"):
            return str(p.name)
    return ""
def active_nvt_prod(root):
    nvt_dir = root / "nvt_prod"
    try:
        target = str(nvt_dir.resolve())
    except OSError:
        target = str(nvt_dir)
    for proc in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            cwd = str((proc / "cwd").resolve())
        except OSError:
            continue
        if cwd != target:
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\\x00", b" ").decode("utf-8", errors="ignore")
        except OSError:
            cmd = ""
        if "mdrun" in cmd or "gmx" in cmd or "gmx_mpi" in cmd:
            return True
    return False
for d in dirs:
    root = pathlib.Path(d)
    row = {{"remote_run_dir": d, "exists": root.exists()}}
    for st in stages:
        sd = root / st
        row[f"{{st}}_done"] = finished(sd, st)
        row[f"{{st}}_fatal_log"] = fatal(sd, st)
        row[f"{{st}}_has_tpr"] = (sd / f"{{st}}.tpr").exists()
        row[f"{{st}}_has_cpt"] = (sd / f"{{st}}.cpt").exists()
        row[f"{{st}}_has_edr"] = (sd / f"{{st}}.edr").exists() or bool(list(sd.glob(f"{{st}}.part*.edr")))
    row["nvt_prod_active"] = active_nvt_prod(root)
    row["pr_edr"] = str(root / "pr" / "pr.edr")
    row["nvt_edr"] = str(root / "nvt_prod" / "nvt_prod.edr")
    print(json.dumps(row, ensure_ascii=False))
"""
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return "python3 -c " + sh_quote(f"import base64; exec(base64.b64decode('{encoded}'))")


def row_source_host(row: dict[str, Any]) -> str:
    host = str(row.get("target_host") or row.get("source_host") or "").strip()
    if host:
        return host
    if str(row.get("source_host_label") or row.get("target_host_label") or "").strip() == "old":
        return OLD_HOST
    if str(row.get("plan_kind") or "").startswith("old_"):
        return OLD_HOST
    return NEW_HOST


def row_source_host_label(row: dict[str, Any]) -> str:
    label = str(row.get("target_host_label") or row.get("source_host_label") or "").strip()
    if label:
        return label
    return "old" if row_source_host(row) == OLD_HOST else "new"


def host_for_remote_run_dir(remote_run_dir: str) -> tuple[str, str]:
    """Infer the concrete SSH host for a clean/extension run directory."""

    remote = str(remote_run_dir or "").strip()
    if remote.startswith("${PELCB100_REMOTE_ROOT}/"):
        return OLD_HOST, "old"
    return NEW_HOST, "new"


def parse_extension_result_payload(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
    return payload if isinstance(payload, dict) else {}


def latest_nvt_extension_sources(
    case_ids: set[str],
    extension_root: Path = NVT_EXTENSION_OUTPUT_DIR,
) -> dict[str, dict[str, Any]]:
    """Return latest successful NVT-extension run source per requested case.

    The plan CSV ``remote_run_dir`` can be the source trajectory, while the
    actual extended trajectory may be stored inside ``result.remote_run_dir``.
    Forced MSD postprocessing must use the latter when present; otherwise it
    silently reprocesses the 400 ns clean run and never reaches the bounded
    800 ns review gate.
    """

    allowed = {str(case_id).strip() for case_id in case_ids if str(case_id).strip()}
    if not allowed or not extension_root.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for plan_path in sorted(extension_root.glob("*/pelcb_clean_nvt_extension_plan.csv")):
        try:
            rows = read_csv(plan_path)
        except Exception:
            continue
        plan_mtime = plan_path.stat().st_mtime
        for row in rows:
            ids = {
                str(row.get("case_id", "")).strip(),
                str(row.get("clean_case_id", "")).strip(),
            }
            matched = sorted(ids & allowed)
            if not matched:
                continue
            payload = parse_extension_result_payload(row.get("result", ""))
            if not truthy_bool(payload.get("ok")):
                continue
            remote_run_dir = str(payload.get("remote_run_dir") or row.get("remote_run_dir") or "").strip()
            if not remote_run_dir:
                continue
            source_host, source_host_label = host_for_remote_run_dir(remote_run_dir)
            record = {
                "remote_run_dir": remote_run_dir,
                "source_host": source_host,
                "source_host_label": source_host_label,
                "extension_plan_csv": str(plan_path),
                "extension_plan_mtime": plan_mtime,
                "extension_result": payload,
            }
            for matched_id in matched:
                prior = latest.get(matched_id)
                if prior is None or plan_mtime >= float(prior.get("extension_plan_mtime", 0.0)):
                    latest[matched_id] = dict(record)
    return latest


def apply_nvt_extension_source_overrides(
    rows: list[dict[str, str]],
    extension_sources: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    if not extension_sources:
        return rows
    out: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        override = extension_sources.get(str(row.get("case_id", "")).strip()) or extension_sources.get(
            str(row.get("clean_case_id", "")).strip()
        )
        if override:
            row["remote_run_dir"] = str(override.get("remote_run_dir", ""))
            row["source_host"] = str(override.get("source_host", ""))
            row["source_host_label"] = str(override.get("source_host_label", ""))
            row["target_host"] = str(override.get("source_host", ""))
            row["target_host_label"] = str(override.get("source_host_label", ""))
            row["extension_source_override"] = "true"
            row["extension_plan_csv"] = str(override.get("extension_plan_csv", ""))
        out.append(row)
    return out


def source_host_filter_for_hosts(source_hosts: set[str]) -> str:
    """Return the orchestrator source-host filter for concrete SSH hosts.

    Older repair-loop rows may carry a stale host label while preserving the
    actual SSH host in ``source_host``/``target_host``.  The postprocessing
    orchestrator ultimately stages from SSH hosts, so the handoff must be
    driven by the concrete host rather than by the potentially stale label.
    """
    hosts = {str(host or "").strip() for host in source_hosts if str(host or "").strip()}
    if not hosts:
        return "all"
    if hosts == {NEW_HOST}:
        return "new"
    if hosts == {OLD_HOST}:
        return "old"
    if hosts == {"${REMOTE_USER}@private-169fa2c7.invalid"}:
        return "newnew"
    return "all"


def fetch_remote_status(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    out: dict[str, dict[str, Any]] = {}
    # Keep each SSH command short.  Windows OpenSSH can hang on long nested
    # command strings, while one directory per command is stable and fast.
    for row in rows:
        remote_dir = row["remote_run_dir"]
        host = row_source_host(row)
        result = ssh_host(host, remote_status_script([remote_dir]), timeout=120)
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            payload = json.loads(line)
            payload["source_host"] = host
            payload["source_host_label"] = row_source_host_label(row)
            out[str(payload["remote_run_dir"])] = payload
    return out


def extract_density_script(remote_dir: str, stage: str, out_name: str, *, source_host: str = NEW_HOST) -> str:
    root = sh_quote(remote_dir)
    stage_q = sh_quote(stage)
    out_q = sh_quote(out_name)
    if source_host == OLD_HOST:
        gmx_setup = f"module purge\nmodule load intel${PELCB100_REMOTE_PATH}\nGMX={sh_quote(OLD_GMX_2024)}"
    else:
        gmx_setup = f"source {sh_quote(GMX_2024_ENV)}\nGMX={sh_quote(GMX_2024_CUDA)}"
    return f"""set -e
{gmx_setup}
cd {root}
stage={stage_q}
out={out_q}
mkdir -p o7_closure
edr="$stage/$stage.edr"
if [ ! -s "$edr" ]; then
  edr=$(ls "$stage/$stage".part*.edr 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -z "$edr" ] || [ ! -s "$edr" ]; then
    echo '{{"ok": false, "reason": "missing_edr"}}'
    exit 0
  fi
fi
printf 'Density\\n0\\n' | "$GMX" energy -f "$edr" -o "o7_closure/$out" -xvg none >/tmp/o7_energy_$$.out 2>/tmp/o7_energy_$$.err || {{
  echo '{{"ok": false, "reason": "gmx_energy_failed"}}'
  cat /tmp/o7_energy_$$.err >&2 || true
  rm -f /tmp/o7_energy_$$.out /tmp/o7_energy_$$.err
  exit 0
}}
python3 - <<'PY'
import json, math, pathlib
p = pathlib.Path("o7_closure") / {json.dumps(out_name)}
rows = []
for line in p.read_text(errors="ignore").splitlines():
    if not line.strip() or line.startswith(("#", "@")):
        continue
    parts = line.split()
    if len(parts) >= 2:
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            pass
if not rows:
    print(json.dumps({{"ok": False, "reason": "empty_xvg"}}))
else:
    t0, t1 = rows[0][0], rows[-1][0]
    start = max(t0, t1 - 10000.0)
    tail = [v for t, v in rows if t >= start]
    # GROMACS density is kg/m^3; divide by 1000 for g/cc.
    mean_kgm3 = sum(tail) / len(tail)
    print(json.dumps({{
        "ok": True,
        "stage": {json.dumps(stage)},
        "xvg": str(p),
        "frames": len(rows),
        "tail_frames": len(tail),
        "time_first_ps": t0,
        "time_last_ps": t1,
        "tail_start_ps": start,
        "density_last10ns_kg_m3": mean_kgm3,
        "density_last10ns_gcc": mean_kgm3 / 1000.0,
    }}))
PY
rm -f /tmp/o7_energy_$$.out /tmp/o7_energy_$$.err
"""


def extract_density(remote_dir: str, stage: str, *, source_host: str = NEW_HOST) -> dict[str, Any]:
    out_name = f"{stage}_density.xvg"
    result = ssh_host(source_host, extract_density_script(remote_dir, stage, out_name, source_host=source_host), check=False, timeout=180)
    payload: dict[str, Any] = {"ok": False, "reason": "no_json", "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
    payload["returncode"] = result.returncode
    payload["stderr"] = result.stderr.strip()
    return payload


def density_record_key(row: dict[str, Any], stage: str | None = None) -> tuple[str, str, str, str, str]:
    """Identify density evidence by clean attempt, not only canonical case."""
    return (
        str(row.get("case_id", "")),
        str(row.get("clean_case_id", "")),
        str(row.get("tag", "")),
        str(row.get("remote_run_dir", "")),
        str(stage if stage is not None else row.get("stage", "")),
    )


def merge_existing_density(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in read_csv(path):
        out[density_record_key(row)] = row
    return out


def write_clean_postprocess_inputs(
    output_dir: Path,
    ready_full: list[dict[str, Any]],
    force_clean_case_ids: set[str] | None = None,
) -> dict[str, str]:
    dashboard_dir = output_dir / "clean_runtime_dashboard"
    config_path = output_dir / "o7clean_config.json"
    orchestrator_dir = output_dir / "postprocess_orchestrator"
    runtime_rows: list[dict[str, Any]] = []
    source_hosts = {
        row_source_host(row)
        for row in ready_full
        if row_source_host(row)
    }
    source_host_filter = source_host_filter_for_hosts(source_hosts)
    for idx, row in enumerate(sorted(ready_full, key=lambda item: item["clean_case_id"]), start=1):
        source_host = row_source_host(row)
        postprocess_status = clean_postprocess_statuses(str(row["clean_case_id"]))
        if force_clean_case_ids and row["clean_case_id"] in force_clean_case_ids:
            # Force-refresh only the evidence domain used by the strict
            # clean-MSD publication gate.  Do not mask already collected
            # dynamics/contact evidence, because those observables are not
            # needed to decide whether the bounded MSD extension can close.
            postprocess_status["status_MSD"] = "MISSING_FORCE_REFRESH"
        runtime_rows.append(
            {
                "case_seq": idx,
                "case_id": row["clean_case_id"],
                "source_case_id": row["case_id"],
                "strict_done": "true",
                "deployed_hosts": source_host,
                "deployed_paths": row["remote_run_dir"],
                "duplicate_reason": "",
                **postprocess_status,
            }
        )
    write_csv(
        dashboard_dir / "runtime_cases.csv",
        runtime_rows,
        [
            "case_seq",
            "case_id",
            "source_case_id",
            "strict_done",
            "deployed_hosts",
            "deployed_paths",
            "duplicate_reason",
            "status_density",
            "status_Rg",
            "status_RDF",
            "status_MSD",
            "status_shape",
            "status_dynamics",
            "status_fsqt_tau_alpha",
            "status_contact",
        ],
    )
    config = {
        "project_id": "o7clean",
        "manifest_csv": "generated_code/emc_alpha_olefin_builder/output/screen_pelcb100/manifest.csv",
        "case_roots": [{"label": "screen_pelcb100", "path": "generated_code/emc_alpha_olefin_builder/output/screen_pelcb100"}],
        "bridge_roots": [],
        "integrated_root": "generated_code/md_analysis/output/o7clean_integrated",
        "runtime_output_dir": str(dashboard_dir).replace("\\", "/"),
        "supervisor_output_dir": "generated_code/md_analysis/output/o7clean_supervisor",
        "emc_output_dir": "generated_code/md_analysis/output/o7clean_emc_orchestrator",
        "remote_md_roots": {
            "new": ["${PELCB100_REMOTE_HOME}/pelcb_clean_engine_runs"],
            "newnew": [],
            "old": ["${PELCB100_REMOTE_ROOT}/pelcb_clean_engine_runs"],
        },
        "old_server_postprocess_root": "${PELCB100_REMOTE_ROOT}/o7clean_postprocess",
        "old_server_input_root": "${PELCB100_REMOTE_ROOT}/o7clean_postprocess/input",
        "old_server": "${REMOTE_USER}@private-3f2a50db.invalid",
        "old_emc_root": "${PELCB100_REMOTE_ROOT}/tools/emc",
        "emc_remote_root": "${PELCB100_REMOTE_ROOT}/lz_project_runs/emc_alpha_olefin_builder/o7clean",
        "new_server": NEW_HOST,
        "newnew_server": "${REMOTE_USER}@private-169fa2c7.invalid",
        "new_gmx": GMX_2024_CUDA,
        "newnew_gmx": "${PELCB100_REMOTE_ROOT}/miniforge3/envs/md/bin/gmx",
        "disabled_md_hosts": ["newnew"],
    }
    write_json(config_path, config)
    command_args = [
        sys.executable,
        str(SCRIPT_DIR / "lcb300_orchestrator.py"),
        "--config",
        str(config_path),
        "--dashboard-dir",
        str(dashboard_dir),
        "--output-dir",
        str(orchestrator_dir),
        "--source-host",
        source_host_filter,
        "--execute",
    ]
    command = subprocess.list2cmdline(command_args)
    write_text(output_dir / "submit_o7clean_full_postprocess.ps1", command + "\n")
    write_text(output_dir / "submit_o7clean_full_postprocess.sh", command + "\n")
    return {
        "dashboard_dir": str(dashboard_dir),
        "runtime_cases_csv": str(dashboard_dir / "runtime_cases.csv"),
        "config_json": str(config_path),
        "orchestrator_output_dir": str(orchestrator_dir),
        "submit_command": command,
        "submit_command_args": command_args,
    }


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


BAD_TERMINAL_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
OK_TERMINAL_STATES = {"COMPLETED"}


def _has_any_file(candidates: list[Path]) -> bool:
    return any(path.exists() and path.stat().st_size > 0 for path in candidates)


def _summary_status_ok(candidates: list[Path]) -> bool:
    for path in candidates:
        if not (path.exists() and path.stat().st_size > 0):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return True
        status = str(payload.get("status", "")).strip().lower()
        if status in {"", "ok", "pass", "completed"}:
            return True
    return False


def clean_postprocess_statuses(clean_case_id: str, output_root: Path | None = None) -> dict[str, str]:
    """Return per-domain availability for clean postprocess scheduling.

    The closure scheduler may be re-entered to fill a single missing domain
    such as Fs(q,t).  Using file-level availability avoids resubmitting RDF,
    density, shape and MSD suites when only the dynamics sidecar is absent.
    """
    clean_case_id = str(clean_case_id or "").strip()
    root = output_root or (SCRIPT_DIR / "output")
    if not clean_case_id:
        return {
            "status_density": "MISSING",
            "status_Rg": "MISSING",
            "status_RDF": "MISSING",
            "status_MSD": "MISSING",
            "status_shape": "MISSING",
            "status_dynamics": "MISSING",
            "status_fsqt_tau_alpha": "MISSING",
            "status_contact": "MISSING",
        }

    density_ok = _has_any_file([root / "o7clean_density_energy" / clean_case_id / "analysis.json"])
    rg_ok = _has_any_file(list(root.glob(f"o7clean_rg_*/cases/{clean_case_id}/rg/rg_summary.json")))
    msd_ok = _summary_status_ok(list(root.glob(f"o7clean_msd_*/cases/{clean_case_id}/msd/msd_summary.json")))
    rdf_ok = _has_any_file(
        [
            root
            / "o7clean_integrated"
            / "RDF"
            / "raw"
            / "cases"
            / clean_case_id
            / "rdf"
            / "rdf_summary.json"
        ]
    )
    shape_ok = _has_any_file(
        [
            root
            / "o7clean_integrated"
            / "ShapeTensor"
            / "raw"
            / "cases"
            / clean_case_id
            / "shape_tensor"
            / "shape_summary.json"
        ]
    )
    contact_ok = _has_any_file(
        [
            root
            / "o7clean_integrated"
            / "IntrachainDynamics"
            / "cases"
            / clean_case_id
            / "intrachain"
            / "contact_probability_summary.json"
        ]
    )
    intrachain_required = [
        root
        / "o7clean_integrated"
        / "IntrachainDynamics"
        / "cases"
        / clean_case_id
        / "intrachain"
        / "g1_monomer_summary.json",
        root
        / "o7clean_integrated"
        / "IntrachainDynamics"
        / "cases"
        / clean_case_id
        / "intrachain"
        / "g1_g3_summary.json",
    ]
    intrachain_ok = all(_summary_status_ok([path]) for path in intrachain_required)
    fsqt_ok = _summary_status_ok(list(root.glob(f"o7clean_dynamics_*/cases/{clean_case_id}/dynamics/fsqt_summary.json")))
    return {
        "status_density": "OK" if density_ok else "MISSING",
        "status_Rg": "OK" if rg_ok else "MISSING",
        "status_RDF": "OK" if rdf_ok else "MISSING",
        "status_MSD": "OK" if msd_ok else "MISSING",
        "status_shape": "OK" if shape_ok else "MISSING",
        "status_dynamics": "OK" if intrachain_ok else "MISSING",
        "status_fsqt_tau_alpha": "OK" if fsqt_ok else "MISSING",
        "status_contact": "OK" if contact_ok else "MISSING",
    }


def full_postprocess_collected(clean_case_id: str, output_root: Path | None = None) -> bool:
    clean_case_id = str(clean_case_id or "").strip()
    if not clean_case_id:
        return False
    statuses = clean_postprocess_statuses(clean_case_id, output_root=output_root)
    return bool(statuses) and all(str(value).upper() == "OK" for value in statuses.values())


def active_case_ids_from_merge_plan(plan_csv: Path) -> set[str]:
    """Return clean case IDs whose source trajectories are still running."""

    rows = read_csv(plan_csv)
    query_rows: list[dict[str, Any]] = []
    case_by_remote: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        source_path = str(row.get("source_path", "")).strip()
        source_host = str(row.get("source_host", "")).strip()
        if not case_id or not source_path:
            continue
        query_rows.append(
            {
                "case_id": case_id,
                "clean_case_id": case_id,
                "remote_run_dir": source_path,
                "source_host": source_host,
            }
        )
        case_by_remote[source_path] = case_id
    if not query_rows:
        return set()
    active_cases: set[str] = set()
    for remote, status in fetch_remote_status(query_rows).items():
        if truthy_bool(status.get("nvt_prod_active")):
            case_id = case_by_remote.get(remote, "")
            if case_id:
                active_cases.add(case_id)
    return active_cases


def filter_plan_csv_excluding_cases(
    plan_csv: Path,
    output_dir: Path,
    excluded_cases: set[str],
    reason: str,
    excluded_sequences: set[str] | None = None,
) -> Path:
    excluded = {str(case_id).strip() for case_id in excluded_cases if str(case_id).strip()}
    excluded_seq = {str(seq).strip() for seq in (excluded_sequences or set()) if str(seq).strip()}
    if not excluded and not excluded_seq:
        return plan_csv
    with plan_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if "case_id" not in fieldnames:
        return plan_csv
    kept_rows = [
        row
        for row in rows
        if str(row.get("case_id", "")).strip() not in excluded
        and not case_has_excluded_sequence(str(row.get("case_id", "")), excluded_seq)
    ]
    if len(kept_rows) == len(rows):
        return plan_csv
    target_dir = output_dir / "postprocess_closure"
    target_dir.mkdir(parents=True, exist_ok=True)
    filtered = target_dir / f"active_protected_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    write_csv(filtered, kept_rows, fieldnames)
    write_json(
        filtered.with_suffix(".json"),
        {
            "source_plan_csv": str(plan_csv),
            "filtered_plan_csv": str(filtered),
            "excluded_cases": sorted(excluded),
            "excluded_case_sequences": sorted(excluded_seq),
            "reason": reason,
        },
    )
    return filtered


def postprocess_job_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    jobs = [str(job).strip() for job in state.get("job_ids", []) if str(job).strip()]
    if not jobs:
        return {"job_ids": [], "active_job_ids": [], "failed_job_ids": [], "completed_job_ids": [], "rows": []}
    cmd = "squeue -h -j " + ",".join(jobs) + " -o '%i %T' 2>/dev/null || true"
    result = run_cmd(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", "${REMOTE_USER}@private-3f2a50db.invalid", cmd],
        check=False,
        timeout=60,
    )
    active_rows = [line.strip().split(maxsplit=1) for line in result.stdout.splitlines() if line.strip()]
    active_job_ids = [row[0] for row in active_rows if row]

    sacct_cmd = (
        "sacct -j "
        + ",".join(jobs)
        + " -n -X --format=JobID,State,ExitCode -P 2>/dev/null || true"
    )
    sacct = run_cmd(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=20",
            "${REMOTE_USER}@private-3f2a50db.invalid",
            sacct_cmd,
        ],
        check=False,
        timeout=60,
    )
    rows: list[dict[str, str]] = []
    failed_job_ids: list[str] = []
    completed_job_ids: list[str] = []
    for line in sacct.stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2 or not parts[0]:
            continue
        state_text = (parts[1].split("+", 1)[0].strip().split() or [""])[0].upper()
        row = {"job_id": parts[0], "state": state_text, "exit_code": parts[2] if len(parts) > 2 else ""}
        rows.append(row)
        if state_text in BAD_TERMINAL_STATES:
            failed_job_ids.append(parts[0])
        elif state_text in OK_TERMINAL_STATES:
            completed_job_ids.append(parts[0])
    return {
        "job_ids": jobs,
        "active_job_ids": active_job_ids,
        "failed_job_ids": sorted(set(failed_job_ids)),
        "completed_job_ids": sorted(set(completed_job_ids)),
        "rows": rows,
    }


def live_o7_postprocess_jobs() -> list[str]:
    """Return live O7 postprocessing Slurm job IDs by job name.

    The state file can briefly lag behind a just-submitted Slurm job.  A live
    queue probe prevents duplicate postprocess submissions when a retry is
    triggered by an older failed job record while a fresh retry is already
    running.
    """
    cmd = "squeue -h -u ${REMOTE_USER} -o '%i|%j|%T' 2>/dev/null || true"
    result = run_cmd(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=20",
            "${REMOTE_USER}@private-3f2a50db.invalid",
            cmd,
        ],
        check=False,
        timeout=60,
    )
    active: list[str] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2:
            continue
        job_id, job_name = parts[0], parts[1]
        if job_name in {"o7clean_fast_bundle", "o7clean_deep_rdf"} or job_name.startswith("o7clean_"):
            if job_id:
                active.append(job_id)
    return sorted(dict.fromkeys(active))


def postprocess_jobs_active(state: dict[str, Any]) -> bool:
    return bool(postprocess_job_snapshot(state).get("active_job_ids")) or bool(live_o7_postprocess_jobs())


def postprocess_active_job_ids(state: dict[str, Any]) -> list[str]:
    snapshot = postprocess_job_snapshot(state)
    ids = [str(job_id) for job_id in snapshot.get("active_job_ids", []) if str(job_id).strip()]
    ids.extend(live_o7_postprocess_jobs())
    return sorted(dict.fromkeys(ids))



def parse_job_ids(text: str) -> list[str]:
    matches = re.findall(r"(?im)^\s*Submitted batch job\s+(\d+)\s*$", text or "")
    if not matches:
        matches = re.findall(r"(?m)^\s*(\d{4,})\s*$", text or "")
    return list(dict.fromkeys(matches))


def latest_file(root: Path, pattern: str) -> Path | None:
    candidates = [path for path in root.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_directory(root: Path, pattern: str) -> Path | None:
    candidates = [path for path in root.glob(pattern) if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def command_result(command: list[str], *, timeout: int = 3600) -> dict[str, Any]:
    result = run_cmd(command, check=False, timeout=timeout)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip()[-4000:],
        "stderr_tail": result.stderr.strip()[-4000:],
    }


def publish_batch_to_default_closure(output_dir: Path) -> dict[str, str]:
    if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        return {"status": "default_output"}
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            "o7_clean_run_status.csv",
            ("case_id", "clean_case_id", "remote_run_dir"),
        ),
        (
            "o7_clean_full_postprocess_ready.csv",
            ("case_id", "clean_case_id", "remote_run_dir"),
        ),
        (
            "o7_clean_density_summary.csv",
            ("case_id", "clean_case_id", "tag", "remote_run_dir", "stage"),
        ),
    ]
    published: dict[str, str] = {}
    for filename, key_fields in specs:
        source = output_dir / filename
        if not source.exists():
            continue
        current = read_csv(source)
        target = DEFAULT_OUTPUT_DIR / filename
        fields = list(current[0].keys()) if current else []
        if not fields:
            continue
        merged = merge_rows_by_key(read_csv(target), current, key_fields)
        write_csv(target, merged, fields)
        published[filename] = str(target)
    return published


def auto_finalize_postprocess(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.output_dir / "o7_clean_postprocess_state.json"
    state = load_json(state_path, {})
    jobs = [str(job) for job in state.get("job_ids", []) if str(job).strip()]
    if not jobs:
        return {"action": "wait", "reason": "no_postprocess_jobs"}
    submitted_this_cycle = {
        str(case_id).strip()
        for case_id in state.get("submitted_this_cycle", [])
        if str(case_id).strip()
    }
    finalized_cases = {
        str(case_id).strip()
        for case_id in state.get("finalized_clean_case_ids", [])
        if str(case_id).strip()
    }
    finalized_job_ids = {
        str(job_id).strip()
        for job_id in state.get("finalized_job_ids", [])
        if str(job_id).strip()
    }
    if submitted_this_cycle and submitted_this_cycle.issubset(finalized_cases) and set(jobs).issubset(finalized_job_ids):
        return {"action": "wait", "reason": "postprocess_already_finalized"}

    snapshot = postprocess_job_snapshot(state)
    live_active_job_ids = postprocess_active_job_ids(state)
    if live_active_job_ids:
        return {
            "action": "wait",
            "reason": "postprocess_jobs_active",
            "active_job_ids": live_active_job_ids,
        }
    failed_job_ids = sorted(set(str(job_id) for job_id in snapshot.get("failed_job_ids", [])))
    completed_roots = {
        re.split(r"[_.]", str(job_id), maxsplit=1)[0]
        for job_id in snapshot.get("completed_job_ids", [])
    }
    terminal_roots = completed_roots | {
        re.split(r"[_.]", str(job_id), maxsplit=1)[0]
        for job_id in failed_job_ids
    }
    if not set(jobs).issubset(terminal_roots):
        return {
            "action": "wait",
            "reason": "postprocess_accounting_pending",
            "completed_job_ids": snapshot.get("completed_job_ids", []),
            "failed_job_ids": failed_job_ids,
        }

    config_path = args.output_dir / "o7clean_config.json"
    orchestrator_dir = args.output_dir / "postprocess_orchestrator"
    plan_csv = latest_file(orchestrator_dir, "*/full_postprocess_plan.csv")
    if not config_path.exists() or plan_csv is None:
        return {
            "action": "blocked",
            "reason": "missing_finalize_inputs",
            "config_json": str(config_path),
            "plan_csv": str(plan_csv or ""),
        }

    steps: list[dict[str, Any]] = []
    if failed_job_ids:
        steps.append(
            {
                "step": "slurm_failed_jobs_present",
                "failed_job_ids": failed_job_ids,
                "policy": "collect_available_outputs_and_let_release_gate_mark_missing_evidence",
            }
        )
    collect = command_result(
        [
            sys.executable,
            str(ORCHESTRATOR),
            "--config",
            str(config_path),
            "--output-dir",
            str(orchestrator_dir),
            "--collect",
        ]
    )
    steps.append({"step": "collect", **collect})
    if collect["returncode"] != 0:
        return {"action": "failed", "reason": "collect_failed", "steps": steps}
    collect_pointer = load_json(orchestrator_dir / "remote_collect_latest.json", {})
    collect_dir_text = str(collect_pointer.get("collect_dir", "")).strip()
    collect_dir = Path(collect_dir_text) if collect_dir_text else latest_directory(orchestrator_dir, "remote_collect_*")
    if collect_dir is None:
        return {"action": "failed", "reason": "collect_dir_missing", "steps": steps}

    closure_dir = args.output_dir / "postprocess_closure"
    merge_plan_csv = augment_plan_csv_with_manual_cases(plan_csv, args.output_dir, state)
    if merge_plan_csv != plan_csv:
        steps.append(
            {
                "step": "augment_merge_plan",
                "source_plan_csv": str(plan_csv),
                "augmented_plan_csv": str(merge_plan_csv),
                "manual_contact_gap_cases": state.get("manual_contact_gap_cases", []),
            }
        )
    active_merge_cases = active_case_ids_from_merge_plan(merge_plan_csv)
    active_merge_sequences = {case_sequence(case_id) for case_id in active_merge_cases if case_sequence(case_id)}
    active_merge_sequences |= active_case_sequences_from_runtime_dashboard()
    filtered_merge_plan_csv = filter_plan_csv_excluding_cases(
        merge_plan_csv,
        args.output_dir,
        active_merge_cases,
        "source_nvt_prod_active",
        excluded_sequences=active_merge_sequences,
    )
    if filtered_merge_plan_csv != merge_plan_csv:
        steps.append(
            {
                "step": "filter_active_source_cases",
                "source_plan_csv": str(merge_plan_csv),
                "filtered_plan_csv": str(filtered_merge_plan_csv),
                "active_cases": sorted(active_merge_cases),
                "active_case_sequences": sorted(active_merge_sequences),
            }
        )
        merge_plan_csv = filtered_merge_plan_csv
    merge = command_result(
        [
            sys.executable,
            str(POSTPROCESS_CLOSURE),
            "--config",
            str(config_path),
            "--output-dir",
            str(closure_dir),
            "--collect-dir",
            str(collect_dir),
            "--merge-plan",
            "--apply-merge",
            "--plan-csv",
            str(merge_plan_csv),
        ]
    )
    steps.append({"step": "apply_merge", **merge})
    if merge["returncode"] != 0:
        return {"action": "failed", "reason": "apply_merge_failed", "steps": steps}

    publish = publish_batch_to_default_closure(args.output_dir)
    regate = command_result([sys.executable, str(CLEAN_REGATE)], timeout=1800)
    steps.append({"step": "clean_regate", **regate})
    if regate["returncode"] != 0:
        return {"action": "failed", "reason": "clean_regate_failed", "steps": steps, "published": publish}

    repair_tag = "auto_after_clean_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    repair_execute = command_result(
        [
            sys.executable,
            str(REPAIR_LOOP),
            "--tag",
            repair_tag + "_msd_followup",
            "--include-optional-validation",
            "--execute",
            "--allowlist-tag",
            "PELCB_REPAIR_LOOP_EXECUTE",
            "--reason",
            "Automatically run strict clean-MSD refit after clean-engine re-gate",
            "--run-strict-msd-refit",
            "--max-execute-msd-window",
            "0",
            "--no-auto-run-ready-closure",
        ],
        timeout=1800,
    )
    steps.append({"step": "execute_msd_followup", **repair_execute})
    if repair_execute["returncode"] != 0:
        return {"action": "failed", "reason": "msd_followup_failed", "steps": steps, "published": publish}

    regate_after_msd = command_result([sys.executable, str(CLEAN_REGATE)], timeout=1800)
    steps.append({"step": "clean_regate_after_msd", **regate_after_msd})
    if regate_after_msd["returncode"] != 0:
        return {"action": "failed", "reason": "post_msd_regate_failed", "steps": steps, "published": publish}

    repair_refresh = command_result(
        [
            sys.executable,
            str(REPAIR_LOOP),
            "--tag",
            repair_tag + "_final",
            "--include-optional-validation",
        ],
        timeout=1800,
    )
    steps.append({"step": "refresh_repair_plan", **repair_refresh})
    if repair_refresh["returncode"] != 0:
        return {"action": "failed", "reason": "repair_plan_refresh_failed", "steps": steps, "published": publish}

    final_repair_plan = REPAIR_LOOP_OUTPUT_DIR / (repair_tag + "_final") / "repair_loop_plan.csv"
    pass_stop = command_result(
        [
            sys.executable,
            str(PASS_STOPPER),
            "--repair-plan",
            str(final_repair_plan),
            "--tag",
            repair_tag + "_pass_stop",
            "--reason",
            "Stop redundant clean NVT after publication-grade PASS and freeze its release evidence",
            "--execute",
            "--allowlist-tag",
            "PELCB_STOP_PASSED_ACTIVE",
        ],
        timeout=900,
    )
    steps.append({"step": "stop_publication_passed_active", **pass_stop})
    if pass_stop["returncode"] != 0:
        return {"action": "failed", "reason": "publication_pass_stop_failed", "steps": steps, "published": publish}

    # Collection scans the shared remote result root, so older cases held in
    # pending_collection are also examined during this finalize pass.  Mark
    # that collection attempt explicitly: if their evidence is still
    # incomplete, auto_submit_postprocess may retry them on the next cycle
    # instead of leaving them pending forever.
    pending_collection_cases = {
        str(case_id).strip()
        for case_id in state.get("pending_collection_case_ids", [])
        if str(case_id).strip()
    }
    state["finalized_at"] = iso_now()
    finalized_this_cycle = {
        case_id
        for case_id in submitted_this_cycle
        if case_id not in active_merge_cases and not case_has_excluded_sequence(case_id, active_merge_sequences)
    }
    submitted_clean_case_ids = {
        str(case_id).strip()
        for case_id in state.get("submitted_clean_case_ids", [])
        if str(case_id).strip()
    } - active_merge_cases
    submitted_clean_case_ids = {
        case_id for case_id in submitted_clean_case_ids if not case_has_excluded_sequence(case_id, active_merge_sequences)
    }
    invalidated_sequences = {
        str(seq).strip()
        for seq in state.get("invalidated_active_case_sequences", [])
        if str(seq).strip()
    }
    invalidated_sequences |= active_merge_sequences
    invalidated_sequences -= {case_sequence(case_id) for case_id in finalized_this_cycle if case_sequence(case_id)}
    state["finalized_clean_case_ids"] = sorted(
        finalized_cases | finalized_this_cycle | pending_collection_cases
    )
    state["finalized_job_ids"] = sorted(set(jobs) | finalized_job_ids)
    state["submitted_clean_case_ids"] = sorted(submitted_clean_case_ids)
    state["submitted_this_cycle"] = sorted(finalized_this_cycle)
    state["invalidated_active_case_sequences"] = sorted(invalidated_sequences)
    if active_merge_cases:
        state["active_source_cases_excluded_from_finalize"] = sorted(active_merge_cases)
    if active_merge_sequences:
        state["active_source_case_sequences_excluded_from_finalize"] = sorted(active_merge_sequences)
    state["finalize_collect_dir"] = str(collect_dir)
    state["finalize_plan_csv"] = str(plan_csv)
    state["finalize_steps"] = steps
    write_json(state_path, state)
    return {
        "action": "finalized",
        "clean_case_ids": sorted(finalized_this_cycle),
        "active_source_cases_excluded": sorted(active_merge_cases),
        "active_source_case_sequences_excluded": sorted(active_merge_sequences),
        "collect_dir": str(collect_dir),
        "plan_csv": str(plan_csv),
        "published": publish,
        "steps": steps,
    }


def auto_submit_postprocess(args: argparse.Namespace, ready_full: list[dict[str, Any]], postprocess_inputs: dict[str, str]) -> dict[str, Any]:
    state_path = args.output_dir / "o7_clean_postprocess_state.json"
    state = load_json(state_path, {})
    active_sequences = active_case_sequences_from_runtime_dashboard()
    if active_sequences:
        invalidated_sequences = {
            str(seq).strip()
            for seq in state.get("invalidated_active_case_sequences", [])
            if str(seq).strip()
        }
        invalidated_sequences |= active_sequences
        state["invalidated_active_case_sequences"] = sorted(invalidated_sequences)
        state["active_case_sequences_excluded_from_submit"] = sorted(active_sequences)
        write_json(state_path, state)
        ready_full = [
            row
            for row in ready_full
            if not case_has_excluded_sequence(str(row.get("clean_case_id") or row.get("case_id", "")), active_sequences)
        ]
    if not ready_full:
        reason = "active_clean_md_cases_protected" if active_sequences else "no_nvt_prod_done"
        return {"action": "wait", "reason": reason, "active_case_sequences": sorted(active_sequences)}
    job_snapshot = postprocess_job_snapshot(state)
    active_job_ids = postprocess_active_job_ids(state)
    failed_job_ids = list(job_snapshot.get("failed_job_ids", []))
    prior_submitted = {str(case_id).strip() for case_id in state.get("submitted_clean_case_ids", []) if str(case_id).strip()}
    finalized_cases = {
        str(case_id).strip()
        for case_id in state.get("finalized_clean_case_ids", [])
        if str(case_id).strip()
    }
    invalidated_sequences = {
        str(seq).strip()
        for seq in state.get("invalidated_active_case_sequences", [])
        if str(seq).strip()
    }
    current_ready = {row["clean_case_id"] for row in ready_full}
    force_requested = {
        str(case_id).strip()
        for case_id in getattr(args, "force_postprocess_cases", []) or []
        if str(case_id).strip()
    }
    force_ready = {
        row["clean_case_id"]
        for row in ready_full
        if row.get("clean_case_id") in force_requested or row.get("case_id") in force_requested
    }
    already_ready = {
        case_id
        for case_id in current_ready
        if not case_has_excluded_sequence(case_id, invalidated_sequences) and full_postprocess_collected(case_id)
    } - force_ready
    submitted_but_incomplete = sorted(prior_submitted - already_ready)
    retry_ready: set[str] = set()
    if failed_job_ids and not active_job_ids:
        retry_ready = {
            str(case_id).strip()
            for case_id in state.get("submitted_this_cycle", [])
            if str(case_id).strip()
        }
        if not retry_ready:
            retry_ready = set(current_ready)
    else:
        retry_ready = (set(submitted_but_incomplete) & finalized_cases & current_ready)
    retry_ready |= force_ready
    if active_job_ids:
        active_cycle_cases = {
            str(case_id).strip()
            for case_id in state.get("submitted_this_cycle", [])
            if str(case_id).strip()
        }
        # Older state can contain cases that were submitted in a previous wave
        # but were never finalized, while the currently active Slurm job belongs
        # to a newer, disjoint wave.  These incomplete non-active cases should
        # be eligible for a new postprocess wave instead of waiting forever.
        retry_ready |= (set(submitted_but_incomplete) - active_cycle_cases) & current_ready
    pending_collection = sorted((current_ready & prior_submitted) - already_ready - retry_ready)
    new_ready = current_ready - prior_submitted - already_ready
    if active_job_ids and not (new_ready or retry_ready):
        return {"action": "wait", "reason": "postprocess_jobs_active", "job_ids": active_job_ids}
    if current_ready and current_ready.issubset(already_ready):
        return {"action": "wait", "reason": "ready_cases_already_submitted", "submitted_clean_case_ids": sorted(already_ready)}
    pending_ids = new_ready | retry_ready
    pending_ready = [row for row in ready_full if row["clean_case_id"] in pending_ids]
    if not pending_ready and pending_collection:
        return {
            "action": "wait",
            "reason": "postprocess_jobs_complete_collection_pending",
            "pending_collection_case_ids": pending_collection,
            "submitted_clean_case_ids": sorted(prior_submitted),
        }
    if not pending_ready:
        return {"action": "wait", "reason": "no_new_postprocess_cases"}
    if force_ready:
        postprocess_inputs = write_clean_postprocess_inputs(
            args.output_dir,
            pending_ready,
            force_clean_case_ids=force_ready,
        )
    elif len(pending_ready) != len(ready_full):
        postprocess_inputs = write_clean_postprocess_inputs(
            args.output_dir,
            pending_ready,
        )
    command_args = postprocess_inputs.get("submit_command_args")
    command = command_args if isinstance(command_args, list) and command_args else postprocess_inputs["submit_command"]
    result = subprocess.run(
        command,
        cwd=Path.cwd(),
        shell=not isinstance(command, list),
        text=True,
        capture_output=True,
        check=False,
    )
    job_ids = parse_job_ids(result.stdout + "\n" + result.stderr)
    submission_ok = result.returncode == 0 and bool(job_ids)
    if not submission_ok:
        failed_state = dict(state)
        failed_state["last_submit_attempt_at"] = iso_now()
        failed_state["last_submit_returncode"] = result.returncode
        failed_state["last_submit_stdout"] = result.stdout.strip()
        failed_state["last_submit_stderr"] = result.stderr.strip()
        failed_state["last_submit_failure_reason"] = "no_slurm_job_ids" if result.returncode == 0 else "command_failed"
        write_json(state_path, failed_state)
        return {
            "action": "submit_failed",
            "returncode": result.returncode,
            "job_ids": job_ids,
            "failure_reason": failed_state["last_submit_failure_reason"],
            "previous_failed_job_ids": failed_job_ids,
            "submitted_but_incomplete_case_ids": submitted_but_incomplete,
            "state_json": str(state_path),
            "submitted_this_cycle": [],
            "pending_collection_case_ids": pending_collection,
            "stdout_tail": result.stdout.strip()[-1000:],
            "stderr_tail": result.stderr.strip()[-1000:],
        }
    submitted_this_cycle = {row["clean_case_id"] for row in pending_ready}
    submitted_sequences = {case_sequence(case_id) for case_id in submitted_this_cycle if case_sequence(case_id)}
    invalidated_sequences -= submitted_sequences
    submitted_clean_case_ids = sorted(prior_submitted | submitted_this_cycle)
    # Keep prior job IDs only while they are still active.  After a failed
    # postprocess wave is retried, stale terminal job IDs must not remain in
    # the live tracking set; otherwise every finalize check keeps querying old
    # Slurm accounting rows and can become unnecessarily slow.
    retained_job_ids = [str(job_id) for job_id in state.get("job_ids", []) if str(job_id).strip()] if active_job_ids else []
    merged_job_ids = sorted({*retained_job_ids, *job_ids})
    new_state = {
        "last_submit_at": iso_now(),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "job_ids": merged_job_ids,
        "submitted_job_ids_this_cycle": job_ids,
        "submitted_clean_case_ids": submitted_clean_case_ids,
        "submitted_this_cycle": sorted(submitted_this_cycle),
        "pending_collection_case_ids": pending_collection,
        "force_postprocess_clean_case_ids": sorted(force_ready),
        "postprocess_runtime_cases_csv": postprocess_inputs["runtime_cases_csv"],
        "postprocess_config_json": postprocess_inputs["config_json"],
        "postprocess_command": postprocess_inputs["submit_command"],
        "postprocess_command_args": command if isinstance(command, list) else [],
    }
    if invalidated_sequences:
        new_state["invalidated_active_case_sequences"] = sorted(invalidated_sequences)
    if finalized_cases:
        new_state["finalized_clean_case_ids"] = sorted(finalized_cases)
    if failed_job_ids:
        new_state["previous_failed_job_ids"] = failed_job_ids
        new_state["retry_reason"] = "previous_postprocess_jobs_failed"
    if submitted_but_incomplete:
        new_state["submitted_but_incomplete_case_ids"] = submitted_but_incomplete
    write_json(state_path, new_state)
    return {
        "action": "submitted" if submission_ok else "submit_failed",
        "returncode": result.returncode,
        "job_ids": job_ids,
        "failure_reason": "" if submission_ok else ("no_slurm_job_ids" if result.returncode == 0 else "command_failed"),
        "previous_failed_job_ids": failed_job_ids,
        "submitted_but_incomplete_case_ids": submitted_but_incomplete,
        "state_json": str(state_path),
        "submitted_this_cycle": sorted(submitted_this_cycle),
        "pending_collection_case_ids": pending_collection,
        "force_postprocess_clean_case_ids": sorted(force_ready),
        "stdout_tail": result.stdout.strip()[-1000:],
        "stderr_tail": result.stderr.strip()[-1000:],
    }


def should_auto_submit_after_finalize(finalize_result: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a submit pass may run after an auto-finalize attempt."""
    action = str(finalize_result.get("action", "") or "")
    reason = str(finalize_result.get("reason", "") or "")
    if action == "disabled":
        return True, "finalize_disabled"
    if action == "finalized":
        return False, "postprocess_finalized_this_cycle"
    if action == "failed":
        return False, f"postprocess_finalize_failed:{reason}"
    if reason in {"postprocess_jobs_active", "postprocess_accounting_pending"}:
        return False, reason
    return True, reason or action or "finalize_allows_submit"


def augment_plan_csv_with_manual_cases(plan_csv: Path, output_dir: Path, state: dict[str, Any]) -> Path:
    """Return a merge-plan CSV that also admits closure-recovery cases.

    Remote collection scans a shared result root.  A case left in
    ``pending_collection_case_ids`` can therefore be present in the collected
    archive even when it was not part of the newest submission wave.  The
    merge layer only needs the case ID as its allowlist, so include those
    recovery cases alongside explicitly submitted manual gaps.
    """
    manual_cases = [
        str(case_id).strip()
        for case_id in state.get("manual_contact_gap_cases", [])
        if str(case_id).strip()
    ]
    pending_collection_cases = [
        str(case_id).strip()
        for case_id in state.get("pending_collection_case_ids", [])
        if str(case_id).strip()
    ]
    recovery_cases = list(dict.fromkeys([*manual_cases, *pending_collection_cases]))
    if not recovery_cases:
        return plan_csv
    with plan_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if "case_id" not in fieldnames:
        return plan_csv
    existing = {str(row.get("case_id", "")).strip() for row in rows}
    added_cases = [case_id for case_id in recovery_cases if case_id not in existing]
    if not added_cases:
        return plan_csv
    defaults = {
        "observable": "contact",
        "suite": "deep_contact",
        "source_host": "",
        "source_path": "",
        "dest_path": "",
        "status_column": "status_contact",
    }
    for case_id in added_cases:
        row = {field: "" for field in fieldnames}
        row["case_id"] = case_id
        for field, value in defaults.items():
            if field in row:
                row[field] = value
        rows.append(row)
    target_dir = output_dir / "postprocess_closure"
    target_dir.mkdir(parents=True, exist_ok=True)
    augmented = target_dir / f"augmented_manual_gap_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    write_csv(augmented, rows, fieldnames)
    write_json(
        augmented.with_suffix(".json"),
        {
            "source_plan_csv": str(plan_csv),
            "augmented_plan_csv": str(augmented),
            "manual_contact_gap_cases": manual_cases,
            "pending_collection_case_ids": pending_collection_cases,
            "added_cases": added_cases,
        },
    )
    return augmented


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    # Finalizing postprocess output does not require a fresh scan of every
    # clean-MD run directory.  That scan can be slow while long extension jobs
    # are active, and it should not delay the lightweight "are Slurm
    # postprocess jobs done yet?" check.
    if (
        getattr(args, "auto_finalize_postprocess", False)
        and not getattr(args, "auto_submit_postprocess", False)
        and not args.tags
        and not args.cases
    ):
        timestamp = iso_now()
        postprocess_finalize = auto_finalize_postprocess(args)
        summary = {
            "scan_timestamp": timestamp,
            "runs": 0,
            "pr_done": 0,
            "nvt_prod_done": 0,
            "density_rows": len(read_csv(args.output_dir / "o7_clean_density_summary.csv")),
            "density_ok": sum(
                1
                for row in read_csv(args.output_dir / "o7_clean_density_summary.csv")
                if str(row.get("ok", "")).lower() == "true"
            ),
            "full_postprocess_ready": len(read_csv(args.output_dir / "o7_clean_full_postprocess_ready.csv")),
            "scan_postprocess_ready": 0,
            "status_csv": str(args.output_dir / "o7_clean_run_status.csv"),
            "density_csv": str(args.output_dir / "o7_clean_density_summary.csv"),
            "postprocess_ready_csv": str(args.output_dir / "o7_clean_full_postprocess_ready.csv"),
            "postprocess_runtime_cases_csv": "",
            "postprocess_config_json": "",
            "postprocess_submit_command": "",
            "postprocess_auto": {"action": "disabled"},
            "postprocess_finalize": postprocess_finalize,
            "published_to_default": {},
            "fast_path": "auto_finalize_postprocess",
        }
        write_json(args.output_dir / "o7_clean_closure_summary.json", summary)
        return summary

    rows = load_plan_rows(args.queue_dir, args.tags, args.cases)
    force_case_ids = {
        str(case_id).strip()
        for case_id in getattr(args, "force_postprocess_cases", []) or []
        if str(case_id).strip()
    }
    extension_sources = latest_nvt_extension_sources(force_case_ids)
    rows = apply_nvt_extension_source_overrides(rows, extension_sources)
    status = fetch_remote_status(rows)
    timestamp = iso_now()

    status_rows: list[dict[str, Any]] = []
    ready_full: list[dict[str, Any]] = []
    for row in rows:
        st = status.get(row["remote_run_dir"], {})
        merged = {
            "case_id": row.get("case_id", ""),
            "clean_case_id": row.get("clean_case_id", ""),
            "tag": row.get("tag", ""),
            "plan_kind": row.get("plan_kind", ""),
            "source_host_label": row_source_host_label(row),
            "source_host": row_source_host(row),
            "node": row.get("node", ""),
            "gpu_id": row.get("gpu_id", ""),
            "threads": row.get("threads", ""),
            "remote_run_dir": row.get("remote_run_dir", ""),
            "extension_source_override": row.get("extension_source_override", ""),
            "extension_plan_csv": row.get("extension_plan_csv", ""),
            "scan_timestamp": timestamp,
            "exists": st.get("exists", False),
        }
        for stage in STAGES:
            merged[f"{stage}_done"] = st.get(f"{stage}_done", False)
            merged[f"{stage}_has_tpr"] = st.get(f"{stage}_has_tpr", False)
            merged[f"{stage}_has_cpt"] = st.get(f"{stage}_has_cpt", False)
            merged[f"{stage}_has_edr"] = st.get(f"{stage}_has_edr", False)
            merged[f"{stage}_fatal_log"] = st.get(f"{stage}_fatal_log", "")
        merged["nvt_prod_active"] = st.get("nvt_prod_active", False)
        eligible, skip_reason = is_full_postprocess_eligible(merged)
        if truthy_bool(merged.get("nvt_prod_active")):
            eligible = False
            skip_reason = "nvt_prod_active"
        merged["full_postprocess_eligible"] = eligible
        merged["ready_skip_reason"] = skip_reason
        status_rows.append(merged)
        if merged["nvt_prod_done"] and eligible:
            ready_full.append(
                {
                    "case_id": merged["case_id"],
                    "clean_case_id": merged["clean_case_id"],
                    "tag": merged["tag"],
                    "remote_run_dir": merged["remote_run_dir"],
                    "source_host_label": merged["source_host_label"],
                    "source_host": merged["source_host"],
                    "postprocess_namespace": "o7_clean_cuda2024",
                    "ready_reason": (
                        "nvt_prod_finished_after_extension_override"
                        if truthy_bool(merged.get("extension_source_override"))
                        else "nvt_prod_finished"
                    ),
                }
            )

    status_fields = [
        "case_id",
        "clean_case_id",
        "tag",
        "plan_kind",
        "source_host_label",
        "source_host",
        "node",
        "gpu_id",
        "threads",
        "remote_run_dir",
        "extension_source_override",
        "extension_plan_csv",
        "scan_timestamp",
        "exists",
        "nvt_prod_active",
        "full_postprocess_eligible",
        "ready_skip_reason",
    ] + [f"{stage}_{suffix}" for stage in STAGES for suffix in ("done", "has_tpr", "has_cpt", "has_edr", "fatal_log")]
    status_path = args.output_dir / "o7_clean_run_status.csv"
    status_rows_merged = merge_rows_by_key(
        read_csv(status_path),
        status_rows,
        ("case_id", "clean_case_id", "remote_run_dir"),
    )
    status_rows_merged = sorted(
        status_rows_merged,
        key=lambda row: (str(row.get("tag", "")), str(row.get("case_id", "")), str(row.get("clean_case_id", ""))),
    )
    write_csv(status_path, status_rows_merged, status_fields)
    ready_path = args.output_dir / "o7_clean_full_postprocess_ready.csv"
    scanned_remote_dirs = {str(row.get("remote_run_dir", "")) for row in status_rows}
    existing_ready = [
        row
        for row in read_csv(ready_path)
        if str(row.get("remote_run_dir", "")) not in scanned_remote_dirs
    ]
    ready_full_merged = merge_rows_by_key(
        existing_ready,
        ready_full,
        ("case_id", "clean_case_id", "remote_run_dir"),
    )
    ready_full_merged = sorted(
        ready_full_merged,
        key=lambda row: (str(row.get("tag", "")), str(row.get("case_id", "")), str(row.get("clean_case_id", ""))),
    )
    write_csv(
        ready_path,
        ready_full_merged,
        [
            "case_id",
            "clean_case_id",
            "tag",
            "remote_run_dir",
            "source_host_label",
            "source_host",
            "postprocess_namespace",
            "ready_reason",
        ],
    )
    # Preserve the historical ready ledger, but only stage and submit cases
    # observed in this scan. A targeted scan must not rebuild a mixed-host
    # historical submission from unrelated ready rows.
    force_ready_full = {
        row["clean_case_id"]
        for row in ready_full
        if row.get("clean_case_id") in force_case_ids or row.get("case_id") in force_case_ids
    }
    postprocess_inputs = write_clean_postprocess_inputs(
        args.output_dir,
        ready_full,
        force_clean_case_ids=force_ready_full,
    )
    postprocess_finalize = {"action": "disabled"}
    if getattr(args, "auto_finalize_postprocess", False):
        postprocess_finalize = auto_finalize_postprocess(args)
    postprocess_auto = {"action": "disabled"}
    if args.auto_submit_postprocess:
        submit_allowed, submit_skip_reason = should_auto_submit_after_finalize(postprocess_finalize)
        if submit_allowed:
            postprocess_auto = auto_submit_postprocess(args, ready_full, postprocess_inputs)
        else:
            postprocess_auto = {"action": "wait", "reason": submit_skip_reason}

    density_path = args.output_dir / "o7_clean_density_summary.csv"
    density_by_key = merge_existing_density(density_path)
    if args.extract_pr_density or args.extract_nvt_density:
        for row in status_rows:
            for stage, enabled in (("pr", args.extract_pr_density), ("nvt_prod", args.extract_nvt_density)):
                if not enabled or not row.get(f"{stage}_done"):
                    continue
                key = density_record_key(row, stage)
                if key in density_by_key and str(density_by_key[key].get("ok", "")).lower() == "true":
                    continue
                payload = extract_density(str(row["remote_run_dir"]), stage, source_host=row_source_host(row))
                density_by_key[key] = {
                    "case_id": row["case_id"],
                    "clean_case_id": row["clean_case_id"],
                    "tag": row["tag"],
                    "stage": stage,
                    "remote_run_dir": row["remote_run_dir"],
                    "ok": payload.get("ok", False),
                    "reason": payload.get("reason", ""),
                    "density_last10ns_gcc": payload.get("density_last10ns_gcc", ""),
                    "density_last10ns_kg_m3": payload.get("density_last10ns_kg_m3", ""),
                    "time_first_ps": payload.get("time_first_ps", ""),
                    "time_last_ps": payload.get("time_last_ps", ""),
                    "tail_start_ps": payload.get("tail_start_ps", ""),
                    "frames": payload.get("frames", ""),
                    "tail_frames": payload.get("tail_frames", ""),
                    "xvg": payload.get("xvg", ""),
                    "extracted_at": timestamp,
                    "stderr": payload.get("stderr", ""),
                }
    density_rows = sorted(
        density_by_key.values(),
        key=lambda r: (
            str(r.get("case_id", "")),
            str(r.get("stage", "")),
            str(r.get("tag", "")),
            str(r.get("remote_run_dir", "")),
        ),
    )
    write_csv(
        density_path,
        density_rows,
        [
            "case_id",
            "clean_case_id",
            "tag",
            "stage",
            "remote_run_dir",
            "ok",
            "reason",
            "density_last10ns_gcc",
            "density_last10ns_kg_m3",
            "time_first_ps",
            "time_last_ps",
            "tail_start_ps",
            "frames",
            "tail_frames",
            "xvg",
            "extracted_at",
            "stderr",
        ],
    )
    published_to_default = publish_batch_to_default_closure(args.output_dir)
    summary = {
        "scan_timestamp": timestamp,
        "runs": len(status_rows_merged),
        "pr_done": sum(1 for row in status_rows_merged if truthy_bool(row.get("pr_done"))),
        "nvt_prod_done": sum(1 for row in status_rows_merged if truthy_bool(row.get("nvt_prod_done"))),
        "density_rows": len(density_rows),
        "density_ok": sum(1 for row in density_rows if str(row.get("ok", "")).lower() == "true"),
        "full_postprocess_ready": len(ready_full_merged),
        "scan_postprocess_ready": len(ready_full),
        "status_csv": str(status_path),
        "density_csv": str(density_path),
        "postprocess_ready_csv": str(ready_path),
        "postprocess_runtime_cases_csv": postprocess_inputs["runtime_cases_csv"],
        "postprocess_config_json": postprocess_inputs["config_json"],
        "postprocess_submit_command": postprocess_inputs["submit_command"],
        "postprocess_auto": postprocess_auto,
        "postprocess_finalize": postprocess_finalize,
        "force_postprocess_case_ids": sorted(force_case_ids),
        "extension_source_override_count": sum(
            1 for row in status_rows if truthy_bool(row.get("extension_source_override"))
        ),
        "extension_source_override_cases": sorted(
            {
                str(row.get("case_id", ""))
                for row in status_rows
                if truthy_bool(row.get("extension_source_override"))
            }
        ),
        "published_to_default": published_to_default,
    }
    write_json(args.output_dir / "o7_clean_closure_summary.json", summary)
    return summary


def main() -> int:
    configure_windows_ssh_path()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tags", nargs="*", help="Specific sidecar tags to scan. Default scans all tags.")
    parser.add_argument("--cases", nargs="*", help="Specific source or clean case IDs to scan within the selected tags.")
    parser.add_argument(
        "--force-postprocess-cases",
        nargs="*",
        default=[],
        help=(
            "Source or clean case IDs whose clean-only postprocessing should be resubmitted even "
            "if prior collected outputs already exist. Used when repair routing requires refreshed "
            "post-extension evidence."
        ),
    )
    parser.add_argument("--extract-pr-density", action="store_true", default=True)
    parser.add_argument("--no-extract-pr-density", dest="extract_pr_density", action="store_false")
    parser.add_argument("--extract-nvt-density", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Run continuously like a lightweight supervisor.")
    parser.add_argument("--interval-sec", type=int, default=600, help="Watch interval in seconds.")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run forever in watch mode.")
    parser.add_argument(
        "--auto-submit-postprocess",
        action="store_true",
        help="Automatically submit clean-only old-server postprocessing when nvt_prod completes.",
    )
    parser.add_argument(
        "--auto-finalize-postprocess",
        action="store_true",
        help="After Slurm postprocessing completes, collect, apply the batch merge, re-gate, and refresh the repair plan.",
    )
    args = parser.parse_args()

    cycle = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_dir: Path | None = None
    if args.watch:
        lock_dir = acquire_watch_lock(args.output_dir)
    try:
        while True:
            cycle += 1
            write_heartbeat(args.output_dir, "cycle_start", cycle=cycle)
            try:
                summary = run_once(args)
                summary["cycle"] = cycle
                write_heartbeat(
                    args.output_dir,
                    "cycle_done",
                    cycle=cycle,
                    scan_timestamp=summary.get("scan_timestamp", ""),
                    scan_postprocess_ready=summary.get("scan_postprocess_ready", 0),
                    postprocess_action=summary.get("postprocess_auto", {}).get("action", ""),
                    finalize_action=summary.get("postprocess_finalize", {}).get("action", ""),
                )
                print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            except Exception as exc:
                error = {
                    "timestamp": iso_now(),
                    "cycle": cycle,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(args.output_dir / "o7_clean_closure_last_error.json", error)
                write_heartbeat(args.output_dir, "cycle_error", cycle=cycle, error=str(exc))
                print(json.dumps(error, indent=2, ensure_ascii=False), file=sys.stderr, flush=True)
                if not args.watch:
                    raise
            if not args.watch:
                break
            if args.max_cycles and cycle >= args.max_cycles:
                break
            time.sleep(max(10, int(args.interval_sec)))
    finally:
        release_watch_lock(lock_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


