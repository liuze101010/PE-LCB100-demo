"""Read-only LCB300 runtime dashboard.

This script intentionally does not submit jobs, kill processes, migrate cases,
or write to any remote filesystem. It combines the LCB300 design manifest,
current integrated analysis table, and live read-only SSH probes into a compact
runtime dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lcb_project_config import load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_CODE_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = (
    GENERATED_CODE_ROOT
    / "emc_alpha_olefin_builder"
    / "output"
    / "screen_lcb300"
    / "manifest.csv"
)
DEFAULT_INTEGRATED = (
    SCRIPT_DIR
    / "output"
    / "lcb300_integrated"
    / "aggregate"
    / "lcb300_summary_wide.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "lcb300_runtime_dashboard"

NEW_SERVER = "${REMOTE_USER}@private-4e2f23ff.invalid"
NEWNEW_SERVER = "${REMOTE_USER}@private-169fa2c7.invalid"
OLD_SERVER = "${REMOTE_USER}@private-3f2a50db.invalid"
NEW_SERVER_ROOTS = ["${PELCB100_REMOTE_HOME}/lcb_manual_runs"]
NEWNEW_SERVER_ROOTS = ["${PELCB100_REMOTE_ROOT}/lcb_manual_runs"]
OLD_STAGED_ROOTS = ["${PELCB100_REMOTE_ROOT}/lcb_rg_20260501/input"]
NEW_SERVER_NODES = [f"compute-node-{i}" for i in range(1, 9)]
PROJECT_ID = "lcb300"

CASE_PATTERN = (
    r"lcb_pe(?:24cuda|24)_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+|"
    r"lcb\d+_[A-Za-z0-9]+_b\d+_l\d+_c\d+|"
    r"lcb_[a-z]+_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+|"
    r"lcb_pe_lin_\d+_c\d+"
)
CASE_RE = re.compile(f"({CASE_PATTERN})")
CASE_SEQ_RE = re.compile(r"(?:lcb(\d+)_|lcb_[a-z]+_(\d+)_|lcb_pe_lin_(\d+)_)", re.IGNORECASE)
STAGES = {"steep", "rlx", "pr", "nvt_prod"}
BASELINE_OBSERVABLES = {
    "Rg": "status_Rg",
    "RDF": "status_RDF",
}
POSTPROCESS_OBSERVABLES = {
    "Rg": "status_Rg",
    "RDF": "status_RDF",
    "MSD": "status_MSD",
    "shape": "status_shape",
    "dynamics": "status_dynamics",
    "contact": "status_contact",
    "density": "status_density",
}


REMOTE_PROCESS_SCAN = r"""
import getpass
import json
import os
import pwd
import re
import subprocess
from pathlib import Path

case_re = re.compile(r"(lcb_pe(?:24cuda|24)_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+|lcb\d+_[A-Za-z0-9]+_b\d+_l\d+_c\d+|lcb_[a-z]+_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+|lcb_pe_lin_\d+_c\d+)")
stages = {"steep", "rlx", "pr", "nvt_prod"}


def run(cmd):
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    except Exception as exc:
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": str(exc)})()


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


def parse_case(path):
    match = case_re.search(str(path))
    return normalize_case_id(match.group(1)) if match else ""


def parse_stage(cwd, deffnm):
    name = Path(cwd).name if cwd else ""
    if name in stages:
        return name
    if deffnm in stages:
        return deffnm
    return deffnm or name


def pid_stats(pid):
    result = run(["ps", "-p", str(pid), "-o", "etimes=", "-o", "pcpu=", "-o", "pmem="])
    parts = result.stdout.split()
    return {
        "etimes": parts[0] if len(parts) > 0 else "",
        "pcpu": parts[1] if len(parts) > 1 else "",
        "pmem": parts[2] if len(parts) > 2 else "",
    }


payload = {
    "ok": True,
    "hostname": os.uname().nodename,
    "user": getpass.getuser(),
    "gpus": [],
    "processes": [],
}

def username_for_pid(pid):
    try:
        status = (Path("/proc") / str(pid) / "status").read_text(errors="ignore")
        uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
        if not uid_line:
            return ""
        return pwd.getpwuid(int(uid_line.split()[1])).pw_name
    except Exception:
        return ""

result = run([
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
])
if result.returncode == 0:
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            payload["gpus"].append({
                "gpu": parts[0],
                "name": parts[1],
                "util_percent": parts[2],
                "memory_used_mib": parts[3],
                "memory_total_mib": parts[4],
            })
else:
    payload["nvidia_smi_error"] = result.stderr.strip()

pmon = run(["nvidia-smi", "pmon", "-c", "1"])
gpu_processes = {str(gpu.get("gpu", "")): [] for gpu in payload["gpus"]}
if pmon.returncode == 0:
    for line in pmon.stdout.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        gpu_idx, pid, command = parts[0], parts[1], parts[7]
        owner = username_for_pid(pid)
        gpu_processes.setdefault(gpu_idx, []).append({"pid": pid, "owner": owner, "command": command})
for gpu in payload["gpus"]:
    visible = gpu_processes.get(str(gpu.get("gpu", "")), [])
    external = [proc for proc in visible if proc.get("owner") and proc.get("owner") != payload["user"]]
    gpu["visible_process_count"] = str(len(visible))
    gpu["external_process_count"] = str(len(external))
    gpu["visible_process_pids"] = ";".join(proc.get("pid", "") for proc in visible)
    gpu["external_process_pids"] = ";".join(proc.get("pid", "") for proc in external)
    gpu["external_process_users"] = ";".join(sorted({proc.get("owner", "") for proc in external if proc.get("owner")}))

uid = os.getuid()
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
    stats = pid_stats(pid)
    payload["processes"].append({
        "pid": pid,
        "cwd": cwd,
        "cmd": cmd,
        "case_id": parse_case(cwd) or parse_case(cmd),
        "stage": parse_stage(cwd, deffnm),
        "gpu_id": parse_cmd_value(tokens, "-gpu_id"),
        "ntomp": parse_cmd_value(tokens, "-ntomp"),
        "deffnm": deffnm,
        **stats,
    })

print(json.dumps(payload, ensure_ascii=False))
"""


REMOTE_DEPLOYED_SCAN_TEMPLATE = r"""
import glob
import json
import os
import re
from pathlib import Path

roots = __ROOTS__
case_re = re.compile(r"(lcb_pe(?:24cuda|24)_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+|lcb\d+_[A-Za-z0-9]+_b\d+_l\d+_c\d+|lcb_[a-z]+_\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+|lcb_pe_lin_\d+_c\d+)")
stage_names = ["steep", "rlx", "pr", "nvt_prod"]


def tail_contains(path, needle, max_bytes=262144):
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            data = handle.read().decode(errors="ignore")
        return needle in data
    except Exception:
        return False


def tail_text(path, max_bytes=262144):
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode(errors="ignore")
    except Exception:
        return ""


def failure_stage_from_path(path, text):
    parts = [str(part) for part in getattr(path, "parts", [])]
    for stage in stage_names:
        if stage in parts:
            return stage
    matches = re.findall(r"Working dir:\s+\S+/(steep|rlx|pr|nvt_prod)\b", text)
    if matches:
        return matches[-1]
    match = re.search(r"stage\s+(steep|rlx|pr|nvt_prod)\s+did not complete", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


failure_markers = (
    "fatal error",
    "segmentation",
    "error in user input",
    "not enough memory",
    "mpi_abort",
    "application called mpi_abort",
    "stage nvt_prod did not complete",
)


def detect_failure(case_dir, latest):
    candidates = []
    if latest:
        candidates.append(latest)
    candidates.extend([case_dir / "manual.err", case_dir / "manual.out"])
    seen = set()
    for path in candidates:
        if not path or not path.exists():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        text = tail_text(path)
        lower = text.lower()
        for marker in failure_markers:
            if marker in lower:
                return True, str(path), marker, text[-1200:], failure_stage_from_path(path, text)
    return False, "", "", "", ""


def case_id_from_path(path):
    match = case_re.search(str(path))
    if not match:
        return ""
    text = match.group(1)
    clean_match = re.match(r"lcb_pe(?:24cuda|24)_(\d+_bp\d+_bc\d+_pd[a-z0-9]+_c\d+)$", text, flags=re.I)
    if clean_match:
        return "lcb_pe_" + clean_match.group(1)
    return text


def latest_log(stage_dir):
    logs = []
    for pattern in ["nvt_prod.log", "nvt_prod.part*.log"]:
        logs.extend(stage_dir.glob(pattern))
    logs = [p for p in logs if p.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def artifact_count(case_dir):
    count = 0
    for stage in stage_names:
        stage_dir = case_dir / stage
        if not stage_dir.exists():
            continue
        for pattern in [f"{stage}.cpt", f"{stage}.part*", f"{stage}.xtc", f"{stage}.edr", f"{stage}.gro", f"{stage}.log"]:
            count += len(list(stage_dir.glob(pattern)))
    return count


def evaluate(case_dir):
    nvt = case_dir / "nvt_prod"
    manual_out = case_dir / "manual.out"
    latest = latest_log(nvt)
    manual_complete = tail_contains(manual_out, "Stage nvt_prod completed") if manual_out.exists() else False
    log_finished = tail_contains(latest, "Finished mdrun") if latest else False
    nvt_outputs_complete = bool(
        (nvt / "nvt_prod.gro").exists()
        and ((nvt / "nvt_prod.xtc").exists() or bool(list(nvt.glob("nvt_prod.part*.xtc"))))
        and ((nvt / "nvt_prod.edr").exists() or bool(list(nvt.glob("nvt_prod.part*.edr"))))
    )
    artifacts = artifact_count(case_dir)
    failure_detected, failure_log, failure_reason, failure_tail, failure_stage_name = detect_failure(case_dir, latest)
    strict_done = bool(log_finished and (manual_complete or nvt_outputs_complete))
    return {
        "case_id": case_id_from_path(case_dir),
        "run_dir": str(case_dir),
        "has_manual_out": manual_out.exists(),
        "manual_nvt_completed": manual_complete,
        "nvt_outputs_complete": nvt_outputs_complete,
        "latest_nvt_log": str(latest) if latest else "",
        "latest_nvt_log_finished": log_finished,
        "strict_done": strict_done,
        "failed_stage": bool((not strict_done) and failure_detected),
        "failure_reason": failure_reason,
        "failure_stage_name": failure_stage_name,
        "latest_failure_log": failure_log,
        "failure_tail": failure_tail,
        "resume_artifact_only": bool((not strict_done) and artifacts > 0 and not failure_detected),
        "artifact_count": artifacts,
        "has_nvt_tpr": (nvt / "nvt_prod.tpr").exists(),
        "has_nvt_xtc": (nvt / "nvt_prod.xtc").exists() or bool(list(nvt.glob("nvt_prod.part*.xtc"))),
        "stage_dirs": ",".join([stage for stage in stage_names if (case_dir / stage).is_dir()]),
    }


items = []
seen = set()
for root in roots:
    root_path = Path(root)
    if not root_path.exists():
        continue
    for current, dirs, files in os.walk(root_path):
        path = Path(current)
        rel_depth = len(path.relative_to(root_path).parts)
        if rel_depth > 2:
            dirs[:] = []
            continue
        if case_re.search(path.name) and (path / "run_manual.sh").exists() or ((path / "manual.out").exists() and case_re.search(path.name)):
            if str(path) not in seen:
                seen.add(str(path))
                items.append(evaluate(path))
            dirs[:] = []
        elif rel_depth >= 2:
            dirs[:] = []

print(json.dumps({
    "ok": True,
    "hostname": os.uname().nodename,
    "roots": roots,
    "deployments": items,
}, ensure_ascii=False))
"""


@dataclass(frozen=True)
class RemoteTarget:
    label: str
    host: str
    user: str
    roots: list[str]
    nodes: list[str] | None = None


TARGETS = {
    "new": RemoteTarget("new", NEW_SERVER, "liuz", NEW_SERVER_ROOTS, NEW_SERVER_NODES),
    "newnew": RemoteTarget("newnew", NEWNEW_SERVER, "${REMOTE_USER}", NEWNEW_SERVER_ROOTS, None),
}


def case_sort_key(case_id: str) -> tuple[int, str]:
    match = CASE_SEQ_RE.search(str(case_id))
    seq = next((group for group in (match.groups() if match else []) if group), None)
    return (int(seq) if seq else 10**9, str(case_id))


def extract_case_id(value: Any) -> str:
    match = CASE_RE.search(str(value))
    return match.group(1) if match else ""


def ensure_status_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column].fillna("MISSING").astype(str)
    return pd.Series(["MISSING"] * len(df), index=df.index)


def load_design(manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv, encoding="utf-8-sig")
    df = df[
        df["case_id"].astype(str).str.match(r"(?:lcb\d+_|lcb_[a-z]+_\d+_|lcb_pe_lin_\d+_)", case=False)
    ].copy()
    df["case_seq"] = df["case_id"].map(lambda x: case_sort_key(str(x))[0])
    return df.sort_values("case_seq").drop_duplicates("case_id", keep="first")


def load_integrated(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["case_id"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "case_id" not in df.columns:
        return pd.DataFrame(columns=["case_id"])
    keep = [
        "case_id",
        "case_seq",
        "topology_slug",
        "topology_family",
        "branch_points",
        "branch_carbons",
        "status_eta",
        "status_Rg",
        "status_RDF",
        "status_shape",
        "status_MSD",
        "status_dynamics",
        "status_contact",
        "status_density",
        "quality_flags",
        "status_ok_count",
        "status_warn_count",
        "status_fail_count",
        "status_missing_count",
    ]
    keep = [col for col in keep if col in df.columns]
    return df[keep].drop_duplicates("case_id", keep="first")


def run_ssh_python(host: str, script: str, timeout: int) -> tuple[bool, dict[str, Any], str]:
    # This workspace uses an SSH wrapper where plain `ssh host ...` works,
    # while OpenSSH `-o` options can be rejected before authentication.
    # subprocess.timeout remains the guard against hanging probes.
    cmd = [
        "ssh",
        host,
        "python3",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            input=script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except Exception as exc:
        return False, {}, str(exc)
    if result.returncode != 0:
        return False, {}, result.stderr.strip() or result.stdout.strip()
    try:
        return True, json.loads(result.stdout), result.stderr.strip()
    except json.JSONDecodeError as exc:
        return False, {}, f"invalid json: {exc}; stdout={result.stdout[:500]!r}; stderr={result.stderr[:500]!r}"


def run_nested_ssh_python(login_host: str, node: str, script: str, timeout: int) -> tuple[bool, dict[str, Any], str]:
    # Keep nested probes on the same plain-ssh path as manual supervisor
    # commands; see run_ssh_python for the environment-specific rationale.
    cmd = [
        "ssh",
        login_host,
        "ssh",
        node,
        "python3",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            input=script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except Exception as exc:
        return False, {}, str(exc)
    if result.returncode != 0:
        return False, {}, result.stderr.strip() or result.stdout.strip()
    try:
        return True, json.loads(result.stdout), result.stderr.strip()
    except json.JSONDecodeError as exc:
        return False, {}, f"invalid json: {exc}; stdout={result.stdout[:500]!r}; stderr={result.stderr[:500]!r}"


def scan_processes(targets: list[RemoteTarget], timeout: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    processes: list[dict[str, Any]] = []
    gpus: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for target in targets:
        if target.nodes:
            for node in target.nodes:
                ok, payload, error = run_nested_ssh_python(target.host, node, REMOTE_PROCESS_SCAN, timeout)
                if not ok:
                    errors.append({"host_label": target.label, "host": target.host, "node": node, "error": error})
                    continue
                for gpu in payload.get("gpus", []):
                    gpus.append({"host_label": target.label, "host": target.host, "node": node, **gpu})
                for proc in payload.get("processes", []):
                    processes.append({"host_label": target.label, "host": target.host, "node": node, **proc})
        else:
            ok, payload, error = run_ssh_python(target.host, REMOTE_PROCESS_SCAN, timeout)
            node = payload.get("hostname", "") if ok else ""
            if not ok:
                errors.append({"host_label": target.label, "host": target.host, "node": node, "error": error})
                continue
            for gpu in payload.get("gpus", []):
                gpus.append({"host_label": target.label, "host": target.host, "node": node, **gpu})
            for proc in payload.get("processes", []):
                processes.append({"host_label": target.label, "host": target.host, "node": node, **proc})
    return processes, gpus, errors


def scan_deployments(targets: list[RemoteTarget], timeout: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deployments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for target in targets:
        script = REMOTE_DEPLOYED_SCAN_TEMPLATE.replace("__ROOTS__", json.dumps(target.roots))
        ok, payload, error = run_ssh_python(target.host, script, timeout)
        if not ok:
            errors.append({"host_label": target.label, "host": target.host, "node": "", "error": error})
            continue
        for row in payload.get("deployments", []):
            deployments.append({"host_label": target.label, "host": target.host, "node": payload.get("hostname", ""), **row})
    return deployments, errors


def status_is_missing(value: Any) -> bool:
    return str(value or "MISSING").strip().upper() in {"", "MISSING", "FAIL", "NAN"}


def classify_runtime_case(
    case_id: str,
    deployments: list[dict[str, Any]],
    active_processes: list[dict[str, Any]],
    integrated_row: dict[str, Any],
) -> dict[str, Any]:
    case_deployments = [row for row in deployments if row.get("case_id") == case_id]
    case_active = [row for row in active_processes if row.get("case_id") == case_id]
    strict_done = any(bool(row.get("strict_done")) for row in case_deployments)
    failed_stage = any(bool(row.get("failed_stage")) for row in case_deployments)
    resume_artifact = any(bool(row.get("resume_artifact_only")) for row in case_deployments)
    active_count = len(case_active)
    deployed_count = len(case_deployments)
    deployed_paths = sorted({str(row.get("run_dir", "")) for row in case_deployments if row.get("run_dir")})
    failure_reasons = sorted({str(row.get("failure_reason", "")) for row in case_deployments if row.get("failure_reason")})
    failure_stage_names = sorted({str(row.get("failure_stage_name", "")) for row in case_deployments if row.get("failure_stage_name")})
    latest_failure_logs = sorted({str(row.get("latest_failure_log", "")) for row in case_deployments if row.get("latest_failure_log")})
    failure_tails = [str(row.get("failure_tail", "")) for row in case_deployments if row.get("failure_tail")]
    active_locations = sorted(
        {
            f"{row.get('host_label','')}:{row.get('node','')}:gpu{row.get('gpu_id','')}:pid{row.get('pid','')}"
            for row in case_active
        }
    )
    active_cwds = [str(row.get("cwd", "")) for row in case_active]
    expected_clean_followup_active = bool(
        active_count > 0
        and strict_done
        and active_cwds
        and all(
            "pelcb_clean_engine_runs" in cwd
            and ("nvt_extension" in cwd or "clean" in cwd)
            for cwd in active_cwds
        )
    )

    duplicate_reason = ""
    if active_count > 1:
        duplicate_reason = "multiple_active_processes"
    elif active_count > 0 and strict_done and not expected_clean_followup_active:
        duplicate_reason = "active_but_strict_done"

    if duplicate_reason:
        runtime_status = "duplicate_active"
    elif expected_clean_followup_active:
        runtime_status = "active_followup_after_strict_done"
    elif active_count > 0:
        runtime_status = "active"
    elif strict_done:
        runtime_status = "strict_done"
    elif failed_stage:
        runtime_status = "failed_stage"
    elif resume_artifact:
        runtime_status = "resume_artifact_only"
    elif deployed_count > 0:
        runtime_status = "deployed_idle"
    else:
        runtime_status = "not_deployed"

    missing_baseline = [
        name
        for name, column in BASELINE_OBSERVABLES.items()
        if status_is_missing(integrated_row.get(column, "MISSING"))
    ]
    missing_postprocess = [
        name
        for name, column in POSTPROCESS_OBSERVABLES.items()
        if status_is_missing(integrated_row.get(column, "MISSING"))
    ]
    status_ok_count = integrated_row.get("status_ok_count", "")
    try:
        analyzed_any = int(float(status_ok_count)) > 0
    except (TypeError, ValueError):
        analyzed_any = any(
            not status_is_missing(integrated_row.get(column, "MISSING"))
            for column in BASELINE_OBSERVABLES.values()
        )

    ready_for_analysis = bool(strict_done and missing_postprocess)
    backlog_submit_candidate = bool(
        runtime_status in {"not_deployed", "failed_stage"}
        and not analyzed_any
    )

    latest_logs = [
        str(row.get("latest_nvt_log", ""))
        for row in case_deployments
        if row.get("latest_nvt_log")
    ]
    return {
        "case_id": case_id,
        "runtime_status": runtime_status,
        "active": bool(active_count),
        "active_count": active_count,
        "active_locations": ";".join(active_locations),
        "strict_done": strict_done,
        "failed_stage": bool(failed_stage and not strict_done),
        "failure_reason": ";".join(failure_reasons),
        "failure_stage_name": ";".join(failure_stage_names),
        "latest_failure_logs": ";".join(latest_failure_logs),
        "failure_tail": "\n---\n".join(failure_tails)[-2000:],
        "resume_artifact_only": bool(resume_artifact and not strict_done),
        "deployed_count": deployed_count,
        "deployed_hosts": ";".join(sorted({str(row.get("host_label", "")) for row in case_deployments if row.get("host_label")})),
        "deployed_paths": ";".join(deployed_paths),
        "latest_nvt_logs": ";".join(latest_logs),
        "duplicate_reason": duplicate_reason,
        "missing_baseline_observables": ";".join(missing_baseline),
        "missing_postprocess_observables": ";".join(missing_postprocess),
        "ready_for_analysis": ready_for_analysis,
        "backlog_submit_candidate": backlog_submit_candidate,
    }


def build_runtime_cases(
    design: pd.DataFrame,
    integrated: pd.DataFrame,
    deployments: list[dict[str, Any]],
    processes: list[dict[str, Any]],
) -> pd.DataFrame:
    merged = design.merge(integrated, on="case_id", how="left", suffixes=("", "_integrated"))
    for column in list(BASELINE_OBSERVABLES.values()) + ["status_eta"]:
        if column not in merged.columns:
            merged[column] = "MISSING"
        merged[column] = merged[column].fillna("MISSING")
    for column in [
        "status_shape",
        "status_MSD",
        "status_dynamics",
        "status_contact",
        "status_density",
        "quality_flags",
    ]:
        if column not in merged.columns:
            merged[column] = "MISSING"
        merged[column] = merged[column].fillna("MISSING")

    runtime_rows = []
    for _, row in merged.iterrows():
        row_dict = row.to_dict()
        runtime_rows.append(classify_runtime_case(str(row["case_id"]), deployments, processes, row_dict))
    runtime = pd.DataFrame(runtime_rows)
    out = merged.merge(runtime, on="case_id", how="left")
    out = out.sort_values("case_id", key=lambda s: s.map(case_sort_key))
    return out


def active_process_frame(processes: list[dict[str, Any]]) -> pd.DataFrame:
    fields = [
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
    df = pd.DataFrame(processes)
    for field in fields:
        if field not in df.columns:
            df[field] = ""
    return df[fields].sort_values(["host_label", "node", "gpu_id", "case_id", "pid"], na_position="last")


def duplicate_frame(runtime_cases: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id",
        "runtime_status",
        "active_count",
        "active_locations",
        "strict_done",
        "deployed_count",
        "deployed_paths",
        "duplicate_reason",
    ]
    return runtime_cases[runtime_cases["duplicate_reason"].astype(str) != ""][cols].copy()


def ready_for_analysis_frame(runtime_cases: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id",
        "runtime_status",
        "strict_done",
        "missing_baseline_observables",
        "missing_postprocess_observables",
        "deployed_hosts",
        "deployed_paths",
        "status_eta",
        "status_Rg",
        "status_RDF",
        "status_MSD",
        "status_shape",
        "status_dynamics",
        "status_contact",
        "status_density",
    ]
    return runtime_cases[runtime_cases["ready_for_analysis"] == True][cols].copy()  # noqa: E712


def backlog_submit_frame(runtime_cases: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "case_id",
        "case_seq",
        "branch_points",
        "branch_carbons",
        "branch_topology",
        "branch_position_pattern",
        "runtime_status",
        "status_eta",
        "status_Rg",
        "status_RDF",
        "quality_flags",
        "failure_reason",
        "failure_stage_name",
        "latest_failure_logs",
    ]
    existing_cols = [col for col in cols if col in runtime_cases.columns]
    return runtime_cases[runtime_cases["backlog_submit_candidate"] == True][existing_cols].copy()  # noqa: E712


def write_outputs(
    output_dir: Path,
    runtime_cases: pd.DataFrame,
    active: pd.DataFrame,
    duplicates: pd.DataFrame,
    ready: pd.DataFrame,
    backlog: pd.DataFrame,
    gpus: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    json_only: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_cases.to_csv(output_dir / "runtime_cases.csv", index=False, encoding="utf-8-sig")
    runtime_cases.to_json(output_dir / "runtime_cases.json", orient="records", force_ascii=False, indent=2)
    active.to_csv(output_dir / "runtime_active_processes.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(output_dir / "runtime_duplicates.csv", index=False, encoding="utf-8-sig")
    ready.to_csv(output_dir / "runtime_ready_for_analysis.csv", index=False, encoding="utf-8-sig")
    backlog.to_csv(output_dir / "runtime_backlog_submit.csv", index=False, encoding="utf-8-sig")
    gpu_fields = [
        "host_label",
        "host",
        "node",
        "gpu",
        "name",
        "util_percent",
        "memory_used_mib",
        "memory_total_mib",
        "visible_process_count",
        "external_process_count",
        "visible_process_pids",
        "external_process_pids",
        "external_process_users",
    ]
    error_fields = ["host_label", "host", "node", "error"]
    pd.DataFrame(gpus, columns=gpu_fields).to_csv(
        output_dir / "runtime_gpu_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(errors, columns=error_fields).to_csv(
        output_dir / "runtime_scan_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "readonly": True,
        "cases": int(len(runtime_cases)),
        "active_processes": int(len(active)),
        "duplicates": int(len(duplicates)),
        "ready_for_analysis": int(len(ready)),
        "backlog_submit_candidates": int(len(backlog)),
        "scan_errors": int(len(errors)),
        "outputs": {
            "runtime_cases_csv": str(output_dir / "runtime_cases.csv"),
            "runtime_active_processes_csv": str(output_dir / "runtime_active_processes.csv"),
            "runtime_duplicates_csv": str(output_dir / "runtime_duplicates.csv"),
            "runtime_ready_for_analysis_csv": str(output_dir / "runtime_ready_for_analysis.csv"),
            "runtime_backlog_submit_csv": str(output_dir / "runtime_backlog_submit.csv"),
            "runtime_gpu_snapshot_csv": str(output_dir / "runtime_gpu_snapshot.csv"),
            "runtime_scan_errors_csv": str(output_dir / "runtime_scan_errors.csv"),
        },
    }
    (output_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not json_only:
        (output_dir / "runtime_dashboard.md").write_text(
            render_markdown(runtime_cases, active, duplicates, ready, backlog, gpus, errors),
            encoding="utf-8",
        )


def render_markdown(
    runtime_cases: pd.DataFrame,
    active: pd.DataFrame,
    duplicates: pd.DataFrame,
    ready: pd.DataFrame,
    backlog: pd.DataFrame,
    gpus: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> str:
    status_counts = runtime_cases["runtime_status"].value_counts().to_dict()
    lines = [
        f"# {PROJECT_ID.upper()} Runtime Dashboard",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Mode: read-only; no jobs submitted, stopped, or migrated.",
        f"- Total design cases: `{len(runtime_cases)}`",
        f"- Active GROMACS processes: `{len(active)}`",
        f"- Duplicate/conflict rows: `{len(duplicates)}`",
        f"- Strict done but missing non-viscosity postprocessing: `{len(ready)}`",
        f"- Submit backlog candidates: `{len(backlog)}`",
        "",
        "## Runtime Status Counts",
        "",
    ]
    for key in sorted(status_counts):
        lines.append(f"- `{key}`: {status_counts[key]}")
    lines.extend(["", "## GPU Snapshot", ""])
    if gpus:
        gpu_df = pd.DataFrame(gpus)
        grouped = gpu_df.groupby(["host_label", "node"], dropna=False).size().reset_index(name="gpu_count")
        for _, row in grouped.iterrows():
            lines.append(f"- `{row['host_label']}:{row['node']}`: {int(row['gpu_count'])} GPUs observed")
    else:
        lines.append("- No GPU snapshot available.")
    if len(duplicates):
        lines.extend(["", "## Duplicate / Conflict Cases", ""])
        for _, row in duplicates.head(30).iterrows():
            lines.append(f"- `{row['case_id']}`: {row['duplicate_reason']} ({row['active_locations']})")
    if len(ready):
        lines.extend(["", "## Ready For Non-Viscosity Postprocessing", ""])
        for _, row in ready.head(30).iterrows():
            lines.append(f"- `{row['case_id']}` missing `{row['missing_postprocess_observables']}`")
    if len(backlog):
        lines.extend(["", "## First Submit Backlog Candidates", ""])
        for _, row in backlog.head(30).iterrows():
            lines.append(f"- `{row['case_id']}`")
    if errors:
        lines.extend(["", "## Scan Errors", ""])
        for row in errors:
            lines.append(f"- `{row.get('host_label')}:{row.get('node','')}`: {row.get('error','')[:220]}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a read-only LCB300 runtime dashboard.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--integrated-csv", type=Path, default=DEFAULT_INTEGRATED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", choices=["new", "newnew", "all"], default="all")
    parser.add_argument("--skip-remote", action="store_true")
    parser.add_argument("--scan-old", action="store_true", help="Also read old-server staged postprocessing dirs.")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--ssh-timeout", type=int, default=45)
    return parser.parse_args()


def selected_targets(host: str) -> list[RemoteTarget]:
    refresh_targets()
    if host == "all":
        return [TARGETS["new"], TARGETS["newnew"]]
    return [TARGETS[host]]


def refresh_targets() -> None:
    TARGETS["new"] = RemoteTarget("new", NEW_SERVER, "liuz", NEW_SERVER_ROOTS, NEW_SERVER_NODES)
    TARGETS["newnew"] = RemoteTarget("newnew", NEWNEW_SERVER, "${REMOTE_USER}", NEWNEW_SERVER_ROOTS, None)


def apply_config(args: argparse.Namespace) -> None:
    global PROJECT_ID, NEW_SERVER, NEWNEW_SERVER, OLD_SERVER, NEW_SERVER_ROOTS, NEWNEW_SERVER_ROOTS, OLD_STAGED_ROOTS
    if not args.config:
        return
    config = load_lcb_config(args.config)
    PROJECT_ID = config.project_id
    args.manifest_csv = config.manifest_csv
    args.integrated_csv = config.integrated_summary_csv
    args.output_dir = config.runtime_output_dir
    NEW_SERVER = config.new_server
    NEWNEW_SERVER = config.newnew_server
    OLD_SERVER = config.old_server
    NEW_SERVER_ROOTS = config.remote_md_roots.get("new", NEW_SERVER_ROOTS)
    NEWNEW_SERVER_ROOTS = config.remote_md_roots.get("newnew", NEWNEW_SERVER_ROOTS)
    OLD_STAGED_ROOTS = [f"{config.old_server_postprocess_root}/input", "${PELCB100_REMOTE_ROOT}/lcb_rg_20260501/input"]
    refresh_targets()


def main() -> int:
    args = parse_args()
    apply_config(args)
    design = load_design(args.manifest_csv)
    integrated = load_integrated(args.integrated_csv)

    processes: list[dict[str, Any]] = []
    gpus: list[dict[str, Any]] = []
    deployments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if not args.skip_remote:
        targets = selected_targets(args.host)
        processes, gpus, proc_errors = scan_processes(targets, args.ssh_timeout)
        deployments, deploy_errors = scan_deployments(targets, args.ssh_timeout)
        errors.extend(proc_errors)
        errors.extend(deploy_errors)
        if args.scan_old:
            old_target = RemoteTarget("old_staged", OLD_SERVER, "${REMOTE_USER}", OLD_STAGED_ROOTS, None)
            old_deployments, old_errors = scan_deployments([old_target], args.ssh_timeout)
            deployments.extend(old_deployments)
            errors.extend(old_errors)

    runtime_cases = build_runtime_cases(design, integrated, deployments, processes)
    active = active_process_frame(processes)
    duplicates = duplicate_frame(runtime_cases)
    ready = ready_for_analysis_frame(runtime_cases)
    backlog = backlog_submit_frame(runtime_cases)
    write_outputs(args.output_dir, runtime_cases, active, duplicates, ready, backlog, gpus, errors, args.json_only)

    print(f"wrote {args.output_dir}")
    print(f"cases={len(runtime_cases)} active={len(active)} duplicates={len(duplicates)} ready={len(ready)} backlog={len(backlog)} errors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
