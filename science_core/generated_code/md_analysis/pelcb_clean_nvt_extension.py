#!/usr/bin/env python3
"""Extend clean-engine nvt_prod from checkpoint for publication-grade MSD.

The tool is deliberately narrow: it does not rebuild systems, does not edit
MDP physics, and does not change canonical PE-LCB100 release tables.  It uses
GROMACS `convert-tpr -extend` on an existing clean nvt_prod checkpoint and
launches a detached resume on a live free GPU slot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from lcb_project_config import load_lcb_config
import o7_clean_supervisor_queue as queue


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "pelcb100.json"
DEFAULT_STATUS_CSV = (
    SCRIPT_DIR
    / "output"
    / "pelcb100_supervisor"
    / "o7_clean_cuda2024_closure"
    / "o7_clean_run_status.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_nvt_extension"
NEW_HOST = os.environ.get("PELCB_NEW_SERVER") or "${REMOTE_USER}@private-4e2f23ff.invalid"
NEW_REMOTE_ROOT = "${PELCB100_REMOTE_HOME}/pelcb_clean_engine_runs"
OLD_HOST = os.environ.get("PELCB_OLD_SERVER") or "${REMOTE_USER}@private-3f2a50db.invalid"
OLD_INTEL_REMOTE_ROOT = "${PELCB100_REMOTE_ROOT}/pelcb_clean_engine_runs"
OLD_INTEL_GMX = "${PELCB100_REMOTE_PATH}"
OLD_INTEL_MODULE = "module load intel${PELCB100_REMOTE_PATH}"
DEFAULT_REFIT_CLASSIFICATION = (
    SCRIPT_DIR
    / "output"
    / "pelcb100_clean_overlay_regate"
    / "msd_extend_vs_refit_20260611.csv"
)
DEFAULT_REFIT_LATEST = SCRIPT_DIR / "output" / "pelcb100_clean_overlay_regate" / "strict_msd_refit_latest.json"
DEFAULT_RUNTIME_ACTIVE_CSV = SCRIPT_DIR / "output" / "pelcb100_runtime_dashboard" / "runtime_active_processes.csv"


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def status_row_priority(row: dict[str, str]) -> tuple[int, int, int]:
    """Prefer the most recent checkpoint-bearing row for a case.

    The closure table can contain both the original 400 ns clean run and a
    live old-server extension for the same case.  For migration/resume, the
    extension row is the valuable one because it carries the latest checkpoint.
    """
    remote_run_dir = str(row.get("remote_run_dir", "") or "")
    tag = str(row.get("tag", "") or "")
    is_oldserver_extension = remote_run_dir.startswith("${PELCB100_REMOTE_ROOT}/") and (
        "oldserver_extension" in tag or "nvt_extension" in remote_run_dir
    )
    is_oldserver = remote_run_dir.startswith("${PELCB100_REMOTE_ROOT}/")
    is_incomplete_nvt = str(row.get("nvt_prod_done", "")).lower() != "true"
    return (
        1 if is_oldserver_extension else 0,
        1 if is_oldserver else 0,
        1 if is_incomplete_nvt else 0,
    )


def dedupe_status_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id", "") or row.get("clean_case_id", "")).strip()
        if not case_id:
            continue
        if case_id not in selected:
            selected[case_id] = row
            order.append(case_id)
            continue
        if status_row_priority(row) > status_row_priority(selected[case_id]):
            selected[case_id] = row
    return [selected[case_id] for case_id in order]


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


def _terminate_process_tree(pid: int) -> None:
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


def _run_hangproof(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    """Run a command without hanging on pipes held by detached grandchildren.

    stdout/stderr are redirected to temp files (never PIPE); on timeout the whole
    process tree is killed.  Returns a CompletedProcess so existing callers that
    read .returncode/.stdout/.stderr are unchanged.
    """
    timed_out = False
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    with tempfile.TemporaryFile(mode="w+b") as out_f, tempfile.TemporaryFile(mode="w+b") as err_f:
        proc = subprocess.Popen(command, stdout=out_f, stderr=err_f, **kwargs)
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
        stderr = (stderr + f"\n[hangproof] TIMEOUT after {timeout}s; process tree terminated").strip()
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def run_cmd(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return _run_hangproof(command, timeout)


def scp_cmd(command: list[str], timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    return _run_hangproof(command, timeout)


def eligible_status_row(row: dict[str, str]) -> bool:
    clean_case_id = str(row.get("clean_case_id", "") or "").strip()
    remote_run_dir = str(row.get("remote_run_dir", "") or "").strip()
    host_label = str(row.get("source_host_label", "") or "").strip()
    host = str(row.get("source_host", "") or "").strip()
    is_new_clean = remote_run_dir.startswith("${PELCB100_REMOTE_HOME}/") and not remote_run_dir.startswith("${PELCB100_REMOTE_ROOT}/")
    is_oldserver_clean = remote_run_dir.startswith("${PELCB100_REMOTE_ROOT}/")
    if "cuda" not in clean_case_id and not is_oldserver_clean:
        return False
    if not (is_new_clean or is_oldserver_clean):
        return False
    if is_new_clean and host_label and host_label not in {"new", "newnew", "cuda2024"}:
        return False
    if is_new_clean and host and "private-4e2f23ff.invalid" not in host and "${REMOTE_USER}@" not in host:
        return False
    if is_oldserver_clean and host and "private-3f2a50db.invalid" not in host and "${REMOTE_USER}@" not in host:
        return False
    if str(row.get("nvt_prod_has_tpr", "")).lower() != "true":
        return False
    if str(row.get("nvt_prod_has_cpt", "")).lower() != "true":
        return False
    # Old-server live extension rows are already checkpoint-extension
    # directories.  They carry root-level nvt_prod.tpr/cpt and may not have a
    # PR marker, so do not require pr_done for that layout.
    if not is_oldserver_clean and str(row.get("pr_done", "")).lower() != "true":
        return False
    return True


def read_classification(path: Path) -> dict[str, str]:
    return {
        str(row.get("case_id", "")).strip(): str(row.get("classification", "")).strip()
        for row in read_csv_rows(path)
        if str(row.get("case_id", "")).strip()
    }


def read_refit_status(latest_path: Path = DEFAULT_REFIT_LATEST) -> dict[str, str]:
    if not latest_path.exists():
        return {}
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    candidate_path = Path(str(payload.get("candidate_csv", "")))
    if not candidate_path.exists():
        return {}
    return {
        str(row.get("case_id", "")).strip(): str(row.get("refit_status", "")).strip()
        for row in read_csv_rows(candidate_path)
        if str(row.get("case_id", "")).strip()
    }


def supervisor_free_slots(
    config_path: Path,
    queue_dir: Path,
    *,
    runtime_max_age_seconds: int = 1800,
    dashboard_timeout_sec: int = 180,
    preferred_node: str = "",
    allow_second_per_gpu: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return free slots under the clean-engine supervisor policy.

    Extension runs are long enough that single-occupancy is the conservative
    default.  Some clean-engine nodes, especially node-d61b45bd, are operated by
    the supervisor at two cases per GPU; callers must opt into that policy.
    """
    config = load_lcb_config(config_path)
    profile = queue.resolve_o7_cluster_profile(config, "o7_clean")
    queue.setup_supervisor_globals(config_path, queue_dir, cluster_profile=profile)
    dashboard = queue.refresh_dashboard_for_queue(config_path, timeout_sec=dashboard_timeout_sec)
    runtime_status = queue.runtime_snapshot_status(
        queue.supervisor.RUNTIME_DIR,
        max_age_seconds=runtime_max_age_seconds,
    )
    if not dashboard.get("ok") and not runtime_status.get("ok"):
        return [], runtime_status, dashboard
    runtime = queue.supervisor.load_runtime()
    reservations = queue.recent_queue_reservations(queue_dir, max_age_seconds=runtime_max_age_seconds)
    runtime_status["queue_reservation_count"] = len(reservations)
    slots = queue.choose_clean_slots(
        runtime,
        plan_rows=[],
        allow_second_per_gpu=allow_second_per_gpu,
        cluster_profile=profile,
        reservation_rows=reservations,
    )
    preferred = str(preferred_node or "").strip()
    # ``compute_slots`` emits one entry per remaining capacity unit.  When a
    # node permits two jobs per GPU, an empty GPU therefore appears twice and
    # a singly occupied GPU once.  Prefer the GPU with the most remaining
    # capacity so long extensions do not share a device while an idle one is
    # available.
    remaining_capacity = Counter(
        (str(slot.get("node", "")), str(slot.get("gpu", ""))) for slot in slots
    )
    slots = sorted(
        slots,
        key=lambda slot: (
            0 if not preferred or str(slot.get("node", "")) == preferred else 1,
            -remaining_capacity[(str(slot.get("node", "")), str(slot.get("gpu", "")))],
            str(slot.get("node", "")),
            int(str(slot.get("gpu", "0")) or 0),
        ),
    )
    return slots, runtime_status, dashboard


def assign_slots(rows: list[dict[str, str]], slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    slot_index = 0
    for index, row in enumerate(rows):
        out: dict[str, Any] = dict(row)
        preassigned_status = str(out.get("capacity_status", "")).strip()
        if preassigned_status and preassigned_status != "SLOT_PENDING":
            planned.append(out)
            continue
        remote_run_dir = str(row.get("remote_run_dir", "") or "")
        if slot_index >= len(slots):
            out["capacity_status"] = "BLOCKED_NO_FREE_GPU_SLOT"
            out["target_node"] = ""
            out["target_gpu_id"] = ""
            out["target_threads"] = ""
        else:
            slot = slots[slot_index]
            slot_index += 1
            out["capacity_status"] = "SLOT_ASSIGNED"
            if remote_run_dir.startswith("${PELCB100_REMOTE_ROOT}/"):
                out["capacity_status"] = "SLOT_ASSIGNED_STAGE_OLDSERVER_TO_NEW"
                out["source_remote_run_dir"] = remote_run_dir
            out["target_node"] = str(slot.get("node", ""))
            out["target_gpu_id"] = str(slot.get("gpu", ""))
            out["target_threads"] = int(slot.get("threads", 1))
            out["target_gmx_bin"] = str(slot.get("gmx_bin", ""))
            out["target_engine_module"] = str(slot.get("engine_module", ""))
            out["target_remote_root"] = str(slot.get("remote_root", NEW_REMOTE_ROOT))
        planned.append(out)
    return planned


def apply_refit_before_extend_skip(rows: list[dict[str, Any]]) -> None:
    """Do not extend trajectories that already passed the strict clean-MSD refit gate."""
    for row in rows:
        if str(row.get("strict_refit_status", "")).strip() != "strict_refit_pass":
            continue
        row["capacity_status"] = "SKIPPED_REFIT_BEFORE_EXTEND"
        row["target_cluster"] = "strict_msd_refit"
        row["target_node"] = ""
        row["target_gpu_id"] = ""


def active_case_ids_from_runtime_dashboard(path: Path = DEFAULT_RUNTIME_ACTIVE_CSV) -> set[str]:
    """Read live case IDs from the runtime dashboard active-process table."""

    return {
        str(row.get("case_id", "")).strip()
        for row in read_csv_rows(path)
        if str(row.get("case_id", "")).strip()
    }


def apply_active_duplicate_skip(rows: list[dict[str, Any]], active_case_ids: set[str]) -> None:
    """Prevent launching a second extension for a case that is already running."""

    launchable_statuses = {
        "SLOT_PENDING",
        "SLOT_ASSIGNED",
        "SLOT_ASSIGNED_STAGE_OLDSERVER_TO_NEW",
        "OLDSERVER_INTELGPU_SLURM_ASSIGNED",
    }
    pending_statuses = {"", "SLOT_PENDING"}
    for row in rows:
        status = str(row.get("capacity_status", "")).strip()
        if status not in pending_statuses and status not in launchable_statuses:
            continue
        case_id = str(row.get("case_id", "")).strip()
        if case_id and case_id in active_case_ids:
            row["capacity_status"] = "SKIPPED_ACTIVE_CASE_ALREADY_RUNNING"
            row["target_cluster"] = "active_runtime_guard"
            row["target_node"] = ""
            row["target_gpu_id"] = ""


def oldserver_slurm_snapshot() -> dict[str, Any]:
    command = (
        "set -e; "
        "sinfo -h -p intel -N -o '%N|%T|%G|%C' 2>/dev/null || true; "
        "echo __SCONTROL__; "
        "scontrol show node -o 2>/dev/null | grep -E 'NodeName=node[4567]' || true; "
        "echo __SQUEUE__; "
        "squeue -h -p intel -u \"$USER\" -o '%i|%T|%j|%R|%b' 2>/dev/null || true"
    )
    result = run_cmd(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", OLD_HOST, f"bash -lc {sh_quote(command)}"],
        timeout=60,
    )
    stdout = result.stdout.strip()
    node_lines: list[str] = []
    scontrol_lines: list[str] = []
    queue_lines: list[str] = []
    section = "nodes"
    for line in stdout.splitlines():
        if line.strip() == "__SCONTROL__":
            section = "scontrol"
            continue
        if line.strip() == "__SQUEUE__":
            section = "queue"
            continue
        if section == "nodes":
            node_lines.append(line)
        elif section == "scontrol":
            scontrol_lines.append(line)
        elif line.strip():
            queue_lines.append(line)
    idle_gpu_slots = 0
    idle_nodes: list[str] = []
    free_by_node: dict[str, int] = {}
    for line in scontrol_lines:
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
        node = fields.get("NodeName", "")
        gres = fields.get("Gres", "")
        total = 0
        for token in gres.split(","):
            if "gpu" not in token.lower():
                continue
            tail = token.split(":")[-1]
            try:
                total = max(total, int(tail))
            except ValueError:
                pass
        alloc = 0
        alloc_tres = fields.get("AllocTRES", "")
        for token in alloc_tres.split(","):
            if token.startswith("gres/gpu="):
                try:
                    alloc = int(token.split("=", 1)[1])
                except ValueError:
                    alloc = 0
        if node and total > 0:
            free_by_node[node] = max(0, total - alloc)
    if free_by_node:
        idle_gpu_slots = sum(free_by_node.values())
        idle_nodes = [node for node, free in free_by_node.items() if free > 0]
    else:
        # Fallback for clusters that do not expose AllocTRES in scontrol.
        idle_gpu_slots = 0
        idle_nodes = []
        for line in node_lines:
            parts = line.split("|")
            if len(parts) < 4:
                continue
            node, state, gres, _cpu = parts[:4]
            state_l = state.lower()
            gpu_count = 0
            for token in gres.split(","):
                if "gpu" not in token.lower():
                    continue
                tail = token.split(":")[-1]
                try:
                    gpu_count = max(gpu_count, int(tail))
                except ValueError:
                    pass
            if state_l in {"idle", "idle*"} and gpu_count > 0:
                idle_nodes.append(node)
                idle_gpu_slots += gpu_count
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip()[-4000:],
        "stderr": result.stderr.strip()[-2000:],
        "node_lines": node_lines,
        "scontrol_lines": scontrol_lines,
        "queue_lines": queue_lines,
        "idle_nodes": idle_nodes,
        "free_gpu_by_node": free_by_node,
        "idle_gpu_slots": idle_gpu_slots,
    }


def oldserver_remote_dir(row: dict[str, Any], stamp: str) -> str:
    case_id = str(row.get("case_id", "")).strip()
    clean_case_id = str(row.get("clean_case_id", "")).strip()
    suffix = clean_case_id or case_id
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in suffix)
    return f"{OLD_INTEL_REMOTE_ROOT}/nvt_extension_{stamp}/{safe}"


def newserver_extension_remote_dir(row: dict[str, Any], stamp: str) -> str:
    case_id = str(row.get("case_id", "")).strip()
    clean_case_id = str(row.get("clean_case_id", "")).strip()
    suffix = clean_case_id or case_id
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in suffix)
    remote_root = str(row.get("target_remote_root") or row.get("remote_root") or NEW_REMOTE_ROOT).rstrip("/")
    return f"{remote_root}/nvt_extension_{stamp}/{safe}"


def case_sequence(row: dict[str, Any]) -> str:
    case_id = str(row.get("case_id", "") or row.get("clean_case_id", "")).strip()
    for part in case_id.split("_"):
        if part.isdigit():
            return part.zfill(3)
    return ""


def oldserver_case_active(row: dict[str, Any], snapshot: dict[str, Any]) -> bool:
    queue_text = "\n".join(str(line) for line in snapshot.get("queue_lines", []))
    if not queue_text:
        return False
    seq = case_sequence(row)
    suffixes = [
        str(row.get("case_id", ""))[-12:],
        str(row.get("clean_case_id", ""))[-12:],
        f"pelcb_ext_{seq}" if seq else "",
    ]
    return any(suffix and suffix in queue_text for suffix in suffixes)


def stage_checkpoint_to_oldserver(row: dict[str, Any], remote_dir: str) -> dict[str, Any]:
    source_dir = str(row.get("remote_run_dir", "")).rstrip("/")
    mkdir = run_cmd(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", OLD_HOST, f"mkdir -p {sh_quote(remote_dir)}"],
        timeout=60,
    )
    if mkdir.returncode != 0:
        return {"ok": False, "stage": "mkdir", "returncode": mkdir.returncode, "stderr": mkdir.stderr.strip()[-2000:]}
    compat_cmd = f"ln -sfn . {sh_quote(remote_dir + '/nvt_prod')}"
    compat = run_cmd(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            OLD_HOST,
            f"bash -lc {sh_quote(compat_cmd)}",
        ],
        timeout=60,
    )
    if compat.returncode != 0:
        return {"ok": False, "stage": "compat_symlink", "returncode": compat.returncode, "stderr": compat.stderr.strip()[-2000:]}
    if source_dir.startswith("${PELCB100_REMOTE_ROOT}/"):
        copy_cmd = (
            f"cp -f {sh_quote(source_dir + '/nvt_prod/nvt_prod.tpr')} {sh_quote(remote_dir + '/nvt_prod.tpr')} && "
            f"cp -f {sh_quote(source_dir + '/nvt_prod/nvt_prod.cpt')} {sh_quote(remote_dir + '/nvt_prod.cpt')}"
        )
        proc = run_cmd(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                OLD_HOST,
                f"bash -lc {sh_quote(copy_cmd)}",
            ],
            timeout=300,
        )
        return {
            "ok": proc.returncode == 0,
            "remote_dir": remote_dir,
            "stage": "oldserver_local_copy",
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[-1000:],
            "stderr": proc.stderr.strip()[-2000:],
        }
    results: list[dict[str, Any]] = []
    for name in ("nvt_prod.tpr", "nvt_prod.cpt"):
        source = f"{NEW_HOST}:{source_dir}/nvt_prod/{name}"
        target = f"{OLD_HOST}:{remote_dir}/{name}"
        proc = scp_cmd(
            [
                "scp",
                "-3",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                source,
                target,
            ],
            timeout=1800,
        )
        results.append(
            {
                "file": name,
                "returncode": proc.returncode,
                "ok": proc.returncode == 0,
                "stdout": proc.stdout.strip()[-1000:],
                "stderr": proc.stderr.strip()[-2000:],
            }
        )
        if proc.returncode != 0:
            return {"ok": False, "stage": "scp", "results": results}
    return {"ok": True, "remote_dir": remote_dir, "results": results}


def stage_oldserver_checkpoint_to_newserver(row: dict[str, Any], remote_dir: str) -> dict[str, Any]:
    source_dir = str(row.get("source_remote_run_dir") or row.get("remote_run_dir", "")).rstrip("/")
    mkdir_cmd = (
        f"mkdir -p {sh_quote(remote_dir)} {sh_quote(remote_dir + '/pr')} && "
        f"ln -sfn . {sh_quote(remote_dir + '/nvt_prod')} && "
        f"printf '%s\\n' 'GMX_BIN={row.get('target_gmx_bin', '')}' > {sh_quote(remote_dir + '/run_manual.sh')} && "
        f"printf '%s\\n' 'Finished mdrun' > {sh_quote(remote_dir + '/pr/pr.log')}"
    )
    mkdir = run_cmd(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", NEW_HOST, f"bash -lc {sh_quote(mkdir_cmd)}"],
        timeout=120,
    )
    if mkdir.returncode != 0:
        return {"ok": False, "stage": "mkdir_newserver", "returncode": mkdir.returncode, "stderr": mkdir.stderr.strip()[-2000:]}
    source_paths: dict[str, str] = {}
    for name in ("nvt_prod.tpr", "nvt_prod.cpt"):
        if name == "nvt_prod.tpr":
            probe_cmd = (
                f"latest=$(ls -t {sh_quote(source_dir)}/nvt_prod.extend_*.tpr "
                f"{sh_quote(source_dir)}/nvt_prod/nvt_prod.extend_*.tpr 2>/dev/null | head -n 1 || true); "
                "if [ -n \"$latest\" ] && [ -s \"$latest\" ]; then printf '%s\\n' \"$latest\"; "
                f"elif [ -s {sh_quote(source_dir + '/nvt_prod/' + name)} ]; then "
                f"printf '%s\\n' {sh_quote(source_dir + '/nvt_prod/' + name)}; "
                f"elif [ -s {sh_quote(source_dir + '/' + name)} ]; then "
                f"printf '%s\\n' {sh_quote(source_dir + '/' + name)}; "
                "else exit 2; fi"
            )
        else:
            probe_cmd = (
                f"if [ -s {sh_quote(source_dir + '/nvt_prod/' + name)} ]; then "
                f"printf '%s\\n' {sh_quote(source_dir + '/nvt_prod/' + name)}; "
                f"elif [ -s {sh_quote(source_dir + '/' + name)} ]; then "
                f"printf '%s\\n' {sh_quote(source_dir + '/' + name)}; "
                "else exit 2; fi"
            )
        probe = run_cmd(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                OLD_HOST,
                f"bash -lc {sh_quote(probe_cmd)}",
            ],
            timeout=120,
        )
        if probe.returncode != 0:
            return {
                "ok": False,
                "stage": "probe_oldserver_source_layout",
                "file": name,
                "source_remote_run_dir": source_dir,
                "returncode": probe.returncode,
                "stdout": probe.stdout.strip()[-1000:],
                "stderr": probe.stderr.strip()[-2000:],
            }
        source_paths[name] = probe.stdout.strip().splitlines()[-1]
    source_tpr_path = source_paths.get("nvt_prod.tpr", "")
    source_tpr_kind = "extended" if ".extend_" in Path(source_tpr_path).name else "base"
    results: list[dict[str, Any]] = []
    for name in ("nvt_prod.tpr", "nvt_prod.cpt"):
        source = f"{OLD_HOST}:{source_paths[name]}"
        target = f"{NEW_HOST}:{remote_dir}/{name}"
        proc = scp_cmd(
            [
                "scp",
                "-3",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                source,
                target,
            ],
            timeout=1800,
        )
        results.append(
            {
                "file": name,
                "returncode": proc.returncode,
                "ok": proc.returncode == 0,
                "stdout": proc.stdout.strip()[-1000:],
                "stderr": proc.stderr.strip()[-2000:],
            }
        )
        if proc.returncode != 0:
            return {"ok": False, "stage": "scp_oldserver_to_newserver", "remote_dir": remote_dir, "results": results}
    return {
        "ok": True,
        "stage": "oldserver_to_newserver_checkpoint_copy",
        "source_remote_run_dir": source_dir,
        "source_tpr_path": source_tpr_path,
        "source_tpr_kind": source_tpr_kind,
        "remote_dir": remote_dir,
        "results": results,
    }


def build_oldserver_slurm_script(row: dict[str, Any], stamp: str, reason: str, extend_ns: float) -> str:
    extend_ps = float(extend_ns) * 1000.0
    provenance = {
        "case_id": row.get("case_id", ""),
        "clean_case_id": row.get("clean_case_id", ""),
        "created_at": stamp,
        "action": "extend_clean_nvt_prod_from_checkpoint",
        "reason": reason,
        "extend_ns": float(extend_ns),
        "mdp_physics_changed": False,
        "uses_convert_tpr_extend": True,
        "target_cluster": "o7_clean_oldserver_intelgpu",
        "target_host": OLD_HOST,
        "slurm_partition": "intel",
        "slurm_gres": "gpu:1",
    }
    provenance_json = json.dumps(provenance, ensure_ascii=False)
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH -J pelcb_ext_{case_sequence(row) or str(row.get('case_id', 'case'))[-8:]}",
            "#SBATCH -p intel",
            "#SBATCH --gres=gpu:1",
            "#SBATCH -N 1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH -o slurm-%j.out",
            "#SBATCH -e slurm-%j.err",
            "set -euo pipefail",
            "module purge >/dev/null 2>&1 || true",
            OLD_INTEL_MODULE,
            f"GMX_BIN={sh_quote(OLD_INTEL_GMX)}",
            "test -x \"$GMX_BIN\" || command -v \"$GMX_BIN\" >/dev/null 2>&1",
            "test -s nvt_prod.tpr",
            "test -s nvt_prod.cpt",
            f"EXTEND_PS={extend_ps:.0f}",
            "EXT_TPR=nvt_prod.extend_${SLURM_JOB_ID:-manual}.tpr",
            "printf '%s\\n' " + sh_quote(provenance_json) + " > pelcb_clean_nvt_extension_${SLURM_JOB_ID:-manual}.json",
            "\"$GMX_BIN\" convert-tpr -s nvt_prod.tpr -extend \"$EXTEND_PS\" -o \"$EXT_TPR\"",
            "\"$GMX_BIN\" mdrun -ntomp \"${SLURM_CPUS_PER_TASK:-2}\" -pin off -nb gpu "
            "-s \"$EXT_TPR\" -deffnm nvt_prod -cpi nvt_prod.cpt -noappend",
        ]
    )


def submit_oldserver_extension(row: dict[str, Any], stamp: str, reason: str, extend_ns: float) -> dict[str, Any]:
    remote_dir = oldserver_remote_dir(row, stamp)
    stage = stage_checkpoint_to_oldserver(row, remote_dir)
    if not stage.get("ok"):
        return {"ok": False, "remote_dir": remote_dir, "stage_result": stage}
    script = build_oldserver_slurm_script(row, stamp, reason, extend_ns)
    upload_cmd = (
        f"cd {sh_quote(remote_dir)} && "
        f"cat > run_extension.sbatch <<'SBATCH'\n{script}\nSBATCH\n"
        "sbatch run_extension.sbatch"
    )
    submit = run_cmd(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", OLD_HOST, f"bash -lc {sh_quote(upload_cmd)}"],
        timeout=120,
    )
    job_id = ""
    for token in submit.stdout.replace("\n", " ").split():
        if token.isdigit():
            job_id = token
    return {
        "ok": submit.returncode == 0 and bool(job_id),
        "returncode": submit.returncode,
        "target_cluster": "o7_clean_oldserver_intelgpu",
        "remote_dir": remote_dir,
        "slurm_job_id": job_id,
        "stdout": submit.stdout.strip()[-2000:],
        "stderr": submit.stderr.strip()[-2000:],
        "stage_result": stage,
    }


def write_oldserver_queue_plan(queue_dir: Path, tag: str, rows: list[dict[str, Any]]) -> str:
    plan_rows: list[dict[str, Any]] = []
    for record in rows:
        result = record.get("result")
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        if result.get("target_cluster") != "o7_clean_oldserver_intelgpu":
            continue
        plan_rows.append(
            {
                "case_id": record.get("case_id", ""),
                "clean_case_id": record.get("clean_case_id", ""),
                "remote_run_dir": result.get("remote_dir", ""),
                "status": "SUBMITTED",
                "plan_kind": "old_intel_2024",
                "source_host_label": "old",
                "source_host": OLD_HOST,
                "target_host_label": "old",
                "target_host": OLD_HOST,
                "target_cluster": "o7_clean_oldserver_intelgpu",
                "slurm_job_id": result.get("slurm_job_id", ""),
                "submitted_at": iso_now(),
            }
        )
    if not plan_rows:
        return ""
    fields = [
        "case_id",
        "clean_case_id",
        "remote_run_dir",
        "status",
        "plan_kind",
        "source_host_label",
        "source_host",
        "target_host_label",
        "target_host",
        "target_cluster",
        "slurm_job_id",
        "submitted_at",
    ]
    plan_path = queue_dir / tag / "o7_stage1_old2024_plan.csv"
    write_csv(plan_path, plan_rows, fields)
    return str(plan_path)


def build_extension_inner_command(
    *,
    row: dict[str, Any],
    stamp: str,
    reason: str,
    extend_ns: float,
) -> str:
    remote_run_dir = str(row.get("remote_run_dir", "") or "").strip()
    gpu_id = str(row.get("target_gpu_id") or row.get("gpu_id") or "").strip()
    threads = str(row.get("target_threads") or row.get("threads") or "2").strip()
    gmx_bin = str(row.get("target_gmx_bin") or "").strip()
    engine_module = str(row.get("target_engine_module") or "").strip()
    extend_ps = float(extend_ns) * 1000.0
    source_hint = str(
        row.get("staged_from_remote_run_dir")
        or row.get("source_remote_run_dir")
        or row.get("remote_run_dir")
        or ""
    )
    tag_hint = str(row.get("tag", "") or "")
    migrated_tpr_flag = row.get("migrated_tpr_is_extension", "")
    if migrated_tpr_flag != "":
        # Old-server migration paths often contain "nvt_extension" even when the
        # copied TPR is the unextended base TPR; trust the staged file provenance.
        resume_existing_extension_tpr = bool(migrated_tpr_flag)
    else:
        resume_existing_extension_tpr = "/nvt_extension_" in source_hint or "oldserver_extension" in tag_hint
    ext_tpr_name = "nvt_prod.tpr" if resume_existing_extension_tpr else f"nvt_prod.extend_{stamp}.tpr"
    provenance = {
        "case_id": row.get("case_id", ""),
        "clean_case_id": row.get("clean_case_id", ""),
        "created_at": stamp,
        "action": "resume_migrated_clean_nvt_prod_from_checkpoint"
        if resume_existing_extension_tpr
        else "extend_clean_nvt_prod_from_checkpoint",
        "reason": reason,
        "extend_ns": float(extend_ns),
        "resume_existing_extension_tpr": resume_existing_extension_tpr,
        "mdp_physics_changed": False,
        "uses_convert_tpr_extend": True,
        "target_node": row.get("target_node", ""),
        "target_gpu_id": gpu_id,
        "target_threads": threads,
    }
    module_line = engine_module if engine_module else "true"
    run_manual = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            module_line,
            f"GMX_BIN={sh_quote(gmx_bin)}",
            f"THREADS={sh_quote(threads)}",
            f"GPU_ID={sh_quote(gpu_id)}",
            "cd \"$(dirname \"$0\")\"",
            f"test -s nvt_prod/{ext_tpr_name}",
            "test -s nvt_prod/nvt_prod.cpt",
            "cd nvt_prod",
            (
                "\"$GMX_BIN\" mdrun -ntmpi 1 -ntomp \"$THREADS\" -pin off "
                "-gpu_id \"$GPU_ID\" -nb gpu -bonded gpu -notunepme "
                f"-s {ext_tpr_name} -deffnm nvt_prod -cpi nvt_prod.cpt -noappend"
            ),
        ]
    )
    convert_lines = []
    if resume_existing_extension_tpr:
        provenance["uses_convert_tpr_extend"] = False
        convert_lines = [
            "test -s nvt_prod/nvt_prod.tpr",
            "printf '%s\\n' 'using existing migrated extension tpr; no convert-tpr extend'",
        ]
    else:
        convert_lines = [
            f"EXTEND_PS={extend_ps:.0f}",
            f"EXT_TPR=nvt_prod/nvt_prod.extend_{stamp}.tpr",
            "\"$GMX_BIN\" convert-tpr -s nvt_prod/nvt_prod.tpr -extend \"$EXTEND_PS\" -o \"$EXT_TPR\"",
        ]
    provenance_json = json.dumps(provenance, ensure_ascii=False)
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {sh_quote(remote_run_dir)}",
            "test -d nvt_prod",
            "test -s nvt_prod/nvt_prod.tpr",
            "test -s nvt_prod/nvt_prod.cpt",
            "test -s run_manual.sh",
            "if ! grep -q 'Finished mdrun' pr/pr*.log 2>/dev/null; then echo 'ERROR: pr is not complete' >&2; exit 4; fi",
            (
                "if command -v pwdx >/dev/null 2>&1; then "
                "for p in $(pgrep -u \"$USER\" -f mdrun || true); do pwdx \"$p\" 2>/dev/null || true; done "
                f"| grep -F {sh_quote(remote_run_dir + '/nvt_prod')} >/dev/null && "
                "echo 'ERROR: nvt_prod mdrun already active in this directory' >&2 && exit 7 || true; fi"
            ),
            module_line,
            "GMX_BIN_FROM_SCRIPT=$(grep '^GMX_BIN=' run_manual.sh | head -n 1 | cut -d= -f2- | tr -d '\"')",
            f"GMX_BIN={sh_quote(gmx_bin)}",
            "[ -n \"$GMX_BIN\" ] || GMX_BIN=\"$GMX_BIN_FROM_SCRIPT\"",
            "test -x \"$GMX_BIN\" || command -v \"$GMX_BIN\" >/dev/null 2>&1",
            *convert_lines,
            "python3 - <<'PY'",
            "from pathlib import Path",
            f"Path('pelcb_clean_nvt_extension_{stamp}.json').write_text({provenance_json!r} + '\\n')",
            "PY",
            f"cp -p run_manual.sh run_manual.sh.bak_nvt_extension_{stamp} 2>/dev/null || true",
            f"cat > run_manual.sh <<'RUNMANUAL'\n{run_manual}\nRUNMANUAL",
            "chmod +x run_manual.sh",
            "printf '%s\\n' 'prepared supervisor-style extension run_manual.sh'",
        ]
    )


def launch_extension(row: dict[str, Any], stamp: str, reason: str, extend_ns: float) -> dict[str, Any]:
    node = str(row.get("target_node") or row.get("node") or "").strip()
    gpu_id = str(row.get("target_gpu_id") or row.get("gpu_id") or "").strip()
    if not node:
        return {"ok": False, "error": "missing target_node"}
    if not gpu_id:
        return {"ok": False, "error": "missing target_gpu_id"}
    stage_result: dict[str, Any] | None = None
    if str(row.get("capacity_status", "")) == "SLOT_ASSIGNED_STAGE_OLDSERVER_TO_NEW":
        staged_dir = newserver_extension_remote_dir(row, stamp)
        stage_result = stage_oldserver_checkpoint_to_newserver(row, staged_dir)
        if not stage_result.get("ok"):
            return {
                "ok": False,
                "error": "stage_oldserver_checkpoint_to_newserver_failed",
                "node": node,
                "gpu_id": gpu_id,
                "source_remote_run_dir": row.get("source_remote_run_dir") or row.get("remote_run_dir", ""),
                "remote_run_dir": staged_dir,
                "stage_result": stage_result,
            }
        row = {
            **row,
            "remote_run_dir": staged_dir,
            "staged_from_remote_run_dir": row.get("source_remote_run_dir") or row.get("remote_run_dir", ""),
            "migrated_tpr_is_extension": stage_result.get("source_tpr_kind") == "extended",
        }
    inner = build_extension_inner_command(row=row, stamp=stamp, reason=reason, extend_ns=extend_ns)
    node_command = f"bash -lc {sh_quote(inner)}"
    prepare_command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        NEW_HOST,
        f"ssh -o BatchMode=yes -o ConnectTimeout=20 {sh_quote(node)} {sh_quote(node_command)}",
    ]
    prepare = run_cmd(prepare_command, timeout=240)
    if prepare.returncode != 0:
        return {
            "ok": False,
            "returncode": prepare.returncode,
            "node": node,
            "gpu_id": gpu_id,
            "threads": row.get("target_threads", ""),
            "remote_run_dir": row.get("remote_run_dir", ""),
            "source_remote_run_dir": row.get("staged_from_remote_run_dir", ""),
            "extend_ns": extend_ns,
            "stdout": prepare.stdout.strip()[-2000:],
            "stderr": prepare.stderr.strip()[-2000:],
            "stage_result": stage_result or {},
            "launch_style": "supervisor_prepare_failed",
        }
    launch_inner = queue.supervisor.remote_launch_command(str(row.get("remote_run_dir", "")))
    launch_node_command = f"bash -lc {sh_quote(launch_inner)}"
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        NEW_HOST,
        f"ssh -o BatchMode=yes -o ConnectTimeout=20 {sh_quote(node)} {sh_quote(launch_node_command)}",
    ]
    result = run_cmd(command, timeout=240)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "node": node,
        "gpu_id": gpu_id,
        "threads": row.get("target_threads", ""),
        "remote_run_dir": row.get("remote_run_dir", ""),
        "source_remote_run_dir": row.get("staged_from_remote_run_dir", ""),
        "extend_ns": extend_ns,
        "stdout": result.stdout.strip()[-2000:],
        "stderr": result.stderr.strip()[-2000:],
        "stage_result": stage_result or {},
        "prepare_stdout": prepare.stdout.strip()[-2000:],
        "prepare_stderr": prepare.stderr.strip()[-2000:],
        "launch_style": "supervisor_remote_launch_command",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("PELCB_PROJECT_CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--queue-dir", type=Path, default=SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_cuda2024_supervisor_queue")
    parser.add_argument("--tag", default="nvt_extension_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--extend-ns", type=float, default=400.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allowlist-tag", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--runtime-max-age-seconds", type=int, default=1800)
    parser.add_argument("--dashboard-timeout-sec", type=int, default=180)
    parser.add_argument("--preferred-node", default="")
    parser.add_argument("--allow-second-per-gpu", action="store_true")
    parser.add_argument("--enable-oldserver-fallover", action="store_true")
    parser.add_argument("--refit-classification-csv", type=Path, default=DEFAULT_REFIT_CLASSIFICATION)
    parser.add_argument("--oldserver-max-gpu-jobs", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue.configure_windows_ssh_path()
    if args.execute:
        if args.allowlist_tag != "PELCB_CLEAN_NVT_EXTENSION":
            raise PermissionError("execution requires --allowlist-tag PELCB_CLEAN_NVT_EXTENSION")
        if len(args.reason.strip()) < 12:
            raise ValueError("execution requires --reason with at least 12 characters")
        if args.extend_ns <= 0:
            raise ValueError("--extend-ns must be positive")

    classification = read_classification(args.refit_classification_csv)
    refit_status = read_refit_status()
    case_filter = set(args.cases)
    rows = dedupe_status_rows([
        row
        for row in read_csv_rows(args.status_csv)
        if (not case_filter or row.get("case_id", "") in case_filter)
        and eligible_status_row(row)
    ])
    for row in rows:
        row["refit_extend_classification"] = classification.get(str(row.get("case_id", "")).strip(), "")
        row["strict_refit_status"] = refit_status.get(str(row.get("case_id", "")).strip(), "")
    slots: list[dict[str, Any]] = []
    runtime_status: dict[str, Any] = {}
    dashboard_status: dict[str, Any] = {}
    oldserver_status: dict[str, Any] = {}
    if args.execute:
        slots, runtime_status, dashboard_status = supervisor_free_slots(
            args.config,
            args.queue_dir,
            runtime_max_age_seconds=int(args.runtime_max_age_seconds),
            dashboard_timeout_sec=int(args.dashboard_timeout_sec),
            preferred_node=str(args.preferred_node),
            allow_second_per_gpu=bool(args.allow_second_per_gpu),
        )
        active_case_ids = active_case_ids_from_runtime_dashboard()
        seed_rows = [dict(row, capacity_status="SLOT_PENDING") for row in rows]
        apply_refit_before_extend_skip(seed_rows)
        apply_active_duplicate_skip(seed_rows, active_case_ids)
        planned = assign_slots(seed_rows, slots)
        oldserver_source_present = any(
            str(row.get("remote_run_dir", "") or "").startswith("${PELCB100_REMOTE_ROOT}/")
            for row in planned
        )
        if args.enable_oldserver_fallover and (not slots or oldserver_source_present):
            oldserver_status = oldserver_slurm_snapshot()
            for row in planned:
                source_dir = str(row.get("source_remote_run_dir") or row.get("remote_run_dir", "") or "")
                if not source_dir.startswith("${PELCB100_REMOTE_ROOT}/"):
                    continue
                if oldserver_case_active(row, oldserver_status):
                    row["capacity_status"] = "OLDSERVER_INTELGPU_ACTIVE"
                    row["target_cluster"] = "o7_clean_oldserver_intelgpu"
                    row["target_node"] = "slurm:intel"
                    row["target_gpu_id"] = "slurm"
            old_limit = max(0, min(int(args.oldserver_max_gpu_jobs), int(oldserver_status.get("idle_gpu_slots", 0) or 0)))
            assigned = 0
            for row in planned:
                if row.get("capacity_status") != "BLOCKED_NO_FREE_GPU_SLOT":
                    continue
                if oldserver_case_active(row, oldserver_status):
                    row["capacity_status"] = "OLDSERVER_INTELGPU_ACTIVE"
                    row["target_cluster"] = "o7_clean_oldserver_intelgpu"
                    row["target_node"] = "slurm:intel"
                    row["target_gpu_id"] = "slurm"
                    continue
                if assigned >= old_limit:
                    continue
                row["capacity_status"] = "OLDSERVER_INTELGPU_SLURM_ASSIGNED"
                row["target_cluster"] = "o7_clean_oldserver_intelgpu"
                row["target_node"] = "slurm:intel"
                row["target_gpu_id"] = "slurm"
                row["target_threads"] = 2
                row["target_gmx_bin"] = OLD_INTEL_GMX
                row["target_engine_module"] = OLD_INTEL_MODULE
                assigned += 1
        apply_active_duplicate_skip(planned, active_case_ids)
    else:
        planned = [dict(row, capacity_status="NOT_CHECKED") for row in rows]

    out_dir = args.output_dir / args.tag
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_rows: list[dict[str, Any]] = []
    for row in planned:
        record: dict[str, Any] = {
            "timestamp": iso_now(),
            "case_id": row.get("case_id", ""),
            "clean_case_id": row.get("clean_case_id", ""),
            "tag": row.get("tag", ""),
            "remote_run_dir": row.get("remote_run_dir", ""),
            "source_remote_run_dir": row.get("source_remote_run_dir", ""),
            "source_node": row.get("node", ""),
            "source_gpu_id": row.get("gpu_id", ""),
            "refit_extend_classification": row.get("refit_extend_classification", ""),
            "strict_refit_status": row.get("strict_refit_status", ""),
            "target_cluster": row.get("target_cluster", "o7_clean"),
            "target_node": row.get("target_node", row.get("node", "")),
            "target_gpu_id": row.get("target_gpu_id", row.get("gpu_id", "")),
            "target_threads": row.get("target_threads", row.get("threads", "")),
            "target_remote_root": row.get("target_remote_root", ""),
            "capacity_status": row.get("capacity_status", ""),
            "extend_ns": args.extend_ns,
            "execute": bool(args.execute),
            "action": "extend_clean_nvt_prod_from_checkpoint",
        }
        if args.execute:
            if row.get("capacity_status") == "BLOCKED_NO_FREE_GPU_SLOT":
                record["result"] = {"ok": False, "error": "no eligible free clean-engine GPU slot"}
            elif row.get("capacity_status") == "SKIPPED_REFIT_BEFORE_EXTEND":
                record["result"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "strict refit already produced publication-grade MSD evidence",
                }
            elif row.get("capacity_status") == "SKIPPED_ACTIVE_CASE_ALREADY_RUNNING":
                record["result"] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "case already has an active clean MD process in runtime dashboard",
                }
            elif row.get("capacity_status") == "OLDSERVER_INTELGPU_SLURM_ASSIGNED":
                record["result"] = submit_oldserver_extension(row, stamp, args.reason, args.extend_ns)
            elif row.get("capacity_status") == "OLDSERVER_INTELGPU_ACTIVE":
                record["result"] = {
                    "ok": False,
                    "active": True,
                    "reason": "matching oldserver Slurm job already active",
                }
            else:
                record["result"] = launch_extension(row, stamp, args.reason, args.extend_ns)
        output_rows.append(record)

    fields = [
        "timestamp",
        "case_id",
        "clean_case_id",
        "tag",
        "remote_run_dir",
        "source_remote_run_dir",
        "source_node",
        "source_gpu_id",
        "refit_extend_classification",
        "strict_refit_status",
        "target_cluster",
        "target_node",
        "target_gpu_id",
        "target_threads",
        "target_remote_root",
        "capacity_status",
        "extend_ns",
        "execute",
        "action",
        "result",
    ]
    write_csv(out_dir / "pelcb_clean_nvt_extension_plan.csv", output_rows, fields)
    oldserver_queue_plan = write_oldserver_queue_plan(args.queue_dir, args.tag, output_rows)
    summary = {
        "tag": args.tag,
        "created_at": iso_now(),
        "execute": bool(args.execute),
        "status_csv": str(args.status_csv),
        "selected_count": len(rows),
        "launched_count": sum(1 for row in output_rows if isinstance(row.get("result"), dict) and row["result"].get("ok")),
        "extend_ns": args.extend_ns,
        "available_slots": slots,
        "oldserver_status": oldserver_status,
        "runtime_status": runtime_status,
        "dashboard_status": dashboard_status,
        "plan_csv": str(out_dir / "pelcb_clean_nvt_extension_plan.csv"),
        "oldserver_queue_plan": oldserver_queue_plan,
        "rows": output_rows,
    }
    write_json(out_dir / "pelcb_clean_nvt_extension_summary.json", summary)
    write_json(args.output_dir / "pelcb_clean_nvt_extension_latest.json", summary)
    print(json.dumps({"tag": args.tag, "execute": bool(args.execute), "selected_count": len(rows), "launched_count": summary["launched_count"], "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
