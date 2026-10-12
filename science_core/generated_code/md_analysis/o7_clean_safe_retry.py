#!/usr/bin/env python3
"""Safe retry/resume helper for incomplete clean-engine runs.

This tool is intentionally narrow.  It does not rebuild EMC inputs, does not
edit MDP physics, and does not touch the PE-LCB100 canonical release.  For an
existing clean sidecar run directory with a fatal stage, it archives the failed
stage directory, patches the launch script, and relaunches the resume-aware
four-stage script.  For checkpoint-pending stages without a fatal marker, it
relaunches the same script without archiving that stage directory so the
existing checkpoint remains available.  Completed previous stages are skipped
by run_manual.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from lcb_project_config import load_lcb_config
import o7_clean_supervisor_queue as queue


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "pelcb100.json"
NEW_HOST = "${REMOTE_USER}@private-4e2f23ff.invalid"
DEFAULT_STATUS_CSV = (
    SCRIPT_DIR
    / "output"
    / "pelcb100_supervisor"
    / "o7_clean_cuda2024_closure"
    / "o7_clean_run_status.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "o7_clean_safe_retry"
STAGES = ("steep", "rlx", "pr", "nvt_prod")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sh_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def fatal_stage(row: dict[str, str]) -> tuple[str, str] | None:
    for stage in reversed(STAGES):
        value = str(row.get(f"{stage}_fatal_log", "") or "").strip()
        if value and value.lower() not in {"nan", "none", "null"}:
            return stage, value
    return None


def checkpoint_resume_stage(row: dict[str, str]) -> tuple[str, str] | None:
    """Return the earliest incomplete stage with a valid checkpoint pair.

    A stage is resumable only when its immediately preceding stage is complete
    and both its TPR and CPT are present.  The immediate predecessor is the
    authoritative chain guard and remains compatible with historical status
    rows that did not materialize every earlier ``*_done`` field.  Requiring
    the checkpoint pair avoids treating an EDR-only dirty directory as a
    checkpoint.  The four-stage runner then performs the actual
    ``mdrun -cpi -noappend`` continuation.
    """
    if str(row.get("nvt_prod_done", "")).lower() == "true":
        return None
    remote_run_dir = str(row.get("remote_run_dir", "") or "").strip()
    if not remote_run_dir:
        return None
    if fatal_stage(row):
        return None

    for stage in reversed(STAGES):
        if str(row.get(f"{stage}_done", "")).lower() == "true":
            continue
        has_tpr = str(row.get(f"{stage}_has_tpr", "")).lower() == "true"
        has_cpt = str(row.get(f"{stage}_has_cpt", "")).lower() == "true"
        predecessor = previous_stage(stage)
        predecessor_done = predecessor is None or str(row.get(f"{predecessor}_done", "")).lower() == "true"
        if predecessor_done and has_tpr and has_cpt:
            return stage, f"checkpoint/incomplete {stage} without fatal marker"
    return None


def previous_stage(stage: str) -> str | None:
    """Return the immediately preceding pipeline stage."""
    if stage not in STAGES:
        raise ValueError(f"unsupported clean stage: {stage}")
    index = STAGES.index(stage)
    return STAGES[index - 1] if index else None


def build_safe_retry_inner_command(
    remote_run_dir: str,
    case_id: str,
    stage: str,
    stamp: str,
    reason: str,
    mode: str = "gpu_nb_only",
    gpu_id: str | None = None,
    threads: int | None = None,
    retry_kind: str = "clean_runtime_fatal",
    archive_incomplete_stage: bool = True,
) -> str:
    """Return a bash command executed inside the remote run directory."""
    predecessor = previous_stage(stage)
    provenance = {
        "case_id": case_id,
        "retry_kind": f"{retry_kind}_{mode}",
        "failed_stage": stage,
        "reason": reason,
        "created_at": stamp,
        "mdp_physics_changed": False,
        "script_changes": {},
    }
    if mode == "cpu_only":
        provenance["script_changes"] = {
            "MDRUN_GPU_ARGS": "-nb cpu",
            "ENABLE_STEEP_GPU": "0",
            "ENABLE_GPU_EXTRA": "0 at launch",
        }
        script_patch = "\n".join(
            [
                "python3 - <<'PY'",
                "from pathlib import Path",
                "import re",
                "p = Path('run_manual.sh')",
                "text = p.read_text()",
                "text = re.sub(r'^MDRUN_GPU_ARGS=.*$', 'MDRUN_GPU_ARGS=\"-nb cpu\"', text, flags=re.M)",
                "text = re.sub(r'^ENABLE_STEEP_GPU=.*$', 'ENABLE_STEEP_GPU=\"0\"', text, flags=re.M)",
                "p.write_text(text)",
                "PY",
            ]
        )
        launch_env = "ENABLE_GPU_EXTRA=0 MERGE_PARTS=1"
    elif mode == "gpu_nb_only":
        provenance["script_changes"] = {
            "GPU_EXTRA_NVT_PROD": "",
            "ENABLE_GPU_EXTRA": "1 at launch",
            "MDRUN_GPU_ARGS": (
                f"-gpu_id {gpu_id} -nb gpu" if gpu_id is not None else "preserve assigned -gpu_id <id> -nb gpu"
            ),
        }
        if gpu_id is not None:
            provenance["target_gpu_id"] = str(gpu_id)
        if threads is not None:
            provenance["target_threads"] = int(threads)
        patch_lines = [
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import re",
            "p = Path('run_manual.sh')",
            "text = p.read_text()",
        ]
        if gpu_id is not None:
            patch_lines.append(
                f"text = re.sub(r'^MDRUN_GPU_ARGS=.*$', 'MDRUN_GPU_ARGS=\"-gpu_id {str(gpu_id)} -nb gpu\"', text, flags=re.M)"
            )
        if threads is not None:
            patch_lines.append(
                f"text = re.sub(r'^THREADS=.*$', 'THREADS=\"{int(threads)}\"', text, flags=re.M)"
            )
        patch_lines.extend(["p.write_text(text)", "PY"])
        script_patch = "\n".join(patch_lines) if gpu_id is not None or threads is not None else "true"
        launch_env = "ENABLE_GPU_EXTRA=1 GPU_EXTRA_NVT_PROD= MERGE_PARTS=1"
    else:
        raise ValueError(f"unsupported safe retry mode: {mode}")
    provenance_json = json.dumps(provenance, ensure_ascii=False)
    archive_command = (
        f"if [ -d {sh_quote(stage)} ] && ! grep -q 'Finished mdrun' "
        f"{sh_quote(stage)}/{sh_quote(stage)}*.log 2>/dev/null; then "
        f"mv {sh_quote(stage)} {sh_quote(stage + '.failed_safe_retry_' + stamp)}; fi"
        if archive_incomplete_stage
        else "true"
    )
    predecessor_guard: list[str] = []
    if predecessor:
        predecessor_guard = [
            f"test -d {sh_quote(predecessor)}",
            (
                f"if ! grep -q 'Finished mdrun' {sh_quote(predecessor)}/{sh_quote(predecessor)}*.log "
                f"2>/dev/null; then echo 'ERROR: {predecessor} is not complete; refusing {stage} retry' "
                ">&2; exit 4; fi"
            ),
        ]
    lock_dir = ".o7_safe_retry.active.lock"
    wrapper_name = f"o7_safe_retry_wrapper_{stamp}.sh"
    wrapper_text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -o pipefail",
            f"LOCK_DIR={sh_quote(lock_dir)}",
            "cleanup_lock() { rm -f \"$LOCK_DIR/pid\"; rmdir \"$LOCK_DIR\" 2>/dev/null || true; }",
            "trap cleanup_lock EXIT",
            f"env {launch_env} bash run_manual.sh",
        ]
    )
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {sh_quote(remote_run_dir)}",
            "test -s run_manual.sh",
            *predecessor_guard,
            (
                f"if ! mkdir {sh_quote(lock_dir)}; then "
                "echo 'ERROR: active safe-retry lock exists; refusing duplicate launch' >&2; exit 6; fi"
            ),
            f"trap 'rmdir {sh_quote(lock_dir)} 2>/dev/null || true' EXIT",
            script_patch,
            archive_command,
            "python3 - <<'PY'",
            "from pathlib import Path",
            f"Path('o7_safe_retry_provenance_{stamp}.json').write_text({provenance_json!r} + '\\n')",
            "PY",
            "python3 - <<'PY'",
            "from pathlib import Path",
            f"Path({wrapper_name!r}).write_text({wrapper_text!r} + '\\n')",
            "PY",
            f"chmod 700 {sh_quote(wrapper_name)}",
            (
                f"nohup bash {sh_quote(wrapper_name)} "
                f"> manual_safe_retry_{stamp}.out 2> manual_safe_retry_{stamp}.err < /dev/null &"
            ),
            f"retry_pid=$!",
            f"printf '%s\\n' \"$retry_pid\" > {sh_quote(lock_dir + '/pid')}",
            "trap - EXIT",
            "printf '%s\\n' \"$retry_pid\"",
        ]
    )


def run_cmd(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)


def plan_safe_retry_targets(
    rows: list[dict[str, str]],
    slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign safe retries to live supervisor slots without overfilling a GPU."""
    planned: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item: dict[str, Any] = dict(row)
        if index >= len(slots):
            item["capacity_status"] = "BLOCKED_NO_SLOTS"
            item["target_node"] = ""
            item["target_gpu_id"] = ""
            item["target_threads"] = ""
        else:
            slot = slots[index]
            item["capacity_status"] = "SLOT_ASSIGNED"
            item["target_node"] = str(slot.get("node", ""))
            item["target_gpu_id"] = str(slot.get("gpu", ""))
            item["target_threads"] = int(slot.get("threads", 1))
        planned.append(item)
    return planned


def prioritize_slots(slots: list[dict[str, Any]], preferred_node: str = "") -> list[dict[str, Any]]:
    """Return slots with the preferred node first while preserving policy order."""
    preferred = str(preferred_node or "").strip()
    if not preferred:
        return list(slots)
    return sorted(slots, key=lambda slot: 0 if str(slot.get("node", "")) == preferred else 1)


def supervisor_slots_for_safe_retry(
    config_path: Path,
    *,
    runtime_max_age_seconds: int = 1800,
    dashboard_timeout_sec: int = 90,
    preferred_node: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return live clean-engine slots using the same policy as the supervisor."""
    config = load_lcb_config(config_path)
    cluster_profile = queue.resolve_o7_cluster_profile(config, "o7_clean")
    queue.setup_supervisor_globals(config_path, config.o7_queue_dir, cluster_profile=cluster_profile)
    dashboard = queue.refresh_dashboard_for_queue(config_path, timeout_sec=dashboard_timeout_sec)
    runtime_status = queue.runtime_snapshot_status(
        queue.supervisor.RUNTIME_DIR,
        max_age_seconds=runtime_max_age_seconds,
    )
    # A fresh dashboard from this invocation is mandatory before any launch.
    # A previously fresh CSV can miss a process that started seconds ago.
    if not dashboard.get("ok"):
        runtime_status["ok"] = False
        runtime_status["reason"] = "runtime_refresh_failed_fail_safe"
        runtime_status["submission_allowed"] = False
        return [], runtime_status, dashboard
    if not runtime_status.get("ok"):
        runtime_status["submission_allowed"] = False
        return [], runtime_status, dashboard
    runtime_status["submission_allowed"] = True
    runtime = queue.supervisor.load_runtime()
    reservations = queue.recent_queue_reservations(
        config.o7_queue_dir,
        max_age_seconds=runtime_max_age_seconds,
    )
    runtime_status["queue_reservation_count"] = len(reservations)
    slots = queue.choose_clean_slots(
        runtime,
        plan_rows=[],
        allow_second_per_gpu=True,
        cluster_profile=cluster_profile,
        reservation_rows=reservations,
    )
    return prioritize_slots(slots, preferred_node), runtime_status, dashboard


def launch_safe_retry(row: dict[str, str], stamp: str, reason: str, mode: str) -> dict[str, Any]:
    fatal_info = fatal_stage(row)
    checkpoint_info = checkpoint_resume_stage(row)
    stage, fatal_log = fatal_info or checkpoint_info or ("", "")
    remote_run_dir = str(row.get("remote_run_dir", "") or "").strip()
    node = str(row.get("target_node") or row.get("node", "") or "").strip()
    gpu_id = str(row.get("target_gpu_id") or row.get("gpu_id", "") or "").strip()
    threads_text = str(row.get("target_threads") or "").strip()
    threads = int(threads_text) if threads_text else None
    if not remote_run_dir:
        return {"ok": False, "error": "missing remote_run_dir"}
    if not node:
        return {"ok": False, "error": "missing node"}
    inner = build_safe_retry_inner_command(
        remote_run_dir,
        str(row.get("case_id", "")),
        stage,
        stamp,
        reason,
        mode,
        gpu_id=gpu_id if mode == "gpu_nb_only" and gpu_id else None,
        threads=threads if mode == "gpu_nb_only" else None,
        retry_kind=(
            "clean_runtime_fatal"
            if fatal_info
            else ("clean_nvt_checkpoint_resume" if stage == "nvt_prod" else f"clean_{stage}_checkpoint_resume")
        ),
        archive_incomplete_stage=bool(fatal_info),
    )
    node_command = f"bash -lc {sh_quote(inner)}"
    command = ["ssh", NEW_HOST, f"ssh -o BatchMode=yes -o ConnectTimeout=20 {sh_quote(node)} {sh_quote(node_command)}"]
    result = run_cmd(command, timeout=180)
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "node": node,
        "gpu_id": gpu_id,
        "threads": threads if threads is not None else "",
        "remote_run_dir": remote_run_dir,
        "failed_stage": stage,
        "fatal_log": fatal_log,
        "mode": mode,
        "stdout": result.stdout.strip()[-2000:],
        "stderr": result.stderr.strip()[-2000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("PELCB_PROJECT_CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="safe_retry_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allowlist-tag", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--mode", choices=("gpu_nb_only", "cpu_only"), default="gpu_nb_only")
    parser.add_argument("--runtime-max-age-seconds", type=int, default=1800)
    parser.add_argument("--dashboard-timeout-sec", type=int, default=90)
    parser.add_argument(
        "--preferred-node",
        default="node-d61b45bd",
        help="Prefer this node's eligible slots before using other supervisor-approved slots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(args.status_csv)
    selected: list[dict[str, str]] = []
    case_filter = set(args.cases)
    for row in rows:
        if case_filter and str(row.get("case_id", "")) not in case_filter:
            continue
        if str(row.get("nvt_prod_done", "")).lower() == "true":
            continue
        if fatal_stage(row) or checkpoint_resume_stage(row):
            selected.append(row)

    out_dir = args.output_dir / args.tag
    out_rows: list[dict[str, Any]] = []
    if args.execute:
        if args.allowlist_tag != "PELCB_CLEAN_SAFE_RETRY":
            raise PermissionError("execution requires --allowlist-tag PELCB_CLEAN_SAFE_RETRY")
        if len(args.reason.strip()) < 12:
            raise ValueError("execution requires --reason with at least 12 characters")

    slots: list[dict[str, Any]] = []
    runtime_status: dict[str, Any] = {}
    dashboard_status: dict[str, Any] = {}
    if args.execute and args.mode == "gpu_nb_only":
        slots, runtime_status, dashboard_status = supervisor_slots_for_safe_retry(
            args.config,
            runtime_max_age_seconds=int(args.runtime_max_age_seconds),
            dashboard_timeout_sec=int(args.dashboard_timeout_sec),
            preferred_node=str(args.preferred_node),
        )
        selected_with_targets = plan_safe_retry_targets(selected, slots)
    else:
        selected_with_targets = [dict(row, capacity_status="NOT_CHECKED") for row in selected]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for row in selected_with_targets:
        fatal_info = fatal_stage(row)
        checkpoint_info = checkpoint_resume_stage(row)
        stage, fatal_log = fatal_info or checkpoint_info or ("", "")
        record: dict[str, Any] = {
            "timestamp": iso_now(),
            "case_id": row.get("case_id", ""),
            "clean_case_id": row.get("clean_case_id", ""),
            "node": row.get("node", ""),
            "gpu_id": row.get("gpu_id", ""),
            "target_node": row.get("target_node", row.get("node", "")),
            "target_gpu_id": row.get("target_gpu_id", row.get("gpu_id", "")),
            "target_threads": row.get("target_threads", ""),
            "capacity_status": row.get("capacity_status", ""),
            "remote_run_dir": row.get("remote_run_dir", ""),
            "failed_stage": stage,
            "fatal_log": fatal_log,
            "mode": args.mode,
            "execute": bool(args.execute),
            "action": (
                "retry_failed_clean_stage_safe_offload"
                if fatal_info
                else f"resume_incomplete_clean_{stage}_from_checkpoint"
            ),
        }
        if args.execute:
            if row.get("capacity_status") == "BLOCKED_NO_SLOTS":
                record["action"] = "retry_waiting_for_supervisor_slot"
                record["result"] = {"ok": False, "error": "no eligible clean-engine GPU slot under supervisor policy"}
            else:
                record["result"] = launch_safe_retry(row, stamp, args.reason, args.mode)
        out_rows.append(record)

    fields = [
        "timestamp",
        "case_id",
        "clean_case_id",
        "node",
        "gpu_id",
        "target_node",
        "target_gpu_id",
        "target_threads",
        "capacity_status",
        "remote_run_dir",
        "failed_stage",
        "fatal_log",
        "mode",
        "execute",
        "action",
        "result",
    ]
    write_csv(out_dir / "o7_clean_safe_retry_plan.csv", out_rows, fields)
    write_json(
        out_dir / "o7_clean_safe_retry_summary.json",
        {
            "tag": args.tag,
            "created_at": iso_now(),
            "execute": bool(args.execute),
            "status_csv": str(args.status_csv),
            "selected_count": len(selected),
            "available_slots": slots,
            "runtime_status": runtime_status,
            "dashboard_status": dashboard_status,
            "plan_csv": str(out_dir / "o7_clean_safe_retry_plan.csv"),
            "rows": out_rows,
        },
    )
    print(
        json.dumps(
            {
                "tag": args.tag,
                "execute": bool(args.execute),
                "selected_count": len(selected),
                "output_dir": str(out_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
