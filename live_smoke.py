"""Optional real-compute smoke hook for the demo (upgrade 5).

OFF BY DEFAULT. When the user opts in with `POLYMER_AGENT_REAL_SMOKE=1`, the demo
can trigger the standalone `run_real_smoke.py` runner as a tracked background job
and report status to the page. This is the only path in the demo that performs a
"write" (a remote Slurm submit), so it is guarded, off by default, and the remote
mode still requires an explicit confirm flag at the API layer.

Two job modes mirror the runner:
- ``reproduce`` : extract an exact density from a completed trajectory (local gmx,
                  no cluster). Fast, fully real.
- ``smoke``     : generate a smoke package and submit it to the remote cluster.
                  Minutes + queue; needs SSH key auth on the host running the demo.

Default-off keeps the demo self-contained: with the env unset, the endpoint
returns ``enabled: false`` and no subprocess/cluster contact happens.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent


def _project_root() -> Path | None:
    configured = os.getenv("POLYMER_AGENT_ROOT")
    if configured:
        return Path(configured)
    for parent in (PACKAGE_ROOT, *PACKAGE_ROOT.parents):
        if (parent / "generated_code" / "lcb_md_pipeline" / "run_real_smoke.py").exists():
            return parent
    return None


def runner_path() -> Path | None:
    root = _project_root()
    if not root:
        return None
    candidate = root / "generated_code" / "lcb_md_pipeline" / "run_real_smoke.py"
    return candidate if candidate.exists() else None


def smoke_enabled() -> bool:
    """True only when explicitly opted in AND the runner script is present."""
    flag = (os.getenv("POLYMER_AGENT_REAL_SMOKE") or "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    return runner_path() is not None


# --------------------------------------------------------------------------- #
# tiny in-process job registry (thread-backed)
# --------------------------------------------------------------------------- #

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _set(job_id: str, **fields: Any) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {})["job_id"] = job_id
        _JOBS[job_id].update(fields)


def get_status(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _run_job(job_id: str, cmd: list[str]) -> None:
    _set(job_id, status="running")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-4000:]
        if proc.returncode != 0:
            _set(job_id, status="failed", returncode=proc.returncode, log=tail)
            return
        result = _parse_result_json(proc.stdout or "")
        _set(job_id, status="done", result=result, log=tail)
    except Exception as exc:  # noqa: BLE001
        _set(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")


def _parse_result_json(stdout: str) -> dict[str, Any] | None:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return None


def start_job(*, mode: str, case_id: str, traj: str | None = None, bridge_dir: str | None = None) -> dict[str, Any]:
    """Launch run_real_smoke.py in the background. Returns {job_id, status}."""
    runner = runner_path()
    if runner is None:
        return {"enabled": False, "reason": "real-smoke runner not found"}

    job_id = uuid.uuid4().hex[:12]
    cmd = [sys.executable, str(runner), "--mode", mode, "--case", case_id]
    if mode == "reproduce":
        if not traj:
            return {"enabled": True, "error": "reproduce mode needs a traj path"}
        cmd += ["--traj", traj]
    else:  # smoke
        if bridge_dir:
            cmd += ["--bridge-dir", bridge_dir]
        cmd += ["--yes"]  # the API-layer confirm gate already approved this

    _set(job_id, status="queued", mode=mode, case_id=case_id)
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()
    return {"enabled": True, "job_id": job_id, "status": "queued", "mode": mode}
