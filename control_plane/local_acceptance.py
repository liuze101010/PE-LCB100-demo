from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import PACKAGE_ROOT, Settings
from .science_core_runtime import analysis_path


_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root(settings: Settings) -> Path:
    if settings.project_root:
        return settings.project_root
    for parent in (PACKAGE_ROOT, *PACKAGE_ROOT.parents):
        if (parent / "generated_code" / "lcb_md_pipeline" / "run_real_smoke.py").is_file():
            return parent
    bundled_root = analysis_path().parents[1]
    if (bundled_root / "generated_code" / "lcb_md_pipeline" / "run_real_smoke.py").is_file():
        return bundled_root
    raise RuntimeError("project root with smoke assets was not found")


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if key not in {"owner_id", "output_dir", "log"}
    }


def _set(job_id: str, **values: Any) -> None:
    with _LOCK:
        values.setdefault("updated_at", _utc_now())
        _JOBS[job_id].update(values)


def _execute(job_id: str, command: list[str], output_dir: Path) -> None:
    _set(job_id, status="running", stage="real_md", message="正在运行短程 GROMACS MD", started_at=_utc_now())
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=900, check=False)
        log = ((result.stdout or "") + "\n" + (result.stderr or ""))[-6000:]
        if result.returncode != 0:
            _set(job_id, status="failed", stage="failed", message="闭环验收失败", error_code="pipeline_core_failed", log=log)
            return
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            _set(job_id, status="failed", stage="failed", message="闭环流程未生成 summary", error_code="summary_missing", log=log)
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _set(
            job_id,
            status="done",
            stage="release",
            message="验收完成，终态发布已校验",
            result={
                key: summary.get(key)
                for key in [
                    "publication_state",
                    "case_status",
                    "density_g_cm3",
                    "rule_verdict",
                    "audit_diagnosis",
                    "audit_route",
                ]
            },
            log=log,
        )
    except subprocess.TimeoutExpired:
        _set(job_id, status="failed", stage="failed", message="闭环验收超时", error_code="timeout")
    except Exception as exc:  # noqa: BLE001
        _set(job_id, status="failed", stage="failed", message="闭环验收失败", error_code=type(exc).__name__)


def start(owner_id: str, settings: Settings) -> dict[str, Any]:
    with _LOCK:
        active = next((job for job in _JOBS.values() if job["owner_id"] == owner_id and job["status"] in {"queued", "running"}), None)
        if active:
            return _public(active)
        job_id = uuid4().hex[:12]
        output_dir = PACKAGE_ROOT / ".runtime" / "local_acceptance" / job_id
        output_dir.mkdir(parents=True, exist_ok=False)
        job = {
            "job_id": job_id,
            "owner_id": owner_id,
            "status": "queued",
            "stage": "queued",
            "message": "闭环验收已排队",
            "output_dir": str(output_dir),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        _JOBS[job_id] = job
    root = _root(settings)
    core = PACKAGE_ROOT / "deploy" / "pelcb_pipeline_core.py"
    command = [
        sys.executable,
        str(core),
        "acceptance-smoke",
        "--project-root",
        str(root),
        "--output",
        str(output_dir),
    ]
    configured_gmx = os.getenv("POLYMER_AGENT_LOCAL_GMX")
    if configured_gmx:
        command.extend(["--gmx", configured_gmx])
    threading.Thread(target=_execute, args=(job_id, command, output_dir), daemon=True).start()
    return _public(job)


def status(job_id: str, owner_id: str, *, admin: bool = False) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job or (not admin and job["owner_id"] != owner_id):
            return None
        return _public(job)


def artifact(job_id: str, owner_id: str, name: str, *, admin: bool = False) -> Path | None:
    if name not in {"release.json", "result.md"}:
        return None
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job or (not admin and job["owner_id"] != owner_id) or job["status"] != "done":
            return None
        path = Path(job["output_dir"]) / name
        return path if path.is_file() else None
