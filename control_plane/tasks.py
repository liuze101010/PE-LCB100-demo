from __future__ import annotations

import json

from .config import load_settings
from .db import SessionLocal
from .models import ClusterConnection, Project, RunEvent, RunStageJob, WorkflowRun
from .slurm import SSHSlurmBackend, runtime_from_model


def submit_stage(
    db,
    *,
    run: WorkflowRun,
    project: Project,
    cluster: ClusterConnection,
    stage: str,
    attempt: int = 1,
) -> RunStageJob:
    settings = load_settings()
    approval = json.loads(run.approval_json or "{}")
    manifest = json.loads(project.manifest_json)
    backend = SSHSlurmBackend(runtime_from_model(cluster, settings.credential_key))
    gpus = int(approval.get("gpus", 1)) if stage == "md" else 0
    command = [
        backend.runtime.runner_command,
        "stage",
        "--stage",
        stage,
        "--project-spec",
        "project.json",
        "--manifest",
        "manifest.json",
        "--work-root",
        "work",
    ]
    max_parallel = min(len(manifest), int(approval.get("max_parallel_cases", 8)))
    script = backend.render_array_script(
        run_id=run.id,
        stage=stage,
        command=command,
        case_count=len(manifest),
        max_parallel=max_parallel,
        cpus=int(approval.get("cpus_per_task", 8)),
        gpus=gpus,
        walltime_hours=int(approval.get("walltime_hours", 168)),
    )
    files = {}
    if stage == "build":
        files = {
            "project.json": json.dumps(json.loads(project.spec_json), ensure_ascii=False, indent=2).encode(),
            "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2).encode(),
        }
    job_id, remote_dir = backend.submit_stage_bundle(run_id=run.id, stage=stage, files=files, script=script)
    record = RunStageJob(
        run_id=run.id,
        stage=stage,
        attempt=attempt,
        state="running",
        slurm_job_id=job_id,
        remote_workdir=remote_dir,
        case_count=len(manifest),
        details_json=json.dumps({"max_parallel_cases": max_parallel, "gpus_per_case": gpus}),
    )
    db.add(record)
    run.remote_job_id = job_id
    run.remote_workdir = remote_dir
    db.add(
        RunEvent(
            run_id=run.id,
            event_type="stage_submitted",
            payload_json=json.dumps({"stage": stage, "attempt": attempt, "job_id": job_id, "case_count": len(manifest)}),
        )
    )
    return record


def submit_repair_stage(
    db,
    *,
    run: WorkflowRun,
    project: Project,
    cluster: ClusterConnection,
    repair_cases: list[dict],
    attempt: int = 1,
) -> RunStageJob:
    if not repair_cases:
        raise ValueError("repair stage requires at least one case")
    settings = load_settings()
    approval = json.loads(run.approval_json or "{}")
    backend = SSHSlurmBackend(runtime_from_model(cluster, settings.credential_key))
    command = [
        backend.runtime.runner_command,
        "stage",
        "--stage",
        "repair",
        "--project-spec",
        "project.json",
        "--manifest",
        "repair_manifest.json",
        "--work-root",
        "work",
    ]
    max_parallel = min(len(repair_cases), int(approval.get("max_parallel_cases", 8)))
    script = backend.render_array_script(
        run_id=run.id,
        stage="repair",
        command=command,
        case_count=len(repair_cases),
        max_parallel=max_parallel,
        cpus=int(approval.get("cpus_per_task", 8)),
        gpus=int(approval.get("gpus", 1)),
        walltime_hours=int(approval.get("walltime_hours", 168)),
    )
    files = {
        "project.json": json.dumps(json.loads(project.spec_json), ensure_ascii=False, indent=2).encode(),
        "repair_manifest.json": json.dumps(repair_cases, ensure_ascii=False, indent=2).encode(),
    }
    job_id, remote_dir = backend.submit_stage_bundle(run_id=run.id, stage="repair", files=files, script=script)
    record = RunStageJob(
        run_id=run.id,
        stage="repair",
        attempt=attempt,
        state="running",
        slurm_job_id=job_id,
        remote_workdir=remote_dir,
        case_count=len(repair_cases),
        details_json=json.dumps({"case_ids": [row["case_id"] for row in repair_cases], "max_parallel_cases": max_parallel}),
    )
    db.add(record)
    run.state = "repairing"
    run.remote_job_id = job_id
    db.add(RunEvent(run_id=run.id, event_type="repair_stage_submitted", payload_json=json.dumps({"job_id": job_id, "case_count": len(repair_cases)})))
    return record


def execute_run(run_id: str) -> dict[str, str]:
    settings = load_settings()
    if not settings.executor_enabled:
        raise RuntimeError("real executor is disabled")
    with SessionLocal() as db:
        run = db.get(WorkflowRun, run_id)
        if run is None:
            raise KeyError(run_id)
        if run.state != "queued":
            return {"run_id": run_id, "state": run.state}
        project = db.get(Project, run.project_id)
        cluster = db.get(ClusterConnection, run.cluster_id)
        if project is None or cluster is None:
            raise RuntimeError("run references missing project or cluster")
        approval = json.loads(run.approval_json or "{}")
        manifest = json.loads(project.manifest_json)
        if len(manifest) > int(approval.get("max_cases", 0)):
            raise RuntimeError("manifest exceeds approved case budget")
        try:
            stage_job = submit_stage(db, run=run, project=project, cluster=cluster, stage="build")
        except Exception as exc:
            run.state = "failed"
            run.error_code = "slurm_submit_failed"
            db.add(
                RunEvent(
                    run_id=run.id,
                    event_type="submit_failed",
                    payload_json=json.dumps({"error": type(exc).__name__}),
                )
            )
            db.commit()
            raise
        run.state = "running"
        db.commit()
        return {"run_id": run.id, "state": run.state, "job_id": str(stage_job.slurm_job_id)}


def enqueue_run(run_id: str) -> str:
    settings = load_settings()
    from redis import Redis
    from rq import Queue

    queue = Queue("polymer-agent", connection=Redis.from_url(settings.redis_url))
    job = queue.enqueue(execute_run, run_id, job_timeout="24h", result_ttl=86400)
    return job.id
