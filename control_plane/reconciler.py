from __future__ import annotations

import hashlib
import json
import time
from pathlib import PurePosixPath

from sqlalchemy import select

from .config import load_settings
from .db import SessionLocal, init_database
from .models import ClusterConnection, ModelCredential, ReleaseSnapshot, ReviewRecord, RunArtifact, RunEvent, RunStageJob, WorkflowRun
from .slurm import SSHSlurmBackend, runtime_from_model
from .science import compare_release_with_frozen_oracle, validate_remote_release
from .models import Project
from .tasks import submit_repair_stage, submit_stage
from .production_evaluation import evaluate_case
from .science_core_runtime import PACKAGE_ROOT
from .production_audit import run_audit_panel
from .security import decrypt_secret
from ..live_execution_plan import build_upper_repair_plan, guard_upper_repair_plan


FAILURE_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}


def store_artifact(db, run_id: str, name: str, payload: dict) -> RunArtifact:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    artifact = db.scalar(select(RunArtifact).where(RunArtifact.run_id == run_id, RunArtifact.name == name))
    if artifact is None:
        artifact = RunArtifact(run_id=run_id, name=name, payload_json=text, payload_sha256=digest)
        db.add(artifact)
    else:
        artifact.payload_json = text
        artifact.payload_sha256 = digest
    return artifact


def collect_initial_evaluation(db, run: WorkflowRun, project: Project, backend: SSHSlurmBackend) -> dict:
    manifest = json.loads(project.manifest_json)
    base = PurePosixPath(run.remote_workdir or "") / "work" / "cases"
    paths = [str(base / str(case["case_id"]) / "canonical_evidence" / "evidence.json") for case in manifest]
    blobs = backend.read_remote_files(paths)
    profile_dir = PACKAGE_ROOT / ".runtime" / "rule_profiles" / (project.manifest_sha256 or project.id)
    evaluations = []
    for case, path in zip(manifest, paths, strict=True):
        raw = json.loads(blobs[path])
        evaluations.append(evaluate_case(raw, case, profile_dir))
    counts: dict[str, int] = {}
    for item in evaluations:
        verdict = str(item["rule_verdict"])
        counts[verdict] = counts.get(verdict, 0) + 1
    payload = {
        "schema_version": "polymer_agent.initial_rule_evaluation.v1",
        "manifest_sha256": project.manifest_sha256,
        "case_count": len(evaluations),
        "counts": counts,
        "cases": evaluations,
    }
    store_artifact(db, run.id, "initial_rule_evaluation", payload)
    return payload


def artifact_payload(db, run_id: str, name: str) -> dict | None:
    row = db.scalar(select(RunArtifact).where(RunArtifact.run_id == run_id, RunArtifact.name == name))
    return json.loads(row.payload_json) if row else None


def model_keys(db, run: WorkflowRun, settings) -> dict[str, str]:
    row = db.scalar(select(ModelCredential).where(ModelCredential.owner_id == run.owner_id))
    if row is None:
        raise RuntimeError("model credentials are missing")
    fields = {
        "openai": row.encrypted_openai_key,
        "deepseek": row.encrypted_deepseek_key,
        "dashscope": row.encrypted_dashscope_key,
        "zhipuai": row.encrypted_zhipuai_key,
    }
    if not all(fields.values()):
        raise RuntimeError("all four model credentials are required for the closed loop")
    return {name: decrypt_secret(str(value), settings.credential_key) for name, value in fields.items()}


def make_release(db, run: WorkflowRun, project: Project, evaluations: list[dict], audits: dict[str, dict], repairs: dict[str, str]) -> dict:
    cases = []
    review_ids = []
    failed_ids = []
    for item in evaluations:
        case_id = str(item["case_id"])
        rule_verdict = str(item.get("rule_verdict") or "WARN")
        passed = rule_verdict == "PASS"
        failed = rule_verdict == "FAIL"
        generic_domains = (item.get("generic_qc") or {}).get("domain_verdicts") or {}
        density_pass = (item.get("density_rule") or {}).get("rule_verdict") == "PASS"
        case_payload = {
            "case_id": case_id,
            "quality_status": "pass" if passed else "failed" if failed else "review",
            "rule_verdict": rule_verdict,
            "warn_sources": list(item.get("warn_sources") or []),
            "density_g_cm3": (item.get("evidence") or {}).get("density_g_cm3"),
            "property_masks": {
                "density": density_pass,
                "static": all(generic_domains.get(name) == "PASS" for name in ("rg", "shape", "rdf", "contact")),
                "diffusion_strict": generic_domains.get("msd") == "PASS",
            },
            "audit_votes": (audits.get(case_id, {}).get("diagnosis") or {}).get("model_votes", []),
            "audit_diagnosis": (audits.get(case_id, {}).get("diagnosis") or {}).get("audit_diagnosis"),
            "repair_history": ([{"attempt": 1, "contract": repairs[case_id]}] if case_id in repairs else []),
            "msd_warn": "msd_warn" in (item.get("warn_sources") or []),
            "msd_total_ns": 800 if repairs.get(case_id) == "clean_msd_bounded_extension" else 400,
            "review_reason": (
                None
                if passed
                else "unsupported_or_failed_configuration"
                if failed
                else ",".join(item.get("warn_sources") or ["persistent_warn"])
            ),
        }
        cases.append(case_payload)
        if failed:
            failed_ids.append(case_id)
        elif not passed:
            review_ids.append(case_id)
    config = {
        "project_spec": json.loads(project.spec_json),
        "manifest_sha256": project.manifest_sha256,
        "run_approval": json.loads(run.approval_json or "{}"),
    }
    payload = {
        "schema_version": "polymer_agent.remote_release.v1",
        "publication_state": "review" if review_ids or failed_ids else "pass",
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "project": {
            "id": project.id,
            "name": project.name,
            "status": project.status,
            "domain_status": project.domain_status,
            "manifest_sha256": project.manifest_sha256,
        },
        "run": {"id": run.id, "state": "review" if review_ids or failed_ids else "pass"},
        "provenance": [
            {"stage": "locked_manifest", "sha256": project.manifest_sha256},
            {"stage": "registered_rules", "case_count": len(cases)},
            {"stage": "tri_model_audit", "audited_case_count": len(audits)},
            {"stage": "bounded_repair", "repaired_case_count": len(repairs)},
            {"stage": "rule_regate", "terminal_case_count": len(cases)},
        ],
        "cases": cases,
        "reviews": [{"case_id": case_id, "reason_code": "persistent_warn", "status": "open"} for case_id in review_ids],
        "summary": {
            "case_count": len(cases),
            "pass": len(cases) - len(review_ids) - len(failed_ids),
            "review": len(review_ids),
            "failed": len(failed_ids),
        },
    }
    payload["benchmark_reproducibility"] = compare_release_with_frozen_oracle(cases)
    validate_remote_release(payload, project_domain_status=project.domain_status)
    previous = db.scalars(
        select(ReleaseSnapshot).where(ReleaseSnapshot.project_id == project.id).order_by(ReleaseSnapshot.version.desc())
    ).first()
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    release = ReleaseSnapshot(
        project_id=project.id,
        run_id=run.id,
        owner_id=run.owner_id,
        version=(previous.version + 1) if previous else 1,
        payload_json=raw,
        payload_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )
    db.add(release)
    db.flush()
    for case_id in review_ids:
        db.add(
            ReviewRecord(
                project_id=project.id,
                run_id=run.id,
                owner_id=run.owner_id,
                reason_code="persistent_warn",
                packet_json=json.dumps(next(item for item in cases if item["case_id"] == case_id), ensure_ascii=False),
            )
        )
    run.state = payload["publication_state"]
    db.add(RunEvent(run_id=run.id, event_type="release_collected", payload_json=json.dumps({"release_id": release.id, **payload["summary"]})))
    return payload


def audit_and_plan(db, run: WorkflowRun, project: Project, cluster: ClusterConnection, settings) -> None:
    initial = artifact_payload(db, run.id, "initial_rule_evaluation")
    if initial is None:
        raise RuntimeError("initial rule evaluation is missing")
    warnings = [item for item in initial["cases"] if item.get("rule_verdict") == "WARN"]
    if not warnings:
        make_release(db, run, project, initial["cases"], {}, {})
        return
    keys = model_keys(db, run, settings)
    approval = json.loads(run.approval_json or "{}")
    audit_payload = artifact_payload(db, run.id, "audit_panel")
    if audit_payload is None:
        cache_dir = PACKAGE_ROOT / ".runtime" / "audit_cache" / run.id
        audits = [
            run_audit_panel(
                item,
                deepseek_key=keys["deepseek"],
                dashscope_key=keys["dashscope"],
                zhipuai_key=keys["zhipuai"],
                cache_dir=cache_dir,
                approval=approval,
            )
            for item in warnings
        ]
        audit_payload = {"schema_version": "polymer_agent.audit_panel.v1", "cases": audits}
        store_artifact(db, run.id, "audit_panel", audit_payload)
    audits_by_id = {item["case_id"]: item for item in audit_payload["cases"]}
    plan_payload = artifact_payload(db, run.id, "upper_repair_plan")
    if plan_payload is None:
        prompt_cases = [
            {
                "case_id": case_id,
                "warn_sources": next(item["warn_sources"] for item in warnings if item["case_id"] == case_id),
                "audit_diagnosis": (audit["diagnosis"] or {}).get("audit_diagnosis"),
                "audit_route": (audit["diagnosis"] or {}).get("recommended_reentry_module"),
                "model_panel_complete": (audit["diagnosis"] or {}).get("model_panel_complete"),
                "cross_model_disagreement": (audit["diagnosis"] or {}).get("cross_model_disagreement"),
                "allowed_contracts": audit["allowed_contracts"],
            }
            for case_id, audit in audits_by_id.items()
        ]
        proposed = build_upper_repair_plan(prompt_cases, approval, api_key=keys["openai"])
        allowed = {case_id: set(item["allowed_contracts"]) for case_id, item in audits_by_id.items()}
        guarded = guard_upper_repair_plan(proposed, allowed)
        plan_payload = {**proposed, "guard": guarded}
        store_artifact(db, run.id, "upper_repair_plan", plan_payload)
    manifest_by_id = {row["case_id"]: row for row in json.loads(project.manifest_json)}
    selected = dict((plan_payload.get("guard") or {}).get("accepted") or {})
    repair_cases = [
        {**manifest_by_id[case_id], "repair_contract": contract, "repair_attempt": 1}
        for case_id, contract in selected.items()
        if contract != "review_hold"
    ]
    if repair_cases:
        submit_repair_stage(db, run=run, project=project, cluster=cluster, repair_cases=repair_cases)
    else:
        make_release(db, run, project, initial["cases"], audits_by_id, {})


def collect_repair_evaluation(db, run: WorkflowRun, project: Project, backend: SSHSlurmBackend) -> dict:
    initial = artifact_payload(db, run.id, "initial_rule_evaluation")
    plan = artifact_payload(db, run.id, "upper_repair_plan")
    audits = artifact_payload(db, run.id, "audit_panel") or {"cases": []}
    if initial is None or plan is None:
        raise RuntimeError("repair collection metadata is incomplete")
    selected = dict((plan.get("guard") or {}).get("accepted") or {})
    repair_ids = [case_id for case_id, contract in selected.items() if contract != "review_hold"]
    base = PurePosixPath(run.remote_workdir or "") / "work" / "cases"
    paths = [str(base / case_id / "repair_overlays" / "attempt_1" / "evidence.json") for case_id in repair_ids]
    blobs = backend.read_remote_files(paths) if paths else {}
    manifest_by_id = {row["case_id"]: row for row in json.loads(project.manifest_json)}
    profile_dir = PACKAGE_ROOT / ".runtime" / "rule_profiles" / (project.manifest_sha256 or project.id)
    repaired = {
        case_id: evaluate_case(json.loads(blobs[path]), manifest_by_id[case_id], profile_dir)
        for case_id, path in zip(repair_ids, paths, strict=True)
    }
    final = [repaired.get(item["case_id"], item) for item in initial["cases"]]
    payload = {
        "schema_version": "polymer_agent.post_repair_rule_evaluation.v1",
        "cases": final,
        "summary": {
            "case_count": len(final),
            "pass": sum(item["rule_verdict"] == "PASS" for item in final),
            "review": sum(item["rule_verdict"] == "WARN" for item in final),
            "failed": sum(item["rule_verdict"] == "FAIL" for item in final),
        },
    }
    store_artifact(db, run.id, "post_repair_rule_evaluation", payload)
    audits_by_id = {item["case_id"]: item for item in audits["cases"]}
    repairs = {case_id: selected[case_id] for case_id in repair_ids}
    make_release(db, run, project, final, audits_by_id, repairs)
    return payload


def reconcile_once() -> int:
    settings = load_settings()
    updated = 0
    with SessionLocal() as db:
        runs = db.scalars(select(WorkflowRun).where(WorkflowRun.state.in_(["running", "collecting", "postprocessing", "evaluating", "repairing"]))).all()
        for run in runs:
            cluster = db.get(ClusterConnection, run.cluster_id)
            if cluster is None:
                continue
            stage_job = db.scalars(
                select(RunStageJob)
                .where(RunStageJob.run_id == run.id, RunStageJob.state == "running")
                .order_by(RunStageJob.created_at.desc())
            ).first()
            backend = SSHSlurmBackend(runtime_from_model(cluster, settings.credential_key))
            if stage_job is None:
                if run.state == "collecting":
                    project = db.get(Project, run.project_id)
                    if project is None:
                        run.state = "failed"
                        run.error_code = "missing_project"
                    else:
                        try:
                            evaluation = collect_initial_evaluation(db, run, project, backend)
                            run.state = "evaluating"
                            db.add(RunEvent(run_id=run.id, event_type="initial_rules_evaluated", payload_json=json.dumps({"counts": evaluation["counts"]})))
                        except Exception as exc:
                            run.state = "failed"
                            run.error_code = "evidence_collection_failed"
                            db.add(RunEvent(run_id=run.id, event_type="evaluation_failed", payload_json=json.dumps({"error": type(exc).__name__})))
                        updated += 1
                elif run.state == "evaluating":
                    project = db.get(Project, run.project_id)
                    if project is None:
                        run.state = "failed"
                        run.error_code = "missing_project"
                    else:
                        try:
                            if artifact_payload(db, run.id, "upper_repair_plan") and db.scalar(
                                select(RunStageJob).where(RunStageJob.run_id == run.id, RunStageJob.stage == "repair", RunStageJob.state == "completed")
                            ):
                                collect_repair_evaluation(db, run, project, backend)
                            else:
                                audit_and_plan(db, run, project, cluster, settings)
                        except Exception as exc:
                            run.state = "failed"
                            run.error_code = "closed_loop_evaluation_failed"
                            db.add(RunEvent(run_id=run.id, event_type="closed_loop_failed", payload_json=json.dumps({"error": type(exc).__name__})))
                        updated += 1
                continue
            if not stage_job.slurm_job_id:
                continue
            try:
                remote_state = backend.status(stage_job.slurm_job_id)
            except Exception as exc:
                db.add(RunEvent(run_id=run.id, event_type="reconcile_error", payload_json=json.dumps({"error": type(exc).__name__})))
                continue
            db.add(RunEvent(run_id=run.id, event_type="slurm_state", payload_json=json.dumps({"stage": stage_job.stage, "state": remote_state})))
            if remote_state == "COMPLETED":
                stage_job.state = "completed"
                project = db.get(Project, run.project_id)
                if project is None:
                    run.state = "failed"
                    run.error_code = "missing_project"
                elif stage_job.stage == "build":
                    submit_stage(db, run=run, project=project, cluster=cluster, stage="md")
                elif stage_job.stage == "md":
                    run.state = "postprocessing"
                    submit_stage(db, run=run, project=project, cluster=cluster, stage="postprocess")
                elif stage_job.stage == "postprocess":
                    run.state = "collecting"
                    db.add(RunEvent(run_id=run.id, event_type="evidence_ready", payload_json=json.dumps({"case_count": stage_job.case_count})))
                elif stage_job.stage == "repair":
                    run.state = "evaluating"
                    db.add(RunEvent(run_id=run.id, event_type="repair_evidence_ready", payload_json=json.dumps({"case_count": stage_job.case_count})))
                updated += 1
            elif remote_state == "CANCELLED":
                run.state = "cancelled"
                updated += 1
            elif remote_state in FAILURE_STATES:
                stage_job.state = "failed"
                project = db.get(Project, run.project_id)
                if remote_state in {"TIMEOUT", "NODE_FAIL"} and stage_job.attempt < 3 and project is not None:
                    if stage_job.stage == "repair":
                        plan = artifact_payload(db, run.id, "upper_repair_plan") or {}
                        selected = dict((plan.get("guard") or {}).get("accepted") or {})
                        manifest_by_id = {row["case_id"]: row for row in json.loads(project.manifest_json)}
                        repair_cases = [
                            {**manifest_by_id[case_id], "repair_contract": contract, "repair_attempt": 1}
                            for case_id, contract in selected.items()
                            if contract != "review_hold"
                        ]
                        submit_repair_stage(db, run=run, project=project, cluster=cluster, repair_cases=repair_cases, attempt=stage_job.attempt + 1)
                    else:
                        submit_stage(db, run=run, project=project, cluster=cluster, stage=stage_job.stage, attempt=stage_job.attempt + 1)
                    db.add(RunEvent(run_id=run.id, event_type="stage_resubmitted", payload_json=json.dumps({"stage": stage_job.stage, "attempt": stage_job.attempt + 1, "reason": remote_state})))
                else:
                    run.state = "failed"
                    run.error_code = "slurm_" + stage_job.stage + "_" + remote_state.lower()
                updated += 1
        db.commit()
    return updated


def main() -> None:
    init_database()
    while True:
        reconcile_once()
        time.sleep(60)


if __name__ == "__main__":
    main()
