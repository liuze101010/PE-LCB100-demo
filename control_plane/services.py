from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, model_configuration, model_stage_readiness
from .models import (
    ClusterConnection,
    LoginSession,
    ModelCredential,
    Project,
    ReleaseSnapshot,
    ReviewRecord,
    RunEvent,
    User,
    WorkflowRun,
)
from .schemas import ClusterCreate, ModelKeyUpdate, ProjectCreate, RunApproval
from .science import PELCBAdapter, build_release_payload
from .security import decrypt_secret, encrypt_secret, generate_ssh_keypair, hash_password, sha256_json
from .state import require_transition


def json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def bootstrap_admin(db: Session, settings: Settings) -> None:
    if not settings.bootstrap_admin or not settings.bootstrap_password:
        return
    existing = db.scalar(select(User).where(User.username == settings.bootstrap_admin))
    if existing:
        if settings.bootstrap_reset:
            existing.password_hash = hash_password(settings.bootstrap_password)
            existing.role = "admin"
            existing.active = True
            db.commit()
        seed_model_keys_from_env(db, existing, settings)
        return
    admin = User(
        username=settings.bootstrap_admin,
        password_hash=hash_password(settings.bootstrap_password),
        role="admin",
    )
    db.add(admin)
    db.commit()
    seed_model_keys_from_env(db, admin, settings)


def serialize_user(user: User) -> dict[str, Any]:
    return {"id": user.id, "username": user.username, "role": user.role, "active": user.active}


MODEL_KEY_FIELDS = {
    "openai": "encrypted_openai_key",
    "deepseek": "encrypted_deepseek_key",
    "dashscope": "encrypted_dashscope_key",
    "zhipuai": "encrypted_zhipuai_key",
}


def _model_credentials(db: Session, owner: User) -> ModelCredential | None:
    return db.scalar(select(ModelCredential).where(ModelCredential.owner_id == owner.id))


def model_key_public(db: Session, owner: User) -> dict[str, Any]:
    row = _model_credentials(db, owner)
    configured = {
        provider: bool(row and getattr(row, field))
        for provider, field in MODEL_KEY_FIELDS.items()
    }
    models = model_configuration()
    providers = {
        "openai": {"label": f"{models['design']} / upper planner", "model": models["design"], "configured": configured["openai"]},
        "deepseek": {"label": "DeepSeek audit vote", "model": models["deepseek"], "configured": configured["deepseek"]},
        "dashscope": {"label": "Qwen audit vote", "model": models["dashscope"], "configured": configured["dashscope"]},
        "zhipuai": {"label": "GLM audit vote", "model": models["zhipuai"], "configured": configured["zhipuai"]},
    }
    return {
        "schema_version": "polymer_agent.model_credentials.v1",
        "providers": providers,
        "openai_configured": configured["openai"],
        **model_stage_readiness(configured),
        "models": models,
        "updated_at": row.updated_at.isoformat() if row else None,
        "status": json_load(row.status_json, {}) if row else {},
    }


def upsert_model_keys(db: Session, owner: User, payload: ModelKeyUpdate, settings: Settings) -> dict[str, Any]:
    row = _model_credentials(db, owner)
    if row is None:
        row = ModelCredential(owner_id=owner.id)
        db.add(row)
    updates = {
        "encrypted_openai_key": payload.openai_api_key,
        "encrypted_deepseek_key": payload.deepseek_api_key,
        "encrypted_dashscope_key": payload.dashscope_api_key,
        "encrypted_zhipuai_key": payload.zhipuai_api_key,
    }
    changed = []
    for field, value in updates.items():
        if value:
            setattr(row, field, encrypt_secret(value, settings.credential_key))
            changed.append(field.replace("encrypted_", "").replace("_key", ""))
    row.status_json = json.dumps({"last_update": utc_iso(), "changed": changed}, ensure_ascii=False)
    db.commit()
    return model_key_public(db, owner)


ENV_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "zhipuai": "ZHIPUAI_API_KEY",
}


def seed_model_keys_from_env(db: Session, owner: User, settings: Settings) -> list[str]:
    """Fill empty model-credential slots from environment variables.

    Only empty slots are populated; keys entered through the UI are never
    overwritten. Set POLYMER_AGENT_SEED_KEYS_FROM_ENV=0 to disable.
    Returns the list of providers that were seeded.
    """
    if os.getenv("POLYMER_AGENT_SEED_KEYS_FROM_ENV", "1").strip().lower() in {"0", "false", "no"}:
        return []
    row = _model_credentials(db, owner)
    seeded: list[str] = []
    for provider, env_var in ENV_KEY_VARS.items():
        value = (os.getenv(env_var) or "").strip()
        if not value:
            continue
        field = MODEL_KEY_FIELDS[provider]
        if row is not None and getattr(row, field):
            continue  # keep any key already stored (e.g. entered via the UI)
        if row is None:
            row = ModelCredential(owner_id=owner.id)
            db.add(row)
        setattr(row, field, encrypt_secret(value, settings.credential_key))
        seeded.append(provider)
    if seeded:
        row.status_json = json.dumps(
            {"last_update": utc_iso(), "changed": seeded, "source": "environment"},
            ensure_ascii=False,
        )
        db.commit()
    return seeded


def clear_model_keys(db: Session, owner: User) -> dict[str, Any]:
    row = _model_credentials(db, owner)
    if row is not None:
        db.delete(row)
        db.commit()
    return model_key_public(db, owner)


def model_api_key_for_user(db: Session, owner: User, provider: str, settings: Settings) -> str | None:
    row = _model_credentials(db, owner)
    field = MODEL_KEY_FIELDS.get(provider)
    encrypted = getattr(row, field, None) if row is not None and field else None
    return decrypt_secret(encrypted, settings.credential_key) if encrypted else None


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_cluster(db: Session, owner: User, payload: ClusterCreate, settings: Settings) -> ClusterConnection:
    private_key, public_key = generate_ssh_keypair()
    limits = {
        "max_cpus": payload.max_cpus,
        "max_gpus": payload.max_gpus,
        "max_walltime_hours": payload.max_walltime_hours,
        "allowed_nodes": payload.allowed_nodes,
        "environment": {
            "mode": payload.environment_mode,
            "container_runtime": payload.container_runtime,
            "container_image": payload.container_image,
            "pipeline_command": payload.pipeline_command,
            "science_install_mode": payload.science_install_mode,
            "gromacs_acceleration": payload.gromacs_acceleration,
            "upstream_terms_accepted": payload.upstream_terms_accepted,
        },
        "provisioning": {"status": "not_started"},
    }
    cluster = ClusterConnection(
        owner_id=owner.id,
        name=payload.name,
        host=payload.host,
        port=payload.port,
        remote_username=payload.remote_username,
        host_key_sha256=payload.host_key_sha256,
        encrypted_private_key=encrypt_secret(private_key, settings.credential_key),
        public_key=public_key,
        partition=payload.partition,
        account=payload.account,
        remote_root=payload.remote_root,
        gromacs_command=payload.gromacs_command,
        emc_command=payload.emc_command,
        python_command=payload.python_command,
        runner_command=payload.runner_command,
        module_init_json=json.dumps(payload.module_init),
        resource_limits_json=json.dumps(limits),
    )
    db.add(cluster)
    db.commit()
    db.refresh(cluster)
    return cluster


def cluster_public(cluster: ClusterConnection) -> dict[str, Any]:
    limits = json_load(cluster.resource_limits_json, {})
    return {
        "id": cluster.id,
        "name": cluster.name,
        "host": cluster.host,
        "port": cluster.port,
        "remote_username": cluster.remote_username,
        "host_key_sha256": cluster.host_key_sha256,
        "public_key": cluster.public_key,
        "partition": cluster.partition,
        "account": cluster.account,
        "status": cluster.status,
        "last_preflight": json_load(cluster.last_preflight_json, None),
        "resource_limits": {
            key: value for key, value in limits.items() if key not in {"environment", "provisioning"}
        },
        "environment": limits.get("environment", {"mode": "module", "pipeline_command": "pelcb-pipeline-core"}),
        "provisioning": limits.get("provisioning", {"status": "not_started"}),
    }


def create_project(db: Session, owner: User, payload: ProjectCreate) -> Project:
    project = Project(
        owner_id=owner.id,
        name=payload.name,
        spec_json=payload.spec.model_dump_json(),
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def project_public(project: Project) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "status": project.status,
        "spec": json_load(project.spec_json, {}),
        "manifest": json_load(project.manifest_json, []),
        "manifest_sha256": project.manifest_sha256,
        "domain_status": project.domain_status,
        "validation": json_load(project.validation_json, None),
    }


def validate_project(db: Session, project: Project) -> dict[str, Any]:
    from .schemas import ProjectSpec

    result = PELCBAdapter().validate(ProjectSpec.model_validate_json(project.spec_json))
    payload = {
        "valid": result.valid,
        "domain_status": result.domain_status,
        "rule_layer_entry": "PASS" if result.domain_status == "validated_domain" else "WARN",
        "errors": result.errors,
        "warnings": result.warnings,
        "manifest_sha256": result.manifest_sha256,
    }
    project.validation_json = json.dumps(payload, ensure_ascii=False)
    project.manifest_json = json.dumps(result.manifest, ensure_ascii=False)
    project.manifest_sha256 = result.manifest_sha256
    project.domain_status = result.domain_status
    project.status = "validated" if result.valid else "draft"
    db.commit()
    return payload


def lock_project(db: Session, project: Project) -> Project:
    if project.status != "validated" or not project.manifest_sha256:
        raise ValueError("project must pass validation before locking")
    project.status = "locked"
    db.commit()
    return project


def event(db: Session, run: WorkflowRun, actor_id: str | None, event_type: str, payload: dict[str, Any]) -> None:
    db.add(
        RunEvent(
            run_id=run.id,
            actor_id=actor_id,
            event_type=event_type,
            payload_json=json.dumps(payload, ensure_ascii=False),
        )
    )


def create_run(db: Session, owner: User, project: Project, cluster: ClusterConnection) -> WorkflowRun:
    if project.status != "locked":
        raise ValueError("project must be locked before a run is planned")
    if cluster.status != "ready":
        raise ValueError("cluster must pass preflight")
    run = WorkflowRun(project_id=project.id, owner_id=owner.id, cluster_id=cluster.id)
    db.add(run)
    db.flush()
    require_transition(run.state, "awaiting_approval")
    run.state = "awaiting_approval"
    event(db, run, owner.id, "run_planned", {"manifest_sha256": project.manifest_sha256})
    db.commit()
    db.refresh(run)
    return run


def run_public(run: WorkflowRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "project_id": run.project_id,
        "cluster_id": run.cluster_id,
        "state": run.state,
        "approval": json_load(run.approval_json, None),
        "remote_job_id": run.remote_job_id,
        "error_code": run.error_code,
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


def approve_run(db: Session, run: WorkflowRun, owner: User, approval: RunApproval) -> WorkflowRun:
    require_transition(run.state, "queued")
    run.approval_json = approval.model_dump_json()
    run.state = "queued"
    event(db, run, owner.id, "execution_approved", approval.model_dump())
    db.commit()
    return run


def build_release(db: Session, owner: User, project: Project, run: WorkflowRun | None) -> ReleaseSnapshot:
    events = []
    if run:
        rows = db.scalars(select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.created_at)).all()
        events = [
            {"type": row.event_type, "payload": json_load(row.payload_json, {}), "created_at": row.created_at.isoformat()}
            for row in rows
        ]
    review_rows = db.scalars(select(ReviewRecord).where(ReviewRecord.project_id == project.id)).all()
    reviews = [
        {"reason_code": row.reason_code, "status": row.status, "packet": json_load(row.packet_json, {})}
        for row in review_rows
    ]
    project_payload = project_public(project)
    payload = build_release_payload(
        project=project_payload,
        run=run_public(run) if run else None,
        events=events,
        reviews=reviews,
    )
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode()
    previous = db.scalars(
        select(ReleaseSnapshot).where(ReleaseSnapshot.project_id == project.id).order_by(ReleaseSnapshot.version.desc())
    ).first()
    release = ReleaseSnapshot(
        project_id=project.id,
        run_id=run.id if run else None,
        owner_id=owner.id,
        version=(previous.version + 1) if previous else 1,
        payload_json=raw.decode(),
        payload_sha256=sha256_json(raw),
    )
    db.add(release)
    db.commit()
    db.refresh(release)
    return release
