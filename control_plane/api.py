from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .config import Settings, load_settings
from .db import get_db
from .models import ClusterConnection, LoginSession, Project, ReleaseSnapshot, ReviewRecord, RunEvent, User, WorkflowRun
from .schemas import (
    ClusterCreate,
    ClusterProvisionRequest,
    DesignDraftRequest,
    HostKeyLookup,
    LoginRequest,
    ModelKeyUpdate,
    PolymerSetupPreviewRequest,
    ProjectCreate,
    PublicKeyInstall,
    ReviewResolution,
    RunApproval,
    RunCreate,
    UserCreate,
)
from .security import hash_password, new_session, token_hash
from .auth import get_auth_backend
from .schemas import ProjectSpec
from .services import (
    approve_run,
    build_release,
    cluster_public,
    create_cluster,
    create_project,
    create_run,
    clear_model_keys,
    event,
    lock_project,
    model_api_key_for_user,
    model_key_public,
    project_public,
    run_public,
    serialize_user,
    upsert_model_keys,
    validate_project,
)
from .slurm import SSHSlurmBackend, install_public_key_with_password, runtime_from_model, scan_host_key_sha256
from .provisioning import provision_cluster
from .science import build_readable_release_markdown
from .local_acceptance import artifact as local_acceptance_artifact
from .local_acceptance import start as start_local_acceptance
from .local_acceptance import status as local_acceptance_status
from ..live_execution_plan import ExecutionPlanError, build_upper_execution_plan, planner_enabled, planner_required
from ..live_design import DesignPlanError, propose_designs, requested_count_from_goal


router = APIRouter(prefix="/api/v1", tags=["control-plane"])
Db = Annotated[Session, Depends(get_db)]


def _refresh_science_install(db: Session, cluster: ClusterConnection, backend: SSHSlurmBackend) -> dict[str, Any]:
    """Reconcile one bounded remote installer status into the cluster record."""
    science = backend.science_runtime_status()
    limits = json.loads(cluster.resource_limits_json)
    provisioning = dict(limits.get("provisioning") or {})
    provisioning["science_install"] = science
    status = str(science.get("status") or "not_started")
    preflight: dict[str, Any] | None = None
    if status == "complete":
        science_bin = f"{cluster.remote_root.rstrip('/')}/.polymer-agent/science/bin"
        cluster.gromacs_command = f"{science_bin}/gmx"
        cluster.emc_command = f"{science_bin}/emc"
        cluster.python_command = f"{science_bin}/python"
        db.flush()
        try:
            preflight = SSHSlurmBackend(runtime_from_model(cluster, load_settings().credential_key)).preflight()
        except Exception as exc:
            preflight = {"ok": False, "error": type(exc).__name__}
        cluster.last_preflight_json = json.dumps(preflight, ensure_ascii=False)
        cluster.status = "ready" if preflight.get("ok") else "needs_admin"
        provisioning["status"] = "installed" if preflight.get("ok") else "needs_admin"
        provisioning["admin_actions"] = list(preflight.get("admin_actions") or [])
    elif status in {"failed", "needs_admin"}:
        cluster.status = "needs_admin"
        provisioning["status"] = "needs_admin"
    elif status in {"starting", "running"}:
        cluster.status = "installing"
        provisioning["status"] = "installing"
    limits["provisioning"] = provisioning
    cluster.resource_limits_json = json.dumps(limits, ensure_ascii=False)
    db.commit()
    public = cluster_public(cluster)
    public["science_install"] = science
    if preflight is not None:
        public["preflight"] = preflight
    return public


def _session_user(
    db: Db,
    polymer_session: Annotated[str | None, Cookie()] = None,
) -> tuple[User, LoginSession]:
    if not polymer_session:
        raise HTTPException(status_code=401, detail="authentication_required")
    session = db.scalar(select(LoginSession).where(LoginSession.token_hash == token_hash(polymer_session)))
    now = datetime.now(timezone.utc)
    if session is None or session.expires_at.replace(tzinfo=timezone.utc) <= now:
        raise HTTPException(status_code=401, detail="session_expired")
    user = db.get(User, session.user_id)
    if user is None or not user.active:
        raise HTTPException(status_code=401, detail="account_disabled")
    return user, session


def current_user(pair: Annotated[tuple[User, LoginSession], Depends(_session_user)]) -> User:
    return pair[0]


def mutating_user(
    pair: Annotated[tuple[User, LoginSession], Depends(_session_user)],
    x_csrf_token: Annotated[str | None, Header()] = None,
) -> User:
    user, session = pair
    if not x_csrf_token or x_csrf_token != session.csrf_token:
        raise HTTPException(status_code=403, detail="csrf_check_failed")
    return user


def _owned(db: Session, model: Any, item_id: str, user: User):
    item = db.get(model, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="not_found")
    if user.role != "admin" and getattr(item, "owner_id", None) != user.id:
        raise HTTPException(status_code=404, detail="not_found")
    return item


@router.get("/health")
def platform_health():
    settings = load_settings()
    from ..live_design import design_enabled

    return {
        "status": "ok",
        "schema_version": "polymer_agent.control_plane.v1",
        "executor_enabled": settings.executor_enabled,
        "design_model_enabled": design_enabled(),
        "design_model": os.getenv("POLYMER_AGENT_DESIGN_MODEL") or "gpt-5.5",
        "upper_planner_enabled": planner_enabled(),
        "upper_planner_model": os.getenv("POLYMER_AGENT_UPPER_PLANNER_MODEL") or os.getenv("POLYMER_AGENT_DESIGN_MODEL") or "gpt-5.5",
        "auth_mode": "local",
    }


@router.post("/local-acceptance")
def local_acceptance_start(user: Annotated[User, Depends(mutating_user)]):
    try:
        return start_local_acceptance(user.id, load_settings())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail={"code": "local_acceptance_unavailable", "message": str(exc)}) from exc


@router.get("/local-acceptance/{job_id}")
def local_acceptance_get(job_id: str, user: Annotated[User, Depends(current_user)]):
    job = local_acceptance_status(job_id, user.id, admin=user.role == "admin")
    if job is None:
        raise HTTPException(status_code=404, detail="not_found")
    return job


@router.get("/local-acceptance/{job_id}/result")
def local_acceptance_result(job_id: str, user: Annotated[User, Depends(current_user)]):
    path = local_acceptance_artifact(job_id, user.id, "result.md", admin=user.role == "admin")
    if path is None:
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="text/markdown; charset=utf-8", filename="pelcb-local-acceptance-result.md")


@router.get("/local-acceptance/{job_id}/release")
def local_acceptance_release(job_id: str, user: Annotated[User, Depends(current_user)]):
    path = local_acceptance_artifact(job_id, user.id, "release.json", admin=user.role == "admin")
    if path is None:
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="application/json", filename="pelcb-local-acceptance-release.json")


@router.post("/auth/login")
def login(payload: LoginRequest, response: Response, db: Db):
    settings = load_settings()
    user = get_auth_backend().authenticate(db, payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token, csrf, expires = new_session(settings.session_hours)
    db.add(LoginSession(token_hash=token_hash(token), csrf_token=csrf, user_id=user.id, expires_at=expires))
    db.commit()
    response.set_cookie(
        "polymer_session",
        token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        max_age=settings.session_hours * 3600,
        path="/",
    )
    return {"user": serialize_user(user), "csrf_token": csrf, "expires_at": expires.isoformat()}


@router.post("/design-drafts")
def design_draft(payload: DesignDraftRequest, db: Db, user: Annotated[User, Depends(mutating_user)]):
    key_status = model_key_public(db, user)
    if not key_status["ready_for_design"]:
        raise HTTPException(status_code=409, detail="model_keys_required")
    try:
        requested_count = requested_count_from_goal(payload.goal)
        settings = load_settings()
        openai_key = model_api_key_for_user(db, user, "openai", settings)
        plan = propose_designs(payload.goal, requested_count, api_key=openai_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_design_count", "message": str(exc)}) from exc
    except DesignPlanError as exc:
        raise HTTPException(status_code=503, detail={"code": "design_model_unavailable", "message": str(exc)}) from exc
    planner: dict[str, Any] = {
        "engine": plan["engine"],
        "model": plan["model"],
        "response_id": plan["response_id"],
        "candidate_source": (
            "gpt_selected_locked_pelcb100_benchmark"
            if plan.get("mode") == "reproduce_pelcb100"
            else "gpt_generated_within_registered_pe_lcb_grammar"
        ),
        "requested_count": plan["requested_count"],
        "catalog_size": plan["catalog_size"],
        "catalog_sha256": plan["catalog_sha256"],
        "strategy_summary": plan["strategy_summary"],
        "attempt": plan["attempt"],
        "mode": plan.get("mode", "new_design"),
        "benchmark_manifest_sha256": plan.get("benchmark_manifest_sha256"),
    }
    spec = ProjectSpec(
        natural_language_goal=payload.goal,
        planning={key: value for key, value in planner.items() if key != "candidate_source"},
        candidates=plan["candidates"],
    )
    return {
        "status": "draft_only",
        "executable": False,
        "requires": ["structured_confirmation", "validation", "manifest_lock", "run_approval"],
        "planner": planner,
        "spec": spec.model_dump(),
    }


@router.post("/polymer-setup/preview")
def polymer_setup_preview(
    payload: PolymerSetupPreviewRequest,
    user: Annotated[User, Depends(mutating_user)],
):
    """Select a registered builder and force field without starting computation."""

    del user
    try:
        from .science_core_runtime import analysis_path

        analysis_path()
        from polymer_design_validator import validate_polymer_design_candidate
        from .production_evaluation import evaluate_preparation_support

        candidate = payload.model_dump(exclude_none=True)
        result = validate_polymer_design_candidate(candidate)
        preparation_case = {
            **candidate,
            "case_id": str(candidate.get("case_id") or "preparation_preview"),
            "builder_spec": result.get("builder_spec") or {},
            "automatic_setup": result.get("automatic_setup") or {},
        }
        preparation_rule = evaluate_preparation_support(preparation_case)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={
                "code": "polymer_setup_unavailable",
                "message": f"自动配置暂时不可用：{type(exc).__name__}",
            },
        ) from exc

    return {
        "status": result.get("status"),
        "validator_status": result.get("validator_status"),
        "valid": result.get("validator_status") == "pass",
        "builder_input_ready": result.get("emc_ready"),
        "production_ready": result.get("production_ready"),
        "next_action": result.get("next_action"),
        "candidate": result.get("normalized_candidate") or candidate,
        "builder_spec": result.get("builder_spec") or {},
        "automatic_setup": result.get("automatic_setup") or {},
        "preparation_rule": preparation_rule,
        "errors": result.get("errors") or [],
    }


@router.get("/model-keys")
def model_keys_get(db: Db, user: Annotated[User, Depends(current_user)]):
    return model_key_public(db, user)


@router.post("/model-keys")
def model_keys_save(payload: ModelKeyUpdate, db: Db, user: Annotated[User, Depends(mutating_user)]):
    return upsert_model_keys(db, user, payload, load_settings())


@router.delete("/model-keys")
def model_keys_delete(db: Db, user: Annotated[User, Depends(mutating_user)]):
    return clear_model_keys(db, user)


@router.post("/clusters/host-key")
def cluster_host_key(payload: HostKeyLookup, user: Annotated[User, Depends(mutating_user)]):
    try:
        fingerprint = scan_host_key_sha256(payload.host, payload.port)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"code": "host_key_scan_failed", "message": f"无法读取主机指纹：{type(exc).__name__}"},
        ) from exc
    return {"host": payload.host, "port": payload.port, "host_key_sha256": fingerprint}


@router.post("/auth/logout")
def logout(
    response: Response,
    db: Db,
    pair: Annotated[tuple[User, LoginSession], Depends(_session_user)],
    x_csrf_token: Annotated[str | None, Header()] = None,
):
    _, session = pair
    if x_csrf_token != session.csrf_token:
        raise HTTPException(status_code=403, detail="csrf_check_failed")
    db.delete(session)
    db.commit()
    response.delete_cookie("polymer_session", path="/")
    return {"status": "logged_out"}


@router.get("/auth/me")
def me(pair: Annotated[tuple[User, LoginSession], Depends(_session_user)]):
    user, session = pair
    return {**serialize_user(user), "csrf_token": session.csrf_token}


@router.post("/users")
def users_create(payload: UserCreate, db: Db, actor: Annotated[User, Depends(mutating_user)]):
    if actor.role != "admin":
        raise HTTPException(status_code=403, detail="admin_required")
    if db.scalar(select(User).where(User.username == payload.username)):
        raise HTTPException(status_code=409, detail="username_exists")
    user = User(username=payload.username, password_hash=hash_password(payload.password), role=payload.role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return serialize_user(user)


@router.post("/clusters")
def clusters_create(payload: ClusterCreate, db: Db, user: Annotated[User, Depends(mutating_user)]):
    cluster = create_cluster(db, user, payload, load_settings())
    if user.role == "admin":
        cluster.status = "authorized"
        db.commit()
        db.refresh(cluster)
    return cluster_public(cluster)


@router.post("/clusters/{cluster_id}/install-key")
def cluster_install_key(cluster_id: str, payload: PublicKeyInstall, db: Db, user: Annotated[User, Depends(mutating_user)]):
    cluster = _owned(db, ClusterConnection, cluster_id, user)
    try:
        result = install_public_key_with_password(
            host=cluster.host,
            port=cluster.port,
            username=cluster.remote_username,
            host_key_sha256=cluster.host_key_sha256,
            password=payload.password,
            public_key=cluster.public_key,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"code": "public_key_install_failed", "message": f"无法自动写入 SSH 公钥：{type(exc).__name__}"},
        ) from exc
    if not result.get("ok"):
        raise HTTPException(status_code=503, detail={"code": "public_key_install_failed", "message": "远端 authorized_keys 写入失败"})
    return {"status": "installed", "cluster": cluster_public(cluster), "result": result}


@router.get("/clusters")
def clusters_list(db: Db, user: Annotated[User, Depends(current_user)]):
    query = select(ClusterConnection)
    if user.role != "admin":
        query = query.where(ClusterConnection.owner_id == user.id)
    return [cluster_public(item) for item in db.scalars(query.order_by(ClusterConnection.created_at)).all()]


@router.post("/clusters/{cluster_id}/autoconfigure")
def cluster_autoconfigure(cluster_id: str, db: Db, user: Annotated[User, Depends(mutating_user)]):
    cluster = _owned(db, ClusterConnection, cluster_id, user)
    if cluster.status not in {"authorized", "ready", "failed", "needs_admin"}:
        raise HTTPException(status_code=409, detail="cluster_requires_admin_authorization")
    try:
        detected = SSHSlurmBackend(
            runtime_from_model(cluster, load_settings().credential_key)
        ).detect_site_defaults()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"code": "cluster_autoconfigure_failed", "message": f"无法自动识别服务器配置：{type(exc).__name__}"},
        ) from exc
    cluster.partition = detected["partition"]
    cluster.remote_root = detected["remote_root"]
    limits = json.loads(cluster.resource_limits_json)
    environment = dict(limits.get("environment") or {})
    environment["scheduler_mode"] = detected["scheduler_mode"]
    environment["manual_nodes"] = detected.get("manual_nodes", [])
    limits["environment"] = environment
    cluster.resource_limits_json = json.dumps(limits, ensure_ascii=False)
    db.commit()
    db.refresh(cluster)
    return {"cluster": cluster_public(cluster), "detected": detected}


@router.post("/clusters/{cluster_id}/authorize")
def cluster_authorize(cluster_id: str, db: Db, actor: Annotated[User, Depends(mutating_user)]):
    if actor.role != "admin":
        raise HTTPException(status_code=403, detail="admin_required")
    cluster = db.get(ClusterConnection, cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="not_found")
    cluster.status = "authorized"
    db.commit()
    return cluster_public(cluster)


@router.post("/clusters/{cluster_id}/preflight")
def cluster_preflight(cluster_id: str, db: Db, user: Annotated[User, Depends(mutating_user)]):
    cluster = _owned(db, ClusterConnection, cluster_id, user)
    if cluster.status not in {"authorized", "ready", "failed", "needs_admin"}:
        raise HTTPException(status_code=409, detail="cluster_requires_admin_authorization")
    try:
        result = SSHSlurmBackend(runtime_from_model(cluster, load_settings().credential_key)).preflight()
    except Exception as exc:
        result = {"ok": False, "error": type(exc).__name__}
    cluster.status = "ready" if result.get("ok") else "needs_admin"
    cluster.last_preflight_json = json.dumps(result, ensure_ascii=False)
    db.commit()
    return cluster_public(cluster)


@router.post("/clusters/{cluster_id}/discover")
def cluster_discover(cluster_id: str, db: Db, user: Annotated[User, Depends(mutating_user)]):
    cluster = _owned(db, ClusterConnection, cluster_id, user)
    if cluster.status not in {"authorized", "ready", "failed", "needs_admin"}:
        raise HTTPException(status_code=409, detail="cluster_requires_admin_authorization")
    try:
        result = SSHSlurmBackend(runtime_from_model(cluster, load_settings().credential_key)).discover_environment()
    except Exception as exc:
        result = {
            "schema_version": "polymer_agent.environment_discovery.v1",
            "ok": False,
            "error": type(exc).__name__,
            "admin_actions": [{"component": "connection", "action": "请检查 SSH、公钥、主机指纹和登录权限。"}],
        }
    limits = json.loads(cluster.resource_limits_json)
    provisioning = dict(limits.get("provisioning") or {})
    provisioning.update({
        "status": "needs_admin" if result.get("admin_actions") else "discovered",
        "discovery": result,
    })
    limits["provisioning"] = provisioning
    cluster.resource_limits_json = json.dumps(limits, ensure_ascii=False)
    if result.get("admin_actions"):
        cluster.status = "needs_admin"
    elif cluster.status != "ready":
        cluster.status = "authorized"
    db.commit()
    return cluster_public(cluster)


@router.post("/clusters/{cluster_id}/provision")
def cluster_provision(
    cluster_id: str,
    payload: ClusterProvisionRequest,
    db: Db,
    user: Annotated[User, Depends(mutating_user)],
):
    cluster = _owned(db, ClusterConnection, cluster_id, user)
    if cluster.status not in {"authorized", "ready", "failed", "needs_admin"}:
        raise HTTPException(status_code=409, detail="cluster_requires_admin_authorization")
    settings = load_settings()
    backend = SSHSlurmBackend(runtime_from_model(cluster, settings.credential_key))
    try:
        result = provision_cluster(backend, overwrite=payload.overwrite)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "remote_provision_failed", "error": type(exc).__name__},
        ) from exc
    installation = dict(result.get("installation") or {})
    science_install = dict(result.get("science_install") or {})
    if installation.get("runner_command"):
        cluster.runner_command = installation["runner_command"]
    limits = json.loads(cluster.resource_limits_json)
    science_running = science_install.get("status") in {"starting", "running"}
    limits["provisioning"] = {
        "status": "installing" if science_running else ("installed" if installation else "needs_admin"),
        "version": installation.get("version"),
        "manifest_sha256": installation.get("manifest_sha256"),
        "admin_actions": result.get("discovery", {}).get("admin_actions", []),
        "science_install": science_install,
    }
    cluster.resource_limits_json = json.dumps(limits, ensure_ascii=False)
    db.flush()
    if science_running:
        preflight = {"ok": False, "pending": "science_runtime_install"}
        cluster.status = "installing"
    else:
        try:
            preflight = SSHSlurmBackend(runtime_from_model(cluster, settings.credential_key)).preflight()
        except Exception as exc:
            preflight = {"ok": False, "error": type(exc).__name__}
        cluster.last_preflight_json = json.dumps(preflight, ensure_ascii=False)
        cluster.status = "ready" if preflight.get("ok") else "needs_admin"
    db.commit()
    public = cluster_public(cluster)
    public["provision_result"] = result
    return public


@router.post("/clusters/{cluster_id}/setup")
def cluster_setup(cluster_id: str, db: Db, user: Annotated[User, Depends(mutating_user)]):
    cluster = _owned(db, ClusterConnection, cluster_id, user)
    if cluster.status not in {"authorized", "ready", "failed", "needs_admin"}:
        raise HTTPException(status_code=409, detail="cluster_requires_admin_authorization")
    settings = load_settings()
    backend = SSHSlurmBackend(runtime_from_model(cluster, settings.credential_key))
    try:
        result = provision_cluster(backend, overwrite=False)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "remote_provision_failed", "error": type(exc).__name__},
        ) from exc
    installation = dict(result.get("installation") or {})
    science_install = dict(result.get("science_install") or {})
    if installation.get("runner_command"):
        cluster.runner_command = installation["runner_command"]
    discovery = dict(result.get("discovery") or {})
    limits = json.loads(cluster.resource_limits_json)
    science_running = science_install.get("status") in {"starting", "running"}
    provisioning = {
        "status": "installing" if science_running else ("installed" if installation else "needs_admin"),
        "version": installation.get("version"),
        "manifest_sha256": installation.get("manifest_sha256"),
        "admin_actions": discovery.get("admin_actions", []),
        "discovery": discovery,
        "auto_setup": True,
        "science_install": science_install,
    }
    limits["provisioning"] = provisioning
    cluster.resource_limits_json = json.dumps(limits, ensure_ascii=False)
    db.flush()
    if science_running:
        preflight = {"ok": False, "pending": "science_runtime_install"}
        cluster.status = "installing"
    else:
        try:
            preflight = SSHSlurmBackend(runtime_from_model(cluster, settings.credential_key)).preflight()
        except Exception as exc:
            preflight = {"ok": False, "error": type(exc).__name__}
        cluster.last_preflight_json = json.dumps(preflight, ensure_ascii=False)
        cluster.status = "ready" if preflight.get("ok") else "needs_admin"
    db.commit()
    public = cluster_public(cluster)
    public["setup_result"] = result
    public["preflight"] = preflight
    return public


@router.post("/clusters/{cluster_id}/setup-status")
def cluster_setup_status(cluster_id: str, db: Db, user: Annotated[User, Depends(mutating_user)]):
    cluster = _owned(db, ClusterConnection, cluster_id, user)
    if cluster.status not in {"installing", "authorized", "ready", "failed", "needs_admin"}:
        raise HTTPException(status_code=409, detail="cluster_setup_not_available")
    backend = SSHSlurmBackend(runtime_from_model(cluster, load_settings().credential_key))
    try:
        return _refresh_science_install(db, cluster, backend)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "science_runtime_status_failed", "error": type(exc).__name__},
        ) from exc


@router.post("/projects")
def projects_create(payload: ProjectCreate, db: Db, user: Annotated[User, Depends(mutating_user)]):
    return project_public(create_project(db, user, payload))


@router.get("/projects")
def projects_list(db: Db, user: Annotated[User, Depends(current_user)]):
    query = select(Project)
    if user.role != "admin":
        query = query.where(Project.owner_id == user.id)
    return [project_public(item) for item in db.scalars(query.order_by(Project.created_at.desc())).all()]


@router.get("/projects/{project_id}")
def project_get(project_id: str, db: Db, user: Annotated[User, Depends(current_user)]):
    return project_public(_owned(db, Project, project_id, user))


@router.post("/projects/{project_id}/validate")
def project_validate(project_id: str, db: Db, user: Annotated[User, Depends(mutating_user)]):
    return validate_project(db, _owned(db, Project, project_id, user))


@router.post("/projects/{project_id}/lock")
def project_lock(project_id: str, db: Db, user: Annotated[User, Depends(mutating_user)]):
    try:
        return project_public(lock_project(db, _owned(db, Project, project_id, user)))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/projects/{project_id}/runs")
def runs_create(project_id: str, payload: RunCreate, db: Db, user: Annotated[User, Depends(mutating_user)]):
    project = _owned(db, Project, project_id, user)
    cluster = _owned(db, ClusterConnection, payload.cluster_id, user)
    try:
        return run_public(create_run(db, user, project, cluster))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
def run_get(run_id: str, db: Db, user: Annotated[User, Depends(current_user)]):
    return run_public(_owned(db, WorkflowRun, run_id, user))


@router.get("/runs")
def runs_list(db: Db, user: Annotated[User, Depends(current_user)]):
    query = select(WorkflowRun)
    if user.role != "admin":
        query = query.where(WorkflowRun.owner_id == user.id)
    return [run_public(item) for item in db.scalars(query.order_by(WorkflowRun.created_at.desc())).all()]


@router.post("/runs/{run_id}/approve")
def run_approve(run_id: str, payload: RunApproval, db: Db, user: Annotated[User, Depends(mutating_user)]):
    settings = load_settings()
    if not settings.executor_enabled:
        raise HTTPException(status_code=503, detail="executor_disabled")
    run = _owned(db, WorkflowRun, run_id, user)
    try:
        project = db.get(Project, run.project_id)
        if project is None:
            raise ValueError("run project is missing")
        try:
            openai_key = model_api_key_for_user(db, user, "openai", settings)
            plan_event = build_upper_execution_plan(project_public(project), run_public(run), payload.model_dump(), api_key=openai_key)
        except ExecutionPlanError as exc:
            plan_event = {
                "schema_version": "polymer_agent.upper_execution_plan.v1",
                "status": "failed_or_skipped",
                "reason": str(exc),
            }
            if planner_required():
                raise HTTPException(status_code=503, detail={"code": "upper_planner_unavailable", "message": str(exc)}) from exc
        approve_run(db, run, user, payload)
        event(db, run, user.id, "upper_llm_execution_plan", plan_event)
        from .tasks import enqueue_run

        queue_job_id = enqueue_run(run.id)
        event(db, run, user.id, "worker_enqueued", {"queue_job_id": queue_job_id})
        db.commit()
        return run_public(run)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/cancel")
def run_cancel(run_id: str, db: Db, user: Annotated[User, Depends(mutating_user)]):
    run = _owned(db, WorkflowRun, run_id, user)
    if run.state in {"pass", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="run_is_terminal")
    if run.remote_job_id:
        cluster = db.get(ClusterConnection, run.cluster_id)
        try:
            SSHSlurmBackend(runtime_from_model(cluster, load_settings().credential_key)).cancel(run.remote_job_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"code": "cancel_failed", "error": type(exc).__name__}) from exc
    run.state = "cancelled"
    event(db, run, user.id, "run_cancelled", {})
    db.commit()
    return run_public(run)


@router.get("/runs/{run_id}/events")
def run_events(run_id: str, db: Db, user: Annotated[User, Depends(current_user)]):
    _owned(db, WorkflowRun, run_id, user)
    rows = db.scalars(select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.created_at)).all()

    def stream():
        for row in rows:
            payload = {"type": row.event_type, "payload": json.loads(row.payload_json), "created_at": row.created_at.isoformat()}
            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/reviews")
def reviews_list(db: Db, user: Annotated[User, Depends(current_user)]):
    query = select(ReviewRecord)
    if user.role not in {"admin", "reviewer"}:
        query = query.where(ReviewRecord.owner_id == user.id)
    return [
        {
            "id": item.id,
            "project_id": item.project_id,
            "run_id": item.run_id,
            "reason_code": item.reason_code,
            "status": item.status,
            "packet": json.loads(item.packet_json),
        }
        for item in db.scalars(query.order_by(ReviewRecord.created_at.desc())).all()
    ]


@router.post("/reviews/{review_id}/resolve")
def review_resolve(review_id: str, payload: ReviewResolution, db: Db, user: Annotated[User, Depends(mutating_user)]):
    review = db.get(ReviewRecord, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="not_found")
    if user.role not in {"admin", "reviewer"} and review.owner_id != user.id:
        raise HTTPException(status_code=403, detail="reviewer_required")
    review.status = "resolved"
    review.resolution_json = payload.model_dump_json()
    db.commit()
    return {"id": review.id, "status": review.status, "resolution": payload.model_dump()}


@router.post("/projects/{project_id}/releases")
def release_create(project_id: str, db: Db, user: Annotated[User, Depends(mutating_user)], run_id: str | None = None):
    project = _owned(db, Project, project_id, user)
    run = _owned(db, WorkflowRun, run_id, user) if run_id else None
    release = build_release(db, user, project, run)
    return {"id": release.id, "version": release.version, "sha256": release.payload_sha256}


@router.get("/projects/{project_id}/releases")
def releases_list(project_id: str, db: Db, user: Annotated[User, Depends(current_user)]):
    _owned(db, Project, project_id, user)
    rows = db.scalars(
        select(ReleaseSnapshot)
        .where(ReleaseSnapshot.project_id == project_id)
        .order_by(ReleaseSnapshot.version.desc(), ReleaseSnapshot.created_at.desc())
    ).all()
    return [
        {
            "id": item.id,
            "project_id": item.project_id,
            "run_id": item.run_id,
            "version": item.version,
            "sha256": item.payload_sha256,
            "created_at": item.created_at.isoformat(),
        }
        for item in rows
    ]


@router.get("/releases/{release_id}/download")
def release_download(release_id: str, db: Db, user: Annotated[User, Depends(current_user)]):
    release = _owned(db, ReleaseSnapshot, release_id, user)
    return Response(
        content=release.payload_json,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="pelcb-release-v{release.version}.json"'},
    )


@router.get("/releases/{release_id}/download-readable")
def release_download_readable(release_id: str, db: Db, user: Annotated[User, Depends(current_user)]):
    return release_download_result(release_id, db, user)


@router.get("/releases/{release_id}/download-result")
def release_download_result(release_id: str, db: Db, user: Annotated[User, Depends(current_user)]):
    release = _owned(db, ReleaseSnapshot, release_id, user)
    body = build_readable_release_markdown(json.loads(release.payload_json))
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="pelcb-release-v{release.version}-result.md"'},
    )
