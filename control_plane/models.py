from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid4().hex


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(20), default="researcher")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LoginSession(Base):
    __tablename__ = "login_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    csrf_token: Mapped[str] = mapped_column(String(64))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelCredential(Base):
    __tablename__ = "model_credentials"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    encrypted_openai_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_deepseek_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_dashscope_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_zhipuai_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ClusterConnection(Base):
    __tablename__ = "cluster_connections"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_cluster_owner_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    host: Mapped[str] = mapped_column(String(255))
    port: Mapped[int] = mapped_column(Integer, default=22)
    remote_username: Mapped[str] = mapped_column(String(100))
    host_key_sha256: Mapped[str] = mapped_column(String(128))
    encrypted_private_key: Mapped[str] = mapped_column(Text)
    public_key: Mapped[str] = mapped_column(Text)
    partition: Mapped[str] = mapped_column(String(100))
    account: Mapped[str | None] = mapped_column(String(100), nullable=True)
    remote_root: Mapped[str] = mapped_column(String(500))
    gromacs_command: Mapped[str] = mapped_column(String(500))
    emc_command: Mapped[str] = mapped_column(String(500))
    python_command: Mapped[str] = mapped_column(String(200), default="python3")
    runner_command: Mapped[str] = mapped_column(String(500), default="polymer-agent-remote")
    module_init_json: Mapped[str] = mapped_column(Text, default="[]")
    resource_limits_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(40), default="unverified")
    last_preflight_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(40), default="draft")
    spec_json: Mapped[str] = mapped_column(Text)
    manifest_json: Mapped[str] = mapped_column(Text, default="[]")
    manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    domain_status: Mapped[str] = mapped_column(String(40), default="unvalidated")
    validation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    cluster_id: Mapped[str] = mapped_column(ForeignKey("cluster_connections.id"), index=True)
    state: Mapped[str] = mapped_column(String(40), default="planned")
    approval_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    remote_job_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    remote_workdir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    actor_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    event_type: Mapped[str] = mapped_column(String(80))
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RunStageJob(Base):
    __tablename__ = "run_stage_jobs"
    __table_args__ = (UniqueConstraint("run_id", "stage", "attempt", name="uq_run_stage_attempt"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(40), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    state: Mapped[str] = mapped_column(String(40), default="planned")
    slurm_job_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    remote_workdir: Mapped[str] = mapped_column(String(500))
    case_count: Mapped[int] = mapped_column(Integer)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class RunArtifact(Base):
    __tablename__ = "run_artifacts"
    __table_args__ = (UniqueConstraint("run_id", "name", name="uq_run_artifact_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    payload_json: Mapped[str] = mapped_column(Text)
    payload_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewRecord(Base):
    __tablename__ = "review_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    reason_code: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(40), default="open")
    packet_json: Mapped[str] = mapped_column(Text, default="{}")
    resolution_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReleaseSnapshot(Base):
    __tablename__ = "release_snapshots"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    payload_json: Mapped[str] = mapped_column(Text)
    payload_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
