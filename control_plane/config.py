from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def model_configuration(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Configured model IDs for the existing planner and three audit providers."""
    env = os.environ if environment is None else environment
    design_model = env.get("POLYMER_AGENT_DESIGN_MODEL") or "gpt-5.5"
    return {
        "design": design_model,
        "upper_planner": env.get("POLYMER_AGENT_UPPER_PLANNER_MODEL") or design_model,
        "deepseek": env.get("POLYMER_AGENT_AUDIT_DEEPSEEK_MODEL") or "deepseek-v4-pro",
        "dashscope": env.get("POLYMER_AGENT_AUDIT_QWEN_MODEL") or "qwen3.7-max",
        "zhipuai": env.get("POLYMER_AGENT_AUDIT_GLM_MODEL") or "glm-4.6",
    }


def model_stage_readiness(configured: Mapping[str, bool]) -> dict[str, bool]:
    """Check credentials for the stage that uses them; values are never returned."""
    planner = bool(configured.get("openai"))
    audit = all(configured.get(provider) for provider in ("deepseek", "dashscope", "zhipuai"))
    return {
        "ready_for_replay": True,
        "ready_for_local_validation": True,
        "ready_for_design": planner,
        "ready_for_upper_planner": planner,
        "ready_for_audit_panel": audit,
        "ready_for_closed_loop": planner and audit,
    }


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    secret_key: str
    credential_key: str
    cookie_secure: bool
    session_hours: int
    project_root: Path | None
    bootstrap_admin: str | None
    bootstrap_password: str | None
    bootstrap_reset: bool
    executor_enabled: bool


def load_settings() -> Settings:
    runtime = PACKAGE_ROOT / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    root_value = os.getenv("POLYMER_AGENT_ROOT")
    settings = Settings(
        database_url=os.getenv(
            "POLYMER_AGENT_DATABASE_URL",
            f"sqlite:///{(runtime / 'control_plane.db').as_posix()}",
        ),
        redis_url=os.getenv("POLYMER_AGENT_REDIS_URL", "redis://redis:6379/0"),
        secret_key=os.getenv("POLYMER_AGENT_SECRET_KEY", "development-only-change-me"),
        credential_key=os.getenv(
            "POLYMER_AGENT_CREDENTIAL_KEY",
            os.getenv("POLYMER_AGENT_SECRET_KEY", "development-only-change-me"),
        ),
        cookie_secure=_truthy(os.getenv("POLYMER_AGENT_COOKIE_SECURE")),
        session_hours=max(1, int(os.getenv("POLYMER_AGENT_SESSION_HOURS", "12"))),
        project_root=Path(root_value).resolve() if root_value else None,
        bootstrap_admin=os.getenv("POLYMER_AGENT_BOOTSTRAP_ADMIN"),
        bootstrap_password=os.getenv("POLYMER_AGENT_BOOTSTRAP_PASSWORD"),
        bootstrap_reset=_truthy(os.getenv("POLYMER_AGENT_BOOTSTRAP_RESET")),
        executor_enabled=_truthy(os.getenv("POLYMER_AGENT_EXECUTOR_ENABLED")),
    )
    if settings.executor_enabled:
        if settings.secret_key == "development-only-change-me" or settings.credential_key == "development-only-change-me":
            raise RuntimeError("production executor requires explicit secret and credential keys")
        if settings.database_url.startswith("sqlite"):
            raise RuntimeError("production executor requires PostgreSQL, not SQLite")
        if not settings.cookie_secure:
            raise RuntimeError("production executor requires secure cookies behind HTTPS")
    return settings
