"""Offline dependency and configuration checks, without importing the platform.

Run ``python -m polymer_agent_demo.scripts.doctor --stage demo``. Configuration
files are read only when explicitly named with ``--env-file``. Credentials are
reported as booleans; this module makes no network or model calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..control_plane.config import model_configuration, model_stage_readiness


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("demo", "design", "local-validation", "audit", "md")
KEY_FIELDS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "zhipuai": "ZHIPUAI_API_KEY",
}
DEPENDENCIES = {
    "demo": ("fastapi", "uvicorn", "pydantic"),
    "design": ("pydantic", "openai", "rdkit"),
    "local-validation": ("pydantic", "rdkit"),
    "audit": ("numpy", "pandas", "yaml", "openai"),
    "md": ("sqlalchemy", "psycopg", "redis", "rq", "paramiko", "cryptography"),
}
EXTRAS = {
    "demo": ".",
    "design": ".[science]",
    "local-validation": ".[science]",
    "audit": ".[science]",
    "md": ".[platform,science]",
}


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _check(name: str, present: bool, requirement: str) -> dict[str, str]:
    return {"name": name, "status": "ready" if present else "missing", "requirement": requirement}


def _snapshot_available(root: Path) -> bool:
    try:
        snapshot = json.loads((root / "data" / "demo_snapshot.json").read_text(encoding="utf-8"))
        return bool(snapshot.get("cases") and snapshot.get("overview") and snapshot.get("data_version"))
    except (OSError, ValueError, AttributeError):
        return False


def _bundle_available(root: Path) -> bool:
    try:
        with zipfile.ZipFile(root / "deploy" / "science_core_bundle.zip") as archive:
            return any(name.endswith("generated_code/md_analysis/polymer_qc_rules.py") for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def build_report(
    stage: str = "demo",
    *,
    environment: Mapping[str, str] | None = None,
    package_root: Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    if stage not in {*STAGES, "all"}:
        raise ValueError("unsupported doctor stage")
    env = os.environ if environment is None else environment
    configured = {provider: bool(str(env.get(field) or "").strip()) for provider, field in KEY_FIELDS.items()}
    selected = STAGES if stage == "all" else (stage,)
    stage_reports: dict[str, Any] = {}
    for name in selected:
        checks = [_check("python", sys.version_info >= (3, 11), "Python 3.11+")]
        checks.extend(_check(module, _available(module), f"Python dependency: {module}") for module in DEPENDENCIES[name])
        if name == "demo":
            checks.append(_check("frozen_snapshot", _snapshot_available(package_root), "Bundled data/demo_snapshot.json"))
            for asset in ("auto.html", "index.html", "auto.js", "auto.css", "app.js", "styles.css"):
                checks.append(_check(asset, (package_root / "static" / asset).is_file(), f"Bundled static/{asset}"))
        if name == "design":
            checks.append(_check("OPENAI_API_KEY", configured["openai"], "OpenAI credential for live design"))
            checks.append(_check("POLYMER_AGENT_LIVE_DESIGN", _enabled(env.get("POLYMER_AGENT_LIVE_DESIGN")), "Enable live design explicitly"))
        if name == "audit":
            checks.append(_check("science_core_bundle", _bundle_available(package_root), "Bundled deploy/science_core_bundle.zip"))
            for provider in ("deepseek", "dashscope", "zhipuai"):
                field = KEY_FIELDS[provider]
                checks.append(_check(field, configured[provider], f"{provider} credential for the three-model audit"))
        if name == "md":
            checks.append(_check("science_core_bundle", _bundle_available(package_root), "Bundled deploy/science_core_bundle.zip"))
            checks.append(_check("POLYMER_AGENT_EXECUTOR_ENABLED", _enabled(env.get("POLYMER_AGENT_EXECUTOR_ENABLED")), "Enable the registered executor explicitly"))
            for field in ("POLYMER_AGENT_SECRET_KEY", "POLYMER_AGENT_CREDENTIAL_KEY"):
                secret = str(env.get(field) or "").strip()
                checks.append(_check(field, bool(secret and secret != "development-only-change-me"), "Explicit platform encryption/session secret"))
            checks.append(_check("POLYMER_AGENT_DATABASE_URL", str(env.get("POLYMER_AGENT_DATABASE_URL") or "").startswith(("postgresql://", "postgresql+psycopg://")), "PostgreSQL configuration for execution"))
            checks.append(_check("POLYMER_AGENT_REDIS_URL", bool(env.get("POLYMER_AGENT_REDIS_URL")), "Redis configuration for the registered worker"))
            checks.append(_check("POLYMER_AGENT_COOKIE_SECURE", _enabled(env.get("POLYMER_AGENT_COOKIE_SECURE")), "Secure cookies behind HTTPS"))
            checks.append({
                "name": "remote_preflight",
                "status": "pending",
                "requirement": "Use the platform cluster preflight to check SSH, Slurm, EMC, GROMACS 2024.4, CPU/GPU availability and account limits",
            })
        stage_reports[name] = {
            "ready": all(item["status"] == "ready" for item in checks),
            "missing": [item["name"] for item in checks if item["status"] == "missing"],
            "pending": [item["name"] for item in checks if item["status"] == "pending"],
            "checks": checks,
            "install_command": f'python -m pip install -e "{EXTRAS[name]}"',
        }
    return {
        "schema_version": "polymer_agent.doctor.v1",
        "stage": stage,
        "scope": "offline_local_dependencies_and_environment",
        "network_checked": False,
        "credentials": configured,
        "credential_scope": "process environment and explicitly named env file; saved user keys are checked by the platform",
        "models": model_configuration(env),
        "model_stage_readiness": model_stage_readiness(configured),
        "private_project_root_configured": bool(env.get("POLYMER_AGENT_ROOT")),
        "stages": stage_reports,
        "ready": all(item["ready"] for item in stage_reports.values()),
    }


def _explicit_environment(path: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    if path is None:
        return env
    # The same simple KEY=value format as the platform loader, without changing
    # os.environ, expanding variables or executing shell expressions.
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip():
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="demo")
    parser.add_argument("--json", action="store_true", help="Print a structured report")
    parser.add_argument("--env-file", type=Path, help="Explicitly inspect this env file; process environment takes precedence")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.stage, environment=_explicit_environment(args.env_file))
    except (OSError, UnicodeError):
        print(json.dumps({"error": "configuration_file_unreadable", "ready": False}))
        return 2
    report["configuration_source"] = "environment_and_explicit_env_file" if args.env_file else "environment"
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("Polymer-Agent doctor: offline checks")
        for name, result in report["stages"].items():
            print(f"{name}: {'ready' if result['ready'] else 'needs configuration'}")
            for check in result["checks"]:
                if check["status"] != "ready":
                    print(f"  {check['status']}: {check['name']}: {check['requirement']}")
            if result["missing"]:
                print(f"  Dependency installation: {result['install_command']}")
        print("Credentials are checked as configured/empty; remote readiness requires the registered preflight.")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
