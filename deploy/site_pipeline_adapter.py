#!/usr/bin/env python3
"""Bounded adapter from the platform runner to a reviewed PE-LCB runtime.

The adapter does not install or emulate EMC/GROMACS. It executes one fixed
``pelcb-pipeline-core`` command supplied by a selected module or container.
Only the versioned project/manifest paths and bounded repair arguments accepted
by the platform contract can cross this boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


COMMAND = re.compile(r"^[A-Za-z0-9_./+-]+$")


def load_config() -> dict[str, Any]:
    raw = os.getenv("POLYMER_AGENT_SITE_CONFIG", "")
    path = Path(raw).resolve()
    if not raw or not path.is_file():
        raise RuntimeError("POLYMER_AGENT_SITE_CONFIG does not reference an installed config")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "polymer_agent.site_config.v1":
        raise ValueError("unsupported site config schema")
    return payload


def checked_command(value: str) -> str:
    if not COMMAND.fullmatch(value):
        raise ValueError("configured commands must be one binary name or absolute path")
    return value


def core_command(config: dict[str, Any]) -> list[str]:
    environment = dict(config.get("environment") or {})
    pipeline = checked_command(str(environment.get("pipeline_command") or "pelcb-pipeline-core"))
    mode = str(environment.get("mode") or "module")
    if mode == "module":
        return [pipeline]
    if mode != "container":
        raise ValueError("unsupported environment mode")
    runtime = checked_command(str(environment.get("container_runtime") or ""))
    if runtime not in {"apptainer", "singularity"}:
        raise ValueError("container runtime must be apptainer or singularity")
    image = Path(str(environment.get("container_image") or ""))
    if not image.is_absolute() or ".." in image.parts:
        raise ValueError("container image must be an absolute normalized path")
    return [runtime, "exec", str(image), pipeline]


def self_test(config: dict[str, Any]) -> int:
    command = [*core_command(config), "self-test", "--json"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
    payload: dict[str, Any] = {
        "schema_version": "polymer_agent.site_adapter.v1",
        "ok": result.returncode == 0,
        "environment_mode": dict(config.get("environment") or {}).get("mode", "module"),
        "core_returncode": result.returncode,
    }
    try:
        core = json.loads(result.stdout)
        payload["core"] = core
        payload["ok"] = payload["ok"] and bool(core.get("ok"))
    except (json.JSONDecodeError, AttributeError):
        payload["core_output_sha256"] = __import__("hashlib").sha256(
            ((result.stdout or "") + (result.stderr or "")).encode()
        ).hexdigest()
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def existing_file(raw: str) -> str:
    path = Path(raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def run(config: dict[str, Any], args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        *core_command(config),
        "run",
        "--project-spec",
        existing_file(args.project_spec),
        "--manifest",
        existing_file(args.manifest),
        "--output",
        str(output),
        "--max-repair-attempts",
        str(args.max_repair_attempts),
        "--max-additional-md-ns",
        str(args.max_additional_md_ns),
    ]
    return int(subprocess.run(command, check=False).returncode)


def stage(config: dict[str, Any], args: argparse.Namespace) -> int:
    command = [
        *core_command(config),
        "stage",
        "--stage",
        args.stage,
        "--project-spec",
        existing_file(args.project_spec),
        "--manifest",
        existing_file(args.manifest),
        "--work-root",
        str(Path(args.work_root).resolve()),
    ]
    if args.case_index is not None:
        command.extend(["--case-index", str(args.case_index)])
    return int(subprocess.run(command, check=False).returncode)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    check = sub.add_parser("self-test")
    check.add_argument("--json", action="store_true")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--project-spec", required=True)
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--max-repair-attempts", type=int, choices=range(0, 4), default=1)
    run_parser.add_argument("--max-additional-md-ns", type=int, choices=range(0, 401), default=400)
    stage_parser = sub.add_parser("stage")
    stage_parser.add_argument("--stage", choices=["build", "md", "postprocess", "repair"], required=True)
    stage_parser.add_argument("--project-spec", required=True)
    stage_parser.add_argument("--manifest", required=True)
    stage_parser.add_argument("--work-root", required=True)
    stage_parser.add_argument("--case-index", type=int)
    return root


def main() -> int:
    args = parser().parse_args()
    config = load_config()
    if args.command == "self-test":
        return self_test(config)
    if args.command == "stage":
        return stage(config, args)
    return run(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
