#!/usr/bin/env python3
"""Pinned cluster-side contract for PE-LCB production execution.

The runner never accepts arbitrary shell text.  A site administrator configures
one absolute adapter executable through ``POLYMER_AGENT_SITE_PIPELINE``.  That
adapter receives only versioned file paths and bounded repair budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def command_path(env_name: str, default: str) -> str:
    value = os.getenv(env_name, default)
    if not re.fullmatch(r"[A-Za-z0-9_./+-]+", value):
        raise ValueError(f"{env_name} must contain one binary name or absolute path")
    return value


def probe(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "output": ((result.stdout or "") + (result.stderr or ""))[:500],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


def self_test() -> int:
    adapter = os.getenv("POLYMER_AGENT_SITE_PIPELINE", "")
    adapter_ok = bool(adapter and Path(adapter).is_absolute() and Path(adapter).is_file() and os.access(adapter, os.X_OK))
    checks = {
        "site_pipeline": {
            "ok": adapter_ok,
            "configured": bool(adapter),
        },
    }
    if adapter_ok:
        checks["site_pipeline_contract"] = probe([adapter, "self-test", "--json"])
    payload = {"schema_version": "polymer_agent.remote_runner.v1", "ok": all(v["ok"] for v in checks.values()), "checks": checks}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def validated_path(raw: str, *, must_exist: bool = True) -> Path:
    path = Path(raw).resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_pipeline(args: argparse.Namespace) -> int:
    project = validated_path(args.project_spec)
    manifest = validated_path(args.manifest)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(project.read_text(encoding="utf-8"))
    cases = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain at least one case")
    if len(cases) > 1000:
        raise ValueError("manifest exceeds runner case limit")
    if spec.get("architecture") != "pe_lcb" or str(spec.get("polymer_family", "")).upper() != "PE":
        raise ValueError("remote runner only accepts PE-LCB projects")
    adapter = command_path("POLYMER_AGENT_SITE_PIPELINE", "")
    if not adapter or not Path(adapter).is_absolute() or not Path(adapter).is_file():
        raise RuntimeError("POLYMER_AGENT_SITE_PIPELINE is not an installed absolute executable")
    command = [
        adapter,
        "run",
        "--project-spec",
        str(project),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--max-repair-attempts",
        str(args.max_repair_attempts),
        "--max-additional-md-ns",
        str(args.max_additional_md_ns),
    ]
    result = subprocess.run(command, check=False)
    return int(result.returncode)


def run_stage(args: argparse.Namespace) -> int:
    project = validated_path(args.project_spec)
    manifest = validated_path(args.manifest)
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    adapter = command_path("POLYMER_AGENT_SITE_PIPELINE", "")
    if not adapter or not Path(adapter).is_absolute() or not Path(adapter).is_file():
        raise RuntimeError("POLYMER_AGENT_SITE_PIPELINE is not an installed absolute executable")
    command = [
        adapter,
        "stage",
        "--stage",
        args.stage,
        "--project-spec",
        str(project),
        "--manifest",
        str(manifest),
        "--work-root",
        str(work_root),
    ]
    if args.case_index is not None:
        command.extend(["--case-index", str(args.case_index)])
    return int(subprocess.run(command, check=False).returncode)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    check = sub.add_parser("self-test")
    check.add_argument("--json", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--project-spec", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--max-repair-attempts", type=int, choices=range(0, 4), default=1)
    run.add_argument("--max-additional-md-ns", type=int, choices=range(0, 401), default=400)
    stage = sub.add_parser("stage")
    stage.add_argument("--stage", choices=["build", "md", "postprocess", "repair"], required=True)
    stage.add_argument("--project-spec", required=True)
    stage.add_argument("--manifest", required=True)
    stage.add_argument("--work-root", required=True)
    stage.add_argument("--case-index", type=int)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "self-test":
        return self_test()
    if args.command == "stage":
        return run_stage(args)
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
