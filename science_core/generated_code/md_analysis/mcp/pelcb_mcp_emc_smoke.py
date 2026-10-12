#!/usr/bin/env python3
"""MCP-mediated generic-validator smoke that materializes EMC build packages.

The default mode is local and non-submitting: it calls the registered FastMCP
validator tool, converts passing builder specs into EMC case directories, and
prepares Slurm-submittable EMC production-build packages.  With
``--remote-emc-smoke`` it submits the prepared packages as a small sidecar EMC
array and waits for completed setup/build artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MD_ANALYSIS_DIR = SCRIPT_DIR.parent
GENERATED_CODE_DIR = MD_ANALYSIS_DIR.parent
PROJECT_ROOT = GENERATED_CODE_DIR.parent
EMC_BUILDER_DIR = GENERATED_CODE_DIR / "emc_alpha_olefin_builder"
PREPARE_EMC = EMC_BUILDER_DIR / "prepare_emc_production_build.py"
SUBMIT_EMC_ARRAY = EMC_BUILDER_DIR / "submit_emc_production_build_array.py"
DEFAULT_OUTPUT_ROOT = MD_ANALYSIS_DIR / "output" / "pelcb_mcp_sidecar" / "emc_smoke"

for import_path in (SCRIPT_DIR, MD_ANALYSIS_DIR, EMC_BUILDER_DIR):
    text = str(import_path)
    if text not in sys.path:
        sys.path.insert(0, text)

import pelcb_mcp_server as server  # noqa: E402
from emc_builder_registry import write_emc_case  # noqa: E402


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_command(command: list[str], *, cwd: Path = PROJECT_ROOT, timeout: int = 3600) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
        "ok": proc.returncode == 0,
    }


async def call_tool(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    _content, data = await server.mcp.call_tool(name, payload)
    if not isinstance(data, dict):
        raise TypeError(f"MCP tool {name} returned non-dict payload: {type(data)}")
    return data


def default_positive_candidates() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "mcp_smoke_linear_c4_c100",
            "polymer_family": "poly_alpha_olefin",
            "architecture": "linear_alpha_olefin",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "chain_count": 12,
            "target_total_carbons_per_chain": 100,
            "sidechain_carbons": 2,
            "repeat_unit_smiles": "C=CCC",
        },
        {
            "candidate_id": "mcp_smoke_linear_c10_c100",
            "polymer_family": "poly_alpha_olefin",
            "architecture": "linear_alpha_olefin",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "chain_count": 12,
            "target_total_carbons_per_chain": 100,
            "sidechain_carbons": 8,
            "repeat_unit_smiles": "C=CCCCCCCC",
        },
        {
            "candidate_id": "mcp_pelcb_b4l4",
            "polymer_family": "PE",
            "architecture": "pe_lcb",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "chain_count": 12,
            "target_total_carbons_per_chain": 100,
            "branch_points": 4,
            "branch_carbons": 4,
            "branch_position_pattern": "comb-uniform-4",
        },
        {
            "candidate_id": "mcp_smoke_linear_pe_c100",
            "polymer_family": "PE",
            "architecture": "linear_homopolymer",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "chain_count": 12,
            "target_total_carbons_per_chain": 100,
            "repeat_unit_smiles": "*CC*",
            "repeat_unit_total_carbons": 2,
            "repeat_units_per_chain": 50,
        },
        {
            "candidate_id": "mcp_smoke_polyether_c100",
            "polymer_family": "polyether",
            "architecture": "linear_homopolymer",
            "builder_backend": "emc_repeat_unit_builder",
            "force_field": "opls-aa",
            "chain_count": 12,
            "target_total_carbons_per_chain": 100,
            "repeat_unit_smiles": "*CCO*",
            "repeat_unit_total_carbons": 2,
            "repeat_units_per_chain": 50,
        },
    ]


def default_negative_candidates() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "mcp_smoke_bad_backend",
            "polymer_family": "poly_alpha_olefin",
            "architecture": "linear_alpha_olefin",
            "builder_backend": "not_registered",
            "force_field": "opls-aa",
            "chain_count": 12,
            "target_total_carbons_per_chain": 100,
            "sidechain_carbons": 2,
            "repeat_unit_smiles": "C=CCC",
        },
        {
            "candidate_id": "mcp_smoke_bad_arch",
            "polymer_family": "poly_alpha_olefin",
            "architecture": "unknown_architecture",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "chain_count": 12,
            "target_total_carbons_per_chain": 100,
            "sidechain_carbons": 2,
            "repeat_unit_smiles": "C=CCC",
        },
        {
            "candidate_id": "mcp_smoke_bad_smiles",
            "polymer_family": "poly_alpha_olefin",
            "architecture": "linear_alpha_olefin",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "chain_count": 12,
            "target_total_carbons_per_chain": 100,
            "sidechain_carbons": 2,
            "repeat_unit_smiles": "not a smiles",
        },
    ]


def prepare_emc_package(case_dir: Path, run_label: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PREPARE_EMC),
        "--case-dir",
        str(case_dir),
        "--package-name",
        "emc_production_build",
        "--run-label",
        run_label,
        "--partition",
        "amd",
        "--cpus-per-task",
        "4",
        "--time-limit",
        "02:00:00",
    ]
    result = run_command(command, timeout=600)
    package_dir = case_dir / "emc_production_build"
    result["package_dir"] = str(package_dir)
    result["package_check"] = verify_package(case_dir, package_dir)
    return result


def verify_package(case_dir: Path, package_dir: Path) -> dict[str, Any]:
    case_esh = sorted(case_dir.glob("*.esh"))
    package_esh = sorted(package_dir.glob("*.esh")) if package_dir.exists() else []
    files = {
        "case_json": case_dir / "case.json",
        "package_manifest": package_dir / "emc_package_manifest.json",
        "submit_script": package_dir / "submit_emc_production_build.sbatch",
        "package_case_json": package_dir / "case.json",
    }
    checks = {name: path.exists() and path.stat().st_size > 0 for name, path in files.items()}
    checks["single_case_esh"] = len(case_esh) == 1 and case_esh[0].stat().st_size > 0
    checks["single_package_esh"] = len(package_esh) == 1 and package_esh[0].stat().st_size > 0
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "case_esh_count": len(case_esh),
        "package_esh_count": len(package_esh),
        "package_dir": str(package_dir),
    }


async def build_local_smoke(output_root: Path, stamp: str) -> dict[str, Any]:
    run_label = f"mcp_generic_emc_smoke_{stamp}"
    run_dir = output_root / run_label
    positive_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []

    for candidate in default_positive_candidates():
        validation = await call_tool("validate_polymer_design_candidate", {"candidate_json": json.dumps(candidate)})
        plan = await call_tool("plan_md_package", {"candidate_json": json.dumps(candidate)})
        row: dict[str, Any] = {
            "candidate": candidate,
            "validation": validation,
            "plan_md_package": plan,
        }
        if validation.get("validator_status") == "pass":
            builder_spec = validation["builder_spec"]
            case_dir = run_dir / "cases" / str(builder_spec["case_id"])
            case_artifacts = write_emc_case(case_dir, builder_spec, validation, plan)
            prepared = prepare_emc_package(case_dir, run_label)
            row.update(
                {
                    **case_artifacts,
                    "package_dir": prepared["package_dir"],
                    "prepare_result": prepared,
                    "package_check": prepared["package_check"],
                }
            )
        positive_rows.append(row)

    for candidate in default_negative_candidates():
        validation = await call_tool("validate_polymer_design_candidate", {"candidate_json": json.dumps(candidate)})
        negative_rows.append({"candidate": candidate, "validation": validation})

    local_ok_count = sum(1 for row in positive_rows if row.get("package_check", {}).get("ok") is True)
    negative_reject_count = sum(1 for row in negative_rows if row["validation"].get("validator_status") == "reject")
    return {
        "created_at": iso_now(),
        "stamp": stamp,
        "run_label": run_label,
        "run_dir": str(run_dir),
        "case_count": len(positive_rows),
        "local_ok_count": local_ok_count,
        "negative_reject_count": negative_reject_count,
        "cases": positive_rows,
        "negative_cases": negative_rows,
    }


def run_remote_emc(
    package_dirs: list[Path],
    *,
    output_root: Path,
    stamp: str,
    remote_host: str,
    remote_root: str,
    partition: str,
    array_concurrency: int,
    poll_seconds: int,
    max_wait_seconds: int,
) -> dict[str, Any]:
    log_root = output_root / f"remote_submit_{stamp}"
    array_name = f"mcp_generic_emc_smoke_{stamp}"
    command = [
        sys.executable,
        str(SUBMIT_EMC_ARRAY),
        "--remote-host",
        remote_host,
        "--remote-root",
        remote_root,
        "--date-tag",
        stamp,
        "--array-name",
        array_name,
        "--partition",
        partition,
        "--array-concurrency",
        str(array_concurrency),
        "--poll-seconds",
        str(poll_seconds),
        "--max-wait-seconds",
        str(max_wait_seconds),
        "--log-root",
        str(log_root),
        "--overwrite-canonical",
    ]
    for package_dir in package_dirs:
        command.extend(["--package-dir", str(package_dir)])
    result = run_command(command, timeout=max(600, max_wait_seconds + 900))
    summary_path = log_root / "array_submit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "status": "ok" if result["ok"] else "failed",
        "command_result": result,
        "summary_path": str(summary_path),
        "summary": summary,
    }


def run_smoke(
    *,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    remote_emc_smoke: bool = False,
    stamp: str | None = None,
    remote_host: str = "${REMOTE_USER}@private-3f2a50db.invalid",
    remote_root: str = "${PELCB100_REMOTE_ROOT}/lz_project_runs/emc_alpha_olefin_builder/pelcb_mcp_generic_validator_smoke",
    partition: str = "amd",
    array_concurrency: int = 3,
    poll_seconds: int = 60,
    max_wait_seconds: int = 14400,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = stamp or default_stamp()
    local = asyncio.run(build_local_smoke(output_root, stamp))
    package_dirs = [Path(row["package_dir"]) for row in local["cases"] if row.get("package_check", {}).get("ok") is True]
    remote = None
    if remote_emc_smoke:
        remote = run_remote_emc(
            package_dirs,
            output_root=output_root,
            stamp=stamp,
            remote_host=remote_host,
            remote_root=remote_root,
            partition=partition,
            array_concurrency=array_concurrency,
            poll_seconds=poll_seconds,
            max_wait_seconds=max_wait_seconds,
        )
    remote_ok_count = None
    if remote is not None:
        remote_summary = remote.get("summary") or {}
        remote_ok_count = remote_summary.get("ok_cases")
    summary = {
        "status": "ok"
        if local["local_ok_count"] == local["case_count"]
        and local["negative_reject_count"] == len(local["negative_cases"])
        and (remote is None or remote.get("status") == "ok")
        else "failed",
        "description": "MCP validator -> builder_spec -> EMC package smoke; optional remote EMC setup/build completion.",
        "remote_emc_smoke": remote_emc_smoke,
        **local,
        "remote_ok_count": remote_ok_count,
        "remote": remote,
    }
    write_json(output_root / "emc_smoke_latest.json", summary)
    if remote is not None:
        write_json(output_root / "generic_validator_remote_emc_latest.json", summary)
    write_json(Path(local["run_dir"]) / "emc_smoke_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--remote-emc-smoke", action="store_true")
    parser.add_argument("--remote-host", default="${REMOTE_USER}@private-3f2a50db.invalid")
    parser.add_argument(
        "--remote-root",
        default="${PELCB100_REMOTE_ROOT}/lz_project_runs/emc_alpha_olefin_builder/pelcb_mcp_generic_validator_smoke",
    )
    parser.add_argument("--partition", default="amd")
    parser.add_argument("--array-concurrency", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=14400)
    args = parser.parse_args()

    summary = run_smoke(
        output_root=args.output_root,
        remote_emc_smoke=args.remote_emc_smoke,
        stamp=args.stamp,
        remote_host=args.remote_host,
        remote_root=args.remote_root,
        partition=args.partition,
        array_concurrency=args.array_concurrency,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
