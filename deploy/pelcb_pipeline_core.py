#!/usr/bin/env python3
"""Versioned PE-LCB pipeline entry point.

The bundled acceptance profile runs a real, short GROMACS calculation and
terminates through the same rule/audit/release contract as the control plane.
It is deliberately non-publication-grade and therefore ends in ``review``.

Production execution is handled by the staged worker in ``production_case.py``:
build, MD, postprocess and bounded repair are submitted as separate Slurm array
stages. The acceptance profile is never used as a silent production fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "polymer_agent.pipeline_core.v1"
CASE_ID = "lcb01_star3_b1_l05_c101"
COMMAND = re.compile(r"^[A-Za-z0-9_./+\\:-]+$")


def project_root(raw: str | None = None) -> Path:
    if raw:
        root = Path(raw).resolve()
    elif os.getenv("POLYMER_AGENT_ROOT"):
        root = Path(os.environ["POLYMER_AGENT_ROOT"]).resolve()
    else:
        root = Path(__file__).resolve().parents[3]
    return root


def assets(root: Path) -> dict[str, Path]:
    bridge = (
        root
        / "generated_code"
        / "emc_alpha_olefin_builder"
        / "output"
        / "lcb100_bridge_fixed_20260424"
        / CASE_ID
    )
    repository_assets = {
        "runner": root / "generated_code" / "lcb_md_pipeline" / "run_real_smoke.py",
        "bridge": bridge,
        "gro": bridge / f"{CASE_ID}.gro",
        "top": bridge / "topol.top",
        "itp": bridge / f"{CASE_ID}_polymer.itp",
        "diagnosis": root / "generated_code" / "md_analysis" / "pelcb_audit_diagnosis.py",
    }
    if repository_assets["runner"].is_file():
        return repository_assets
    bundled = Path(__file__).resolve().parent
    return {
        "runner": bundled / "run_real_smoke.py",
        "bridge": bundled / "fixture",
        "gro": bundled / "fixture" / f"{CASE_ID}.gro",
        "top": bundled / "fixture" / "topol.top",
        "itp": bundled / "fixture" / f"{CASE_ID}_polymer.itp",
        "diagnosis": bundled / "pelcb_audit_diagnosis.py",
    }


def resolve_gmx(raw: str | None = None) -> str | None:
    candidates = [
        raw,
        os.getenv("POLYMER_AGENT_GROMACS"),
        os.getenv("POLYMER_AGENT_LOCAL_GMX"),
        shutil.which("gmx"),
        shutil.which("gmx_mpi"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return str(candidate)
    return None


def probe(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error_summary": str(exc)[-500:]}
    payload: dict[str, Any] | None = None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload = None
    return {
        "ok": completed.returncode == 0 and bool(payload.get("ok") if isinstance(payload, dict) else completed.returncode == 0),
        "returncode": completed.returncode,
        "payload": payload,
        "stderr_tail": completed.stderr[-1000:],
    }


def self_test(root: Path, gmx: str | None) -> dict[str, Any]:
    paths = assets(root)
    checks = {name: path.is_file() if name != "bridge" else path.is_dir() for name, path in paths.items()}
    checks["gromacs"] = bool(gmx)
    science_root_raw = os.getenv("POLYMER_AGENT_SCIENCE_ROOT", "")
    science_root = Path(science_root_raw) if science_root_raw else None
    science_required = [
        "generated_code/md_analysis/polymer_qc_rules.py",
        "generated_code/md_analysis/rule_profile_registry.py",
        "generated_code/md_analysis/pelcb_warn_audit_llm.py",
        "generated_code/md_analysis/pelcb_audit_diagnosis.py",
        "generated_code/md_analysis/pelcb_clean_overlay_regate.py",
        "generated_code/md_analysis/polymer_design_validator.py",
        "generated_code/md_analysis/emc_builder_registry.py",
        "generated_code/emc_alpha_olefin_builder/emc_to_gromacs.py",
        "generated_code/lcb_md_pipeline/mdp/nvt_prod.mdp",
    ]
    science_checks = {
        name: bool(science_root and (science_root / name).is_file())
        for name in science_required
    }
    science_core_installed = bool(science_checks) and all(science_checks.values())
    production_runtime = os.getenv("POLYMER_AGENT_PRODUCTION_PIPELINE")
    bundled_worker = Path(__file__).resolve().parent / "production_case.py"
    emc = os.getenv("POLYMER_AGENT_EMC") or shutil.which("emc")
    worker_probe = probe([sys.executable, str(bundled_worker), "--self-test", "--gmx", str(gmx or "gmx"), "--emc", str(emc or "emc")]) if bundled_worker.is_file() else {"ok": False}
    production_ready = bool(
        (production_runtime and Path(production_runtime).is_file())
        or (bundled_worker.is_file() and science_core_installed and gmx and emc and worker_probe.get("ok"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": production_ready,
        "acceptance_ready": all(checks.values()),
        "production_ready": production_ready,
        "science_core_installed": science_core_installed,
        "science_checks": science_checks,
        "production_worker": worker_probe,
        "checks": checks,
        "capabilities": ["minute_closed_loop_acceptance"] + (["production"] if production_ready else []),
    }


def run_smoke(root: Path, output: Path, gmx: str, steep_nsteps: int, npt_steps: int) -> dict[str, Any]:
    paths = assets(root)
    work = output / "md"
    command = [
        sys.executable,
        str(paths["runner"]),
        "--mode",
        "local-smoke",
        "--case",
        CASE_ID,
        "--bridge-dir",
        str(paths["bridge"]),
        "--gmx-bin",
        gmx,
        "--work-dir",
        str(work),
        "--steep-nsteps",
        str(steep_nsteps),
        "--npt-steps",
        str(npt_steps),
    ]
    environment = os.environ.copy()
    gmx_path = Path(gmx)
    if gmx_path.is_file() and not environment.get("GMXLIB"):
        candidate = gmx_path.parent.parent / "share" / "gromacs" / "top"
        if candidate.is_dir():
            environment["GMXLIB"] = str(candidate)
    result = subprocess.run(command, text=True, capture_output=True, timeout=900, check=False, env=environment)
    if result.returncode != 0:
        raise RuntimeError("local GROMACS smoke failed: " + ((result.stdout or "") + (result.stderr or ""))[-1200:])
    result_path = work / "local_smoke_result.json"
    if not result_path.is_file():
        raise RuntimeError("local smoke did not produce local_smoke_result.json")
    return json.loads(result_path.read_text(encoding="utf-8"))


def deterministic_audit(root: Path, smoke: dict[str, Any]) -> dict[str, Any]:
    analysis_dir = root / "generated_code" / "md_analysis"
    if not analysis_dir.is_dir():
        analysis_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(analysis_dir))
    from pelcb_audit_diagnosis import diagnose_case  # noqa: PLC0415

    return diagnose_case(
        {
            "case_id": smoke["case"],
            "case_status": "completed_engine_qualified_warn",
            "density_tier_si_o": smoke["verdict"],
            "rho_final_gcc_si_o": smoke["density_g_per_cc"],
            "status_density": "OK",
            "ml_usable": False,
        }
    )


def build_review_release(output: Path, smoke: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    energy = output / "md" / "rlx.edr"
    config = {
        "profile": "minute_closed_loop_acceptance",
        "steep_nsteps": smoke["smoke_scale"]["steep_nsteps"],
        "npt_steps": smoke["smoke_scale"]["npt_steps"],
        "terminal_policy": "smoke_evidence_is_never_publication_grade",
    }
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return {
        "schema_version": "polymer_agent.remote_release.v1",
        "publication_state": "review",
        "config_sha256": config_sha,
        "project": {
            "name": "PE-LCB minute closed-loop acceptance",
            "status": "locked",
            "domain_status": "validated_domain",
            "manifest_sha256": hashlib.sha256(CASE_ID.encode()).hexdigest(),
        },
        "run": {"state": "review", "mode": "minute_closed_loop_acceptance"},
        "provenance": [
            {"stage": "real_md", "artifact_sha256": hashlib.sha256(energy.read_bytes()).hexdigest()},
            {"stage": "rule_layer", "verdict": smoke["verdict"], "warn_source": smoke.get("warn_source")},
            {
                "stage": "audit",
                "diagnosis": diagnosis.get("primary_audit_diagnosis"),
                "route": diagnosis.get("recommended_reentry_module"),
            },
            {"stage": "release_validation", "terminal_state": "review"},
        ],
        "cases": [
            {
                "case_id": CASE_ID,
                "quality_status": "review",
                "density_g_cm3": smoke["density_g_per_cc"],
                "property_masks": {"density": False, "static": False, "diffusion_strict": False},
                "audit_votes": [],
                "repair_history": [],
                "msd_warn": False,
                "review_reason": "Minute-scale MD is not publication-grade; bounded closure terminates in review.",
                "audit_diagnosis": diagnosis.get("primary_audit_diagnosis"),
                "recommended_reentry_module": diagnosis.get("recommended_reentry_module"),
                "recommended_action": diagnosis.get("recommended_action"),
                "evidence_sufficiency": diagnosis.get("evidence_sufficiency", {}),
            }
        ],
        "reviews": [
            {"reason_code": "insufficient_smoke_evidence", "status": "open", "case_id": CASE_ID}
        ],
    }


def validate_release(root: Path, payload: dict[str, Any]) -> str:
    sys.path.insert(0, str(root))
    try:
        from polymer_agent_demo.control_plane.science import (  # noqa: PLC0415
            build_readable_release_markdown,
            validate_remote_release,
        )

        validate_remote_release(payload, project_domain_status="validated_domain")
        return build_readable_release_markdown(payload)
    except ModuleNotFoundError:
        if payload.get("schema_version") != "polymer_agent.remote_release.v1":
            raise ValueError("unexpected remote release schema")
        if payload.get("publication_state") not in {"pass", "review"}:
            raise ValueError("release has no terminal publication state")
        cases = payload.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("release has no case records")
        case = cases[0]
        if case.get("quality_status") not in {"pass", "review", "failed"}:
            raise ValueError("case has no terminal quality status")
        return "\n".join(
            [
                "# PE-LCB minute closed-loop acceptance result",
                "",
                f"- Publication state: {payload['publication_state']}",
                f"- Case: {case['case_id']}",
                f"- Case status: {case['quality_status']}",
                f"- Density: {case.get('density_g_cm3')} g/cm3",
                f"- Audit diagnosis: {case.get('audit_diagnosis')}",
                f"- Review reason: {case.get('review_reason')}",
                "",
                "This minute-scale result validates execution only and is not publication-grade data.",
            ]
        )


def acceptance_smoke(args: argparse.Namespace) -> int:
    root = project_root(args.project_root)
    gmx = resolve_gmx(args.gmx)
    report = self_test(root, gmx)
    if not report["acceptance_ready"] or not gmx:
        print(json.dumps(report, ensure_ascii=False))
        return 2
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    smoke = run_smoke(root, output, gmx, args.steep_nsteps, args.npt_steps)
    diagnosis = deterministic_audit(root, smoke)
    release = build_review_release(output, smoke, diagnosis)
    readable = validate_release(root, release)
    (output / "release.json").write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "result.md").write_text(readable, encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "publication_state": release["publication_state"],
        "case_status": release["cases"][0]["quality_status"],
        "density_g_cm3": smoke["density_g_per_cc"],
        "rule_verdict": smoke["verdict"],
        "audit_diagnosis": diagnosis.get("primary_audit_diagnosis"),
        "audit_route": diagnosis.get("recommended_reentry_module"),
        "release_path": str(output / "release.json"),
        "result_path": str(output / "result.md"),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def delegate_production(args: argparse.Namespace) -> int:
    target = os.getenv("POLYMER_AGENT_PRODUCTION_PIPELINE", "")
    if not target or not Path(target).is_file() or not COMMAND.fullmatch(target):
        raise RuntimeError("production runtime is not configured; acceptance smoke is not a production fallback")
    command = [
        target,
        "run",
        "--project-spec",
        args.project_spec,
        "--manifest",
        args.manifest,
        "--output",
        args.output,
        "--max-repair-attempts",
        str(args.max_repair_attempts),
        "--max-additional-md-ns",
        str(args.max_additional_md_ns),
    ]
    return int(subprocess.run(command, check=False).returncode)


def run_production_stage(args: argparse.Namespace) -> int:
    worker = Path(__file__).resolve().parent / "production_case.py"
    if not worker.is_file():
        raise RuntimeError("the versioned production-case worker is not installed")
    command = [
        sys.executable,
        str(worker),
        "--stage",
        args.stage,
        "--project-spec",
        args.project_spec,
        "--manifest",
        args.manifest,
        "--work-root",
        args.work_root,
        "--gmx",
        os.getenv("POLYMER_AGENT_GROMACS", "gmx"),
        "--emc",
        os.getenv("POLYMER_AGENT_EMC", "emc"),
    ]
    if args.case_index is not None:
        command.extend(["--case-index", str(args.case_index)])
    return int(subprocess.run(command, check=False).returncode)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pelcb-pipeline-core")
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("self-test")
    check.add_argument("--json", action="store_true")
    check.add_argument("--project-root")
    check.add_argument("--gmx")
    smoke = commands.add_parser("acceptance-smoke")
    smoke.add_argument("--project-root")
    smoke.add_argument("--gmx")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--steep-nsteps", type=int, default=200)
    smoke.add_argument("--npt-steps", type=int, default=500)
    run = commands.add_parser("run")
    run.add_argument("--project-spec", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--max-repair-attempts", type=int, choices=range(0, 4), default=1)
    run.add_argument("--max-additional-md-ns", type=int, choices=range(0, 401), default=400)
    stage = commands.add_parser("stage")
    stage.add_argument("--stage", choices=["build", "md", "postprocess", "repair"], required=True)
    stage.add_argument("--project-spec", required=True)
    stage.add_argument("--manifest", required=True)
    stage.add_argument("--work-root", required=True)
    stage.add_argument("--case-index", type=int)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "self-test":
        report = self_test(project_root(args.project_root), resolve_gmx(args.gmx))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "acceptance-smoke":
        return acceptance_smoke(args)
    if args.command == "stage":
        return run_production_stage(args)
    return delegate_production(args)


if __name__ == "__main__":
    raise SystemExit(main())
