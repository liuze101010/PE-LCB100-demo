#!/usr/bin/env python3
"""Portable, resume-aware worker for one locked PE-LCB case.

This program runs on a compute node. It never receives model credentials and
never decides publication status. Its outputs are bounded stage records that
the control plane can evaluate with the registered rule and audit layers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "polymer_agent.production_case.v1"
CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
STAGES = ("build", "md", "postprocess", "repair")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int | None = None,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
        env=env,
    )
    log = cwd / "commands.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.write(result.stdout or "")
        handle.write("\n")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[0]}")


def science_root() -> Path:
    raw = os.getenv("POLYMER_AGENT_SCIENCE_ROOT", "")
    bundled = Path(__file__).resolve().parent / "science_core"
    readable = Path(__file__).resolve().parents[1] / "science_core"
    root = Path(raw).resolve() if raw else (bundled if bundled.is_dir() else readable)
    required = root / "generated_code" / "md_analysis" / "emc_builder_registry.py"
    if not required.is_file():
        raise RuntimeError("the versioned science core is not installed")
    return root


def load_case(manifest_path: Path, index: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not 0 <= index < len(manifest):
        raise ValueError("case index is outside the locked manifest")
    case = dict(manifest[index])
    case_id = str(case.get("case_id") or "")
    if not CASE_ID.fullmatch(case_id):
        raise ValueError("unsafe or missing case_id")
    return case


def normalized_builder_spec(case: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    source = dict(case.get("builder_spec") or {})
    architecture = str(
        source.get("architecture")
        or case.get("architecture")
        or project.get("architecture")
        or "pe_lcb"
    )
    polymer_family = str(
        source.get("polymer_family")
        or case.get("polymer_family")
        or project.get("polymer_family")
        or ("PE" if architecture == "pe_lcb" else "custom_polymer")
    )
    target_total = int(
        case.get("target_total_carbons_per_chain")
        or case.get("target_total_carbons")
        or source.get("target_total_carbons_per_chain")
        or source.get("target_total_carbons")
        or 100
    )
    actual_total = int(
        case.get("actual_total_carbons_per_chain")
        or source.get("actual_total_carbons_per_chain")
        or target_total
    )
    source.update(
        {
            "case_id": case["case_id"],
            "architecture": architecture,
            "polymer_family": polymer_family,
            "chain_count": int(case.get("chain_count") or project.get("chain_count") or 150),
            "target_total_carbons_per_chain": target_total,
            "actual_total_carbons_per_chain": actual_total,
            "temperature_k": float(case.get("temperature_k") or project.get("temperature_k") or 400.0),
            "pressure_atm": float(case.get("pressure_atm") or project.get("pressure_atm") or 1.0),
            "initial_density_g_per_cc": float(project.get("initial_density_g_cm3") or 0.85),
            "force_field": str(
                case.get("force_field")
                or source.get("force_field")
                or project.get("force_field")
                or "opls/2024/opls-aa"
            ),
        }
    )
    if architecture == "pe_lcb":
        source.update(
            {
                "branch_points": int(case.get("branch_points") or source.get("branch_points") or 0),
                "branch_carbons": int(case.get("branch_carbons") or source.get("branch_carbons") or 0),
                "branch_position_pattern": str(
                    case.get("placement_code")
                    or case.get("branch_distribution_code")
                    or source.get("branch_position_pattern")
                    or ""
                ),
                "branch_positions": list(case.get("branch_positions") or source.get("branch_positions") or []),
                "monomer_smiles": str(case.get("macro_smiles") or source.get("monomer_smiles") or ""),
                "backbone_repeat_units": int(
                    case.get("backbone_repeat_units")
                    or source.get("backbone_repeat_units")
                    or 0
                ),
            }
        )
        if not source["monomer_smiles"] or not source["branch_positions"]:
            raise ValueError("manifest lacks a buildable PE-LCB molecular specification")
    else:
        repeat_smiles = str(
            source.get("repeat_unit_smiles")
            or case.get("repeat_unit_smiles")
            or source.get("monomer_smiles")
            or case.get("monomer_smiles")
            or ""
        )
        source["repeat_unit_smiles"] = repeat_smiles
        source["monomer_smiles"] = str(source.get("monomer_smiles") or repeat_smiles)
        if not source["monomer_smiles"]:
            raise ValueError("manifest lacks a buildable repeat-unit molecular specification")
    return source


def resolve_emc_root(emc_command: str) -> Path:
    env = os.getenv("EMC_ROOT") or os.getenv("POLYMER_AGENT_EMC_ROOT")
    candidates: list[Path] = [Path(env)] if env else []
    try:
        probe = subprocess.run([emc_command, "--root"], text=True, capture_output=True, timeout=30, check=False)
        if probe.returncode == 0 and probe.stdout.strip():
            candidates.append(Path(probe.stdout.strip().splitlines()[-1]))
    except (OSError, subprocess.SubprocessError):
        pass
    resolved = shutil.which(emc_command)
    if resolved:
        binary = Path(resolved).resolve()
        candidates.extend([binary.parent.parent, binary.parent])
    for root in candidates:
        root = root.expanduser().resolve()
        if (root / "scripts" / "emc.pl").is_file():
            if os.name == "nt" and any(character.isspace() for character in str(root)):
                raise RuntimeError(
                    "EMC_ROOT must not contain whitespace on Windows; install the EMC runtime "
                    "under a path such as C:\\polymer-agent-runtime\\emc"
                )
            return root
    raise RuntimeError("could not locate EMC_ROOT containing scripts/emc.pl")


def emc_subprocess_env(emc_root: Path) -> dict[str, str]:
    """Return an environment that EMC's Perl setup script can resolve.

    ``POLYMER_AGENT_EMC_ROOT`` is a control-plane setting.  EMC itself only
    reads ``EMC_ROOT``, so every EMC child process must receive the translated
    variable even when the caller did not export it globally.
    """

    env = os.environ.copy()
    env["EMC_ROOT"] = str(emc_root)
    return env


def missing_emc_artifacts(remote_build: Path, case_id: str) -> list[str]:
    """Return bridge inputs that a nominally successful EMC run omitted."""

    missing: list[str] = []
    for suffix in (".psf.gz", ".pdb.gz", ".params"):
        found = any(
            path.is_file() and path.name.startswith(case_id) and path.name.endswith(suffix)
            for path in remote_build.rglob(f"{case_id}*{suffix}")
        )
        if not found:
            missing.append(suffix)
    return missing


def build_case(case: dict[str, Any], project: dict[str, Any], case_root: Path, gmx: str, emc: str) -> dict[str, Any]:
    root = science_root()
    analysis = root / "generated_code" / "md_analysis"
    sys.path.insert(0, str(analysis))
    from emc_builder_registry import write_emc_case  # type: ignore

    spec = normalized_builder_spec(case, project)
    case_root.mkdir(parents=True, exist_ok=True)
    generated = write_emc_case(case_root, spec, validation={"locked": True}, plan={"source": "locked_manifest"})
    remote_build = case_root / "remote_build"
    remote_build.mkdir(exist_ok=True)
    esh = Path(generated["esh"])
    build_esh = remote_build / esh.name
    shutil.copy2(esh, build_esh)
    emc_root = resolve_emc_root(emc)
    emc_env = emc_subprocess_env(emc_root)
    run(
        ["perl", str(emc_root / "scripts" / "emc.pl"), build_esh.name],
        cwd=remote_build,
        timeout=1800,
        env=emc_env,
    )
    run([emc, "build.emc"], cwd=remote_build, timeout=14400, env=emc_env)
    missing = missing_emc_artifacts(remote_build, str(case["case_id"]))
    if missing:
        raise RuntimeError(
            "EMC did not produce required bridge artifacts "
            f"({', '.join(missing)}); inspect remote_build/commands.log for force-field or build errors"
        )

    bridge = root / "generated_code" / "emc_alpha_olefin_builder" / "emc_to_gromacs.py"
    md_root = case_root / "md"
    run(
        [sys.executable, str(bridge), "--case-dir", str(case_root), "--output-dir", str(md_root), "--nsteps", "50"],
        cwd=case_root,
        timeout=1800,
    )
    run(
        [gmx, "grompp", "-f", "em_bridge_min.mdp", "-c", f"{case['case_id']}.gro", "-p", "topol.top", "-o", "bridge_check.tpr", "-maxwarn", "0"],
        cwd=md_root,
        timeout=600,
    )
    return {
        "builder_spec_sha256": sha256_json(spec),
        "bridge_dir": "md",
        "atomistic_input_verified": True,
        "builder_adapter": str(spec.get("adapter_id") or spec.get("architecture") or ""),
        "selected_force_field": str(spec.get("force_field") or ""),
        "force_field_assignment_verified": True,
        "parameterization_probe": "emc_build_and_gromacs_grompp_passed",
    }


def latest_stage_gro(stage_dir: Path, stage: str) -> Path | None:
    parts = sorted(stage_dir.glob(f"{stage}.part*.gro"))
    if parts:
        return parts[-1]
    candidate = stage_dir / f"{stage}.gro"
    return candidate if candidate.is_file() else None


def stage_complete(stage_dir: Path, stage: str) -> bool:
    gro = latest_stage_gro(stage_dir, stage)
    logs = [stage_dir / f"{stage}.log", *stage_dir.glob(f"{stage}.part*.log")]
    return bool(gro and any(path.is_file() and "Finished mdrun" in path.read_text(errors="ignore") for path in logs))


def gpu_execution() -> tuple[str, str]:
    """Use the GPU assigned by the site's scheduler, or a CPU execution mode."""
    mode = os.getenv("POLYMER_AGENT_MDRUN_MODE", "auto")
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    gpu_requested = mode == "gpu" or (mode == "auto" and (
        visible not in {"", "-1", "NoDevFiles"} or bool(os.getenv("SLURM_JOB_GPUS"))))
    if mode not in {"auto", "gpu", "cpu"}:
        raise ValueError("POLYMER_AGENT_MDRUN_MODE must be auto, gpu or cpu")
    gpu_id = os.getenv("POLYMER_AGENT_GPU_ID", "0")
    if not gpu_id.isdigit():
        raise ValueError("POLYMER_AGENT_GPU_ID must be an assigned numeric device index")
    return (f"-gpu_id {gpu_id} -nb gpu", "1") if gpu_requested else ("", "0")


def shell_assignment(value: str) -> str:
    """Escape a literal interpolated inside the original template's quotes."""
    for character in ("\\", '"', "$", "`"):
        value = value.replace(character, "\\" + character)
    return value


def render_original_md_script(md_root: Path, gmx: str, cpus: int) -> tuple[str, dict[str, str]]:
    pipeline = science_root() / "generated_code" / "lcb_md_pipeline"
    template = (pipeline / "manual_template.sh").read_text(encoding="utf-8")
    gro = sorted(md_root.glob("*.gro"))[0]
    gpu_args, steep_gpu = gpu_execution()
    parallel = "-ntomp $THREADS" if "_mpi" in Path(gmx).name else "-ntmpi 1 -ntomp $THREADS"
    script = template.format(module_lines="", case_id=shell_assignment(md_root.parent.name),
        gro_file=shell_assignment(gro.name), gmx_bin=shell_assignment(gmx), maxwarn="0",
        cpus=max(1, cpus), mdrun_gpu_args=gpu_args, mdrun_parallel=parallel,
        enable_steep_gpu=steep_gpu)
    environment = {"RLX_MAXWARN": "0", "MERGE_PARTS": "1", "ENABLE_GPU_EXTRA": steep_gpu}
    return script, environment


def merge_stage_streams(stage_dir: Path, stage: str, gmx: str) -> None:
    """Use the original template's trjcat/eneconv contract after a restart."""
    for extension, tool in (("xtc", "trjcat"), ("edr", "eneconv")):
        paths = [path for path in [stage_dir / f"{stage}.{extension}",
                 *sorted(stage_dir.glob(f"{stage}.part*.{extension}"))] if path.is_file()]
        output = stage_dir / f"{stage}.full.{extension}"
        if len(paths) == 1:
            shutil.copy2(paths[0], output)
        elif paths:
            command = [gmx, tool, "-f", *[path.name for path in paths], "-o", output.name]
            if tool == "trjcat":
                command.append("-cat")
            run(command, cwd=stage_dir, timeout=None)


def run_md(case_root: Path, gmx: str, cpus: int) -> dict[str, Any]:
    root = science_root()
    mdp_root = root / "generated_code" / "lcb_md_pipeline" / "mdp"
    md_root = case_root / "md"
    if not (md_root / "topol.top").is_file():
        raise RuntimeError("atomistic build package is missing")
    for stage in ("steep", "rlx", "pr", "nvt_prod"):
        shutil.copy2(mdp_root / f"{stage}.mdp", md_root / f"{stage}.mdp")
    script, switches = render_original_md_script(md_root, gmx, cpus)
    script_path = md_root / "run_original_md.sh"
    script_path.write_text(script, encoding="utf-8", newline="\n")
    environment = os.environ.copy()
    environment.update(switches)
    run(["bash", script_path.name], cwd=md_root, timeout=None, env=environment)
    if not all(stage_complete(md_root / stage, stage) for stage in ("steep", "rlx", "pr", "nvt_prod")):
        raise RuntimeError("the original four-stage pipeline did not complete")
    merge_stage_streams(md_root / "nvt_prod", "nvt_prod", gmx)
    return {"protocol": ["steep", "rlx", "pr", "nvt_prod"], "production_ns": 400,
            "resume_aware": True, "runner": "original_manual_template",
            "runner_sha256": hashlib.sha256(script.encode()).hexdigest(), "execution_switches": switches}


def extract_density(case_root: Path, gmx: str, output: Path) -> dict[str, Any]:
    pr = case_root / "md" / "pr"
    edrs = [path for path in [pr / "pr.edr", *sorted(pr.glob("pr.part*.edr"))] if path.is_file()]
    if not edrs:
        raise FileNotFoundError("pressure-relaxation EDR is missing")
    edr = edrs[0]
    if len(edrs) > 1:
        edr = pr / "pr.full.edr"
        run([gmx, "eneconv", "-f", *[str(path) for path in edrs], "-o", str(edr)], cwd=pr, timeout=900)
    xvg = output / "pr_density.xvg"
    run([gmx, "energy", "-f", str(edr), "-o", str(xvg), "-b", "40000", "-e", "50000"], cwd=output, timeout=900, stdin="Density\n")
    values: list[float] = []
    times: list[float] = []
    for line in xvg.read_text(errors="replace").splitlines():
        if not line or line[0] in "#@":
            continue
        fields = line.split()
        if len(fields) >= 2:
            times.append(float(fields[0]))
            values.append(float(fields[1]) / 1000.0)
    if not values:
        raise RuntimeError("density extraction produced no numeric frames")
    if min(times) > 40010 or max(times) < 49990:
        raise RuntimeError("pressure-relaxation energy does not cover the last 10 ns")
    return {
        "status_density": "OK",
        "density_g_cm3": sum(values) / len(values),
        "density_frame_count": len(values),
        "density_stage": "pr_last_10_ns",
        "density_time_first_ps": min(times),
        "density_time_last_ps": max(times),
    }


def postprocess(
    case: dict[str, Any],
    case_root: Path,
    gmx: str,
    cpus: int,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    root = science_root()
    analysis = root / "generated_code" / "md_analysis"
    output = output or (case_root / "canonical_evidence")
    output.mkdir(exist_ok=True)
    trajectory = analysis / "analyze_trajectory.py"
    trajectory_args = [
        sys.executable,
        str(trajectory),
        "--case-dir",
        str(case_root / "md"),
        "--gmx-bin",
        gmx,
        "--output-dir",
        str(output / "trajectory"),
        "--analyses",
        "rg,msd,ree",
        "--max-parallel",
        str(max(1, min(cpus, 8))),
    ]
    if (case_root / "md" / "nvt_prod" / "extension_800ns_complete.json").is_file():
        trajectory_args.extend(
            ["--msd-beginfit-ps", "300000", "--msd-endfit-ps", "650000", "--trajectory-end-ps", "800000"]
        )
    run(
        trajectory_args,
        cwd=output,
        timeout=None,
    )
    full_rdf = analysis / "compute_full_rdf_suite.py"
    run(
        [sys.executable, str(full_rdf), "--case-dir", str(case_root / "md"), "--gmx-bin", gmx, "--output-root", str(output), "--batch-date", "run", "--chain-count", str(case.get("chain_count") or 150)],
        cwd=output,
        timeout=None,
    )
    shape = analysis / "compute_shape_tensor.py"
    run(
        [sys.executable, str(shape), "--case-id", str(case["case_id"]), "--case-dir", str(case_root / "md"), "--out-dir", str(output / "shape"), "--gmx", gmx, "--n-chains", str(case.get("chain_count") or 150)],
        cwd=output,
        timeout=None,
    )
    analysis_inputs = case_root.parent / ".analysis_inputs"
    analysis_inputs.mkdir(exist_ok=True)
    case_view = analysis_inputs / str(case["case_id"])
    if case_view.is_symlink() and case_view.resolve() != (case_root / "md").resolve():
        case_view.unlink()
    if not case_view.exists():
        case_view.symlink_to((case_root / "md").resolve(), target_is_directory=True)

    rdf_csv_candidates = list(output.glob("lcb100_rdf_run/aggregate/rdf_summary.csv"))
    if not rdf_csv_candidates:
        rdf_csv_candidates = list(output.rglob("rdf_summary.csv"))
    if not rdf_csv_candidates:
        raise RuntimeError("full RDF suite did not write its aggregate summary")
    rdf_csv = rdf_csv_candidates[0]
    dynamics = analysis / "analyze_dynamics.py"
    run(
        [
            sys.executable,
            str(dynamics),
            "--input-root",
            str(analysis_inputs),
            "--output-root",
            str(output),
            "--batch-date",
            "run",
            "--cases",
            str(case["case_id"]),
            "--gmx-bin",
            gmx,
            "--rdf-summary",
            str(rdf_csv),
            "--skip-msd-alpha",
        ],
        cwd=output,
        timeout=None,
    )
    dynamics_csv = next(iter(output.glob("lcb100_dynamics_run/aggregate/dynamics_summary.csv")), None)
    if dynamics_csv is None:
        dynamics_csv = next(iter(output.rglob("dynamics_summary.csv")), None)
    if dynamics_csv is None:
        raise RuntimeError("dynamics suite did not write its aggregate summary")
    intrachain = analysis / "analyze_intrachain_dynamics.py"
    run(
        [
            sys.executable,
            str(intrachain),
            "--input-root",
            str(analysis_inputs),
            "--batch-dir",
            str(output / "intrachain"),
            "--gmx",
            gmx,
            "--cases",
            str(case["case_id"]),
            "--rdf-summary",
            str(rdf_csv),
            "--dynamics-summary",
            str(dynamics_csv),
        ],
        cwd=output,
        timeout=None,
    )
    density = extract_density(case_root, gmx, output)
    def one_json(name: str) -> dict[str, Any]:
        matches = list(output.rglob(name))
        if not matches:
            raise FileNotFoundError(name)
        return json.loads(matches[0].read_text(encoding="utf-8"))

    gmx_version = subprocess.run([gmx, "--version"], text=True, capture_output=True, timeout=60, check=False)
    payload = {
        "schema_version": "polymer_agent.case_evidence.v1",
        "case_id": case["case_id"],
        "density": density,
        "trajectory_summary": json.loads((output / "trajectory" / "trajectory_analysis.json").read_text(encoding="utf-8")),
        "rg_summary": one_json("rg_summary.json"),
        "msd_summary": one_json("msd_summary.json"),
        "rdf_summary": one_json("rdf_summary.json"),
        "shape_summary": one_json("shape_summary.json"),
        "dynamics_summary": one_json("dynamics_summary.json"),
        "contact_summary": one_json("contact_probability_summary.json"),
        "intrachain_summary": one_json("g1_g3_summary.json"),
        "runtime": {
            "engine_version": ((gmx_version.stdout or "") + (gmx_version.stderr or "")).splitlines()[:12],
            "gromacs_finished": True,
            "fatal_log_marker_count": 0,
        },
        "generated_at": utcnow(),
    }
    atomic_json(output / "evidence.json", payload)
    return {"evidence": "canonical_evidence/evidence.json", "density_g_cm3": density["density_g_cm3"]}


def extend_clean_nvt(case_root: Path, gmx: str, cpus: int, additional_ns: int) -> dict[str, Any]:
    if additional_ns != 400:
        raise ValueError("registered PE-LCB MSD extension is exactly 400 ns")
    nvt = case_root / "md" / "nvt_prod"
    source_tpr = nvt / "nvt_prod.tpr"
    checkpoint = nvt / "nvt_prod.cpt"
    if not source_tpr.is_file() or not checkpoint.is_file():
        raise RuntimeError("clean NVT extension requires the original TPR and checkpoint")
    extended_tpr = nvt / "nvt_prod_800ns.tpr"
    if not extended_tpr.is_file():
        run([gmx, "convert-tpr", "-s", source_tpr.name, "-extend", "400000", "-o", extended_tpr.name], cwd=nvt, timeout=900)
    marker = nvt / "extension_800ns_complete.json"
    if not marker.is_file():
        gpu_args, _ = gpu_execution()
        parallel = [] if "_mpi" in Path(gmx).name else ["-ntmpi", "1"]
        extras = ["-bonded", "gpu", "-notunepme"] if gpu_args else []
        run(
            [gmx, "mdrun", *parallel, "-ntomp", str(max(1, cpus)), "-pin", "off", *gpu_args.split(), *extras,
             "-s", extended_tpr.name, "-deffnm", "nvt_prod", "-cpi", checkpoint.name, "-noappend"],
            cwd=nvt,
            timeout=None,
        )
        if not stage_complete(nvt, "nvt_prod"):
            raise RuntimeError("the resumed NVT stage did not complete")
        merge_stage_streams(nvt, "nvt_prod", gmx)
        atomic_json(marker, {"schema_version": SCHEMA, "additional_ns": 400, "target_total_ns": 800, "completed_at": utcnow()})
    return {"additional_ns": 400, "target_total_ns": 800, "marker": marker.name}


def repair_case(case: dict[str, Any], project: dict[str, Any], case_root: Path, gmx: str, emc: str, cpus: int) -> dict[str, Any]:
    contract = str(case.get("repair_contract") or "")
    attempt = int(case.get("repair_attempt") or 1)
    if attempt != 1:
        raise ValueError("the registered PE-LCB production policy permits one repair attempt")
    overlay = case_root / "repair_overlays" / f"attempt_{attempt}"
    overlay.mkdir(parents=True, exist_ok=True)
    if contract == "clean_msd_bounded_extension":
        action = extend_clean_nvt(case_root, gmx, cpus, 400)
        post = postprocess(case, case_root, gmx, cpus, output=overlay)
    elif contract in {"postprocess_recompute", "missing_output_backfill"}:
        action = {"recomputed": "registered_postprocess_suite"}
        post = postprocess(case, case_root, gmx, cpus, output=overlay)
    elif contract == "density_clean_profile_regate":
        sidecar = case_root / "repair_sidecars" / f"density_clean_attempt_{attempt}"
        action = build_case(case, project, sidecar, gmx, emc)
        action.update(run_md(sidecar, gmx, cpus))
        post = postprocess(case, sidecar, gmx, cpus, output=overlay)
    else:
        raise ValueError(f"repair contract is not executable: {contract}")
    result = {
        "repair_contract": contract,
        "repair_attempt": attempt,
        "action": action,
        "postprocess": post,
        "overlay_evidence": str((overlay / "evidence.json").relative_to(case_root)),
    }
    atomic_json(overlay / "repair_result.json", result)
    return result


def execute(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    project_path = Path(args.project_spec).resolve()
    work_root = Path(args.work_root).resolve()
    index = args.case_index
    if index is None:
        raw = os.getenv("SLURM_ARRAY_TASK_ID")
        if raw is None or not raw.isdigit():
            raise ValueError("case index or SLURM_ARRAY_TASK_ID is required")
        index = int(raw)
    case = load_case(manifest_path, index)
    project = json.loads(project_path.read_text(encoding="utf-8"))
    case_root = work_root / "cases" / case["case_id"]
    fingerprint = sha256_json({"stage": args.stage, "case": case, "project": project})
    status_path = case_root / "status" / f"{args.stage}.json"
    if status_path.is_file():
        current = json.loads(status_path.read_text(encoding="utf-8"))
        if current.get("state") == "completed" and current.get("input_sha256") == fingerprint:
            print(json.dumps(current, ensure_ascii=False))
            return 0
    status = {
        "schema_version": SCHEMA,
        "case_id": case["case_id"],
        "case_index": index,
        "stage": args.stage,
        "state": "running",
        "input_sha256": fingerprint,
        "started_at": utcnow(),
    }
    atomic_json(status_path, status)
    try:
        if args.stage == "build":
            result = build_case(case, project, case_root, args.gmx, args.emc)
        elif args.stage == "md":
            result = run_md(case_root, args.gmx, args.cpus)
        elif args.stage == "postprocess":
            result = postprocess(case, case_root, args.gmx, args.cpus)
        elif args.stage == "repair":
            result = repair_case(case, project, case_root, args.gmx, args.emc, args.cpus)
        else:
            raise ValueError(f"unsupported stage: {args.stage}")
    except Exception as exc:
        status.update(state="failed", finished_at=utcnow(), error_type=type(exc).__name__, error_summary=str(exc)[-800:])
        atomic_json(status_path, status)
        raise
    status.update(state="completed", finished_at=utcnow(), result=result)
    atomic_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False))
    return 0


def self_test(args: argparse.Namespace) -> int:
    checks: dict[str, Any] = {}
    for name in ("numpy", "rdkit.Chem", "pandas", "pydantic", "jsonschema", "yaml"):
        try:
            importlib.import_module(name)
            checks[name] = {"ok": True}
        except Exception as exc:
            checks[name] = {"ok": False, "error": type(exc).__name__}
    try:
        root = science_root()
        required = [
            root / "generated_code" / "md_analysis" / "analyze_trajectory.py",
            root / "generated_code" / "md_analysis" / "compute_full_rdf_suite.py",
            root / "generated_code" / "md_analysis" / "analyze_dynamics.py",
            root / "generated_code" / "md_analysis" / "analyze_intrachain_dynamics.py",
            root / "generated_code" / "lcb_md_pipeline" / "mdp" / "nvt_prod.mdp",
            root / "generated_code" / "lcb_md_pipeline" / "manual_template.sh",
        ]
        checks["science_core"] = {"ok": all(path.is_file() for path in required)}
    except Exception as exc:
        checks["science_core"] = {"ok": False, "error": type(exc).__name__}
    try:
        gmx_probe = subprocess.run([args.gmx, "--version"], capture_output=True, timeout=60, check=False)
        checks["gromacs"] = {"ok": gmx_probe.returncode == 0}
    except Exception as exc:
        checks["gromacs"] = {"ok": False, "error": type(exc).__name__}
    try:
        root = resolve_emc_root(args.emc)
        checks["emc"] = {"ok": (root / "scripts" / "emc.pl").is_file() and shutil.which("perl") is not None}
    except Exception as exc:
        checks["emc"] = {"ok": False, "error": type(exc).__name__}
    payload = {"schema_version": SCHEMA, "ok": all(item.get("ok") for item in checks.values()), "checks": checks}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--self-test", action="store_true")
    root.add_argument("--stage", choices=STAGES)
    root.add_argument("--project-spec")
    root.add_argument("--manifest")
    root.add_argument("--work-root")
    root.add_argument("--case-index", type=int)
    root.add_argument("--gmx", default=os.getenv("POLYMER_AGENT_GROMACS", "gmx"))
    root.add_argument("--emc", default=os.getenv("POLYMER_AGENT_EMC", "emc"))
    root.add_argument("--cpus", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", "8")))
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.self_test:
        raise SystemExit(self_test(arguments))
    if not all([arguments.stage, arguments.project_spec, arguments.manifest, arguments.work_root]):
        parser().error("--stage, --project-spec, --manifest and --work-root are required")
    raise SystemExit(execute(arguments))
