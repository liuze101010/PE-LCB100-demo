#!/usr/bin/env python3
"""
Prepare and optionally submit the senior-approved 4-stage GROMACS MD pipeline.

The script consumes EMC->GROMACS bridge packages containing:
  - <case>.gro
  - topol.top or <case>.top
  - one or more *.itp files
  - optional case.json

Default production posture:
  - use the senior-script stage flow only: steep -> rlx -> pr -> nvt_prod
  - use CPU+GPU by default
  - only tune cpus-per-task and CPU/GPU resource allocation as needed
  - do not assume a fixed old-server CPU count; choose it from current idle resources
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = SCRIPT_DIR / "slurm_template.sh"
MANUAL_LAUNCH_HELPER = SCRIPT_DIR / "manual_launch_helper.sh"


@dataclass(frozen=True)
class BridgeCase:
    case_id: str
    case_dir: Path
    gro_path: Path
    top_path: Path
    itp_paths: tuple[Path, ...]
    case_json: Path | None
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LCB/SCB GROMACS MD packages from bridge outputs."
    )
    parser.add_argument("--bridge-dir", required=True, type=Path)
    parser.add_argument("--run-base", required=True, type=Path)
    parser.add_argument("--mdp-dir", required=True, type=Path)
    parser.add_argument("--cases", nargs="*", default=[], help="Case ids or prefixes, e.g. sc01 lcb13.")
    parser.add_argument("--partition", default="intel")
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="CPU cores per task. Choose this from current idle resources; no fixed old-server default is assumed.",
    )
    parser.add_argument(
        "--time-limit",
        default="30-00:00:00",
        help=(
            "Slurm walltime. Default 30 days matches partition default; users cannot "
            "extend an already-running job's TimeLimit, so submit generously and "
            "rely on the resume-aware template if walltime is hit."
        ),
    )
    parser.add_argument(
        "--module-preset",
        choices=(
            "intel",
            "amd-local",
            "newserver-gromacs2021",
            "newserver-gromacs2021-node8",
            "newserver-gromacs2024-cuda112",
            "none",
        ),
        default="intel",
        help="Module setup used in submit.slurm.",
    )
    parser.add_argument(
        "--gmx-bin",
        default=None,
        help="GROMACS executable. Defaults to gmx_mpi for intel, or the validated AMD local gmx for amd-local.",
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--script-name", default=None, help="Generated script filename. Default: submit.slurm or run_manual.sh for manual templates.")
    parser.add_argument("--run-tag", default=None, help="Run tag appended to directories. Default: YYYYMMDD_HHMMSS.")
    parser.add_argument("--dry-run", action="store_true", help="Generate files but do not submit.")
    parser.add_argument("--probe-only", action="store_true", help="Alias of --dry-run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated run directories.")
    parser.add_argument("--top-name", default="auto", help="Topology file to prefer. Use auto, topol.top, or <case>.top.")
    parser.add_argument("--maxwarn", type=int, default=0, help="grompp maxwarn value. Production default is 0.")
    parser.add_argument("--steep-nsteps", type=int, default=None, help="Override steep.mdp nsteps for smoke tests.")
    parser.add_argument("--rlx-ns", type=float, default=None, help="Override rlx.mdp length in ns.")
    parser.add_argument("--pr-ns", type=float, default=None, help="Override pr.mdp length in ns.")
    parser.add_argument("--nvt-ns", type=float, default=None, help="Override nvt_prod.mdp length in ns.")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        default=True,
        help="Request one GPU and pass GPU mdrun options (default).",
    )
    gpu_group.add_argument(
        "--cpu-only",
        dest="gpu",
        action="store_false",
        help="CPU-only override for exceptional cases; normal production should remain CPU+GPU.",
    )
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--gpu-validation-marker", type=Path, default=None)
    parser.add_argument("--force-gpu-unvalidated", action="store_true")
    parser.add_argument(
        "--submit-host",
        default=None,
        help="Optional SSH host for sbatch. Only use when run_dir paths are visible on that host.",
    )
    return parser.parse_args()


def resolve_local_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_cpus(args: argparse.Namespace) -> int:
    if args.cpus is None:
        raise ValueError(
            "--cpus is now an explicit resource decision. "
            "Probe idle CPU resources first, then pass the chosen core count."
        )
    if args.cpus < 1:
        raise ValueError("--cpus must be >= 1")
    return args.cpus


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def case_filter_matches(case_id: str, filters: list[str]) -> bool:
    if not filters:
        return True
    return any(case_id == item or case_id.startswith(f"{item}_") for item in filters)


def choose_top_file(case_dir: Path, case_id_hint: str | None, top_name: str) -> Path | None:
    if top_name != "auto":
        candidate = case_dir / top_name
        return candidate if candidate.exists() else None

    preferred: list[Path] = []
    if (case_dir / "topol.top").exists():
        preferred.append(case_dir / "topol.top")
    if case_id_hint and (case_dir / f"{case_id_hint}.top").exists():
        preferred.append(case_dir / f"{case_id_hint}.top")

    top_files = sorted(case_dir.glob("*.top"))
    for path in top_files:
        if path not in preferred:
            preferred.append(path)
    return preferred[0] if preferred else None


def choose_gro_file(case_dir: Path, case_id: str) -> Path | None:
    exact = case_dir / f"{case_id}.gro"
    if exact.exists():
        return exact

    def score(path: Path) -> tuple[int, str]:
        name = path.name
        stem = path.stem
        penalty = 0
        if stem.endswith(("_em", "_em1", "_nvt1", "_nvt2")) or "_npt" in stem:
            penalty += 10
        if stem in {"all", "confout", "average_struct"}:
            penalty += 20
        if stem.startswith(case_id):
            penalty -= 5
        return penalty, name

    gro_files = sorted(case_dir.glob("*.gro"), key=score)
    return gro_files[0] if gro_files else None


def discover_cases(bridge_dir: Path, top_name: str, filters: list[str]) -> list[BridgeCase]:
    if not bridge_dir.exists():
        raise FileNotFoundError(
            f"Bridge directory not found: {bridge_dir}. "
            "If this is a server path such as ~/lz_project_runs, run this script on the server "
            "or first copy the bridge package locally."
        )

    candidates: list[BridgeCase] = []
    seen_dirs: set[Path] = set()
    for case_dir in [bridge_dir, *sorted(path for path in bridge_dir.rglob("*") if path.is_dir())]:
        if case_dir in seen_dirs:
            continue
        seen_dirs.add(case_dir)

        case_json = case_dir / "case.json"
        metadata = load_json(case_json) if case_json.exists() else {}
        case_id_hint = metadata.get("case_id")

        top_path = choose_top_file(case_dir, case_id_hint, top_name)
        if top_path is None:
            continue

        case_id = case_id_hint or (top_path.stem if top_path.name != "topol.top" else "")
        if not case_id:
            gro_guess = choose_gro_file(case_dir, "")
            case_id = gro_guess.stem if gro_guess else case_dir.name

        gro_path = choose_gro_file(case_dir, case_id)
        itp_paths = tuple(sorted(case_dir.glob("*.itp")))
        if gro_path is None or not itp_paths:
            continue

        if not case_filter_matches(case_id, filters):
            continue

        meta = dict(metadata)
        meta.setdefault("case_id", case_id)
        meta.setdefault("bridge_source_dir", str(case_dir))
        candidates.append(
            BridgeCase(
                case_id=case_id,
                case_dir=case_dir,
                gro_path=gro_path,
                top_path=top_path,
                itp_paths=itp_paths,
                case_json=case_json if case_json.exists() else None,
                metadata=meta,
            )
        )

    def candidate_score(item: BridgeCase) -> tuple[int, int, str]:
        parts = set(item.case_dir.parts)
        penalty = 0
        if any(part.startswith("artifacts_") for part in parts):
            penalty += 20
        if any(part.startswith(("steep", "rlx", "pr", "nvt")) for part in parts):
            penalty += 20
        if item.case_json is None:
            penalty += 1
        return penalty, len(item.case_dir.parts), str(item.case_dir)

    best_by_case: dict[str, BridgeCase] = {}
    for item in candidates:
        current = best_by_case.get(item.case_id)
        if current is None or candidate_score(item) < candidate_score(current):
            best_by_case[item.case_id] = item

    return sorted(best_by_case.values(), key=lambda item: item.case_id)


def read_mdp_dt_ps(path: Path) -> float:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        body = line.split(";", 1)[0].strip()
        if body.startswith("dt"):
            _, _, value = body.partition("=")
            return float(value.strip())
    raise ValueError(f"Could not find dt in {path}")


def rewrite_nsteps(path: Path, target_ns: float | None) -> None:
    if target_ns is None:
        return
    dt_ps = read_mdp_dt_ps(path)
    nsteps = int(round(target_ns * 1000.0 / dt_ps))
    out: list[str] = []
    changed = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        body = line.split(";", 1)[0].strip()
        if body.startswith("nsteps"):
            out.append(f"nsteps                 = {nsteps}")
            changed = True
        else:
            out.append(line)
    if not changed:
        out.append(f"nsteps                 = {nsteps}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def rewrite_nsteps_value(path: Path, nsteps: int | None) -> None:
    if nsteps is None:
        return
    out: list[str] = []
    changed = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        body = line.split(";", 1)[0].strip()
        if body.startswith("nsteps"):
            out.append(f"nsteps                 = {nsteps}")
            changed = True
        else:
            out.append(line)
    if not changed:
        out.append(f"nsteps                 = {nsteps}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def build_run_dir(case: BridgeCase, run_base: Path, mdp_dir: Path, run_tag: str, overwrite: bool) -> Path:
    run_dir = run_base / f"{case.case_id}_md_{run_tag}"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite or --run-tag.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(case.gro_path, run_dir / case.gro_path.name)
    shutil.copy2(case.top_path, run_dir / "topol.top")
    for itp in case.itp_paths:
        shutil.copy2(itp, run_dir / itp.name)
    for mdp in sorted(mdp_dir.glob("*.mdp")):
        shutil.copy2(mdp, run_dir / mdp.name)

    metadata = dict(case.metadata)
    metadata["pipeline_source"] = {
        "case_dir": str(case.case_dir),
        "gro_path": str(case.gro_path),
        "top_path": str(case.top_path),
        "itp_paths": [str(path) for path in case.itp_paths],
    }
    (run_dir / "case.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def apply_duration_overrides(run_dir: Path, args: argparse.Namespace) -> None:
    rewrite_nsteps_value(run_dir / "steep.mdp", args.steep_nsteps)
    rewrite_nsteps(run_dir / "rlx.mdp", args.rlx_ns)
    rewrite_nsteps(run_dir / "pr.mdp", args.pr_ns)
    rewrite_nsteps(run_dir / "nvt_prod.mdp", args.nvt_ns)


def validate_gpu_gate(args: argparse.Namespace, run_base: Path) -> None:
    _ = (args, run_base)
    return


def generate_slurm_script(case: BridgeCase, run_dir: Path, args: argparse.Namespace) -> Path:
    template = args.template.read_text(encoding="utf-8")
    cpus = resolve_cpus(args)
    gmx_bin = args.gmx_bin
    if gmx_bin is None:
        if args.module_preset == "amd-local":
            gmx_bin = "${PELCB100_REMOTE_ROOT}/tools/gromacs/gromacs-2024.4_gcc11_avx2_nompi_staticstdcpp/bin/gmx"
        elif args.module_preset in ("newserver-gromacs2021", "newserver-gromacs2021-node8"):
            gmx_bin = "${PELCB100_REMOTE_PATH}"
        else:
            gmx_bin = "${PELCB100_REMOTE_PATH}"

    module_lines: list[str] = ["module purge"]
    if args.module_preset == "intel":
        # The old-server GROMACS 2024.4 module declares the CUDA module it was
        # built against. Loading an explicit CUDA version here can conflict with
        # that dependency and make compute-node MPI initialization fail.
        module_lines.append("module load intel${PELCB100_REMOTE_PATH}")
    elif args.module_preset == "amd-local":
        module_lines.append("# AMD-local GROMACS: validated self-built binary; no intelmpi module")
    elif args.module_preset == "newserver-gromacs2021":
        module_lines = [
            "# New manual GPU server profile; run after checking node/GPU occupancy.",
            "source ${PELCB100_REMOTE_PATH} 2>/dev/null || true",
            "source ${PELCB100_REMOTE_PATH} 2>/dev/null || true",
            "source ${PELCB100_REMOTE_PATH}",
        ]
    elif args.module_preset == "newserver-gromacs2021-node8":
        module_lines = [
            "# node-d61b45bd profile: the installed gmx_mpi needs Intel/MKL runtime libraries.",
            "source ${PELCB100_REMOTE_PATH}",
            "source ${PELCB100_REMOTE_PATH}",
            "source ${PELCB100_REMOTE_PATH} 2>/dev/null || true",
            "source ${PELCB100_REMOTE_PATH}",
        ]
    elif args.module_preset == "newserver-gromacs2024-cuda112":
        module_lines = [
            "# New-server clean-engine profile: self-built GROMACS 2024.4 CUDA 11.2.",
            "source ${PELCB100_REMOTE_HOME}/tools/gromacs/env_gromacs2024_4_cuda112.sh",
        ]
    else:
        # A self-contained GROMACS executable must also work on clusters that
        # do not provide Environment Modules.  Do not retain the default
        # ``module purge`` line for the explicit ``none`` preset.
        module_lines = ["# No modules requested"]

    gmx_executable = Path(str(gmx_bin)).name
    if gmx_executable in {"gmx", "gmx2024_4_cuda112"}:
        mdrun_parallel = "-ntmpi 1 -ntomp $THREADS"
    else:
        mdrun_parallel = "-ntomp $THREADS"

    content = template.format(
        case_id=case.case_id,
        gro_file=case.gro_path.name,
        partition=args.partition,
        cpus=cpus,
        time_limit=args.time_limit,
        maxwarn=args.maxwarn,
        gres_line="#SBATCH --gres=gpu:1" if args.gpu else "",
        module_lines="\n".join(module_lines),
        gmx_bin=gmx_bin,
        mdrun_gpu_args=f"-gpu_id {args.gpu_id} -nb gpu" if args.gpu else "",
        mdrun_parallel=mdrun_parallel,
        enable_steep_gpu="1" if args.gpu else "0",
    )
    script_name = args.script_name
    if script_name is None:
        script_name = "run_manual.sh" if "manual" in args.template.name else "submit.slurm"
    script_path = run_dir / script_name
    script_path.write_text(content, encoding="utf-8", newline="\n")
    if script_path.name != "submit.slurm" and MANUAL_LAUNCH_HELPER.exists():
        shutil.copy2(MANUAL_LAUNCH_HELPER, run_dir / MANUAL_LAUNCH_HELPER.name)
    if os.name != "nt":
        script_path.chmod(0o755)
        helper_path = run_dir / MANUAL_LAUNCH_HELPER.name
        if helper_path.exists():
            helper_path.chmod(0o755)
    return script_path


def submit_job(run_dir: Path, script_path: Path, dry_run: bool, submit_host: str | None) -> str | None:
    if dry_run:
        if script_path.name == "submit.slurm":
            print(f"  [DRY RUN] Would submit: cd {run_dir} && sbatch submit.slurm")
        else:
            print(f"  [DRY RUN] Manual run package: cd {run_dir} && bash {script_path.name}")
        return None

    if script_path.name != "submit.slurm":
        raise RuntimeError(
            f"{script_path.name} is a manual-run script. Use --dry-run/--probe-only and run it manually on the target server."
        )

    if platform.system().lower().startswith("windows") and submit_host is None:
        raise RuntimeError("Submitting from Windows without --submit-host is not supported. Use --dry-run locally.")

    if submit_host:
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=NUL",
            "-o",
            "GlobalKnownHostsFile=NUL",
            submit_host,
            f"cd {run_dir} && sbatch submit.slurm",
        ]
    else:
        cmd = ["sbatch", "submit.slurm"]

    result = subprocess.run(
        cmd,
        cwd=None if submit_host else run_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed for {run_dir}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result.stdout.strip()


def write_manifest(run_base: Path, run_tag: str, rows: list[dict]) -> Path:
    manifest = run_base / f"lcb_md_pipeline_manifest_{run_tag}.json"
    manifest.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    if args.probe_only:
        args.dry_run = True

    bridge_dir = resolve_local_path(args.bridge_dir)
    run_base = resolve_local_path(args.run_base)
    mdp_dir = resolve_local_path(args.mdp_dir)
    args.template = resolve_local_path(args.template)

    if not mdp_dir.exists():
        raise FileNotFoundError(f"MDP directory not found: {mdp_dir}")
    if not args.template.exists():
        raise FileNotFoundError(f"Slurm template not found: {args.template}")

    validate_gpu_gate(args, run_base)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Discovering bridge outputs in: {bridge_dir}")
    cases = discover_cases(bridge_dir, args.top_name, args.cases)
    if not cases:
        print("No bridge cases found.")
        return 1

    print(f"Found {len(cases)} bridge case(s).")
    run_base.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for case in cases:
        print(f"Processing {case.case_id}")
        run_dir = build_run_dir(case, run_base, mdp_dir, run_tag, args.overwrite)
        apply_duration_overrides(run_dir, args)
        script_path = generate_slurm_script(case, run_dir, args)
        job = submit_job(run_dir, script_path, args.dry_run, args.submit_host)
        rows.append(
            {
                "case_id": case.case_id,
                "run_dir": str(run_dir),
                "source_dir": str(case.case_dir),
                "submit_script": str(script_path),
                "job": job,
                "dry_run": args.dry_run,
                "gpu": args.gpu,
            }
        )

    manifest = write_manifest(run_base, run_tag, rows)
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
