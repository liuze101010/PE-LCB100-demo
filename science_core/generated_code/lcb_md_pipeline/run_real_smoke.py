#!/usr/bin/env python3
"""Run a REAL smoke of the polymer MD framework end to end and emit real data.

Three modes:

- ``reproduce``  : extract density from an already-completed GROMACS energy file
                   (``gmx energy``) and apply the frozen density rule. No cluster,
                   no new MD -> reproduces an exact, correct density. Verifiable
                   locally with the bundled Windows ``gmx.exe``.
- ``probe-only`` : generate a smoke Slurm package (senior 4-stage, nsteps cut to
                   smoke scale) and print the plan. Does NOT submit.
- ``smoke``      : generate the smoke package, submit it to the remote Slurm
                   cluster, wait, download artifacts, then extract+classify the
                   real (short-run, NOT converged) density.

This reuses the existing senior pipeline rather than reinventing it:
- ``lcb_md_submit.py``                 -> generate the 4-stage package
- ``submit_remote_slurm_package.py``   -> upload / sbatch / poll / download
- ``md_analysis/configs/pelcb100.json``-> frozen density-tier thresholds

SMOKE NUMBERS ARE NOT PUBLICATION-GRADE. A ~2 ps run gives a real but
unequilibrated density; the point is to prove the pipeline truly executes and
the rule truly classifies. Use ``reproduce`` for an exact correct value.

Security: SSH key auth only (never a password in code); the remote submit step
probes first and asks for confirmation unless ``--yes`` is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # .../lz project
LCB_MD_SUBMIT = SCRIPT_DIR / "lcb_md_submit.py"
MDP_DIR = SCRIPT_DIR / "mdp"
SUBMIT_REMOTE = (
    PROJECT_ROOT / "generated_code" / "emc_alpha_olefin_builder" / "submit_remote_slurm_package.py"
)
PELCB_CONFIG = PROJECT_ROOT / "generated_code" / "md_analysis" / "configs" / "pelcb100.json"
LOCAL_GMX = Path(r"${LOCAL_ROOT}\GROMACS\gmx2020.6_GPU\bin\gmx.exe")

# Smoke scale: steep iterations + ~2 ps per dynamics stage. Senior physics
# params (Tcoupl/Pcoupl/ref_t/...) are untouched; only nsteps are cut.
SMOKE_STEEP_NSTEPS = 2000
SMOKE_STAGE_NS = 0.002

DENSITY_TERM = "Density"
DENSITY_LINE_RE = re.compile(r"^\s*Density\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b")


# --------------------------------------------------------------------------- #
# density rule (reuse frozen thresholds; do not reinvent)
# --------------------------------------------------------------------------- #

def load_density_thresholds(config_path: Path = PELCB_CONFIG) -> dict[str, float]:
    """PE-LCB density screening thresholds in g/cm^3."""
    pass_low, pass_high, warn_low = 0.73, 0.87, 0.66
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        tiers = cfg.get("compaction_density_tiers", {})
        pass_low = float(cfg.get("rho_pass_low_g_per_cc", tiers.get("pass_low", pass_low)))
        pass_high = float(cfg.get("rho_pass_high_g_per_cc", tiers.get("pass_high", pass_high)))
        warn_low = float(cfg.get("rho_warn_low_g_per_cc", tiers.get("warn_low", warn_low)))
    except Exception:
        pass
    return {"pass_low": pass_low, "pass_high": pass_high, "warn_low": warn_low}


def classify_density(density_g_per_cc: float, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Apply the frozen density-tier rule -> PASS / WARN (+ tier)."""
    defaults = load_density_thresholds()
    th = thresholds or defaults
    pass_low = th.get("pass_low", defaults["pass_low"])
    pass_high = th.get("pass_high", defaults["pass_high"])
    warn_low = th.get("warn_low", defaults["warn_low"])
    if pass_low <= density_g_per_cc <= pass_high:
        verdict, tier, warn_source = "PASS", "in_range", None
    elif density_g_per_cc > pass_high:
        verdict, tier, warn_source = "WARN", "above_pass_high", "density_warn"
    elif density_g_per_cc >= warn_low:
        verdict, tier, warn_source = "WARN", "below_pass_low", "density_warn"
    else:
        verdict, tier, warn_source = "WARN", "below_warn_low", "density_warn"
    return {
        "density_g_per_cc": round(density_g_per_cc, 4),
        "verdict": verdict,
        "tier": tier,
        "warn_source": warn_source,
        "rule": {"pass_low": pass_low, "pass_high": pass_high, "warn_low": warn_low, "unit": "g_per_cc"},
    }


# --------------------------------------------------------------------------- #
# density extraction via gmx energy (real computation)
# --------------------------------------------------------------------------- #

def resolve_gmx(gmx_bin: str | None) -> str:
    if gmx_bin:
        return gmx_bin
    if LOCAL_GMX.exists():
        return str(LOCAL_GMX)
    return "gmx"


def extract_density_from_edr(edr_path: Path, gmx_bin: str | None = None, work_dir: Path | None = None) -> float:
    """Run ``gmx energy`` on an .edr and return mean density in g/cm^3.

    gmx reports density in kg/m^3; we divide by 1000. Selection is by the term
    name 'Density' fed on stdin (no BOM, unlike a PowerShell pipe).
    """
    edr_path = Path(edr_path)
    if not edr_path.exists():
        raise FileNotFoundError(f"energy file not found: {edr_path}")
    gmx = resolve_gmx(gmx_bin)
    work = work_dir or edr_path.parent
    out_xvg = Path(work) / "_density_smoke.xvg"
    proc = subprocess.run(
        [gmx, "energy", "-f", str(edr_path), "-o", str(out_xvg)],
        input=f"{DENSITY_TERM}\n\n",
        text=True,
        capture_output=True,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in combined.splitlines():
        match = DENSITY_LINE_RE.match(line)
        if match:
            return float(match.group(1)) / 1000.0
    raise RuntimeError(
        "could not parse Density from gmx energy output (no 'Density' term?).\n"
        + combined[-1500:]
    )


def find_energy_file(artifacts_dir: Path) -> Path:
    """Pick the most production-like .edr from a downloaded run (nvt_prod > pr)."""
    artifacts_dir = Path(artifacts_dir)
    for pref in ("nvt_prod", "pr", "rlx"):
        hits = sorted(artifacts_dir.rglob(f"{pref}*.edr"))
        if hits:
            return hits[0]
    any_edr = sorted(artifacts_dir.rglob("*.edr"))
    if not any_edr:
        raise FileNotFoundError(f"no .edr under {artifacts_dir}")
    return any_edr[0]


# --------------------------------------------------------------------------- #
# local real smoke (steep + short NPT) using a local GROMACS — no cluster
# --------------------------------------------------------------------------- #

def _find_case_inputs(bridge_dir: Path, case: str) -> tuple[Path, Path, list[Path]]:
    gro = bridge_dir / f"{case}.gro"
    if not gro.exists():
        gros = sorted(bridge_dir.glob("*.gro"))
        if not gros:
            raise FileNotFoundError(f"no .gro under {bridge_dir}")
        gro = gros[0]
    top = bridge_dir / "topol.top"
    if not top.exists():
        top = bridge_dir / f"{case}.top"
    if not top.exists():
        tops = sorted(bridge_dir.glob("*.top"))
        if not tops:
            raise FileNotFoundError(f"no .top under {bridge_dir}")
        top = tops[0]
    return gro, top, sorted(bridge_dir.glob("*.itp"))


def _smoke_npt_mdp(npt_steps: int) -> str:
    # Short NPT smoke: frequent energy frames + velocity generation so a 2-3 ps
    # run yields a usable density. Senior physics knobs are otherwise preserved.
    lines = [
        "integrator = md", "dt = 0.001", f"nsteps = {npt_steps}",
        "nstcomm = 100", "nstlog = 500", "nstenergy = 100", "nstxout-compressed = 0",
        "cutoff-scheme = Verlet", "nstlist = 10", "pbc = xyz", "rlist = 1.0",
        "coulombtype = PME", "rcoulomb = 1.0", "fourierspacing = 0.15",
        "vdwtype = cut-off", "rvdw = 1.0", "DispCorr = EnerPres",
        "Tcoupl = V-rescale", "tc-grps = system", "tau_t = 0.5", "ref_t = 298",
        "pcoupl = Berendsen", "tau-p = 0.8", "compressibility = 4.5e-5", "ref-p = 1.0",
        "gen_vel = yes", "gen_temp = 298", "gen_seed = -1", "",
    ]
    return "\n".join(lines)


def _gmx(gmx_bin: str, gmx_args: list[str], cwd: Path) -> None:
    proc = subprocess.run(
        [gmx_bin, *gmx_args], cwd=str(cwd), input="", text=True, capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gmx {' '.join(gmx_args[:2])} failed (rc={proc.returncode}):\n"
            + ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
        )


def run_local_smoke(
    case: str,
    bridge_dir: Path,
    gmx_bin: str | None,
    work_dir: Path,
    steep_nsteps: int,
    npt_steps: int,
) -> Path:
    """Run a REAL short MD locally (steep -> short NPT). Returns the NPT .edr.

    Uses a built bridge case + a local GROMACS. No cluster, no SSH. The density
    is real but smoke-scale (and bridge-test FF) -> not a production value.
    """
    gmx = resolve_gmx(gmx_bin)
    gro, top, itps = _find_case_inputs(bridge_dir, case)
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(gro, work_dir / "conf.gro")
    shutil.copy(top, work_dir / "topol.top")
    for itp in itps:
        shutil.copy(itp, work_dir / itp.name)
    shutil.copy(MDP_DIR / "steep.mdp", work_dir / "steep.mdp")
    (work_dir / "rlx_smoke.mdp").write_text(_smoke_npt_mdp(npt_steps), encoding="utf-8")

    print(f"[local-smoke] work dir: {work_dir}")
    _gmx(gmx, ["editconf", "-f", "conf.gro", "-o", "all.gro", "-c"], work_dir)
    _gmx(gmx, ["grompp", "-f", "steep.mdp", "-c", "all.gro", "-p", "topol.top",
               "-o", "steep.tpr", "-maxwarn", "20"], work_dir)
    print(f"[local-smoke] steep minimization (cap {steep_nsteps} steps)…")
    _gmx(gmx, ["mdrun", "-deffnm", "steep", "-nsteps", str(steep_nsteps), "-pin", "off"], work_dir)
    _gmx(gmx, ["grompp", "-f", "rlx_smoke.mdp", "-c", "steep.gro", "-p", "topol.top",
               "-o", "rlx.tpr", "-maxwarn", "20"], work_dir)
    print(f"[local-smoke] short NPT ({npt_steps} steps ~ {npt_steps / 1000:.1f} ps)…")
    _gmx(gmx, ["mdrun", "-deffnm", "rlx", "-pin", "off"], work_dir)
    edr = work_dir / "rlx.edr"
    if not edr.exists():
        raise RuntimeError("local smoke produced no rlx.edr")
    return edr


# --------------------------------------------------------------------------- #
# smoke package generation + remote submit (reuse existing scripts)
# --------------------------------------------------------------------------- #

def generate_smoke_package(
    case: str,
    bridge_dir: Path,
    run_base: Path,
    cpus: int,
    run_tag: str,
    probe_only: bool,
) -> None:
    cmd = [
        sys.executable, str(LCB_MD_SUBMIT),
        "--bridge-dir", str(bridge_dir),
        "--run-base", str(run_base),
        "--mdp-dir", str(MDP_DIR),
        "--cases", case,
        "--partition", "intel",
        "--module-preset", "intel",
        "--cpus", str(cpus),
        "--run-tag", run_tag,
        "--overwrite",
        "--steep-nsteps", str(SMOKE_STEEP_NSTEPS),
        "--rlx-ns", str(SMOKE_STAGE_NS),
        "--pr-ns", str(SMOKE_STAGE_NS),
        "--nvt-ns", str(SMOKE_STAGE_NS),
    ]
    if probe_only:
        cmd.append("--probe-only")
    print("[smoke] generating package:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def submit_and_collect(
    local_dir: Path,
    remote_dir: str,
    submit_script: str,
    remote_host: str,
    artifact_dir: Path,
) -> Path:
    cmd = [
        sys.executable, str(SUBMIT_REMOTE),
        "--local-dir", str(local_dir),
        "--remote-dir", remote_dir,
        "--submit-script", submit_script,
        "--remote-host", remote_host,
        "--adaptive-slurm-cpus",
        "--artifact-dir-name", artifact_dir.name,
    ]
    print("[smoke] submitting to cluster:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return artifact_dir


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def write_result(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[result] wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("reproduce", "local-smoke", "probe-only", "smoke"), required=True)
    p.add_argument("--case", default="", help="Case id (e.g. lcb01...). Required for probe-only/smoke.")
    p.add_argument("--traj", type=Path, help="reproduce mode: path to a completed .edr (or run dir).")
    p.add_argument("--bridge-dir", type=Path, help="Built EMC->GROMACS bridge dir (has <case>.gro/.top).")
    p.add_argument("--run-base", type=Path, default=SCRIPT_DIR / "output" / "real_smoke")
    p.add_argument("--cpus", type=int, default=8, help="cpus-per-task for the smoke job.")
    p.add_argument("--run-tag", default="real_smoke")
    p.add_argument("--remote-host", default="")
    p.add_argument("--remote-root", default="/tmp/polymer_agent_runs")
    p.add_argument("--gmx-bin", default=None, help="GROMACS binary for local density extraction.")
    p.add_argument("--out", type=Path, default=None, help="Result JSON path.")
    p.add_argument("--yes", action="store_true", help="Skip the interactive confirm before remote submit.")
    p.add_argument("--work-dir", type=Path, default=None, help="local-smoke: scratch dir (default run-base/local_smoke_<case>).")
    p.add_argument("--steep-nsteps", type=int, default=1000, help="local-smoke: steepest-descent step cap.")
    p.add_argument("--npt-steps", type=int, default=2500, help="local-smoke: short NPT steps (dt=1 fs).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = load_density_thresholds()

    if args.mode == "reproduce":
        if not args.traj:
            print("ERROR: --traj <edr or run dir> is required for reproduce mode", file=sys.stderr)
            return 2
        edr = args.traj if args.traj.suffix == ".edr" else find_energy_file(args.traj)
        density = extract_density_from_edr(edr, args.gmx_bin)
        verdict = classify_density(density, thresholds)
        result = {
            "mode": "reproduce",
            "case": args.case or None,
            "is_smoke": False,
            "energy_file": str(edr),
            **verdict,
            "note": "Reproduced exact density from a completed trajectory via gmx energy.",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.out:
            write_result(args.out, result)
        return 0

    if args.mode == "local-smoke":
        if not args.case or not args.bridge_dir:
            print("ERROR: --case and --bridge-dir are required for local-smoke", file=sys.stderr)
            return 2
        work = args.work_dir or (args.run_base / f"local_smoke_{args.case}")
        edr = run_local_smoke(
            args.case, args.bridge_dir, args.gmx_bin, work, args.steep_nsteps, args.npt_steps
        )
        density = extract_density_from_edr(edr, args.gmx_bin)
        verdict = classify_density(density, thresholds)
        result = {
            "mode": "local-smoke",
            "case": args.case,
            "is_smoke": True,
            "energy_file": str(edr),
            "smoke_scale": {
                "steep_nsteps": args.steep_nsteps,
                "npt_steps": args.npt_steps,
                "npt_ps": round(args.npt_steps / 1000, 2),
            },
            **verdict,
            "note": "Local real short MD (steep + short NPT). SMOKE scale + bridge-test FF -> real but NOT a production density.",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        write_result(args.out or (work / "local_smoke_result.json"), result)
        return 0

    if not args.case or not args.bridge_dir:
        print("ERROR: --case and --bridge-dir are required for probe-only/smoke", file=sys.stderr)
        return 2

    generate_smoke_package(
        case=args.case,
        bridge_dir=args.bridge_dir,
        run_base=args.run_base,
        cpus=args.cpus,
        run_tag=args.run_tag,
        probe_only=(args.mode == "probe-only"),
    )

    if args.mode == "probe-only":
        print("[probe-only] package generated; NOT submitted. Inspect under:", args.run_base)
        return 0

    # smoke: confirm before touching the shared cluster
    if not args.yes:
        print(
            f"\nAbout to submit a REAL smoke job for '{args.case}' to {args.remote_host}.\n"
            "This consumes shared cluster resources. Type 'yes' to proceed: ",
            end="",
        )
        if input().strip().lower() != "yes":
            print("[smoke] aborted by user.")
            return 1

    case_dirs = sorted(args.run_base.glob(f"{args.case}*"))
    if not case_dirs:
        print(f"ERROR: generated package dir for {args.case} not found under {args.run_base}", file=sys.stderr)
        return 3
    local_dir = case_dirs[0]
    submit_script = "submit.slurm"
    remote_dir = f"{args.remote_root.rstrip(chr(47))}/{local_dir.name}"
    artifact_dir = args.run_base / f"artifacts_{args.case}"

    submit_and_collect(local_dir, remote_dir, submit_script, args.remote_host, artifact_dir)

    edr = find_energy_file(artifact_dir)
    density = extract_density_from_edr(edr, args.gmx_bin)
    verdict = classify_density(density, thresholds)
    result = {
        "mode": "smoke",
        "case": args.case,
        "is_smoke": True,
        "remote_host": args.remote_host,
        "remote_dir": remote_dir,
        "energy_file": str(edr),
        "smoke_scale": {"steep_nsteps": SMOKE_STEEP_NSTEPS, "stage_ns": SMOKE_STAGE_NS},
        **verdict,
        "note": "SMOKE scale (~2 ps): real pipeline + real density, but NOT converged.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = args.out or (artifact_dir / "real_smoke_result.json")
    write_result(out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
