from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def write_text_unix(path: Path, content: str, encoding: str = "utf-8") -> None:
    with path.open("w", encoding=encoding, newline="\n") as handle:
        handle.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Slurm-submittable EMC production-build packages from case directories."
    )
    parser.add_argument(
        "--case-dir",
        action="append",
        type=Path,
        required=True,
        help="Case directory containing case.json and a single .esh file. May be repeated.",
    )
    parser.add_argument(
        "--package-name",
        default="emc_production_build",
        help="Per-case package directory name. Default: emc_production_build.",
    )
    parser.add_argument(
        "--run-label",
        help="Optional EMC output subdirectory label. Defaults to the parent run directory name, e.g. screen_c1_c20.",
    )
    parser.add_argument(
        "--partition",
        default="amd",
        help="Default Slurm partition for the packaged submit script. Default: amd.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="Conservative local fallback for direct sbatch. Dynamic production submission may rewrite this value.",
    )
    parser.add_argument(
        "--time-limit",
        default="02:00:00",
        help="Conservative local fallback for direct sbatch. Dynamic production submission may rewrite this value.",
    )
    parser.add_argument(
        "--remote-emc-root",
        default="${PELCB100_REMOTE_ROOT}/tools/emc",
        help="Remote EMC installation root.",
    )
    parser.add_argument(
        "--job-name-suffix",
        default="emcbuild",
        help="Suffix used in the packaged Slurm job name.",
    )
    return parser.parse_args()


def load_case_metadata(case_dir: Path) -> dict:
    case_json = case_dir / "case.json"
    if not case_json.exists():
        raise FileNotFoundError(f"Missing case.json in {case_dir}")
    return json.loads(case_json.read_text(encoding="utf-8"))


def detect_esh(case_dir: Path) -> Path:
    matches = sorted(case_dir.glob("*.esh"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one .esh file in {case_dir}, found {len(matches)}")
    return matches[0]


def detect_run_label(case_dir: Path, explicit_run_label: str | None) -> str:
    if explicit_run_label:
        return explicit_run_label
    return case_dir.parent.name


def copy_required_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required input file: {src}")
    shutil.copy2(src, dst)


def write_submit_script(
    path: Path,
    *,
    case_id: str,
    esh_name: str,
    partition: str,
    cpus_per_task: int,
    time_limit: str,
    remote_emc_root: str,
    run_label: str,
    job_name_suffix: str,
) -> None:
    content = f"""#!/bin/bash
#SBATCH --job-name={case_id}-{job_name_suffix}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time_limit}
#SBATCH --output=slurm-%j.out

set -euo pipefail

# NOTE:
# - This local sbatch uses a conservative fallback resource header.
# - For slot-aware production submission, prefer submit_remote_slurm_package.py
#   or submit_slurm_batch_dynamic.py so cpus/time can be adapted at submit time.

module purge >/dev/null 2>&1 || true

WORK_DIR="${{SLURM_SUBMIT_DIR:-$PWD}}"
cd "$WORK_DIR"

export EMC_ROOT="{remote_emc_root}"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

RUN_DIR="$WORK_DIR/{run_label}"
mkdir -p "$RUN_DIR"
cp "{esh_name}" "$RUN_DIR/{esh_name}"

echo "EMC_BUILD_START=$(date -Is)"
echo "HOSTNAME=$(hostname)"
echo "PWD=$WORK_DIR"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-unknown}}"
echo "SLURM_JOB_NODELIST=${{SLURM_JOB_NODELIST:-unknown}}"
echo "EMC_ROOT=$EMC_ROOT"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

cd "$RUN_DIR"

set +e
perl "$EMC_ROOT/scripts/emc.pl" "{esh_name}" > setup.log 2>&1
setup_status=$?
set -e
printf '%s\\n' "$setup_status" > setup.exitcode
if [ "$setup_status" -ne 0 ]; then
  exit "$setup_status"
fi

rm -f "{esh_name}"

set +e
"$EMC_ROOT/bin/emc_linux_x86_64" build.emc > build.log 2>&1
build_status=$?
set -e
printf '%s\\n' "$build_status" > build.exitcode

echo "EMC_BUILD_END=$(date -Is)"
exit "$build_status"
"""
    write_text_unix(path, content, encoding="ascii")


def write_readme(
    path: Path,
    *,
    case_id: str,
    package_dir: Path,
    case_dir: Path,
    run_label: str,
    esh_name: str,
    partition: str,
    cpus_per_task: int,
    time_limit: str,
    remote_emc_root: str,
) -> None:
    content = f"""# EMC Production Build Package: {case_id}

This package wraps the EMC production-build step for `{case_id}` into a Slurm-
submittable directory so the remaining `.esh`-only cases can advance through a
queue-aware CPU workflow instead of a direct SSH-only build.

Protocol:

1. Copy `{esh_name}` into a run-label subdirectory
2. Run `perl $EMC_ROOT/scripts/emc.pl {esh_name}` to generate `build.emc`
3. Run `emc_linux_x86_64 build.emc`
4. Download the package artifacts, then sync `{run_label}` back into the case's canonical `remote_build` tree

Key settings:

- package directory = `{package_dir}`
- case directory = `{case_dir}`
- run label = `{run_label}`
- partition = `{partition}`
- fallback `cpus-per-task` = `{cpus_per_task}`
- fallback `time limit` = `{time_limit}`
- remote EMC root = `{remote_emc_root}`

Scope:

- move `.esh + case.json` cases into a reusable Slurm package path
- preserve the established EMC output convention under `remote_build/{run_label}`
- keep the local sbatch conservative; production slot-aware submission may rewrite CPU/time at submit time

Main files:

- `{esh_name}`
- `case.json`
- `submit_emc_production_build.sbatch`
- `emc_package_manifest.json`
"""
    write_text_unix(path, content, encoding="utf-8")


def prepare_case(
    case_dir: Path,
    *,
    package_name: str,
    run_label: str,
    partition: str,
    cpus_per_task: int,
    time_limit: str,
    remote_emc_root: str,
    job_name_suffix: str,
) -> Path:
    case_dir = case_dir.resolve()
    meta = load_case_metadata(case_dir)
    case_id = meta["case_id"]
    esh_path = detect_esh(case_dir)
    package_dir = (case_dir / package_name).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)

    copy_required_file(case_dir / "case.json", package_dir / "case.json")
    copy_required_file(esh_path, package_dir / esh_path.name)

    write_submit_script(
        package_dir / "submit_emc_production_build.sbatch",
        case_id=case_id,
        esh_name=esh_path.name,
        partition=partition,
        cpus_per_task=cpus_per_task,
        time_limit=time_limit,
        remote_emc_root=remote_emc_root,
        run_label=run_label,
        job_name_suffix=job_name_suffix,
    )
    manifest = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "package_dir": str(package_dir),
        "run_label": run_label,
        "submit_script": "submit_emc_production_build.sbatch",
        "esh_name": esh_path.name,
        "canonical_remote_build_dir": str(case_dir / "remote_build" / run_label),
        "remote_emc_root": remote_emc_root,
        "fallback_partition": partition,
        "fallback_cpus_per_task": cpus_per_task,
        "fallback_time_limit": time_limit,
    }
    write_text_unix(
        package_dir / "emc_package_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_readme(
        package_dir / "README.md",
        case_id=case_id,
        package_dir=package_dir,
        case_dir=case_dir,
        run_label=run_label,
        esh_name=esh_path.name,
        partition=partition,
        cpus_per_task=cpus_per_task,
        time_limit=time_limit,
        remote_emc_root=remote_emc_root,
    )
    return package_dir


def main() -> int:
    args = parse_args()
    prepared: list[Path] = []
    for raw_case_dir in args.case_dir:
        case_dir = raw_case_dir.resolve()
        run_label = detect_run_label(case_dir, args.run_label)
        package_dir = prepare_case(
            case_dir,
            package_name=args.package_name,
            run_label=run_label,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            time_limit=args.time_limit,
            remote_emc_root=args.remote_emc_root,
            job_name_suffix=args.job_name_suffix,
        )
        prepared.append(package_dir)
        print(package_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
