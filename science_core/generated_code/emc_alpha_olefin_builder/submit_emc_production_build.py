from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a prepared EMC production-build package and sync the outputs into remote_build/<run_label>."
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="Path to the prepared emc_production_build directory.",
    )
    parser.add_argument(
        "--remote-root",
        default="${PELCB100_REMOTE_ROOT}/lz_project_runs/emc_alpha_olefin_builder",
        help="Remote root used to construct the package upload directory.",
    )
    parser.add_argument(
        "--remote-host",
        default="${REMOTE_USER}@private-3f2a50db.invalid",
        help="Remote SSH host.",
    )
    parser.add_argument(
        "--date-tag",
        default=datetime.now().strftime("%Y%m%d"),
        help="Date tag used in the remote package directory name.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=7200,
    )
    parser.add_argument(
        "--max-unknown-polls",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--adaptive-slurm-cpus",
        dest="adaptive_slurm_cpus",
        action="store_true",
        default=True,
        help="Probe current idle slots and adapt cpus/time before submit. Default: enabled.",
    )
    parser.add_argument(
        "--no-adaptive-slurm-cpus",
        dest="adaptive_slurm_cpus",
        action="store_false",
        help="Disable adaptive resource rewriting and use the packaged fallback sbatch header as-is.",
    )
    parser.add_argument("--min-adaptive-cpus", type=int, default=4)
    parser.add_argument("--max-adaptive-cpus", type=int, default=16)
    parser.add_argument(
        "--baseline-runtime-hours-at-8cpu",
        type=float,
        default=0.45,
        help="Observed EMC build runtime at the current production scale. Default: 0.45 h.",
    )
    parser.add_argument("--time-buffer-hours", type=float, default=0.75)
    parser.add_argument("--min-time-limit-hours", type=float, default=1.0)
    parser.add_argument("--max-time-limit-hours", type=float, default=2.0)
    parser.add_argument("--time-round-minutes", type=int, default=15)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only print the adaptive plan. Do not upload or submit.",
    )
    parser.add_argument(
        "--overwrite-canonical",
        action="store_true",
        help="Allow replacing an existing canonical remote_build/<run_label> directory after download.",
    )
    return parser.parse_args()


def load_manifest(package_dir: Path) -> dict:
    manifest_path = package_dir / "emc_package_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing package manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def latest_artifacts_dir(package_dir: Path) -> Path:
    matches = [path for path in package_dir.glob("artifacts_*") if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No artifacts_* directory found in {package_dir}")

    def sort_key(path: Path) -> tuple[int, float]:
        match = re.search(r"artifacts_(\d+)$", path.name)
        numeric = int(match.group(1)) if match else -1
        return (numeric, path.stat().st_mtime)

    return max(matches, key=sort_key)


def sync_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source directory to sync: {src}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"Canonical EMC build directory already exists: {dst}. Re-run with --overwrite-canonical to replace it."
            )
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def build_remote_dir(remote_root: str, case_id: str, package_name: str, date_tag: str) -> str:
    return f"{remote_root}/{case_id}_{package_name}_{date_tag}/{package_name}"


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    manifest = load_manifest(package_dir)
    case_id = manifest["case_id"]
    package_name = package_dir.name
    remote_dir = build_remote_dir(args.remote_root, case_id, package_name, args.date_tag)
    submit_script = manifest["submit_script"]

    command = [
        sys.executable,
        str(SCRIPT_DIR / "submit_remote_slurm_package.py"),
        "--local-dir",
        str(package_dir),
        "--remote-dir",
        remote_dir,
        "--submit-script",
        submit_script,
        "--remote-host",
        args.remote_host,
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-wait-seconds",
        str(args.max_wait_seconds),
        "--max-unknown-polls",
        str(args.max_unknown_polls),
    ]
    if args.adaptive_slurm_cpus:
        command.extend(
            [
                "--adaptive-slurm-cpus",
                "--min-adaptive-cpus",
                str(args.min_adaptive_cpus),
                "--max-adaptive-cpus",
                str(args.max_adaptive_cpus),
                "--baseline-runtime-hours-at-8cpu",
                str(args.baseline_runtime_hours_at_8cpu),
                "--time-buffer-hours",
                str(args.time_buffer_hours),
                "--min-time-limit-hours",
                str(args.min_time_limit_hours),
                "--max-time-limit-hours",
                str(args.max_time_limit_hours),
                "--time-round-minutes",
                str(args.time_round_minutes),
            ]
        )
    if args.probe_only:
        command.append("--probe-only")

    result = subprocess.run(command, cwd=str(SCRIPT_DIR), check=False)
    if result.returncode != 0:
        return result.returncode
    if args.probe_only:
        return 0

    artifacts_dir = latest_artifacts_dir(package_dir)
    run_label = manifest["run_label"]
    source_dir = artifacts_dir / run_label
    canonical_dir = Path(manifest["canonical_remote_build_dir"]).resolve()
    sync_tree(source_dir, canonical_dir, overwrite=args.overwrite_canonical)

    print(f"artifacts={artifacts_dir}")
    print(f"canonical={canonical_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
