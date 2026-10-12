from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath


SCRIPT_DIR = Path(__file__).resolve().parent
SAFE_REMOTE_ROOT = PurePosixPath("${PELCB100_REMOTE_ROOT}/lz_project_runs")
REQUIRED_SUFFIXES = (".data", ".params", ".vmd")


@dataclass
class PackageRecord:
    task_index: int
    case_id: str
    package_dir: Path
    package_name: str
    run_label: str
    esh_name: str
    canonical_remote_build_dir: Path


@dataclass
class CaseArrayResult:
    task_index: int
    case_id: str
    package_dir: str
    artifact_dir: str | None
    canonical_dir: str
    setup_exitcode: str | None
    build_exitcode: str | None
    has_data: bool
    has_params: bool
    has_vmd: bool
    has_pdb_gz: bool
    has_psf_gz: bool
    ok: bool
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit many prepared EMC production-build packages as one Slurm array, "
            "then sync artifacts into canonical remote_build/<run_label>."
        )
    )
    parser.add_argument("--package-dir", action="append", type=Path, required=True)
    parser.add_argument("--remote-host", default="${REMOTE_USER}@private-3f2a50db.invalid")
    parser.add_argument(
        "--remote-root",
        default="${PELCB100_REMOTE_ROOT}/lz_project_runs/emc_alpha_olefin_builder",
    )
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--array-name", help="Remote/local bundle name. Defaults to emc_array_<date-tag>.")
    parser.add_argument("--partition", default="amd")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--array-concurrency", type=int, default=10)
    parser.add_argument("--remote-emc-root", default="${PELCB100_REMOTE_ROOT}/tools/emc")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=14400)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--overwrite-canonical", action="store_true")
    parser.add_argument("--submit-only", action="store_true", help="Upload and submit only; skip wait/download.")
    return parser.parse_args()


def run_command(args: list[str], *, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=capture_output, check=False)


def run_ssh(remote_host: str, command: str) -> str:
    result = run_command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", remote_host, command])
    if result.returncode != 0:
        raise RuntimeError(f"SSH failed: {command}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result.stdout


def run_scp(args: list[str]) -> None:
    result = run_command(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", *args])
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def validate_remote_dir(remote_root: str, array_name: str) -> PurePosixPath:
    remote_root_path = PurePosixPath(remote_root.replace("\\", "/"))
    if not remote_root_path.is_absolute():
        raise ValueError(f"Remote root must be absolute: {remote_root}")
    try:
        remote_root_path.relative_to(SAFE_REMOTE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Remote root must stay under {SAFE_REMOTE_ROOT}: {remote_root_path}") from exc
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", array_name):
        raise ValueError(f"Unsafe array name: {array_name!r}")
    return remote_root_path / array_name


def load_package_record(package_dir: Path, task_index: int) -> PackageRecord:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "emc_package_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing package manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PackageRecord(
        task_index=task_index,
        case_id=manifest["case_id"],
        package_dir=package_dir,
        package_name=package_dir.name,
        run_label=manifest["run_label"],
        esh_name=manifest["esh_name"],
        canonical_remote_build_dir=Path(manifest["canonical_remote_build_dir"]).resolve(),
    )


def write_text_unix(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.write_text(content, encoding=encoding, newline="\n")


def prepare_local_bundle(
    records: list[PackageRecord],
    *,
    bundle_dir: Path,
    partition: str,
    cpus_per_task: int,
    time_limit: str,
    array_concurrency: int,
    remote_emc_root: str,
) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "packages").mkdir(parents=True)
    (bundle_dir / "logs").mkdir()

    lines = []
    for record in records:
        destination = bundle_dir / "packages" / record.case_id / record.package_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(record.package_dir, destination)
        lines.append(
            "\t".join(
                [
                    str(record.task_index),
                    record.case_id,
                    record.package_name,
                    record.run_label,
                    record.esh_name,
                ]
            )
        )
    write_text_unix(bundle_dir / "case_list.tsv", "\n".join(lines) + "\n", encoding="utf-8")

    max_task = len(records) - 1
    array_script = f"""#!/bin/bash
#SBATCH --job-name=emc-array-{records[0].run_label}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time_limit}
#SBATCH --array=0-{max_task}%{array_concurrency}
#SBATCH --output=logs/%A_%a.out

set -euo pipefail

ROOT="${{SLURM_SUBMIT_DIR:-$PWD}}"
TASK_ID="${{SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}}"
LINE="$(awk -F '\\t' -v idx="$TASK_ID" '$1 == idx {{ print; exit }}' "$ROOT/case_list.tsv")"
if [ -z "$LINE" ]; then
  echo "No case record for task $TASK_ID" >&2
  exit 2
fi

IFS=$'\\t' read -r _TASK_INDEX CASE_ID PACKAGE_NAME RUN_LABEL ESH_NAME <<< "$LINE"
PACKAGE_DIR="$ROOT/packages/$CASE_ID/$PACKAGE_NAME"
RUN_DIR="$PACKAGE_DIR/$RUN_LABEL"

module purge >/dev/null 2>&1 || true
export EMC_ROOT="{remote_emc_root}"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

echo "EMC_ARRAY_TASK_START=$(date -Is)"
echo "CASE_ID=$CASE_ID"
echo "HOSTNAME=$(hostname)"
echo "PACKAGE_DIR=$PACKAGE_DIR"
echo "RUN_LABEL=$RUN_LABEL"
echo "EMC_ROOT=$EMC_ROOT"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

cd "$PACKAGE_DIR"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp "$ESH_NAME" "$RUN_DIR/$ESH_NAME"
cd "$RUN_DIR"

set +e
perl "$EMC_ROOT/scripts/emc.pl" "$ESH_NAME" > setup.log 2>&1
setup_status=$?
set -e
printf '%s\\n' "$setup_status" > setup.exitcode
if [ "$setup_status" -ne 0 ]; then
  echo "setup failed: $setup_status" >&2
  exit "$setup_status"
fi

rm -f "$ESH_NAME"

set +e
"$EMC_ROOT/bin/emc_linux_x86_64" build.emc > build.log 2>&1
build_status=$?
set -e
printf '%s\\n' "$build_status" > build.exitcode
echo "EMC_ARRAY_TASK_END=$(date -Is)"
exit "$build_status"
"""
    write_text_unix(bundle_dir / "array.sbatch", array_script, encoding="ascii")

    bundle_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_label": records[0].run_label,
        "total_packages": len(records),
        "partition": partition,
        "cpus_per_task": cpus_per_task,
        "time_limit": time_limit,
        "array_concurrency": array_concurrency,
        "remote_emc_root": remote_emc_root,
        "cases": [asdict(record) | {"package_dir": str(record.package_dir), "canonical_remote_build_dir": str(record.canonical_remote_build_dir)} for record in records],
    }
    write_text_unix(
        bundle_dir / "array_manifest.json",
        json.dumps(bundle_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def upload_bundle(bundle_dir: Path, remote_host: str, remote_dir: PurePosixPath) -> None:
    parent = remote_dir.parent.as_posix()
    remote = remote_dir.as_posix()
    run_ssh(remote_host, f"mkdir -p '{parent}' && rm -rf '{remote}'")
    run_scp(["-r", str(bundle_dir), f"{remote_host}:{parent}"])


def submit_array(remote_host: str, remote_dir: PurePosixPath) -> str:
    output = run_ssh(remote_host, f"cd '{remote_dir.as_posix()}' && sbatch --parsable array.sbatch")
    job_id = output.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse Slurm array job id from:\n{output}")
    return job_id


def wait_for_array(remote_host: str, job_id: str, poll_seconds: int, max_wait_seconds: int) -> None:
    start = time.monotonic()
    while True:
        output = run_ssh(remote_host, f"squeue -h -j {job_id} -o '%i|%T|%R' 2>/dev/null || true").strip()
        if not output:
            return
        lines = output.splitlines()
        print(f"[array {job_id}] active={len(lines)} sample={lines[0]}")
        if max_wait_seconds > 0 and (time.monotonic() - start) > max_wait_seconds:
            raise TimeoutError(f"Timed out waiting for Slurm array {job_id} after {max_wait_seconds} seconds")
        time.sleep(poll_seconds)


def read_exitcode(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def validate_canonical(case_id: str, canonical_dir: Path) -> tuple[bool, dict[str, str | bool | None]]:
    setup_exitcode = read_exitcode(canonical_dir / "setup.exitcode")
    build_exitcode = read_exitcode(canonical_dir / "build.exitcode")
    has_data = (canonical_dir / f"{case_id}.data").exists()
    has_params = (canonical_dir / f"{case_id}.params").exists()
    has_vmd = (canonical_dir / f"{case_id}.vmd").exists()
    has_pdb_gz = (canonical_dir / f"{case_id}.pdb.gz").exists()
    has_psf_gz = (canonical_dir / f"{case_id}.psf.gz").exists()
    ok = setup_exitcode == "0" and build_exitcode == "0" and has_data and has_params and has_vmd
    return ok, {
        "setup_exitcode": setup_exitcode,
        "build_exitcode": build_exitcode,
        "has_data": has_data,
        "has_params": has_params,
        "has_vmd": has_vmd,
        "has_pdb_gz": has_pdb_gz,
        "has_psf_gz": has_psf_gz,
    }


def sync_record(
    record: PackageRecord,
    *,
    remote_host: str,
    remote_dir: PurePosixPath,
    job_id: str,
    overwrite_canonical: bool,
) -> CaseArrayResult:
    artifact_dir = record.package_dir / f"artifacts_{job_id}_{record.task_index}"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    remote_run_dir = (
        remote_dir
        / "packages"
        / record.case_id
        / record.package_name
        / record.run_label
    ).as_posix()

    try:
        run_scp(["-r", f"{remote_host}:{remote_run_dir}", str(artifact_dir)])
        source_dir = artifact_dir / record.run_label
        if record.canonical_remote_build_dir.exists():
            if not overwrite_canonical:
                ok, details = validate_canonical(record.case_id, record.canonical_remote_build_dir)
                return CaseArrayResult(
                    task_index=record.task_index,
                    case_id=record.case_id,
                    package_dir=str(record.package_dir),
                    artifact_dir=str(artifact_dir),
                    canonical_dir=str(record.canonical_remote_build_dir),
                    setup_exitcode=details["setup_exitcode"],
                    build_exitcode=details["build_exitcode"],
                    has_data=bool(details["has_data"]),
                    has_params=bool(details["has_params"]),
                    has_vmd=bool(details["has_vmd"]),
                    has_pdb_gz=bool(details["has_pdb_gz"]),
                    has_psf_gz=bool(details["has_psf_gz"]),
                    ok=ok,
                    error=None if ok else "canonical exists but is incomplete; rerun with --overwrite-canonical",
                )
            shutil.rmtree(record.canonical_remote_build_dir)
        record.canonical_remote_build_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, record.canonical_remote_build_dir)
        (artifact_dir / "remote_job.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "task_index": record.task_index,
                    "case_id": record.case_id,
                    "remote_dir": remote_run_dir,
                    "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        ok, details = validate_canonical(record.case_id, record.canonical_remote_build_dir)
        return CaseArrayResult(
            task_index=record.task_index,
            case_id=record.case_id,
            package_dir=str(record.package_dir),
            artifact_dir=str(artifact_dir),
            canonical_dir=str(record.canonical_remote_build_dir),
            setup_exitcode=details["setup_exitcode"],
            build_exitcode=details["build_exitcode"],
            has_data=bool(details["has_data"]),
            has_params=bool(details["has_params"]),
            has_vmd=bool(details["has_vmd"]),
            has_pdb_gz=bool(details["has_pdb_gz"]),
            has_psf_gz=bool(details["has_psf_gz"]),
            ok=ok,
            error=None if ok else "canonical validation failed",
        )
    except Exception as exc:
        return CaseArrayResult(
            task_index=record.task_index,
            case_id=record.case_id,
            package_dir=str(record.package_dir),
            artifact_dir=str(artifact_dir),
            canonical_dir=str(record.canonical_remote_build_dir),
            setup_exitcode=None,
            build_exitcode=None,
            has_data=False,
            has_params=False,
            has_vmd=False,
            has_pdb_gz=False,
            has_psf_gz=False,
            ok=False,
            error=str(exc),
        )


def main() -> int:
    args = parse_args()
    records = [load_package_record(path, index) for index, path in enumerate(args.package_dir)]
    if not records:
        raise RuntimeError("No packages provided")
    run_labels = {record.run_label for record in records}
    if len(run_labels) != 1:
        raise ValueError(f"All packages in one array must use the same run_label, got {sorted(run_labels)}")
    if args.array_concurrency < 1:
        raise ValueError("--array-concurrency must be >= 1")

    array_name = args.array_name or f"emc_array_{args.date_tag}"
    remote_dir = validate_remote_dir(args.remote_root, array_name)
    log_root = (args.log_root or (SCRIPT_DIR / "output" / f"emc_array_submit_{args.date_tag}")).resolve()
    bundle_dir = log_root / array_name
    log_root.mkdir(parents=True, exist_ok=True)

    prepare_local_bundle(
        records,
        bundle_dir=bundle_dir,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        time_limit=args.time_limit,
        array_concurrency=args.array_concurrency,
        remote_emc_root=args.remote_emc_root,
    )
    print(f"[bundle] {bundle_dir}")
    print(f"[upload] {bundle_dir} -> {args.remote_host}:{remote_dir}")
    upload_bundle(bundle_dir, args.remote_host, remote_dir)
    job_id = submit_array(args.remote_host, remote_dir)
    print(f"[submit] array_job_id={job_id}")

    summary_path = log_root / "array_submit_summary.json"
    if args.submit_only:
        payload = {
            "array_job_id": job_id,
            "remote_host": args.remote_host,
            "remote_dir": remote_dir.as_posix(),
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "total_cases": len(records),
            "cases": [asdict(record) | {"package_dir": str(record.package_dir), "canonical_remote_build_dir": str(record.canonical_remote_build_dir)} for record in records],
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[summary] {summary_path}")
        return 0

    wait_for_array(args.remote_host, job_id, args.poll_seconds, args.max_wait_seconds)
    print(f"[done] array_job_id={job_id}; downloading artifacts")

    results = [
        sync_record(
            record,
            remote_host=args.remote_host,
            remote_dir=remote_dir,
            job_id=job_id,
            overwrite_canonical=args.overwrite_canonical,
        )
        for record in records
    ]
    ok_count = sum(1 for result in results if result.ok)
    payload = {
        "array_job_id": job_id,
        "remote_host": args.remote_host,
        "remote_dir": remote_dir.as_posix(),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "total_cases": len(records),
        "ok_cases": ok_count,
        "failed_cases": len(records) - ok_count,
        "results": [asdict(result) for result in results],
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[summary] ok={ok_count}/{len(records)} path={summary_path}")
    return 0 if ok_count == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
