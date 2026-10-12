from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath


TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
}

SAFE_REMOTE_ROOT = PurePosixPath("${PELCB100_REMOTE_ROOT}/lz_project_runs")
MIN_REMOTE_EXTRA_DEPTH = 2
ARTIFACT_DIR_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SBATCH_PARTITION_RE = re.compile(r"^#SBATCH\s+--partition(?:=|\s+)(?P<value>\S+)\s*$")
SBATCH_CPUS_RE = re.compile(r"^#SBATCH\s+--cpus-per-task(?:=|\s+)(?P<value>\d+)\s*$")
SBATCH_TIME_RE = re.compile(r"^#SBATCH\s+--time(?:=|\s+)(?P<value>\S+)\s*$")
DISALLOWED_NODE_STATES = ("down", "drain", "drng", "fail", "maint", "resv", "unk")


@dataclass
class SlurmNodeAvailability:
    node: str
    state: str
    total_cpus: int
    allocated_cpus: int
    idle_cpus: int
    other_cpus: int


@dataclass
class SlurmAdaptivePlan:
    partition: str
    requested_cpus_per_task: int
    requested_time_limit: str
    selected_cpus_per_task: int
    selected_time_limit: str
    adaptive_applied: bool
    total_idle_cpus: int
    largest_idle_cpus: int
    largest_idle_node: str
    min_adaptive_cpus: int
    max_adaptive_cpus: int | None
    nodes: list[SlurmNodeAvailability]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local Slurm package to the remote server, submit it, wait for completion, and download artifacts."
    )
    parser.add_argument("--local-dir", type=Path, required=True, help="Local package directory.")
    parser.add_argument("--remote-dir", required=True, help="Exact remote package directory path.")
    parser.add_argument("--submit-script", required=True, help="Submit script filename inside the package.")
    parser.add_argument("--remote-host", default="${REMOTE_USER}@private-3f2a50db.invalid")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--artifact-dir-name", help="Optional local artifact directory name. Defaults to artifacts_<jobid>.")
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=7200,
        help="Maximum wall-clock wait time for a Slurm job before aborting.",
    )
    parser.add_argument(
        "--max-unknown-polls",
        type=int,
        default=10,
        help="Abort if job state stays UNKNOWN for this many consecutive polls.",
    )
    parser.add_argument(
        "--adaptive-slurm-cpus",
        action="store_true",
        help=(
            "Probe the current Slurm partition state from the package submit script "
            "and adapt --cpus-per-task toward the largest currently idle single-node slot."
        ),
    )
    parser.add_argument(
        "--min-adaptive-cpus",
        type=int,
        default=4,
        help="Do not adapt to fewer than this many CPUs per task. Default: 4.",
    )
    parser.add_argument(
        "--max-adaptive-cpus",
        type=int,
        help="Optional upper bound for adaptive cpus-per-task. Default: no cap.",
    )
    parser.add_argument(
        "--baseline-runtime-hours-at-8cpu",
        type=float,
        default=1.9,
        help="Observed runtime at 8 CPU threads, used to derive adaptive walltime. Default: 1.9 h.",
    )
    parser.add_argument(
        "--time-buffer-hours",
        type=float,
        default=1.0,
        help="Walltime buffer added after simple 8cpu runtime scaling. Default: 1.0 h.",
    )
    parser.add_argument(
        "--min-time-limit-hours",
        type=float,
        default=3.0,
        help="Lower bound for the derived Slurm time limit. Default: 3.0 h.",
    )
    parser.add_argument(
        "--max-time-limit-hours",
        type=float,
        default=6.0,
        help="Upper bound for the derived Slurm time limit. Default: 6.0 h.",
    )
    parser.add_argument(
        "--time-round-minutes",
        type=int,
        default=30,
        help="Round derived walltime up to this many minutes. Default: 30.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only probe and print the Slurm resource plan; do not upload or submit anything.",
    )
    return parser.parse_args()


def run_command(args: list[str], *, cwd: Path | None = None, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=capture_output,
        check=False,
    )


def run_ssh(remote_host: str, command: str) -> str:
    result = run_command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", remote_host, command])
    if result.returncode != 0:
        raise RuntimeError(f"SSH failed: {command}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result.stdout


def run_scp(args: list[str]) -> None:
    result = run_command(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *args])
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def validate_remote_dir(local_dir: Path, remote_dir: str) -> PurePosixPath:
    remote_path = PurePosixPath(remote_dir.replace("\\", "/"))
    if not remote_path.is_absolute():
        raise ValueError(f"Remote dir must be an absolute POSIX path: {remote_dir}")
    if local_dir.name != remote_path.name:
        raise ValueError(f"Local package dir name '{local_dir.name}' must match remote package basename '{remote_path.name}'.")
    try:
        relative = remote_path.relative_to(SAFE_REMOTE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Remote dir must stay under {SAFE_REMOTE_ROOT.as_posix()}: {remote_path.as_posix()}"
        ) from exc
    if len(relative.parts) < MIN_REMOTE_EXTRA_DEPTH:
        raise ValueError(
            f"Remote dir must be at least {MIN_REMOTE_EXTRA_DEPTH} levels below {SAFE_REMOTE_ROOT.as_posix()}: "
            f"{remote_path.as_posix()}"
        )
    return remote_path


def read_submit_script_lines(submit_script_path: Path) -> list[str]:
    if not submit_script_path.exists():
        raise FileNotFoundError(f"Submit script not found: {submit_script_path}")
    return submit_script_path.read_text(encoding="utf-8").splitlines()


def detect_submit_script_partition(submit_script_path: Path) -> str:
    for line in read_submit_script_lines(submit_script_path):
        match = SBATCH_PARTITION_RE.match(line.strip())
        if match:
            return match.group("value")
    raise ValueError(f"Could not detect #SBATCH --partition from {submit_script_path}")


def detect_submit_script_cpus_per_task(submit_script_path: Path) -> int:
    for line in read_submit_script_lines(submit_script_path):
        match = SBATCH_CPUS_RE.match(line.strip())
        if match:
            return int(match.group("value"))
    raise ValueError(f"Could not detect #SBATCH --cpus-per-task from {submit_script_path}")


def detect_submit_script_time_limit(submit_script_path: Path) -> str:
    for line in read_submit_script_lines(submit_script_path):
        match = SBATCH_TIME_RE.match(line.strip())
        if match:
            return match.group("value")
    raise ValueError(f"Could not detect #SBATCH --time from {submit_script_path}")


def ceil_time_limit(hours: float, round_minutes: int) -> str:
    total_minutes = max(1, math.ceil(hours * 60 / round_minutes) * round_minutes)
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}:00"


def derive_time_limit(
    cpus_per_task: int,
    *,
    baseline_runtime_hours_at_8cpu: float,
    time_buffer_hours: float,
    min_time_limit_hours: float,
    max_time_limit_hours: float | None,
    round_minutes: int,
) -> str:
    predicted_hours = baseline_runtime_hours_at_8cpu * (8.0 / float(cpus_per_task))
    limit_hours = max(predicted_hours + time_buffer_hours, min_time_limit_hours)
    if max_time_limit_hours is not None:
        limit_hours = min(limit_hours, max_time_limit_hours)
    return ceil_time_limit(limit_hours, round_minutes)


def probe_slurm_partition_nodes(remote_host: str, partition: str) -> list[SlurmNodeAvailability]:
    output = run_ssh(remote_host, f"sinfo -p {partition} -N -h -o '%N|%t|%c|%C'")
    nodes: list[SlurmNodeAvailability] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 4:
            continue
        node, state, total_cpus_raw, cpu_tuple = parts
        try:
            allocated_raw, idle_raw, other_raw, total_raw = cpu_tuple.split("/")
            nodes.append(
                SlurmNodeAvailability(
                    node=node,
                    state=state.lower(),
                    total_cpus=int(total_cpus_raw),
                    allocated_cpus=int(allocated_raw),
                    idle_cpus=int(idle_raw),
                    other_cpus=int(other_raw),
                )
            )
        except ValueError:
            continue
    if not nodes:
        raise RuntimeError(f"No node-level sinfo data returned for partition {partition!r}")
    return nodes


def make_slurm_adaptive_plan(
    remote_host: str,
    submit_script_path: Path,
    *,
    min_adaptive_cpus: int,
    max_adaptive_cpus: int | None,
    baseline_runtime_hours_at_8cpu: float,
    time_buffer_hours: float,
    min_time_limit_hours: float,
    max_time_limit_hours: float | None,
    time_round_minutes: int,
) -> SlurmAdaptivePlan:
    partition = detect_submit_script_partition(submit_script_path)
    requested_cpus = detect_submit_script_cpus_per_task(submit_script_path)
    requested_time_limit = detect_submit_script_time_limit(submit_script_path)
    nodes = probe_slurm_partition_nodes(remote_host, partition)
    eligible_nodes = [
        node
        for node in nodes
        if node.idle_cpus > 0 and not any(token in node.state for token in DISALLOWED_NODE_STATES)
    ]
    total_idle_cpus = sum(node.idle_cpus for node in eligible_nodes)
    if eligible_nodes:
        best_node = max(eligible_nodes, key=lambda item: (item.idle_cpus, item.total_cpus, item.node))
        selected_cpus = best_node.idle_cpus
        if max_adaptive_cpus is not None:
            selected_cpus = min(selected_cpus, max_adaptive_cpus)
        if selected_cpus < min_adaptive_cpus:
            selected_cpus = requested_cpus
            adaptive_applied = False
        else:
            adaptive_applied = selected_cpus != requested_cpus
    else:
        best_node = SlurmNodeAvailability(
            node="none",
            state="unavailable",
            total_cpus=0,
            allocated_cpus=0,
            idle_cpus=0,
            other_cpus=0,
        )
        selected_cpus = requested_cpus
        adaptive_applied = False

    return SlurmAdaptivePlan(
        partition=partition,
        requested_cpus_per_task=requested_cpus,
        requested_time_limit=requested_time_limit,
        selected_cpus_per_task=selected_cpus,
        selected_time_limit=derive_time_limit(
            selected_cpus,
            baseline_runtime_hours_at_8cpu=baseline_runtime_hours_at_8cpu,
            time_buffer_hours=time_buffer_hours,
            min_time_limit_hours=min_time_limit_hours,
            max_time_limit_hours=max_time_limit_hours,
            round_minutes=time_round_minutes,
        ),
        adaptive_applied=adaptive_applied,
        total_idle_cpus=total_idle_cpus,
        largest_idle_cpus=best_node.idle_cpus,
        largest_idle_node=best_node.node,
        min_adaptive_cpus=min_adaptive_cpus,
        max_adaptive_cpus=max_adaptive_cpus,
        nodes=eligible_nodes,
    )


def rewrite_submit_resources(submit_script_path: Path, *, new_cpus: int, new_time_limit: str) -> None:
    lines = read_submit_script_lines(submit_script_path)
    updated_cpus = False
    updated_time = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if SBATCH_CPUS_RE.match(stripped):
            new_lines.append(f"#SBATCH --cpus-per-task={new_cpus}")
            updated_cpus = True
        elif SBATCH_TIME_RE.match(stripped):
            new_lines.append(f"#SBATCH --time={new_time_limit}")
            updated_time = True
        else:
            new_lines.append(line)
    if not updated_cpus or not updated_time:
        raise ValueError(f"Could not rewrite cpus/time in {submit_script_path}")
    submit_script_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8", newline="\n")


def write_resource_plan(local_dir: Path, plan: SlurmAdaptivePlan) -> None:
    payload = asdict(plan)
    (local_dir / "slurm_resource_plan.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_remote_package(local_dir: Path, remote_host: str, remote_dir: str) -> None:
    remote_path = validate_remote_dir(local_dir, remote_dir)
    parent = remote_path.parent.as_posix()
    remote_dir_posix = remote_path.as_posix()
    run_ssh(remote_host, f"mkdir -p '{parent}' && rm -rf '{remote_dir_posix}'")
    run_scp(["-r", str(local_dir), f"{remote_host}:{parent}"])


def submit_job(remote_host: str, remote_dir: str, submit_script: str) -> str:
    output = run_ssh(remote_host, f"cd '{remote_dir}' && sbatch '{submit_script}'")
    for token in output.strip().split():
        if token.isdigit():
            return token
    raise RuntimeError(f"Could not parse job id from sbatch output:\n{output}")


def query_job_state(remote_host: str, job_id: str) -> str:
    sacct_output = run_ssh(
        remote_host,
        f"sacct -j {job_id} --format=State --noheader -X -P 2>/dev/null || true",
    )
    states = [line.strip().split("|")[0] for line in sacct_output.splitlines() if line.strip()]
    for state in states:
        root = state.split()[0]
        if root:
            return root
    squeue_output = run_ssh(remote_host, f"squeue -h -j {job_id} -o %T 2>/dev/null || true").strip()
    if squeue_output:
        return squeue_output.splitlines()[0].strip()
    return "UNKNOWN"


def wait_for_job(
    remote_host: str,
    job_id: str,
    poll_seconds: int,
    max_wait_seconds: int,
    max_unknown_polls: int,
) -> str:
    last_state = ""
    start = time.monotonic()
    unknown_polls = 0
    while True:
        state = query_job_state(remote_host, job_id)
        if state != last_state:
            print(f"[job {job_id}] state={state}")
            last_state = state
        if state == "UNKNOWN":
            unknown_polls += 1
            if unknown_polls >= max_unknown_polls:
                raise TimeoutError(
                    f"Job {job_id} stayed UNKNOWN for {unknown_polls} consecutive polls; aborting."
                )
        else:
            unknown_polls = 0
        if state in TERMINAL_STATES:
            return state
        if max_wait_seconds > 0 and (time.monotonic() - start) > max_wait_seconds:
            raise TimeoutError(f"Timed out waiting for job {job_id} after {max_wait_seconds} seconds.")
        time.sleep(poll_seconds)


def validate_artifact_dir_name(name: str) -> str:
    if not name or not ARTIFACT_DIR_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Artifact dir name must match {ARTIFACT_DIR_NAME_RE.pattern!r}; got {name!r}"
        )
    return name


def download_artifacts(local_dir: Path, remote_host: str, remote_dir: str, artifact_dir_name: str) -> Path:
    artifact_dir = local_dir / artifact_dir_name
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_scp(["-r", f"{remote_host}:{remote_dir}/.", str(artifact_dir)])
    return artifact_dir


def write_metadata(artifact_dir: Path, payload: dict[str, str]) -> None:
    (artifact_dir / "remote_job.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    local_dir = args.local_dir.resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local package directory not found: {local_dir}")
    submit_script_path = local_dir / args.submit_script
    if not submit_script_path.exists():
        raise FileNotFoundError(f"Submit script not found in package: {submit_script_path}")
    remote_dir = validate_remote_dir(local_dir, args.remote_dir).as_posix()
    adaptive_plan: SlurmAdaptivePlan | None = None

    if args.adaptive_slurm_cpus or args.probe_only:
        adaptive_plan = make_slurm_adaptive_plan(
            args.remote_host,
            submit_script_path,
            min_adaptive_cpus=args.min_adaptive_cpus,
            max_adaptive_cpus=args.max_adaptive_cpus,
            baseline_runtime_hours_at_8cpu=args.baseline_runtime_hours_at_8cpu,
            time_buffer_hours=args.time_buffer_hours,
            min_time_limit_hours=args.min_time_limit_hours,
            max_time_limit_hours=args.max_time_limit_hours,
            time_round_minutes=args.time_round_minutes,
        )
        print(
            "[probe] "
            f"partition={adaptive_plan.partition} requested_cpus={adaptive_plan.requested_cpus_per_task} "
            f"requested_time={adaptive_plan.requested_time_limit} selected_cpus={adaptive_plan.selected_cpus_per_task} "
            f"selected_time={adaptive_plan.selected_time_limit} total_idle={adaptive_plan.total_idle_cpus} "
            f"largest_idle={adaptive_plan.largest_idle_cpus} on {adaptive_plan.largest_idle_node}"
        )
        needs_rewrite = (
            adaptive_plan.selected_cpus_per_task != adaptive_plan.requested_cpus_per_task
            or adaptive_plan.selected_time_limit != adaptive_plan.requested_time_limit
        )
        if args.adaptive_slurm_cpus and needs_rewrite:
            if args.probe_only:
                print(
                    f"[probe] would update {submit_script_path.name} "
                    f"--cpus-per-task {adaptive_plan.requested_cpus_per_task} -> {adaptive_plan.selected_cpus_per_task}, "
                    f"--time {adaptive_plan.requested_time_limit} -> {adaptive_plan.selected_time_limit}"
                )
            else:
                rewrite_submit_resources(
                    submit_script_path,
                    new_cpus=adaptive_plan.selected_cpus_per_task,
                    new_time_limit=adaptive_plan.selected_time_limit,
                )
                print(
                    f"[adapt] updated {submit_script_path.name} "
                    f"--cpus-per-task {adaptive_plan.requested_cpus_per_task} -> {adaptive_plan.selected_cpus_per_task}, "
                    f"--time {adaptive_plan.requested_time_limit} -> {adaptive_plan.selected_time_limit}"
                )
        write_resource_plan(local_dir, adaptive_plan)
        if args.probe_only:
            print(f"[probe] wrote {local_dir / 'slurm_resource_plan.json'}")
            return 0

    print(f"[upload] {local_dir} -> {args.remote_host}:{remote_dir}")
    ensure_remote_package(local_dir, args.remote_host, remote_dir)

    print(f"[submit] {args.submit_script}")
    job_id = submit_job(args.remote_host, remote_dir, args.submit_script)
    print(f"[submit] job_id={job_id}")

    terminal_state = wait_for_job(
        args.remote_host,
        job_id,
        args.poll_seconds,
        args.max_wait_seconds,
        args.max_unknown_polls,
    )
    print(f"[done] job_id={job_id} terminal_state={terminal_state}")

    artifact_dir_name = validate_artifact_dir_name(args.artifact_dir_name or f"artifacts_{job_id}")
    artifact_dir = download_artifacts(local_dir, args.remote_host, remote_dir, artifact_dir_name)
    write_metadata(
        artifact_dir,
        {
            "job_id": job_id,
            "terminal_state": terminal_state,
            "remote_host": args.remote_host,
            "remote_dir": remote_dir,
            "submit_script": args.submit_script,
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            **(
                {
                    "slurm_partition": adaptive_plan.partition,
                    "requested_cpus_per_task": str(adaptive_plan.requested_cpus_per_task),
                    "requested_time_limit": adaptive_plan.requested_time_limit,
                    "selected_cpus_per_task": str(adaptive_plan.selected_cpus_per_task),
                    "selected_time_limit": adaptive_plan.selected_time_limit,
                    "total_idle_cpus_at_submit": str(adaptive_plan.total_idle_cpus),
                    "largest_idle_node_at_submit": adaptive_plan.largest_idle_node,
                    "largest_idle_cpus_at_submit": str(adaptive_plan.largest_idle_cpus),
                }
                if adaptive_plan is not None
                else {}
            ),
        },
    )
    print(f"[download] {artifact_dir}")

    if terminal_state != "COMPLETED":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
