from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from submit_remote_slurm_package import (
    detect_submit_script_partition,
    probe_slurm_partition_nodes,
    query_job_state,
    run_scp,
    run_ssh,
    validate_remote_dir,
)


SBATCH_CPUS_RE = re.compile(r"^#SBATCH\s+--cpus-per-task(?:=|\s+)(?P<value>\d+)\s*$")
SBATCH_TIME_RE = re.compile(r"^#SBATCH\s+--time(?:=|\s+)(?P<value>\S+)\s*$")
CASE_ID_RE = re.compile(r"^sc\d+_.+")


@dataclass
class BatchJobPlan:
    local_dir: str
    case_label: str
    package_name: str
    submit_script: str
    remote_dir: str
    requested_cpus_per_task: int
    requested_time_limit: str
    selected_cpus_per_task: int
    selected_time_limit: str
    planned_idle_node: str
    planned_idle_cpus: int
    submitted_job_id: str | None = None
    submitted_state: str | None = None


@dataclass
class DynamicBatchPlan:
    remote_host: str
    partition: str
    generated_at: str
    submit_mode: bool
    allow_overwrite_remote: bool
    planned_idle_node_semantics: str
    queue_probe_filter: str
    min_cpus_per_task: int
    max_cpus_per_task: int | None
    baseline_runtime_hours_at_8cpu: float
    time_buffer_hours: float
    min_time_limit_hours: float
    max_time_limit_hours: float | None
    jobs: list[BatchJobPlan]
    queue_snapshot: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or submit multiple packaged Slurm jobs with a dynamic "
            "single-node idle-slot aware resource policy."
        )
    )
    parser.add_argument(
        "--package-dir",
        action="append",
        type=Path,
        required=True,
        help="Local packaged Slurm directory. May be repeated.",
    )
    parser.add_argument(
        "--submit-script-name",
        help="Optional explicit submit script name inside every package directory. Defaults to auto-detecting a single *.sbatch.",
    )
    parser.add_argument(
        "--case-label",
        action="append",
        default=[],
        help=(
            "Optional per-package case/run label override, in the same order as "
            "--package-dir. Defaults to the parent directory name when possible."
        ),
    )
    parser.add_argument(
        "--remote-host",
        default="${REMOTE_USER}@private-3f2a50db.invalid",
        help="Remote SSH host.",
    )
    parser.add_argument(
        "--remote-root",
        default="${PELCB100_REMOTE_ROOT}/lz_project_runs/emc_gromacs_relax",
        help="Remote run root used for auto-generated remote directories.",
    )
    parser.add_argument(
        "--date-tag",
        default=datetime.now().strftime("%Y%m%d"),
        help="Date tag used in auto-generated remote directories.",
    )
    parser.add_argument(
        "--min-cpus-per-task",
        type=int,
        default=4,
        help="Minimum cpus-per-task to request for each submitted job.",
    )
    parser.add_argument(
        "--max-cpus-per-task",
        type=int,
        help="Optional upper bound for cpus-per-task.",
    )
    parser.add_argument(
        "--baseline-runtime-hours-at-8cpu",
        type=float,
        default=1.9,
        help="Observed runtime at 8 CPU threads, used to derive dynamic walltime.",
    )
    parser.add_argument(
        "--time-buffer-hours",
        type=float,
        default=1.0,
        help="Walltime buffer added after simple 8cpu runtime scaling.",
    )
    parser.add_argument(
        "--min-time-limit-hours",
        type=float,
        default=3.0,
        help="Lower bound for the derived Slurm time limit.",
    )
    parser.add_argument(
        "--max-time-limit-hours",
        type=float,
        default=6.0,
        help="Upper bound for the derived Slurm time limit.",
    )
    parser.add_argument(
        "--time-round-minutes",
        type=int,
        default=30,
        help="Round derived walltime up to this many minutes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that receives the generated plan summary files.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually upload and submit the packages. Without this flag, only write the plan.",
    )
    parser.add_argument(
        "--allow-overwrite-remote",
        action="store_true",
        help="Allow replacing an existing remote package directory during submit.",
    )
    parser.add_argument(
        "--cancel-jobid",
        action="append",
        default=[],
        help="Optional job id(s) to cancel before submitting.",
    )
    return parser.parse_args()


def read_submit_script_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Submit script not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def detect_submit_script_name(package_dir: Path, explicit_name: str | None) -> str:
    if explicit_name:
        path = package_dir / explicit_name
        if not path.exists():
            raise FileNotFoundError(f"Submit script not found: {path}")
        return explicit_name
    matches = sorted(path.name for path in package_dir.glob("*.sbatch"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one *.sbatch in {package_dir}, found {len(matches)}: {matches}"
        )
    return matches[0]


def read_submit_resources(path: Path) -> tuple[int, str]:
    cpus: int | None = None
    time_limit: str | None = None
    for line in read_submit_script_lines(path):
        stripped = line.strip()
        match_cpus = SBATCH_CPUS_RE.match(stripped)
        if match_cpus:
            cpus = int(match_cpus.group("value"))
        match_time = SBATCH_TIME_RE.match(stripped)
        if match_time:
            time_limit = match_time.group("value")
    if cpus is None or time_limit is None:
        raise ValueError(f"Could not parse cpus/time from {path}")
    return cpus, time_limit


def rewrite_submit_resources(path: Path, *, cpus_per_task: int, time_limit: str) -> None:
    updated_cpus = False
    updated_time = False
    new_lines: list[str] = []
    for line in read_submit_script_lines(path):
        stripped = line.strip()
        if SBATCH_CPUS_RE.match(stripped):
            new_lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
            updated_cpus = True
        elif SBATCH_TIME_RE.match(stripped):
            new_lines.append(f"#SBATCH --time={time_limit}")
            updated_time = True
        else:
            new_lines.append(line)
    if not updated_cpus or not updated_time:
        raise ValueError(f"Could not rewrite cpus/time in {path}")
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8", newline="\n")


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


def infer_case_label(package_dir: Path, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    parent_name = package_dir.parent.name
    if CASE_ID_RE.fullmatch(parent_name):
        return parent_name
    return package_dir.name


def remote_dir_for_package(remote_root: str, case_label: str, package_name: str, date_tag: str) -> str:
    run_label = package_name.removeprefix("gromacs_")
    return f"{remote_root}/{case_label}_{run_label}_{date_tag}/{package_name}"


def remote_dir_exists(remote_host: str, remote_dir: str) -> bool:
    output = run_ssh(remote_host, f"if [ -d '{remote_dir}' ]; then echo exists; else echo missing; fi")
    return output.strip() == "exists"


def ensure_remote_package(local_dir: Path, remote_host: str, remote_dir: str, *, allow_overwrite: bool) -> None:
    remote_path = validate_remote_dir(local_dir, remote_dir)
    remote_dir_posix = remote_path.as_posix()
    if remote_dir_exists(remote_host, remote_dir_posix):
        if not allow_overwrite:
            raise FileExistsError(
                f"Remote package directory already exists: {remote_dir_posix}. "
                "Re-run with --allow-overwrite-remote only if you intentionally want to replace it."
            )
        run_ssh(remote_host, f"rm -rf '{remote_dir_posix}'")
    parent = remote_path.parent.as_posix()
    run_ssh(remote_host, f"mkdir -p '{parent}'")
    run_scp(["-r", str(local_dir), f"{remote_host}:{parent}"])


def submit_job(remote_host: str, remote_dir: str, submit_script: str) -> str:
    output = run_ssh(remote_host, f"cd '{remote_dir}' && sbatch '{submit_script}'")
    for token in output.strip().split():
        if token.isdigit():
            return token
    raise RuntimeError(f"Could not parse job id from sbatch output:\n{output}")


def cancel_jobs(remote_host: str, job_ids: list[str]) -> None:
    if job_ids:
        run_ssh(remote_host, "scancel " + " ".join(job_ids))


def build_plan(args: argparse.Namespace) -> DynamicBatchPlan:
    package_dirs = [path.resolve() for path in args.package_dir]
    explicit_labels = list(args.case_label)
    if explicit_labels and len(explicit_labels) != len(package_dirs):
        raise ValueError("--case-label must be given either zero times or exactly once per --package-dir")

    first_submit_script_name = detect_submit_script_name(package_dirs[0], args.submit_script_name)
    first_submit_script_path = package_dirs[0] / first_submit_script_name
    partition = detect_submit_script_partition(first_submit_script_path)
    queue_nodes = probe_slurm_partition_nodes(args.remote_host, partition)
    eligible_nodes = sorted(
        [node for node in queue_nodes if node.idle_cpus > 0],
        key=lambda item: (item.idle_cpus, item.total_cpus, item.node),
        reverse=True,
    )

    jobs: list[BatchJobPlan] = []
    for index, package_dir in enumerate(package_dirs):
        submit_script_name = detect_submit_script_name(package_dir, args.submit_script_name)
        submit_script_path = package_dir / submit_script_name
        package_partition = detect_submit_script_partition(submit_script_path)
        if package_partition != partition:
            raise ValueError(
                f"Mixed partitions are not supported in one batch plan: {submit_script_path} uses {package_partition}, "
                f"but the first package uses {partition}."
            )
        requested_cpus, requested_time = read_submit_resources(submit_script_path)
        if index < len(eligible_nodes):
            node = eligible_nodes[index]
            selected_cpus = node.idle_cpus
            planned_idle_node = node.node
            planned_idle_cpus = node.idle_cpus
        else:
            selected_cpus = args.min_cpus_per_task
            planned_idle_node = "queue-only"
            planned_idle_cpus = 0
        if args.max_cpus_per_task is not None:
            selected_cpus = min(selected_cpus, args.max_cpus_per_task)
        selected_cpus = max(selected_cpus, args.min_cpus_per_task)
        selected_time = derive_time_limit(
            selected_cpus,
            baseline_runtime_hours_at_8cpu=args.baseline_runtime_hours_at_8cpu,
            time_buffer_hours=args.time_buffer_hours,
            min_time_limit_hours=args.min_time_limit_hours,
            max_time_limit_hours=args.max_time_limit_hours,
            round_minutes=args.time_round_minutes,
        )
        case_label = infer_case_label(package_dir, explicit_labels[index] if explicit_labels else None)
        jobs.append(
            BatchJobPlan(
                local_dir=str(package_dir),
                case_label=case_label,
                package_name=package_dir.name,
                submit_script=submit_script_name,
                remote_dir=remote_dir_for_package(args.remote_root, case_label, package_dir.name, args.date_tag),
                requested_cpus_per_task=requested_cpus,
                requested_time_limit=requested_time,
                selected_cpus_per_task=selected_cpus,
                selected_time_limit=selected_time,
                planned_idle_node=planned_idle_node,
                planned_idle_cpus=planned_idle_cpus,
            )
        )

    return DynamicBatchPlan(
        remote_host=args.remote_host,
        partition=partition,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        submit_mode=bool(args.submit),
        allow_overwrite_remote=bool(args.allow_overwrite_remote),
        planned_idle_node_semantics=(
            "plan-time greedy idle-slot suggestion from the captured queue snapshot; "
            "Slurm may place the job on a different node or keep it pending."
        ),
        queue_probe_filter=(
            "All probed partition nodes are recorded in queue_snapshot; "
            "only nodes with idle_cpus > 0 are eligible for immediate slot assignment."
        ),
        min_cpus_per_task=args.min_cpus_per_task,
        max_cpus_per_task=args.max_cpus_per_task,
        baseline_runtime_hours_at_8cpu=args.baseline_runtime_hours_at_8cpu,
        time_buffer_hours=args.time_buffer_hours,
        min_time_limit_hours=args.min_time_limit_hours,
        max_time_limit_hours=args.max_time_limit_hours,
        jobs=jobs,
        queue_snapshot=[asdict(node) for node in queue_nodes],
    )


def write_markdown(path: Path, plan: DynamicBatchPlan) -> None:
    lines = [
        "# Dynamic Slurm Batch Submit Plan",
        "",
        f"- generated_at: `{plan.generated_at}`",
        f"- remote_host: `{plan.remote_host}`",
        f"- partition: `{plan.partition}`",
        f"- submit_mode: `{plan.submit_mode}`",
        f"- allow_overwrite_remote: `{plan.allow_overwrite_remote}`",
        f"- planned_idle_node_semantics: {plan.planned_idle_node_semantics}",
        f"- queue_probe_filter: {plan.queue_probe_filter}",
        "",
        "## Queue Snapshot",
        "",
        "| Node | State | Total CPUs | Allocated | Idle | Other |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for node in plan.queue_snapshot:
        lines.append(
            f"| {node['node']} | {node['state']} | {node['total_cpus']} | {node['allocated_cpus']} | {node['idle_cpus']} | {node['other_cpus']} |"
        )
    lines.extend(
        [
            "",
            "## Job Plan",
            "",
            "| Case Label | Package | Planned Idle Node | Idle CPUs | Requested CPUs | Selected CPUs | Requested Time | Selected Time | Submitted Job | State |",
            "|---|---|---|---:|---:|---:|---|---|---|---|",
        ]
    )
    for job in plan.jobs:
        lines.append(
            f"| {job.case_label} | `{job.local_dir}` | {job.planned_idle_node} | {job.planned_idle_cpus} | "
            f"{job.requested_cpus_per_task} | {job.selected_cpus_per_task} | {job.requested_time_limit} | "
            f"{job.selected_time_limit} | {job.submitted_job_id or ''} | {job.submitted_state or ''} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This plan is intentionally dynamic. It does not hard-code any historical `cpu / walltime` split.",
            "- Each package consumes the largest currently visible single-node idle slot that is still available in the greedy input order.",
            "- `planned_idle_node` is only the plan-time slot suggestion from the captured queue snapshot. The final runtime node is assigned by Slurm.",
            "- `queue_snapshot` records all probed partition nodes; only rows with `idle_cpus > 0` are eligible for immediate slot assignment.",
            "- If a node exposes more idle CPUs than the historical example, the corresponding package can scale up automatically unless an explicit cap is set.",
            "- The derived walltime is scaled from the observed runtime at `8 CPU`, then padded with a fixed time buffer.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan(output_dir: Path, plan: DynamicBatchPlan) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(plan), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "summary.md", plan)


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    if args.submit:
        cancel_jobs(args.remote_host, args.cancel_jobid)
        for job in plan.jobs:
            local_dir = Path(job.local_dir)
            submit_script_path = local_dir / job.submit_script
            rewrite_submit_resources(
                submit_script_path,
                cpus_per_task=job.selected_cpus_per_task,
                time_limit=job.selected_time_limit,
            )
            ensure_remote_package(
                local_dir,
                args.remote_host,
                job.remote_dir,
                allow_overwrite=args.allow_overwrite_remote,
            )
            job_id = submit_job(args.remote_host, job.remote_dir, job.submit_script)
            job.submitted_job_id = job_id
            job.submitted_state = query_job_state(args.remote_host, job_id)

    write_plan(args.output_dir.resolve(), plan)
    for job in plan.jobs:
        print(
            f"{job.case_label}: {job.selected_cpus_per_task} cpu, {job.selected_time_limit}, "
            f"idle_slot={job.planned_idle_node}:{job.planned_idle_cpus}, "
            f"job_id={job.submitted_job_id or '-'} state={job.submitted_state or '-'}"
        )
    print(f"[plan] wrote {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
