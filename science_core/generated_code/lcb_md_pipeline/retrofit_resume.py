#!/usr/bin/env python3
"""Retrofit existing run_manual.sh / submit.slurm packages with the resume-aware
4-stage pipeline, reusing each package's original variable values.

Safe defaults:
  - dry-run mode prints per-case variable dumps without writing files
  - live mode backs up the original to <name>.pre-resume.<ts> before writing
  - skips already-retrofitted scripts (marker: '# --- Resume-aware helpers')
  - skips completed packages (nvt_prod confout + 'Finished mdrun') unless
    --include-completed is passed
  - does NOT try to detect remote live state; user is responsible for only
    overlaying scripts onto stopped remote directories

Usage:
  python retrofit_resume.py --scan-dir "${LOCAL_ROOT}\\...\\output" [--dry-run]
                            [--cases lcb15 sc01] [--include-completed]
"""

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
MANUAL_TEMPLATE = SCRIPT_DIR / "manual_template.sh"
SLURM_TEMPLATE = SCRIPT_DIR / "slurm_template.sh"

RESUME_MARKER = "# --- Resume-aware helpers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan-dir", required=True, type=Path)
    parser.add_argument("--cases", nargs="*", default=[],
                        help="Case id prefixes; retrofits only matching package dirs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-completed", action="store_true",
                        help="Also retrofit packages whose nvt_prod already finished.")
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_assignment(text: str, key: str) -> Optional[str]:
    pattern = re.compile(rf'^\s*{re.escape(key)}\s*=\s*"([^"]*)"\s*$', re.MULTILINE)
    match = pattern.search(text)
    return match.group(1) if match else None


def extract_sbatch(text: str, flag: str) -> Optional[str]:
    pattern = re.compile(rf'^\s*#SBATCH\s+--{re.escape(flag)}=(\S+)\s*$', re.MULTILINE)
    match = pattern.search(text)
    return match.group(1) if match else None


def extract_module_lines(text: str) -> str:
    """Block between `set -e... pipefail` and the first `CASE_ID=` line."""
    lines = text.splitlines()
    start = None  # type: Optional[int]
    end = None  # type: Optional[int]
    for idx, line in enumerate(lines):
        if start is None and re.match(r'^\s*set\s+-e[uo]*\s+pipefail\s*$', line):
            start = idx + 1
        if start is not None and line.strip().startswith("CASE_ID="):
            end = idx
            break
    if start is None or end is None:
        return ""
    return "\n".join(lines[start:end]).strip("\n")


def log_has_finished(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "Finished mdrun" in line:
                    return True
    except OSError:
        pass
    return False


def nvt_prod_completed(pkg_dir: Path) -> bool:
    nvt = pkg_dir / "nvt_prod"
    if not nvt.is_dir():
        return False
    gro_candidates = [nvt / "nvt_prod.gro", *sorted(nvt.glob("nvt_prod.part*.gro"))]
    if not any(p.exists() and p.stat().st_size > 0 for p in gro_candidates):
        return False
    log_candidates = [nvt / "nvt_prod.log", *sorted(nvt.glob("nvt_prod.part*.log"))]
    return any(p.exists() and log_has_finished(p) for p in log_candidates)


def case_matches(name: str, filters: List[str]) -> bool:
    if not filters:
        return True
    return any(name == item or name.startswith(item) or item in name for item in filters)


def render_manual(values: Dict[str, str]) -> str:
    template = read_text(MANUAL_TEMPLATE)
    return template.format(
        case_id=values["case_id"],
        gro_file=values["gro"],
        module_lines=values["module_lines"],
        cpus=values["threads"],
        gmx_bin=values["gmx_bin"],
        maxwarn=values["maxwarn"],
        mdrun_gpu_args=values["mdrun_gpu_args"],
    )


def render_slurm(values: Dict[str, str]) -> str:
    template = read_text(SLURM_TEMPLATE)
    gres_line = f"#SBATCH --gres={values['gres']}" if values.get("gres") else ""
    return template.format(
        case_id=values["case_id"],
        partition=values["partition"],
        cpus=values["cpus"],
        time_limit=values["time_limit"],
        gres_line=gres_line,
        module_lines=values["module_lines"],
        gro_file=values["gro"],
        gmx_bin=values["gmx_bin"],
        maxwarn=values["maxwarn"],
        mdrun_gpu_args=values["mdrun_gpu_args"],
    )


def parse_manual(text: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    values = {
        "case_id": extract_assignment(text, "CASE_ID"),
        "gro": extract_assignment(text, "GRO"),
        "gmx_bin": extract_assignment(text, "GMX_BIN"),
        "maxwarn": extract_assignment(text, "MAXWARN"),
        "threads": extract_assignment(text, "THREADS"),
        "mdrun_gpu_args": extract_assignment(text, "MDRUN_GPU_ARGS") or "",
        "module_lines": extract_module_lines(text),
    }
    missing = [k for k in ("case_id", "gro", "gmx_bin", "maxwarn", "threads") if values[k] is None]
    if missing:
        return None, f"missing {missing}"
    return values, None


def parse_slurm(text: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    values = {
        "case_id": extract_assignment(text, "CASE_ID"),
        "gro": extract_assignment(text, "GRO"),
        "gmx_bin": extract_assignment(text, "GMX_BIN"),
        "maxwarn": extract_assignment(text, "MAXWARN"),
        "mdrun_gpu_args": extract_assignment(text, "MDRUN_GPU_ARGS") or "",
        "module_lines": extract_module_lines(text),
        "partition": extract_sbatch(text, "partition"),
        "cpus": extract_sbatch(text, "cpus-per-task"),
        "time_limit": extract_sbatch(text, "time"),
        "gres": extract_sbatch(text, "gres"),
    }
    missing = [k for k in ("case_id", "gro", "gmx_bin", "maxwarn",
                           "partition", "cpus", "time_limit") if values[k] is None]
    if missing:
        return None, f"missing {missing}"
    return values, None


def print_values(values: Dict[str, str], kind: str, lines: List[str]) -> None:
    common_keys = ["case_id", "gro", "gmx_bin", "maxwarn", "mdrun_gpu_args"]
    keys = common_keys + (["threads"] if kind == "manual"
                          else ["partition", "cpus", "time_limit", "gres"])
    for key in keys:
        lines.append(f"    {key:18s} = {values.get(key)!r}")
    module_lines = values.get("module_lines", "")
    module_preview = module_lines.splitlines() if module_lines else []
    if module_preview:
        lines.append(f"    module_lines ({len(module_preview)} line(s)):")
        for module_line in module_preview:
            lines.append(f"      | {module_line}")


def retrofit_script(
    script_path: Path,
    kind: str,
    dry_run: bool,
    messages: List[str],
) -> bool:
    text = read_text(script_path)
    if RESUME_MARKER in text:
        messages.append(f"  {script_path.name}: already retrofitted; skip")
        return False

    parser_fn = parse_manual if kind == "manual" else parse_slurm
    values, err = parser_fn(text)
    if values is None:
        messages.append(f"  {script_path.name}: parse failed ({err}); skip")
        return False

    messages.append(f"  {script_path.name}:")
    print_values(values, kind, messages)

    try:
        new_text = render_manual(values) if kind == "manual" else render_slurm(values)
    except KeyError as exc:
        messages.append(f"  {script_path.name}: render failed ({exc}); skip")
        return False

    if dry_run:
        messages.append(f"  action: rewrite (dry-run)")
        return True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = script_path.with_name(f"{script_path.name}.pre-resume.{timestamp}")
    shutil.copy2(script_path, backup)
    with script_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(new_text)
    messages.append(f"  action: rewrote (backup={backup.name})")
    return True


def retrofit_package(pkg_dir: Path, dry_run: bool, include_completed: bool) -> Tuple[str, List[str]]:
    messages = []  # type: List[str]
    manual = pkg_dir / "run_manual.sh"
    slurm = pkg_dir / "submit.slurm"
    if not manual.exists() and not slurm.exists():
        return "noop", messages

    if not include_completed and nvt_prod_completed(pkg_dir):
        messages.append("  nvt_prod already completed; skip")
        return "skipped_completed", messages

    touched = []  # type: List[str]
    if manual.exists() and retrofit_script(manual, "manual", dry_run, messages):
        touched.append("manual")
    if slurm.exists() and retrofit_script(slurm, "slurm", dry_run, messages):
        touched.append("slurm")

    if not touched:
        return "skipped_or_failed", messages
    if len(touched) == 2:
        return "rewrote_both", messages
    return f"rewrote_{touched[0]}", messages


def main() -> int:
    args = parse_args()
    scan_dir = args.scan_dir.resolve()
    if not scan_dir.is_dir():
        print(f"ERROR: scan-dir not found: {scan_dir}", file=sys.stderr)
        return 2
    if not MANUAL_TEMPLATE.exists() or not SLURM_TEMPLATE.exists():
        print(
            f"ERROR: template files missing: {MANUAL_TEMPLATE} / {SLURM_TEMPLATE}",
            file=sys.stderr,
        )
        return 2

    package_dirs = set()  # type: set
    for root, dirnames, filenames in os.walk(str(scan_dir), followlinks=False):
        # Prune symlinked directories explicitly so historical looped run trees
        # under remote lz_project_runs do not break the scan on older servers.
        dirnames[:] = [
            name for name in dirnames
            if not os.path.islink(os.path.join(root, name))
        ]
        for name in ("run_manual.sh", "submit.slurm"):
            if name in filenames:
                package_dirs.add(Path(root))

    counts = {}  # type: Dict[str, int]
    for pkg_dir in sorted(package_dirs):
        if not case_matches(pkg_dir.name, args.cases):
            continue
        status, messages = retrofit_package(pkg_dir, args.dry_run, args.include_completed)
        counts[status] = counts.get(status, 0) + 1
        try:
            rel = pkg_dir.relative_to(scan_dir)
        except ValueError:
            rel = pkg_dir
        print(f"[{status}] {rel}")
        for line in messages:
            print(line)

    print()
    print("Summary:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"mode: {'dry-run' if args.dry_run else 'live'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
