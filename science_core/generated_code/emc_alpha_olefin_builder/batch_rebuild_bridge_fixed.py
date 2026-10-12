#!/usr/bin/env python3
"""
Batch rebuild LCB GROMACS bridge packages from canonical EMC outputs.

The input tree is expected to be:
  output/<run_label>/<case_id>/remote_build/<run_label>/

Each case is converted through emc_to_gromacs.py and written into a clean bridge
directory that is ready for representative grompp validation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "output" / "screen_lcb100"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / f"lcb100_bridge_fixed_{datetime.now().strftime('%Y%m%d')}"
DEFAULT_RUN_LABEL = "screen_lcb100"


@dataclass
class CaseResult:
    case_id: str
    case_dir: str
    output_dir: str
    emc_ready: bool
    returncode: int
    ready_for_md: bool
    gro_file: str | None
    polymer_itp: str | None
    topol_file: str | None
    stderr_tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the full LCB100 GROMACS bridge from local canonical EMC outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Local canonical LCB EMC tree. Default: output/screen_lcb100",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for rebuilt bridge packages.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Optional case_id filter. May be repeated.",
    )
    parser.add_argument(
        "--case-min",
        type=int,
        help="Optional minimum numeric LCB sequence, e.g. 101 for lcb101.",
    )
    parser.add_argument(
        "--case-max",
        type=int,
        help="Optional maximum numeric LCB sequence, e.g. 200 for lcb200.",
    )
    parser.add_argument(
        "--run-label",
        default=DEFAULT_RUN_LABEL,
        help="Canonical remote_build label to check. Default: screen_lcb100.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove output-root before rebuilding.",
    )
    return parser.parse_args()


def load_case_metadata(case_dir: Path) -> dict:
    case_json = case_dir / "case.json"
    if not case_json.exists():
        raise FileNotFoundError(f"Missing case.json in {case_dir}")
    return json.loads(case_json.read_text(encoding="utf-8"))


def lcb_sequence(case_id: str) -> int | None:
    if not case_id.startswith("lcb"):
        return None
    digits = []
    for char in case_id[3:]:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    return int("".join(digits))


def iter_case_dirs(
    input_root: Path,
    selected_cases: set[str],
    case_min: int | None,
    case_max: int | None,
) -> list[Path]:
    case_dirs: list[Path] = []
    for case_dir in sorted(path for path in input_root.iterdir() if path.is_dir() and path.name.startswith("lcb")):
        meta = load_case_metadata(case_dir)
        case_id = meta["case_id"]
        if selected_cases and case_id not in selected_cases:
            continue
        sequence = lcb_sequence(case_id)
        if case_min is not None and (sequence is None or sequence < case_min):
            continue
        if case_max is not None and (sequence is None or sequence > case_max):
            continue
        case_dirs.append(case_dir)
    return case_dirs


def emc_case_ready(case_dir: Path, run_label: str) -> bool:
    remote_build_dir = case_dir / "remote_build" / run_label
    setup_exitcode = remote_build_dir / "setup.exitcode"
    build_exitcode = remote_build_dir / "build.exitcode"
    if not setup_exitcode.exists() or not build_exitcode.exists():
        return False
    return (
        setup_exitcode.read_text(encoding="utf-8").strip() == "0"
        and build_exitcode.read_text(encoding="utf-8").strip() == "0"
    )


def verify_bridge_output(output_dir: Path, case_id: str) -> tuple[bool, Path | None, Path | None, Path | None]:
    gro_file = output_dir / f"{case_id}.gro"
    polymer_itp = output_dir / f"{case_id}_polymer.itp"
    topol_file = output_dir / "topol.top"
    ready = gro_file.exists() and polymer_itp.exists() and topol_file.exists()
    return (
        ready,
        gro_file if gro_file.exists() else None,
        polymer_itp if polymer_itp.exists() else None,
        topol_file if topol_file.exists() else None,
    )


def run_conversion(case_dir: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "emc_to_gromacs.py"),
        "--case-dir",
        str(case_dir),
        "--output-dir",
        str(output_dir),
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def write_summary(output_root: Path, results: list[CaseResult], *, input_root: Path, run_label: str) -> None:
    ready_cases = [result for result in results if result.ready_for_md]
    payload = {
        "description": f"LCB bridge rebuilt locally from canonical {run_label} EMC outputs",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root),
        "run_label": run_label,
        "total_cases": len(results),
        "ready_for_md": len(ready_cases),
        "failed_cases": len(results) - len(ready_cases),
        "cases": [asdict(result) for result in results],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    selected_cases = set(args.case)
    run_label = args.run_label

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    case_dirs = iter_case_dirs(input_root, selected_cases, args.case_min, args.case_max)
    if not case_dirs:
        raise RuntimeError(f"No matching LCB case directories found under {input_root}")

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Cases:       {len(case_dirs)}")
    print()

    for index, case_dir in enumerate(case_dirs, start=1):
        meta = load_case_metadata(case_dir)
        case_id = meta["case_id"]
        case_output_dir = output_root / case_id

        if case_output_dir.exists():
            shutil.rmtree(case_output_dir)
        case_output_dir.mkdir(parents=True, exist_ok=True)

        ready_emc = emc_case_ready(case_dir, run_label)
        print(f"[{index:3d}/{len(case_dirs)}] {case_id}", end=" ", flush=True)

        if not ready_emc:
            print("FAIL (canonical EMC output not ready)")
            results.append(
                CaseResult(
                    case_id=case_id,
                    case_dir=str(case_dir),
                    output_dir=str(case_output_dir),
                    emc_ready=False,
                    returncode=1,
                    ready_for_md=False,
                    gro_file=None,
                    polymer_itp=None,
                    topol_file=None,
                    stderr_tail="canonical EMC output missing or non-zero exitcode",
                )
            )
            continue

        result = run_conversion(case_dir, case_output_dir)
        ready_for_md, gro_file, polymer_itp, topol_file = verify_bridge_output(case_output_dir, case_id)

        if result.returncode == 0 and ready_for_md:
            print("OK")
        elif result.returncode == 0:
            print("FAIL (missing output files)")
        else:
            print(f"FAIL (returncode={result.returncode})")

        results.append(
            CaseResult(
                case_id=case_id,
                case_dir=str(case_dir),
                output_dir=str(case_output_dir),
                emc_ready=True,
                returncode=result.returncode,
                ready_for_md=ready_for_md and result.returncode == 0,
                gro_file=str(gro_file) if gro_file else None,
                polymer_itp=str(polymer_itp) if polymer_itp else None,
                topol_file=str(topol_file) if topol_file else None,
                stderr_tail=result.stderr.strip()[-500:],
            )
        )

    write_summary(output_root, results, input_root=input_root, run_label=run_label)

    ready_count = sum(1 for result in results if result.ready_for_md)
    failed_count = len(results) - ready_count
    print()
    print(f"Ready for MD: {ready_count}/{len(results)}")
    print(f"Failed:       {failed_count}/{len(results)}")
    print(f"Manifest:     {output_root / 'manifest.json'}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
