#!/usr/bin/env python3
"""Run strict single-chain-average Rg using gmx polystat."""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

try:
    from canonical_outputs import ANALYSIS_VERSION, batch_dir, case_dir, default_output_root
    from trajectory_helpers import parse_xvg_numeric, resolve_analysis_tpr, resolve_case_root, resolve_nvt_dir, resolve_trajectory, time_metadata
except ImportError:  # pragma: no cover
    from .canonical_outputs import ANALYSIS_VERSION, batch_dir, case_dir, default_output_root  # type: ignore
    from .trajectory_helpers import parse_xvg_numeric, resolve_analysis_tpr, resolve_case_root, resolve_nvt_dir, resolve_trajectory, time_metadata  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Run gmx polystat Rg for one LCB case.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--gmx-bin", default="gmx")
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--batch-date", default="20260501")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_polystat_rg_column(xvg_path):
    legends = {}
    for line in Path(xvg_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r'@\s+s(\d+)\s+legend\s+"([^"]+)"', line)
        if match:
            legends[int(match.group(1)) + 1] = match.group(2)
    for col, name in legends.items():
        if name.strip() == r"<R\sg\N>":
            return col
    for col, name in legends.items():
        lname = name.lower()
        if ("r\\sg" in lname or "rg" in lname or "gyr" in lname) and "eig" not in lname and "end" not in lname:
            return col
    return 2


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main():
    args = parse_args()
    case_root = resolve_case_root(args.case_dir)
    nvt_dir = resolve_nvt_dir(args.case_dir)
    out_dir = case_dir(batch_dir(args.output_root, "rg", args.batch_date), args.case_id, "rg")
    out_dir.mkdir(parents=True, exist_ok=True)
    tpr = resolve_analysis_tpr(args.gmx_bin, case_root, nvt_dir, force=args.force)
    traj = resolve_trajectory(args.gmx_bin, nvt_dir, out_dir, force=args.force)
    xvg = out_dir / "rg_polystat.xvg"
    stdout = out_dir / "polystat.stdout"
    stderr = out_dir / "polystat.stderr"

    if args.force or not xvg.exists() or xvg.stat().st_size == 0:
        proc = subprocess.run(
            [str(args.gmx_bin), "polystat", "-s", str(tpr), "-f", str(traj), "-o", str(xvg)],
            input="0\n",
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout.write_text(proc.stdout, encoding="utf-8")
        stderr.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"gmx polystat failed for {args.case_id}\n{proc.stderr}")

    data = parse_xvg_numeric(xvg)
    if data.shape[1] < 3:
        raise ValueError(f"Unexpected polystat width for {args.case_id}: {data.shape[1]}")
    rg_col = parse_polystat_rg_column(xvg)
    if rg_col >= data.shape[1]:
        raise ValueError(f"Rg column {rg_col} outside polystat width {data.shape[1]}")
    values = data[:, rg_col]
    times = data[:, 0]
    err_text = stderr.read_text(encoding="utf-8", errors="ignore") if stderr.exists() else ""
    match = re.search(r"Group .* consists of\s+(\d+)\s+molecules", err_text)
    n_chains = int(match.group(1)) if match else ""
    status = "OK" if n_chains == 150 and values.size > 0 else "CHECK"
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    summary = {
        "case_id": args.case_id,
        "status": status,
        "n_chains": n_chains,
        **time_metadata(times),
        "method": "per-chain gmx polystat on old-server AMD gmx2026; group 0 System split into molecules",
        "analysis_version": ANALYSIS_VERSION,
        "source_case_dir": str(case_root),
        "rg_mean_nm": float(np.mean(values)),
        "rg_std_time_nm": std,
        "rg_sem_time_nm": std / math.sqrt(values.size) if values.size else "",
        "rg2_mean_nm2": float(np.mean(values * values)),
        "rg2_over_linear_rg2": "",
        "linear_baseline_case_id": "",
        "baseline_status": "missing_matching_linear_400ns_baseline",
        "polystat_xvg": str(xvg),
        "trajectory": str(traj),
        "rg_column_index": rg_col,
    }
    write_json(out_dir / "rg_summary.json", summary)
    print(f"{args.case_id}\t{status}\tRg={summary['rg_mean_nm']:.6g}\tn_chains={n_chains}")


if __name__ == "__main__":
    raise SystemExit(main())
