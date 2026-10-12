#!/usr/bin/env python3
"""Backfill clean-chain COM g3 MSD into the clean IntrachainDynamics bundle.

The clean intrachain bundle produced on 2026-06-07 contains repeat-unit COM
``g1_monomer`` outputs but did not carry the already computed clean chain-COM
MSD.  This utility bridges the existing ``o7clean_msd*`` outputs into the
IntrachainDynamics tree without rerunning MD or changing physical parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


ANALYSIS_VERSION = "canonical_v2_clean_g3_backfill"
TIMEPOINTS: Tuple[Tuple[float, str], ...] = (
    (1000.0, "1ns"),
    (10000.0, "10ns"),
    (50000.0, "50ns"),
    (100000.0, "100ns"),
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def finite_ratio(numerator: Any, denominator: Any) -> Any:
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(n) or not math.isfinite(d) or d == 0:
        return ""
    return n / d


def interp_value(times: np.ndarray, values: np.ndarray, target_ps: float) -> Any:
    if len(times) == 0 or target_ps < float(times[0]) or target_ps > float(times[-1]):
        return ""
    return float(np.interp(target_ps, times, values))


def local_alpha(
    times: np.ndarray,
    values: np.ndarray,
    begin_ps: float = 100000.0,
    end_ps: float = 300000.0,
) -> Any:
    mask = (times >= begin_ps) & (times <= end_ps) & (times > 0) & (values > 0)
    if int(mask.sum()) < 3:
        return ""
    coeff = np.polyfit(np.log(times[mask]), np.log(values[mask]), 1)
    return float(coeff[0])


def numeric_series(rows: List[Dict[str, str]], value_field: str) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    values: List[float] = []
    for row in rows:
        try:
            times.append(float(row["time_ps"]))
            values.append(float(row[value_field]))
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def build_case_summary(
    case_id: str,
    case_out: Path,
    msd_case_dir: Path,
    g3_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    times, values = numeric_series(g3_rows, "msd_nm2")
    g1_summary = read_json(case_out / "g1_monomer_summary.json")
    msd_summary = read_json(msd_case_dir / "msd" / "msd_summary.json")

    summary: Dict[str, Any] = {
        "case_id": case_id,
        "status": "ok" if len(times) else "missing_g3_timeseries",
        "analysis_version": ANALYSIS_VERSION,
        "method": "clean g3 chain COM MSD backfilled from available o7clean_msd* outputs; g1 is repeat-unit COM MSD when present",
        "g3_source": str((msd_case_dir / "msd" / "msd_timeseries.csv").resolve()),
        "n_frames_g3": int(len(times)),
        "time_first_ps": float(times[0]) if len(times) else "",
        "time_last_ps": float(times[-1]) if len(times) else "",
        "dt_ps": float(np.median(np.diff(times))) if len(times) > 1 else "",
        "g3_alpha_100_300": local_alpha(times, values),
        "g3_loglog_slope_source": msd_summary.get("loglog_slope", ""),
        "g3_diffusive_regime_ok": msd_summary.get("diffusive_regime_ok", ""),
        "g3_D_mean_m2_s": msd_summary.get("D_mean_m2_s", ""),
        "g3_D_median_m2_s": msd_summary.get("D_median_m2_s", ""),
        "g3_window_D_cv_pct": msd_summary.get("window_D_cv_pct", ""),
        "g3_regime_warning": msd_summary.get("regime_warning", ""),
    }
    for target_ps, label in TIMEPOINTS:
        summary[f"g3_{label}_nm2"] = interp_value(times, values, target_ps)
        g1_value = g1_summary.get(f"g1_monomer_{label}_nm2", "")
        summary[f"g1_monomer_{label}_nm2"] = g1_value
        summary[f"g1_monomer_over_g3_{label}"] = finite_ratio(
            g1_value,
            summary[f"g3_{label}_nm2"],
        )
    if g1_summary:
        summary["n_frames_g1_monomer"] = g1_summary.get("n_frames_g1_monomer", "")
        summary["g1_monomer_alpha_100_300"] = g1_summary.get("g1_monomer_alpha_100_300", "")
        summary["g1_source"] = str((case_out / "g1_monomer_com.csv").resolve())
    return summary


def update_g1_summary(case_out: Path, g3_summary: Dict[str, Any]) -> Dict[str, Any]:
    path = case_out / "g1_monomer_summary.json"
    summary = read_json(path)
    if not summary:
        return {}
    for _target_ps, label in TIMEPOINTS:
        summary[f"g1_monomer_over_g3_{label}"] = g3_summary.get(
            f"g1_monomer_over_g3_{label}",
            "",
        )
    summary["g3_backfill_status"] = g3_summary.get("status", "")
    summary["g3_backfill_source"] = g3_summary.get("g3_source", "")
    write_json(path, summary)
    return summary


def discover_msd_roots(output_root: Path) -> List[Path]:
    roots = sorted(
        (path / "cases" for path in output_root.glob("o7clean_msd*") if (path / "cases").exists()),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    return roots


def find_msd_case_dir(case_id: str, msd_roots: List[Path]) -> Optional[Path]:
    for root in msd_roots:
        candidate = root / case_id
        if (candidate / "msd" / "msd_timeseries.csv").exists():
            return candidate
    return None


def run_backfill(intrachain_root: Path, msd_roots: List[Path]) -> Dict[str, Any]:
    cases_root = intrachain_root / "cases"
    if not cases_root.exists():
        raise FileNotFoundError(f"missing IntrachainDynamics cases root: {cases_root}")

    g3_long: List[Dict[str, Any]] = []
    g1_long: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    g1_summaries: List[Dict[str, Any]] = []
    missing: List[str] = []

    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        case_out = case_dir / "intrachain"
        msd_case_dir = find_msd_case_dir(case_id, msd_roots)
        if msd_case_dir is None:
            missing.append(case_id)
            continue
        source_csv = msd_case_dir / "msd" / "msd_timeseries.csv"
        rows = read_csv(source_csv)
        if not rows:
            missing.append(case_id)
            continue

        case_g3_rows: List[Dict[str, Any]] = []
        for row in rows:
            out_row = {
                "time_ps": float(row["time_ps"]),
                "g3_com_msd_nm2": float(row["msd_nm2"]),
            }
            case_g3_rows.append(out_row)
            g3_long.append({"case_id": case_id, **out_row})
        write_csv(case_out / "g3_com_msd.csv", case_g3_rows, ["time_ps", "g3_com_msd_nm2"])

        g3_summary = build_case_summary(case_id, case_out, msd_case_dir, rows)
        write_json(case_out / "g1_g3_summary.json", g3_summary)
        summaries.append(g3_summary)

        g1_summary = update_g1_summary(case_out, g3_summary)
        if g1_summary:
            g1_summaries.append(g1_summary)
        for row in read_csv(case_out / "g1_monomer_com.csv"):
            row["case_id"] = case_id
            g1_long.append(row)

    agg = intrachain_root / "aggregate"
    summaries.sort(key=lambda row: row["case_id"])
    g1_summaries.sort(key=lambda row: row["case_id"])
    g3_long.sort(key=lambda row: (row["case_id"], float(row["time_ps"])))
    g1_long.sort(key=lambda row: (row["case_id"], float(row["time_ps"])))

    summary_fields = [
        "case_id",
        "status",
        "analysis_version",
        "method",
        "n_frames_g3",
        "time_first_ps",
        "time_last_ps",
        "dt_ps",
        "g3_1ns_nm2",
        "g3_10ns_nm2",
        "g3_50ns_nm2",
        "g3_100ns_nm2",
        "g3_alpha_100_300",
        "g3_loglog_slope_source",
        "g3_diffusive_regime_ok",
        "g3_D_mean_m2_s",
        "g3_D_median_m2_s",
        "g3_window_D_cv_pct",
        "g3_regime_warning",
        "n_frames_g1_monomer",
        "g1_monomer_1ns_nm2",
        "g1_monomer_10ns_nm2",
        "g1_monomer_50ns_nm2",
        "g1_monomer_100ns_nm2",
        "g1_monomer_alpha_100_300",
        "g1_monomer_over_g3_1ns",
        "g1_monomer_over_g3_10ns",
        "g1_monomer_over_g3_50ns",
        "g1_monomer_over_g3_100ns",
        "g3_source",
        "g1_source",
    ]
    g1_summary_fields = [
        "case_id",
        "status",
        "analysis_version",
        "method",
        "monomer_mapping_method",
        "n_chains",
        "n_frames_g1_monomer",
        "time_first_ps",
        "time_last_ps",
        "dt_ps",
        "carbon_per_chain",
        "backbone_repeat_units",
        "monomers_per_chain",
        "g1_monomer_1ns_nm2",
        "g1_monomer_10ns_nm2",
        "g1_monomer_50ns_nm2",
        "g1_monomer_100ns_nm2",
        "g1_monomer_alpha_100_300",
        "g1_monomer_over_g3_1ns",
        "g1_monomer_over_g3_10ns",
        "g1_monomer_over_g3_50ns",
        "g1_monomer_over_g3_100ns",
        "g3_backfill_status",
        "g3_backfill_source",
        "source_case_dir",
    ]
    write_csv(agg / "g3_long.csv", g3_long, ["case_id", "time_ps", "g3_com_msd_nm2"])
    write_csv(agg / "g1_g3_summary.csv", summaries, summary_fields)
    write_csv(agg / "g1_monomer_long.csv", g1_long, ["case_id", "time_ps", "g1_monomer_com_msd_nm2"])
    write_csv(agg / "g1_monomer_summary.csv", g1_summaries, g1_summary_fields)
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "method": "backfill clean g3 chain-COM MSD from available o7clean_msd* outputs into clean IntrachainDynamics",
        "intrachain_root": str(intrachain_root.resolve()),
        "msd_roots": [str(path.resolve()) for path in msd_roots],
        "case_count": len(summaries),
        "g3_long_rows": len(g3_long),
        "g1_monomer_case_count": len(g1_summaries),
        "missing_g3_case_count": len(missing),
        "missing_g3_cases": missing,
        "physical_parameters_changed": False,
        "md_rerun": False,
    }
    write_json(agg / "intrachain_clean_g3_backfill_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intrachain-root",
        type=Path,
        default=Path("output/o7clean_integrated/IntrachainDynamics"),
    )
    parser.add_argument(
        "--msd-root",
        type=Path,
        action="append",
        dest="msd_roots",
        help="Clean MSD cases root. May be supplied more than once. Defaults to all output/o7clean_msd*/cases roots, newest first.",
    )
    args = parser.parse_args()
    msd_roots = args.msd_roots or discover_msd_roots(args.intrachain_root.parents[1])
    manifest = run_backfill(args.intrachain_root, msd_roots)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
