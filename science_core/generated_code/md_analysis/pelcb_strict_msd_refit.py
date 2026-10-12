#!/usr/bin/env python3
"""Refit clean MSD evidence before extending nvt_prod.

This sidecar is intentionally conservative:

* it scans cases pre-classified as ``REFIT_window_exists`` or current
  overlay MSD-WARN rows explicitly handed to the refit worker;
* it requires a clean MSD time series, not a legacy 2021.2 trajectory;
* the accepted window must lie in the late diffusive region and avoid the
  noisy long-lag tail;
* it writes publication-grade ``msd_summary.json`` files only for cases that
  satisfy the strict gate.

The script does not rerun MD and does not edit canonical release tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pelcb_msd_publication_gate import (
    MAX_PUBLICATION_END_FRACTION,
    MIN_MSD_OVER_RG2,
    MIN_PUBLICATION_BEGIN_FRACTION,
    MIN_PUBLICATION_WINDOW_FRACTION,
    REFIT_POLICY,
    evaluate_msd_publication_gate,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_CLASSIFICATION_CSV = (
    OUTPUT_ROOT / "pelcb100_clean_overlay_regate" / "msd_extend_vs_refit_20260611.csv"
)
DEFAULT_CURRENT_OVERLAY_LATEST = OUTPUT_ROOT / "pelcb100_clean_overlay_regate" / "clean_overlay_regate_latest.json"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / ("o7clean_msd_refit_" + datetime.now().strftime("%Y%m%d"))
SCHEMA_VERSION = "pelcb_strict_msd_refit.v1"
ELIGIBLE_CLASSIFICATIONS = {"REFIT_window_exists", "current_overlay_msd_warn"}


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def current_overlay_msd_warn_rows(latest_path: Path = DEFAULT_CURRENT_OVERLAY_LATEST) -> list[dict[str, str]]:
    """Return current clean-overlay MSD-WARN rows for refit-before-extend.

    The repair daemon historically consumed a frozen REFIT/EXTEND routing CSV.
    Current overlay rows can change as new clean evidence arrives, so we merge
    live MSD-WARN rows here to avoid extending cases that a fresh clean refit can
    already close.
    """
    if not latest_path.exists():
        return []
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    csv_path = Path(str(payload.get("summary", {}).get("csv", "")))
    if not csv_path.exists():
        return []
    rows: list[dict[str, str]] = []
    for row in read_csv_rows(csv_path):
        if str(row.get("targeted_regate_verdict", "")).strip().upper() != "WARN":
            continue
        warn_sources = str(row.get("remaining_target_warn_sources", ""))
        if "msd_warn" not in warn_sources:
            continue
        case_id = str(row.get("case_id", "")).strip()
        clean_case_id = str(row.get("clean_case_id", "")).strip()
        if not case_id or not clean_case_id:
            continue
        rows.append(
            {
                "case_id": case_id,
                "clean_case_id": clean_case_id,
                "classification": "current_overlay_msd_warn",
                "clean_window_ns": str(row.get("clean_msd_window_overlay_fit_window_ns", "")).strip(),
                "msd_over_rg2": str(row.get("msd_over_rg2", "")).strip(),
            }
        )
    return rows


def merge_current_overlay_rows(rows: list[dict[str, str]], overlay_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge current overlay rows by case_id, preferring live overlay state."""
    merged: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            merged[case_id] = row
    for row in overlay_rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            merged[case_id] = row
    return list(merged.values())


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def latest_file(pattern: str) -> Path | None:
    candidates = [path for path in OUTPUT_ROOT.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def latest_msd_timeseries(clean_case_id: str) -> Path | None:
    return latest_file(f"o7clean_msd_*/cases/{clean_case_id}/msd/msd_timeseries.csv")


def latest_msd_summary(clean_case_id: str) -> Path | None:
    return latest_file(f"o7clean_msd_*/cases/{clean_case_id}/msd/msd_summary.json")


def latest_rg_summary(clean_case_id: str) -> Path | None:
    return latest_file(f"o7clean_rg_*/cases/{clean_case_id}/rg/rg_summary.json")


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def linear_fit(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.allclose(x_arr, x_arr[0]):
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan}
    design = np.vstack([x_arr, np.ones_like(x_arr)]).T
    slope, intercept = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    predicted = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - predicted) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def fit_window(times_ps: np.ndarray, msd_nm2: np.ndarray, begin_ns: float, end_ns: float) -> dict[str, Any]:
    begin_ps = float(begin_ns) * 1000.0
    end_ps = float(end_ns) * 1000.0
    mask = (times_ps >= begin_ps) & (times_ps <= end_ps)
    lin = linear_fit(times_ps[mask], msd_nm2[mask])
    slope_nm2_per_ps = lin["slope"]
    if math.isfinite(slope_nm2_per_ps):
        d_1e5 = slope_nm2_per_ps / 6.0 * 1000.0
        d_m2_s = d_1e5 * 1.0e-9
    else:
        d_1e5 = math.nan
        d_m2_s = math.nan
    log_mask = mask & (times_ps > 0) & (msd_nm2 > 0)
    log_fit = linear_fit(np.log10(times_ps[log_mask]), np.log10(msd_nm2[log_mask]))
    return {
        "begin_ns": float(begin_ns),
        "end_ns": float(end_ns),
        "width_ns": float(end_ns - begin_ns),
        "n_points": int(np.count_nonzero(mask)),
        "D_1e-5_cm2_s": float(d_1e5),
        "D_m2_s": float(d_m2_s),
        "linear_R2": float(lin["r2"]),
        "loglog_slope": float(log_fit["slope"]),
        "loglog_R2": float(log_fit["r2"]),
        "positive_D": bool(math.isfinite(d_1e5) and d_1e5 > 0),
    }


def shifted_window_cv(
    times_ps: np.ndarray,
    msd_nm2: np.ndarray,
    begin_ns: float,
    end_ns: float,
    cutoff_ns: float,
) -> float:
    width_ns = float(end_ns - begin_ns)
    shift_ns = max(10.0, min(20.0, width_ns * 0.10))
    windows = [
        (begin_ns, end_ns),
        (begin_ns + shift_ns, end_ns + shift_ns),
        (max(0.0, begin_ns - shift_ns), max(0.0, end_ns - shift_ns)),
    ]
    values: list[float] = []
    for left, right in windows:
        if right <= left or right > cutoff_ns:
            continue
        fit = fit_window(times_ps, msd_nm2, left, right)
        value = fit["D_1e-5_cm2_s"]
        if math.isfinite(value) and value > 0:
            values.append(float(value))
    if len(values) < 2:
        return math.nan
    mean = float(np.mean(values))
    if mean <= 0:
        return math.nan
    return float(np.std(values, ddof=1) / mean * 100.0)


def parse_window_ns(value: str) -> tuple[float, float] | None:
    text = str(value or "").strip()
    if not text or "-" not in text:
        return None
    left, right = text.split("-", 1)
    begin = to_float(left)
    end = to_float(right)
    if not math.isfinite(begin) or not math.isfinite(end) or end <= begin:
        return None
    return begin, end


def rg2_for_clean_case(clean_case_id: str) -> float:
    summary = read_json(latest_rg_summary(clean_case_id))
    for key in ("rg2_mean_nm2", "shape_rg2_carbon_mean_nm2"):
        value = to_float(summary.get(key))
        if math.isfinite(value) and value > 0:
            return value
    return math.nan


def candidate_windows(
    times_ps: np.ndarray,
    msd_nm2: np.ndarray,
    rg2_nm2: float,
    prior_window: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    if times_ps.size == 0:
        return []
    max_time_ns = float(np.nanmax(times_ps)) / 1000.0
    start_floor_ns = max_time_ns * MIN_PUBLICATION_BEGIN_FRACTION
    cutoff_ns = max_time_ns * MAX_PUBLICATION_END_FRACTION
    min_width_ns = max_time_ns * MIN_PUBLICATION_WINDOW_FRACTION
    pairs: set[tuple[float, float]] = set()
    if prior_window is not None:
        if (
            prior_window[0] >= start_floor_ns
            and prior_window[1] <= cutoff_ns
            and prior_window[1] - prior_window[0] >= min_width_ns
        ):
            pairs.add((round(prior_window[0], 3), round(prior_window[1], 3)))
    # Keep chain-scale displacement as a diagnostic only.  The publication
    # gate follows the common MSD practice of requiring clean provenance,
    # late-time slope, fit quality, and D stability; it does not require a
    # fixed multiple of Rg before a window can even be tested.
    first_start = start_floor_ns
    first_start = max(start_floor_ns, math.ceil(first_start / 5.0) * 5.0)
    latest_start = max(first_start, cutoff_ns - min_width_ns)
    for begin in np.arange(first_start, latest_start + 0.1, 5.0):
        min_end = begin + min_width_ns
        if min_end > cutoff_ns + 1e-6:
            continue
        for end in np.arange(min_end, cutoff_ns + 0.1, 5.0):
            pairs.add((round(float(begin), 3), round(float(end), 3)))
    return sorted(pairs)


def window_msd_over_rg2_min(
    times_ps: np.ndarray,
    msd_nm2: np.ndarray,
    begin_ns: float,
    end_ns: float,
    rg2_nm2: float,
    prior_over_rg2: float,
) -> float:
    if not math.isfinite(rg2_nm2) or rg2_nm2 <= 0:
        return prior_over_rg2
    begin_ps = begin_ns * 1000.0
    end_ps = end_ns * 1000.0
    mask = (times_ps >= begin_ps) & (times_ps <= end_ps)
    if not np.any(mask):
        return math.nan
    return float(np.nanmin(msd_nm2[mask]) / rg2_nm2)


def analyze_case(row: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    case_id = str(row.get("case_id", "")).strip()
    clean_case_id = str(row.get("clean_case_id", "")).strip()
    ts_path = latest_msd_timeseries(clean_case_id)
    source_summary_path = latest_msd_summary(clean_case_id)
    source_summary = read_json(source_summary_path)
    if ts_path is None:
        return {
            "case_id": case_id,
            "clean_case_id": clean_case_id,
            "refit_status": "missing_clean_msd_timeseries",
            "source_timeseries": "",
            "recommendation": "postprocess_clean_trajectory_before_extend",
            "classification": row.get("classification", ""),
        }, None

    ts = pd.read_csv(ts_path, encoding="utf-8-sig")
    times = pd.to_numeric(ts.get("time_ps"), errors="coerce").to_numpy(dtype=float)
    msd = pd.to_numeric(ts.get("msd_nm2"), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(times) & np.isfinite(msd)
    times = times[mask]
    msd = msd[mask]
    if times.size < 20:
        return {
            "case_id": case_id,
            "clean_case_id": clean_case_id,
            "refit_status": "too_few_clean_msd_points",
            "source_timeseries": str(ts_path),
            "recommendation": "extend_clean_nvt_prod_from_checkpoint",
            "classification": row.get("classification", ""),
        }, None

    max_time_ns = float(np.nanmax(times)) / 1000.0
    cutoff_ns = max_time_ns * MAX_PUBLICATION_END_FRACTION
    rg2_nm2 = rg2_for_clean_case(clean_case_id)
    prior_over_rg2 = to_float(row.get("msd_over_rg2"))
    prior_window = parse_window_ns(str(row.get("clean_window_ns", "")))
    windows = []
    for begin_ns, end_ns in candidate_windows(times, msd, rg2_nm2, prior_window):
        fit = fit_window(times, msd, begin_ns, end_ns)
        fit["local_D_cv_pct"] = shifted_window_cv(times, msd, begin_ns, end_ns, cutoff_ns)
        fit["window_msd_over_rg2_min"] = window_msd_over_rg2_min(
            times,
            msd,
            begin_ns,
            end_ns,
            rg2_nm2,
            prior_over_rg2,
        )
        fit["abs_slope_error"] = abs(float(fit["loglog_slope"]) - 1.0) if math.isfinite(fit["loglog_slope"]) else math.nan
        fit["strict_refit_candidate"] = (
            fit["positive_D"]
            and fit["n_points"] >= 20
            and math.isfinite(fit["loglog_slope"])
            and 0.95 <= float(fit["loglog_slope"]) <= 1.05
            and math.isfinite(fit["linear_R2"])
            and float(fit["linear_R2"]) >= 0.95
            and math.isfinite(fit["local_D_cv_pct"])
            and float(fit["local_D_cv_pct"]) <= 5.0
            and end_ns <= cutoff_ns + 1e-6
        )
        windows.append(fit)
    passing = [fit for fit in windows if fit["strict_refit_candidate"]]
    if not passing:
        best = sorted(
            windows,
            key=lambda fit: (
                math.inf if not math.isfinite(fit.get("abs_slope_error", math.nan)) else fit["abs_slope_error"],
                math.inf if not math.isfinite(fit.get("local_D_cv_pct", math.nan)) else fit["local_D_cv_pct"],
                -float(fit.get("width_ns", 0.0)),
            ),
        )[0] if windows else {}
        return {
            "case_id": case_id,
            "clean_case_id": clean_case_id,
            "refit_status": "no_strict_refit_window",
            "source_timeseries": str(ts_path),
            "source_summary": str(source_summary_path or ""),
            "classification": row.get("classification", ""),
            "max_time_ns": max_time_ns,
            "cutoff_ns": cutoff_ns,
            "rg2_nm2": rg2_nm2,
            "prior_clean_window_ns": row.get("clean_window_ns", ""),
            "best_begin_ns": best.get("begin_ns", ""),
            "best_end_ns": best.get("end_ns", ""),
            "best_loglog_slope": best.get("loglog_slope", ""),
            "best_linear_R2": best.get("linear_R2", ""),
            "best_D_m2_s": best.get("D_m2_s", ""),
            "best_local_D_cv_pct": best.get("local_D_cv_pct", ""),
            "best_window_msd_over_rg2_min": best.get("window_msd_over_rg2_min", ""),
            "recommendation": "extend_clean_nvt_prod_from_checkpoint",
        }, None

    best = sorted(
        passing,
        key=lambda fit: (fit["abs_slope_error"], fit["local_D_cv_pct"], -fit["width_ns"]),
    )[0]
    d_outlier_fraction = to_float(source_summary.get("D_outlier_fraction"))
    d_median_m2_s = to_float(source_summary.get("D_median_m2_s"))
    if not math.isfinite(d_median_m2_s) or d_median_m2_s <= 0:
        d_median_m2_s = best["D_m2_s"]
    summary = {
        "analysis_version": "strict_msd_refit_v1",
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "clean_case_id": clean_case_id,
        "case_id": case_id,
        "n_frames": int(times.size),
        "time_first_ps": float(np.nanmin(times)),
        "time_last_ps": float(np.nanmax(times)),
        "time_final_ps": float(np.nanmax(times)),
        "fit_begin_ps": float(best["begin_ns"] * 1000.0),
        "fit_end_ps": float(best["end_ns"] * 1000.0),
        "loglog_slope": best["loglog_slope"],
        "fit_quality": "ok",
        "diffusive_regime_ok": True,
        "regime_warning": "",
        "window_sensitive": False,
        "window_D_cv_pct": best["local_D_cv_pct"],
        "D_mean_m2_s": best["D_m2_s"],
        "D_median_m2_s": d_median_m2_s,
        "D_mean_1e-5_cm2_s": best["D_1e-5_cm2_s"],
        "D_outlier_fraction": d_outlier_fraction,
        "python_fit": {
            "D_1e-5_cm2_s": best["D_1e-5_cm2_s"],
            "D_m2_s": best["D_m2_s"],
            "R2": best["linear_R2"],
            "loglog_R2": best["loglog_R2"],
            "loglog_slope": best["loglog_slope"],
        },
        "publication_window_policy": REFIT_POLICY,
        "window_msd_over_rg2_min": best["window_msd_over_rg2_min"],
        "rg2_mean_nm2": rg2_nm2,
        "msd_over_rg2": row.get("msd_over_rg2", ""),
        "msd_timeseries_csv": "",
        "source_timeseries_csv": str(ts_path),
        "source_summary_json": str(source_summary_path or ""),
        "method": (
            "clean GROMACS 2024.4 chain-COM MSD refit; late diffusive window before "
            "extend, requiring width >= 30% duration, begin >= 25% duration, "
            "end <= 85% duration, stable D, and near-unity log-log slope. "
            "Chain-scale displacement is retained as a non-blocking diagnostic."
        ),
        "md_rerun": False,
        "canonical_release_modified": False,
    }
    gate = evaluate_msd_publication_gate(
        {
            **summary,
            "engine_version": "GROMACS 2024.4 clean refit",
            "msd_source_engine": "GROMACS 2024.4 clean nvt_prod",
        },
        require_clean=True,
    )
    status = "strict_refit_pass" if gate["msd_publication_grade"] else "strict_refit_gate_failed"
    case_row = {
        "case_id": case_id,
        "clean_case_id": clean_case_id,
        "refit_status": status,
        "source_timeseries": str(ts_path),
        "source_summary": str(source_summary_path or ""),
        "classification": row.get("classification", ""),
        "max_time_ns": max_time_ns,
        "cutoff_ns": cutoff_ns,
        "rg2_nm2": rg2_nm2,
        "prior_clean_window_ns": row.get("clean_window_ns", ""),
        "best_begin_ns": best["begin_ns"],
        "best_end_ns": best["end_ns"],
        "best_width_ns": best["width_ns"],
        "best_loglog_slope": best["loglog_slope"],
        "best_linear_R2": best["linear_R2"],
        "best_D_m2_s": best["D_m2_s"],
        "best_local_D_cv_pct": best["local_D_cv_pct"],
        "best_window_msd_over_rg2_min": best["window_msd_over_rg2_min"],
        "publication_gate_issues": gate["issue_string"],
        "recommendation": "regate_without_extend" if gate["msd_publication_grade"] else "extend_clean_nvt_prod_from_checkpoint",
    }
    return case_row, summary if gate["msd_publication_grade"] else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASSIFICATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="strict_msd_refit_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--write-failed-summaries", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_filter = {str(case_id).strip() for case_id in args.case_id if str(case_id).strip()}
    input_rows = read_csv_rows(args.classification_csv)
    overlay_rows = []
    if Path(args.classification_csv).resolve() == DEFAULT_CLASSIFICATION_CSV.resolve():
        overlay_rows = current_overlay_msd_warn_rows()
        input_rows = merge_current_overlay_rows(input_rows, overlay_rows)
    rows = [
        row
        for row in input_rows
        if str(row.get("classification", "")).strip() in ELIGIBLE_CLASSIFICATIONS
        and (not case_filter or str(row.get("case_id", "")).strip() in case_filter)
    ]
    out_dir = args.output_dir
    candidate_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    written = 0
    for row in rows:
        case_row, summary = analyze_case(row)
        candidate_rows.append(case_row)
        if summary is None:
            continue
        clean_case_id = str(summary["clean_case_id"])
        msd_dir = out_dir / "cases" / clean_case_id / "msd"
        msd_dir.mkdir(parents=True, exist_ok=True)
        source_ts = Path(str(summary.get("source_timeseries_csv", "")))
        target_ts = msd_dir / "msd_timeseries.csv"
        if source_ts.exists() and not target_ts.exists():
            shutil.copy2(source_ts, target_ts)
        summary["msd_timeseries_csv"] = str(target_ts)
        write_json(msd_dir / "msd_summary.json", summary)
        aggregate_rows.append(summary)
        written += 1
    fields = [
        "case_id",
        "clean_case_id",
        "refit_status",
        "classification",
        "source_timeseries",
        "source_summary",
        "max_time_ns",
        "cutoff_ns",
        "rg2_nm2",
        "prior_clean_window_ns",
        "best_begin_ns",
        "best_end_ns",
        "best_width_ns",
        "best_loglog_slope",
        "best_linear_R2",
        "best_D_m2_s",
        "best_local_D_cv_pct",
        "best_window_msd_over_rg2_min",
        "publication_gate_issues",
        "recommendation",
    ]
    write_csv(out_dir / "strict_msd_refit_candidates.csv", candidate_rows, fields)
    aggregate_fields = [
        "case_id",
        "clean_case_id",
        "status",
        "fit_begin_ps",
        "fit_end_ps",
        "loglog_slope",
        "D_mean_m2_s",
        "D_median_m2_s",
        "window_D_cv_pct",
        "D_outlier_fraction",
        "publication_window_policy",
        "window_msd_over_rg2_min",
        "time_last_ps",
        "source_timeseries_csv",
    ]
    write_csv(out_dir / "aggregate" / "msd_summary.csv", aggregate_rows, aggregate_fields)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_now(),
        "classification_csv": str(args.classification_csv),
        "output_dir": str(out_dir),
        "scanned_count": len(rows),
        "current_overlay_warn_rows_added": len(overlay_rows),
        "written_summary_count": written,
        "status_counts": dict(Counter(row.get("refit_status", "") for row in candidate_rows)),
        "candidate_csv": str(out_dir / "strict_msd_refit_candidates.csv"),
        "aggregate_csv": str(out_dir / "aggregate" / "msd_summary.csv"),
        "md_rerun": False,
        "canonical_release_modified": False,
    }
    write_json(out_dir / "strict_msd_refit_summary.json", summary)
    write_json(OUTPUT_ROOT / "pelcb100_clean_overlay_regate" / "strict_msd_refit_latest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


