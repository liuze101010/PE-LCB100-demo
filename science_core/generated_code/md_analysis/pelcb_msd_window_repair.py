#!/usr/bin/env python3
"""Dry-run C100-aware MSD fit-window repair planner.

This script reads existing chain-COM MSD time series and evaluates alternative
fit windows for cases currently routed as MSD-WARN.  It never reruns MD and
does not edit canonical release tables.  Outputs are sidecar diagnostics that
can later support a separately frozen postprocessing-rule update.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
AGGREGATE_DIR = SCRIPT_DIR / "output" / "pelcb100_integrated" / "aggregate"
DEFAULT_MSD_ROOT = SCRIPT_DIR / "output" / "pelcb100_msd_20260523"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "pelcb100_repair_loop"
SCHEMA_VERSION = "pelcb_msd_window_repair.v1"
MIN_PUBLICATION_TAIL_FRACTION = 0.30
MAX_PUBLICATION_TAIL_GAP_FRACTION = 0.05


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


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
    n_points = int(np.count_nonzero(mask))
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
    slope = log_fit["slope"]
    return {
        "begin_ns": float(begin_ns),
        "end_ns": float(end_ns),
        "width_ns": float(end_ns - begin_ns),
        "n_points": n_points,
        "D_1e-5_cm2_s": float(d_1e5),
        "D_m2_s": float(d_m2_s),
        "linear_R2": float(lin["r2"]),
        "loglog_slope": float(slope),
        "loglog_R2": float(log_fit["r2"]),
        "strict_slope_ok": bool(math.isfinite(slope) and 0.95 <= slope <= 1.05),
        "c100_relaxed_slope_ok": bool(math.isfinite(slope) and 0.85 <= slope <= 1.05),
        "positive_D": bool(math.isfinite(d_1e5) and d_1e5 > 0),
    }


def local_window_cv(
    times_ps: np.ndarray,
    msd_nm2: np.ndarray,
    begin_ns: float,
    end_ns: float,
    shift_ns: float = 20.0,
) -> float:
    windows = [
        (begin_ns, end_ns),
        (begin_ns + shift_ns, end_ns + shift_ns),
        (max(0.0, begin_ns - shift_ns), max(0.0, end_ns - shift_ns)),
    ]
    values: list[float] = []
    max_time_ns = float(np.nanmax(times_ps)) / 1000.0 if times_ps.size else 0.0
    for left, right in windows:
        if right <= left or right > max_time_ns:
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


def candidate_windows() -> list[tuple[float, float]]:
    starts = [40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 180.0, 200.0]
    ends = [250.0, 300.0, 320.0, 350.0, 360.0, 380.0, 400.0]
    presets = [(40.0, 360.0), (80.0, 320.0), (100.0, 300.0), (150.0, 350.0), (200.0, 400.0)]
    pairs = set(presets)
    for start in starts:
        for end in ends:
            if end - start >= 100.0:
                pairs.add((start, end))
    return sorted(pairs)


def choose_best(rows: list[dict[str, Any]], strict_cv_pct: float, relaxed_cv_pct: float) -> dict[str, Any] | None:
    valid = [row for row in rows if row["positive_D"] and row["n_points"] >= 20 and math.isfinite(row["loglog_slope"])]
    if not valid:
        return None
    for row in valid:
        row["abs_slope_error"] = abs(float(row["loglog_slope"]) - 1.0)
        row["strict_pass_candidate"] = (
            row["strict_slope_ok"]
            and row["positive_D"]
            and float(row["linear_R2"]) >= 0.95
            and float(row["local_D_cv_pct"]) <= strict_cv_pct
        )
        row["c100_relaxed_candidate"] = (
            row["c100_relaxed_slope_ok"]
            and row["positive_D"]
            and float(row["linear_R2"]) >= 0.95
            and float(row["local_D_cv_pct"]) <= relaxed_cv_pct
        )
    strict = [row for row in valid if row["strict_pass_candidate"]]
    if strict:
        return sorted(strict, key=lambda row: (row["abs_slope_error"], row["local_D_cv_pct"], -row["width_ns"]))[0]
    relaxed = [row for row in valid if row["c100_relaxed_candidate"]]
    if relaxed:
        return sorted(relaxed, key=lambda row: (row["abs_slope_error"], row["local_D_cv_pct"], -row["width_ns"]))[0]
    return sorted(valid, key=lambda row: (row["abs_slope_error"], -float(row["linear_R2"]), row["local_D_cv_pct"]))[0]


def original_diffusive_ok(original: dict[str, Any]) -> bool:
    warning = str(original.get("regime_warning", "") or "").strip().lower()
    return (
        truthy(original.get("diffusive_regime_ok"))
        and not truthy(original.get("window_sensitive"))
        and warning in {"", "ok", "pass"}
    )


def broad_tail_candidate(best: dict[str, Any], max_time_ns: float) -> bool:
    if not max_time_ns or not math.isfinite(max_time_ns) or max_time_ns <= 0:
        return False
    begin_ns = to_float(best.get("begin_ns"))
    end_ns = to_float(best.get("end_ns"))
    width_ns = to_float(best.get("width_ns"))
    if not all(math.isfinite(value) for value in (begin_ns, end_ns, width_ns)):
        return False
    if width_ns / max_time_ns < MIN_PUBLICATION_TAIL_FRACTION:
        return False
    max_gap_ns = max(1.0, max_time_ns * MAX_PUBLICATION_TAIL_GAP_FRACTION)
    return max_time_ns - end_ns <= max_gap_ns


def warn_case_ids(mask: pd.DataFrame, explicit_case_ids: list[str]) -> list[str]:
    if explicit_case_ids:
        return explicit_case_ids
    if mask.empty:
        return []
    rows = mask[
        (mask.get("case_status", "").astype(str) == "completed_msd_warn")
        | mask.get("warning_reason", "").astype(str).str.contains("MSD:WARN", na=False)
    ]
    return sorted(rows["case_id"].dropna().astype(str).unique().tolist())


def analyze_case(
    case_id: str,
    msd_root: Path,
    summary_by_case: dict[str, dict[str, Any]],
    strict_cv_pct: float,
    relaxed_cv_pct: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ts_path = msd_root / "cases" / case_id / "msd" / "msd_timeseries.csv"
    original = summary_by_case.get(case_id, {})
    if not ts_path.exists():
        row = {
            "case_id": case_id,
            "repair_status": "missing_timeseries",
            "recommendation": "cannot_plan_from_local_outputs",
            "timeseries_path": str(ts_path),
            "original_loglog_slope": original.get("loglog_slope", ""),
            "original_fit_quality": original.get("fit_quality", ""),
            "original_window_sensitive": original.get("window_sensitive", ""),
        }
        return row, []
    ts = read_csv(ts_path)
    times = pd.to_numeric(ts.get("time_ps"), errors="coerce").to_numpy(dtype=float)
    msd = pd.to_numeric(ts.get("msd_nm2"), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(times) & np.isfinite(msd)
    times = times[mask]
    msd = msd[mask]
    max_time_ns = float(np.nanmax(times)) / 1000.0 if times.size else math.nan
    window_rows: list[dict[str, Any]] = []
    for begin_ns, end_ns in candidate_windows():
        fit = fit_window(times, msd, begin_ns, end_ns)
        fit["case_id"] = case_id
        fit["local_D_cv_pct"] = local_window_cv(times, msd, begin_ns, end_ns)
        window_rows.append(fit)
    best = choose_best(window_rows, strict_cv_pct=strict_cv_pct, relaxed_cv_pct=relaxed_cv_pct)
    if best is None:
        row = {
            "case_id": case_id,
            "repair_status": "no_valid_window",
            "recommendation": "retain_msd_warn",
            "timeseries_path": str(ts_path),
            "original_loglog_slope": original.get("loglog_slope", ""),
            "original_fit_quality": original.get("fit_quality", ""),
            "original_window_sensitive": original.get("window_sensitive", ""),
        }
        return row, window_rows
    globally_diffusive = original_diffusive_ok(original)
    broad_tail = broad_tail_candidate(best, max_time_ns)
    publication_window_candidate = bool(best.get("strict_pass_candidate") and globally_diffusive and broad_tail)
    if publication_window_candidate:
        repair_status = "publication_tail_window_candidate"
        recommendation = "review_only_recompute_same_clean_tail_window_then_regate"
    elif best.get("strict_pass_candidate"):
        repair_status = "local_strict_window_only"
        recommendation = "do_not_promote_extend_clean_nvt_prod_to_true_diffusion"
    elif best.get("c100_relaxed_candidate"):
        repair_status = "c100_relaxed_window_candidate"
        recommendation = "do_not_promote_extend_clean_nvt_prod_to_true_diffusion"
    else:
        repair_status = "no_pass_window_candidate"
        recommendation = "retain_msd_warn_or_extend_clean_nvt_prod"
    row = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "repair_status": repair_status,
        "recommendation": recommendation,
        "timeseries_path": str(ts_path),
        "original_fit_begin_ps": original.get("fit_begin_ps", ""),
        "original_fit_end_ps": original.get("fit_end_ps", ""),
        "original_loglog_slope": original.get("loglog_slope", ""),
        "original_fit_quality": original.get("fit_quality", ""),
        "original_regime_warning": original.get("regime_warning", ""),
        "original_window_sensitive": original.get("window_sensitive", ""),
        "original_diffusive_regime_ok": original.get("diffusive_regime_ok", ""),
        "global_diffusive_ok": globally_diffusive,
        "max_time_ns": max_time_ns,
        "best_begin_ns": best.get("begin_ns", ""),
        "best_end_ns": best.get("end_ns", ""),
        "best_width_ns": best.get("width_ns", ""),
        "best_is_broad_tail_window": broad_tail,
        "best_loglog_slope": best.get("loglog_slope", ""),
        "best_loglog_R2": best.get("loglog_R2", ""),
        "best_linear_R2": best.get("linear_R2", ""),
        "best_D_m2_s": best.get("D_m2_s", ""),
        "best_local_D_cv_pct": best.get("local_D_cv_pct", ""),
        "strict_slope_ok": best.get("strict_slope_ok", False),
        "c100_relaxed_slope_ok": best.get("c100_relaxed_slope_ok", False),
        "strict_pass_candidate": False,
        "local_strict_pass_candidate": best.get("strict_pass_candidate", False),
        "publication_tail_window_candidate": publication_window_candidate,
        "c100_relaxed_candidate": best.get("c100_relaxed_candidate", False),
    }
    return row, window_rows


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts = Counter(row.get("repair_status", "") for row in rows)
    lines = [
        "# PE-LCB100 MSD window-repair dry run",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Cases scanned: `{len(rows)}`",
        f"- MSD root: `{args.msd_root}`",
        "- No MD trajectories were rerun and no canonical release tables were edited.",
        "",
        "## Status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- `publication_tail_window_candidate` is review-only evidence: the original trajectory must already be globally diffusive, and the candidate must be a broad tail window.",
            "- `local_strict_window_only` means a narrow/local segment can be fitted with slope near 1, but the whole trajectory is not publication-grade diffusion evidence.",
            "- `c100_relaxed_window_candidate` and `no_pass_window_candidate` remain MSD-WARN; the publication-grade repair is longer clean nvt_prod plus re-gate.",
            "",
            "## Top candidates",
            "",
            "| Case | Status | Original slope | Best window ns | Best slope | Local D CV % | Recommendation |",
            "|---|---|---:|---|---:|---:|---|",
        ]
    )
    order = {
        "publication_tail_window_candidate": 0,
        "local_strict_window_only": 1,
        "c100_relaxed_window_candidate": 2,
        "no_pass_window_candidate": 3,
    }
    for row in sorted(rows, key=lambda item: (order.get(item.get("repair_status", ""), 9), item.get("case_id", "")))[:40]:
        window = f"{row.get('best_begin_ns', '')}-{row.get('best_end_ns', '')}"
        lines.append(
            f"| `{row.get('case_id', '')}` | `{row.get('repair_status', '')}` | "
            f"{row.get('original_loglog_slope', '')} | `{window}` | {row.get('best_loglog_slope', '')} | "
            f"{row.get('best_local_D_cv_pct', '')} | {row.get('recommendation', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=AGGREGATE_DIR / "ml_quality_mask.csv")
    parser.add_argument("--msd-root", type=Path, default=DEFAULT_MSD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="msd_window_repair_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--case-id", action="append", default=[], help="Restrict to one case; repeatable.")
    parser.add_argument("--strict-cv-pct", type=float, default=5.0)
    parser.add_argument("--relaxed-cv-pct", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_dir / args.tag
    mask = read_csv(args.mask)
    summary = read_csv(args.msd_root / "aggregate" / "msd_summary.csv")
    summary_by_case = {
        str(row["case_id"]): row.to_dict()
        for _, row in summary.iterrows()
        if "case_id" in summary.columns
    }
    cases = warn_case_ids(mask, args.case_id)
    case_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for case_id in cases:
        case_row, windows = analyze_case(
            case_id,
            args.msd_root,
            summary_by_case,
            strict_cv_pct=float(args.strict_cv_pct),
            relaxed_cv_pct=float(args.relaxed_cv_pct),
        )
        case_rows.append(case_row)
        window_rows.extend(windows)

    case_fields = [
        "schema_version",
        "case_id",
        "repair_status",
        "recommendation",
        "timeseries_path",
        "original_fit_begin_ps",
        "original_fit_end_ps",
        "original_loglog_slope",
        "original_fit_quality",
        "original_regime_warning",
        "original_window_sensitive",
        "original_diffusive_regime_ok",
        "global_diffusive_ok",
        "max_time_ns",
        "best_begin_ns",
        "best_end_ns",
        "best_width_ns",
        "best_is_broad_tail_window",
        "best_loglog_slope",
        "best_loglog_R2",
        "best_linear_R2",
        "best_D_m2_s",
        "best_local_D_cv_pct",
        "strict_slope_ok",
        "c100_relaxed_slope_ok",
        "strict_pass_candidate",
        "local_strict_pass_candidate",
        "publication_tail_window_candidate",
        "c100_relaxed_candidate",
    ]
    window_fields = [
        "case_id",
        "begin_ns",
        "end_ns",
        "width_ns",
        "n_points",
        "D_1e-5_cm2_s",
        "D_m2_s",
        "linear_R2",
        "loglog_slope",
        "loglog_R2",
        "local_D_cv_pct",
        "strict_slope_ok",
        "c100_relaxed_slope_ok",
        "positive_D",
        "abs_slope_error",
        "strict_pass_candidate",
        "c100_relaxed_candidate",
    ]
    write_csv(run_dir / "msd_window_repair_candidates.csv", case_rows, case_fields)
    write_csv(run_dir / "msd_window_repair_all_windows.csv", window_rows, window_fields)
    write_report(run_dir / "msd_window_repair_report.md", case_rows, args)
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_now(),
        "mode": "dry-run",
        "case_count": len(case_rows),
        "status_counts": dict(Counter(row.get("repair_status", "") for row in case_rows)),
        "candidate_csv": str(run_dir / "msd_window_repair_candidates.csv"),
        "all_windows_csv": str(run_dir / "msd_window_repair_all_windows.csv"),
        "report_md": str(run_dir / "msd_window_repair_report.md"),
        "canonical_release_modified": False,
        "md_rerun": False,
    }
    write_json(run_dir / "msd_window_repair_summary.json", summary_payload)
    write_json(args.output_dir / "msd_window_repair_latest.json", summary_payload)
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
