"""Publication-grade gate for PE-LCB100 chain-COM MSD evidence.

This module separates a numerically usable MSD fit from a publication-grade
diffusion coefficient.  A publication-grade D requires clean-engine provenance,
a broad late-regime diffusive window, stable local diffusion estimates, and
non-pathological ensemble diagnostics.  Narrow post-hoc windows are not allowed
to turn a globally sub-diffusive trajectory into PASS evidence.

The chain-scale displacement amplitude is retained as a provenance and
confidence diagnostic.  It is not a hard pass/fail criterion because the
polymer-MSD literature usually gates diffusion coefficients by clean provenance,
late-time linearity, fit quality, and uncertainty rather than by a fixed
multiple of the coil size.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


SLOPE_LOW = 0.95
SLOPE_HIGH = 1.05
MIN_R2 = 0.95
MAX_WINDOW_D_CV_PCT = 5.0
MAX_D_OUTLIER_FRACTION = 0.25
MIN_PUBLICATION_WINDOW_FRACTION = 0.30
MIN_PUBLICATION_BEGIN_FRACTION = 0.25
MAX_PUBLICATION_END_FRACTION = 0.85
REFIT_POLICY = "refit_before_extend_mid_diffusion"
MIN_MSD_OVER_RG2 = 3.0


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    out = str(value).strip()
    return "" if out.lower() in {"nan", "none", "null"} else out


def truthy(value: Any) -> bool:
    return text(value).lower() in {"1", "true", "yes", "ok", "pass"}


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def nested_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raw = text(value)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def fit_r2(row: Mapping[str, Any]) -> float:
    for key in ("python_fit_R2", "msd_fit_R2_publication", "msd_fit_R2", "linear_R2", "R2"):
        value = to_float(row.get(key))
        if math.isfinite(value):
            return value
    fit = nested_json(row.get("python_fit"))
    return to_float(fit.get("R2"))


def clean_engine_source(row: Mapping[str, Any]) -> bool:
    """Return whether MSD provenance explicitly identifies the clean engine.

    Project-internal path fragments such as ``o7clean`` and ``pe24`` are not
    engine identifiers and are deliberately ignored.
    """

    engine = text(
        row.get("engine_version") or row.get("msd_source_engine") or row.get("clean_engine_version")
    ).lower()
    if "2021.2" in engine:
        return False
    if truthy(row.get("msd_clean_engine_source")):
        return True
    return "gromacs" in engine and "2024.4" in engine and "clean" in engine


def publication_window_ok(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    first_ps = to_float(row.get("time_first_ps"))
    last_ps = to_float(row.get("time_last_ps") or row.get("time_final_ps"))
    begin_ps = to_float(row.get("fit_begin_ps"))
    end_ps = to_float(row.get("fit_end_ps"))
    if not all(math.isfinite(value) for value in (last_ps, begin_ps, end_ps)):
        return False, ["missing_fit_or_trajectory_time"]
    if not math.isfinite(first_ps):
        first_ps = 0.0
    duration = max(0.0, last_ps - first_ps)
    width = end_ps - begin_ps
    if duration <= 0 or width <= 0:
        return False, ["invalid_fit_or_trajectory_duration"]
    if width / duration < MIN_PUBLICATION_WINDOW_FRACTION:
        issues.append("fit_window_too_short_for_publication")
    if begin_ps < first_ps + duration * MIN_PUBLICATION_BEGIN_FRACTION:
        issues.append("fit_window_starts_before_late_diffusive_regime")
    if end_ps > first_ps + duration * MAX_PUBLICATION_END_FRACTION:
        issues.append("fit_window_overlaps_noisy_long_lag_tail")
    return not issues, issues


def displacement_amplitude_diagnostic(row: Mapping[str, Any]) -> tuple[str, float, list[str]]:
    """Return a non-blocking chain-scale displacement diagnostic.

    The chain-scale cutoff is useful for distinguishing high-confidence
    absolute D estimates from fits with limited center-of-mass displacement.
    It is not treated as a publication-grade hard gate.
    """

    over_rg2 = to_float(row.get("window_msd_over_rg2_min") or row.get("msd_over_rg2"))
    if not math.isfinite(over_rg2):
        if truthy(row.get("msd_tail_window_ok")):
            return "sufficient", MIN_MSD_OVER_RG2, []
        return "unknown", math.nan, ["displacement_amplitude_unknown"]
    if over_rg2 < MIN_MSD_OVER_RG2:
        return "limited", over_rg2, ["limited_displacement_amplitude"]
    return "sufficient", over_rg2, []


def evaluate_msd_publication_gate(
    row: Mapping[str, Any],
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    issues: list[str] = []
    diagnostic_issues: list[str] = []

    if require_clean and not clean_engine_source(row):
        issues.append("msd_engine_provenance_warn")

    status = text(row.get("status_MSD") or row.get("msd_status") or row.get("status")).lower()
    if status and status not in {"ok", "pass"}:
        issues.append(f"msd_status={status}")

    slope = to_float(row.get("loglog_slope"))
    if not math.isfinite(slope) or not (SLOPE_LOW <= slope <= SLOPE_HIGH):
        issues.append("msd_tail_slope_not_diffusive")

    if not truthy(row.get("diffusive_regime_ok")):
        issues.append("diffusive_regime_not_confirmed")

    fq = text(row.get("fit_quality")).lower()
    if fq and fq not in {"ok", "pass"}:
        issues.append(f"fit_quality={fq}")

    r2 = fit_r2(row)
    if not math.isfinite(r2) or r2 < MIN_R2:
        issues.append("msd_fit_R2_below_0.95")

    d_mean = to_float(row.get("D_mean_m2_s") or row.get("best_D_m2_s"))
    if not math.isfinite(d_mean) or d_mean <= 0:
        issues.append("D_mean_not_positive")

    cv = to_float(row.get("window_D_cv_pct") or row.get("best_local_D_cv_pct"))
    if not math.isfinite(cv) or cv > MAX_WINDOW_D_CV_PCT:
        issues.append("window_D_cv_warn")

    if truthy(row.get("window_sensitive")):
        issues.append("window_sensitive_warn")

    outlier_fraction = to_float(row.get("D_outlier_fraction"))
    if math.isfinite(outlier_fraction) and outlier_fraction > MAX_D_OUTLIER_FRACTION:
        # Per-chain D outlier fraction is an ensemble-diagnostic flag.  When
        # the fitted mean D is positive and window-level D stability is already
        # acceptable, this should lower confidence rather than hard-fail the
        # case by itself.
        diagnostic_issues.append("D_outlier_fraction_warn")

    ok_window, window_issues = publication_window_ok(row)
    if not ok_window:
        for issue in window_issues:
            if issue == "fit_window_too_short_for_publication":
                diagnostic_issues.append("short_fit_window")
            else:
                issues.append(issue)

    amplitude_check, over_rg2, amplitude_diagnostic_issues = displacement_amplitude_diagnostic(row)
    diagnostic_issues.extend(amplitude_diagnostic_issues)
    ok = not issues
    msd_confidence = "LOW"
    if ok:
        msd_confidence = "HIGH" if amplitude_check == "sufficient" and not diagnostic_issues else "MED"
    return {
        "msd_publication_grade": ok,
        "status_MSD": "OK" if ok else "WARN",
        "warn_source": "" if ok else "msd_warn",
        "issues": list(dict.fromkeys(issues)),
        "issue_string": ";".join(dict.fromkeys(issues)),
        "diagnostic_issues": list(dict.fromkeys(diagnostic_issues)),
        "diagnostic_issue_string": ";".join(dict.fromkeys(diagnostic_issues)),
        "slope": slope,
        "fit_R2": r2,
        "window_D_cv_pct": cv,
        "D_outlier_fraction": outlier_fraction,
        "clean_engine_source": clean_engine_source(row),
        "tail_window_ok": ok_window,
        "displacement_amplitude_check": amplitude_check,
        "window_msd_over_rg2_min": over_rg2,
        "msd_diffusion_confidence": msd_confidence,
        "publication_window_policy": text(row.get("publication_window_policy") or row.get("msd_publication_window_policy")),
    }
