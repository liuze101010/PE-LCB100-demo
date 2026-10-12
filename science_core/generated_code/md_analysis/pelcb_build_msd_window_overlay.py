#!/usr/bin/env python3
"""Build a sidecar diagnostic overlay from strict MSD window candidates.

The overlay is intentionally separate from the PE-LCB100 v1 canonical release.
It records cases whose MSD repair dry-run found a strict-window candidate, but
it no longer promotes any row to publication-grade or ML-usable status. A local
window candidate can explain an MSD warning; it cannot close a clean
publication-grade gap without clean-trajectory provenance and the strict MSD
publication gate.
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

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
AGGREGATE_DIR = SCRIPT_DIR / "output" / "pelcb100_integrated" / "aggregate"
REPAIR_DIR = SCRIPT_DIR / "output" / "pelcb100_repair_loop"
DEFAULT_CANDIDATES = REPAIR_DIR / "msd_window_repair_20260528_check" / "msd_window_repair_candidates.csv"
DEFAULT_OUTPUT_ROOT = AGGREGATE_DIR / "overlays"
SCHEMA_VERSION = "pelcb_msd_window_overlay.v3"
DIAGNOSTIC_POLICY_NOTE = (
    "Diagnostic-only MSD-window overlay; local window candidates do not modify "
    "publication_grade, ml_usable, warning_reason, or case_status."
)


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def stringify(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def build_overlay(
    targets: pd.DataFrame,
    mask: pd.DataFrame,
    candidates: pd.DataFrame,
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if targets.empty or mask.empty:
        raise FileNotFoundError("ml_targets.csv and ml_quality_mask.csv must both be present")
    if candidates.empty:
        raise FileNotFoundError("MSD window-repair candidates table is empty")

    strict = candidates[candidates["repair_status"].astype(str) == "strict_pass_window_candidate"].copy()
    relaxed = candidates[candidates["repair_status"].astype(str) == "c100_relaxed_window_candidate"].copy()
    strict_by_case = {str(row["case_id"]): row.to_dict() for _, row in strict.iterrows()}
    relaxed_by_case = {str(row["case_id"]): row.to_dict() for _, row in relaxed.iterrows()}

    targets_v3 = pd.DataFrame(targets.to_numpy(dtype=object), columns=targets.columns).astype(object)
    mask_v3 = pd.DataFrame(mask.to_numpy(dtype=object), columns=mask.columns).astype(object)

    for col in ["D_mean_m2_s", "D_mean_1e-5_cm2_s", "loglog_slope"]:
        if col in targets_v3.columns and f"v1_{col}" not in targets_v3.columns:
            targets_v3[f"v1_{col}"] = targets_v3[col]
    for col in ["publication_grade", "ml_usable", "warning_reason", "case_status"]:
        if col in mask_v3.columns and f"v1_{col}" not in mask_v3.columns:
            mask_v3[f"v1_{col}"] = mask_v3[col]

    target_overlay_cols = {
        "msd_window_overlay_applied": False,
        "msd_window_overlay_status": "not_applicable",
        "msd_window_overlay_source_tag": "",
        "msd_window_overlay_fit_window_ns": "",
        "msd_window_overlay_loglog_slope": "",
        "msd_window_overlay_D_m2_s": "",
        "msd_window_overlay_local_D_cv_pct": "",
        "msd_window_overlay_policy": "diagnostic_only",
    }
    mask_overlay_cols = {
        "msd_window_overlay_applied": False,
        "msd_window_overlay_status": "not_applicable",
        "msd_window_overlay_source_tag": "",
        "msd_window_overlay_fit_window_ns": "",
        "msd_window_overlay_note": DIAGNOSTIC_POLICY_NOTE,
        "msd_window_overlay_policy": "diagnostic_only",
        "publication_grade_modified_by_msd_window_overlay": False,
    }
    for col, default in target_overlay_cols.items():
        targets_v3[col] = pd.Series([default] * len(targets_v3), dtype=object)
    for col, default in mask_overlay_cols.items():
        mask_v3[col] = pd.Series([default] * len(mask_v3), dtype=object)

    changes: list[dict[str, Any]] = []
    for idx, row in targets_v3.iterrows():
        case_id = str(row["case_id"])
        if case_id not in strict_by_case and case_id not in relaxed_by_case:
            continue
        candidate = strict_by_case.get(case_id) or relaxed_by_case.get(case_id)
        status = str(candidate.get("repair_status", ""))
        fit_window = f"{candidate.get('best_begin_ns', '')}-{candidate.get('best_end_ns', '')}"
        targets_v3.at[idx, "msd_window_overlay_status"] = status
        targets_v3.at[idx, "msd_window_overlay_source_tag"] = tag
        targets_v3.at[idx, "msd_window_overlay_fit_window_ns"] = fit_window
        targets_v3.at[idx, "msd_window_overlay_loglog_slope"] = candidate.get("best_loglog_slope", "")
        targets_v3.at[idx, "msd_window_overlay_D_m2_s"] = candidate.get("best_D_m2_s", "")
        targets_v3.at[idx, "msd_window_overlay_local_D_cv_pct"] = candidate.get("best_local_D_cv_pct", "")

        mask_idx = mask_v3.index[mask_v3["case_id"].astype(str) == case_id].tolist()
        if mask_idx:
            mi = mask_idx[0]
            mask_v3.at[mi, "msd_window_overlay_status"] = status
            mask_v3.at[mi, "msd_window_overlay_source_tag"] = tag
            mask_v3.at[mi, "msd_window_overlay_fit_window_ns"] = fit_window
        if case_id not in strict_by_case:
            if mask_idx:
                mask_v3.at[mask_idx[0], "msd_window_overlay_note"] = (
                    "C100-relaxed candidate only; " + DIAGNOSTIC_POLICY_NOTE
                )
            continue

        before = {
            "D_mean_m2_s": row.get("D_mean_m2_s", ""),
            "D_mean_1e-5_cm2_s": row.get("D_mean_1e-5_cm2_s", ""),
            "loglog_slope": row.get("loglog_slope", ""),
        }
        new_d_m2 = finite_float(candidate.get("best_D_m2_s"))
        new_slope = finite_float(candidate.get("best_loglog_slope"))
        if new_d_m2 is None or new_slope is None:
            continue
        targets_v3.at[idx, "D_mean_m2_s"] = new_d_m2
        targets_v3.at[idx, "D_mean_1e-5_cm2_s"] = new_d_m2 / 1.0e-9
        targets_v3.at[idx, "loglog_slope"] = new_slope
        targets_v3.at[idx, "msd_window_overlay_applied"] = True

        if mask_idx:
            mi = mask_idx[0]
            mask_v3.at[mi, "msd_window_overlay_applied"] = True
            mask_v3.at[mi, "msd_window_overlay_note"] = (
                "Strict postprocessing-window candidate recorded. " + DIAGNOSTIC_POLICY_NOTE
            )

        changes.append(
            {
                "case_id": case_id,
                "overlay_status": status,
                "fit_window_ns": fit_window,
                "v1_D_mean_m2_s": before["D_mean_m2_s"],
                "overlay_D_mean_m2_s": new_d_m2,
                "v1_loglog_slope": before["loglog_slope"],
                "overlay_loglog_slope": new_slope,
                "local_D_cv_pct": candidate.get("best_local_D_cv_pct", ""),
                "release_table_action": "diagnostic_sidecar_only",
                "publication_grade_modified": False,
            }
        )
    return targets_v3, mask_v3, changes


def write_report(path: Path, targets_v3: pd.DataFrame, mask_v3: pd.DataFrame, changes: list[dict[str, Any]]) -> None:
    usable = int(mask_v3["ml_usable"].map(boolish).sum()) if "ml_usable" in mask_v3.columns else 0
    case_counts = mask_v3["case_status"].fillna("missing").value_counts().to_dict() if "case_status" in mask_v3.columns else {}
    overlay_counts = Counter(mask_v3.get("msd_window_overlay_status", pd.Series(dtype=str)).astype(str))
    lines = [
        "# PE-LCB100 MSD-window ML overlay v3",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Rows: targets `{len(targets_v3)}`, quality mask `{len(mask_v3)}`",
        f"- Strict MSD-window diagnostic records: `{len(changes)}`",
        f"- Input/unchanged `ml_usable=True`: `{usable}`",
        "- PE-LCB100 v1 canonical tables are not overwritten by this overlay.",
        "- This overlay is diagnostic-only and never modifies `publication_grade`, `ml_usable`, `warning_reason`, or `case_status`.",
        "",
        "## Overlay status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(overlay_counts.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Case status counts", "", "| Case status | Count |", "|---|---:|"])
    for key, value in sorted(case_counts.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- `strict_pass_window_candidate` rows are recorded as diagnostic sidecar values only.",
            "- `c100_relaxed_window_candidate` rows are annotated but remain non-promoted.",
            "- The overlay may record alternate MSD target values in sidecar target files for review.",
            "- The overlay does not update ML-usability flags or publication-grade status.",
            "- Publication-grade MSD closure requires clean-trajectory provenance and the strict MSD publication gate.",
            "- Canonical v1 release, locked labels, rule thresholds, and source trajectories are unchanged.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=AGGREGATE_DIR / "ml_targets.csv")
    parser.add_argument("--mask", type=Path, default=AGGREGATE_DIR / "ml_quality_mask.csv")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="msd_window_overlay_v3_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    targets = read_csv(args.targets)
    mask = read_csv(args.mask)
    candidates = read_csv(args.candidates)
    targets_v3, mask_v3, changes = build_overlay(targets, mask, candidates, tag=args.tag)

    targets_path = run_dir / "ml_targets_msd_window_overlay_v3.csv"
    mask_path = run_dir / "ml_quality_mask_msd_window_overlay_v3.csv"
    changes_path = run_dir / "msd_window_overlay_changes.csv"
    report_path = run_dir / "msd_window_overlay_report.md"
    targets_v3.to_csv(targets_path, index=False, encoding="utf-8-sig")
    mask_v3.to_csv(mask_path, index=False, encoding="utf-8-sig")
    write_csv(changes_path, changes, list(changes[0].keys()) if changes else ["case_id"])
    write_report(report_path, targets_v3, mask_v3, changes)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_now(),
        "tag": args.tag,
        "mode": "sidecar_overlay",
        "canonical_release_modified": False,
        "md_rerun": False,
        "rows": {
            "targets": int(len(targets_v3)),
            "quality_mask": int(len(mask_v3)),
        },
        "strict_overlay_applied_count": 0,
        "strict_diagnostic_record_count": int(len(changes)),
        "publication_grade_modified_by_overlay": False,
        "ml_usable_modified_by_overlay": False,
        "ml_usable_count": int(mask_v3["ml_usable"].map(boolish).sum()) if "ml_usable" in mask_v3.columns else None,
        "case_status_counts": mask_v3["case_status"].fillna("missing").value_counts().to_dict()
        if "case_status" in mask_v3.columns
        else {},
        "targets_csv": str(targets_path),
        "quality_mask_csv": str(mask_path),
        "changes_csv": str(changes_path),
        "report_md": str(report_path),
        "source_candidates_csv": str(args.candidates),
    }
    write_json(run_dir / "msd_window_overlay_summary.json", summary)
    write_json(args.output_root / "msd_window_overlay_latest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
