#!/usr/bin/env python3
"""Convert intrachain nonbonded C-C curves to tail-normalized descriptors.

This migration keeps the same pair selection
(``intra_chain_nonbonded_cc``: same-chain C-C pairs with graph distance > 3)
but changes the reported long curve from the raw eligible-partner-density
normalization to a tail-normalized pair-distribution descriptor. The raw curve
is retained as ``g_intra_nonbonded_raw``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compute_intra_nonbonded_rdf import find_first_minimum_after_peak  # noqa: E402
from compute_full_rdf_suite import rebuild_long_csvs, rebuild_summary_from_case_jsons  # noqa: E402


METHOD = "tail-normalized intrachain nonbonded C-C pair-distribution from gmx traj coordinates + TPR bond graph"
NORMALIZATION = "tail-normalized intrachain pair-distribution descriptor; raw eligible-partner-density curve retained"
STATUS = "TAIL_NORMALIZED_1P5_2P0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def normalize_long_csv(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = read_csv(path)
    if not rows:
        return None

    r = np.asarray([safe_float(row.get("r_nm")) for row in rows], dtype=float)
    raw_values = []
    for row in rows:
        raw = safe_float(row.get("g_intra_nonbonded_raw"))
        if not math.isfinite(raw):
            raw = safe_float(row.get("g_intra_nonbonded"))
        raw_values.append(raw)
    raw_g = np.asarray(raw_values, dtype=float)
    finite = np.isfinite(r) & np.isfinite(raw_g)
    tail_mask = finite & (r >= 1.5) & (r <= 2.0)
    if int(np.count_nonzero(tail_mask)) == 0:
        return None
    raw_tail_mean = float(np.mean(raw_g[tail_mask]))
    raw_tail_std = float(np.std(raw_g[tail_mask]))
    if not math.isfinite(raw_tail_mean) or raw_tail_mean <= 0:
        return None

    norm_g = raw_g / raw_tail_mean
    fields = list(rows[0].keys())
    if "g_intra_nonbonded_raw" not in fields:
        fields.insert(fields.index("g_intra_nonbonded") + 1, "g_intra_nonbonded_raw")
    for row, norm, raw in zip(rows, norm_g, raw_g):
        row["g_intra_nonbonded"] = f"{norm:.10g}" if math.isfinite(float(norm)) else ""
        row["g_intra_nonbonded_raw"] = f"{raw:.10g}" if math.isfinite(float(raw)) else ""
    write_csv(path, rows, fields)

    peak_mask = finite & (r >= 0.2) & (r <= 1.2)
    peak_r = math.nan
    peak_g = math.nan
    if int(np.count_nonzero(peak_mask)) > 0:
        sub_idx = np.where(peak_mask)[0]
        peak_idx = int(sub_idx[int(np.argmax(norm_g[peak_mask]))])
        peak_r = float(r[peak_idx])
        peak_g = float(norm_g[peak_idx])
    first_min = find_first_minimum_after_peak(r, norm_g, peak_r)
    tail = norm_g[tail_mask]
    return {
        "first_peak_r_nm": peak_r,
        "first_peak_g": peak_g,
        "first_min_r_nm": first_min,
        "tail_g_mean": float(np.mean(tail)),
        "tail_g_std": float(np.std(tail)),
        "raw_tail_g_mean": raw_tail_mean,
        "raw_tail_g_std": raw_tail_std,
    }


def update_json(path: Path, stats: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["method"] = METHOD
    payload["normalization"] = NORMALIZATION
    payload["tail_normalization_status"] = STATUS
    payload["first_peak_r_nm_0p2_1p2"] = stats["first_peak_r_nm"]
    payload["first_peak_g_0p2_1p2"] = stats["first_peak_g"]
    payload["first_min_r_nm"] = stats["first_min_r_nm"]
    payload["tail_g_mean_1p5_2p0"] = stats["tail_g_mean"]
    payload["tail_g_std_1p5_2p0"] = stats["tail_g_std"]
    payload["raw_tail_g_mean_1p5_2p0"] = stats["raw_tail_g_mean"]
    payload["raw_tail_g_std_1p5_2p0"] = stats["raw_tail_g_std"]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def update_case_summary(path: Path, stats: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    changed = False
    for entry in entries:
        if entry.get("rdf_kind") != "intra_chain_nonbonded_cc":
            continue
        entry["method"] = METHOD
        entry["selection"] = "tail-normalized same-chain carbon-carbon pair distribution excluding graph-distance <= 3"
        entry["normalization"] = NORMALIZATION
        entry["tail_normalization_status"] = STATUS
        entry["first_peak_r_nm"] = stats["first_peak_r_nm"]
        entry["first_peak_g"] = stats["first_peak_g"]
        entry["first_min_r_nm"] = stats["first_min_r_nm"]
        entry["tail_g_mean"] = stats["tail_g_mean"]
        entry["tail_g_std"] = stats["tail_g_std"]
        entry["raw_tail_g_mean"] = stats["raw_tail_g_mean"]
        entry["raw_tail_g_std"] = stats["raw_tail_g_std"]
        changed = True
    if changed:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return changed


def normalize_case_dir(case_rdf_dir: Path) -> bool:
    stats = normalize_long_csv(case_rdf_dir / "rdf_cc_intra_nonbonded_long.csv")
    if not stats:
        return False
    update_json(case_rdf_dir / "rdf_intra_nonbonded_summary.json", stats)
    update_case_summary(case_rdf_dir / "rdf_summary.json", stats)
    return True


def normalize_case_tree(raw_root: Path) -> int:
    count = 0
    for case_rdf_dir in sorted(raw_root.glob("*/rdf")):
        if normalize_case_dir(case_rdf_dir):
            count += 1
    return count


def normalize_aggregate_only(root: Path) -> bool:
    stats_by_case: dict[str, dict[str, Any]] = {}
    long_path = root / "rdf_cc_intra_nonbonded_long.csv"
    rows = read_csv(long_path) if long_path.exists() else []
    if not rows:
        return False
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row.get("case_id", ""), []).append(row)
    all_fields = list(rows[0].keys())
    if "g_intra_nonbonded_raw" not in all_fields:
        all_fields.insert(all_fields.index("g_intra_nonbonded") + 1, "g_intra_nonbonded_raw")
    new_rows: list[dict[str, Any]] = []
    for case_id, case_rows in by_case.items():
        tmp = root / f".tmp_{case_id}_intra.csv"
        write_csv(tmp, case_rows, all_fields)
        stats = normalize_long_csv(tmp)
        if stats:
            stats_by_case[case_id] = stats
            new_rows.extend(read_csv(tmp))
        tmp.unlink(missing_ok=True)
    if not stats_by_case:
        return False
    write_csv(long_path, new_rows, all_fields)

    summary_path = root / "rdf_summary.csv"
    if summary_path.exists():
        summary_rows = read_csv(summary_path)
        fields = list(summary_rows[0].keys()) if summary_rows else []
        for field in ("normalization", "tail_normalization_status", "raw_tail_g_mean", "raw_tail_g_std"):
            if field not in fields:
                fields.append(field)
        for row in summary_rows:
            if row.get("rdf_kind") != "intra_chain_nonbonded_cc":
                continue
            stats = stats_by_case.get(row.get("case_id", ""))
            if not stats:
                continue
            row["method"] = METHOD
            row["selection"] = "tail-normalized same-chain carbon-carbon pair distribution excluding graph-distance <= 3"
            row["first_peak_r_nm"] = stats["first_peak_r_nm"]
            row["first_peak_g"] = stats["first_peak_g"]
            row["first_min_r_nm"] = stats["first_min_r_nm"]
            row["tail_g_mean"] = stats["tail_g_mean"]
            row["tail_g_std"] = stats["tail_g_std"]
            row["normalization"] = NORMALIZATION
            row["tail_normalization_status"] = STATUS
            row["raw_tail_g_mean"] = stats["raw_tail_g_mean"]
            row["raw_tail_g_std"] = stats["raw_tail_g_std"]
        write_csv(summary_path, summary_rows, fields)
    return True


def process_root(root: Path) -> tuple[int, str]:
    root = Path(root)
    if (root / "cases").exists():
        count = normalize_case_tree(root / "cases")
        rebuild_summary_from_case_jsons(root / "cases", root / "aggregate")
        rebuild_long_csvs(root / "cases", root / "aggregate")
        return count, "batch"
    if (root / "raw" / "cases").exists():
        count = normalize_case_tree(root / "raw" / "cases")
        rebuild_summary_from_case_jsons(root / "raw" / "cases", root / "summary")
        rebuild_long_csvs(root / "raw" / "cases", root / "summary")
        return count, "integrated"
    if (root / "rdf_summary.csv").exists() and (root / "rdf_cc_intra_nonbonded_long.csv").exists():
        ok = normalize_aggregate_only(root)
        return (1 if ok else 0), "aggregate-only"
    return 0, "unrecognized"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        count, kind = process_root(root)
        print(f"{root}: kind={kind} normalized_cases={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
