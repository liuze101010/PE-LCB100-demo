#!/usr/bin/env python3
"""Verify that the RadonPy-only density rule preserves PE-LCB100 outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from radonpy_density_screening import load_case_match_pack, load_reference_pack


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_ROOT = ROOT.parent / "research-vault" / "40 Drafts" / "pelcb100_paper"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "output"
    / "radonpy_density_screening"
    / "density_rule_regression_report.json"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference_pack = load_reference_pack()
    case_match_pack = load_case_match_pack()
    pass_low, pass_high = reference_pack["density_range"]["pass_range_g_cm3"]

    paired_path = args.paper_root / "figures" / "fig5_density_warn_paired.csv"
    paired_rows = _read_csv(paired_path)
    before = [float(row["density_before_g_cm3"]) for row in paired_rows]
    after = [float(row["density_after_g_cm3"]) for row in paired_rows]
    initial_warn_count = sum(value < pass_low or value > pass_high for value in before)
    repaired_pass_count = sum(pass_low <= value <= pass_high for value in after)

    ledger_path = args.paper_root / "figures" / "fig4_closed_loop_case_ledger.csv"
    ledger_rows = _read_csv(ledger_path)
    final_pass_count = sum(
        str(row.get("effective_publication_grade") or "").strip().lower() == "true"
        for row in ledger_rows
    )
    final_review_count = len(ledger_rows) - final_pass_count

    assert len(reference_pack["records"]) == 40
    assert reference_pack["density_range"]["raw_range_g_cm3"] == [
        0.7315756468839724,
        0.866582790659682,
    ]
    assert [pass_low, pass_high] == [0.73, 0.87]
    assert case_match_pack["case_count"] == 100
    assert len(paired_rows) == 77
    assert initial_warn_count == 77
    assert repaired_pass_count == 77
    assert (final_pass_count, final_review_count) == (92, 8)

    report = {
        "status": "PASS",
        "reference_count": len(reference_pack["records"]),
        "raw_density_range_g_cm3": reference_pack["density_range"]["raw_range_g_cm3"],
        "active_density_range_g_cm3": [pass_low, pass_high],
        "case_structure_match_count": case_match_pack["case_count"],
        "structure_similarity_tier_counts": case_match_pack["tier_counts"],
        "initial_density_warn_count": initial_warn_count,
        "repaired_density_pass_count": repaired_pass_count,
        "final_pass_count": final_pass_count,
        "final_review_count": final_review_count,
        "inputs": {
            "paired_density_csv": str(paired_path),
            "final_case_ledger_csv": str(ledger_path),
            "radonpy_source_sha256": reference_pack["source"]["sha256"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
