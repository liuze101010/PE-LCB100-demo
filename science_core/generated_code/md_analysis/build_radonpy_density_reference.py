#!/usr/bin/env python3
"""Build the frozen RadonPy density reference pack and PE-LCB case matches."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from radonpy_density_screening import (
    DEFAULT_CASE_MATCH_CSV,
    DEFAULT_CASE_MATCH_PACK,
    DEFAULT_REFERENCE_PACK,
    DEFAULT_REFERENCE_CSV,
    EXPECTED_REFERENCE_COUNT,
    RADONPY_PI1070_URL,
    RAW_HIGH_G_CM3,
    RAW_LOW_G_CM3,
    build_case_match_pack,
    build_reference_pack,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "generated_code"
    / "emc_alpha_olefin_builder"
    / "output"
    / "screen_pelcb100"
    / "manifest.csv"
)
DEFAULT_CASE_MATCHES = (
    DEFAULT_CASE_MATCH_PACK
)


def _read_source(source: str) -> bytes:
    local_path = Path(source)
    if local_path.is_file():
        return local_path.read_bytes()
    with urllib.request.urlopen(source, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=RADONPY_PI1070_URL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-pack", type=Path, default=DEFAULT_REFERENCE_PACK)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--case-matches", type=Path, default=DEFAULT_CASE_MATCHES)
    parser.add_argument("--case-matches-csv", type=Path, default=DEFAULT_CASE_MATCH_CSV)
    args = parser.parse_args()

    source_bytes = _read_source(args.source)
    downloaded_at = datetime.now(timezone.utc).isoformat()
    reference_pack = build_reference_pack(
        source_bytes,
        downloaded_at_utc=downloaded_at,
        source_url=args.source,
    )
    if len(reference_pack["records"]) != EXPECTED_REFERENCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_REFERENCE_COUNT} RadonPy records, "
            f"found {len(reference_pack['records'])}."
        )
    raw_low, raw_high = reference_pack["density_range"]["raw_range_g_cm3"]
    if abs(raw_low - RAW_LOW_G_CM3) > 1e-12 or abs(raw_high - RAW_HIGH_G_CM3) > 1e-12:
        raise RuntimeError(
            "The RadonPy density envelope changed. Review the source before updating the active profile."
        )

    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    case_match_pack = build_case_match_pack(manifest_rows, reference_pack)
    if case_match_pack["case_count"] != 100:
        raise RuntimeError(f"Expected 100 PE-LCB cases, found {case_match_pack['case_count']}.")

    args.reference_pack.parent.mkdir(parents=True, exist_ok=True)
    args.reference_csv.parent.mkdir(parents=True, exist_ok=True)
    args.case_matches.parent.mkdir(parents=True, exist_ok=True)
    args.case_matches_csv.parent.mkdir(parents=True, exist_ok=True)
    args.reference_pack.write_text(
        json.dumps(reference_pack, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reference_rows = reference_pack["records"]
    with args.reference_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)
    args.case_matches.write_text(
        json.dumps(case_match_pack, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_rows = case_match_pack["matches"]
    with args.case_matches_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        json.dumps(
            {
                "reference_pack": str(args.reference_pack),
                "reference_csv": str(args.reference_csv),
                "case_matches": str(args.case_matches),
                "case_matches_csv": str(args.case_matches_csv),
                "reference_count": len(reference_pack["records"]),
                "density_range": reference_pack["density_range"],
                "tier_counts": case_match_pack["tier_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
