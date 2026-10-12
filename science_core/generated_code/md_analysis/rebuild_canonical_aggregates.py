#!/usr/bin/env python3
"""Rebuild canonical v2 aggregate CSV files from per-case JSON summaries."""

import argparse
import csv
import json
from pathlib import Path

try:
    from canonical_outputs import DYNAMICS_FIELDS, MSD_FIELDS, RDF_FIELDS, REE_FIELDS, RG_FIELDS, aggregate_dir, batch_dir, default_output_root
except ImportError:  # pragma: no cover
    from .canonical_outputs import DYNAMICS_FIELDS, MSD_FIELDS, RDF_FIELDS, REE_FIELDS, RG_FIELDS, aggregate_dir, batch_dir, default_output_root  # type: ignore


SPECS = {
    "rg": ("rg_summary.json", "rg_summary.csv", RG_FIELDS),
    "msd": ("msd_summary.json", "msd_summary.csv", MSD_FIELDS),
    "ree": ("ree_summary.json", "ree_summary.csv", REE_FIELDS),
    "rdf": ("rdf_summary.json", "rdf_summary.csv", RDF_FIELDS),
    "dynamics": ("dynamics_summary.json", "dynamics_summary.csv", DYNAMICS_FIELDS),
}

RDF_LONG_FILES = [
    "rdf_cc_all_long.csv",
    "rdf_cc_excl_long.csv",
    "rdf_cc_inter_long.csv",
    "rdf_cc_intra_nonbonded_long.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild md_analysis canonical aggregate CSVs.")
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--date-tag", default="20260501")
    parser.add_argument("--observables", default="rg,msd,ree,rdf")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def case_id_from_summary(path, payload):
    return payload.get("case_id") or Path(path).parents[1].name


def rebuild_one(output_root, observable, date_tag):
    summary_name, csv_name, fields = SPECS[observable]
    root = batch_dir(output_root, observable, date_tag)
    rows = []
    for path in sorted((root / "cases").glob(f"*/{observable}/{summary_name}")):
        payload = read_json(path)
        if observable == "rdf" and isinstance(payload.get("entries"), list):
            entries = payload["entries"]
        else:
            entries = [payload]
        for entry in entries:
            row = dict(entry)
            row["case_id"] = case_id_from_summary(path, row)
            row.setdefault("source_case_dir", "")
            row.setdefault("analysis_version", "canonical_v2")
            rows.append(row)
    rows.sort(key=lambda row: (row.get("case_id", ""), row.get("rdf_kind", ""), row.get("selection", "")))
    out = aggregate_dir(root) / csv_name
    write_csv(out, rows, fields)
    if observable == "rdf":
        rebuild_rdf_long_csvs(root)
    return out, rows


def rebuild_rdf_long_csvs(root):
    cases_root = Path(root) / "cases"
    aggregate = aggregate_dir(root)
    for filename in RDF_LONG_FILES:
        rows = []
        fields = None
        for path in sorted(cases_root.glob(f"*/rdf/{filename}")):
            file_rows = read_csv_rows(path)
            if not file_rows:
                continue
            if fields is None:
                fields = list(file_rows[0].keys())
            rows.extend(file_rows)
        if fields:
            rows.sort(key=lambda row: (row.get("case_id", ""), safe_float(row.get("r_nm"))))
            write_csv(aggregate / filename, rows, fields)


def main():
    args = parse_args()
    for observable in [item.strip() for item in args.observables.split(",") if item.strip()]:
        if observable not in SPECS:
            raise ValueError(f"Unknown observable: {observable}")
        out, rows = rebuild_one(args.output_root, observable, args.date_tag)
        unique_cases = len({row.get("case_id", "") for row in rows})
        print(f"{observable}: rows={len(rows)} cases={unique_cases} -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
