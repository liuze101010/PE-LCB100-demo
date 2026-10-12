#!/usr/bin/env python3
"""PE-LCB100 literature-grounded design-origin evidence ledger.

This module turns the retained PE-LCB100 manifest and local design-source
catalog into sidecar evidence artifacts.  It is intentionally read-only with
respect to the canonical manifest, ML tables, MD outputs, and rule thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polymer_design_validator import validate_polymer_design_candidate


SCRIPT_PATH = Path(__file__).resolve()
MD_ANALYSIS_DIR = SCRIPT_PATH.parent
GENERATED_CODE_DIR = MD_ANALYSIS_DIR.parent
PROJECT_ROOT = GENERATED_CODE_DIR.parent
DEFAULT_MANIFEST_JSON = GENERATED_CODE_DIR / "emc_alpha_olefin_builder" / "output" / "screen_pelcb100" / "manifest.json"
DEFAULT_MANIFEST_CSV = GENERATED_CODE_DIR / "emc_alpha_olefin_builder" / "output" / "screen_pelcb100" / "manifest.csv"
DEFAULT_SOURCE_CATALOG = MD_ANALYSIS_DIR / "plugins" / "pelcb100_design_source_catalog.json"
DEFAULT_PROVENANCE_CASES = MD_ANALYSIS_DIR / "output" / "pelcb100_integrated" / "aggregate" / "provenance_cases.csv"
DEFAULT_OUTPUT_DIR = MD_ANALYSIS_DIR / "output" / "pelcb100_design_origin"


OUTPUT_FILENAMES = {
    "design_evidence_pack_jsonl": "design_evidence_pack.jsonl",
    "case_design_origin_table_csv": "case_design_origin_table.csv",
    "design_prior_summary_md": "design_prior_summary.md",
    "candidate_manifest_validation_report_md": "candidate_manifest_validation_report.md",
    "pelcb100_design_origin_ledger_jsonl": "pelcb100_design_origin_ledger.jsonl",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _sha256_bytes(blob: bytes) -> str:
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return _sha256_bytes(blob)


def _sha256_path(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_source_catalog(catalog_path: str | Path = DEFAULT_SOURCE_CATALOG) -> dict[str, Any]:
    path = Path(catalog_path)
    payload = _read_json(path)
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("design-source catalog must contain a list field named 'sources'")
    return payload


def _source_index(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in catalog.get("sources", []) or []:
        aliases = [str(alias) for alias in source.get("aliases", []) or []]
        title = str(source.get("title", "") or "")
        source_id = str(source.get("source_id", "") or "")
        out.append({**source, "_match_terms": [term for term in [source_id, title, *aliases] if term]})
    return out


def _matched_sources(anchor: dict[str, Any], catalog: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    text = " ".join(
        [
            str(anchor.get("primary") or ""),
            str(anchor.get("reason") or ""),
        ]
    ).lower()
    matched: list[dict[str, Any]] = []
    for source in _source_index(catalog):
        terms = [str(term).lower() for term in source.get("_match_terms", []) if str(term).strip()]
        if any(term in text for term in terms):
            matched.append({key: value for key, value in source.items() if key != "_match_terms"})
    ids = sorted({str(source.get("source_id")) for source in matched if source.get("source_id")})
    return matched, ids


def _case_json_path(case_id: str) -> Path:
    return DEFAULT_MANIFEST_JSON.parent / case_id / "case.json"


def _manifest_case_lookup(manifest_json: Path = DEFAULT_MANIFEST_JSON) -> dict[str, dict[str, Any]]:
    payload = _read_json(Path(manifest_json))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a list field named 'cases'")
    return {str(case.get("case_id")): dict(case) for case in cases}


def _csv_case_lookup(manifest_csv: Path = DEFAULT_MANIFEST_CSV) -> dict[str, dict[str, str]]:
    return {str(row.get("case_id")): row for row in _read_csv_rows(Path(manifest_csv))}


def _provenance_lookup(path: Path = DEFAULT_PROVENANCE_CASES) -> dict[str, dict[str, str]]:
    return {str(row.get("case_id")): row for row in _read_csv_rows(path)}


def _int_field(row: dict[str, Any], field: str, default: int = 0) -> int:
    try:
        return int(row.get(field, default))
    except Exception:
        return default


def _candidate_from_manifest_case(case: dict[str, Any]) -> dict[str, Any]:
    actual_total = _int_field(case, "actual_total_carbons_per_chain", 100)
    return {
        "candidate_id": case.get("case_id"),
        "polymer_family": "PE",
        "architecture": "pe_lcb",
        "builder_backend": "emc_alpha_olefin_builder",
        "force_field": case.get("force_field") or "opls/2024/opls-aa",
        "chain_count": _int_field(case, "chain_count", 150),
        "target_total_carbons_per_chain": _int_field(case, "target_total_carbons_per_chain", 100),
        "carbon_budget": actual_total,
        "branch_points": _int_field(case, "branch_points"),
        "branch_carbons": _int_field(case, "branch_carbons"),
        "branch_position_pattern": case.get("branch_position_pattern"),
    }


def _design_variable(case: dict[str, Any]) -> str:
    bp = _int_field(case, "branch_points")
    bc = _int_field(case, "branch_carbons")
    pattern = str(case.get("branch_position_pattern") or "")
    return f"bp{bp:02d}_bc{bc:03d}_pd{pattern}"


def _duplicate_key_counts(cases: list[dict[str, Any]]) -> dict[tuple[int, int, str], int]:
    counts: dict[tuple[int, int, str], int] = {}
    for case in cases:
        key = (
            _int_field(case, "branch_points"),
            _int_field(case, "branch_carbons"),
            str(case.get("branch_position_pattern") or ""),
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_design_origin_records(
    *,
    manifest_json: str | Path = DEFAULT_MANIFEST_JSON,
    manifest_csv: str | Path = DEFAULT_MANIFEST_CSV,
    catalog_path: str | Path = DEFAULT_SOURCE_CATALOG,
    provenance_cases_csv: str | Path = DEFAULT_PROVENANCE_CASES,
) -> dict[str, Any]:
    manifest_json = Path(manifest_json)
    manifest_csv = Path(manifest_csv)
    catalog = load_source_catalog(catalog_path)
    case_lookup = _manifest_case_lookup(manifest_json)
    csv_lookup = _csv_case_lookup(manifest_csv)
    provenance_lookup = _provenance_lookup(Path(provenance_cases_csv))
    cases = [case_lookup[key] for key in sorted(case_lookup)]
    duplicate_counts = _duplicate_key_counts(cases)
    manifest_hash = _sha256_path(manifest_csv) or _sha256_path(manifest_json)

    records: list[dict[str, Any]] = []
    review_notes: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("case_id") or "")
        anchor = case.get("literature_anchor") if isinstance(case.get("literature_anchor"), dict) else {}
        matched_sources, matched_source_ids = _matched_sources(anchor, catalog)
        candidate = _candidate_from_manifest_case(case)
        validation = validate_polymer_design_candidate(candidate)
        key = (
            _int_field(case, "branch_points"),
            _int_field(case, "branch_carbons"),
            str(case.get("branch_position_pattern") or ""),
        )
        duplicate_status = "pass" if duplicate_counts.get(key, 0) == 1 else "reject"
        validator_errors = list(validation.get("errors", []) or [])
        if duplicate_status != "pass":
            validator_errors.append("duplicate design key in frozen PE-LCB100 manifest")
        validator_status = "pass" if validation.get("validator_status") == "pass" and duplicate_status == "pass" else "reject"

        case_json = _case_json_path(case_id)
        provenance = provenance_lookup.get(case_id, {})
        source_metadata_flags = sorted(
            {
                str(source.get("metadata_status"))
                for source in matched_sources
                if str(source.get("metadata_status") or "") == "source_metadata_incomplete"
            }
        )
        record_review_notes: list[str] = []
        if not matched_source_ids:
            record_review_notes.append("unmapped_anchor")
        if source_metadata_flags:
            record_review_notes.append("source_metadata_incomplete")
        if validator_status != "pass":
            record_review_notes.append("validator_reject")
        for note in record_review_notes:
            review_notes.append({"case_id": case_id, "note": note})

        builder_spec = validation.get("builder_spec", {}) if isinstance(validation, dict) else {}
        row = {
            "schema_version": "pelcb100_design_origin.v1",
            "case_id": case_id,
            "preset": case.get("preset", "screen_pelcb100"),
            "human_scope": "PE-LCB/C100",
            "agent_role": "design_prior_extraction_and_manifest_generation",
            "deterministic_validator": "freeze_gate",
            "canonical_manifest_modified": False,
            "design_variable": _design_variable(case),
            "branch_points": _int_field(case, "branch_points"),
            "branch_carbons": _int_field(case, "branch_carbons"),
            "branch_position_pattern": case.get("branch_position_pattern"),
            "branch_positions": case.get("branch_positions", []),
            "branch_topology": case.get("branch_topology", ""),
            "topology_class": case.get("topology_class", ""),
            "source_case_id": case.get("source_case_id", ""),
            "polymer_family": case.get("polymer_family", "PE"),
            "architecture": case.get("architecture", "pe_lcb"),
            "backbone_monomer": case.get("backbone_monomer", "ethylene"),
            "chain_count": _int_field(case, "chain_count", 150),
            "target_total_carbons_per_chain": _int_field(case, "target_total_carbons_per_chain", 100),
            "actual_total_carbons_per_chain": _int_field(case, "actual_total_carbons_per_chain", 100),
            "force_field": case.get("force_field", ""),
            "temperature_k": case.get("temperature_k", ""),
            "pressure_atm": case.get("pressure_atm", ""),
            "initial_density_g_per_cc": case.get("initial_density_g_per_cc", ""),
            "source_anchor": anchor.get("primary", ""),
            "design_rationale": anchor.get("reason", ""),
            "mapped_source_ids": matched_source_ids,
            "mapped_sources": matched_sources,
            "source_metadata_flags": source_metadata_flags,
            "review_notes": record_review_notes,
            "manifest_hash": manifest_hash,
            "manifest_row_hash": _sha256_json(case),
            "manifest_csv_row_hash": _sha256_json(csv_lookup.get(case_id, {})),
            "case_json_path": str(case_json),
            "case_json_hash": _sha256_path(case_json),
            "provenance_manifest_row_hash": provenance.get("manifest_row_hash", ""),
            "provenance_case_json_sha256": ("sha256:" + provenance.get("case_json_sha256", ""))
            if provenance.get("case_json_sha256")
            else "",
            "validator_status": validator_status,
            "validator_errors": validator_errors,
            "validator_warnings": list(validation.get("warnings", []) or []),
            "builder_spec_hash": _sha256_json(builder_spec) if builder_spec else "",
            "builder_spec": builder_spec,
        }
        records.append(row)

    summary = {
        "schema_version": "pelcb100_design_origin.v1",
        "created_at_utc": _utc_now(),
        "case_count": len(records),
        "validator_pass_count": sum(1 for row in records if row["validator_status"] == "pass"),
        "validator_reject_count": sum(1 for row in records if row["validator_status"] != "pass"),
        "unmapped_anchor_count": sum(1 for row in records if "unmapped_anchor" in row["review_notes"]),
        "review_note_count": len(review_notes),
        "source_metadata_incomplete_count": sum(
            1 for row in records if "source_metadata_incomplete" in row["review_notes"]
        ),
        "canonical_manifest_modified": False,
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "manifest_hash": manifest_hash,
        "catalog_path": str(catalog_path),
        "catalog_id": catalog.get("catalog_id", ""),
        "source_count": len(catalog.get("sources", []) or []),
        "human_scope": "PE-LCB/C100",
        "agent_role": "design_prior_extraction_and_manifest_generation",
        "deterministic_validator": "freeze_gate",
    }
    return {"records": records, "summary": summary, "review_notes": review_notes, "catalog": catalog}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "design_variable",
        "branch_points",
        "branch_carbons",
        "branch_position_pattern",
        "branch_topology",
        "topology_class",
        "source_case_id",
        "source_anchor",
        "design_rationale",
        "mapped_source_ids",
        "review_notes",
        "validator_status",
        "validator_errors",
        "manifest_hash",
        "manifest_row_hash",
        "case_json_hash",
        "builder_spec_hash",
        "canonical_manifest_modified",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field, "")
                    for field in fields
                }
            )


def _summary_markdown(summary: dict[str, Any], catalog: dict[str, Any]) -> str:
    lines = [
        "# PE-LCB100 design-prior summary",
        "",
        "This sidecar report records retained design-origin evidence for the PE-LCB100 manifest. It does not claim unconstrained autonomous discovery and does not modify the canonical manifest.",
        "",
        "## Scope",
        "",
        "- Human-specified scope: `PE-LCB/C100`.",
        "- Agent role: literature/design-prior extraction and grammar-constrained manifest generation.",
        "- Deterministic validator role: freeze gate for schema, PE-LCB topology, carbon budget, backend and builder-spec compatibility.",
        "",
        "## Counts",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Validator pass: `{summary['validator_pass_count']}`",
        f"- Validator reject: `{summary['validator_reject_count']}`",
        f"- Unmapped anchors: `{summary['unmapped_anchor_count']}`",
        f"- Source metadata incomplete notes: `{summary['source_metadata_incomplete_count']}`",
        "",
        "## Source catalog",
        "",
    ]
    for source in catalog.get("sources", []) or []:
        lines.append(
            f"- `{source.get('source_id')}`: {source.get('role', '')} "
            f"(`{source.get('metadata_status', '')}`)"
        )
    lines.extend(
        [
            "",
            "## Safe manuscript claim",
            "",
            "The LLM planning layer used literature-grounded design priors to construct a grammar-constrained PE-LCB100 candidate manifest under a human-specified PE-LCB/C100 scope; Codex materialized the manifest, deterministic validation froze the accepted candidates, and all design-origin evidence was ledgered.",
            "",
        ]
    )
    return "\n".join(lines)


def _validation_report_markdown(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    reject_rows = [row for row in records if row["validator_status"] != "pass"]
    lines = [
        "# PE-LCB100 candidate-manifest validation report",
        "",
        "Validation is read-only and does not regenerate molecules, submit EMC/MD jobs, or alter canonical release tables.",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Manifest cases | {summary['case_count']} |",
        f"| Validator pass | {summary['validator_pass_count']} |",
        f"| Validator reject | {summary['validator_reject_count']} |",
        f"| Unmapped anchor records | {summary['unmapped_anchor_count']} |",
        "",
        "## Rejected candidates",
        "",
    ]
    if not reject_rows:
        lines.append("None.")
    else:
        lines.append("| case_id | errors |")
        lines.append("|---|---|")
        for row in reject_rows:
            lines.append(f"| `{row['case_id']}` | `{'; '.join(row.get('validator_errors', []))}` |")
    lines.append("")
    return "\n".join(lines)


def build_design_origin_outputs(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_json: str | Path = DEFAULT_MANIFEST_JSON,
    manifest_csv: str | Path = DEFAULT_MANIFEST_CSV,
    catalog_path: str | Path = DEFAULT_SOURCE_CATALOG,
    provenance_cases_csv: str | Path = DEFAULT_PROVENANCE_CASES,
) -> dict[str, Any]:
    payload = build_design_origin_records(
        manifest_json=manifest_json,
        manifest_csv=manifest_csv,
        catalog_path=catalog_path,
        provenance_cases_csv=provenance_cases_csv,
    )
    output_dir = Path(output_dir)
    outputs = {key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()}

    _write_jsonl(outputs["design_evidence_pack_jsonl"], payload["records"])
    _write_csv(outputs["case_design_origin_table_csv"], payload["records"])
    outputs["design_prior_summary_md"].write_text(
        _summary_markdown(payload["summary"], payload["catalog"]),
        encoding="utf-8",
    )
    outputs["candidate_manifest_validation_report_md"].write_text(
        _validation_report_markdown(payload["records"], payload["summary"]),
        encoding="utf-8",
    )
    ledger_rows = [
        {
            "timestamp_utc": payload["summary"]["created_at_utc"],
            "case_id": row["case_id"],
            "human_scope": row["human_scope"],
            "agent_role": row["agent_role"],
            "deterministic_validator": row["deterministic_validator"],
            "validator_status": row["validator_status"],
            "manifest_row_hash": row["manifest_row_hash"],
            "builder_spec_hash": row["builder_spec_hash"],
            "canonical_manifest_modified": False,
            "review_notes": row["review_notes"],
        }
        for row in payload["records"]
    ]
    _write_jsonl(outputs["pelcb100_design_origin_ledger_jsonl"], ledger_rows)

    return {
        "status": "ok",
        "summary": payload["summary"],
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def get_design_origin_record(case_id: str) -> dict[str, Any]:
    payload = build_design_origin_records()
    for row in payload["records"]:
        if row["case_id"] == case_id:
            return {"status": "ok", "record": row, "summary": payload["summary"]}
    return {"status": "not_found", "case_id": case_id, "summary": payload["summary"]}


def get_design_origin_summary() -> dict[str, Any]:
    payload = build_design_origin_records()
    return {"status": "ok", "summary": payload["summary"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_SOURCE_CATALOG)
    args = parser.parse_args(argv)
    result = build_design_origin_outputs(
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        manifest_csv=args.manifest_csv,
        catalog_path=args.catalog,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
