from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "pass"}


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_terminal_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "case_seq": int(row["case_seq"]),
        "terminal_verdict": row["result_case_verdict"],
        "terminal_gate_pass": _as_bool(row["result_terminal_gate_pass"]),
        "terminal_release_state": row["current_terminal_release_state"],
        "publication_state": row["current_publication_state"],
        "publication_state_reason": row["current_publication_state_reason"],
        "repair_route": row["current_repair_route"],
        "repair_action": row["current_repair_action"],
        "clean_repair_state": row["current_clean_repair_state"],
        "clean_postprocess_state": row["current_clean_postprocess_state"],
        "clean_regate_verdict": row["current_clean_release_overlay_verdict"],
        "density_g_cm3": _as_float(row["result_density_g_cm3"]),
        "msd_loglog_slope": _as_float(row["result_loglog_slope"]),
        "checks": {
            "required_values_complete": _as_bool(row["result_required_values_complete"]),
            "all_physical_checks_pass": _as_bool(row["result_all_physical_checks_pass"]),
            "density": _as_bool(row["result_density_check"]),
            "rg": _as_bool(row["result_rg_check"]),
            "shape": _as_bool(row["result_shape_check"]),
            "rdf": _as_bool(row["result_rdf_check"]),
            "contact": _as_bool(row["result_contact_check"]),
            "msd": _as_bool(row["result_msd_check"]),
            "fsqt": _as_bool(row["result_fsqt_check"]),
            "intrachain_dynamics": _as_bool(row["result_intrachain_dynamics_check"]),
        },
    }


def build_terminal_outcomes(release_table: Path) -> dict[str, Any]:
    with release_table.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [_safe_terminal_record(row) for row in rows]
    case_ids = [record["case_id"] for record in records]
    if len(records) != 100 or len(set(case_ids)) != 100:
        raise ValueError("terminal release table must contain 100 unique cases")
    counts = Counter(record["terminal_verdict"] for record in records)
    if counts != {"PASS": 92, "REVIEW": 8}:
        raise ValueError(f"expected 92 PASS / 8 REVIEW, got {dict(counts)}")
    return {
        "schema_version": "polymer_agent_demo.terminal_outcomes.v1",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_sha256": _sha256(release_table),
        "summary": {"case_count": 100, "pass": 92, "review": 8},
        "cases": sorted(records, key=lambda item: item["case_id"]),
    }


def validate_audit_ledger(audit_jsonl: Path) -> dict[str, int]:
    rows = [
        json.loads(line)
        for line in audit_jsonl.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    case_count = len(rows)
    unique_cases = len({row.get("case_id") for row in rows})
    model_votes = sum(len(row.get("model_votes") or []) for row in rows)
    complete_panels = sum(len(row.get("model_votes") or []) == 3 for row in rows)
    if (case_count, unique_cases, model_votes, complete_panels) != (97, 97, 291, 97):
        raise ValueError(
            "audit ledger must contain 97 unique cases and 291 votes "
            f"with 97 complete panels; got {case_count}/{unique_cases}/{model_votes}/{complete_panels}"
        )
    return {
        "case_count": case_count,
        "unique_cases": unique_cases,
        "model_votes": model_votes,
        "complete_panels": complete_panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the reduced public terminal ledger and validate the locked audit ledger."
    )
    parser.add_argument("--release-table", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "terminal_release_outcomes.json",
    )
    args = parser.parse_args()
    audit = validate_audit_ledger(args.audit_jsonl.resolve())
    payload = build_terminal_outcomes(args.release_table.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.output}: {payload['summary']['pass']} PASS / "
        f"{payload['summary']['review']} REVIEW; "
        f"audit {audit['case_count']} cases / {audit['model_votes']} votes"
    )


if __name__ == "__main__":
    main()
