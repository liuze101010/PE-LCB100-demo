#!/usr/bin/env python3
"""Build and replay the frozen PE-LCB100 history for a ReAct baseline.

The replay is deliberately separate from the production workflow.  Public
tool responses are assembled from time-ordered, frozen evidence.  Terminal
publication labels remain private and are used only by the scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = Path(r"${PELCB100_PAPER_ROOT}")
DEFAULT_INITIAL_LEDGER = PAPER_ROOT / "figures" / "fig4_closed_loop_case_ledger.csv"
DEFAULT_ROUTE_LEDGER = PAPER_ROOT / "figures" / "fig4_planning_route_ledger.csv"
DEFAULT_FINAL_LEDGER = PAPER_ROOT / "figures" / "fig9_rule_audit_ablation_case_source.csv"
DEFAULT_MSD_PASS_LEDGER = PAPER_ROOT / "figures" / "fig6_msd_window_candidates.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "pelcb100_react_baseline"

SCHEMA_VERSION = "pelcb100_react_replay.v1"
MAX_RESULT_CHANGING_ACTIONS = 2
DECISIONS = {"PASS", "REVIEW", "FAIL"}
REGISTERED_ROUTES = {
    "submit_md_clean_engine",
    "run_postprocess_repair",
    "open_review_packet",
}
FORBIDDEN_PUBLIC_KEYS = {
    "reference_decision",
    "reference_full_pass",
    "publication_state",
    "publication_grade",
    "effective_publication_grade",
    "final_state",
    "final_route",
    "expected_decision",
}
FORBIDDEN_PUBLIC_TEXT = (
    "reference_decision",
    "reference_full_pass",
    "effective_publication_grade",
    "publication_state",
    "final diagnosis",
    "terminal state",
    "retained for review",
    "92 pass",
    "8 review",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _case_short(row: Mapping[str, Any]) -> str:
    value = str(row.get("case_seq", "")).strip()
    try:
        return f"PE{int(float(value)):03d}"
    except (TypeError, ValueError):
        return str(row.get("case_id", ""))


def _initial_conditions(row: Mapping[str, str]) -> list[str]:
    generic = str(row.get("generic_qc_verdict", "")).strip().upper()
    density = str(row.get("density_rule_verdict", "")).strip().upper()
    conditions: list[str] = []
    if generic == "WARN_INCOMPLETE" or density == "MISSING":
        conditions.append("missing_output")
    elif generic == "WARN":
        conditions.append("msd")
    if density == "WARN":
        conditions.append("density")
    if not conditions and "audit_disagreement_warn" in str(row.get("warn_source", "")):
        conditions.append("retained_audit_signal")
    return conditions


def _missing_outputs(row: Mapping[str, str]) -> list[str]:
    names = {
        "status_density": "density",
        "status_Rg": "Rg",
        "status_RDF": "RDF",
        "status_shape": "shape_anisotropy",
        "status_MSD": "MSD",
        "status_fsqt_tau_alpha": "Fsqt",
        "status_contact": "contact_probability",
    }
    return [label for key, label in names.items() if str(row.get(key, "")).strip().upper() == "MISSING"]


def _presence(row: Mapping[str, str]) -> dict[str, bool]:
    return {
        "density": str(row.get("status_density", "")).strip().upper() != "MISSING",
        "Rg": str(row.get("status_Rg", "")).strip().upper() != "MISSING",
        "RDF": str(row.get("status_RDF", "")).strip().upper() != "MISSING",
        "shape_anisotropy": str(row.get("status_shape", "")).strip().upper() != "MISSING",
        "MSD": str(row.get("status_MSD", "")).strip().upper() != "MISSING",
        "Fsqt": str(row.get("status_fsqt_tau_alpha", "")).strip().upper() != "MISSING",
        "contact_probability": str(row.get("status_contact", "")).strip().upper() != "MISSING",
    }


def _public_initial_evidence(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "case_short": _case_short(row),
        "design": {
            "case_sequence": int(float(row["case_seq"])),
            "branch_points": _as_float(row.get("branch_points")),
            "branch_carbons_per_branch": _as_float(row.get("branch_carbons")),
        },
        "first_postprocessing_record": {
            "output_available": _presence(row),
            "missing_outputs": _missing_outputs(row),
            "density_g_cm3": _as_float(row.get("rho_final_gcc")),
        },
    }


def _msd_pass_record(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "strict_reassessment": "PASS",
        "trajectory_length_ns": _as_float(row.get("max_time_ns")),
        "fit_begin_ns": _as_float(row.get("best_begin_ns")),
        "fit_end_ns": _as_float(row.get("best_end_ns")),
        "loglog_slope": _as_float(row.get("best_loglog_slope")),
        "linear_R2": _as_float(row.get("best_linear_R2")),
        "local_D_cv_percent": _as_float(row.get("best_local_D_cv_pct")),
    }


def _msd_review_record(row: Mapping[str, str]) -> dict[str, Any]:
    reason = str(row.get("publication_state_reason", "")).strip()
    time_match = re.search(r"time_last_ps=([0-9.eE+-]+)", reason)
    slope_match = re.search(r"loglog_slope=([0-9.eE+-]+)", reason)
    time_ns = float(time_match.group(1)) / 1000.0 if time_match else None
    slope = float(slope_match.group(1)) if slope_match else _as_float(row.get("best_loglog_slope"))
    return {
        "strict_reassessment": "WARN",
        "unresolved_check": "No fitting window in the available bounded trajectory satisfied every strict MSD condition.",
        "trajectory_length_ns": time_ns,
        "fit_begin_ns": _as_float(row.get("best_begin_ns")),
        "fit_end_ns": _as_float(row.get("best_end_ns")),
        "loglog_slope": slope,
        "linear_R2": _as_float(row.get("best_loglog_R2")),
        "local_D_cv_percent": _as_float(row.get("best_local_D_cv_pct")),
    }


def _parse_allowed_routes(row: Mapping[str, str] | None) -> list[dict[str, Any]]:
    if not row:
        return []
    try:
        routes = json.loads(row.get("allowed_routes", "[]"))
    except json.JSONDecodeError:
        return []
    return [route for route in routes if route.get("repair_tool") in REGISTERED_ROUTES]


def _find_public_key_leaks(value: Any, path: str = "") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key).lower() in FORBIDDEN_PUBLIC_KEYS:
                leaks.append(here)
            leaks.extend(_find_public_key_leaks(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaks.extend(_find_public_key_leaks(child, f"{path}[{index}]"))
    return leaks


def _find_public_text_leaks(value: Any, path: str = "") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}" if path else str(key)
            leaks.extend(_find_public_text_leaks(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaks.extend(_find_public_text_leaks(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for token in FORBIDDEN_PUBLIC_TEXT:
            if token in lowered:
                leaks.append(f"{path}:{token}")
    return leaks


def build_fixtures(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    initial_ledger: Path = DEFAULT_INITIAL_LEDGER,
    route_ledger: Path = DEFAULT_ROUTE_LEDGER,
    final_ledger: Path = DEFAULT_FINAL_LEDGER,
    msd_pass_ledger: Path = DEFAULT_MSD_PASS_LEDGER,
) -> dict[str, Any]:
    initial_rows = _read_csv(initial_ledger)
    route_rows = _read_csv(route_ledger)
    final_rows = _read_csv(final_ledger)
    msd_pass_rows = _read_csv(msd_pass_ledger)
    initial = {row["case_id"]: row for row in initial_rows}
    route = {row["case_id"]: row for row in route_rows}
    final = {row["case_id"]: row for row in final_rows}
    msd_pass = {row["case_id"]: row for row in msd_pass_rows}
    if set(initial) != set(final):
        raise ValueError("initial and terminal ledgers contain different case sets")
    if len(msd_pass) != 39:
        raise ValueError(f"expected 39 strict-MSD PASS records, found {len(msd_pass)}")

    final_review = {case_id for case_id, row in final.items() if not _truthy(row["reference_full_pass"])}
    if len(final_review) != 8:
        raise ValueError(f"expected 8 terminal REVIEW cases, found {len(final_review)}")
    strict_msd_cohort = set(msd_pass) | final_review
    if len(strict_msd_cohort) != 47:
        raise ValueError(f"expected 47 strict-MSD reassessment cases, found {len(strict_msd_cohort)}")

    fixture_dir = output_root / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_index: list[dict[str, Any]] = []
    public_leaks: list[dict[str, Any]] = []
    for case_id in sorted(initial, key=lambda key: int(float(initial[key]["case_seq"]))):
        source = initial[case_id]
        conditions = _initial_conditions(source)
        terminal = "PASS" if _truthy(final[case_id]["reference_full_pass"]) else "REVIEW"
        strict_record: dict[str, Any] | None = None
        if case_id in msd_pass:
            strict_record = _msd_pass_record(msd_pass[case_id])
        elif case_id in final_review:
            strict_record = _msd_review_record(source)
        fixture = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "case_short": _case_short(source),
            "public_initial_evidence": _public_initial_evidence(source),
            "private_history": {
                "initial_conditions": conditions,
                "initial_audit": {
                    "primary_diagnosis": (route.get(case_id) or {}).get("primary_audit_diagnosis", ""),
                    "model_disagreement": _truthy((route.get(case_id) or {}).get("current_panel_disagreement", "")),
                    "historical_allowed_routes": _parse_allowed_routes(route.get(case_id)),
                },
                "density_after_g_cm3": _as_float(source.get("density_after_g_cm3")),
                "output_recovery": {
                    "status": source.get("clean_record_ok", ""),
                    "recovered_outputs": _missing_outputs(source),
                },
                "initial_msd_result": {
                    "fit_begin_ns": _as_float(source.get("best_begin_ns")),
                    "fit_end_ns": _as_float(source.get("best_end_ns")),
                    "loglog_slope": _as_float(source.get("best_loglog_slope")),
                    "linear_R2": _as_float(source.get("best_loglog_R2")),
                    "local_D_cv_percent": _as_float(source.get("best_local_D_cv_pct")),
                },
                "strict_msd_record": strict_record,
                "reference_decision": terminal,
                "reference_publication_state": final[case_id].get("publication_state", ""),
            },
        }
        leaks = _find_public_key_leaks(fixture["public_initial_evidence"])
        leaks.extend(_find_public_text_leaks(fixture["public_initial_evidence"]))
        if strict_record:
            leaks.extend(_find_public_text_leaks(strict_record, "strict_msd_record"))
        if leaks:
            public_leaks.append({"case_id": case_id, "locations": leaks})
        path = fixture_dir / f"{case_id}.json"
        _write_json(path, fixture)
        fixture_index.append(
            {
                "case_id": case_id,
                "case_short": fixture["case_short"],
                "fixture_path": str(path),
                "fixture_sha256": _sha256_file(path),
                "initial_conditions": "|".join(conditions) if conditions else "none",
                "strict_msd_member": case_id in strict_msd_cohort,
                "reference_decision": terminal,
            }
        )

    condition_counts = Counter()
    initial_warn_cases = 0
    for row in fixture_index:
        domains = [] if row["initial_conditions"] == "none" else str(row["initial_conditions"]).split("|")
        if domains:
            initial_warn_cases += 1
        condition_counts.update(domains)
    if len(fixture_index) != 100:
        raise ValueError(f"expected 100 fixtures, found {len(fixture_index)}")
    if condition_counts["density"] != 77 or condition_counts["msd"] != 41 or condition_counts["missing_output"] != 6:
        raise ValueError(f"unexpected initial condition counts: {dict(condition_counts)}")
    if initial_warn_cases != 97:
        raise ValueError(f"expected 97 audit-applicable cases, found {initial_warn_cases}")
    if public_leaks:
        raise ValueError(f"public evidence leaks terminal keys: {public_leaks[:3]}")

    _write_csv(output_root / "fixture_index.csv", fixture_index)
    source_paths = {
        "initial_ledger": initial_ledger,
        "route_ledger": route_ledger,
        "final_ledger": final_ledger,
        "msd_pass_ledger": msd_pass_ledger,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(fixture_index),
        "initial_audit_applicable_cases": initial_warn_cases,
        "initial_condition_counts": dict(condition_counts),
        "strict_msd_reassessment_count": len(strict_msd_cohort),
        "strict_msd_pass_count": len(msd_pass),
        "terminal_counts": dict(Counter(row["reference_decision"] for row in fixture_index)),
        "fixture_index": str(output_root / "fixture_index.csv"),
        "sources": {
            key: {"path": str(path), "sha256": _sha256_file(path)} for key, path in source_paths.items()
        },
        "time_layers": [
            "first_postprocessing_record",
            "initial_result_checks_and_audit",
            "registered_historical_processing",
            "strict_msd_reassessment",
            "terminal_reference_hidden_from_agent",
        ],
    }
    _write_json(output_root / "fixture_manifest.json", manifest)
    select_pilot(output_root / "fixture_index.csv", output_root / "pilot_12_cases.json")
    return manifest


def select_pilot(index_csv: Path, output_path: Path) -> dict[str, Any]:
    rows = _read_csv(index_csv)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def add(label: str, predicate: Any, count: int = 2) -> None:
        candidates = [row for row in rows if row["case_id"] not in used and predicate(row)]
        if len(candidates) < count:
            raise ValueError(f"pilot stratum {label!r} has only {len(candidates)} unused cases")
        for row in candidates[:count]:
            used.add(row["case_id"])
            selected.append({"stratum": label, "case_id": row["case_id"], "case_short": row["case_short"]})

    add("no_initial_warning", lambda row: row["initial_conditions"] == "none")
    add("density_only", lambda row: row["initial_conditions"] == "density")
    add("msd_only", lambda row: row["initial_conditions"] == "msd")
    add("density_and_msd", lambda row: row["initial_conditions"] in {"msd|density", "density|msd"})
    add("missing_output", lambda row: "missing_output" in row["initial_conditions"])
    add("terminal_review", lambda row: row["reference_decision"] == "REVIEW")
    payload = {"schema_version": SCHEMA_VERSION, "case_count": len(selected), "cases": selected}
    _write_json(output_path, payload)
    return payload


@dataclass
class ReplayState:
    fixture: dict[str, Any]
    state_path: Path | None = None
    tool_log_path: Path | None = None
    started: bool = False
    phase: str = "not_started"
    pending_conditions: list[str] = field(default_factory=list)
    unlocked_results: list[dict[str, Any]] = field(default_factory=list)
    last_check: dict[str, Any] | None = None
    last_audit: dict[str, Any] | None = None
    authorized_routes: list[dict[str, Any]] = field(default_factory=list)
    result_changing_actions: int = 0
    invalid_route_calls: int = 0
    strict_reassessment_opened: bool = False
    strict_route_attempted: bool = False
    finished: bool = False
    agent_decision: str = ""
    agent_reason: str = ""
    tool_call_count: int = 0

    @classmethod
    def from_file(
        cls,
        fixture_path: Path,
        state_path: Path | None = None,
        tool_log_path: Path | None = None,
    ) -> "ReplayState":
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        return cls(fixture=fixture, state_path=state_path, tool_log_path=tool_log_path)

    @property
    def case_id(self) -> str:
        return self.fixture["case_id"]

    def _public_state(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": self.case_id,
            "started": self.started,
            "phase": self.phase,
            "pending_conditions": list(self.pending_conditions),
            "unlocked_results": list(self.unlocked_results),
            "last_check": self.last_check,
            "authorized_route_count": len(self.authorized_routes),
            "result_changing_actions_used": self.result_changing_actions,
            "result_changing_actions_remaining": MAX_RESULT_CHANGING_ACTIONS - self.result_changing_actions,
            "finished": self.finished,
            "agent_decision": self.agent_decision,
            "agent_reason": self.agent_reason,
            "invalid_route_calls": self.invalid_route_calls,
            "tool_call_count": self.tool_call_count,
        }

    def _persist(self) -> None:
        if self.state_path:
            _write_json(self.state_path, self._public_state())

    def _record(self, tool: str, arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
        self.tool_call_count += 1
        public_state = self._public_state()
        visible = {
            "initial_evidence": self.fixture["public_initial_evidence"] if self.started else {},
            "unlocked_results": self.unlocked_results,
            "last_check": self.last_check,
        }
        record = {
            "recorded_at": datetime.now().isoformat(timespec="milliseconds"),
            "case_id": self.case_id,
            "tool": tool,
            "arguments": dict(arguments),
            "result": dict(result),
            "visible_evidence_sha256": _sha256_payload(visible),
            "remaining_unclosed_conditions": list(self.pending_conditions),
            "public_state_sha256": _sha256_payload(public_state),
        }
        if self.tool_log_path:
            _append_jsonl(self.tool_log_path, record)
        self._persist()
        return dict(result)

    def _error(self, tool: str, message: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._record(tool, arguments or {}, {"status": "error", "message": message})

    def start_case(self, case_id: str) -> dict[str, Any]:
        if self.started:
            return self._error("start_case", "This isolated session already has an active case.", {"case_id": case_id})
        if case_id != self.case_id:
            return self._error("start_case", "The requested case is unavailable in this isolated session.", {"case_id": case_id})
        self.started = True
        self.phase = "initial_evidence"
        self.pending_conditions = list(self.fixture["private_history"]["initial_conditions"])
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "message": "Case started. Use get_current_evidence or run_result_checks to continue.",
            "maximum_result_changing_actions": MAX_RESULT_CHANGING_ACTIONS,
        }
        return self._record("start_case", {"case_id": case_id}, result)

    def get_current_evidence(self) -> dict[str, Any]:
        if not self.started:
            return self._error("get_current_evidence", "Call start_case first.")
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "current_phase": self.phase,
            "initial_evidence": self.fixture["public_initial_evidence"],
            "unlocked_results": list(self.unlocked_results),
            "visible_evidence_sha256": _sha256_payload(
                {"initial": self.fixture["public_initial_evidence"], "unlocked": self.unlocked_results}
            ),
        }
        return self._record("get_current_evidence", {}, result)

    def _condition_details(self) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        density = self.fixture["public_initial_evidence"]["first_postprocessing_record"]["density_g_cm3"]
        for condition in self.pending_conditions:
            if condition == "density":
                details.append(
                    {
                        "condition": condition,
                        "observation": {"density_g_cm3": density},
                        "acceptance_condition": "0.73 <= density_g_cm3 <= 0.87",
                    }
                )
            elif condition == "msd":
                details.append(
                    {
                        "condition": condition,
                        "observation": "The first MSD analysis did not satisfy every fitting-window condition.",
                        "acceptance_condition": (
                            "slope 0.95-1.05, linear R2 >= 0.95, local D CV <= 5%, "
                            "window start >= 0.25T, end <= 0.85T, width >= 0.30T"
                        ),
                    }
                )
            elif condition == "missing_output":
                details.append(
                    {
                        "condition": condition,
                        "observation": {
                            "missing_outputs": self.fixture["public_initial_evidence"]["first_postprocessing_record"][
                                "missing_outputs"
                            ]
                        },
                        "acceptance_condition": "all required outputs are present",
                    }
                )
            elif condition == "retained_audit_signal":
                details.append(
                    {
                        "condition": condition,
                        "observation": "A retained provenance record requires cause review before release.",
                        "acceptance_condition": "review record is closed",
                    }
                )
            elif condition == "strict_msd":
                details.append(
                    {
                        "condition": condition,
                        "observation": self.fixture["private_history"]["strict_msd_record"],
                        "acceptance_condition": (
                            "slope 0.95-1.05, linear R2 >= 0.95, local D CV <= 5%, "
                            "window start >= 0.25T, end <= 0.85T, width >= 0.30T"
                        ),
                    }
                )
        return details

    def _open_strict_reassessment(self) -> None:
        if self.strict_reassessment_opened or self.pending_conditions:
            return
        self.strict_reassessment_opened = True
        strict_record = self.fixture["private_history"].get("strict_msd_record")
        if not strict_record:
            return
        self.unlocked_results.append({"stage": "strict_msd_reassessment", **strict_record})
        if strict_record.get("strict_reassessment") == "WARN":
            self.pending_conditions.append("strict_msd")

    def _run_checks(self, tool: str) -> dict[str, Any]:
        if not self.started:
            return self._error(tool, "Call start_case first.")
        if self.finished:
            return self._error(tool, "The case has already been finished.")
        self._open_strict_reassessment()
        verdict = "WARN" if self.pending_conditions else "PASS"
        self.phase = "checks_warn" if verdict == "WARN" else "checks_pass"
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "verdict": verdict,
            "open_conditions": self._condition_details(),
            "result_changing_actions_remaining": MAX_RESULT_CHANGING_ACTIONS - self.result_changing_actions,
        }
        self.last_check = result
        self.last_audit = None
        return self._record(tool, {}, result)

    def run_result_checks(self) -> dict[str, Any]:
        return self._run_checks("run_result_checks")

    def rerun_result_checks(self) -> dict[str, Any]:
        return self._run_checks("rerun_result_checks")

    def _route_for(self, condition: str) -> dict[str, Any]:
        if condition in {"density", "missing_output"}:
            parameters: dict[str, Any] = {
                "engine_profile": "gromacs_2024_4_clean",
                "reentry_stage": "md",
            }
            if condition == "missing_output":
                parameters["input_source"] = "archived_source_inputs"
                parameters["repair_kind"] = "missing_output_recovery"
            else:
                parameters["rerun_scope"] = "same_input_clean_engine_validation"
            return {
                "route": "submit_md_clean_engine",
                "parameters": parameters,
                "addresses": [condition],
            }
        if condition == "msd":
            return {
                "route": "run_postprocess_repair",
                "parameters": {"repair_kind": "msd_window", "window_strategy": "c100_aware"},
                "addresses": [condition],
            }
        if condition == "strict_msd":
            if self.strict_route_attempted or self.result_changing_actions >= MAX_RESULT_CHANGING_ACTIONS:
                return {
                    "route": "open_review_packet",
                    "parameters": {"review_kind": "bounded_msd_conditions_unresolved"},
                    "addresses": [condition],
                }
            return {
                "route": "run_postprocess_repair",
                "parameters": {"repair_kind": "msd_bounded_extension", "target_time_ns": 800},
                "addresses": [condition],
            }
        return {
            "route": "open_review_packet",
            "parameters": {"review_kind": "retained_audit_signal"},
            "addresses": [condition],
        }

    def request_warn_audit(self) -> dict[str, Any]:
        if not self.started:
            return self._error("request_warn_audit", "Call start_case first.")
        if not self.last_check or self.last_check.get("verdict") != "WARN":
            return self._error("request_warn_audit", "A WARN result from run_result_checks is required first.")
        routes: dict[str, dict[str, Any]] = {}
        for condition in self.pending_conditions:
            route = self._route_for(condition)
            key = route["route"] + json.dumps(route["parameters"], sort_keys=True)
            if key in routes:
                routes[key]["addresses"] = sorted(set(routes[key]["addresses"] + route["addresses"]))
            else:
                routes[key] = route
        history = self.fixture["private_history"]["initial_audit"]
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "audit_summary": {
                "primary_diagnosis": history.get("primary_diagnosis")
                or ("strict_msd_conditions_unresolved" if "strict_msd" in self.pending_conditions else "review_required"),
                "model_disagreement": bool(history.get("model_disagreement")),
            },
            "allowed_routes": list(routes.values()),
            "note": "Only the listed route and parameter combinations have replayable historical results.",
        }
        for candidate in result["allowed_routes"]:
            key = (candidate["route"], json.dumps(candidate["parameters"], sort_keys=True))
            known = {
                (item["route"], json.dumps(item["parameters"], sort_keys=True)) for item in self.authorized_routes
            }
            if key not in known:
                self.authorized_routes.append(dict(candidate))
        self.last_audit = result
        self.phase = "audit_available"
        return self._record("request_warn_audit", {}, result)

    @staticmethod
    def _parameters_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
        return dict(expected) == dict(actual)

    def execute_registered_route(self, route: str, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        parameters = dict(parameters or {})
        arguments = {"route": route, "parameters": parameters}
        if not self.started:
            return self._error("execute_registered_route", "Call start_case first.", arguments)
        if self.finished:
            return self._error("execute_registered_route", "The case has already been finished.", arguments)
        allowed = self.authorized_routes
        match = next(
            (
                item
                for item in allowed
                if item.get("route") == route and self._parameters_match(item.get("parameters", {}), parameters)
            ),
            None,
        )
        if not match:
            self.invalid_route_calls += 1
            return self._error(
                "execute_registered_route",
                "No replayable result exists for this route and parameter combination.",
                arguments,
            )
        addresses = [condition for condition in match.get("addresses", []) if condition in self.pending_conditions]
        if not addresses:
            self.invalid_route_calls += 1
            return self._error(
                "execute_registered_route",
                "This previously listed route no longer addresses an open condition.",
                arguments,
            )
        if route != "open_review_packet" and self.result_changing_actions >= MAX_RESULT_CHANGING_ACTIONS:
            self.invalid_route_calls += 1
            return self._error(
                "execute_registered_route",
                "The two result-changing actions have already been used.",
                arguments,
            )

        if route == "submit_md_clean_engine":
            self.result_changing_actions += 1
            closed = [condition for condition in ("missing_output", "density") if condition in self.pending_conditions]
            self.pending_conditions = [condition for condition in self.pending_conditions if condition not in closed]
            result_payload = {
                "route_result": "historical_result_replayed",
                "closed_conditions": closed,
                "density_g_cm3": self.fixture["private_history"].get("density_after_g_cm3"),
                "required_outputs_recovered": "missing_output" in closed,
            }
        elif route == "run_postprocess_repair" and "strict_msd" in addresses:
            self.result_changing_actions += 1
            self.strict_route_attempted = True
            result_payload = {
                "route_result": "historical_result_replayed",
                "closed_conditions": [],
                "remaining_condition": "strict_msd",
                "strict_msd_result": self.fixture["private_history"].get("strict_msd_record"),
            }
        elif route == "run_postprocess_repair":
            self.result_changing_actions += 1
            self.pending_conditions = [condition for condition in self.pending_conditions if condition != "msd"]
            result_payload = {
                "route_result": "historical_result_replayed",
                "closed_conditions": ["msd"],
                "msd_result": self.fixture["private_history"].get("initial_msd_result"),
            }
        else:
            closed = [condition for condition in addresses if condition == "retained_audit_signal"]
            self.pending_conditions = [condition for condition in self.pending_conditions if condition not in closed]
            result_payload = {
                "route_result": "historical_review_record_opened",
                "closed_conditions": closed,
                "remaining_conditions": list(self.pending_conditions),
            }
        self.authorized_routes = [item for item in self.authorized_routes if item is not match]
        self.unlocked_results.append({"stage": "registered_route", "route": route, **result_payload})
        self.phase = "route_executed_recheck_required"
        self.last_check = None
        self.last_audit = None
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "route": route,
            **result_payload,
            "next_step": "Call rerun_result_checks.",
        }
        return self._record("execute_registered_route", arguments, result)

    def finish_case(self, decision: str, reason: str) -> dict[str, Any]:
        arguments = {"decision": decision, "reason": reason}
        if not self.started:
            return self._error("finish_case", "Call start_case first.", arguments)
        if self.finished:
            return self._error("finish_case", "The case has already been finished.", arguments)
        normalized = str(decision).strip().upper()
        if normalized not in DECISIONS:
            return self._error("finish_case", "decision must be PASS, REVIEW, or FAIL.", arguments)
        if not str(reason).strip():
            return self._error("finish_case", "A concise reason is required.", arguments)
        self.finished = True
        self.agent_decision = normalized
        self.agent_reason = str(reason).strip()
        self.phase = "finished"
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "recorded_decision": normalized,
            "all_result_conditions_closed": not self.pending_conditions,
            "remaining_unclosed_conditions": list(self.pending_conditions),
            "message": "Decision recorded. The hidden reference label is not returned to the agent.",
        }
        return self._record("finish_case", arguments, result)


def replay_full_framework(fixture_dir: Path) -> list[dict[str, Any]]:
    """Replay the deterministic historical route policy as an integrity test."""

    rows: list[dict[str, Any]] = []
    for fixture_path in sorted(fixture_dir.glob("*.json")):
        state = ReplayState.from_file(fixture_path)
        state.start_case(state.case_id)
        check = state.run_result_checks()
        guard = 0
        while check.get("verdict") == "WARN" and guard < 8:
            guard += 1
            audit = state.request_warn_audit()
            routes = audit.get("allowed_routes", [])
            if not routes:
                break
            priority = {
                "submit_md_clean_engine": 0,
                "run_postprocess_repair": 1,
                "open_review_packet": 2,
            }
            chosen = sorted(routes, key=lambda item: priority.get(item["route"], 99))[0]
            result = state.execute_registered_route(chosen["route"], chosen["parameters"])
            if result.get("status") != "ok":
                break
            check = state.rerun_result_checks()
            if chosen["route"] == "open_review_packet" and check.get("verdict") == "WARN":
                break
        decision = "PASS" if check.get("verdict") == "PASS" else "REVIEW"
        state.finish_case(decision, "deterministic full-workflow replay")
        reference = state.fixture["private_history"]["reference_decision"]
        rows.append(
            {
                "case_id": state.case_id,
                "replay_decision": decision,
                "reference_decision": reference,
                "match": decision == reference,
                "result_changing_actions": state.result_changing_actions,
                "tool_calls": state.tool_call_count,
            }
        )
    return rows


def validate_fixtures(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    fixture_paths = sorted((output_root / "fixtures").glob("*.json"))
    if len(fixture_paths) != 100:
        raise ValueError(f"expected 100 fixture files, found {len(fixture_paths)}")
    leak_rows: list[dict[str, Any]] = []
    for path in fixture_paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        leaks = _find_public_key_leaks(fixture["public_initial_evidence"])
        leaks.extend(_find_public_text_leaks(fixture["public_initial_evidence"]))
        strict_record = fixture["private_history"].get("strict_msd_record")
        if strict_record:
            leaks.extend(_find_public_text_leaks(strict_record, "strict_msd_record"))
        if leaks:
            leak_rows.append({"case_id": fixture["case_id"], "locations": "|".join(leaks)})
    replay_rows = replay_full_framework(output_root / "fixtures")
    counts = Counter(row["replay_decision"] for row in replay_rows)
    mismatch = [row for row in replay_rows if not row["match"]]
    _write_csv(output_root / "full_framework_replay.csv", replay_rows)
    _write_csv(output_root / "fixture_leak_report.csv", leak_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "fixture_count": len(fixture_paths),
        "public_leak_count": len(leak_rows),
        "full_framework_replay_counts": dict(counts),
        "full_framework_reference_mismatch_count": len(mismatch),
        "valid": not leak_rows and not mismatch and counts == Counter({"PASS": 92, "REVIEW": 8}),
    }
    _write_json(output_root / "fixture_validation.json", summary)
    if not summary["valid"]:
        raise RuntimeError(f"fixture validation failed: {summary}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-fixtures")
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate = sub.add_parser("validate")
    validate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    replay = sub.add_parser("replay-full")
    replay.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    pilot = sub.add_parser("select-pilot")
    pilot.add_argument("--index-csv", type=Path, default=DEFAULT_OUTPUT_ROOT / "fixture_index.csv")
    pilot.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "pilot_12_cases.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-fixtures":
        result = build_fixtures(args.output_root)
    elif args.command == "validate":
        result = validate_fixtures(args.output_root)
    elif args.command == "replay-full":
        rows = replay_full_framework(args.output_root / "fixtures")
        result = {"counts": dict(Counter(row["replay_decision"] for row in rows)), "rows": len(rows)}
    else:
        result = select_pilot(args.index_csv, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
