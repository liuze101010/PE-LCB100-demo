#!/usr/bin/env python3
"""Build and replay the PE-LCB100 ordinary ReAct baseline.

This environment removes the rule and audit interfaces. The Agent receives
raw evidence and primitive scientific operations. Reference decisions and
open-condition bookkeeping remain private and are used only by the scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = Path(r"${PELCB100_PAPER_ROOT}")
GUARDED_ROOT = SCRIPT_DIR / "output" / "pelcb100_react_baseline"
DEFAULT_RELEASE_TABLE = PAPER_ROOT / "SI" / "figures" / "si_pelcb100_ml_release_table_20260718.csv"
DEFAULT_MSD_CANDIDATES = PAPER_ROOT / "figures" / "fig6_msd_window_candidates.csv"
DEFAULT_INITIAL_MSD_DIAGNOSTICS = SCRIPT_DIR / "output" / "lcb300_integrated" / "MSD" / "summary" / "msd_quality_diagnostics.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "pelcb100_policy_free_react_baseline"

SCHEMA_VERSION = "pelcb100_policy_free_replay.v1"
MAX_RESULT_CHANGING_ACTIONS = 2
DECISIONS = {"PASS", "REVIEW", "FAIL"}

FORBIDDEN_PUBLIC_KEYS = {
    "initial_conditions",
    "reference_decision",
    "reference_full_pass",
    "publication_state",
    "final_state",
    "expected_decision",
    "allowed_routes",
    "primary_diagnosis",
    "open_conditions",
    "pending_conditions",
}
FORBIDDEN_PUBLIC_TEXT = (
    "reference_decision",
    "reference_full_pass",
    "publication_state",
    "final diagnosis",
    "terminal state",
    "retained for review",
    "allowed route",
    "primary diagnosis",
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_public_leaks(value: Any, path: str = "") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key).lower() in FORBIDDEN_PUBLIC_KEYS:
                leaks.append(here)
            leaks.extend(_find_public_leaks(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaks.extend(_find_public_leaks(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for token in FORBIDDEN_PUBLIC_TEXT:
            if token in lowered:
                leaks.append(f"{path}:{token}")
    return leaks


def _raw_values(row: Mapping[str, str], availability: Mapping[str, bool], density: float | None) -> dict[str, Any]:
    def present(name: str, value: Any) -> Any:
        return value if availability.get(name, False) else None

    return {
        "density_g_cm3": present("density", density),
        "radius_of_gyration_nm": present("Rg", _as_float(row.get("rg_mean_nm"))),
        "shape_relative_anisotropy": present(
            "shape_anisotropy", _as_float(row.get("shape_relative_anisotropy_mean"))
        ),
        "inter_chain_cc_rdf_first_peak": present(
            "RDF", _as_float(row.get("rdf_inter_chain_cc_first_peak_g"))
        ),
        "contact_probability_s10": present("contact_probability", _as_float(row.get("P_contact_rc0p8_s10"))),
        "diffusion_coefficient_m2_s": present("MSD", _as_float(row.get("target_D_mean_m2_s"))),
        "msd_loglog_slope": present("MSD", _as_float(row.get("target_loglog_slope"))),
        "fsqt_relaxation_time_ps": present("Fsqt", _as_float(row.get("target_tau_fs_ps"))),
    }


def _raw_initial_msd(
    row: Mapping[str, str],
    availability: Mapping[str, bool],
    detailed: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not availability.get("MSD", False):
        return {"available": False, "reason": "The first output package contains no MSD result."}
    record = {
        "available": True,
        "trajectory_length_ns": 400.0,
        "reported_fit_begin_ns": 100.0,
        "reported_fit_end_ns": 300.0,
        "loglog_slope": _as_float(row.get("target_loglog_slope")),
        "diffusion_coefficient_m2_s": _as_float(row.get("target_D_mean_m2_s")),
        "linear_R2": None,
        "local_D_cv_percent": None,
        "note": "The archived first report did not retain every window-stability statistic.",
    }
    if detailed:
        record.update(
            {
                "reported_fit_begin_ns": 100.0,
                "reported_fit_end_ns": 300.0,
                "loglog_slope": _as_float(detailed.get("loglog_slope_100_300ns")),
                "linear_R2": _as_float(detailed.get("R2_100_300ns")),
                "local_D_cv_percent": _as_float(detailed.get("window_D_cv_pct")),
                "note": "Raw statistics retained by the first MSD analysis.",
            }
        )
    return record


def _clean_raw_values(row: Mapping[str, str], density: float | None) -> dict[str, Any]:
    return {
        "density_g_cm3": density,
        "radius_of_gyration_nm": _as_float(row.get("result_rg_mean_nm")),
        "shape_relative_anisotropy": _as_float(row.get("result_shape_relative_anisotropy_mean")),
        "inter_chain_cc_rdf_first_peak": _as_float(row.get("result_rdf_inter_chain_cc_first_peak_g")),
        "contact_probability_s10": _as_float(row.get("result_contact_rc0p8_s10")),
        "diffusion_coefficient_m2_s": _as_float(row.get("result_D_mean_m2_s")),
        "msd_loglog_slope": _as_float(row.get("result_loglog_slope")),
        "fsqt_relaxation_time_ps": _as_float(row.get("result_tau_fs_ps")),
    }


def _clean_msd_record(row: Mapping[str, str], prior: Mapping[str, Any] | None) -> dict[str, Any]:
    if prior:
        return {"available": True, **dict(prior)}
    return {
        "available": True,
        "trajectory_length_ns": 400.0,
        "fit_begin_ns": 100.0,
        "fit_end_ns": 300.0,
        "loglog_slope": _as_float(row.get("result_loglog_slope")),
        "linear_R2": None,
        "local_D_cv_percent": _as_float(row.get("result_msd_window_D_cv_pct")),
    }


def _raw_strict_record(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    return {
        "trajectory_length_ns": record.get("trajectory_length_ns"),
        "fit_begin_ns": record.get("fit_begin_ns"),
        "fit_end_ns": record.get("fit_end_ns"),
        "loglog_slope": record.get("loglog_slope"),
        "linear_R2": record.get("linear_R2"),
        "local_D_cv_percent": record.get("local_D_cv_percent"),
        "fitting_window_found": record.get("strict_reassessment") == "PASS",
    }


def _raw_candidate_prior(row: Mapping[str, str] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "trajectory_length_ns": _as_float(row.get("max_time_ns")),
        "fit_begin_ns": _as_float(row.get("prior_begin_ns")),
        "fit_end_ns": _as_float(row.get("prior_end_ns")),
        "loglog_slope": _as_float(row.get("prior_alpha")),
        "linear_R2": None,
        "local_D_cv_percent": None,
    }


def build_fixtures(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    release_table: Path = DEFAULT_RELEASE_TABLE,
    msd_candidates: Path = DEFAULT_MSD_CANDIDATES,
    initial_msd_diagnostics: Path = DEFAULT_INITIAL_MSD_DIAGNOSTICS,
) -> dict[str, Any]:
    guarded_index = _read_csv(GUARDED_ROOT / "fixture_index.csv")
    release = {row["case_id"]: row for row in _read_csv(release_table)}
    candidates = {row["case_id"]: row for row in _read_csv(msd_candidates)}
    detailed_msd: dict[int, dict[str, str]] = {}
    for diagnostic in _read_csv(initial_msd_diagnostics):
        prefix = str(diagnostic.get("case_id", "")).split("_", 1)[0]
        if prefix.startswith("lcb") and prefix[3:].isdigit():
            detailed_msd[int(prefix[3:])] = diagnostic
    if len(guarded_index) != 100 or len(release) != 100:
        raise ValueError("expected 100 guarded fixtures and 100 release-table rows")

    fixture_dir = output_root / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    evidence_audit_rows: list[dict[str, Any]] = []
    leak_rows: list[dict[str, Any]] = []
    for index_row in guarded_index:
        case_id = index_row["case_id"]
        guarded_path = Path(index_row["fixture_path"])
        guarded = _load_json(guarded_path)
        row = release[case_id]
        initial = guarded["public_initial_evidence"]
        availability = dict(initial["first_postprocessing_record"]["output_available"])
        case_sequence = int(float(row["case_seq"]))
        public_initial = {
            "case_id": case_id,
            "case_short": guarded["case_short"],
            "design": {
                "branch_points": _as_float(row.get("branch_points")),
                "branch_carbons_per_branch": _as_float(row.get("branch_carbons")),
                "actual_total_carbons_per_chain": _as_float(row.get("actual_total_carbons_per_chain")),
                "branch_position_pattern": row.get("branch_position_pattern", ""),
            },
            "first_output_package": {
                "output_available": availability,
                "missing_outputs": list(initial["first_postprocessing_record"]["missing_outputs"]),
                "raw_values": _raw_values(
                    row, availability, initial["first_postprocessing_record"].get("density_g_cm3")
                ),
                "msd_fit_record": _raw_initial_msd(row, availability, detailed_msd.get(case_sequence)),
            },
        }
        initial_conditions = list(guarded["private_history"]["initial_conditions"])
        strict_record = _raw_strict_record(guarded["private_history"].get("strict_msd_record"))
        clean_prior = _raw_candidate_prior(candidates.get(case_id))
        clean_outputs = {
            "output_available": {key: True for key in availability},
            "missing_outputs": [],
            "raw_values": _clean_raw_values(row, guarded["private_history"].get("density_after_g_cm3")),
            "msd_fit_record": _clean_msd_record(row, clean_prior),
        }
        if guarded["private_history"]["reference_decision"] == "REVIEW" and strict_record:
            reference_path_msd = strict_record
        elif "msd" in initial_conditions or clean_prior:
            reference_path_msd = strict_record or guarded["private_history"].get("initial_msd_result") or {}
        elif {"density", "missing_output"} & set(initial_conditions):
            reference_path_msd = clean_outputs["msd_fit_record"]
        else:
            reference_path_msd = public_initial["first_output_package"]["msd_fit_record"]
        msd_statistics_complete = all(
            reference_path_msd.get(key) is not None
            for key in ("loglog_slope", "linear_R2", "local_D_cv_percent")
        )
        fixture = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "case_short": guarded["case_short"],
            "public_initial_evidence": public_initial,
            "raw_history": {
                "initial_configuration": {
                    "force_field": row.get("quality_force_field", ""),
                    "engine_version": row.get("quality_engine_version", ""),
                    "pressure_verlet_buffer_tolerance": None,
                    "simulation_temperature_K": 400.0,
                    "simulation_pressure_bar": 1.0,
                },
                "clean_configuration": {
                    "engine_profile": "gromacs_2024_4_clean",
                    "pressure_verlet_buffer_tolerance": 0.5,
                },
                "clean_output_package": clean_outputs,
                "clean_msd_before_refit": clean_prior,
                "msd_refit_result": strict_record or guarded["private_history"].get("initial_msd_result"),
                "trajectory_extension_result": strict_record,
            },
            "private_reference": {
                "initial_conditions": initial_conditions,
                "reference_decision": guarded["private_history"]["reference_decision"],
                "md_rerun_available": bool({"density", "missing_output"} & set(initial_conditions)),
                "msd_refit_available": "msd" in initial_conditions or strict_record is not None,
                "trajectory_extension_available": (
                    guarded["private_history"]["reference_decision"] == "REVIEW" and strict_record is not None
                ),
                "reference_path_msd_statistics_complete": msd_statistics_complete,
            },
        }
        leaks = _find_public_leaks(public_initial)
        if leaks:
            leak_rows.append({"case_id": case_id, "locations": leaks})
        path = fixture_dir / f"{case_id}.json"
        _write_json(path, fixture)
        index_rows.append(
            {
                "case_id": case_id,
                "case_short": guarded["case_short"],
                "fixture_path": str(path),
                "fixture_sha256": _sha256_file(path),
                "initial_conditions": "|".join(initial_conditions) if initial_conditions else "none",
                "reference_decision": guarded["private_history"]["reference_decision"],
                "reference_path_msd_statistics_complete": msd_statistics_complete,
            }
        )
        evidence_audit_rows.append(
            {
                "case_id": case_id,
                "case_short": guarded["case_short"],
                "reference_decision": guarded["private_history"]["reference_decision"],
                "initial_conditions": "|".join(initial_conditions) if initial_conditions else "none",
                "reference_path_msd_statistics_complete": msd_statistics_complete,
                "loglog_slope_available": reference_path_msd.get("loglog_slope") is not None,
                "linear_R2_available": reference_path_msd.get("linear_R2") is not None,
                "local_D_cv_available": reference_path_msd.get("local_D_cv_percent") is not None,
            }
        )

    if leak_rows:
        raise ValueError(f"public evidence leaks private benchmark state: {leak_rows[:3]}")
    _write_csv(output_root / "fixture_index.csv", index_rows)
    _write_csv(output_root / "evidence_completeness_audit.csv", evidence_audit_rows)
    pilot = _load_json(GUARDED_ROOT / "pilot_12_cases.json")
    pilot["schema_version"] = SCHEMA_VERSION
    _write_json(output_root / "pilot_12_cases.json", pilot)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(index_rows),
        "terminal_counts": dict(Counter(row["reference_decision"] for row in index_rows)),
        "interface": "raw evidence and primitive operations; no rule or audit result is exposed",
        "maximum_result_changing_actions": MAX_RESULT_CHANGING_ACTIONS,
        "sources": {
            "guarded_fixture_index": {
                "path": str(GUARDED_ROOT / "fixture_index.csv"),
                "sha256": _sha256_file(GUARDED_ROOT / "fixture_index.csv"),
            },
            "release_table": {"path": str(release_table), "sha256": _sha256_file(release_table)},
            "msd_candidates": {"path": str(msd_candidates), "sha256": _sha256_file(msd_candidates)},
            "initial_msd_diagnostics": {
                "path": str(initial_msd_diagnostics),
                "sha256": _sha256_file(initial_msd_diagnostics),
            },
        },
    }
    _write_json(output_root / "fixture_manifest.json", manifest)
    return manifest


@dataclass
class PolicyFreeReplayState:
    fixture: dict[str, Any]
    state_path: Path | None = None
    tool_log_path: Path | None = None
    started: bool = False
    phase: str = "not_started"
    current_output_package: dict[str, Any] = field(default_factory=dict)
    current_configuration: dict[str, Any] = field(default_factory=dict)
    latest_msd_record: dict[str, Any] = field(default_factory=dict)
    unlocked_evidence: list[dict[str, Any]] = field(default_factory=list)
    evaluator_open_issues: list[str] = field(default_factory=list)
    result_changing_actions: int = 0
    invalid_operation_calls: int = 0
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
    ) -> "PolicyFreeReplayState":
        return cls(fixture=_load_json(fixture_path), state_path=state_path, tool_log_path=tool_log_path)

    @property
    def case_id(self) -> str:
        return self.fixture["case_id"]

    def _state_for_scorer(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": self.case_id,
            "started": self.started,
            "phase": self.phase,
            "unlocked_evidence": self.unlocked_evidence,
            "evaluator_open_issues": self.evaluator_open_issues,
            "result_changing_actions_used": self.result_changing_actions,
            "result_changing_actions_remaining": MAX_RESULT_CHANGING_ACTIONS - self.result_changing_actions,
            "invalid_operation_calls": self.invalid_operation_calls,
            "finished": self.finished,
            "agent_decision": self.agent_decision,
            "agent_reason": self.agent_reason,
            "tool_call_count": self.tool_call_count,
        }

    def _persist(self) -> None:
        if self.state_path:
            _write_json(self.state_path, self._state_for_scorer())

    def _visible_hash(self) -> str:
        return _sha256_payload(
            {
                "initial": self.fixture["public_initial_evidence"] if self.started else {},
                "unlocked": self.unlocked_evidence,
            }
        )

    def _record(self, tool: str, arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
        self.tool_call_count += 1
        record = {
            "recorded_at": datetime.now().isoformat(timespec="milliseconds"),
            "case_id": self.case_id,
            "tool": tool,
            "arguments": dict(arguments),
            "result": dict(result),
            "visible_evidence_sha256": self._visible_hash(),
            "evaluator_open_issues_after_step": list(self.evaluator_open_issues),
            "scorer_state_sha256": _sha256_payload(self._state_for_scorer()),
        }
        if self.tool_log_path:
            _append_jsonl(self.tool_log_path, record)
        self._persist()
        return dict(result)

    def _error(self, tool: str, message: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._record(tool, arguments or {}, {"status": "error", "message": message})

    def _require_active(self, tool: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.started:
            return self._error(tool, "Call start_case first.", arguments)
        if self.finished:
            return self._error(tool, "The case has already been finished.", arguments)
        return None

    def _consume_action(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.result_changing_actions >= MAX_RESULT_CHANGING_ACTIONS:
            self.invalid_operation_calls += 1
            return self._error(tool, "The two result-changing actions have already been used.", arguments)
        self.result_changing_actions += 1
        return None

    def _unlock(self, stage: str, payload: Mapping[str, Any]) -> None:
        self.unlocked_evidence.append({"stage": stage, **dict(payload)})

    def start_case(self, case_id: str) -> dict[str, Any]:
        if self.started:
            return self._error("start_case", "This isolated session already has an active case.", {"case_id": case_id})
        if case_id != self.case_id:
            return self._error("start_case", "The requested case is unavailable in this session.", {"case_id": case_id})
        self.started = True
        self.phase = "initial_evidence"
        self.current_output_package = dict(self.fixture["public_initial_evidence"]["first_output_package"])
        self.current_configuration = dict(self.fixture["raw_history"]["initial_configuration"])
        self.latest_msd_record = dict(self.current_output_package.get("msd_fit_record", {}))
        initial = [item for item in self.fixture["private_reference"]["initial_conditions"] if item != "retained_audit_signal"]
        self.evaluator_open_issues = list(initial)
        if self.fixture["private_reference"]["reference_decision"] == "REVIEW" and "strict_msd" not in self.evaluator_open_issues:
            self.evaluator_open_issues.append("strict_msd")
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "design": self.fixture["public_initial_evidence"]["design"],
            "message": "The first output package is available through the inspection tools.",
            "maximum_result_changing_actions": MAX_RESULT_CHANGING_ACTIONS,
        }
        return self._record("start_case", {"case_id": case_id}, result)

    def inspect_outputs(self) -> dict[str, Any]:
        blocked = self._require_active("inspect_outputs")
        if blocked:
            return blocked
        return self._record(
            "inspect_outputs",
            {},
            {"status": "ok", "case_id": self.case_id, "output_package": self.current_output_package},
        )

    def inspect_run_configuration(self) -> dict[str, Any]:
        blocked = self._require_active("inspect_run_configuration")
        if blocked:
            return blocked
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "current_configuration": self.current_configuration,
            "registered_runtime_profiles": [
                {
                    "engine_profile": "gromacs_2021_2_archive",
                    "pressure_verlet_buffer_tolerance": None,
                },
                self.fixture["raw_history"]["clean_configuration"],
            ],
            "registered_analysis_requests": [
                {"window_strategy": "bounded_window_scan"},
                {"target_time_ns": 800.0},
            ],
        }
        return self._record("inspect_run_configuration", {}, result)

    def query_reference_data(self, topic: str) -> dict[str, Any]:
        arguments = {"topic": topic}
        blocked = self._require_active("query_reference_data", arguments)
        if blocked:
            return blocked
        normalized = str(topic).strip().lower()
        if normalized == "density":
            evidence = {
                "topic": "density",
                "system": "approximately 400 K and 1 bar polyolefin melt",
                "radonpy_record_count": 40,
                "converted_external_min_g_cm3": 0.7316,
                "converted_external_max_g_cm3": 0.8666,
                "rounded_screening_interval_g_cm3": [0.73, 0.87],
            }
        elif normalized == "msd":
            evidence = {
                "topic": "msd",
                "scientific_target": "stable free-diffusion window",
                "bounded_window_protocol": {
                    "loglog_slope_interval": [0.95, 1.05],
                    "minimum_linear_R2": 0.95,
                    "maximum_local_D_cv_percent": 5.0,
                    "minimum_start_fraction_of_trajectory": 0.25,
                    "maximum_end_fraction_of_trajectory": 0.85,
                    "minimum_width_fraction_of_trajectory": 0.30,
                },
            }
        elif normalized == "required_outputs":
            evidence = {
                "topic": "required_outputs",
                "names": ["density", "Rg", "RDF", "shape_anisotropy", "MSD", "Fsqt", "contact_probability"],
            }
        else:
            return self._error("query_reference_data", "topic must be density, msd, or required_outputs.", arguments)
        self._unlock("reference_data", evidence)
        return self._record("query_reference_data", arguments, {"status": "ok", "reference_data": evidence})

    def inspect_msd_diagnostics(self) -> dict[str, Any]:
        blocked = self._require_active("inspect_msd_diagnostics")
        if blocked:
            return blocked
        return self._record(
            "inspect_msd_diagnostics",
            {},
            {"status": "ok", "case_id": self.case_id, "msd_record": self.latest_msd_record},
        )

    def rerun_md(self, engine_profile: str, pressure_verlet_buffer_tolerance: float | None = None) -> dict[str, Any]:
        arguments = {
            "engine_profile": engine_profile,
            "pressure_verlet_buffer_tolerance": pressure_verlet_buffer_tolerance,
        }
        blocked = self._require_active("rerun_md", arguments)
        if blocked:
            return blocked
        expected = self.fixture["raw_history"]["clean_configuration"]
        matches = (
            engine_profile == expected["engine_profile"]
            and pressure_verlet_buffer_tolerance == expected["pressure_verlet_buffer_tolerance"]
        )
        if not self.fixture["private_reference"]["md_rerun_available"] or not matches:
            self.invalid_operation_calls += 1
            return self._error("rerun_md", "No archived result exists for this case and setting combination.", arguments)
        consumed = self._consume_action("rerun_md", arguments)
        if consumed:
            return consumed
        self.current_configuration = dict(expected)
        self.current_output_package = dict(self.fixture["raw_history"]["clean_output_package"])
        self.latest_msd_record = dict(self.current_output_package.get("msd_fit_record", {}))
        self.evaluator_open_issues = [
            issue for issue in self.evaluator_open_issues if issue not in {"density", "missing_output"}
        ]
        payload = {
            "operation": "md_rerun",
            "configuration": self.current_configuration,
            "new_output_package": self.current_output_package,
        }
        self._unlock("md_rerun", payload)
        self.phase = "md_rerun_complete"
        return self._record("rerun_md", arguments, {"status": "ok", **payload})

    def run_msd_analysis(self, window_strategy: str) -> dict[str, Any]:
        arguments = {"window_strategy": window_strategy}
        blocked = self._require_active("run_msd_analysis", arguments)
        if blocked:
            return blocked
        if window_strategy != "bounded_window_scan" or not self.fixture["private_reference"]["msd_refit_available"]:
            self.invalid_operation_calls += 1
            return self._error("run_msd_analysis", "No archived result exists for this analysis request.", arguments)
        if not self.current_output_package.get("output_available", {}).get("MSD", False):
            self.invalid_operation_calls += 1
            return self._error("run_msd_analysis", "The current output package contains no MSD trajectory.", arguments)
        consumed = self._consume_action("run_msd_analysis", arguments)
        if consumed:
            return consumed
        raw = dict(self.fixture["raw_history"].get("msd_refit_result") or {})
        raw.pop("strict_reassessment", None)
        raw.pop("unresolved_check", None)
        if not raw:
            self.invalid_operation_calls += 1
            return self._error("run_msd_analysis", "The archived analysis result is empty.", arguments)
        self.latest_msd_record = {"available": True, **raw}
        fitting_window_found = raw.get("fitting_window_found")
        if fitting_window_found is False:
            if "strict_msd" not in self.evaluator_open_issues:
                self.evaluator_open_issues.append("strict_msd")
            self.evaluator_open_issues = [issue for issue in self.evaluator_open_issues if issue != "msd"]
        else:
            self.evaluator_open_issues = [
                issue for issue in self.evaluator_open_issues if issue not in {"msd", "strict_msd"}
            ]
        payload = {"operation": "msd_bounded_window_scan", "msd_record": self.latest_msd_record}
        self._unlock("msd_analysis", payload)
        self.phase = "msd_analysis_complete"
        return self._record("run_msd_analysis", arguments, {"status": "ok", **payload})

    def extend_trajectory(self, target_time_ns: float) -> dict[str, Any]:
        arguments = {"target_time_ns": target_time_ns}
        blocked = self._require_active("extend_trajectory", arguments)
        if blocked:
            return blocked
        if target_time_ns != 800 or not self.fixture["private_reference"]["trajectory_extension_available"]:
            self.invalid_operation_calls += 1
            return self._error("extend_trajectory", "No archived result exists for this trajectory request.", arguments)
        consumed = self._consume_action("extend_trajectory", arguments)
        if consumed:
            return consumed
        raw = dict(self.fixture["raw_history"].get("trajectory_extension_result") or {})
        raw.pop("strict_reassessment", None)
        raw.pop("unresolved_check", None)
        self.latest_msd_record = {"available": True, **raw}
        if "strict_msd" not in self.evaluator_open_issues:
            self.evaluator_open_issues.append("strict_msd")
        payload = {"operation": "trajectory_extension", "target_time_ns": 800.0, "msd_record": self.latest_msd_record}
        self._unlock("trajectory_extension", payload)
        self.phase = "trajectory_extension_complete"
        return self._record("extend_trajectory", arguments, {"status": "ok", **payload})

    def finish_case(self, decision: str, reason: str) -> dict[str, Any]:
        arguments = {"decision": decision, "reason": reason}
        blocked = self._require_active("finish_case", arguments)
        if blocked:
            return blocked
        normalized = str(decision).strip().upper()
        if normalized not in DECISIONS:
            return self._error("finish_case", "decision must be PASS, REVIEW, or FAIL.", arguments)
        if not str(reason).strip():
            return self._error("finish_case", "A concise reason is required.", arguments)
        self.finished = True
        self.agent_decision = normalized
        self.agent_reason = str(reason).strip()
        self.phase = "finished"
        return self._record(
            "finish_case",
            arguments,
            {
                "status": "ok",
                "case_id": self.case_id,
                "recorded_decision": normalized,
                "message": "Decision recorded.",
            },
        )


def validate_fixtures(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    rows = _read_csv(output_root / "fixture_index.csv")
    leaks: list[dict[str, Any]] = []
    for row in rows:
        fixture = _load_json(Path(row["fixture_path"]))
        found = _find_public_leaks(fixture["public_initial_evidence"])
        if found:
            leaks.append({"case_id": row["case_id"], "locations": found})
    result = {
        "case_count": len(rows),
        "public_initial_evidence_leak_count": len(leaks),
        "terminal_counts": dict(Counter(row["reference_decision"] for row in rows)),
        "leaks": leaks,
    }
    _write_json(output_root / "fixture_validation.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        result = build_fixtures(args.output_root)
    else:
        result = validate_fixtures(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
