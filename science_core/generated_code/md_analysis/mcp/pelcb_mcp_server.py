#!/usr/bin/env python3
"""Guarded MCP server for polymer molecular-dynamics automation.

The server exposes polymer-general preparation, rule-profile, quality-control,
and guarded routing tools. PE-LCB100 tools remain available as a validated
case-study implementation. The server does not accept arbitrary shell commands
or edit MD physics, rule thresholds, locked splits, or expert labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from mcp.server.fastmcp import FastMCP


SCRIPT_PATH = Path(__file__).resolve()
MD_ANALYSIS_DIR = SCRIPT_PATH.parents[1]
GENERATED_CODE_DIR = MD_ANALYSIS_DIR.parent
PROJECT_ROOT = GENERATED_CODE_DIR.parent
if str(MD_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(MD_ANALYSIS_DIR))

from polymer_design_validator import (  # noqa: E402
    list_registered_adapters,
    validate_pe_lcb_legacy_candidate,
    validate_polymer_design_candidate as validate_polymer_design_candidate_core,
)
from polymer_runtime_registry import list_runtime_capabilities  # noqa: E402
from polymer_builder_dispatch import (  # noqa: E402
    list_builder_materializers,
    materialize_registered_polymer_input,
)
from pelcb_design_origin import (  # noqa: E402
    get_design_origin_record as get_design_origin_record_core,
    get_design_origin_summary as get_design_origin_summary_core,
)
from pelcb_audit_diagnosis import (  # noqa: E402
    diagnose_case,
    registered_module_specs,
    repair_tool_affordances,
    route_from_diagnosis,
    route_from_primary_diagnosis,
    validate_repair_tool_call as validate_repair_tool_call_core,
    warn_diagnosis_taxonomy,
)
from diagnosis_candidate_registry import (  # noqa: E402
    ALLOWLIST_TAG as DIAGNOSIS_PROFILE_ALLOWLIST_TAG,
    DEFAULT_DIAGNOSIS_PROFILE_DIR,
    PROJECT_ID as DIAGNOSIS_PROFILE_PROJECT_ID,
    activate_diagnosis_profile as activate_diagnosis_profile_core,
    collect_diagnosis_candidates,
    list_active_diagnosis_profiles,
    propose_diagnosis_profile_from_candidates as propose_diagnosis_profile_from_candidates_core,
    write_candidate_ledger,
)
from pelcb_warn_audit_llm import (  # noqa: E402
    ALLOWLIST_TAG as WARN_AUDIT_ALLOWLIST_TAG,
    DEFAULT_OUTPUT_DIR as WARN_AUDIT_OUTPUT_DIR,
    audit_warn_case,
    make_static_clients,
    run_warn_audit_batch as run_warn_audit_batch_core,
)
from polymer_qc_rules import apply_polymer_qc_rules as apply_polymer_qc_rules_core  # noqa: E402
from rule_evidence_discovery import (  # noqa: E402
    discover_rule_evidence as discover_rule_evidence_core,
    list_rule_evidence_providers as list_rule_evidence_providers_core,
)
from rule_profile_registry import (  # noqa: E402
    DEFAULT_RULE_PROFILE_DIR,
    activate_rule_profile as activate_rule_profile_core,
    apply_rule_gate as apply_rule_gate_core,
    get_rule_profile as get_rule_profile_core,
    list_supported_rule_properties as list_supported_rule_properties_core,
    propose_rule_profile_from_sources as propose_rule_profile_from_sources_core,
    propose_similarity_transfer_profile as propose_similarity_transfer_profile_core,
    rule_profile_automation_policy,
    rule_profile_missing_warn_source,
    rule_profile_source_taxonomy,
)
from structure_similarity_transfer import (  # noqa: E402
    DEFAULT_HIGH_SIMILARITY,
    DEFAULT_MIN_SIMILARITY,
)
from zotero_rag_design_prior import (  # noqa: E402
    ALLOWLIST_TAG as DESIGN_PRIOR_ALLOWLIST_TAG,
    DEFAULT_PELCB_DESIGN_DOIS,
    DEFAULT_OUTPUT_DIR as DEFAULT_ZOTERO_RAG_DIR,
    PROJECT_ID as DESIGN_PRIOR_PROJECT_ID,
    activate_design_prior_profile as activate_design_prior_profile_core,
    get_design_prior_profile as get_design_prior_profile_core,
    get_zotero_rag_corpus_summary as get_zotero_rag_corpus_summary_core,
    propose_design_prior_profile_from_rag as propose_design_prior_profile_from_rag_core,
    retrieve_design_prior_evidence as retrieve_design_prior_evidence_core,
)
from lcb_project_config import load_lcb_config  # noqa: E402


def _project_config_path() -> str | Path:
    return (
        os.environ.get("PELCB_PROJECT_CONFIG")
        or os.environ.get("LCB_PROJECT_CONFIG")
        or MD_ANALYSIS_DIR / "configs" / "pelcb100.json"
    )


PROJECT_CONFIG_PATH = Path(os.path.expandvars(str(_project_config_path()))).expanduser()
if not PROJECT_CONFIG_PATH.is_absolute():
    PROJECT_CONFIG_PATH = PROJECT_ROOT / PROJECT_CONFIG_PATH
PROJECT_CONFIG_PATH = PROJECT_CONFIG_PATH.resolve()
PROJECT_CONFIG = load_lcb_config(PROJECT_CONFIG_PATH)
AGGREGATE_DIR = PROJECT_CONFIG.integrated_output_dir
A3_DIR = PROJECT_CONFIG.a3_results_dir
RUNTIME_DIR = PROJECT_CONFIG.runtime_output_dir
O7_QUEUE_DIR = PROJECT_CONFIG.o7_queue_dir
O7_CLOSURE_DIR = PROJECT_CONFIG.o7_closure_dir
SIDECAR_DIR = MD_ANALYSIS_DIR / "output" / "pelcb_mcp_sidecar"
POLYMER_PREPARATION_DIR = SIDECAR_DIR / "polymer_preparation"
LEDGER_PATH = SIDECAR_DIR / "mcp_tool_ledger.jsonl"
REPAIR_LOOP = MD_ANALYSIS_DIR / "pelcb_repair_loop.py"
NVT_EXTENSION = MD_ANALYSIS_DIR / "pelcb_clean_nvt_extension.py"
REPAIR_LOOP_DIR = PROJECT_CONFIG.repair_loop_dir
MSD_WINDOW_REPAIR = MD_ANALYSIS_DIR / "pelcb_msd_window_repair.py"
MSD_WINDOW_OVERLAY = MD_ANALYSIS_DIR / "pelcb_build_msd_window_overlay.py"
PACKING_VOID_REPAIR = MD_ANALYSIS_DIR / "pelcb_packing_void.py"
OVERLAY_DIR = PROJECT_CONFIG.overlay_dir
PACKING_VOID_OUTPUT_DIR = PROJECT_CONFIG.packing_void_output_dir
WARN_AUDIT_OUTPUT_DIR = PROJECT_CONFIG.warn_audit_output_dir
RULE_PROFILE_DIR = PROJECT_CONFIG.rule_profile_dir or DEFAULT_RULE_PROFILE_DIR
DIAGNOSIS_PROFILE_DIR = Path(os.environ.get("PELCB_DIAGNOSIS_PROFILE_DIR", DEFAULT_DIAGNOSIS_PROFILE_DIR))
ZOTERO_RAG_DIR = Path(os.environ.get("PELCB_ZOTERO_RAG_DIR", DEFAULT_ZOTERO_RAG_DIR))

EXECUTION_ALLOWLIST = {
    "polymer_md_preparation": {"POLYMER_MCP_EXECUTE_PREPARATION"},
    "o7clean": {"PELCB_MCP_EXECUTE_O7CLEAN"},
    "pelcb_agent_design_pilot": {"PELCB_MCP_EXECUTE_AGENT_DESIGN_PILOT"},
    "pelcb_repair_loop": {"PELCB_MCP_EXECUTE_REPAIR_LOOP"},
    "pelcb_postprocess_repair": {"PELCB_MCP_EXECUTE_POSTPROCESS_REPAIR"},
    "pelcb_rule_profiles": {"PELCB_MCP_EXECUTE_RULE_PROFILE"},
    DIAGNOSIS_PROFILE_PROJECT_ID: {DIAGNOSIS_PROFILE_ALLOWLIST_TAG},
    DESIGN_PRIOR_PROJECT_ID: {DESIGN_PRIOR_ALLOWLIST_TAG},
}

ALLOWED_BRANCH_POINTS = {1, 2, 3, 4, 5}
ALLOWED_BRANCH_CARBONS = {3, 4, 5, 8, 12, 16, 20}


def _ensure_import_paths() -> None:
    for path in (MD_ANALYSIS_DIR, GENERATED_CODE_DIR / "lcb_md_pipeline"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_jsonish(value: str | dict[str, Any] | list[Any]) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _slug_text(value: Any) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "unspecified"


def _ledger(tool_name: str, request: dict[str, Any], result: dict[str, Any]) -> None:
    _write_jsonl(
        LEDGER_PATH,
        {
            "timestamp_utc": _utc_now(),
            "tool_name": tool_name,
            "input_hash": _hash_payload(request),
            "execute": bool(request.get("execute", False)),
            "project_id": request.get("project_id"),
            "allowlist_tag": request.get("allowlist_tag"),
            "reason": request.get("reason"),
            "result_status": result.get("status"),
            "affected_path": result.get("affected_path"),
            "result_hash": _hash_payload(result),
        },
    )


def _require_execution(tool_name: str, request: dict[str, Any]) -> dict[str, Any] | None:
    project_id = str(request.get("project_id", "") or "")
    tag = str(request.get("allowlist_tag", "") or "")
    reason = str(request.get("reason", "") or "").strip()
    if not request.get("execute"):
        return {"status": "rejected", "reason": "execute=true is required for this mutating MCP tool"}
    if project_id not in EXECUTION_ALLOWLIST:
        return {"status": "rejected", "reason": f"project_id {project_id!r} is not in the sidecar execution allowlist"}
    if tag not in EXECUTION_ALLOWLIST[project_id]:
        return {"status": "rejected", "reason": "allowlist_tag does not match the requested project_id"}
    if len(reason) < 12:
        return {"status": "rejected", "reason": "a specific reason with at least 12 characters is required"}
    return None


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "features": _read_csv(AGGREGATE_DIR / "ml_features.csv"),
        "targets": _read_csv(AGGREGATE_DIR / "ml_targets.csv"),
        "mask": _read_csv(AGGREGATE_DIR / "ml_quality_mask.csv"),
        "confidence": _read_csv(AGGREGATE_DIR / "result_confidence.csv"),
        "design": _read_csv(AGGREGATE_DIR / "pelcb100_design_matrix_audit.csv"),
    }


def _case_row(df: pd.DataFrame, case_id: str) -> dict[str, Any]:
    if df.empty or "case_id" not in df.columns:
        return {}
    rows = df[df["case_id"].astype(str) == str(case_id)]
    if rows.empty:
        return {}
    return _json_safe(rows.iloc[0].to_dict())


def _case_confidence(tables: dict[str, pd.DataFrame], case_id: str, mask: Mapping[str, Any]) -> dict[str, Any]:
    confidence = tables.get("confidence", pd.DataFrame())
    if confidence is not None and not confidence.empty and "case_id" in confidence.columns:
        rows = confidence[confidence["case_id"].astype(str) == str(case_id)]
        if not rows.empty:
            observables: dict[str, Any] = {}
            for _, row in rows.iterrows():
                payload = _json_safe(row.to_dict())
                breakdown = payload.get("confidence_breakdown_json")
                if isinstance(breakdown, str) and breakdown.strip():
                    try:
                        payload["confidence_breakdown"] = json.loads(breakdown)
                    except json.JSONDecodeError:
                        payload["confidence_breakdown"] = breakdown
                observables[str(payload.get("observable") or "")] = payload
            return {"schema_version": "result_confidence.v1", "case_id": case_id, "observables": observables}

    observables = {}
    for key, value in mask.items():
        if not str(key).startswith("confidence_") or not str(key).endswith("_confidence_tier"):
            continue
        observable = str(key)[len("confidence_") : -len("_confidence_tier")]
        observables[observable] = {"confidence_tier": value}
    return {"schema_version": "result_confidence.v1", "case_id": case_id, "observables": observables}


def _existing_design_keys(features: pd.DataFrame) -> set[tuple[int, int, str]]:
    if features.empty:
        return set()
    keys: set[tuple[int, int, str]] = set()
    for _, row in features.iterrows():
        try:
            keys.add(
                (
                    int(row["branch_points"]),
                    int(row["branch_carbons"]),
                    str(row["branch_position_pattern"]),
                )
            )
        except Exception:
            continue
    return keys


def _allowed_patterns_by_bp(features: pd.DataFrame) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {bp: set() for bp in ALLOWED_BRANCH_POINTS}
    if features.empty:
        return out
    for _, row in features.iterrows():
        try:
            out[int(row["branch_points"])].add(str(row["branch_position_pattern"]))
        except Exception:
            continue
    return out


def _actual_carbon_budget(branch_points: int, branch_carbons: int, carbon_budget: int) -> tuple[int, int]:
    branch_total = branch_points * branch_carbons
    actual_total = carbon_budget if (carbon_budget - branch_total) % 2 == 0 else carbon_budget + 1
    backbone_repeat_units = (actual_total - branch_total) // 2
    return actual_total, backbone_repeat_units


def get_case_summary_impl(case_id: str) -> dict[str, Any]:
    tables = _tables()
    features = _case_row(tables["features"], case_id)
    targets = _case_row(tables["targets"], case_id)
    mask = _case_row(tables["mask"], case_id)
    if not features and not targets and not mask:
        return {"status": "not_found", "case_id": case_id}
    return {
        "status": "ok",
        "case_id": case_id,
        "features": features,
        "targets": targets,
        "quality_mask": mask,
        "result_confidence": _case_confidence(tables, case_id, mask),
    }


def get_release_manifest_impl() -> dict[str, Any]:
    tables = _tables()
    manifest = _read_json(AGGREGATE_DIR / "ml_dataset_manifest.json", {})
    mask = tables["mask"]
    summary: dict[str, Any] = {}
    if not mask.empty:
        summary = {
            "case_count": int(len(mask)),
            "ml_usable_count": int(mask["ml_usable"].fillna(False).astype(bool).sum()) if "ml_usable" in mask else None,
            "case_status_counts": mask["case_status"].fillna("missing").value_counts().to_dict()
            if "case_status" in mask
            else {},
            "density_tier_counts": mask["density_tier_si_o"].fillna("missing").value_counts().to_dict()
            if "density_tier_si_o" in mask
            else {},
            "locked_eval_count": int(mask["a3_locked_eval_member"].fillna(False).astype(bool).sum())
            if "a3_locked_eval_member" in mask
            else None,
        }
    return {"status": "ok", "manifest": _json_safe(manifest), "summary": _json_safe(summary)}


def get_design_origin_impl(case_id: str) -> dict[str, Any]:
    """Return read-only PE-LCB100 design-origin provenance for one case."""
    return get_design_origin_record_core(case_id)


def get_design_origin_summary_impl() -> dict[str, Any]:
    """Return read-only PE-LCB100 design-origin provenance summary."""
    return get_design_origin_summary_core()


def get_a3_metrics_impl() -> dict[str, Any]:
    metrics = _read_json(A3_DIR / "ablation_metrics.json", {})
    if not metrics:
        return {"status": "not_found", "path": str(A3_DIR / "ablation_metrics.json")}
    compact = {
        "created_at": metrics.get("created_at"),
        "split": metrics.get("split"),
        "n_samples": metrics.get("n_samples"),
        "n_groups": metrics.get("n_groups"),
        "model_names": metrics.get("model_names"),
        "class_distribution_global": metrics.get("class_distribution_global"),
        "per_baseline": {},
    }
    for name, row in (metrics.get("per_baseline") or {}).items():
        compact["per_baseline"][name] = {
            "n": row.get("n"),
            "accuracy": row.get("accuracy"),
            "macro_f1": row.get("macro_f1"),
            "balanced_accuracy": row.get("balanced_accuracy"),
            "ci_95_percentile_bootstrap": row.get("ci_95_percentile_bootstrap"),
        }
    return {"status": "ok", "metrics": _json_safe(compact)}


def get_audit_record_impl(case_id: str, decision_type: str | None = None, mode: str | None = None) -> dict[str, Any]:
    rows = _read_jsonl(A3_DIR / "raw_judgments.jsonl")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("case_id")) != str(case_id):
            continue
        if decision_type and str(row.get("decision_type")) != str(decision_type):
            continue
        if mode and str(row.get("mode")) != str(mode):
            continue
        selected.append(row)
    return {"status": "ok", "case_id": case_id, "count": len(selected), "records": _json_safe(selected)}


def get_registered_repair_modules_impl() -> dict[str, Any]:
    return {
        "status": "ok",
        "modules": registered_module_specs(),
        "boundary": {
            "rule_gate": "flags PASS/WARN/FAIL from frozen release evidence; WARN triggers diagnosis rather than naming a cause",
            "audit_diagnosis": "classifies cause, evidence basis, deterministic re-entry module, and concrete tool parameters",
            "repair_router": "invokes only registered deterministic tools under sidecar execution guards",
        },
    }


def list_warn_diagnosis_taxonomy_impl() -> dict[str, Any]:
    taxonomy = warn_diagnosis_taxonomy(profile_dir=DIAGNOSIS_PROFILE_DIR)
    return {"status": "ok", **_json_safe(taxonomy)}


def _read_json_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_diagnosis_candidates_impl(diagnosis_jsonl: str | None = None) -> dict[str, Any]:
    if diagnosis_jsonl:
        records = _read_json_records(diagnosis_jsonl)
        summary = write_candidate_ledger(records, DIAGNOSIS_PROFILE_DIR / "candidate_review")
        return {"status": "ok", **_json_safe(summary)}
    candidates = []
    candidate_dir = DIAGNOSIS_PROFILE_DIR / "candidate_review"
    candidate_jsonl = candidate_dir / "diagnosis_candidate_ledger.jsonl"
    if candidate_jsonl.exists():
        candidates = _read_json_records(candidate_jsonl)
    return {
        "status": "ok",
        "candidate_count": len(candidates),
        "eligible_count": sum(1 for item in candidates if item.get("freeze_readiness") == "eligible_for_profile_freeze"),
        "candidates": _json_safe(candidates),
        "active_profiles": _json_safe(list_active_diagnosis_profiles(DIAGNOSIS_PROFILE_DIR)),
    }


def propose_diagnosis_profile_from_candidates_impl(candidate_json: str, candidate_id: str | None = None) -> dict[str, Any]:
    payload = json.loads(candidate_json)
    profile = propose_diagnosis_profile_from_candidates_core(payload, candidate_id=candidate_id)
    return {"status": "ok", "profile": _json_safe(profile)}


def activate_diagnosis_profile_impl(
    profile_json: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    request = {
        "tool": "activate_diagnosis_profile",
        "project_id": project_id,
        "reason": reason,
        "allowlist_tag": allowlist_tag,
        "execute": execute,
    }
    profile = json.loads(profile_json)
    result = activate_diagnosis_profile_core(
        profile,
        profile_dir=DIAGNOSIS_PROFILE_DIR,
        project_id=project_id,
        reason=reason,
        allowlist_tag=allowlist_tag,
        execute=execute,
    )
    _ledger("activate_diagnosis_profile", request, result)
    return _json_safe(result)


def get_audit_diagnosis_impl(case_id: str) -> dict[str, Any]:
    mask_row = _case_row(_tables()["mask"], case_id)
    if not mask_row:
        return {"status": "not_found", "case_id": case_id}
    diagnosis = diagnose_case(mask_row)
    return {
        "status": "ok",
        "case_id": case_id,
        "diagnosis": _json_safe(diagnosis),
        "deterministic_module": _json_safe(registered_module_specs().get(diagnosis["recommended_reentry_module"], {})),
    }


def _planning_route_options(diagnosis: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnosis_names = [
        diagnosis.get("primary_audit_diagnosis") or diagnosis.get("audit_diagnosis"),
        *(diagnosis.get("secondary_audit_diagnoses") or []),
    ]
    options: list[dict[str, Any]] = []
    seen_tools: set[str] = set()
    for diagnosis_name in diagnosis_names:
        if not diagnosis_name:
            continue
        route = route_from_primary_diagnosis(str(diagnosis_name))
        tool = str(route.get("repair_tool") or "")
        if not tool or tool in seen_tools:
            continue
        seen_tools.add(tool)
        options.append(
            {
                "diagnosis": str(diagnosis_name),
                "repair_tool": tool,
                "tool_parameters": dict(route.get("tool_parameters") or {}),
                "repair_route": route.get("repair_route"),
                "recommended_reentry_module": route.get("module"),
            }
        )
    return options


def plan_repair_route_impl(
    case_id: str,
    planning_proposal_json: str = "",
) -> dict[str, Any]:
    diagnosis_result = get_audit_diagnosis_impl(case_id)
    if diagnosis_result["status"] != "ok":
        return diagnosis_result
    diagnosis = diagnosis_result["diagnosis"]
    planning_options = _planning_route_options(diagnosis)
    if not planning_proposal_json:
        return {
            "status": "ok",
            "dry_run": True,
            "case_id": case_id,
            "rule_verdict": diagnosis.get("rule_verdict"),
            "warn_sources": diagnosis.get("warn_sources", []),
            "warn_source": diagnosis.get("warn_source"),
            "primary_audit_diagnosis": diagnosis.get("primary_audit_diagnosis"),
            "secondary_audit_diagnoses": diagnosis.get("secondary_audit_diagnoses", []),
            "audit_diagnosis": diagnosis.get("audit_diagnosis"),
            "diagnosis_basis": diagnosis.get("diagnosis_basis", []),
            "allowed_routes": planning_options,
            "planning_selection_status": "awaiting_planning_model",
            "route_selected_by": "none",
            "planning_selection": {},
            "planning_guard": {},
            "repair_tool": "",
            "tool_parameters": {},
            "repair_route": "",
            "repair_action": "",
            "automation_level": "awaiting_guarded_selection",
            "executor": "",
            "deterministic_module": {},
            "tool_plan": {},
            "execution_guard": {
                "mutating_tools_require": ["execute=true", "project_id", "reason", "allowlist_tag"],
                "canonical_release_modified": False,
            },
        }
    planning_selection: dict[str, Any] | None = None
    planning_guard: dict[str, Any] | None = None
    try:
        proposal = json.loads(planning_proposal_json)
        if not isinstance(proposal, dict):
            raise ValueError("planning_proposal_json must decode to an object")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "case_id": case_id,
            "planning_selection_status": "rejected",
            "planning_rejection_reason": str(exc),
            "allowed_routes": planning_options,
        }
    allowed_by_tool = {str(item["repair_tool"]): item for item in planning_options}
    selected_tool = str(proposal.get("tool") or proposal.get("repair_tool") or "")
    if selected_tool not in allowed_by_tool:
        return {
            "status": "error",
            "case_id": case_id,
            "planning_selection_status": "rejected",
            "planning_rejection_reason": f"tool {selected_tool or '<missing>'} is not allowed for this audit diagnosis",
            "allowed_routes": planning_options,
        }
    planning_rationale = str(proposal.get("rationale") or "").strip()
    if not planning_rationale:
        return {
            "status": "error",
            "case_id": case_id,
            "planning_selection_status": "rejected",
            "planning_rejection_reason": "planning model must explain why the selected route matches the audit result",
            "allowed_routes": planning_options,
        }
    if not proposal.get("parameters") and not proposal.get("tool_parameters"):
        proposal["parameters"] = dict(allowed_by_tool[selected_tool].get("tool_parameters") or {})
    planning_guard = validate_repair_tool_call_core(proposal, diagnosis)
    if planning_guard.get("guard_verdict") != "pass":
        return {
            "status": "error",
            "case_id": case_id,
            "planning_selection_status": "rejected",
            "planning_rejection_reason": planning_guard.get("guard_rejection_reason", "guard rejected planning proposal"),
            "allowed_routes": planning_options,
            "planning_guard": _json_safe(planning_guard),
        }
    planning_selection = dict(planning_guard["proposal"])
    diagnosis = {
        **diagnosis,
        "guard_verdict": "pass",
        "guard_rejection_reason": "",
        "llm_tool_call_proposal": planning_selection,
    }
    route = route_from_diagnosis(diagnosis)
    return {
        "status": "ok",
        "dry_run": True,
        "case_id": case_id,
        "rule_verdict": diagnosis.get("rule_verdict"),
        "warn_sources": diagnosis.get("warn_sources", []),
        "warn_source": diagnosis.get("warn_source"),
        "primary_audit_diagnosis": diagnosis.get("primary_audit_diagnosis"),
        "secondary_audit_diagnoses": diagnosis.get("secondary_audit_diagnoses", []),
        "audit_diagnosis": diagnosis.get("audit_diagnosis"),
        "diagnosis_basis": diagnosis.get("diagnosis_basis", []),
        "recommended_reentry_module": route.get("module"),
        "recommended_action": route.get("repair_action"),
        "repair_tool": route.get("repair_tool"),
        "tool_parameters": route.get("tool_parameters", {}),
        "repair_route": route["repair_route"],
        "repair_action": route["repair_action"],
        "automation_level": route["automation_level"],
        "executor": route["executor"],
        "allowed_routes": planning_options,
        "planning_selection_status": "selected",
        "route_selected_by": "planning_model",
        "planning_selection": _json_safe(planning_selection or {}),
        "planning_selection_rationale": planning_rationale,
        "planning_guard": _json_safe(planning_guard or {}),
        "deterministic_module": _json_safe(route["deterministic_module"]),
        "tool_plan": _json_safe(route["tool_plan"]),
        "execution_guard": {
            "mutating_tools_require": ["execute=true", "project_id", "reason", "allowlist_tag"],
            "canonical_release_modified": False,
        },
    }


def list_repair_tool_affordances_impl() -> dict[str, Any]:
    return {
        "status": "ok",
        "tools": _json_safe(repair_tool_affordances()),
        "guard_policy": {
            "tool_names_must_be_registered": True,
            "parameters_must_be_allowlisted": True,
            "forbidden_runtime_changes": [
                "arbitrary shell",
                "runtime rule threshold edits",
                "unallowlisted mdp physics edits",
                "canonical release edits",
                "locked label edits",
            ],
        },
    }


def validate_repair_tool_call_impl(proposal_json: str, context_json: str | None = None) -> dict[str, Any]:
    try:
        proposal = json.loads(proposal_json)
        if not isinstance(proposal, dict):
            raise ValueError("proposal_json must decode to an object")
        context = json.loads(context_json) if context_json else {}
        if not isinstance(context, dict):
            raise ValueError("context_json must decode to an object")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "guard_verdict": "reject",
            "guard_rejection_reason": str(exc),
        }
    result = validate_repair_tool_call_core(proposal, context)
    return {"status": "ok", **_json_safe(result)}


def run_warn_audit_diagnosis_impl(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    force_refresh: bool = False,
    use_mock_clients: bool = False,
) -> dict[str, Any]:
    """Run tri-model WARN cause attribution plus repair/modification planning."""
    mask_row = _case_row(_tables()["mask"], case_id)
    if not mask_row:
        return {"status": "not_found", "case_id": case_id}
    fallback = diagnose_case(mask_row)
    if fallback.get("rule_verdict") == "PASS":
        return {"status": "ok", "case_id": case_id, "dry_run": True, "diagnosis": _json_safe(fallback)}
    request = {
        "tool": "run_warn_audit_diagnosis",
        "case_id": case_id,
        "project_id": project_id,
        "reason": reason,
        "allowlist_tag": allowlist_tag,
        "execute": execute,
        "force_refresh": force_refresh,
        "use_mock_clients": use_mock_clients,
    }
    if not execute:
        route = route_from_diagnosis(fallback)
        return {
            "status": "ok",
            "dry_run": True,
            "case_id": case_id,
            "diagnosis": _json_safe(
                {
                    **fallback,
                    "diagnosis_source": "dry_run_deterministic_preview",
                    "modification_plan": {},
                    "model_votes": [],
                    "model_errors": [],
                    "llm_model_count": 0,
                }
            ),
            "tool_plan": _json_safe(route["tool_plan"]),
            "execution_guard": {
                "required_for_real_llm_call": [
                    "execute=true",
                    "project_id=pelcb_repair_loop",
                    f"allowlist_tag={WARN_AUDIT_ALLOWLIST_TAG}",
                    "reason",
                ]
            },
        }
    if project_id != "pelcb_repair_loop" or allowlist_tag != WARN_AUDIT_ALLOWLIST_TAG or len(str(reason).strip()) < 12:
        rejected = {
            "status": "rejected",
            "reason": "tri-model WARN audit requires project_id=pelcb_repair_loop, execute=true, allowlist tag, and reason",
            "required_allowlist_tag": WARN_AUDIT_ALLOWLIST_TAG,
        }
        _ledger("run_warn_audit_diagnosis", request, rejected)
        return rejected
    clients = make_static_clients() if use_mock_clients else None
    diagnosis = audit_warn_case(mask_row, clients=clients, cache_dir=WARN_AUDIT_OUTPUT_DIR / "mcp_cache", force_refresh=force_refresh)
    route = route_from_diagnosis(diagnosis)
    result = {
        "status": "ok",
        "dry_run": False,
        "case_id": case_id,
        "diagnosis": _json_safe(diagnosis),
        "tool_plan": _json_safe(route["tool_plan"]),
    }
    _ledger("run_warn_audit_diagnosis", request, result)
    return result


def run_warn_audit_batch_impl(
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    max_cases: int = 0,
    case_ids: list[str] | None = None,
    use_mock_clients: bool = False,
) -> dict[str, Any]:
    """Run the tri-model WARN diagnosis loop over all or selected WARN cases."""
    request = {
        "tool": "run_warn_audit_batch",
        "project_id": project_id,
        "reason": reason,
        "allowlist_tag": allowlist_tag,
        "execute": execute,
        "max_cases": max_cases,
        "case_ids": case_ids or [],
        "use_mock_clients": use_mock_clients,
    }
    if not execute:
        summary = run_warn_audit_batch_core(
            output_dir=WARN_AUDIT_OUTPUT_DIR,
            cases=case_ids or None,
            max_cases=max_cases,
            execute=False,
        )
        return {"status": "ok", "dry_run": True, "summary": _json_safe(summary)}
    if project_id != "pelcb_repair_loop" or allowlist_tag != WARN_AUDIT_ALLOWLIST_TAG or len(str(reason).strip()) < 12:
        rejected = {
            "status": "rejected",
            "reason": "tri-model WARN audit batch requires project_id=pelcb_repair_loop, execute=true, allowlist tag, and reason",
            "required_allowlist_tag": WARN_AUDIT_ALLOWLIST_TAG,
        }
        _ledger("run_warn_audit_batch", request, rejected)
        return rejected
    summary = run_warn_audit_batch_core(
        output_dir=WARN_AUDIT_OUTPUT_DIR,
        cases=case_ids or None,
        max_cases=max_cases,
        clients=make_static_clients() if use_mock_clients else None,
        execute=True,
        allowlist_tag=allowlist_tag,
        reason=reason,
    )
    result = {"status": "ok", "dry_run": False, "summary": _json_safe(summary)}
    _ledger("run_warn_audit_batch", request, result)
    return result


def get_compute_status_impl() -> dict[str, Any]:
    status = {
        "runtime_cases": str(RUNTIME_DIR / "runtime_cases.csv"),
        "runtime_active_processes": str(RUNTIME_DIR / "runtime_active_processes.csv"),
        "runtime_gpu_snapshot": str(RUNTIME_DIR / "runtime_gpu_snapshot.csv"),
        "o7_queue_dir": str(O7_QUEUE_DIR),
        "o7_closure_summary": str(O7_CLOSURE_DIR / "o7_clean_closure_summary.json"),
    }
    tables = {
        "cases": _read_csv(RUNTIME_DIR / "runtime_cases.csv"),
        "active": _read_csv(RUNTIME_DIR / "runtime_active_processes.csv"),
        "gpus": _read_csv(RUNTIME_DIR / "runtime_gpu_snapshot.csv"),
    }
    summary = {name: int(len(df)) for name, df in tables.items()}
    closure = _read_json(O7_CLOSURE_DIR / "o7_clean_closure_summary.json", {})
    return {"status": "ok", "paths": status, "summary": summary, "o7_closure": _json_safe(closure)}


def validate_design_candidate_impl(
    branch_points: int,
    branch_carbons: int,
    branch_position_pattern: str,
    carbon_budget: int = 100,
) -> dict[str, Any]:
    return validate_pe_lcb_legacy_candidate(
        branch_points=branch_points,
        branch_carbons=branch_carbons,
        branch_position_pattern=branch_position_pattern,
        carbon_budget=carbon_budget,
        existing_features=_tables()["features"],
    )


def validate_polymer_design_candidate_impl(candidate_json: str | dict[str, Any]) -> dict[str, Any]:
    return validate_polymer_design_candidate_core(candidate_json, existing_features=_tables()["features"])


def list_polymer_preparation_capabilities_impl() -> dict[str, Any]:
    """Return registered chemistry adapters and executable runtime capabilities."""

    return {
        "status": "ok",
        "polymer_adapters": list_registered_adapters(),
        "runtime_capabilities": list_runtime_capabilities(),
        "builder_materializers": list_builder_materializers(),
        "scope_policy": {
            "registered_means_technically_checkable": True,
            "parameterization_probe_precedes_md_when_required": True,
            "external_reference_qualification_precedes_publication_pass": True,
            "unregistered_chemistry_is_routed_to_review": True,
        },
    }


def score_candidate_against_reliability_map_impl(
    branch_points: int,
    branch_carbons: int,
    branch_position_pattern: str,
    target_contact: float | None = None,
) -> dict[str, Any]:
    validation = validate_design_candidate_impl(branch_points, branch_carbons, branch_position_pattern)
    tables = _tables()
    features = tables["features"]
    mask = tables["mask"]
    joined = features.merge(mask[["case_id", "ml_usable", "case_status"]], on="case_id", how="left") if not features.empty and not mask.empty else pd.DataFrame()

    cell = joined[
        (joined["branch_points"].astype(int) == int(branch_points))
        & (joined["branch_carbons"].astype(int) == int(branch_carbons))
    ] if not joined.empty else pd.DataFrame()
    bp_slice = joined[joined["branch_points"].astype(int) == int(branch_points)] if not joined.empty else pd.DataFrame()
    global_rate = float(joined["ml_usable"].fillna(False).astype(bool).mean()) if not joined.empty else None
    cell_rate = float(cell["ml_usable"].fillna(False).astype(bool).mean()) if not cell.empty else None
    bp_rate = float(bp_slice["ml_usable"].fillna(False).astype(bool).mean()) if not bp_slice.empty else None
    predicted_ml_usable = cell_rate if cell_rate is not None else bp_rate if bp_rate is not None else global_rate

    design = tables["design"]
    design_cell = design[
        (design["branch_points"].astype(int) == int(branch_points))
        & (design["branch_carbons"].astype(int) == int(branch_carbons))
    ] if not design.empty else pd.DataFrame()
    case_count = int(design_cell.iloc[0]["case_count"]) if not design_cell.empty else 0
    coverage_gain_score = 1.0 / (1.0 + case_count)

    contact_col = "P_contact_rc0p8_s20"
    contact_estimate = None
    contact_hit_score = None
    if not cell.empty and contact_col in cell:
        contact_estimate = float(cell[contact_col].dropna().median()) if not cell[contact_col].dropna().empty else None
    if contact_estimate is None and contact_col in joined and not joined[contact_col].dropna().empty:
        contact_estimate = float(joined[contact_col].dropna().median())
    if target_contact is not None and contact_estimate is not None:
        contact_hit_score = max(0.0, 1.0 - abs(float(target_contact) - contact_estimate) / max(abs(float(target_contact)), 1e-6))

    return {
        "status": "ok",
        "validation": validation,
        "scores": {
            "coverage_gain_score": coverage_gain_score,
            "cell_case_count": case_count,
            "predicted_ml_usable_probability": predicted_ml_usable,
            "target_contact": target_contact,
            "contact_estimate_P_contact_rc0p8_s20": contact_estimate,
            "target_contact_hit_score": contact_hit_score,
        },
    }


def plan_md_package_impl(
    candidate_id: str | None = None,
    branch_points: int | None = None,
    branch_carbons: int | None = None,
    branch_position_pattern: str | None = None,
    candidate_json: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if candidate_json:
        validation = validate_polymer_design_candidate_impl(candidate_json)
        planned_candidate_id = str(validation.get("builder_spec", {}).get("case_id") or validation.get("normalized_candidate", {}).get("candidate_id") or "")
        return {
            "status": "ok",
            "dry_run": True,
            "candidate_id": planned_candidate_id,
            "validation": validation,
            "builder_spec": validation.get("builder_spec", {}),
            "planned_outputs": {
                "sidecar_project": "polymer_md_design_pilot",
                "candidate_manifest_row": f"{planned_candidate_id}.json" if planned_candidate_id else "<candidate_id>.json",
                "md_package": f"{planned_candidate_id}_md_<run_tag>" if planned_candidate_id else "<candidate_id>_md_<run_tag>",
                "note": "No files are generated by this MCP dry-run planner.",
            },
        }
    missing = [
        name
        for name, value in {
            "candidate_id": candidate_id,
            "branch_points": branch_points,
            "branch_carbons": branch_carbons,
            "branch_position_pattern": branch_position_pattern,
        }.items()
        if value is None
    ]
    if missing:
        return {"status": "rejected", "reason": f"missing legacy package fields: {', '.join(missing)}"}
    validation = validate_design_candidate_impl(int(branch_points), int(branch_carbons), str(branch_position_pattern))
    return {
        "status": "ok",
        "dry_run": True,
        "candidate_id": candidate_id,
        "validation": validation,
        "planned_outputs": {
            "sidecar_project": "pelcb_agent_design_pilot",
            "candidate_manifest_row": f"{candidate_id}.json",
            "md_package": f"{candidate_id}_md_<run_tag>",
            "note": "No files are generated by this MCP dry-run planner.",
        },
    }


def validate_or_build_emc_input_impl(candidate_json: str | dict[str, Any]) -> dict[str, Any]:
    validation = validate_polymer_design_candidate_impl(candidate_json)
    status = "ok" if validation.get("validator_status") == "pass" and validation.get("builder_ready") else "rejected"
    return {
        "status": status,
        "dry_run": True,
        "validation": validation,
        "builder_spec": validation.get("builder_spec", {}),
        "automatic_setup": validation.get("automatic_setup", {}),
        "next_action": validation.get("next_action"),
        "input_validation_route": validation.get("input_validation_route"),
        "md_submission_allowed": validation.get("md_submission_allowed"),
        "return_to_researcher": validation.get("return_to_researcher"),
        "stop_reason": validation.get("stop_reason"),
        "planned_outputs": {
            "case_json": "<sidecar>/<candidate_id>/case.json",
            "builder_input": "<sidecar>/<candidate_id>/<builder-specific-input>",
            "parameterization_probe": "registered builder followed by topology and engine-input checks",
            "note": "This tool validates registered construction and force-field options. Builder-specific materialization verifies atom typing and topology before MD.",
        },
    }


def validate_or_plan_polymer_input_impl(candidate_json: str | dict[str, Any]) -> dict[str, Any]:
    """Polymer-general name for the guarded preparation planning entry point."""

    return validate_or_build_emc_input_impl(candidate_json)


def materialize_polymer_input_impl(
    candidate_json: str | dict[str, Any],
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    request = locals().copy()
    rejected = _require_execution("materialize_polymer_input", request)
    if rejected:
        _ledger("materialize_polymer_input", request, rejected)
        return rejected
    validation = validate_polymer_design_candidate_impl(candidate_json)
    if validation.get("validator_status") != "pass" or not validation.get("builder_ready"):
        result = {
            "status": "rejected",
            "reason": "polymer candidate did not pass molecule-preparation validation",
            "validation": validation,
            "input_validation_route": validation.get("input_validation_route"),
            "return_to_researcher": validation.get("return_to_researcher"),
            "stop_reason": validation.get("stop_reason"),
        }
        _ledger("materialize_polymer_input", request, result)
        return result
    builder_spec = dict(validation.get("builder_spec") or {})
    case_id = str(builder_spec.get("case_id") or "")
    try:
        artifacts = materialize_registered_polymer_input(
            POLYMER_PREPARATION_DIR / case_id,
            builder_spec,
            validation=validation,
            plan={"source": "materialize_polymer_input", "project_id": project_id},
        )
        result = {
            "status": "ok",
            "affected_path": artifacts.get("case_dir"),
            "validation": validation,
            "artifacts": artifacts,
            "next_action": validation.get("next_action"),
        }
    except Exception as exc:
        result = {"status": "error", "reason": str(exc), "validation": validation}
    _ledger("materialize_polymer_input", request, result)
    return _json_safe(result)


def plan_supervisor_slots_impl(refresh: bool = False, max_slots: int = 12) -> dict[str, Any]:
    _ensure_import_paths()
    import lcb300_supervisor as supervisor  # noqa: PLC0415

    dashboard = None
    if refresh:
        dashboard = supervisor.refresh_dashboard()
        if not dashboard.get("ok"):
            return {"status": "dashboard_failed", "dashboard": dashboard}
    runtime = supervisor.load_runtime()
    slots = supervisor.compute_slots(runtime.get("gpus", pd.DataFrame()), runtime.get("active", pd.DataFrame()))
    return {
        "status": "ok",
        "dry_run": True,
        "dashboard": dashboard,
        "slot_count": len(slots),
        "slots": _json_safe(slots[: max(0, int(max_slots))]),
    }


def plan_postprocess_closure_impl(project_id: str = "o7clean") -> dict[str, Any]:
    if project_id != "o7clean":
        return {"status": "rejected", "reason": "only o7clean postprocess closure is currently exposed"}
    ready = _read_csv(O7_CLOSURE_DIR / "o7_clean_full_postprocess_ready.csv")
    summary = _read_json(O7_CLOSURE_DIR / "o7_clean_closure_summary.json", {})
    return {
        "status": "ok",
        "dry_run": True,
        "project_id": project_id,
        "ready_count": int(len(ready)),
        "ready_rows": _json_safe(ready.head(50).to_dict(orient="records")) if not ready.empty else [],
        "summary": _json_safe(summary),
    }


def get_repair_loop_latest_impl() -> dict[str, Any]:
    latest = _read_json(REPAIR_LOOP_DIR / "repair_loop_latest.json", {})
    if not latest:
        return {"status": "not_found", "path": str(REPAIR_LOOP_DIR / "repair_loop_latest.json")}
    if "route_counts" in latest:
        summary = latest
    else:
        summary_path_text = str(latest.get("summary_json") or latest.get("summary") or "")
        summary_path = Path(summary_path_text) if summary_path_text else None
        summary = _read_json(summary_path, {}) if summary_path and summary_path.exists() and summary_path.is_file() else {}
    return {
        "status": "ok",
        "latest": _json_safe(latest),
        "summary": _json_safe(summary),
    }


def plan_repair_loop_impl(include_optional_validation: bool = False, tag: str | None = None) -> dict[str, Any]:
    run_tag = tag or f"mcp_repair_loop_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command = [sys.executable, str(REPAIR_LOOP), "--config", str(PROJECT_CONFIG_PATH), "--tag", run_tag]
    if include_optional_validation:
        command.append("--include-optional-validation")
    run_result = _run_subprocess(command, timeout=3600)
    latest = get_repair_loop_latest_impl()
    return {
        "status": "planned" if run_result["ok"] else "failed",
        "dry_run": True,
        "tag": run_tag,
        "affected_path": str(REPAIR_LOOP_DIR / run_tag),
        "command": command,
        "subprocess": run_result,
        "latest": latest,
    }


def get_msd_window_repair_latest_impl() -> dict[str, Any]:
    latest = _read_json(REPAIR_LOOP_DIR / "msd_window_repair_latest.json", {})
    if not latest:
        return {"status": "not_found", "path": str(REPAIR_LOOP_DIR / "msd_window_repair_latest.json")}
    return {"status": "ok", "summary": _json_safe(latest)}


def plan_msd_window_repair_impl(tag: str | None = None, case_ids: list[str] | None = None) -> dict[str, Any]:
    run_tag = tag or f"mcp_msd_window_repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command = [sys.executable, str(MSD_WINDOW_REPAIR), "--mask", str(AGGREGATE_DIR / "ml_quality_mask.csv"),
               "--output-dir", str(REPAIR_LOOP_DIR), "--tag", run_tag]
    for case_id in case_ids or []:
        command.extend(["--case-id", str(case_id)])
    run_result = _run_subprocess(command, timeout=3600)
    latest = get_msd_window_repair_latest_impl()
    return {
        "status": "planned" if run_result["ok"] else "failed",
        "dry_run": True,
        "tag": run_tag,
        "affected_path": str(REPAIR_LOOP_DIR / run_tag),
        "command": command,
        "subprocess": run_result,
        "latest": latest,
    }


def get_msd_window_overlay_latest_impl() -> dict[str, Any]:
    latest = _read_json(OVERLAY_DIR / "msd_window_overlay_latest.json", {})
    if not latest:
        return {"status": "not_found", "path": str(OVERLAY_DIR / "msd_window_overlay_latest.json")}
    return {"status": "ok", "summary": _json_safe(latest)}


def plan_msd_window_overlay_impl(tag: str | None = None) -> dict[str, Any]:
    run_tag = tag or f"mcp_msd_window_overlay_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command = [sys.executable, str(MSD_WINDOW_OVERLAY), "--mask", str(AGGREGATE_DIR / "ml_quality_mask.csv"),
               "--targets", str(AGGREGATE_DIR / "ml_targets.csv"), "--output-root", str(OVERLAY_DIR),
               "--tag", run_tag]
    run_result = _run_subprocess(command, timeout=1800)
    latest = get_msd_window_overlay_latest_impl()
    return {
        "status": "planned" if run_result["ok"] else "failed",
        "dry_run": True,
        "tag": run_tag,
        "affected_path": str(OVERLAY_DIR / run_tag),
        "command": command,
        "subprocess": run_result,
        "latest": latest,
    }


def get_packing_void_latest_impl() -> dict[str, Any]:
    latest = _read_json(PACKING_VOID_OUTPUT_DIR / "packing_void_latest.json", {})
    if not latest:
        return {"status": "not_found", "path": str(PACKING_VOID_OUTPUT_DIR / "packing_void_latest.json")}
    return {"status": "ok", "summary": _json_safe(latest)}


def plan_packing_void_repair_impl(tag: str | None = None, case_ids: list[str] | None = None) -> dict[str, Any]:
    run_tag = tag or f"mcp_packing_void_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command = [
        sys.executable,
        str(PACKING_VOID_REPAIR),
        "--summary",
        str(PROJECT_CONFIG.integrated_summary_csv),
        "--tag",
        run_tag,
        "--output-dir",
        str(PACKING_VOID_OUTPUT_DIR),
    ]
    for case_id in case_ids or []:
        command.extend(["--case-id", str(case_id)])
    run_result = _run_subprocess(command, timeout=3600)
    latest = get_packing_void_latest_impl()
    return {
        "status": "planned" if run_result["ok"] else "failed",
        "dry_run": True,
        "repair_kind": "packing_void",
        "tag": run_tag,
        "affected_path": str(PACKING_VOID_OUTPUT_DIR / run_tag),
        "command": command,
        "subprocess": run_result,
        "latest": latest,
    }


def run_repair_loop_impl(
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    max_execute_md: int = 0,
    max_execute_density: int = 0,
    run_closure: bool = False,
    auto_submit_postprocess: bool = False,
    refresh_publication_assets: bool = False,
    include_optional_validation: bool = False,
) -> dict[str, Any]:
    request = locals().copy()
    rejected = _require_execution("run_repair_loop", request)
    if rejected:
        _ledger("run_repair_loop", request, rejected)
        return rejected
    if project_id != "pelcb_repair_loop":
        result = {"status": "rejected", "reason": "only pelcb_repair_loop execution is implemented for this tool"}
        _ledger("run_repair_loop", request, result)
        return result

    run_tag = f"mcp_repair_loop_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command = [
        sys.executable,
        str(REPAIR_LOOP),
        "--config",
        str(PROJECT_CONFIG_PATH),
        "--execute",
        "--allowlist-tag",
        "PELCB_REPAIR_LOOP_EXECUTE",
        "--reason",
        reason,
        "--max-execute-md",
        str(max(0, int(max_execute_md))),
        "--max-execute-density",
        str(max(0, int(max_execute_density))),
        "--tag",
        run_tag,
    ]
    if run_closure:
        command.append("--run-closure")
    if auto_submit_postprocess:
        command.append("--auto-submit-postprocess")
    if refresh_publication_assets:
        command.append("--refresh-publication-assets")
    if include_optional_validation:
        command.append("--include-optional-validation")

    run_result = _run_subprocess(command, timeout=7200)
    result = {
        "status": "executed" if run_result["ok"] else "failed",
        "affected_path": str(REPAIR_LOOP_DIR / run_tag),
        "command": command,
        "subprocess": run_result,
        "latest": get_repair_loop_latest_impl(),
    }
    _ledger("run_repair_loop", request, result)
    return result


def extend_clean_nvt_prod_impl(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    """Submit one checkpoint-aware clean NVT extension through the registered guard.

    The extension length and MD physics remain owned by
    ``pelcb_clean_nvt_extension.py``. This MCP boundary intentionally exposes no
    temperature, timestep, coupling, cutoff, constraint, or ``.mdp`` options.
    """
    request = locals().copy()
    rejected = _require_execution("extend_clean_nvt_prod", request)
    if rejected:
        _ledger("extend_clean_nvt_prod", request, rejected)
        return rejected
    if project_id != "pelcb_repair_loop":
        result = {
            "status": "rejected",
            "reason": "only pelcb_repair_loop clean NVT extension is implemented for this tool",
        }
        _ledger("extend_clean_nvt_prod", request, result)
        return result
    if not str(case_id or "").strip():
        result = {"status": "rejected", "reason": "case_id is required"}
        _ledger("extend_clean_nvt_prod", request, result)
        return result

    run_tag = f"mcp_clean_nvt_extension_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    extension_output_dir = PROJECT_CONFIG.supervisor_output_dir / "o7_clean_nvt_extension"
    command = [
        sys.executable,
        str(NVT_EXTENSION),
        "--config",
        str(PROJECT_CONFIG_PATH),
        "--status-csv",
        str(O7_CLOSURE_DIR / "o7_clean_run_status.csv"),
        "--queue-dir",
        str(O7_QUEUE_DIR),
        "--output-dir",
        str(extension_output_dir),
        "--tag",
        run_tag,
        "--cases",
        str(case_id).strip(),
        "--execute",
        "--allowlist-tag",
        "PELCB_CLEAN_NVT_EXTENSION",
        "--reason",
        reason,
    ]
    run_result = _run_subprocess(command, timeout=1200)
    result = {
        "status": "submitted" if run_result["ok"] else "failed",
        "case_id": str(case_id).strip(),
        "affected_path": str(extension_output_dir / run_tag),
        "command": command,
        "subprocess": run_result,
        "execution_policy": {
            "checkpoint_resume_required": True,
            "extension_length_source": "pelcb_clean_nvt_extension.py deterministic default",
            "mdp_physics_editable": False,
            "completion_requires": [
                "clean trajectory completion",
                "full clean postprocessing",
                "strict MSD publication re-gate",
            ],
        },
    }
    _ledger("extend_clean_nvt_prod", request, result)
    return result


def _terminate_process_tree(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        else:
            import signal

            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def _run_subprocess(command: list[str], cwd: Path = PROJECT_ROOT, timeout: int = 1800) -> dict[str, Any]:
    # Hang-proof: child stdout/stderr go to temp files, never to a PIPE, so a
    # detached grandchild (e.g. a backgrounded remote launch that inherits the
    # child's handles) cannot block this process; on timeout the whole process
    # tree is terminated and the temp files are read (file reads never block).
    timed_out = False
    popen_kwargs: dict[str, Any] = {"cwd": str(cwd), "stdin": subprocess.DEVNULL}
    child_environment = os.environ.copy()
    child_environment["PELCB_PROJECT_CONFIG"] = str(PROJECT_CONFIG_PATH)
    child_environment["LCB_PROJECT_CONFIG"] = str(PROJECT_CONFIG_PATH)
    popen_kwargs["env"] = child_environment
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    with tempfile.TemporaryFile(mode="w+b") as out_f, tempfile.TemporaryFile(mode="w+b") as err_f:
        proc = subprocess.Popen(command, stdout=out_f, stderr=err_f, **popen_kwargs)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(proc.pid)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().decode("utf-8", "replace")
        stderr = err_f.read().decode("utf-8", "replace")
    returncode = proc.returncode if proc.returncode is not None else -1
    if timed_out:
        stderr = (stderr + f"\n[_run_subprocess] TIMEOUT after {timeout}s; process tree terminated").strip()
    return {
        "returncode": returncode,
        "stdout": stdout[-8000:],
        "stderr": stderr[-8000:],
        "ok": returncode == 0 and not timed_out,
    }


def submit_supervisor_case_impl(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    max_submit: int = 1,
) -> dict[str, Any]:
    request = locals().copy()
    rejected = _require_execution("submit_supervisor_case", request)
    if rejected:
        _ledger("submit_supervisor_case", request, rejected)
        return rejected
    if project_id != "o7clean":
        result = {"status": "rejected", "reason": "only o7clean submission is implemented in this sidecar"}
        _ledger("submit_supervisor_case", request, result)
        return result
    command = [
        sys.executable,
        str(MD_ANALYSIS_DIR / "o7_clean_supervisor_queue.py"),
        "--config",
        str(PROJECT_CONFIG_PATH),
        "--queue-dir",
        str(O7_QUEUE_DIR),
        "--source-csv",
        str(PROJECT_CONFIG.supervisor_output_dir / "o7_clean_engine_audit_20260522" / "source_dirs_combined_unique.csv"),
        "--cases",
        case_id,
        "--max-submit",
        str(max(1, int(max_submit))),
        "--execute",
    ]
    run_result = _run_subprocess(command, timeout=3600)
    result = {
        "status": "executed" if run_result["ok"] else "failed",
        "affected_path": str(O7_QUEUE_DIR),
        "command": command,
        "subprocess": run_result,
    }
    _ledger("submit_supervisor_case", request, result)
    return result


def run_postprocess_closure_impl(
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    auto_submit_postprocess: bool = False,
) -> dict[str, Any]:
    request = locals().copy()
    rejected = _require_execution("run_postprocess_closure", request)
    if rejected:
        _ledger("run_postprocess_closure", request, rejected)
        return rejected
    if project_id != "o7clean":
        result = {"status": "rejected", "reason": "only o7clean closure is implemented in this sidecar"}
        _ledger("run_postprocess_closure", request, result)
        return result
    command = [sys.executable, str(MD_ANALYSIS_DIR / "o7_clean_closure.py"),
               "--queue-dir", str(O7_QUEUE_DIR), "--output-dir", str(O7_CLOSURE_DIR)]
    if auto_submit_postprocess:
        command.append("--auto-submit-postprocess")
    run_result = _run_subprocess(command, timeout=3600)
    result = {
        "status": "executed" if run_result["ok"] else "failed",
        "affected_path": str(O7_CLOSURE_DIR),
        "command": command,
        "subprocess": run_result,
    }
    _ledger("run_postprocess_closure", request, result)
    return result


def submit_md_clean_engine_impl(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    max_submit: int = 1,
) -> dict[str, Any]:
    return submit_supervisor_case_impl(
        case_id=case_id,
        project_id=project_id,
        reason=reason,
        allowlist_tag=allowlist_tag,
        execute=execute,
        max_submit=max_submit,
    )


def run_postprocess_repair_impl(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    repair_kind: str = "msd_window",
    tag: str | None = None,
) -> dict[str, Any]:
    request = locals().copy()
    rejected = _require_execution("run_postprocess_repair", request)
    if rejected:
        _ledger("run_postprocess_repair", request, rejected)
        return rejected
    if project_id != "pelcb_postprocess_repair":
        result = {"status": "rejected", "reason": "only pelcb_postprocess_repair is implemented for this tool"}
        _ledger("run_postprocess_repair", request, result)
        return result
    if repair_kind not in {"msd_window", "packing_void"}:
        result = {"status": "rejected", "reason": "only msd_window and packing_void postprocess repair are implemented"}
        _ledger("run_postprocess_repair", request, result)
        return result
    if repair_kind == "msd_window":
        result = plan_msd_window_repair_impl(tag=tag, case_ids=[case_id])
    else:
        result = plan_packing_void_repair_impl(tag=tag, case_ids=[case_id])
    result["status"] = "executed" if result.get("status") == "planned" else result.get("status", "failed")
    _ledger("run_postprocess_repair", request, result)
    return result


def run_audit_subset_impl(
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    request = locals().copy()
    rejected = _require_execution("run_audit_subset", request)
    if rejected:
        _ledger("run_audit_subset", request, rejected)
        return rejected
    result = {
        "status": "rejected",
        "reason": "locked A3 audit execution is intentionally not exposed as a mutating MCP tool; use existing frozen ledgers or run the benchmark script manually after a new freeze",
        "case_ids": case_ids or [],
    }
    _ledger("run_audit_subset", request, result)
    return result


def list_rule_profile_source_taxonomy_impl() -> dict[str, Any]:
    return rule_profile_source_taxonomy()


def list_supported_rule_properties_impl() -> dict[str, Any]:
    return _json_safe(list_supported_rule_properties_core())


def list_online_rule_sources_impl() -> dict[str, Any]:
    """Return structured online/control rule-evidence sources and access boundaries."""

    providers = list_rule_evidence_providers_core()
    source_notes = {
        "radonpy_pi1070": {
            "display_name": "RadonPy PI1070",
            "access_policy": "live_structured_csv",
            "scraping_allowed": True,
            "boundary": "Read the public structured RadonPy CSV; do not infer values from prose.",
            "claim_support": "1070 amorphous polymers and multiple MD-calculated physical properties.",
        },
        "htpmd_public_index": {
            "display_name": "HTPMD public index",
            "access_policy": "live_structured_csv",
            "scraping_allowed": True,
            "boundary": "Read the public structured HTPMD CSV for polymer-electrolyte transport properties.",
            "claim_support": "High-throughput polymer electrolyte trajectories and automated post-processing properties.",
        },
        "polyinfo_export": {
            "display_name": "PoLyInfo export/cache",
            "access_policy": "official_export_or_local_cache_only",
            "scraping_allowed": False,
            "boundary": "Use only official exports, authorized APIs, or user-provided local cache; no web scraping.",
            "claim_support": "PoLyInfo covers about 100 property classes and literature-derived polymer data, but prohibits mass mechanical downloading.",
        },
        "registered_structured_seed_provider": {
            "display_name": "Registered structured seed/control evidence",
            "access_policy": "local_structured_seed_or_control",
            "scraping_allowed": False,
            "boundary": "Use only versioned local seed/control records, such as same-input clean controls.",
            "claim_support": "Supports deterministic rule-profile smoke tests and PE-LCB control evidence.",
        },
    }
    online_sources: list[dict[str, Any]] = []
    for provider_id, meta in sorted(providers.items()):
        notes = source_notes.get(provider_id, {})
        online_sources.append(
            {
                "provider_id": provider_id,
                "display_name": notes.get("display_name", provider_id),
                "mode": meta.get("mode"),
                "access_policy": notes.get("access_policy", meta.get("mode")),
                "scraping_allowed": bool(notes.get("scraping_allowed", False)),
                "supported_properties": list(meta.get("supported_properties", [])),
                "boundary": notes.get("boundary", "Provider must emit structured reference records only."),
                "claim_support": notes.get("claim_support", ""),
            }
        )
    return {
        "status": "ok",
        "schema_version": "online_rule_sources.v1",
        "online_rule_sources": online_sources,
        "freeze_boundary": [
            "Provider plugins emit structured reference records only.",
            "Deterministic proposal normalizes units, checks compatibility, and detects conflicts.",
            "Machine-freeze activation requires guarded MCP execution.",
        ],
        "runtime_boundary": [
            "Runtime rule gates use locally activated profiles only.",
            "No live web lookup is used during PASS/WARN/FAIL evaluation.",
            "LLM summaries and ordinary web pages cannot directly activate or alter rule thresholds.",
        ],
    }


def _profile_build_artifact_dir(context: dict[str, Any], property_name: str, profile: dict[str, Any] | None) -> Path:
    profile_id = (profile or {}).get("profile_id")
    if not profile_id:
        profile_id = "_".join(
            [
                _slug_text(context.get("polymer_family")),
                _slug_text(context.get("force_field")),
                _slug_text(context.get("engine_scope") or "any"),
                _slug_text(property_name),
                _hash_payload(context)[:10],
            ]
        )
    return RULE_PROFILE_DIR / "online_builds" / str(profile_id)


def _write_rule_build_artifacts(
    build_dir: Path,
    *,
    discovery: dict[str, Any],
    profile: dict[str, Any] | None,
) -> dict[str, str]:
    build_dir.mkdir(parents=True, exist_ok=True)
    discovery_path = build_dir / "discovery_result.json"
    reference_record_pack = build_dir / "reference_record_pack.jsonl"
    candidate_path = build_dir / "candidate_rule_profile.json"
    _write_json(discovery_path, discovery)
    reference_record_pack.write_text("", encoding="utf-8")
    reference_records = discovery.get("reference_records", discovery.get("source_evidence", [])) or []
    for source in reference_records:
        _write_jsonl(reference_record_pack, {"reference_record": source})
    if not reference_records:
        similarity = discovery.get("similarity_transfer") or {}
        for candidate in similarity.get("candidates", []) or []:
            if candidate.get("status") == "similarity_transfer_candidate":
                _write_jsonl(reference_record_pack, {"similarity_transfer_reference": candidate})
    if profile is not None:
        _write_json(candidate_path, profile)
    return {
        "discovery_result_json": str(discovery_path),
        "reference_record_pack_jsonl": str(reference_record_pack),
        # Backward-compatible alias for historical MCP clients and ledgers.
        "rule_evidence_pack_jsonl": str(reference_record_pack),
        "candidate_rule_profile_json": str(candidate_path) if profile is not None else "",
    }


def _profile_mean_evidence(profile: dict[str, Any]) -> dict[str, Any]:
    manifest = profile.get("source_manifest") or {}
    prop = str(profile.get("property") or "")
    mean_value = manifest.get("mean_normalized_value")
    evidence = {
        "value": mean_value,
        "units": profile.get("units"),
        "evidence_source": "candidate_rule_profile_source_mean",
    }
    if prop == "density":
        evidence["rho_final_gcc_si_o"] = mean_value
    else:
        evidence[prop] = mean_value
    return {
        key: value for key, value in evidence.items() if key
    }


def _missing_profile_rule_result(context: dict[str, Any], property_name: str, reason: str) -> dict[str, Any]:
    policy = rule_profile_automation_policy(property_name)
    return {
        "status": "ok",
        "rule_verdict": "WARN",
        "audit_required": True,
        "warn_source": rule_profile_missing_warn_source(property_name),
        "profile_id": None,
        "property": policy["property"],
        "polymer_type": str(context.get("polymer_type") or context.get("polymer_family") or ""),
        "condition": {
            "temperature_K": context.get("temperature_K"),
            "pressure_bar": context.get("pressure_bar"),
            "state": context.get("state"),
        },
        "force_field": str(context.get("force_field") or ""),
        "engine": str(context.get("engine") or context.get("engine_scope") or ""),
        "missing_reason": "no_frozen_profile_match",
        "automation_level": policy["automation_level"],
        "calibration_route": "auto_rule_profile_calibration"
        if policy["profile_freezable"]
        else "quality_or_audit_only",
        "fallback_status": policy["fallback_status"],
        "reason": reason,
    }


def discover_rule_evidence_impl(
    context_json: str | dict[str, Any],
    property_name: str,
    use_live: bool = True,
    include_seed: bool = True,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    high_similarity: float = DEFAULT_HIGH_SIMILARITY,
) -> dict[str, Any]:
    try:
        return _json_safe(
            discover_rule_evidence_core(
                context_json,
                property_name,
                use_live=use_live,
                include_seed=include_seed,
                min_similarity=min_similarity,
                high_similarity=high_similarity,
            )
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def propose_rule_profile_from_sources_impl(
    context_json: str | dict[str, Any],
    property_name: str,
    source_evidence_json: str | list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        profile = propose_rule_profile_from_sources_core(
            context_json,
            property_name,
            source_evidence_json,
            output_dir=RULE_PROFILE_DIR,
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    return {"status": "ok", "profile": _json_safe(profile)}


def activate_rule_profile_impl(
    profile_json: str | dict[str, Any],
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    request = locals().copy()
    rejected = _require_execution("activate_rule_profile", request)
    if rejected:
        _ledger("activate_rule_profile", request, rejected)
        return rejected
    try:
        result = activate_rule_profile_core(
            profile_json,
            output_dir=RULE_PROFILE_DIR,
            actor="codex",
            reason=reason,
        )
    except Exception as exc:
        result = {"status": "error", "reason": str(exc)}
    _ledger("activate_rule_profile", request, result)
    return _json_safe(result)


def build_rule_profile_from_online_evidence_impl(
    context_json: str | dict[str, Any],
    property_name: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    use_live: bool = True,
    include_seed: bool = True,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    high_similarity: float = DEFAULT_HIGH_SIMILARITY,
) -> dict[str, Any]:
    """Compare reference records, propose, optionally activate, and dry-check a rule profile."""

    request = {
        "context_json": context_json,
        "property_name": property_name,
        "project_id": project_id,
        "reason": reason,
        "allowlist_tag": allowlist_tag,
        "execute": execute,
        "use_live": use_live,
        "include_seed": include_seed,
        "min_similarity": min_similarity,
        "high_similarity": high_similarity,
    }
    try:
        context = dict(_parse_jsonish(context_json))
        discovery = discover_rule_evidence_core(
            context_json,
            property_name,
            use_live=use_live,
            include_seed=include_seed,
            min_similarity=min_similarity,
            high_similarity=high_similarity,
        )
        source_evidence = discovery.get("reference_records", discovery.get("source_evidence", [])) or []
        similarity_transfer = discovery.get("similarity_transfer") or {}
        has_similarity_candidate = (
            similarity_transfer.get("status") == "similarity_transfer_candidate"
            and bool(
                similarity_transfer.get(
                    "machine_activation_eligible",
                    similarity_transfer.get("machine_freeze_eligible"),
                )
            )
        )
        if not source_evidence and not has_similarity_candidate:
            status = "profile_unfreezable"
            if discovery.get("freeze_readiness") == "blocked_unsupported_property":
                status = "unsupported_property"
            result = {
                "status": status,
                "activation_outcome": "profile_not_activatable",
                "reason": "No compatible independent reference records were found; profile threshold guessing is disabled.",
                "discovery": discovery,
                "profile": None,
                "activation": {"status": "not_executed"},
                "dry_gate_check": None,
                "rule_result": _missing_profile_rule_result(
                    context,
                    property_name,
                    "No active rule profile can be built from the available compatible reference records.",
                ),
                "artifacts": {},
            }
            _ledger("build_rule_profile_from_online_evidence", request, result)
            return _json_safe(result)

        if source_evidence:
            profile = propose_rule_profile_from_sources_core(
                context_json,
                property_name,
                source_evidence,
                output_dir=RULE_PROFILE_DIR,
            )
        else:
            # Similarity transfer enters only after direct profile evidence is absent;
            # the registry still applies the ordinary freeze/conflict/confidence gates.
            profile = propose_similarity_transfer_profile_core(
                context_json,
                property_name,
                similarity_transfer,
                output_dir=RULE_PROFILE_DIR,
            )
        if not execute:
            return _json_safe(
                {
                    "status": "dry_run",
                    "activation_outcome": "proposal_ready_for_review",
                    "reason": "Profile proposal completed; execute=true is required to write artifacts and activate.",
                    "discovery": discovery,
                    "profile": profile,
                    "activation": {"status": "not_executed"},
                    "dry_gate_check": None,
                    "artifacts": {},
                }
            )

        rejected = _require_execution("build_rule_profile_from_online_evidence", request)
        if rejected:
            _ledger("build_rule_profile_from_online_evidence", request, rejected)
            return rejected

        artifacts = _write_rule_build_artifacts(
            _profile_build_artifact_dir(context, property_name, profile),
            discovery=discovery,
            profile=profile,
        )
        if not bool(profile.get("profile_activatable", profile.get("activation_status") == "eligible_for_machine_freeze")):
            blocked_status = str(profile.get("activation_status") or "")
            status = "profile_conflict" if blocked_status == "blocked_conflict" else "profile_unfreezable"
            result = {
                "status": status,
                "activation_outcome": "profile_not_activatable",
                "reason": f"profile is not eligible for activation: {profile.get('activation_readiness', profile.get('activation_status'))}",
                "discovery": discovery,
                "profile": profile,
                "activation": {"status": "not_executed"},
                "dry_gate_check": None,
                "rule_result": _missing_profile_rule_result(
                    context,
                    profile.get("property", property_name),
                    "Candidate profile was blocked before activation.",
                ),
                "artifacts": artifacts,
            }
            _ledger("build_rule_profile_from_online_evidence", request, result)
            return _json_safe(result)

        activation = activate_rule_profile_core(
            profile,
            output_dir=RULE_PROFILE_DIR,
            actor="codex",
            reason=reason,
        )
        dry_gate_check = None
        if activation.get("status") == "ok":
            dry_gate_check = apply_rule_gate_core(
                _profile_mean_evidence(profile),
                context=context_json,
                property_name=property_name,
                output_dir=RULE_PROFILE_DIR,
            )
        result = {
            "status": "profile_frozen" if activation.get("status") == "ok" else "error",
            "activation_outcome": "profile_activated" if activation.get("status") == "ok" else "activation_error",
            "discovery": discovery,
            "profile": profile,
            "activation": activation,
            "dry_gate_check": dry_gate_check,
            "artifacts": artifacts,
        }
        _ledger("build_rule_profile_from_online_evidence", request, result)
        return _json_safe(result)
    except Exception as exc:
        result = {"status": "error", "reason": str(exc)}
        _ledger("build_rule_profile_from_online_evidence", request, result)
        return result


def get_rule_profile_impl(context_json: str | dict[str, Any], property_name: str) -> dict[str, Any]:
    try:
        return _json_safe(get_rule_profile_core(context_json, property_name, output_dir=RULE_PROFILE_DIR))
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def apply_rule_gate_impl(
    context_json: str | dict[str, Any],
    property_name: str,
    evidence_json: str | dict[str, Any],
) -> dict[str, Any]:
    try:
        result = apply_rule_gate_core(
            evidence_json,
            context=context_json,
            property_name=property_name,
            output_dir=RULE_PROFILE_DIR,
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    return {"status": "ok", "rule_result": _json_safe(result)}


def apply_polymer_qc_rules_impl(evidence_json: str | dict[str, Any]) -> dict[str, Any]:
    try:
        result = apply_polymer_qc_rules_core(evidence_json)
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    return {"status": "ok", "qc_result": _json_safe(result)}


def get_zotero_rag_corpus_summary_impl() -> dict[str, Any]:
    """Return the frozen local Zotero-RAG corpus summary."""
    try:
        return _json_safe(get_zotero_rag_corpus_summary_core(output_dir=ZOTERO_RAG_DIR))
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def retrieve_design_prior_evidence_impl(
    context_json: str | dict[str, Any],
    query: str | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Retrieve Zotero-RAG chunks for a polymer-design prior query."""
    try:
        return _json_safe(
            retrieve_design_prior_evidence_core(
                context_json,
                query=query,
                output_dir=ZOTERO_RAG_DIR,
                top_k=top_k,
            )
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def get_pelcb100_design_literature_evidence_impl() -> dict[str, Any]:
    """Return the exact Zotero evidence used to define PE-LCB100 design axes."""

    context = {
        "polymer_family": "PE",
        "architecture": "pe_lcb",
        "target": "PE-LCB100 branched-polyethylene molecular design",
        "design_variables": ["branch_points", "branch_carbons", "branch_position_pattern"],
        "required_dois": list(DEFAULT_PELCB_DESIGN_DOIS),
        "design_value_provenance": {
            "literature_role": "defines the branch topology, length/count and placement/distribution axes",
            "project_role": (
                "defines the discrete bp/bc/pd grid under the approximately C100 carbon budget "
                "and registered builder constraints"
            ),
        },
    }
    query = "branched polyethylene branch number branch length branch distribution star topology"
    retrieval = retrieve_design_prior_evidence_impl(context, query=query, top_k=12)
    if retrieval.get("status") != "ok":
        return {"status": retrieval.get("status", "review_required"), "context": context, "retrieval": retrieval}
    profile = propose_design_prior_profile_from_rag_impl(context, retrieval.get("records", []))
    return {
        "status": "ok" if profile.get("profile", {}).get("activation_status") == "eligible_for_machine_freeze" else "review_required",
        "context": context,
        "retrieval": retrieval,
        "profile": profile.get("profile", {}),
    }


def propose_design_prior_profile_from_rag_impl(
    context_json: str | dict[str, Any],
    retrieval_records_json: str | list[dict[str, Any]],
) -> dict[str, Any]:
    """Propose a design-prior profile from retrieved Zotero-RAG evidence."""
    try:
        return _json_safe(
            propose_design_prior_profile_from_rag_core(
                context_json,
                retrieval_records_json,
                output_dir=ZOTERO_RAG_DIR,
            )
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def get_design_prior_profile_impl(profile_id: str) -> dict[str, Any]:
    """Return a proposed or activated Zotero-RAG design-prior profile."""
    try:
        return _json_safe(get_design_prior_profile_core(profile_id, output_dir=ZOTERO_RAG_DIR))
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def activate_design_prior_profile_impl(
    profile_json: str | dict[str, Any],
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    request = locals().copy()
    rejected = _require_execution("activate_design_prior_profile", request)
    if rejected:
        _ledger("activate_design_prior_profile", request, rejected)
        return rejected
    try:
        result = activate_design_prior_profile_core(
            profile_json,
            output_dir=ZOTERO_RAG_DIR,
            actor="codex",
            reason=reason,
        )
    except Exception as exc:
        result = {"status": "error", "reason": str(exc)}
    _ledger("activate_design_prior_profile", request, result)
    return _json_safe(result)


mcp = FastMCP(
    "Polymer MD guarded automation server",
    instructions=(
        "Guarded MCP interface for polymer molecular-dynamics workflows. "
        "Polymer-general preparation and result checks use registered capability "
        "profiles; PE-LCB100 endpoints remain as validated case-study tools. "
        "Before proposing a PE-LCB100 molecular-design set, call "
        "get_pelcb100_design_literature_evidence; if any required source is missing, "
        "route the design task to REVIEW instead of claiming literature grounding. "
        "Read-only and dry-run tools are safe by default. Mutating tools require "
        "execute=true plus project_id/reason/allowlist_tag and write a ledger."
    ),
)


@mcp.tool()
def get_case_summary(case_id: str) -> dict[str, Any]:
    """Return joined features, targets, and quality mask for one PE-LCB case."""
    return get_case_summary_impl(case_id)


@mcp.tool()
def get_release_manifest() -> dict[str, Any]:
    """Return the ML release manifest plus key reliability-map counts."""
    return get_release_manifest_impl()


@mcp.tool()
def get_design_origin(case_id: str) -> dict[str, Any]:
    """Return read-only design-origin evidence for one PE-LCB100 case."""
    return get_design_origin_impl(case_id)


@mcp.tool()
def get_design_origin_summary() -> dict[str, Any]:
    """Return read-only PE-LCB100 design-origin evidence summary."""
    return get_design_origin_summary_impl()


@mcp.tool()
def get_a3_metrics() -> dict[str, Any]:
    """Return compact locked A3 benchmark metrics."""
    return get_a3_metrics_impl()


@mcp.tool()
def get_audit_record(case_id: str, decision_type: str | None = None, mode: str | None = None) -> dict[str, Any]:
    """Return frozen A3 audit records for a case."""
    return get_audit_record_impl(case_id, decision_type, mode)


@mcp.tool()
def get_registered_repair_modules() -> dict[str, Any]:
    """Return deterministic EMC/MD/postprocessing/review module contracts."""
    return get_registered_repair_modules_impl()


@mcp.tool()
def list_warn_diagnosis_taxonomy() -> dict[str, Any]:
    """Return registered WARN sources, diagnosis classes, and repair tool routes."""
    return list_warn_diagnosis_taxonomy_impl()


@mcp.tool()
def list_repair_tool_affordances() -> dict[str, Any]:
    """Return bounded repair tools, allowed parameters, and guard policy."""
    return list_repair_tool_affordances_impl()


@mcp.tool()
def validate_repair_tool_call(proposal_json: str, context_json: str | None = None) -> dict[str, Any]:
    """Dry-run guard validation for an LLM-proposed repair tool call."""
    return validate_repair_tool_call_impl(proposal_json, context_json)


@mcp.tool()
def list_diagnosis_candidates(diagnosis_jsonl: str | None = None) -> dict[str, Any]:
    """Return candidate open WARN causes, optionally rebuilt from a diagnosis JSONL ledger."""
    return list_diagnosis_candidates_impl(diagnosis_jsonl)


@mcp.tool()
def propose_diagnosis_profile_from_candidates(candidate_json: str, candidate_id: str | None = None) -> dict[str, Any]:
    """Propose a frozen diagnosis profile from an eligible open-cause candidate."""
    return propose_diagnosis_profile_from_candidates_impl(candidate_json, candidate_id)


@mcp.tool()
def activate_diagnosis_profile(
    profile_json: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    """Guarded activation of a versioned open-diagnosis profile."""
    return activate_diagnosis_profile_impl(profile_json, project_id, reason, allowlist_tag, execute)


@mcp.tool()
def get_audit_diagnosis(case_id: str) -> dict[str, Any]:
    """Return rule-triggered audit diagnosis and recommended module re-entry for one case."""
    return get_audit_diagnosis_impl(case_id)


@mcp.tool()
def plan_repair_route(case_id: str, planning_proposal_json: str = "") -> dict[str, Any]:
    """Let the planning model select one guarded route allowed by the audit diagnosis."""
    return plan_repair_route_impl(case_id, planning_proposal_json)


@mcp.tool()
def run_warn_audit_diagnosis(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    force_refresh: bool = False,
    use_mock_clients: bool = False,
) -> dict[str, Any]:
    """Run guarded tri-model WARN diagnosis plus bounded repair/modification planning for one case."""
    return run_warn_audit_diagnosis_impl(
        case_id=case_id,
        project_id=project_id,
        reason=reason,
        allowlist_tag=allowlist_tag,
        execute=execute,
        force_refresh=force_refresh,
        use_mock_clients=use_mock_clients,
    )


@mcp.tool()
def run_warn_audit_batch(
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    max_cases: int = 0,
    case_ids: list[str] | None = None,
    use_mock_clients: bool = False,
) -> dict[str, Any]:
    """Run guarded tri-model WARN diagnosis over selected/all WARN cases and write a sidecar ledger."""
    return run_warn_audit_batch_impl(
        project_id=project_id,
        reason=reason,
        allowlist_tag=allowlist_tag,
        execute=execute,
        max_cases=max_cases,
        case_ids=case_ids,
        use_mock_clients=use_mock_clients,
    )


@mcp.tool()
def get_compute_status() -> dict[str, Any]:
    """Return local supervisor/runtime status snapshots without probing remotes."""
    return get_compute_status_impl()


@mcp.tool()
def list_rule_profile_source_taxonomy() -> dict[str, Any]:
    """Return trusted source classes for automatic rule-profile calibration."""
    return list_rule_profile_source_taxonomy_impl()


@mcp.tool()
def list_supported_rule_properties() -> dict[str, Any]:
    """Return profile-freezable properties and generic QC-only rule sources."""
    return list_supported_rule_properties_impl()


@mcp.tool()
def list_online_rule_sources() -> dict[str, Any]:
    """Return trusted online/structured sources for automatic rule-profile construction."""
    return list_online_rule_sources_impl()


@mcp.tool()
def discover_rule_evidence(
    context_json: str | dict[str, Any],
    property_name: str,
    use_live: bool = True,
    include_seed: bool = True,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    high_similarity: float = DEFAULT_HIGH_SIMILARITY,
) -> dict[str, Any]:
    """Discover structured public/control evidence for a rule-profile proposal without activating it."""
    return discover_rule_evidence_impl(
        context_json,
        property_name,
        use_live=use_live,
        include_seed=include_seed,
        min_similarity=min_similarity,
        high_similarity=high_similarity,
    )


@mcp.tool()
def build_rule_profile_from_online_evidence(
    context_json: str | dict[str, Any],
    property_name: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    use_live: bool = True,
    include_seed: bool = True,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    high_similarity: float = DEFAULT_HIGH_SIMILARITY,
) -> dict[str, Any]:
    """Build and optionally activate a rule profile from compatible structured reference records."""
    return build_rule_profile_from_online_evidence_impl(
        context_json,
        property_name,
        project_id=project_id,
        reason=reason,
        allowlist_tag=allowlist_tag,
        execute=execute,
        use_live=use_live,
        include_seed=include_seed,
        min_similarity=min_similarity,
        high_similarity=high_similarity,
    )


@mcp.tool()
def propose_rule_profile_from_sources(
    context_json: str | dict[str, Any],
    property_name: str,
    source_evidence_json: str | list[dict[str, Any]],
) -> dict[str, Any]:
    """Propose a candidate frozen rule profile from normalized public/control evidence."""
    return propose_rule_profile_from_sources_impl(context_json, property_name, source_evidence_json)


@mcp.tool()
def activate_rule_profile(
    profile_json: str | dict[str, Any],
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    """Activate an eligible machine-frozen rule profile through the guarded ledger."""
    return activate_rule_profile_impl(profile_json, project_id, reason, allowlist_tag, execute)


@mcp.tool()
def get_rule_profile(context_json: str | dict[str, Any], property_name: str) -> dict[str, Any]:
    """Return the active rule profile matching polymer family/property/scope."""
    return get_rule_profile_impl(context_json, property_name)


@mcp.tool()
def apply_rule_gate(
    context_json: str | dict[str, Any],
    property_name: str,
    evidence_json: str | dict[str, Any],
) -> dict[str, Any]:
    """Apply the active frozen rule profile to evidence and return PASS/WARN/FAIL."""
    return apply_rule_gate_impl(context_json, property_name, evidence_json)


@mcp.tool()
def apply_polymer_qc_rules(evidence_json: str | dict[str, Any]) -> dict[str, Any]:
    """Apply polymer-general QC rule sources for MSD/Rg/shape/RDF/contact/Fs(q,t)/runtime/provenance."""
    return apply_polymer_qc_rules_impl(evidence_json)


@mcp.tool()
def get_zotero_rag_corpus_summary() -> dict[str, Any]:
    """Return the frozen local Zotero-RAG corpus summary."""
    return get_zotero_rag_corpus_summary_impl()


@mcp.tool()
def get_pelcb100_design_literature_evidence() -> dict[str, Any]:
    """Read the three DOI-locked Zotero sources that define PE-LCB100 design axes."""
    return get_pelcb100_design_literature_evidence_impl()


@mcp.tool()
def retrieve_design_prior_evidence(
    context_json: str | dict[str, Any],
    query: str | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Retrieve frozen Zotero chunks supporting polymer design-prior extraction."""
    return retrieve_design_prior_evidence_impl(context_json, query=query, top_k=top_k)


@mcp.tool()
def propose_design_prior_profile_from_rag(
    context_json: str | dict[str, Any],
    retrieval_records_json: str | list[dict[str, Any]],
) -> dict[str, Any]:
    """Propose a guarded design-prior profile from retrieved Zotero-RAG evidence."""
    return propose_design_prior_profile_from_rag_impl(context_json, retrieval_records_json)


@mcp.tool()
def get_design_prior_profile(profile_id: str) -> dict[str, Any]:
    """Return a proposed or activated Zotero-RAG design-prior profile."""
    return get_design_prior_profile_impl(profile_id)


@mcp.tool()
def activate_design_prior_profile(
    profile_json: str | dict[str, Any],
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    """Activate an eligible Zotero-RAG design-prior profile through guarded ledger."""
    return activate_design_prior_profile_impl(profile_json, project_id, reason, allowlist_tag, execute)


@mcp.tool()
def validate_design_candidate(
    branch_points: int,
    branch_carbons: int,
    branch_position_pattern: str,
    carbon_budget: int = 100,
) -> dict[str, Any]:
    """Validate a grammar-constrained PE-LCB design candidate."""
    return validate_design_candidate_impl(branch_points, branch_carbons, branch_position_pattern, carbon_budget)


@mcp.tool()
def validate_polymer_design_candidate(candidate_json: str | dict[str, Any]) -> dict[str, Any]:
    """Automatically select registered construction and force-field options, then validate a polymer candidate."""
    return validate_polymer_design_candidate_impl(candidate_json)


@mcp.tool()
def list_polymer_preparation_capabilities() -> dict[str, Any]:
    """List registered polymer adapters, builders, force fields, and validated scopes."""
    return list_polymer_preparation_capabilities_impl()


@mcp.tool()
def validate_or_plan_polymer_input(candidate_json: str | dict[str, Any]) -> dict[str, Any]:
    """Validate a polymer candidate and plan its builder and topology checks."""
    return validate_or_plan_polymer_input_impl(candidate_json)


@mcp.tool()
def materialize_polymer_input(
    candidate_json: str | dict[str, Any],
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    """Materialize inputs with a registered builder after guarded validation."""
    return materialize_polymer_input_impl(
        candidate_json,
        project_id,
        reason,
        allowlist_tag,
        execute,
    )


@mcp.tool()
def score_candidate_against_reliability_map(
    branch_points: int,
    branch_carbons: int,
    branch_position_pattern: str,
    target_contact: float | None = None,
) -> dict[str, Any]:
    """Score a candidate against current coverage and release quality patterns."""
    return score_candidate_against_reliability_map_impl(branch_points, branch_carbons, branch_position_pattern, target_contact)


@mcp.tool()
def plan_md_package(
    candidate_id: str | None = None,
    branch_points: int | None = None,
    branch_carbons: int | None = None,
    branch_position_pattern: str | None = None,
    candidate_json: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dry-run an MD package plan for a validated legacy PE-LCB or generic polymer candidate."""
    return plan_md_package_impl(candidate_id, branch_points, branch_carbons, branch_position_pattern, candidate_json)


@mcp.tool()
def validate_or_build_emc_input(candidate_json: str | dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible EMC planning alias for registered polymer candidates."""
    return validate_or_build_emc_input_impl(candidate_json)


@mcp.tool()
def plan_supervisor_slots(refresh: bool = False, max_slots: int = 12) -> dict[str, Any]:
    """Dry-run supervisor slot planning. Optionally refreshes runtime dashboard."""
    return plan_supervisor_slots_impl(refresh=refresh, max_slots=max_slots)


@mcp.tool()
def plan_postprocess_closure(project_id: str = "o7clean") -> dict[str, Any]:
    """Dry-run clean sidecar postprocess closure planning."""
    return plan_postprocess_closure_impl(project_id)


@mcp.tool()
def get_repair_loop_latest() -> dict[str, Any]:
    """Return the most recent PE-LCB repair-loop plan summary."""
    return get_repair_loop_latest_impl()


@mcp.tool()
def plan_repair_loop(include_optional_validation: bool = False, tag: str | None = None) -> dict[str, Any]:
    """Dry-run repair routing for missing, warning, and audit-disagreement cases."""
    return plan_repair_loop_impl(include_optional_validation=include_optional_validation, tag=tag)


@mcp.tool()
def get_msd_window_repair_latest() -> dict[str, Any]:
    """Return the latest MSD fit-window repair dry-run summary."""
    return get_msd_window_repair_latest_impl()


@mcp.tool()
def plan_msd_window_repair(tag: str | None = None, case_ids: list[str] | None = None) -> dict[str, Any]:
    """Dry-run C100-aware MSD fit-window repair diagnostics without editing release tables."""
    return plan_msd_window_repair_impl(tag=tag, case_ids=case_ids)


@mcp.tool()
def get_msd_window_overlay_latest() -> dict[str, Any]:
    """Return the latest sidecar MSD-window ML overlay summary."""
    return get_msd_window_overlay_latest_impl()


@mcp.tool()
def plan_msd_window_overlay(tag: str | None = None) -> dict[str, Any]:
    """Regenerate the sidecar v3 MSD-window overlay without modifying v1 tables."""
    return plan_msd_window_overlay_impl(tag=tag)


@mcp.tool()
def get_packing_void_latest() -> dict[str, Any]:
    """Return the latest packing/void sidecar overlay summary."""
    return get_packing_void_latest_impl()


@mcp.tool()
def plan_packing_void_repair(tag: str | None = None, case_ids: list[str] | None = None) -> dict[str, Any]:
    """Run sidecar packing/void postprocessing and overlay generation without modifying v1 tables."""
    return plan_packing_void_repair_impl(tag=tag, case_ids=case_ids)


@mcp.tool()
def submit_supervisor_case(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    max_submit: int = 1,
) -> dict[str, Any]:
    """Submit one sidecar case through the guarded supervisor path."""
    return submit_supervisor_case_impl(case_id, project_id, reason, allowlist_tag, execute, max_submit)


@mcp.tool()
def submit_md_clean_engine(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    max_submit: int = 1,
) -> dict[str, Any]:
    """Submit a case to the guarded clean-engine MD module."""
    return submit_md_clean_engine_impl(case_id, project_id, reason, allowlist_tag, execute, max_submit)


@mcp.tool()
def run_postprocess_closure(
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    auto_submit_postprocess: bool = False,
) -> dict[str, Any]:
    """Run guarded clean sidecar postprocess closure."""
    return run_postprocess_closure_impl(project_id, reason, allowlist_tag, execute, auto_submit_postprocess)


@mcp.tool()
def run_postprocess_repair(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    repair_kind: str = "msd_window",
    tag: str | None = None,
) -> dict[str, Any]:
    """Run a guarded postprocessing repair module for one sidecar case."""
    return run_postprocess_repair_impl(case_id, project_id, reason, allowlist_tag, execute, repair_kind, tag)


@mcp.tool()
def run_audit_subset(
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Guarded placeholder for audit execution; frozen A3 reruns stay manual."""
    return run_audit_subset_impl(project_id, reason, allowlist_tag, execute, case_ids)


@mcp.tool()
def extend_clean_nvt_prod(
    case_id: str,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    """Submit one guarded checkpoint-aware clean NVT production extension."""
    return extend_clean_nvt_prod_impl(case_id, project_id, reason, allowlist_tag, execute)


@mcp.tool()
def run_repair_loop(
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
    max_execute_md: int = 0,
    max_execute_density: int = 0,
    run_closure: bool = False,
    auto_submit_postprocess: bool = False,
    refresh_publication_assets: bool = False,
    include_optional_validation: bool = False,
) -> dict[str, Any]:
    """Run the guarded repair-loop controller for sidecar MD/postprocess repair routes."""
    return run_repair_loop_impl(
        project_id=project_id,
        reason=reason,
        allowlist_tag=allowlist_tag,
        execute=execute,
        max_execute_md=max_execute_md,
        max_execute_density=max_execute_density,
        run_closure=run_closure,
        auto_submit_postprocess=auto_submit_postprocess,
        refresh_publication_assets=refresh_publication_assets,
        include_optional_validation=include_optional_validation,
    )


@mcp.resource("pelcb://release/ml_manifest")
def release_manifest_resource() -> str:
    return json.dumps(get_release_manifest_impl(), ensure_ascii=False, indent=2)


@mcp.resource("pelcb://case/{case_id}/summary")
def case_summary_resource(case_id: str) -> str:
    return json.dumps(get_case_summary_impl(case_id), ensure_ascii=False, indent=2)


@mcp.resource("pelcb://a3/metrics")
def a3_metrics_resource() -> str:
    return json.dumps(get_a3_metrics_impl(), ensure_ascii=False, indent=2)


@mcp.resource("pelcb://supervisor/status")
def supervisor_status_resource() -> str:
    return json.dumps(get_compute_status_impl(), ensure_ascii=False, indent=2)


@mcp.resource("pelcb://repair/latest")
def repair_latest_resource() -> str:
    return json.dumps(get_repair_loop_latest_impl(), ensure_ascii=False, indent=2)


@mcp.resource("pelcb://repair/msd_window_latest")
def msd_window_repair_latest_resource() -> str:
    return json.dumps(get_msd_window_repair_latest_impl(), ensure_ascii=False, indent=2)


@mcp.resource("pelcb://overlay/msd_window_latest")
def msd_window_overlay_latest_resource() -> str:
    return json.dumps(get_msd_window_overlay_latest_impl(), ensure_ascii=False, indent=2)


@mcp.resource("polymermd://preparation/capabilities")
def polymer_preparation_capabilities_resource() -> str:
    return json.dumps(
        list_polymer_preparation_capabilities_impl(),
        ensure_ascii=False,
        indent=2,
    )


@mcp.prompt()
def polymer_design_candidate_review(candidate_json: str) -> str:
    return (
        "Review a polymer candidate using only registered capabilities. "
        "Call list_polymer_preparation_capabilities, then "
        "validate_or_plan_polymer_input. Distinguish technical builder support, "
        "parameterization-probe status, and scientific qualification. "
        f"Candidate: {candidate_json}\n"
        "Do not infer support for unregistered chemistry or edit MD physics."
    )


@mcp.prompt()
def pelcb_design_candidate_review(objective: str, constraints: str) -> str:
    return (
        "Review a PE-LCB sidecar design candidate under the frozen grammar. "
        "Use validate_design_candidate and score_candidate_against_reliability_map first. "
        f"Objective: {objective}\nConstraints: {constraints}\n"
        "Do not suggest mdp-physics or rule-threshold edits."
    )


@mcp.prompt()
def pelcb_failure_route_review(case_id: str) -> str:
    return (
        f"Review failure routing for {case_id}. Use get_case_summary, get_audit_record, "
        "and get_compute_status. Recommend only deterministic repair routes: rerun/resume, "
        "postprocess closure, rule/evidence review, or automatic REVIEW assignment with an evidence packet."
    )


@mcp.prompt()
def pelcb_release_audit_review(case_id: str) -> str:
    return (
        f"Audit release evidence for {case_id}. Use frozen release tables and A3 records. "
        "Report descriptors, provenance, missingness, rule verdicts, audit disagreement, "
        "and whether the case is ML-usable. Do not invent missing evidence."
    )


@mcp.prompt()
def pelcb_repair_route_review(case_id: str) -> str:
    return (
        f"Review deterministic repair routing for {case_id}. Use get_case_summary, "
        "get_repair_loop_latest, get_msd_window_repair_latest, and any relevant A3 audit records. Distinguish incomplete MD "
        "repair, postprocessing-window repair, audit-disagreement review, and no-action cases. "
        "Do not recommend threshold edits or unbounded reruns."
    )


def self_test() -> dict[str, Any]:
    return {
        "case_summary": get_case_summary_impl("lcb_pe_004_bp01_bc016_pdmid_c100")["status"],
        "manifest": get_release_manifest_impl()["summary"],
        "design_origin": get_design_origin_summary_impl()["summary"],
        "design_origin_case": get_design_origin_impl("lcb_pe_004_bp01_bc016_pdmid_c100")["record"][
            "validator_status"
        ],
        "a3_n_groups": get_a3_metrics_impl().get("metrics", {}).get("n_groups"),
        "repair_modules": sorted(get_registered_repair_modules_impl()["modules"].keys()),
        "warn_taxonomy_sources": len(list_warn_diagnosis_taxonomy_impl()["warn_sources"]),
        "audit_diagnosis_density": get_audit_diagnosis_impl("lcb_pe_006_bp01_bc008_pdend_c100")["diagnosis"][
            "audit_diagnosis"
        ],
        "repair_route_msd": plan_repair_route_impl(
            "lcb_pe_002_bp01_bc008_pdmid_c100",
            json.dumps(
                {
                    "tool": "run_postprocess_repair",
                    "parameters": {
                        "repair_kind": "msd_window",
                        "window_strategy": "c100_aware",
                    },
                    "rationale": (
                        "The audit identifies a fit-window issue, so the registered "
                        "MSD window recomputation directly addresses the open warning."
                    ),
                }
            ),
        )["repair_route"],
        "warn_audit_dry_run": run_warn_audit_diagnosis_impl(
            case_id="lcb_pe_002_bp01_bc008_pdmid_c100",
            project_id="pelcb_repair_loop",
            reason="self-test dry-run tri-model warn diagnosis guard",
            allowlist_tag=WARN_AUDIT_ALLOWLIST_TAG,
            execute=False,
        )["diagnosis"]["diagnosis_source"],
        "warn_audit_execute_reject": run_warn_audit_diagnosis_impl(
            case_id="lcb_pe_002_bp01_bc008_pdmid_c100",
            project_id="pelcb_repair_loop",
            reason="self-test rejection path",
            allowlist_tag="wrong",
            execute=True,
            use_mock_clients=True,
        )["status"],
        "valid_candidate": validate_design_candidate_impl(4, 4, "comb-uniform-4")["validator_status"],
        "duplicate_candidate": validate_design_candidate_impl(1, 16, "mid")["validator_status"],
        "generic_linear_candidate": validate_polymer_design_candidate_impl(
            json.dumps(
                {
                    "candidate_id": "selftest_linear_alpha_octene",
                    "polymer_family": "poly_alpha_olefin",
                    "architecture": "linear_alpha_olefin",
                    "builder_backend": "emc_alpha_olefin_builder",
                    "force_field": "opls-aa",
                    "chain_count": 150,
                    "target_total_carbons_per_chain": 100,
                    "sidechain_carbons": 6,
                    "repeat_unit_smiles": "C=CCCCCC",
                }
            )
        )["validator_status"],
        "generic_homopolymer_candidate": validate_polymer_design_candidate_impl(
            json.dumps(
                {
                    "candidate_id": "selftest_linear_pe",
                    "polymer_family": "PE",
                    "architecture": "linear_homopolymer",
                    "builder_backend": "emc_alpha_olefin_builder",
                    "force_field": "opls-aa",
                    "chain_count": 150,
                    "target_total_carbons_per_chain": 100,
                    "repeat_unit_smiles": "*CC*",
                    "repeat_unit_total_carbons": 2,
                    "repeat_units_per_chain": 50,
                }
            )
        )["validator_status"],
        "repair_latest": get_repair_loop_latest_impl()["status"],
        "msd_window_repair_latest": get_msd_window_repair_latest_impl()["status"],
        "msd_window_overlay_latest": get_msd_window_overlay_latest_impl()["status"],
        "packing_void_latest": get_packing_void_latest_impl()["status"],
        "zotero_rag_corpus": get_zotero_rag_corpus_summary_impl()["status"],
        "rule_profile_source_taxonomy": len(list_rule_profile_source_taxonomy_impl()["high_trust_source_types"]),
        "supported_rule_properties": len(list_supported_rule_properties_impl()["public_data_profile_properties"]),
        "online_rule_sources": len(list_online_rule_sources_impl()["online_rule_sources"]),
        "rule_evidence_discovery_pe_density": discover_rule_evidence_impl(
            json.dumps(
                {
                    "polymer_family": "PE",
                    "repeat_unit_smiles": "*CC*",
                    "synonyms": ["polyethylene", "PE-LCB100"],
                    "force_field": "opls-aa",
                    "engine_scope": "any",
                    "temperature_K": 450.0,
                    "pressure_bar": 1.0,
                    "state": "melt",
                }
            ),
            "density",
            use_live=False,
            include_seed=True,
        )["source_count"],
        "online_rule_profile_build_preview": build_rule_profile_from_online_evidence_impl(
            json.dumps(
                {
                    "polymer_family": "PE",
                    "repeat_unit_smiles": "*CC*",
                    "synonyms": ["polyethylene", "PE-LCB100"],
                    "force_field": "opls-aa",
                    "engine_scope": "any",
                    "temperature_K": 450.0,
                    "pressure_bar": 1.0,
                    "state": "melt",
                }
            ),
            "density",
            project_id="pelcb_rule_profiles",
            reason="self-test online rule-profile build preview",
            allowlist_tag="PELCB_MCP_EXECUTE_RULE_PROFILE",
            execute=False,
            use_live=False,
            include_seed=True,
        )["status"],
        "polymer_qc_rules_pass": apply_polymer_qc_rules_impl(
            json.dumps(
                {
                    "case_id": "selftest_qc_pass",
                    "case_status": "completed_quality_passed",
                    "force_field": "opls-aa",
                    "force_field_qualified": True,
                    "builder_backend": "emc_alpha_olefin_builder",
                    "topology_format": "gromacs_atomistic",
                    "force_field_supported": True,
                    "builder_backend_supported": True,
                    "topology_supported": True,
                    "support_assessment_source": (
                        "registered_capability:polymer_md_runtime_capabilities.v1"
                    ),
                    "engine_version": "GROMACS 2024.4 clean nvt_prod",
                    "engine_qualified": True,
                    "status_Rg": "OK",
                    "rg_mean_nm": 1.5,
                    "rg2_mean_nm2": 2.25,
                    "status_shape": "OK",
                    "shape_rg2_carbon_mean_nm2": 2.5,
                    "shape_asphericity_mean_nm2": 1.0,
                    "shape_acylindricity_mean_nm2": 0.4,
                    "shape_relative_anisotropy_mean": 0.35,
                    "status_RDF": "OK",
                    "rdf_validation_status": "current",
                    "rdf_all_first_peak_g": 12.5,
                    "rdf_excl_first_peak_g": 1.4,
                    "rdf_inter_chain_cc_first_peak_g": 1.0,
                    "rdf_intra_chain_nonbonded_cc_first_peak_g": 12.0,
                    "status_contact": "OK",
                    "P_contact_rc0p8_s10": 0.10,
                    "P_contact_rc0p8_s20": 0.04,
                    "P_contact_rc0p8_s50": 0.01,
                    "status_MSD": "OK",
                    "fit_quality": "ok",
                    "D_mean_m2_s": 5e-11,
                    "D_median_m2_s": 4.9e-11,
                    "D_outlier_fraction": 0.0,
                    "loglog_slope": 1.0,
                    "diffusive_regime_ok": True,
                    "window_sensitive": False,
                    "window_D_cv_pct": 2.0,
                    "window_msd_over_rg2_min": 3.5,
                    "fit_begin_ps": 100000,
                    "fit_end_ps": 300000,
                    "time_first_ps": 0,
                    "time_last_ps": 400000,
                    "python_fit": {"R2": 0.99},
                    "status_fsqt_tau_alpha": "OK",
                    "tau_alpha_ps": 64.0,
                    "status_dynamics": "OK",
                }
            )
        )["qc_result"]["rule_verdict"],
        "execution_reject": submit_supervisor_case_impl(
            case_id="lcb_pe_004_bp01_bc016_pdmid_c100",
            project_id="o7clean",
            reason="self-test rejection path",
            allowlist_tag="PELCB_MCP_EXECUTE_O7CLEAN",
            execute=False,
        )["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run local function smoke checks and exit.")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_json_safe(self_test()), ensure_ascii=False, indent=2))
        return 0
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
