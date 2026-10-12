#!/usr/bin/env python3
"""Tri-model WARN diagnosis and repair-plan audit runner.

The deterministic rule layer only flags PASS/WARN/FAIL.  This module covers
the next agentic step for every WARN source: ask three audit models to return a
cause attribution plus a concrete modification plan, then normalize the result
back onto the registered deterministic repair taxonomy.

The LLM output is advisory.  It cannot introduce arbitrary tools, shell
commands, rule-threshold edits, or mdp physics changes; selected tool calls are
always constrained by ``pelcb_audit_diagnosis.DIAGNOSIS_TO_ROUTE``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

import pandas as pd

try:
    from agentic.tri_llm_judge import DeepSeekClient, GLMClient, QwenClient, StaticJSONClient
except ImportError:  # pragma: no cover - direct path fallback
    from generated_code.md_analysis.agentic.tri_llm_judge import (  # type: ignore
        DeepSeekClient,
        GLMClient,
        QwenClient,
        StaticJSONClient,
    )

from pelcb_audit_diagnosis import (
    DIAGNOSIS_TO_ROUTE,
    DETERMINISTIC_MODULES,
    WARN_SOURCES,
    build_warn_evidence_packets,
    collect_warn_sources,
    diagnose_case,
    normalize_tool_parameters,
    repair_tool_affordances,
    route_from_diagnosis,
    route_from_primary_diagnosis,
    validate_repair_tool_call,
    warn_diagnosis_taxonomy,
)
from diagnosis_candidate_registry import write_candidate_ledger
from lcb_project_config import load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent


def _project_config_path() -> str | Path:
    return (
        os.environ.get("PELCB_PROJECT_CONFIG")
        or os.environ.get("LCB_PROJECT_CONFIG")
        or SCRIPT_DIR / "configs" / "pelcb100.json"
    )


PROJECT_CONFIG = load_lcb_config(_project_config_path())
DEFAULT_MASK = Path(
    os.environ.get("PELCB_AUDIT_MASK")
    or PROJECT_CONFIG.integrated_output_dir / "ml_quality_mask.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PELCB_WARN_AUDIT_OUTPUT_DIR")
    or PROJECT_CONFIG.warn_audit_output_dir
)
WARN_AUDIT_SCHEMA_VERSION = "pelcb_warn_audit_tri_model.v2.3"
AUDIT_PROMPT_MODE = "independent_evidence_only"
ALLOWLIST_TAG = "PELCB_WARN_AUDIT_EXECUTE"
AGGREGATOR_ONLY_DIAGNOSES = {"cross_model_disagreement", "low_confidence_audit"}
AUDIT_MODEL_DISPLAY_NAMES = {
    "deepseek-v4-pro": "DeepSeek V4-Pro",
    "qwen3-max": "Qwen3.7-Max",
    "qwen3.7-max": "Qwen3.7-Max",
    "glm-4.6": "GLM 4.6",
}
CASE_EVIDENCE_ROUTE_HINT_KEYS = {
    "audit_diagnosis",
    "available_repair_tools",
    "candidate_diagnosis",
    "candidate_diagnoses",
    "candidate_route",
    "deterministic_module",
    "diagnosis_candidate",
    "diagnosis_candidates",
    "final_diagnosis",
    "final_repair_tool",
    "llm_tool_call_proposal",
    "modification_plan",
    "primary_audit_diagnosis",
    "recommended_action",
    "recommended_reentry_module",
    "repair_affordances",
    "repair_action",
    "repair_instruction",
    "repair_route",
    "repair_tool",
    "route",
    "route_plan",
    "tool_parameters",
    "tool_plan",
}
AUDIT_STRATEGIES = {
    "single-model",
    "tri-model-majority",
    "conditional-deliberation",
    "always-deliberation",
}
DELIBERATION_CONFIDENCE_THRESHOLD = 0.75


class AuditClient(Protocol):
    name: str

    def generate(self, prompt: str) -> str | dict[str, Any]:
        ...


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def audit_model_display_name(provider_name: str) -> str:
    """Return the manuscript-facing name while preserving provider aliases in ledgers."""

    name = str(provider_name or "").strip()
    return AUDIT_MODEL_DISPLAY_NAMES.get(name.lower(), name)


def classify_audit_disagreement_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Distinguish a current vote split from a disagreement retained from an earlier audit."""

    raw_sources = record.get("warn_sources") or record.get("warn_source") or []
    if isinstance(raw_sources, str):
        sources = {token.strip() for token in re.split(r"[,;|\s]+", raw_sources) if token.strip()}
    else:
        sources = {str(token).strip() for token in raw_sources if str(token).strip()}

    raw_votes = record.get("deliberation_votes") or record.get("model_votes") or []
    vote_labels = {
        str(vote.get("primary_audit_diagnosis") or vote.get("cause_diagnosis") or "").strip()
        for vote in raw_votes
        if isinstance(vote, Mapping)
        and str(vote.get("primary_audit_diagnosis") or vote.get("cause_diagnosis") or "").strip()
    }
    raw_current_flag = record.get("cross_model_disagreement")
    if isinstance(raw_current_flag, str):
        current_flag = raw_current_flag.strip().lower() in {"1", "true", "yes"}
    else:
        current_flag = bool(raw_current_flag)
    current_panel_disagreement = current_flag or len(vote_labels) > 1
    primary = str(
        record.get("primary_audit_diagnosis")
        or record.get("audit_diagnosis")
        or record.get("final_diagnosis")
        or ""
    ).strip()
    retained_prior_signal = (
        "audit_disagreement_warn" in sources
        and primary == "cross_model_disagreement"
        and not current_panel_disagreement
    )
    if current_panel_disagreement:
        origin = "current_panel_vote_split"
    elif retained_prior_signal:
        origin = "retained_prior_audit_signal"
    else:
        origin = "none"
    return {
        "current_panel_disagreement": current_panel_disagreement,
        "retained_prior_audit_disagreement": retained_prior_signal,
        "audit_disagreement_origin": origin,
        "current_panel_vote_labels": sorted(vote_labels),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def parse_json_payload(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if match:
            text = match.group(1)
    if not text.startswith("{"):
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            text = match.group(1)
    return json.loads(text)


def _as_list(value: Any) -> list[Any]:
    """Normalize nullable model array fields without discarding scalar evidence."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def default_clients() -> list[AuditClient]:
    timeout_s = float(os.environ.get("WARN_AUDIT_TIMEOUT_S", "120"))
    max_tokens = int(os.environ.get("WARN_AUDIT_MAX_TOKENS", "2000"))
    # The configured DeepSeek reasoning endpoint needs a larger completion
    # budget for the 20k-character evidence prompt; provider-specific knobs
    # avoid increasing the cost of Qwen and GLM calls.
    deepseek_timeout_s = float(os.environ.get("DEEPSEEK_WARN_AUDIT_TIMEOUT_S", "240"))
    deepseek_max_tokens = int(os.environ.get("DEEPSEEK_WARN_AUDIT_MAX_TOKENS", "6000"))
    qwen_timeout_s = float(os.environ.get("QWEN_WARN_AUDIT_TIMEOUT_S", str(timeout_s)))
    qwen_max_tokens = int(os.environ.get("QWEN_WARN_AUDIT_MAX_TOKENS", str(max_tokens)))
    glm_timeout_s = float(os.environ.get("GLM_WARN_AUDIT_TIMEOUT_S", "180"))
    glm_max_tokens = int(os.environ.get("GLM_WARN_AUDIT_MAX_TOKENS", str(max_tokens)))
    deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    qwen_model = os.environ.get("QWEN_MODEL", "qwen3.7-max")
    glm_model = os.environ.get("GLM_MODEL") or os.environ.get("ZHIPU_MODEL", "glm-4.6")
    return [
        DeepSeekClient(
            model=deepseek_model,
            name=deepseek_model,
            timeout_s=deepseek_timeout_s,
            max_tokens=deepseek_max_tokens,
        ),
        QwenClient(
            model=qwen_model,
            name=qwen_model,
            timeout_s=qwen_timeout_s,
            max_tokens=qwen_max_tokens,
        ),
        GLMClient(
            model=glm_model,
            name=glm_model,
            timeout_s=glm_timeout_s,
            max_tokens=glm_max_tokens,
        ),
    ]


def make_static_clients(payload: dict[str, Any] | None = None) -> list[StaticJSONClient]:
    payload = payload or {
        "primary_audit_diagnosis": "insufficient_evidence_for_diagnosis",
        "secondary_audit_diagnoses": [],
        "diagnosis_basis": ["static_test_payload"],
        "recommended_reentry_module": "audit_review",
        "recommended_action": "open_review_packet",
        "repair_tool": "open_review_packet",
        "tool_parameters": {"review_kind": "insufficient_evidence_for_diagnosis"},
        "confidence": 0.5,
        "requires_freeze": True,
        "rationale": "static local audit payload",
        "modification_plan": {
            "repair_objective": "open review packet",
            "tool_adjustments": {},
            "expected_success_condition": "sufficient evidence is gathered",
            "do_not_change": ["rule thresholds", "mdp physics", "canonical release"],
        },
    }
    return [
        StaticJSONClient("deepseek-v3-plus", payload),
        StaticJSONClient("claude-sonnet-4-6", payload),
        StaticJSONClient("glm-4.6", payload),
    ]


def compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keep = [
        "case_id",
        "case_status",
        "warning_reason",
        "density_tier_si_o",
        "status_density",
        "status_MSD",
        "status_Rg",
        "status_RDF",
        "rdf_validation_status",
        "status_shape",
        "status_ShapeTensor",
        "status_contact",
        "status_FsQ",
        "engine_qualified",
        "force_field_supported",
        "ml_usable",
        "audit_disagreement_count",
        "rule_equilibration",
        "rule_fit_quality",
        "rule_anomaly",
        "status_packing",
        "void_flag",
        "void_fraction",
        "low_density_voxel_fraction",
        "largest_connected_low_density_fraction",
        "largest_void_radius_nm",
        "min_nonbonded_distance_nm",
        "packing_uniformity_score",
        "box_occupancy_ratio",
        "packing_method",
        "packing_source_path",
        "packing_missing_reason",
    ]
    return {key: _json_safe(row.get(key, "")) for key in keep if key in row}


def _independent_evidence_view(evidence_packets: Mapping[str, Any]) -> dict[str, Any]:
    """Remove case-specific deterministic route hints before LLM audit."""

    cleaned: dict[str, Any] = {}
    for warn_source, raw_packet in evidence_packets.items():
        cleaned[str(warn_source)] = _strip_case_route_hints(raw_packet)
    return cleaned


def _strip_case_route_hints(value: Any) -> Any:
    """Return a JSON-safe evidence view without diagnosis or repair-route hints."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key in CASE_EVIDENCE_ROUTE_HINT_KEYS:
                continue
            cleaned[key] = _strip_case_route_hints(raw_value)
        return cleaned
    if isinstance(value, list):
        return [_strip_case_route_hints(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_case_route_hints(item) for item in value]
    return _json_safe(value)


def build_prompt(row: Mapping[str, Any], deterministic_diagnosis: Mapping[str, Any]) -> str:
    taxonomy = warn_diagnosis_taxonomy()
    warn_sources = list(deterministic_diagnosis.get("warn_sources") or collect_warn_sources(row))
    evidence_packets = deterministic_diagnosis.get("warn_evidence_packets")
    if not isinstance(evidence_packets, dict):
        evidence_packets = build_warn_evidence_packets(row, warn_sources)
    independent_packets = _independent_evidence_view(evidence_packets)
    payload = {
        "audit_prompt_mode": AUDIT_PROMPT_MODE,
        "task": (
            "Diagnose a polymer-MD WARN case and propose a concrete, bounded repair/modification plan. "
            "WARN is only a trigger, not a cause. The case_evidence contains frozen multimodal evidence packets: "
            "scalar descriptors, protocol/log text, structure/trajectory paths, diagnostic plot paths. "
            "Use these evidence modalities together when attributing cause, for example scalar density tier + mdout.mdp + clean-control overlay. "
            "The model receives structured text and frozen path metadata only; do not assume permission to open binary trajectories, images, "
            "or unlisted files. Prefer the best primary_audit_diagnosis from the allowed taxonomy. "
            "If the allowed taxonomy cannot explain the evidence, provide a free-text cause diagnosis and "
            "open_cause_hypothesis. You may propose a registered MCP tool call, but it will only be used if "
            "a deterministic guard validates the tool name, parameters, evidence sufficiency, and forbidden changes."
        ),
        "schema": {
            "primary_audit_diagnosis": "string; prefer one allowed diagnosis, but may be a free-text cause if taxonomy is insufficient",
            "secondary_audit_diagnoses": "array of allowed diagnosis strings",
            "diagnosis_basis": "array of evidence-field strings used",
            "recommended_reentry_module": "one of emc, md, postprocess, rule_review, audit_review, unsupported",
            "recommended_action": "string; registered action, not free shell",
            "repair_tool": "registered MCP tool name",
            "tool_parameters": "object; bounded parameters for the registered tool",
            "confidence": "number in [0,1]",
            "requires_freeze": "boolean",
            "rationale": "short scientific rationale",
            "cause_diagnosis": "same as primary_audit_diagnosis; included for explicit cause/action separation",
            "repair_instruction": {
                "repair_tool": "registered MCP tool name",
                "tool_parameters": "bounded parameters selected from the route allowlist",
                "expected_success_condition": "observable condition for PASS/review after re-gate",
            },
            "modification_plan": {
                "repair_objective": "what the agent should try to change or rerun",
                "tool_adjustments": "bounded tool parameters; do not include shell commands",
                "expected_success_condition": "observable condition for PASS/review",
                "do_not_change": "array of forbidden changes such as rule thresholds or mdp physics unless routed to review",
            },
            "open_cause_hypothesis": "optional string; free-text candidate cause if allowed taxonomy is insufficient",
            "hypothesis_evidence_basis": "optional array of evidence-field strings supporting the open hypothesis",
            "hypothesis_test_suggestion": "optional string describing how to validate the open hypothesis",
            "is_registered_diagnosis": "boolean; false when using open_cause_hypothesis instead of a registered diagnosis",
            "raw_model_diagnosis": "optional raw natural-language cause before normalization",
            "raw_model_repair_suggestion": "optional raw natural-language repair suggestion before normalization",
            "llm_tool_call_proposal": {
                "tool": "registered MCP tool name",
                "parameters": "bounded parameters for that tool",
                "rationale": "why this tool matches the evidence",
                "expected_success_condition": "observable condition after re-gate",
            },
        },
        "allowed_warn_sources": taxonomy["warn_sources"],
        "allowed_diagnoses_by_category": taxonomy["taxonomy"],
        "allowed_routes": taxonomy["routes"],
        "repair_tool_affordances": repair_tool_affordances(),
        "case_evidence": {
            "summary": compact_row(row),
            "warn_evidence_packets": independent_packets,
            "warn_evidence_packet_note": (
                "Each packet is a frozen multimodal evidence packet. evidence_modalities separates scalar descriptors, "
                "protocol/log evidence, structure/trajectory paths, and diagnostic plot paths. "
                "Paths are provenance references; do not treat them as permission to inspect mutable or unlisted files."
            ),
            "evidence_sufficiency": _strip_case_route_hints(deterministic_diagnosis.get("evidence_sufficiency", {})),
        },
        "hard_boundaries": [
            "Do not execute shell commands.",
            "Do not change mdp physics.",
            "Do not change rule thresholds at runtime.",
            "Do not decide PASS/FAIL directly.",
            "Select only registered modules and repair tools.",
            "Tool-call proposals are advisory until deterministic guard validation passes.",
            "No deterministic fallback diagnosis or case-specific route hint is provided to the model.",
        ],
    }
    return (
        "Return exactly one JSON object matching the schema. No markdown.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _prompt_payload(prompt: str) -> dict[str, Any]:
    return json.loads(prompt.split("\n\n", 1)[1])


def _walk_mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            keys.append(str(raw_key))
            keys.extend(_walk_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_mapping_keys(item))
    return keys


def prompt_independence_report(prompt: str, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit whether case evidence sent to an LLM contains deterministic route hints."""

    payload = _prompt_payload(prompt)
    case_evidence = payload.get("case_evidence", {})
    case_evidence_text = json.dumps(case_evidence, ensure_ascii=False, sort_keys=True)
    leaked_keys = sorted(set(_walk_mapping_keys(case_evidence)) & CASE_EVIDENCE_ROUTE_HINT_KEYS)
    fallback_diag = str((fallback or {}).get("primary_audit_diagnosis", "") or "")
    contains_fallback_diag = bool(fallback_diag and fallback_diag in case_evidence_text)
    return {
        "prompt_independence_pass": bool(not leaked_keys and not contains_fallback_diag),
        "case_evidence_route_hint_key_count": len(leaked_keys),
        "case_evidence_route_hint_keys": leaked_keys,
        "case_evidence_contains_fallback_diagnosis": contains_fallback_diag,
        "fallback_diagnosis_checked": fallback_diag,
        "audit_prompt_mode": payload.get("audit_prompt_mode", ""),
    }


def validate_vote(
    payload: dict[str, Any],
    fallback: Mapping[str, Any],
    *,
    allow_fallback: bool = False,
    allow_aggregator_labels: bool = False,
) -> dict[str, Any]:
    explicit_diagnosis = str(
        payload.get("cause_diagnosis") or payload.get("primary_audit_diagnosis") or ""
    ).strip()
    if not explicit_diagnosis and not allow_fallback:
        raise ValueError("model response is missing an explicit audit diagnosis")
    diagnosis = explicit_diagnosis or str(fallback.get("primary_audit_diagnosis") or "")
    if explicit_diagnosis in AGGREGATOR_ONLY_DIAGNOSES and not allow_aggregator_labels:
        raise ValueError(f"individual model response used aggregator-only diagnosis {explicit_diagnosis!r}")
    raw_model_diagnosis = str(payload.get("raw_model_diagnosis") or diagnosis).strip()
    raw_model_repair_suggestion = str(
        payload.get("raw_model_repair_suggestion")
        or payload.get("recommended_action")
        or payload.get("rationale")
        or ""
    ).strip()
    is_registered = diagnosis in DIAGNOSIS_TO_ROUTE
    open_hypothesis = str(payload.get("open_cause_hypothesis") or "").strip()
    if not is_registered and not open_hypothesis:
        open_hypothesis = diagnosis
    route_diagnosis = diagnosis if is_registered else "insufficient_evidence_for_diagnosis"
    route = route_from_primary_diagnosis(route_diagnosis)
    module_name = str(payload.get("recommended_reentry_module") or route["module"])
    if module_name not in DETERMINISTIC_MODULES:
        module_name = route["module"]
    confidence = float(payload.get("confidence", fallback.get("confidence", 0.5)))
    confidence = max(0.0, min(1.0, confidence))
    secondary = [
        str(item)
        for item in _as_list(payload.get("secondary_audit_diagnoses"))
        if str(item) in DIAGNOSIS_TO_ROUTE and str(item) != diagnosis
    ]
    modification_plan = payload.get("modification_plan") if isinstance(payload.get("modification_plan"), dict) else {}
    repair_instruction = payload.get("repair_instruction") if isinstance(payload.get("repair_instruction"), dict) else {}
    requested_parameters: dict[str, Any] = {}
    if isinstance(repair_instruction.get("tool_parameters"), dict):
        requested_parameters.update(repair_instruction["tool_parameters"])
    if isinstance(payload.get("tool_parameters"), dict):
        requested_parameters.update(payload["tool_parameters"])

    explicit_tool_proposal = payload.get("llm_tool_call_proposal") if isinstance(payload.get("llm_tool_call_proposal"), dict) else None
    proposal_parameters = dict(requested_parameters)
    if explicit_tool_proposal and isinstance(explicit_tool_proposal.get("parameters"), dict):
        proposal_parameters.update(explicit_tool_proposal["parameters"])
    proposal = {
        "case_id": fallback.get("case_id", ""),
        "warn_sources": list(fallback.get("warn_sources", [])),
        "tool": str(
            (explicit_tool_proposal or {}).get("tool")
            or repair_instruction.get("repair_tool")
            or payload.get("repair_tool")
            or (route["repair_tool"] if is_registered else "")
        ),
        "parameters": proposal_parameters,
        "rationale": str((explicit_tool_proposal or {}).get("rationale") or payload.get("rationale") or ""),
        "expected_success_condition": str(
            (explicit_tool_proposal or {}).get("expected_success_condition")
            or repair_instruction.get("expected_success_condition")
            or ""
        ),
    }

    guard = {
        "guard_verdict": "not_evaluated",
        "guard_rejection_reason": "",
        "tool_affordance_match": "",
        "proposal": {
            "tool": route["repair_tool"],
            "parameters": normalize_tool_parameters(route_diagnosis, requested_parameters),
            "rationale": str(payload.get("rationale") or ""),
            "expected_success_condition": str(repair_instruction.get("expected_success_condition") or ""),
        },
    }
    if explicit_tool_proposal or not is_registered:
        guard = validate_repair_tool_call(proposal, fallback)

    if not is_registered:
        if not open_hypothesis:
            open_hypothesis = diagnosis
    guard_verdict = str(guard.get("guard_verdict", "not_evaluated"))
    guarded_proposal = guard.get("proposal") if isinstance(guard.get("proposal"), dict) else {}
    if guard_verdict == "pass":
        tool_parameters = dict(guarded_proposal.get("parameters") or {})
        repair_tool = str(guarded_proposal.get("tool") or route["repair_tool"])
        module_name = str(repair_tool_affordances().get(repair_tool, {}).get("module") or module_name)
    elif guard_verdict in {"reject", "review"}:
        tool_parameters = {"review_kind": "guard_rejected_tool_call"}
        repair_tool = "open_review_packet"
        module_name = "audit_review"
    else:
        tool_parameters = normalize_tool_parameters(route_diagnosis, requested_parameters)
        repair_tool = route["repair_tool"]
    if not modification_plan:
        modification_plan = {
            "repair_objective": route["repair_action"] if guard_verdict != "pass" else "execute guarded tool proposal",
            "tool_adjustments": dict(tool_parameters),
            "expected_success_condition": "rule gate no longer emits the same WARN source, or review packet is complete",
            "do_not_change": ["rule thresholds", "mdp physics", "canonical release"],
        }
    else:
        modification_plan = dict(modification_plan)
        modification_plan["tool_adjustments"] = dict(tool_parameters)
    normalized_repair_instruction = {
        "repair_tool": repair_tool,
        "tool_parameters": dict(tool_parameters),
        "expected_success_condition": str(
            repair_instruction.get("expected_success_condition")
            or guarded_proposal.get("expected_success_condition")
            or modification_plan.get("expected_success_condition")
            or "run the selected deterministic module and re-enter the rule gate"
        ),
    }
    return {
        "cause_diagnosis": diagnosis,
        "primary_audit_diagnosis": diagnosis,
        "secondary_audit_diagnoses": secondary,
        "diagnosis_basis": [
            str(item)
            for item in _as_list(payload.get("diagnosis_basis", fallback.get("diagnosis_basis", [])))
        ],
        "recommended_reentry_module": module_name,
        "recommended_action": str(payload.get("recommended_action") or route["repair_action"]),
        "repair_tool": repair_tool,
        "tool_parameters": dict(tool_parameters),
        "confidence": round(confidence, 3),
        "requires_freeze": bool(payload.get("requires_freeze", fallback.get("requires_freeze", False))),
        "rationale": str(payload.get("rationale") or fallback.get("rationale") or ""),
        "repair_instruction": normalized_repair_instruction,
        "modification_plan": _json_safe(modification_plan),
        "open_cause_hypothesis": open_hypothesis,
        "hypothesis_evidence_basis": [
            str(item)
            for item in _as_list(payload.get("hypothesis_evidence_basis", payload.get("diagnosis_basis", [])))
            if str(item).strip()
        ],
        "hypothesis_test_suggestion": str(payload.get("hypothesis_test_suggestion") or ""),
        "is_registered_diagnosis": bool(is_registered),
        "raw_model_diagnosis": raw_model_diagnosis,
        "raw_model_repair_suggestion": raw_model_repair_suggestion,
        "llm_tool_call_proposal": dict(guarded_proposal or proposal),
        "tool_affordance_match": str(guard.get("tool_affordance_match", "")),
        "guard_verdict": guard_verdict,
        "guard_rejection_reason": str(guard.get("guard_rejection_reason", "")),
    }


def call_client(client: AuditClient, prompt: str, fallback: Mapping[str, Any], retries: int = 1) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(max(1, int(retries) + 1)):
        try:
            return validate_vote(parse_json_payload(client.generate(prompt)), fallback)
        except Exception as exc:  # noqa: BLE001 - preserve model diagnostics
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(str(last_error) if last_error else "model call failed")


def make_cache_key(
    row: Mapping[str, Any],
    fallback: Mapping[str, Any],
    client_names: list[str],
    prompt: str,
    *,
    audit_strategy: str = "tri-model-majority",
) -> str:
    payload = {
        "schema_version": WARN_AUDIT_SCHEMA_VERSION,
        "audit_strategy": audit_strategy,
        "case_id": row.get("case_id", ""),
        "row": compact_row(row),
        "fallback": fallback,
        "client_names": client_names,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def aggregate_votes(votes: list[dict[str, Any]], fallback: Mapping[str, Any]) -> dict[str, Any]:
    if not votes:
        selected = validate_vote({}, fallback, allow_fallback=True)
        selected["selected_by"] = "deterministic_fallback_no_valid_model_vote"
        selected["action_selected_by"] = "deterministic_fallback_no_valid_model_vote"
        selected["cross_model_disagreement"] = False
        return selected
    counts = Counter(vote["primary_audit_diagnosis"] for vote in votes)
    top_count = counts.most_common(1)[0][1]
    top = sorted([diagnosis for diagnosis, count in counts.items() if count == top_count])[0]
    if top_count < 2:
        labels = sorted(counts)
        selected = validate_vote(
            {
                "primary_audit_diagnosis": "cross_model_disagreement",
                "cause_diagnosis": "cross_model_disagreement",
                "diagnosis_basis": ["independent_model_vote_split"],
                "recommended_reentry_module": "audit_review",
                "recommended_action": "open_audit_review_packet",
                "repair_tool": "open_review_packet",
                "confidence": 0.5,
                "requires_freeze": True,
                "rationale": "Independent audit votes have no majority: " + ", ".join(labels),
            },
            fallback,
            allow_aggregator_labels=True,
        )
        selected["selected_by"] = "cross_model_disagreement_no_majority"
        selected["action_selected_by"] = "cross_model_disagreement_no_majority"
    else:
        cause_votes = [vote for vote in votes if vote["primary_audit_diagnosis"] == top]
        action_counts = Counter(
            json.dumps(vote.get("tool_parameters", {}), ensure_ascii=False, sort_keys=True)
            for vote in cause_votes
        )
        top_action_count = action_counts.most_common(1)[0][1]
        top_actions = sorted(action for action, count in action_counts.items() if count == top_action_count)
        selected_action = top_actions[0]
        action_votes = [
            vote
            for vote in cause_votes
            if json.dumps(vote.get("tool_parameters", {}), ensure_ascii=False, sort_keys=True) == selected_action
        ]
        selected = dict(sorted(action_votes, key=lambda item: -float(item.get("confidence", 0.0)))[0])
        selected["selected_by"] = "tri_model_majority"
        selected["action_selected_by"] = "repair_instruction_majority" if top_action_count >= 2 else "highest_confidence_action"
    selected["cross_model_disagreement"] = len(counts) > 1
    return selected


def incomplete_panel_review(
    votes: list[dict[str, Any]],
    fallback: Mapping[str, Any],
    *,
    required_votes: int,
) -> dict[str, Any]:
    selected = validate_vote(
        {
            "primary_audit_diagnosis": "low_confidence_audit",
            "cause_diagnosis": "low_confidence_audit",
            "diagnosis_basis": ["incomplete_model_panel"],
            "recommended_reentry_module": "audit_review",
            "recommended_action": "open_review_packet",
            "repair_tool": "open_review_packet",
            "confidence": 0.0,
            "requires_freeze": True,
            "rationale": f"Only {len(votes)} of {required_votes} required independent model votes are valid.",
        },
        fallback,
        allow_aggregator_labels=True,
    )
    selected["selected_by"] = "incomplete_model_panel_review"
    selected["action_selected_by"] = "incomplete_model_panel_review"
    selected["cross_model_disagreement"] = False
    return selected


def _call_clients(
    clients: list[AuditClient],
    prompt: str,
    fallback: Mapping[str, Any],
    *,
    retries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    votes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def invoke(client: AuditClient) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            vote = call_client(client, prompt, fallback, retries=retries)
            return client.name, vote, None
        except Exception as exc:  # noqa: BLE001
            return client.name, None, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(clients))) as executor:
        futures = {executor.submit(invoke, client): idx for idx, client in enumerate(clients)}
        results: list[tuple[int, str, dict[str, Any] | None, str | None]] = []
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            name, vote, error = future.result()
            results.append((idx, name, vote, error))
    for _, name, vote, error in sorted(results, key=lambda item: item[0]):
        if vote is not None:
            votes.append({"client": name, **vote})
        elif error is not None:
            errors.append({"client": name, "error": error})
    return votes, errors


def _strip_client(votes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in vote.items() if key != "client"} for vote in votes]


def _action_key(vote: Mapping[str, Any]) -> str:
    return json.dumps(vote.get("tool_parameters", {}), ensure_ascii=False, sort_keys=True)


def classify_initial_consensus(votes: list[dict[str, Any]], *, confidence_threshold: float) -> tuple[str, str]:
    if not votes:
        return "no_valid_votes", "no_valid_model_vote"
    if any(str(vote.get("open_cause_hypothesis", "")).strip() for vote in votes):
        return "open_hypothesis", "open_cause_hypothesis"
    diagnoses = [str(vote.get("primary_audit_diagnosis", "")) for vote in votes]
    diagnosis_counts = Counter(diagnoses)
    if len(diagnosis_counts) > 1:
        return "split", "initial_diagnosis_split"
    confidences = [float(vote.get("confidence", 0.0) or 0.0) for vote in votes]
    if confidences and min(confidences) < confidence_threshold:
        return "low_confidence", "low_confidence_consensus"
    action_counts = Counter(_action_key(vote) for vote in votes)
    if len(action_counts) > 1:
        return "cause_consensus_action_split", "initial_action_split"
    return "full_consensus", ""


def build_deliberation_prompt(
    row: Mapping[str, Any],
    deterministic_diagnosis: Mapping[str, Any],
    initial_votes: list[dict[str, Any]],
    initial_selected: Mapping[str, Any],
    *,
    deliberation_reason: str,
) -> str:
    taxonomy = warn_diagnosis_taxonomy()
    reviewer_votes = []
    for index, vote in enumerate(initial_votes):
        reviewer_votes.append(
            {
                "reviewer": f"Reviewer {chr(ord('A') + index)}",
                "primary_audit_diagnosis": vote.get("primary_audit_diagnosis"),
                "diagnosis_basis": vote.get("diagnosis_basis", []),
                "repair_instruction": vote.get("repair_instruction", {}),
                "tool_parameters": vote.get("tool_parameters", {}),
                "confidence": vote.get("confidence"),
                "rationale": vote.get("rationale", ""),
                "open_cause_hypothesis": vote.get("open_cause_hypothesis", ""),
            }
        )
    payload = {
        "task": (
            "Deliberate over anonymous independent WARN diagnoses for the same polymer-MD case. "
            "Choose the evidence-supported registered cause and bounded repair action. "
            "If the registered taxonomy is too narrow, preserve a free-text cause and propose a guarded MCP tool call. "
            "If the evidence remains insufficient or the reviewers cannot be reconciled, route to review."
        ),
        "deliberation_reason": deliberation_reason,
        "schema": {
            "primary_audit_diagnosis": "string; prefer one allowed diagnosis, but may be a free-text cause if taxonomy is insufficient",
            "secondary_audit_diagnoses": "array of allowed diagnosis strings",
            "diagnosis_basis": "array of evidence-field strings used",
            "recommended_reentry_module": "one of emc, md, postprocess, rule_review, audit_review, unsupported",
            "recommended_action": "string; registered action, not free shell",
            "repair_tool": "registered MCP tool name",
            "tool_parameters": "object; bounded parameters for the registered tool",
            "confidence": "number in [0,1]",
            "requires_freeze": "boolean",
            "rationale": "short rationale that directly compares the reviewer votes",
            "cause_diagnosis": "same as primary_audit_diagnosis",
            "repair_instruction": {
                "repair_tool": "registered MCP tool name",
                "tool_parameters": "bounded parameters selected from the route allowlist",
                "expected_success_condition": "observable condition after re-gate",
            },
            "open_cause_hypothesis": "optional; only if registered taxonomy is insufficient",
            "hypothesis_evidence_basis": "optional evidence fields",
            "hypothesis_test_suggestion": "optional validation test",
            "is_registered_diagnosis": "boolean",
            "raw_model_diagnosis": "optional raw natural-language cause before normalization",
            "raw_model_repair_suggestion": "optional raw natural-language repair suggestion before normalization",
            "llm_tool_call_proposal": {
                "tool": "registered MCP tool name",
                "parameters": "bounded parameters for that tool",
                "rationale": "why this tool matches the evidence",
                "expected_success_condition": "observable condition after re-gate",
            },
        },
        "allowed_diagnoses_by_category": taxonomy["taxonomy"],
        "allowed_routes": taxonomy["routes"],
        "repair_tool_affordances": repair_tool_affordances(),
        "audit_prompt_mode": AUDIT_PROMPT_MODE,
        "initial_selected_record": initial_selected,
        "anonymous_initial_votes": reviewer_votes,
        "case_evidence": {
            "summary": compact_row(row),
            "warn_evidence_packets": _independent_evidence_view(
                deterministic_diagnosis.get("warn_evidence_packets", {})
            ),
            "evidence_sufficiency": _strip_case_route_hints(deterministic_diagnosis.get("evidence_sufficiency", {})),
        },
        "hard_boundaries": [
            "Do not execute shell commands.",
            "Do not change mdp physics.",
            "Do not change rule thresholds at runtime.",
            "Do not decide PASS/FAIL directly.",
            "Select only registered modules and repair tools.",
            "Tool-call proposals are advisory until deterministic guard validation passes.",
        ],
    }
    return (
        "Return exactly one JSON object matching the schema. No markdown.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


MSD_CLEAN_MD_REQUIRED_MARKERS = (
    "msd_engine_provenance_warn",
    "diffusive_regime_not_confirmed",
    "msd_tail_slope_not_diffusive",
)


def apply_publication_grade_repair_guard(
    selected: Mapping[str, Any],
    row: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    """Align an MSD repair action with frozen publication-grade evidence policy.

    A local window repair remains available for clean trajectories whose global
    diffusive regime is already confirmed.  Engine-provenance failure or an
    unconfirmed global diffusive regime requires clean MD production/resume,
    full postprocessing and re-gating.  This guard selects a registered route;
    it does not infer physics, edit thresholds or alter the model-voted cause.
    """

    guarded = dict(selected)
    diagnosis = str(guarded.get("primary_audit_diagnosis") or "")
    if diagnosis not in {"fit_window_or_method_issue", "insufficient_equilibration"}:
        return guarded
    warn_sources = {str(item) for item in fallback.get("warn_sources", [])}
    if "msd_warn" not in warn_sources:
        return guarded

    evidence_text = "\n".join(
        [
            str(row.get("warning_reason") or ""),
            json.dumps(fallback.get("warn_evidence_packets", {}), ensure_ascii=False, sort_keys=True),
        ]
    ).lower()
    trigger_markers = [marker for marker in MSD_CLEAN_MD_REQUIRED_MARKERS if marker in evidence_text]
    if not trigger_markers:
        return guarded

    route = {
        "repair_route": "repair_md_clean_nvt_extension",
        "repair_action": "extend_clean_nvt_prod_from_checkpoint",
        "automation_level": "sidecar_md_execution_extension",
        "executor": "pelcb_clean_nvt_extension.py",
        "module": "md",
        "repair_tool": "extend_clean_nvt_prod",
        "tool_parameters": {
            "repair_kind": "clean_nvt_extension",
            "reentry_stage": "nvt_prod",
            "expected_success_condition": (
                "clean tail-window diffusion passes the strict MSD publication gate"
            ),
        },
    }
    prior_tool = str(guarded.get("repair_tool") or "")
    prior_action_selected_by = str(guarded.get("action_selected_by") or "")
    expected = str(
        route.get("tool_parameters", {}).get("expected_success_condition")
        or "clean MSD evidence passes the strict publication gate after full postprocessing and re-gating"
    )
    guarded.update(
        {
            "repair_route": route["repair_route"],
            "recommended_reentry_module": route["module"],
            "recommended_action": route["repair_action"],
            "recommended_automation_level": route["automation_level"],
            "recommended_executor": route["executor"],
            "repair_tool": route["repair_tool"],
            "tool_parameters": dict(route["tool_parameters"]),
            "repair_instruction": {
                "repair_tool": route["repair_tool"],
                "tool_parameters": dict(route["tool_parameters"]),
                "expected_success_condition": expected,
            },
            "llm_tool_call_proposal": {
                "tool": route["repair_tool"],
                "parameters": dict(route["tool_parameters"]),
                "rationale": "deterministic publication-grade MSD evidence guard",
                "expected_success_condition": expected,
            },
            "tool_affordance_match": route["repair_tool"],
            "guard_verdict": "pass",
            "guard_rejection_reason": "",
            "model_action_selected_by": prior_action_selected_by,
            "action_selected_by": "deterministic_publication_grade_guard",
            "publication_repair_guard": {
                "applied": True,
                "policy": "clean_msd_before_publication.v1",
                "reason": "clean_source_or_global_diffusion_not_confirmed",
                "trigger_markers": trigger_markers,
                "model_selected_tool": prior_tool,
                "guard_selected_tool": route["repair_tool"],
                "requires_full_postprocess": True,
                "requires_rule_regate": True,
            },
        }
    )
    modification_plan = dict(guarded.get("modification_plan") or {})
    modification_plan.update(
        {
            "repair_objective": route["repair_action"],
            "tool_adjustments": dict(route["tool_parameters"]),
            "expected_success_condition": expected,
            "do_not_change": ["rule thresholds", "mdp physics", "canonical release"],
        }
    )
    guarded["modification_plan"] = modification_plan
    return guarded


def audit_warn_case(
    row: Mapping[str, Any],
    *,
    clients: list[AuditClient] | None = None,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
    retries: int = 1,
    audit_strategy: str = "tri-model-majority",
    deliberation_confidence_threshold: float = DELIBERATION_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    if audit_strategy not in AUDIT_STRATEGIES:
        raise ValueError(f"unknown audit_strategy {audit_strategy!r}; expected one of {sorted(AUDIT_STRATEGIES)}")
    fallback = diagnose_case(row)
    warn_sources = fallback.get("warn_sources", [])
    if fallback.get("rule_verdict") == "PASS" or not warn_sources:
        return {
            **fallback,
            "strategy": audit_strategy,
            "diagnosis_source": "no_warn_no_llm_call",
            "modification_plan": {},
            "model_votes": [],
            "initial_model_votes": [],
            "deliberation_votes": [],
            "model_errors": [],
            "llm_model_count": 0,
            "model_call_count": 0,
            "deliberation_model_count": 0,
            "cross_model_disagreement": False,
            "current_panel_disagreement": False,
            "retained_prior_audit_disagreement": False,
            "audit_disagreement_origin": "none",
            "current_panel_vote_labels": [],
            "deliberation_triggered": False,
            "final_diagnosis": fallback.get("primary_audit_diagnosis", ""),
            "final_repair_tool": "",
            "final_tool_parameters": {},
            "route_plan": {},
        }
    clients = clients or default_clients()
    if audit_strategy == "single-model":
        if not clients:
            raise ValueError("single-model WARN audit requires at least 1 client")
        active_clients = clients[:1]
    else:
        if len(clients) != 3:
            raise ValueError(f"{audit_strategy} WARN audit requires exactly 3 clients, got {len(clients)}")
        active_clients = clients
    if audit_strategy != "single-model" and len(active_clients) != 3:
        raise ValueError(f"tri-model WARN audit requires exactly 3 clients, got {len(clients)}")
    prompt = build_prompt(row, fallback)
    initial_prompt_independence = prompt_independence_report(prompt, fallback)
    cache_root = Path(cache_dir) if cache_dir else DEFAULT_OUTPUT_DIR / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = make_cache_key(row, fallback, [client.name for client in active_clients], prompt, audit_strategy=audit_strategy)
    cache_path = cache_root / f"{cache_key}.json"
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))["diagnosis"]

    votes, errors = _call_clients(active_clients, prompt, fallback, retries=retries)
    required_valid_votes = 1 if audit_strategy == "single-model" else 3
    panel_complete = len(votes) >= required_valid_votes
    if audit_strategy == "single-model" and votes:
        initial_selected = dict(_strip_client(votes)[0])
        initial_selected["selected_by"] = "single_model"
        initial_selected["action_selected_by"] = "single_model"
        initial_selected["cross_model_disagreement"] = False
    elif votes and not panel_complete:
        initial_selected = incomplete_panel_review(
            _strip_client(votes),
            fallback,
            required_votes=required_valid_votes,
        )
    else:
        initial_selected = aggregate_votes(_strip_client(votes), fallback)
    if votes and not panel_complete:
        initial_status, deliberation_reason = "incomplete_panel", "incomplete_model_panel"
    else:
        initial_status, deliberation_reason = classify_initial_consensus(
            votes,
            confidence_threshold=deliberation_confidence_threshold,
        )
    deliberation_votes: list[dict[str, Any]] = []
    deliberation_errors: list[dict[str, str]] = []
    deliberation_prompt_independence: dict[str, Any] = {}
    deliberation_triggered = False
    if panel_complete and audit_strategy == "always-deliberation":
        deliberation_triggered = True
        deliberation_reason = "always_deliberation"
    elif panel_complete and audit_strategy == "conditional-deliberation" and initial_status != "full_consensus":
        deliberation_triggered = True
    if deliberation_triggered:
        deliberation_prompt = build_deliberation_prompt(
            row,
            fallback,
            votes,
            initial_selected,
            deliberation_reason=deliberation_reason,
        )
        deliberation_prompt_independence = prompt_independence_report(deliberation_prompt, fallback)
        deliberation_votes, deliberation_errors = _call_clients(active_clients, deliberation_prompt, fallback, retries=retries)
        selected = aggregate_votes(_strip_client(deliberation_votes), fallback)
        selected["selected_by"] = (
            "deliberation_majority"
            if selected.get("selected_by") == "tri_model_majority"
            else f"deliberation_{selected.get('selected_by')}"
        )
    else:
        selected = initial_selected
    selected = apply_publication_grade_repair_guard(selected, row, fallback)
    selected_route = route_from_diagnosis(selected)
    route_selected_by = str(selected.get("selected_by", ""))
    if str(selected.get("guard_verdict", "")) == "pass" and not bool(selected.get("is_registered_diagnosis", True)):
        route_selected_by = "llm_tool_router_guarded"
    elif str(selected.get("guard_verdict", "")) in {"reject", "review"}:
        route_selected_by = "review_fallback"
    initial_diag = str(initial_selected.get("primary_audit_diagnosis", ""))
    final_diag = str(selected.get("primary_audit_diagnosis", ""))
    fallback_diag = str(fallback.get("primary_audit_diagnosis", ""))
    initial_params = dict(initial_selected.get("tool_parameters", {}) or {})
    final_params = dict(selected.get("tool_parameters", {}) or {})
    all_errors = [*errors, *deliberation_errors]
    selected_by = str(selected.get("selected_by", ""))
    if selected_by.startswith("deterministic_fallback"):
        diagnosis_source = "deterministic_fallback_no_valid_model_vote"
    elif selected_by == "incomplete_model_panel_review":
        diagnosis_source = "tri_model_incomplete_review"
    elif final_diag == "cross_model_disagreement":
        diagnosis_source = "tri_model_disagreement_review"
    else:
        diagnosis_source = "tri_model_llm"
    disagreement_provenance = classify_audit_disagreement_provenance(
        {
            **fallback,
            **selected,
            "warn_sources": warn_sources,
            "model_votes": votes,
            "deliberation_votes": deliberation_votes,
        }
    )
    diagnosis = {
        **fallback,
        **selected,
        "schema_version": WARN_AUDIT_SCHEMA_VERSION,
        "strategy": audit_strategy,
        "diagnosis_source": diagnosis_source,
        "audit_prompt_mode": AUDIT_PROMPT_MODE,
        "case_id": str(row.get("case_id", fallback.get("case_id", ""))),
        "rule_verdict": "FAIL"
        if selected["primary_audit_diagnosis"]
        in {"unsupported_by_current_system", "topology_unsupported", "force_field_assignment_unsupported"}
        else "WARN",
        "warn_sources": list(warn_sources),
        "warn_source": ",".join(warn_sources),
        "audit_diagnosis": selected["primary_audit_diagnosis"],
        "recommended_action": selected_route["repair_action"],
        "repair_tool": selected_route["tool_plan"]["tool"],
        "tool_parameters": selected_route["tool_plan"]["parameters"],
        "model_votes": _json_safe(votes),
        "initial_model_votes": _json_safe(votes),
        "deliberation_votes": _json_safe(deliberation_votes),
        "model_errors": all_errors,
        "llm_model_count": len(votes),
        "required_valid_model_count": required_valid_votes,
        "model_panel_complete": bool(panel_complete),
        "model_provider_count": len(active_clients),
        "deliberation_model_count": len(deliberation_votes),
        "model_call_count": len(votes) + len(deliberation_votes),
        "model_display_names": [audit_model_display_name(client.name) for client in active_clients],
        "initial_consensus_status": initial_status,
        "deliberation_triggered": deliberation_triggered,
        "deliberation_reason": deliberation_reason if deliberation_triggered else "",
        "final_diagnosis": final_diag,
        "final_repair_tool": selected_route["tool_plan"]["tool"],
        "final_tool_parameters": selected_route["tool_plan"]["parameters"],
        "final_repair_instruction": selected.get("repair_instruction", {}),
        "deterministic_fallback_diagnosis": fallback_diag,
        "matches_deterministic_fallback": bool(final_diag == fallback_diag),
        "initial_prompt_independence": initial_prompt_independence,
        "deliberation_prompt_independence": deliberation_prompt_independence,
        "changed_after_deliberation": bool(
            deliberation_triggered and (final_diag != initial_diag or final_params != initial_params)
        ),
        "selected_by": route_selected_by,
        "consensus_selected_by": selected.get("selected_by", ""),
        "dissenting_opinions": [
            {
                "client": vote.get("client", ""),
                "primary_audit_diagnosis": vote.get("primary_audit_diagnosis", ""),
                "tool_parameters": vote.get("tool_parameters", {}),
                "rationale": vote.get("rationale", ""),
            }
            for vote in (deliberation_votes or votes)
            if vote.get("primary_audit_diagnosis") != final_diag
            or dict(vote.get("tool_parameters", {}) or {}) != final_params
        ],
        "route_plan": selected_route,
        **disagreement_provenance,
        "cache_key": cache_key,
        "created_at": iso_now(),
    }
    cache_payload = {
        "schema_version": WARN_AUDIT_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "case_id": diagnosis["case_id"],
        "cache_key": cache_key,
        "client_names": [client.name for client in active_clients],
        "prompt": prompt,
        "diagnosis": _json_safe(diagnosis),
    }
    cache_path.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return _json_safe(diagnosis)


def read_mask(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def merge_evidence_overlay(mask: pd.DataFrame, evidence_path: Path | None) -> pd.DataFrame:
    if not evidence_path:
        return mask
    evidence = read_mask(evidence_path)
    if evidence.empty:
        return mask
    if "case_id" not in mask.columns or "case_id" not in evidence.columns:
        raise ValueError("mask and evidence overlay must both contain case_id")
    overlay_cols = [col for col in evidence.columns if col != "case_id"]
    merged = mask.merge(evidence[["case_id", *overlay_cols]], on="case_id", how="left", suffixes=("", "__overlay"))
    for col in overlay_cols:
        overlay_col = f"{col}__overlay"
        if overlay_col not in merged.columns:
            continue
        if col not in mask.columns:
            merged[col] = merged[overlay_col]
        else:
            base = merged[col]
            overlay = merged[overlay_col]
            merged[col] = overlay.where(~overlay.isna() & (overlay.astype(str).str.strip() != ""), base)
        merged = merged.drop(columns=[overlay_col])
    return merged


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def model_count(row: Mapping[str, Any]) -> int:
    try:
        return int(float(row.get("llm_model_count", 0) or 0))
    except Exception:
        return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "strategy",
        "case_id",
        "rule_verdict",
        "warn_source",
        "cause_diagnosis",
        "primary_audit_diagnosis",
        "final_diagnosis",
        "recommended_reentry_module",
        "recommended_action",
        "repair_tool",
        "final_repair_tool",
        "tool_parameters",
        "final_tool_parameters",
        "confidence",
        "llm_model_count",
        "deliberation_model_count",
        "model_call_count",
        "model_display_names",
        "cross_model_disagreement",
        "current_panel_disagreement",
        "retained_prior_audit_disagreement",
        "audit_disagreement_origin",
        "current_panel_vote_labels",
        "initial_consensus_status",
        "deliberation_triggered",
        "deliberation_reason",
        "changed_after_deliberation",
        "selected_by",
        "action_selected_by",
        "consensus_selected_by",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_warn_audit_batch(
    mask_path: Path = DEFAULT_MASK,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    evidence_path: Path | None = None,
    cases: list[str] | None = None,
    max_cases: int = 0,
    clients: list[AuditClient] | None = None,
    execute: bool = False,
    allowlist_tag: str = "",
    reason: str = "",
    force_refresh: bool = False,
    retry_incomplete_jsonl: Path | None = None,
    min_model_count: int = 3,
    audit_strategy: str = "tri-model-majority",
    model_retries: int = 1,
) -> dict[str, Any]:
    if audit_strategy not in AUDIT_STRATEGIES:
        raise ValueError(f"unknown audit_strategy {audit_strategy!r}; expected one of {sorted(AUDIT_STRATEGIES)}")
    if execute:
        if allowlist_tag != ALLOWLIST_TAG:
            raise PermissionError(f"execute requires --allowlist-tag {ALLOWLIST_TAG}")
        if len(str(reason).strip()) < 12:
            raise ValueError("execute requires --reason with at least 12 characters")
    mask = merge_evidence_overlay(read_mask(mask_path), evidence_path)
    selected = mask
    previous_rows: list[dict[str, Any]] = []
    previous_by_case: dict[str, dict[str, Any]] = {}
    previous_order: list[str] = []
    retry_case_ids: set[str] | None = None
    if retry_incomplete_jsonl:
        previous_rows = read_jsonl(retry_incomplete_jsonl)
        for previous in previous_rows:
            case_id = str(previous.get("case_id", "")).strip()
            if not case_id:
                continue
            previous_by_case[case_id] = previous
            previous_order.append(case_id)
        retry_case_ids = {
            case_id
            for case_id, previous in previous_by_case.items()
            if model_count(previous) < int(min_model_count)
        }
        if cases:
            # Explicit targeted re-audits may replace a complete historical row
            # after a prompt/schema guard fix while preserving the raw ledger.
            retry_case_ids = set(cases)
        selected = selected[selected["case_id"].astype(str).isin(retry_case_ids)]
    if cases:
        selected = selected[selected["case_id"].astype(str).isin(set(cases))]
    warn_rows = []
    for _, row in selected.iterrows():
        if collect_warn_sources(row.to_dict()):
            warn_rows.append(row.to_dict())
    if max_cases > 0:
        warn_rows = warn_rows[:max_cases]
    out_dir = output_dir / ("execute_" + datetime.now().strftime("%Y%m%d_%H%M%S") if execute else "dry_run_latest")
    diagnoses: list[dict[str, Any]] = []
    for row in warn_rows:
        if execute:
            diagnoses.append(
                audit_warn_case(
                    row,
                    clients=clients,
                    cache_dir=out_dir / "cache",
                    force_refresh=force_refresh,
                    retries=model_retries,
                    audit_strategy=audit_strategy,
                )
            )
        else:
            fallback = diagnose_case(row)
            diagnoses.append(
                {
                    **fallback,
                    "strategy": audit_strategy,
                    "diagnosis_source": "dry_run_deterministic_preview",
                    "cause_diagnosis": fallback.get("primary_audit_diagnosis", ""),
                    "repair_instruction": fallback.get("repair_instruction", {}),
                    "modification_plan": {},
                    "model_votes": [],
                    "initial_model_votes": [],
                    "deliberation_votes": [],
                    "model_errors": [],
                    "llm_model_count": 0,
                    "deliberation_model_count": 0,
                    "model_call_count": 0,
                    "cross_model_disagreement": False,
                    "current_panel_disagreement": False,
                    "retained_prior_audit_disagreement": False,
                    "audit_disagreement_origin": "none",
                    "current_panel_vote_labels": [],
                    "initial_consensus_status": "",
                    "deliberation_triggered": False,
                    "deliberation_reason": "",
                    "changed_after_deliberation": False,
                    "final_diagnosis": fallback.get("primary_audit_diagnosis", ""),
                    "final_repair_tool": "",
                    "final_tool_parameters": {},
                    "route_plan": {},
                }
            )
    retried_cases = [str(row.get("case_id", "")) for row in warn_rows]
    if previous_by_case:
        merged = dict(previous_by_case)
        for diagnosis in diagnoses:
            case_id = str(diagnosis.get("case_id", "")).strip()
            if not case_id:
                continue
            previous = merged.get(case_id)
            if previous is None or model_count(diagnosis) >= model_count(previous):
                merged[case_id] = diagnosis
        ordered_cases = [case_id for case_id in previous_order if case_id in merged]
        ordered_cases.extend(sorted(case_id for case_id in merged if case_id not in set(ordered_cases)))
        diagnoses = [merged[case_id] for case_id in ordered_cases]
    write_jsonl(out_dir / "warn_audit_diagnosis.jsonl", diagnoses)
    write_csv(out_dir / "warn_audit_diagnosis_summary.csv", diagnoses)
    candidate_summary = write_candidate_ledger(diagnoses, out_dir / "diagnosis_candidates")
    summary = {
        "schema_version": WARN_AUDIT_SCHEMA_VERSION,
        "created_at": iso_now(),
        "mode": "execute" if execute else "dry-run",
        "audit_strategy": audit_strategy,
        "warn_cases": len(warn_rows),
        "diagnosed_cases": len(diagnoses),
        "llm_model_calls_recorded": sum(int(row.get("model_call_count", row.get("llm_model_count", 0)) or 0) for row in diagnoses),
        "valid_model_votes_recorded": sum(int(row.get("llm_model_count", 0) or 0) for row in diagnoses),
        "model_provider_invocations_recorded": sum(int(row.get("model_provider_count", 0) or 0) for row in diagnoses),
        "model_panel_complete_cases": sum(1 for row in diagnoses if bool(row.get("model_panel_complete"))),
        "incomplete_model_panel_cases": sum(
            1
            for row in diagnoses
            if row.get("model_panel_complete") is False and int(row.get("model_provider_count", 0) or 0) > 0
        ),
        "primary_diagnosis_counts": dict(Counter(str(row.get("primary_audit_diagnosis", "")) for row in diagnoses)),
        "llm_model_count_distribution": dict(Counter(str(model_count(row)) for row in diagnoses)),
        "deliberation_model_count_distribution": dict(
            Counter(str(int(row.get("deliberation_model_count") or 0)) for row in diagnoses)
        ),
        "model_call_count_distribution": dict(Counter(str(int(row.get("model_call_count") or 0)) for row in diagnoses)),
        "deliberation_triggered_cases": sum(1 for row in diagnoses if bool(row.get("deliberation_triggered"))),
        "changed_after_deliberation_cases": sum(1 for row in diagnoses if bool(row.get("changed_after_deliberation"))),
        "diagnosis_candidate_count": candidate_summary["candidate_count"],
        "diagnosis_candidate_eligible_count": candidate_summary["eligible_count"],
        "diagnosis_candidate_jsonl": candidate_summary["jsonl"],
        "retry_incomplete_from": str(retry_incomplete_jsonl) if retry_incomplete_jsonl else "",
        "evidence_overlay": str(evidence_path) if evidence_path else "",
        "retried_cases": retried_cases,
        "min_model_count": int(min_model_count),
        "model_retries": int(model_retries),
        "jsonl": str(out_dir / "warn_audit_diagnosis.jsonl"),
        "summary_csv": str(out_dir / "warn_audit_diagnosis_summary.csv"),
    }
    (out_dir / "warn_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "warn_audit_latest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--evidence", type=Path, default=None, help="Optional overlay evidence CSV merged by case_id before WARN selection.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Call the three configured model providers and write cache.")
    parser.add_argument("--mock-clients", action="store_true", help="Use local static clients for offline integration testing.")
    parser.add_argument("--allowlist-tag", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--retry-incomplete-jsonl",
        type=Path,
        default=None,
        help="Merge retries into an existing ledger; --cases explicitly selects complete rows for targeted replacement.",
    )
    parser.add_argument("--min-model-count", type=int, default=3)
    parser.add_argument("--model-retries", type=int, default=1, help="Retries per model provider after an invalid or failed response.")
    parser.add_argument("--audit-strategy", choices=sorted(AUDIT_STRATEGIES), default="tri-model-majority")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clients = make_static_clients() if args.mock_clients else None
    summary = run_warn_audit_batch(
        args.mask,
        args.output_dir,
        evidence_path=args.evidence,
        cases=args.cases,
        max_cases=args.max_cases,
        clients=clients,
        execute=args.execute,
        allowlist_tag=args.allowlist_tag,
        reason=args.reason,
        force_refresh=args.force_refresh,
        retry_incomplete_jsonl=args.retry_incomplete_jsonl,
        min_model_count=args.min_model_count,
        audit_strategy=args.audit_strategy,
        model_retries=args.model_retries,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
