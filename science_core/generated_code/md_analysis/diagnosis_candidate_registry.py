"""Open WARN-cause hypothesis registry and diagnosis-profile freeze helpers.

LLM audit models may propose novel causes, but novel causes do not get
execution authority.  This module stores them as candidate failure modes and
only turns eligible candidates into versioned diagnosis profiles.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "pelcb_open_diagnosis_profile.v1"
ALLOWLIST_TAG = "PELCB_MCP_EXECUTE_DIAGNOSIS_PROFILE"
PROJECT_ID = "pelcb_diagnosis_profiles"
DEFAULT_DIAGNOSIS_PROFILE_DIR = Path(__file__).resolve().parent / "output" / "diagnosis_profiles"

KNOWN_TOOL_DEFAULTS: dict[str, dict[str, Any]] = {
    "submit_md_clean_engine": {
        "module": "md",
        "tool_parameters": {"engine_profile": "gromacs_2024_4_clean", "reentry_stage": "md"},
        "repair_route": "open_profile_md_repair",
        "repair_action": "md_clean_engine_rerun",
        "automation_level": "profile_guarded_md_execution",
        "executor": "registered MCP submit_md_clean_engine",
    },
    "run_postprocess_repair": {
        "module": "postprocess",
        "tool_parameters": {"repair_kind": "input_repair"},
        "repair_route": "open_profile_postprocess_repair",
        "repair_action": "postprocess_repair",
        "automation_level": "profile_guarded_postprocess_execution",
        "executor": "registered MCP run_postprocess_repair",
    },
    "validate_or_build_emc_input": {
        "module": "emc",
        "tool_parameters": {"repair_kind": "builder_input_completion"},
        "repair_route": "open_profile_emc_repair",
        "repair_action": "emc_rebuild",
        "automation_level": "profile_guarded_emc_execution",
        "executor": "registered MCP validate_or_build_emc_input",
    },
    "open_review_packet": {
        "module": "audit_review",
        "tool_parameters": {"review_kind": "automatic_review_required"},
        "repair_route": "open_profile_review",
        "repair_action": "open_review_packet",
        "automation_level": "automatic_review_packet",
        "executor": "registered MCP open_review_packet",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_hypothesis_text(text: str) -> str:
    """Return a coarse cluster key for semantically similar open hypotheses."""
    lowered = text.lower()
    replacements = {
        "gromacs": "engine",
        "version": "engine",
        "neighbor-list": "buffer",
        "neighbor list": "buffer",
        "verlet": "buffer",
        "buffering": "buffer",
        "density-only": "density",
        "density-specific": "density",
        "shift": "artifact",
        "change": "artifact",
    }
    for old, new in replacements.items():
        lowered = lowered.replace(old, new)
    tokens = re.findall(r"[a-z0-9]+", lowered)
    stop = {"a", "an", "the", "from", "by", "of", "and", "or", "to", "causes", "creates", "specific", "only"}
    kept = sorted({token for token in tokens if token not in stop and len(token) > 2})
    if {"engine", "buffer", "density"} <= set(kept):
        return "engine_buffer_density_artifact"
    return "_".join(kept[:8]) or "open_unknown_cause"


def _candidate_id(cluster_key: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cluster_key.lower()).strip("_")
    return cleaned[:80] or "open_unknown_cause"


def _profile_diagnosis(candidate_id: str) -> str:
    return f"open__{_candidate_id(candidate_id)}"


def _iter_hypotheses(records: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in records:
        case_id = _safe_text(record.get("case_id"))
        items: list[Mapping[str, Any]] = []
        if record.get("open_cause_hypothesis"):
            items.append(record)
        for vote in record.get("model_votes", []) or []:
            if isinstance(vote, Mapping) and vote.get("open_cause_hypothesis"):
                items.append(vote)
        for item in items:
            hypothesis = _safe_text(item.get("open_cause_hypothesis"))
            if not hypothesis:
                continue
            yield {
                "case_id": _safe_text(item.get("case_id")) or case_id,
                "model_name": _safe_text(item.get("client") or item.get("model_name") or item.get("model")),
                "open_cause_hypothesis": hypothesis,
                "cluster_key": normalize_hypothesis_text(hypothesis),
                "hypothesis_evidence_basis": list(item.get("hypothesis_evidence_basis") or item.get("diagnosis_basis") or []),
                "hypothesis_test_suggestion": _safe_text(item.get("hypothesis_test_suggestion")),
                "recommended_reentry_module": _safe_text(item.get("recommended_reentry_module")),
                "repair_tool": _safe_text(item.get("repair_tool")),
                "tool_parameters": dict(item.get("tool_parameters") or {}),
            }


def _freeze_readiness(candidate: Mapping[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if int(candidate.get("case_count", 0)) < 2:
        reasons.append("needs_at_least_two_cases")
    if int(candidate.get("model_count", 0)) < 2:
        reasons.append("needs_at_least_two_models")
    if not candidate.get("evidence_basis"):
        reasons.append("needs_evidence_basis")
    if not candidate.get("test_suggestions"):
        reasons.append("needs_test_suggestion")
    if not candidate.get("repair_tools"):
        reasons.append("needs_registered_or_review_tool")
    return ("eligible_for_profile_freeze" if not reasons else "blocked_" + "_".join(reasons), reasons)


def collect_diagnosis_candidates(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _iter_hypotheses(records):
        grouped[item["cluster_key"]].append(item)

    candidates: list[dict[str, Any]] = []
    for cluster_key, items in grouped.items():
        case_ids = sorted({item["case_id"] for item in items if item["case_id"]})
        model_names = sorted({item["model_name"] for item in items if item["model_name"]})
        evidence_basis = sorted({str(field) for item in items for field in item["hypothesis_evidence_basis"] if str(field).strip()})
        test_suggestions = sorted({item["hypothesis_test_suggestion"] for item in items if item["hypothesis_test_suggestion"]})
        modules = [item["recommended_reentry_module"] for item in items if item["recommended_reentry_module"]]
        tools = [item["repair_tool"] for item in items if item["repair_tool"]]
        tool_counter = Counter(tools)
        selected_tool = tool_counter.most_common(1)[0][0] if tool_counter else "open_review_packet"
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": _candidate_id(cluster_key),
            "canonical_hypothesis": sorted(items, key=lambda item: len(item["open_cause_hypothesis"]))[0][
                "open_cause_hypothesis"
            ],
            "open_cause_hypotheses": sorted({item["open_cause_hypothesis"] for item in items}),
            "case_ids": case_ids,
            "case_count": len(case_ids),
            "model_names": model_names,
            "model_count": len(model_names),
            "evidence_basis": evidence_basis,
            "test_suggestions": test_suggestions,
            "recommended_reentry_modules": sorted(set(modules)),
            "repair_tools": sorted(set(tools)),
            "repair_tool": selected_tool,
            "tool_parameters": _select_tool_parameters(items, selected_tool),
            "created_at": utc_now(),
        }
        readiness, reasons = _freeze_readiness(candidate)
        candidate["freeze_readiness"] = readiness
        candidate["freeze_block_reasons"] = reasons
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: (item["freeze_readiness"] != "eligible_for_profile_freeze", item["candidate_id"]))


def _select_tool_parameters(items: list[dict[str, Any]], selected_tool: str) -> dict[str, Any]:
    for item in items:
        if item["repair_tool"] == selected_tool and item["tool_parameters"]:
            defaults = KNOWN_TOOL_DEFAULTS.get(selected_tool, {}).get("tool_parameters", {})
            return {**defaults, **dict(item["tool_parameters"])}
    return dict(KNOWN_TOOL_DEFAULTS.get(selected_tool, {}).get("tool_parameters", {}))


def propose_diagnosis_profile_from_candidates(
    candidate_or_candidates: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    candidates = candidate_or_candidates if isinstance(candidate_or_candidates, list) else [candidate_or_candidates]
    selected = None
    for candidate in candidates:
        if candidate_id is None or str(candidate.get("candidate_id")) == str(candidate_id):
            selected = dict(candidate)
            break
    if selected is None:
        raise ValueError(f"candidate_id not found: {candidate_id}")

    repair_tools = list(selected.get("repair_tools") or [])
    selected_tool = _safe_text(selected.get("repair_tool")) or (_safe_text(repair_tools[0]) if repair_tools else "open_review_packet")
    defaults = KNOWN_TOOL_DEFAULTS.get(selected_tool, KNOWN_TOOL_DEFAULTS["open_review_packet"])
    route = {
        "repair_route": defaults["repair_route"],
        "repair_action": defaults["repair_action"],
        "automation_level": defaults["automation_level"],
        "executor": defaults["executor"],
        "module": _safe_text(selected.get("recommended_reentry_module")) or defaults["module"],
        "repair_tool": selected_tool if selected_tool in KNOWN_TOOL_DEFAULTS else "open_review_packet",
        "tool_parameters": {**dict(defaults["tool_parameters"]), **dict(selected.get("tool_parameters") or {})},
    }
    activation_status = (
        "eligible_for_machine_freeze"
        if selected.get("freeze_readiness") == "eligible_for_profile_freeze"
        and route["repair_tool"] in KNOWN_TOOL_DEFAULTS
        else "blocked_candidate_not_ready"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": _candidate_id(_safe_text(selected.get("candidate_id"))),
        "diagnosis": _profile_diagnosis(_safe_text(selected.get("candidate_id"))),
        "canonical_hypothesis": selected.get("canonical_hypothesis", ""),
        "source_candidate_id": selected.get("candidate_id", ""),
        "case_ids": list(selected.get("case_ids", [])),
        "model_names": list(selected.get("model_names", [])),
        "evidence_basis": list(selected.get("evidence_basis", [])),
        "test_suggestions": list(selected.get("test_suggestions", [])),
        "route": route,
        "activation_status": activation_status,
        "freeze_block_reasons": list(selected.get("freeze_block_reasons", [])),
        "created_at": utc_now(),
    }


def activate_diagnosis_profile(
    profile: Mapping[str, Any],
    *,
    profile_dir: str | Path = DEFAULT_DIAGNOSIS_PROFILE_DIR,
    project_id: str,
    reason: str,
    allowlist_tag: str,
    execute: bool = False,
) -> dict[str, Any]:
    if not execute:
        return {"status": "rejected", "reason": "activate_diagnosis_profile requires execute=true"}
    if project_id != PROJECT_ID or allowlist_tag != ALLOWLIST_TAG or len(_safe_text(reason)) < 12:
        return {
            "status": "rejected",
            "reason": "activate_diagnosis_profile requires project_id=pelcb_diagnosis_profiles, allowlist tag, and reason",
            "required_allowlist_tag": ALLOWLIST_TAG,
        }
    if profile.get("activation_status") != "eligible_for_machine_freeze":
        return {"status": "rejected", "reason": "diagnosis profile is not eligible_for_machine_freeze"}
    root = Path(profile_dir)
    root.mkdir(parents=True, exist_ok=True)
    frozen = {**dict(profile), "freeze_status": "machine_frozen_by_codex", "frozen_at": utc_now(), "freeze_reason": reason}
    profile_path = root / f"{_candidate_id(str(frozen['profile_id']))}.json"
    profile_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    ledger_path = root / "diagnosis_profile_freeze_ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"profile_path": str(profile_path), **frozen}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"status": "ok", "profile_path": str(profile_path), "profile": frozen}


def list_active_diagnosis_profiles(profile_dir: str | Path = DEFAULT_DIAGNOSIS_PROFILE_DIR) -> list[dict[str, Any]]:
    root = Path(profile_dir)
    if not root.exists():
        return []
    profiles: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if path.name == "diagnosis_profile_freeze_ledger.jsonl":
            continue
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if profile.get("freeze_status") == "machine_frozen_by_codex":
            profiles.append(profile)
    return profiles


def route_from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    route = dict(profile.get("route") or {})
    tool = _safe_text(route.get("repair_tool")) or "open_review_packet"
    defaults = KNOWN_TOOL_DEFAULTS.get(tool, KNOWN_TOOL_DEFAULTS["open_review_packet"])
    return {
        "repair_route": route.get("repair_route") or defaults["repair_route"],
        "repair_action": route.get("repair_action") or defaults["repair_action"],
        "automation_level": route.get("automation_level") or defaults["automation_level"],
        "executor": route.get("executor") or defaults["executor"],
        "module": route.get("module") or defaults["module"],
        "repair_tool": tool if tool in KNOWN_TOOL_DEFAULTS else "open_review_packet",
        "tool_parameters": dict(route.get("tool_parameters") or defaults["tool_parameters"]),
    }


def write_candidate_ledger(records: Iterable[Mapping[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    candidates = collect_diagnosis_candidates(records)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl = root / "diagnosis_candidate_ledger.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "status": "ok",
        "candidate_count": len(candidates),
        "eligible_count": sum(1 for item in candidates if item["freeze_readiness"] == "eligible_for_profile_freeze"),
        "jsonl": str(jsonl),
        "candidates": candidates,
    }
    (root / "diagnosis_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "ALLOWLIST_TAG",
    "DEFAULT_DIAGNOSIS_PROFILE_DIR",
    "PROJECT_ID",
    "activate_diagnosis_profile",
    "collect_diagnosis_candidates",
    "list_active_diagnosis_profiles",
    "normalize_hypothesis_text",
    "propose_diagnosis_profile_from_candidates",
    "route_from_profile",
    "write_candidate_ledger",
]
