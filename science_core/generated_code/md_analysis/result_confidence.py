#!/usr/bin/env python3
"""Per-observable uncertainty, evidence checklists, and review routing.

Evidence tiers are assigned from explicit required checks rather than additive
weights. ``HIGH`` requires every applicable supporting check, ``MED`` means the
primary result checks pass while supporting evidence remains incomplete, and
``LOW`` means the value is missing or a primary result check fails.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from force_field_qualification import DEFAULT_OUTPUT_DIR as DEFAULT_FF_OUTPUT_DIR
from force_field_qualification import find_active_qualification_profile
from pelcb_msd_publication_gate import evaluate_msd_publication_gate
from phase_state_profile import (
    classify_phase_state,
    FLUID_REFERENCE_STATES,
    phase_compatibility,
    reference_phase_family,
)
from polymer_qc_rules import apply_polymer_qc_rules
from rule_profile_registry import _json_safe, canonical_property_name


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AGGREGATE_DIR = SCRIPT_DIR / "output" / "pelcb100_integrated" / "aggregate"
DEFAULT_CONFIDENCE_CSV = DEFAULT_AGGREGATE_DIR / "result_confidence.csv"
RESULT_CONFIDENCE_SCHEMA_VERSION = "result_confidence.v2"


AUTOMATIC_CHECK_ACTIONS = {
    "value_present": "rerun_missing_output_analysis",
    "primary_qc": "route_failed_result_to_audit",
    "force_field_qualification": "find_and_validate_force_field_reference",
    "engine_qualification": "run_same_input_engine_control",
    "method_verification": "run_replicate_or_analysis_sensitivity",
    "external_calibration": "find_matching_external_reference",
    "phase_compatibility": "run_phase_state_review",
    "statistical_uncertainty": "estimate_uncertainty_from_trajectory_blocks",
}

AUTOMATIC_CHECK_TOOLS = {
    "value_present": "run_postprocess_repair",
    "primary_qc": "plan_repair_route",
    "force_field_qualification": "discover_rule_evidence",
    "engine_qualification": "submit_md_clean_engine",
    "method_verification": "run_postprocess_repair",
    "external_calibration": "discover_rule_evidence",
    "phase_compatibility": "apply_polymer_qc_rules",
    "statistical_uncertainty": "run_postprocess_repair",
}


OBSERVABLES = {
    "density": {
        "value_keys": ["result_density_g_cm3", "rho_final_gcc_si_o", "rho_final_gcc", "density_gcc"],
        "property": "density",
        "units": "g/cm3",
        "status_keys": ["result_density_check", "status_density"],
        "stat_keys": ["rho_final_sem_gcc", "density_sem_gcc", "rho_sem_gcc"],
        "relative_stat_fallback": 0.01,
        "requires_ff": True,
        "external_absolute": True,
    },
    "D_mean_m2_s": {
        "value_keys": ["result_D_mean_m2_s", "D_mean_m2_s"],
        "property": "self_diffusion",
        "units": "m2/s",
        "status_keys": ["result_msd_check", "status_MSD"],
        "stat_keys": ["D_std_m2_s", "D_mad_m2_s"],
        "cv_keys": ["window_D_cv_pct", "best_local_D_cv_pct"],
        "requires_ff": False,
        "external_absolute": True,
    },
    "rg_mean_nm": {
        "value_keys": ["result_rg_mean_nm", "rg_mean_nm"],
        "property": "rg",
        "units": "nm",
        "status_keys": ["result_rg_check", "status_Rg", "status_rg"],
        "stat_keys": ["rg_sem_time_nm", "rg_std_time_nm"],
        "requires_ff": False,
        "external_absolute": False,
    },
    "ree_mean_nm": {
        "value_keys": ["ree_mean_nm"],
        "property": "ree",
        "units": "nm",
        "status_keys": ["status_Ree"],
        "stat_keys": ["ree_std_time_nm"],
        "requires_ff": False,
        "external_absolute": False,
    },
    "shape_relative_anisotropy_mean": {
        "value_keys": ["result_shape_relative_anisotropy_mean", "shape_relative_anisotropy_mean", "relative_shape_anisotropy_mean"],
        "property": "shape",
        "units": "dimensionless",
        "status_keys": ["result_shape_check", "status_shape"],
        "stat_keys": [],
        "requires_ff": False,
        "external_absolute": False,
    },
    "rdf_inter_chain_cc_first_peak_g": {
        "value_keys": ["result_rdf_inter_chain_cc_first_peak_g", "rdf_inter_chain_cc_first_peak_g", "rdf_all_first_peak_g"],
        "property": "rdf",
        "units": "dimensionless",
        "status_keys": ["result_rdf_check", "status_RDF"],
        "stat_keys": ["tail_g_std"],
        "requires_ff": False,
        "external_absolute": False,
    },
}


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    out = str(value).strip()
    return "" if out.lower() in {"nan", "none", "null"} else out


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def truthy(value: Any) -> bool:
    return text(value).lower() in {"1", "true", "yes", "ok", "pass"}


def canonical_force_field_label(value: Any) -> str:
    raw = text(value).lower()
    if "opls" in raw:
        return "opls-aa"
    return text(value) or "opls-aa"


def infer_polymer_family(row: Mapping[str, Any]) -> str:
    family = text(row.get("polymer_family"))
    case_id = text(row.get("case_id")).lower()
    if case_id.startswith("lcb_pe_") or family.upper() == "PE":
        return "PE-LCB100" if case_id.startswith("lcb_pe_") else "PE"
    return family or text(row.get("topology_family")) or "PE"


def first_float(row: Mapping[str, Any], keys: list[str]) -> tuple[float, str]:
    for key in keys:
        value = to_float(row.get(key))
        if math.isfinite(value):
            return value, key
    return math.nan, ""


def first_text(row: Mapping[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = text(row.get(key))
        if value:
            return value
    return ""


def phase_output_dir_for_ff_dir(ff_output_dir: str | Path) -> Path:
    path = Path(ff_output_dir)
    if path.name == "force_field_qualification":
        return path.parent / "phase_state_profiles"
    return path / "phase_state_profiles"


def qc_pass_for_observable(row: Mapping[str, Any], observable: str, prop: str) -> tuple[bool, str, dict[str, Any]]:
    status_keys = list(OBSERVABLES[observable]["status_keys"])
    result_status = first_text(row, [key for key in status_keys if key.startswith("result_")]).upper()
    if result_status in {"OK", "PASS", "CURRENT", "VALIDATED"}:
        return True, "", {"explicit_result_check": result_status}
    if result_status in {"WARN", "REVIEW", "FAIL", "MISSING"}:
        return False, f"explicit_result_check={result_status}", {"explicit_result_check": result_status}
    if prop == "self_diffusion":
        gate = evaluate_msd_publication_gate(row, require_clean=True)
        return bool(gate.get("msd_publication_grade")), str(gate.get("issue_string") or ""), gate
    explicit_status = first_text(row, [key for key in status_keys if not key.startswith("result_")]).upper()
    if explicit_status in {"OK", "PASS", "CURRENT", "VALIDATED"}:
        return True, "", {"explicit_result_check": explicit_status}
    if explicit_status in {"WARN", "REVIEW", "FAIL", "MISSING"}:
        return False, f"explicit_result_check={explicit_status}", {"explicit_result_check": explicit_status}
    try:
        qc = apply_polymer_qc_rules(row)
    except Exception as exc:  # defensive: confidence should not block release export
        return False, f"qc_error:{exc}", {}
    domain = "msd" if prop == "self_diffusion" else prop
    verdict = str((qc.get("domain_verdicts") or {}).get(domain) or "").upper()
    if not verdict:
        verdict = "PASS" if explicit_status in {"OK", "PASS", "CURRENT", "VALIDATED"} else "WARN"
    return verdict == "PASS", "" if verdict == "PASS" else f"{domain}_qc={verdict or 'unknown'}", qc


def statistical_uncertainty(row: Mapping[str, Any], spec: Mapping[str, Any], value: float) -> tuple[float, dict[str, Any]]:
    stat, stat_key = first_float(row, list(spec.get("stat_keys", [])))
    if math.isfinite(stat):
        return abs(stat), {"u_stat_source": stat_key}
    cv, cv_key = first_float(row, list(spec.get("cv_keys", [])))
    if math.isfinite(cv) and math.isfinite(value):
        return abs(value) * abs(cv) / 100.0, {"u_stat_source": cv_key, "u_stat_cv_pct": cv}
    fallback = spec.get("relative_stat_fallback")
    if fallback is not None and math.isfinite(value):
        return abs(value) * float(fallback), {"u_stat_source": "relative_fallback", "u_stat_relative": float(fallback)}
    return 0.0, {"u_stat_source": "not_available"}


def ff_adjustment(
    row: Mapping[str, Any],
    prop: str,
    value: float,
    *,
    ff_output_dir: str | Path = DEFAULT_FF_OUTPUT_DIR,
) -> tuple[float, float | None, bool, str, dict[str, Any]]:
    context = {
        "polymer_family": infer_polymer_family(row),
        "force_field": canonical_force_field_label(row.get("force_field") or row.get("quality_force_field")),
        "engine_scope": first_text(
            row,
            ["engine_version", "engine_scope", "quality_engine_version", "current_clean_engine_version"],
        ) or "GROMACS2024.4-clean",
        "property": canonical_property_name(prop),
    }
    profile = find_active_qualification_profile(context, output_dir=ff_output_dir, profile_kind="force_field_qualification")
    if not profile:
        return 0.0, None, False, "", {"ff_qualified": False, "ff_reason": "no_active_ff_qualification_profile"}
    ref_phase_family = reference_phase_family(profile)
    if profile.get("phase_validation_status") != "phase_validated":
        return (
            0.0,
            None,
            False,
            "",
            {
                "ff_qualified": False,
                "ff_reason": "ff_profile_phase_unverified",
                "ff_profile_id": profile.get("profile_id"),
                "ff_profile_hash": profile.get("profile_hash"),
                "ff_reference_phase_family": ref_phase_family,
                "ff_phase_validation_status": profile.get("phase_validation_status") or "missing",
            },
        )
    rel = to_float(profile.get("systematic_uncertainty"))
    offset = to_float(profile.get("systematic_offset"))
    corrected = value
    correction_applied = False
    if math.isfinite(offset) and math.isfinite(value) and (1.0 + offset) > 0:
        corrected = value / (1.0 + offset)
        correction_applied = True
    u_ff = abs(corrected) * abs(rel) if math.isfinite(rel) and math.isfinite(corrected) else 0.0
    return (
        u_ff,
        corrected if correction_applied else None,
        True,
        str(profile.get("profile_id") or ""),
        {
            "ff_qualified": True,
            "ff_profile_id": profile.get("profile_id"),
            "ff_profile_hash": profile.get("profile_hash"),
            "ff_systematic_offset": offset if math.isfinite(offset) else None,
            "ff_systematic_uncertainty": rel if math.isfinite(rel) else None,
            "ff_n_references": profile.get("n_references"),
            "ff_bias_correction_applied": correction_applied,
            "ff_reference_phase_family": ref_phase_family,
            "ff_phase_validation_status": profile.get("phase_validation_status"),
        },
    )


def apply_phase_confidence_gate(
    *,
    tier: str,
    tier_reasons: list[str],
    phase_payload: Mapping[str, Any],
    reference_phase: str,
    external_absolute: bool,
) -> tuple[str, list[str], dict[str, Any]]:
    compatibility = phase_compatibility(str(phase_payload.get("phase_state") or ""), reference_phase)
    target_state = str(phase_payload.get("target_state") or "").lower()
    target_declared_fluid = target_state in FLUID_REFERENCE_STATES
    target_phase_mismatch = target_declared_fluid and phase_payload.get("phase_state") == "hard_core_arrest"
    if target_phase_mismatch:
        compatibility = {"status": "mismatch", "confidence_cap": "MED", "reason": "phase_mismatch_review"}
    phase_reasons = list(tier_reasons)
    gated_tier = tier
    if external_absolute and compatibility.get("confidence_cap") == "MED":
        reason = str(compatibility.get("reason") or "phase_review")
        if reason not in phase_reasons:
            phase_reasons.append(reason)
        if phase_payload.get("arrest_detected") and "arrest_detected" not in phase_reasons:
            phase_reasons.append("arrest_detected")
        if gated_tier == "HIGH":
            gated_tier = "MED"
    phase_breakdown = {
        "phase_state": phase_payload.get("phase_state"),
        "phase_rule_profile_id": phase_payload.get("phase_rule_profile_id"),
        "phase_rule_profile_hash": phase_payload.get("phase_rule_profile_hash"),
        "phase_evidence_complete": phase_payload.get("phase_evidence_complete"),
        "phase_reasons": phase_payload.get("phase_reasons"),
        "phase_reference_family": reference_phase,
        "phase_target_state": target_state or None,
        "phase_compatibility_status": compatibility.get("status"),
        "phase_confidence_cap": compatibility.get("confidence_cap"),
        "phase_gate_reason": compatibility.get("reason"),
        "phase_mismatch_review": compatibility.get("status") == "mismatch",
        "phase_boundary_review": compatibility.get("status") == "boundary_review",
        "arrest_detected": phase_payload.get("arrest_detected"),
        "phase_engine": phase_payload.get("phase_engine"),
        "phase_g3_provenance": phase_payload.get("phase_g3_provenance"),
        "phase_alpha_main": phase_payload.get("alpha_main"),
        "phase_alpha_main_key": phase_payload.get("alpha_main_key"),
        "phase_alpha_late": phase_payload.get("alpha_late"),
        "phase_alpha_late_key": phase_payload.get("alpha_late_key"),
        "phase_msd_ratio": phase_payload.get("msd_ratio"),
        "phase_msd_ratio_key": phase_payload.get("msd_ratio_key"),
        "phase_rdf_peak": phase_payload.get("rdf_peak"),
        "phase_rdf_peak_key": phase_payload.get("rdf_peak_key"),
    }
    return gated_tier, phase_reasons, phase_breakdown


def engine_uncertainty(
    row: Mapping[str, Any],
    prop: str,
    value: float,
    *,
    ff_output_dir: str | Path = DEFAULT_FF_OUTPUT_DIR,
) -> tuple[float, bool, str, dict[str, Any]]:
    engine = first_text(
        row,
        ["engine_version", "engine_scope", "quality_engine_version", "current_clean_engine_version"],
    ).lower()
    clean = "2024.4" in engine or "clean" in engine or (
        canonical_property_name(prop) == "self_diffusion" and truthy(row.get("msd_clean_engine_source"))
    )
    if clean:
        return 0.0, True, "clean_engine_or_provenance", {"engine_transfer_qualified": True, "u_engine_source": "clean_engine_or_provenance"}
    if "2021.2" in engine:
        profile = find_active_qualification_profile(
            {
                "polymer_family": infer_polymer_family(row),
                "force_field": canonical_force_field_label(row.get("force_field")),
                "property": canonical_property_name(prop),
            },
            output_dir=ff_output_dir,
            profile_kind="engine_transfer",
        )
        if profile:
            factor = to_float(profile.get("transfer_factor"))
            transfer_uncertainty = to_float(profile.get("transfer_uncertainty"))
            bias = abs((1.0 / factor) - 1.0) if math.isfinite(factor) and factor > 0 else math.nan
            spread = abs(transfer_uncertainty / factor) if math.isfinite(transfer_uncertainty) and math.isfinite(factor) and factor > 0 else 0.0
            rel = max(bias if math.isfinite(bias) else 0.0, spread)
            corrected_candidate = value / factor if math.isfinite(value) and math.isfinite(factor) and factor > 0 else None
            return abs(value) * rel if math.isfinite(value) else 0.0, False, "legacy_engine_transfer_profile", {
                "engine_transfer_qualified": False,
                "u_engine_source": "engine_transfer_profile_bias_budget",
                "engine_transfer_profile_id": profile.get("profile_id"),
                "engine_transfer_factor": factor if math.isfinite(factor) else None,
                "engine_transfer_uncertainty": transfer_uncertainty if math.isfinite(transfer_uncertainty) else None,
                "engine_corrected_candidate_value": corrected_candidate,
            }
        return 0.0, False, "legacy_engine_untransferred_no_profile", {
            "engine_transfer_qualified": False,
            "u_engine_source": "legacy_engine_untransferred_no_profile",
        }
    return 0.0, False, "engine_unverified", {"engine_transfer_qualified": False, "u_engine_source": "engine_unverified"}


def method_uncertainty(row: Mapping[str, Any], value: float) -> tuple[float, bool, dict[str, Any]]:
    for key in ["method_replicate_cv_pct", "cross_engine_cv_pct", "replicate_cv_pct"]:
        cv = to_float(row.get(key))
        if math.isfinite(cv) and math.isfinite(value):
            ok = cv <= 5.0
            return abs(value) * abs(cv) / 100.0, ok, {"method_verified": ok, "u_method_source": key, "u_method_cv_pct": cv}
    if truthy(row.get("same_input_clean_engine_control")) or truthy(row.get("msd_clean_engine_source")):
        return 0.0, True, {"method_verified": True, "u_method_source": "same_input_or_clean_control"}
    return 0.0, False, {"method_verified": False, "u_method_source": "single_run_or_no_control"}


def similarity_transfer_uncertainty(row: Mapping[str, Any], prop: str, value: float) -> tuple[float, bool, dict[str, Any]]:
    """Read optional similar-structure transfer metadata from a result row."""

    del prop  # Property-specific hooks can be added when descriptor profiles mature.
    reference_id = first_text(row, ["similarity_reference_id", "rule_similarity_reference_id"])
    reason = first_text(row, ["similarity_distance_reason", "rule_similarity_distance_reason"])
    rule_basis = first_text(row, ["rule_basis", "rule_profile_basis"])
    score = to_float(row.get("similarity_score") if row.get("similarity_score") is not None else row.get("rule_similarity_score"))
    source_count = to_float(
        row.get("similarity_transfer_source_count")
        if row.get("similarity_transfer_source_count") is not None
        else row.get("similarity_reference_count")
    )
    rel = to_float(
        row.get("u_similarity_rel")
        if row.get("u_similarity_rel") is not None
        else row.get("transfer_uncertainty_rel")
    )
    u_abs = to_float(row.get("u_similarity"))
    uses_similarity = bool(reference_id or rule_basis == "similarity_transfer")
    if not uses_similarity:
        return (
            0.0,
            False,
            {
                "rule_basis": rule_basis or "direct_profile",
                "similarity_reference_id": None,
                "similarity_score": None,
                "similarity_score_interpretation": "not_applicable",
                "similarity_distance_reason": None,
                "u_similarity_source": "not_applicable",
                "similarity_transfer_review": False,
                "similarity_transfer_high_confidence": False,
            },
        )
    if not math.isfinite(u_abs):
        u_abs = abs(value) * abs(rel) if math.isfinite(value) and math.isfinite(rel) else 0.0
    reason_norm = reason or "matched"
    review = reason_norm not in {"matched", "direct_profile_match"} or not math.isfinite(score) or score < 0.80
    high_confidence = (
        not review
        and math.isfinite(score)
        and score >= 0.90
        and math.isfinite(source_count)
        and source_count >= 2
    )
    return (
        abs(u_abs),
        True,
        {
            "rule_basis": "similarity_transfer",
            "similarity_reference_id": reference_id,
            "similarity_score": score if math.isfinite(score) else None,
            "similarity_score_interpretation": "composite_comparability_score_not_property_error",
            "similarity_distance_reason": reason_norm,
            "u_similarity_source": "similarity_transfer_profile",
            "u_similarity_rel": rel if math.isfinite(rel) else None,
            "similarity_transfer_source_count": int(source_count) if math.isfinite(source_count) else None,
            "similarity_transfer_review": review,
            "similarity_transfer_high_confidence": high_confidence,
        },
    )


def confidence_tier(
    *,
    qc_pass: bool,
    ff_required: bool,
    ff_qualified: bool,
    engine_qualified: bool,
    method_verified: bool,
    external_absolute: bool,
    value_present: bool,
    calibrated_absolute: bool,
) -> tuple[str, list[str], dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, *, primary: bool = False) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else ("FAIL" if primary else "PENDING"),
                "primary": primary,
                "automatic_action": None if passed else AUTOMATIC_CHECK_ACTIONS[name],
                "registered_tool": None if passed else AUTOMATIC_CHECK_TOOLS[name],
            }
        )

    add_check("value_present", value_present, primary=True)
    add_check("primary_qc", qc_pass, primary=True)
    if ff_required or external_absolute:
        add_check("force_field_qualification", ff_qualified)
    add_check("engine_qualification", engine_qualified)
    add_check("method_verification", method_verified)
    add_check("external_calibration", bool(external_absolute and calibrated_absolute))

    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    pending = [item["name"] for item in checks if item["status"] == "PENDING"]
    passed = [item["name"] for item in checks if item["status"] == "PASS"]
    actions = [item["automatic_action"] for item in checks if item.get("automatic_action")]
    tools = list(dict.fromkeys(item["registered_tool"] for item in checks if item.get("registered_tool")))
    reasons: list[str] = []
    reason_by_check = {
        "value_present": "value_missing",
        "primary_qc": "qc_or_convergence_not_passed",
        "force_field_qualification": "force_field_unqualified",
        "engine_qualification": "engine_or_protocol_unqualified",
        "method_verification": "method_unverified",
        "external_calibration": (
            "external_absolute_not_force_field_calibrated"
            if external_absolute
            else "relative_structural_descriptor_high_not_applicable"
        ),
    }
    reasons.extend(reason_by_check[name] for name in [*failed, *pending])

    if failed:
        tier = "LOW"
    elif pending:
        tier = "MED"
    else:
        tier = "HIGH"
    checklist = {
        "tier_basis": "deterministic_required_checks",
        "checks": checks,
        "passed_checks": passed,
        "failed_checks": failed,
        "pending_checks": pending,
        "automatic_check_actions": actions,
        "automatic_check_tools": tools,
    }
    return tier, reasons, checklist


def confidence_for_observable(
    row: Mapping[str, Any],
    observable: str,
    *,
    ff_output_dir: str | Path = DEFAULT_FF_OUTPUT_DIR,
) -> dict[str, Any]:
    spec = OBSERVABLES[observable]
    value, value_key = first_float(row, list(spec["value_keys"]))
    raw_value = value
    prop = str(spec["property"])
    value_present = math.isfinite(value)
    qc_pass, qc_reason, qc_payload = qc_pass_for_observable(row, observable, prop)
    raw_u_stat, stat_breakdown = statistical_uncertainty(row, spec, raw_value)
    phase_payload = classify_phase_state(row, output_dir=phase_output_dir_for_ff_dir(ff_output_dir))
    u_ff = 0.0
    ff_qualified = False
    ff_profile_id = ""
    ff_breakdown: dict[str, Any] = {"ff_qualified": False, "ff_reason": "not_required"}
    value_ff_corrected = None
    if bool(spec.get("requires_ff")) or bool(spec.get("external_absolute")):
        u_ff, value_ff_corrected, ff_qualified, ff_profile_id, ff_breakdown = ff_adjustment(row, prop, raw_value, ff_output_dir=ff_output_dir)
    reported_value = value_ff_corrected if value_ff_corrected is not None else raw_value
    scale = abs(reported_value / raw_value) if math.isfinite(raw_value) and raw_value != 0 and math.isfinite(reported_value) else 1.0
    u_stat = raw_u_stat * scale
    u_engine, engine_qualified, engine_reason, engine_breakdown = engine_uncertainty(row, prop, raw_value, ff_output_dir=ff_output_dir)
    u_method, method_verified, method_breakdown = method_uncertainty(row, reported_value)
    u_similarity, similarity_used, similarity_breakdown = similarity_transfer_uncertainty(row, prop, reported_value)
    u_known = math.sqrt(sum(component * component for component in [u_stat, u_ff, u_engine, u_method, u_similarity]))
    tier, tier_reasons, check_payload = confidence_tier(
        qc_pass=qc_pass,
        ff_required=bool(spec.get("requires_ff")),
        ff_qualified=ff_qualified,
        engine_qualified=engine_qualified,
        method_verified=method_verified,
        external_absolute=bool(spec.get("external_absolute")),
        value_present=value_present,
        calibrated_absolute=bool(ff_qualified and value_ff_corrected is not None),
    )
    reference_phase = str(ff_breakdown.get("ff_reference_phase_family") or "no_reference_profile")
    tier, tier_reasons, phase_breakdown = apply_phase_confidence_gate(
        tier=tier,
        tier_reasons=tier_reasons,
        phase_payload=phase_payload,
        reference_phase=reference_phase,
        external_absolute=bool(spec.get("external_absolute")),
    )
    if similarity_used and not similarity_breakdown.get("similarity_transfer_high_confidence"):
        if "similarity_transfer_estimated_reference" not in tier_reasons:
            tier_reasons.append("similarity_transfer_estimated_reference")
        if tier == "HIGH":
            tier = "MED"
    if similarity_breakdown.get("similarity_transfer_review"):
        if "similarity_transfer_review" not in tier_reasons:
            tier_reasons.append("similarity_transfer_review")
        if tier == "HIGH":
            tier = "MED"
    if phase_breakdown.get("phase_mismatch_review") or phase_breakdown.get("phase_boundary_review"):
        if "phase_compatibility" not in check_payload["pending_checks"]:
            check_payload["pending_checks"].append("phase_compatibility")
            check_payload["automatic_check_actions"].append(AUTOMATIC_CHECK_ACTIONS["phase_compatibility"])
            tool = AUTOMATIC_CHECK_TOOLS["phase_compatibility"]
            if tool not in check_payload["automatic_check_tools"]:
                check_payload["automatic_check_tools"].append(tool)

    uncertainty_status = {
        "statistical_uncertainty": stat_breakdown.get("u_stat_source") != "not_available",
        "force_field_uncertainty": (
            not (bool(spec.get("requires_ff")) or bool(spec.get("external_absolute")))
            or bool(ff_qualified)
        ),
        "engine_uncertainty": engine_breakdown.get("u_engine_source") not in {
            "engine_unverified",
            "legacy_engine_untransferred_no_profile",
        },
        "method_uncertainty": method_breakdown.get("u_method_source") != "single_run_or_no_control",
        "similarity_uncertainty": (
            not similarity_used or similarity_breakdown.get("u_similarity_source") != "not_available"
        ),
    }
    known_uncertainty_components = [name for name, available in uncertainty_status.items() if available]
    unknown_uncertainty_components = [name for name, available in uncertainty_status.items() if not available]
    uncertainty_complete = not unknown_uncertainty_components
    if not uncertainty_complete:
        action = AUTOMATIC_CHECK_ACTIONS["statistical_uncertainty"]
        if "statistical_uncertainty" in unknown_uncertainty_components and action not in check_payload["automatic_check_actions"]:
            check_payload["automatic_check_actions"].append(action)
            tool = AUTOMATIC_CHECK_TOOLS["statistical_uncertainty"]
            if tool not in check_payload["automatic_check_tools"]:
                check_payload["automatic_check_tools"].append(tool)
    breakdown = {
        "schema_version": RESULT_CONFIDENCE_SCHEMA_VERSION,
        "observable": observable,
        "property": prop,
        "value_key": value_key,
        "confidence_scope": "external_absolute_calibrated" if bool(spec.get("external_absolute")) else "relative_structural_or_internal",
        "confidence_tier_basis": check_payload["tier_basis"],
        "evidence_checks": check_payload["checks"],
        "passed_checks": check_payload["passed_checks"],
        "failed_checks": check_payload["failed_checks"],
        "pending_checks": check_payload["pending_checks"],
        "automatic_check_actions": check_payload["automatic_check_actions"],
        "automatic_check_tools": check_payload["automatic_check_tools"],
        "uncertainty_complete": uncertainty_complete,
        "known_uncertainty_components": known_uncertainty_components,
        "unknown_uncertainty_components": unknown_uncertainty_components,
        "qc_pass": qc_pass,
        "qc_reason": qc_reason,
        "tier_reasons": tier_reasons,
        **stat_breakdown,
        **ff_breakdown,
        **engine_breakdown,
        **method_breakdown,
        **similarity_breakdown,
        **phase_breakdown,
    }
    review_flag = bool(
        similarity_breakdown.get("similarity_transfer_review")
        or phase_breakdown.get("phase_mismatch_review")
        or phase_breakdown.get("phase_boundary_review")
        or (not qc_pass)
    )
    return {
        "case_id": text(row.get("case_id")),
        "observable": observable,
        "property": prop,
        "value": reported_value if value_present else None,
        "reported_value": reported_value if value_present else None,
        "raw_value": raw_value if value_present else None,
        "value_ff_corrected": value_ff_corrected,
        "bias_correction_applied": value_ff_corrected is not None,
        "units": spec.get("units"),
        "u_stat": round(u_stat, 8),
        "u_ff": round(u_ff, 8),
        "u_engine": round(u_engine, 8),
        "u_method": round(u_method, 8),
        "u_similarity": round(u_similarity, 8),
        "u_known": round(u_known, 8),
        "u_total": round(u_known, 8) if uncertainty_complete else None,
        "uncertainty_complete": uncertainty_complete,
        "known_uncertainty_components_json": json.dumps(known_uncertainty_components, ensure_ascii=False),
        "unknown_uncertainty_components_json": json.dumps(unknown_uncertainty_components, ensure_ascii=False),
        "confidence_tier": tier,
        "confidence_tier_basis": check_payload["tier_basis"],
        "evidence_checks_json": json.dumps(check_payload["checks"], ensure_ascii=False, sort_keys=True),
        "pending_checks_json": json.dumps(check_payload["pending_checks"], ensure_ascii=False),
        "automatic_check_actions_json": json.dumps(check_payload["automatic_check_actions"], ensure_ascii=False),
        "automatic_check_tools_json": json.dumps(check_payload["automatic_check_tools"], ensure_ascii=False),
        "confidence_breakdown": breakdown,
        "confidence_breakdown_json": json.dumps(_json_safe(breakdown), ensure_ascii=False, sort_keys=True),
        "ff_qualified": ff_qualified,
        "ff_profile_id": ff_profile_id,
        "engine_transfer_qualified": engine_qualified,
        "method_verified": method_verified,
        "qc_pass": qc_pass,
        "engine_reason": engine_reason,
        "rule_basis": similarity_breakdown.get("rule_basis"),
        "similarity_reference_id": similarity_breakdown.get("similarity_reference_id"),
        "similarity_score": similarity_breakdown.get("similarity_score"),
        "similarity_score_interpretation": similarity_breakdown.get("similarity_score_interpretation"),
        "similarity_distance_reason": similarity_breakdown.get("similarity_distance_reason"),
        "review_flag": review_flag,
        "phase_state": phase_breakdown.get("phase_state"),
        "phase_mismatch_review": phase_breakdown.get("phase_mismatch_review"),
        "phase_boundary_review": phase_breakdown.get("phase_boundary_review"),
        "arrest_detected": phase_breakdown.get("arrest_detected"),
        "phase_rule_profile_id": phase_breakdown.get("phase_rule_profile_id"),
        "phase_g3_provenance": phase_breakdown.get("phase_g3_provenance"),
    }


def confidence_for_case(row: Mapping[str, Any], *, ff_output_dir: str | Path = DEFAULT_FF_OUTPUT_DIR) -> dict[str, Any]:
    records = [confidence_for_observable(row, observable, ff_output_dir=ff_output_dir) for observable in OBSERVABLES]
    return {
        "schema_version": RESULT_CONFIDENCE_SCHEMA_VERSION,
        "case_id": text(row.get("case_id")),
        "observables": {record["observable"]: record for record in records},
    }


def build_result_confidence_table(
    wide: pd.DataFrame,
    *,
    ff_output_dir: str | Path = DEFAULT_FF_OUTPUT_DIR,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in wide.iterrows():
        payload = confidence_for_case(row.to_dict(), ff_output_dir=ff_output_dir)
        records.extend(payload["observables"].values())
    return pd.DataFrame(records)


def confidence_wide_columns(confidence: pd.DataFrame) -> pd.DataFrame:
    if confidence.empty:
        return pd.DataFrame()
    keep = [
        "case_id",
        "observable",
        "raw_value",
        "reported_value",
        "value_ff_corrected",
        "bias_correction_applied",
        "u_similarity",
        "u_known",
        "u_total",
        "uncertainty_complete",
        "known_uncertainty_components_json",
        "unknown_uncertainty_components_json",
        "confidence_tier",
        "confidence_tier_basis",
        "evidence_checks_json",
        "pending_checks_json",
        "automatic_check_actions_json",
        "automatic_check_tools_json",
        "confidence_breakdown_json",
        "ff_qualified",
        "ff_profile_id",
        "engine_transfer_qualified",
        "method_verified",
        "qc_pass",
        "rule_basis",
        "similarity_reference_id",
        "similarity_score",
        "similarity_score_interpretation",
        "similarity_distance_reason",
        "review_flag",
        "phase_state",
        "phase_mismatch_review",
        "phase_boundary_review",
        "arrest_detected",
        "phase_rule_profile_id",
        "phase_g3_provenance",
    ]
    data = confidence[[col for col in keep if col in confidence.columns]].copy()
    wide = data.pivot(index="case_id", columns="observable")
    wide.columns = [f"confidence_{observable}_{field}" for field, observable in wide.columns]
    return wide.reset_index()


def write_result_confidence_outputs(
    wide: pd.DataFrame,
    *,
    output_csv: str | Path = DEFAULT_CONFIDENCE_CSV,
    ff_output_dir: str | Path = DEFAULT_FF_OUTPUT_DIR,
) -> dict[str, Any]:
    table = build_result_confidence_table(wide, ff_output_dir=ff_output_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "ok",
        "result_confidence_csv": str(output_path),
        "record_count": int(len(table)),
        "case_count": int(table["case_id"].nunique()) if not table.empty else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide-csv", default=str(DEFAULT_AGGREGATE_DIR / "integrated_summary.csv"))
    parser.add_argument("--output-csv", default=str(DEFAULT_CONFIDENCE_CSV))
    parser.add_argument("--ff-output-dir", default=str(DEFAULT_FF_OUTPUT_DIR))
    args = parser.parse_args(argv)
    wide = pd.read_csv(args.wide_csv, encoding="utf-8-sig")
    result = write_result_confidence_outputs(wide, output_csv=args.output_csv, ff_output_dir=args.ff_output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
