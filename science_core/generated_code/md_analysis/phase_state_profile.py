#!/usr/bin/env python3
"""Deterministic phase-state profiles for result confidence.

The profile is intentionally narrow.  It does not try to name every possible
polymer phase; it only prevents a fluid/melt reference profile from conferring
``HIGH`` confidence on a target whose clean trajectory shows local-order-
associated kinetic arrest.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rule_profile_registry import _hash_payload, _json_safe, _slug, _write_json


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PHASE_OUTPUT_DIR = SCRIPT_DIR / "output" / "phase_state_profiles"
PHASE_PROFILE_SCHEMA_VERSION = "phase_state_rule_profile.v1"
DEFAULT_PHASE_PROFILE_ID = "clean_msd_phase_state_v1"
FLUID_REFERENCE_STATES = {"liquid", "melt", "fluid"}
SUPPORTED_PIPELINE_STATES = {"", "melt"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def first_float(row: Mapping[str, Any], keys: list[str]) -> tuple[float, str]:
    for key in keys:
        value = to_float(row.get(key))
        if math.isfinite(value):
            return value, key
    return math.nan, ""


def build_default_phase_profile() -> dict[str, Any]:
    profile = {
        "schema_version": PHASE_PROFILE_SCHEMA_VERSION,
        "profile_kind": "phase_state_rule",
        "profile_id": DEFAULT_PHASE_PROFILE_ID,
        "scope": {
            "engine": "clean GROMACS 2024.4 preferred",
            "force_field": "any; rule uses observed clean dynamics",
            "observables": ["chain COM MSD", "Rg", "RDF first peak as co-symptom only"],
        },
        "hard_core_arrest_rule": {
            "requires_all": True,
            "alpha_100_300_max": 0.80,
            "alpha_late_max": 0.85,
            "msd_over_3rg2_max": 0.75,
            "interpretation": "local-order-associated kinetic arrest",
        },
        "diffusive_melt_rule": {
            "requires_all": True,
            "alpha_min": 0.95,
            "alpha_max": 1.05,
            "msd_over_rg2_min": 3.0,
            "interpretation": "diffusive melt or liquid-like trajectory",
        },
        "boundary_policy": {
            "confidence_cap": "MED",
            "reason": "boundary_or_partial_order_requires_review",
        },
        "rdf_policy": {
            "role": "co_symptom_only",
            "note": "RDF first-peak elevation is recorded but never used alone to assign phase.",
        },
        "threshold_source": "PE-LCB100 clean-2024.4 hardening note, 2026-06-14",
        "activation_status": "eligible_for_machine_freeze",  # legacy schema value
        "activation_readiness": "eligible_for_activation",
        "created_at": utc_now(),
        "frozen_at": None,
        "frozen_by": None,
        "freeze_reason": "",
    }
    profile["profile_hash"] = _hash_payload(profile)
    return profile


def _profile_path(output_dir: Path, profile: Mapping[str, Any]) -> Path:
    return output_dir / "profiles" / "phase_state_rule" / f"{profile['profile_id']}.json"


def _ledger_path(output_dir: Path) -> Path:
    return output_dir / "phase_state_freeze_ledger.jsonl"


def activate_phase_profile(
    profile: Mapping[str, Any] | None = None,
    *,
    output_dir: str | Path = DEFAULT_PHASE_OUTPUT_DIR,
    actor: str = "codex",
    reason: str = "activate deterministic phase-state confidence gate",
) -> dict[str, Any]:
    prof = dict(profile or build_default_phase_profile())
    if prof.get("activation_status") not in {"eligible_for_machine_freeze", "active"}:
        return {
            "status": "rejected",
            "profile_id": prof.get("profile_id"),
            "errors": [f"profile is not eligible for freeze: {prof.get('activation_status')}"],
        }
    frozen = dict(prof)
    frozen["activation_status"] = "active"
    frozen["activation_readiness"] = "active"
    frozen["activated_at"] = utc_now()
    frozen["activated_by"] = "machine_activated_by_codex" if _slug(actor) == "codex" else str(actor)
    frozen["activation_reason"] = reason
    frozen["frozen_at"] = frozen["activated_at"]
    frozen["frozen_by"] = frozen["activated_by"]
    frozen["freeze_reason"] = reason
    frozen["profile_hash"] = _hash_payload({k: v for k, v in frozen.items() if k != "profile_hash"})

    out_dir = Path(output_dir)
    path = _profile_path(out_dir, frozen)
    _write_json(path, frozen)
    ledger = {
        "timestamp_utc": utc_now(),
        "event": "activate_phase_profile",
        "status": "ok",
        "profile_id": frozen.get("profile_id"),
        "actor": actor,
        "reason": reason,
        "profile_path": str(path),
        "profile_hash": frozen.get("profile_hash"),
    }
    with _ledger_path(out_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(ledger), ensure_ascii=False, sort_keys=True) + "\n")
    return {"status": "ok", "profile_id": frozen.get("profile_id"), "affected_path": str(path)}


def find_active_phase_profile(*, output_dir: str | Path = DEFAULT_PHASE_OUTPUT_DIR) -> dict[str, Any] | None:
    out_dir = Path(output_dir)
    profile_dir = out_dir / "profiles" / "phase_state_rule"
    if not profile_dir.exists():
        return None
    for path in sorted(profile_dir.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
        if profile.get("activation_status") == "active":
            return profile
    return None


def ensure_default_phase_profile(*, output_dir: str | Path = DEFAULT_PHASE_OUTPUT_DIR) -> dict[str, Any]:
    active = find_active_phase_profile(output_dir=output_dir)
    if active:
        return active
    activation = activate_phase_profile(output_dir=output_dir)
    if activation.get("status") == "ok":
        return json.loads(Path(activation["affected_path"]).read_text(encoding="utf-8-sig"))
    profile = build_default_phase_profile()
    profile["activation_status"] = "not_frozen_fallback"
    return profile


def reference_phase_family(profile: Mapping[str, Any] | None) -> str:
    if not profile:
        return "no_reference_profile"
    states = {
        text(item.get("state")).lower()
        for item in profile.get("references", []) or []
        if text(item.get("state"))
    }
    if not states:
        return "unknown_reference_phase"
    if states <= FLUID_REFERENCE_STATES:
        return "fluid_reference"
    if states & FLUID_REFERENCE_STATES:
        return "mixed_reference_phase"
    return "nonfluid_reference"


def classify_phase_state(
    row: Mapping[str, Any],
    *,
    profile: Mapping[str, Any] | None = None,
    output_dir: str | Path = DEFAULT_PHASE_OUTPUT_DIR,
) -> dict[str, Any]:
    prof = dict(profile or ensure_default_phase_profile(output_dir=output_dir))
    # Phase must be read from a FIXED-window chain-COM slope.  ``best_loglog_slope``
    # is the publication gate's *searched* best window and is biased toward ~1.0
    # (it can find a diffusive-looking window even inside an arrested trajectory),
    # so it is intentionally excluded here.
    alpha_main, alpha_main_key = first_float(
        row,
        [
            "phase_alpha_100_300",
            "alpha_100_300",
            "g3_alpha_100_300",
            "loglog_slope",
        ],
    )
    alpha_late, alpha_late_key = first_float(
        row,
        [
            "phase_alpha_200_340",
            "alpha_200_340",
            "alpha_200_400",
            "late_loglog_slope",
        ],
    )
    # The thresholds (hard-core < 0.75, melt >= 3.0) are expressed in MSD/(3*Rg^2).
    # Some sources report MSD/Rg^2 instead (e.g. ``window_msd_over_rg2_min`` follows
    # the publication-gate convention where the melt floor is MSD/Rg^2 >= 3).  Those
    # are 3x larger, so they MUST be divided by 3 before comparison; otherwise the
    # strongest arrest cases silently fail the hard-core rule and escape the gate.
    msd_ratio, msd_ratio_key = first_float(
        row,
        ["msd_over_3rg2_final", "msd_over_3rg2", "window_msd_over_3rg2_min"],
    )
    msd_ratio_unit = "msd_over_3rg2"
    if not math.isfinite(msd_ratio):
        raw_over_rg2, msd_ratio_key = first_float(
            row,
            ["msd_over_rg2_final", "window_msd_over_rg2_min", "best_window_msd_over_rg2_min", "msd_over_rg2"],
        )
        if math.isfinite(raw_over_rg2):
            msd_ratio = raw_over_rg2 / 3.0
            msd_ratio_unit = "msd_over_rg2_div3"
    rdf_peak, rdf_peak_key = first_float(
        row,
        ["rdf_inter_chain_cc_first_peak_g", "rdf_all_first_peak_g", "peak1", "rdfg"],
    )
    target_state = text(row.get("state") or row.get("target_state") or row.get("requested_state") or "melt").lower()
    hard_rule = prof.get("hard_core_arrest_rule") or {}
    melt_rule = prof.get("diffusive_melt_rule") or {}

    # Arrest is engine-dependent: the same case diffuses under legacy 2021.2 but
    # arrests under clean 2024.4.  Phase must be read from clean-engine dynamics; a
    # legacy-engine slope (alpha~=1) must not be allowed to assert "diffusive_melt".
    engine = text(row.get("engine_version") or row.get("engine_scope") or row.get("msd_engine")).lower()
    nonclean_engine = ("2021.2" in engine) and ("2024.4" not in engine) and ("clean" not in engine)
    g3_provenance = text(
        row.get("intrachain_g3_provenance")
        or row.get("phase_g3_provenance")
        or row.get("g3_provenance")
    )

    reasons: list[str] = []
    hard_core = (
        math.isfinite(alpha_main)
        and math.isfinite(alpha_late)
        and math.isfinite(msd_ratio)
        and alpha_main < float(hard_rule.get("alpha_100_300_max", 0.80))
        and alpha_late < float(hard_rule.get("alpha_late_max", 0.85))
        and msd_ratio < float(hard_rule.get("msd_over_3rg2_max", 0.75))
    )
    melt = (
        math.isfinite(alpha_main)
        and math.isfinite(msd_ratio)
        and float(melt_rule.get("alpha_min", 0.95)) <= alpha_main <= float(melt_rule.get("alpha_max", 1.05))
        and msd_ratio >= float(melt_rule.get("msd_over_rg2_min", 3.0))
    )

    if target_state not in SUPPORTED_PIPELINE_STATES:
        phase_state = "unsupported_target_state"
        reasons.append("phase_profile_supports_melt_only")
    elif nonclean_engine:
        phase_state = "phase_engine_provenance_unverified"
        reasons.append("phase_dynamics_from_nonclean_engine")
    elif hard_core:
        phase_state = "hard_core_arrest"
        reasons.append("hard_core_arrest_rule_matched")
    elif melt:
        phase_state = "diffusive_melt"
        reasons.append("diffusive_melt_rule_matched")
    elif not math.isfinite(alpha_main) or not math.isfinite(msd_ratio):
        phase_state = "phase_evidence_incomplete"
        reasons.append("phase_evidence_incomplete")
    else:
        phase_state = "boundary_or_partial_order"
        reasons.append("boundary_or_partial_order")

    if math.isfinite(rdf_peak):
        reasons.append("rdf_peak_recorded_as_co_symptom_only")
    if g3_provenance:
        if "backfill" in g3_provenance.lower():
            reasons.append("phase_g3_from_clean_com_msd_backfill")
        elif "native" in g3_provenance.lower():
            reasons.append("phase_g3_from_native_intrachain_analysis")

    return {
        "phase_state": phase_state,
        "phase_rule_profile_id": prof.get("profile_id"),
        "phase_rule_profile_hash": prof.get("profile_hash"),
        "phase_evidence_complete": phase_state
        not in {"phase_evidence_incomplete", "phase_engine_provenance_unverified", "unsupported_target_state"},
        "arrest_detected": phase_state == "hard_core_arrest",
        "phase_boundary": phase_state == "boundary_or_partial_order",
        "phase_engine_provenance_ok": not nonclean_engine,
        "phase_engine": engine or None,
        "phase_g3_provenance": g3_provenance or None,
        "phase_reasons": reasons,
        "alpha_main": alpha_main if math.isfinite(alpha_main) else None,
        "alpha_main_key": alpha_main_key,
        "alpha_late": alpha_late if math.isfinite(alpha_late) else None,
        "alpha_late_key": alpha_late_key,
        "msd_ratio": msd_ratio if math.isfinite(msd_ratio) else None,
        "msd_ratio_key": msd_ratio_key,
        "msd_ratio_unit": msd_ratio_unit,
        "rdf_peak": rdf_peak if math.isfinite(rdf_peak) else None,
        "rdf_peak_key": rdf_peak_key,
        "target_state": target_state,
    }


def phase_compatibility(target_phase: str, reference_family: str) -> dict[str, Any]:
    if target_phase == "unsupported_target_state":
        return {"status": "unsupported", "confidence_cap": "MED", "reason": "melt_only_profile_boundary"}
    if reference_family == "fluid_reference":
        if target_phase == "diffusive_melt":
            return {"status": "compatible", "confidence_cap": None, "reason": "target_matches_fluid_reference_phase"}
        if target_phase == "hard_core_arrest":
            return {"status": "mismatch", "confidence_cap": "MED", "reason": "phase_mismatch_review"}
        if target_phase == "boundary_or_partial_order":
            return {"status": "boundary_review", "confidence_cap": "MED", "reason": "phase_boundary_review"}
        return {"status": "unknown_review", "confidence_cap": "MED", "reason": "phase_evidence_incomplete_review"}
    if reference_family in {"unknown_reference_phase", "mixed_reference_phase", "no_reference_profile"}:
        return {"status": "unknown_review", "confidence_cap": "MED", "reason": "reference_phase_unverified"}
    return {"status": "review", "confidence_cap": "MED", "reason": "nonfluid_reference_requires_review"}


__all__ = [
    "DEFAULT_PHASE_OUTPUT_DIR",
    "PHASE_PROFILE_SCHEMA_VERSION",
    "activate_phase_profile",
    "build_default_phase_profile",
    "classify_phase_state",
    "ensure_default_phase_profile",
    "find_active_phase_profile",
    "phase_compatibility",
    "reference_phase_family",
]
