#!/usr/bin/env python3
"""Transferable force-field and engine qualification profiles.

These profiles do not assert that a new material has an external acceptance
threshold.  They quantify systematic force-field and engine/protocol
uncertainty from reference substances or same-input controls, then expose that
budget to result-confidence reporting.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from rule_profile_registry import (
    DEFAULT_RULE_PROFILE_DIR,
    HIGH_TRUST_SOURCE_TYPES,
    _confidence_score,
    _hash_payload,
    _json_safe,
    _slug,
    _source_trust_score,
    _write_json,
    canonical_property_name,
    normalize_property_value,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH = SCRIPT_DIR / "plugins" / "force_field_reference_substances.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "force_field_qualification"
FF_QUALIFICATION_SCHEMA_VERSION = "force_field_qualification_profile.v1"
ENGINE_TRANSFER_SCHEMA_VERSION = "engine_transfer_profile.v1"
MIN_REFERENCE_COUNT = 2
MIN_CONFIDENCE_SCORE = 0.65
QUALIFICATION_HIGH_TRUST_SOURCE_TYPES = set(HIGH_TRUST_SOURCE_TYPES) | {"si_p_control"}
PHASE_VALIDATED_REFERENCE_STATUSES = {
    "verified_same_phase",
    "verified_fluid",
    "verified_melt",
    "phase_validated",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_reference_registry(path: str | Path = DEFAULT_REFERENCE_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"schema_version": "force_field_reference_substances.v1", "references": []}
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _qualification_source_trust_score(source_type: str, confidence: float) -> float:
    if _slug(source_type) == "si_p_control":
        return _source_trust_score("internal_clean_control", confidence)
    return _source_trust_score(source_type, confidence)


def _reference_matches(
    item: Mapping[str, Any],
    *,
    polymer_family: str,
    force_field: str,
    property_name: str,
    engine_scope: str | None,
) -> bool:
    profile_kind = str(item.get("profile_kind") or "force_field").lower()
    if profile_kind not in {"force_field", "force_field_reference"}:
        return False
    prop = canonical_property_name(item.get("property"))
    if prop != canonical_property_name(property_name):
        return False
    if _slug(item.get("force_field")) != _slug(force_field):
        return False
    families = {
        _slug(item.get("polymer_family")),
        _slug(item.get("chemical_family")),
    }
    target_family = _slug(polymer_family)
    if target_family not in families and not (target_family.startswith("pe") and "pe" in families):
        return False
    if engine_scope and item.get("engine_scope") and _slug(item.get("engine_scope")) != _slug(engine_scope):
        return False
    return True


def _engine_transfer_matches(
    item: Mapping[str, Any],
    *,
    polymer_family: str,
    force_field: str,
    property_name: str,
    source_engine_scope: str | None,
    target_engine_scope: str | None,
) -> bool:
    if str(item.get("profile_kind") or "").lower() != "engine_transfer":
        return False
    if canonical_property_name(item.get("property")) != canonical_property_name(property_name):
        return False
    if _slug(item.get("force_field")) != _slug(force_field):
        return False
    families = {_slug(item.get("polymer_family")), _slug(item.get("chemical_family"))}
    target_family = _slug(polymer_family)
    if target_family not in families and not (target_family.startswith("pe") and "pe" in families):
        return False
    if source_engine_scope and item.get("source_engine_scope") and _slug(item.get("source_engine_scope")) != _slug(source_engine_scope):
        return False
    if target_engine_scope and item.get("target_engine_scope") and _slug(item.get("target_engine_scope")) != _slug(target_engine_scope):
        return False
    return True


def _normalise_reference(item: Mapping[str, Any], property_name: str) -> dict[str, Any] | None:
    prop = canonical_property_name(property_name)
    ref_value = normalize_property_value(item.get("ref_value"), str(item.get("ref_units") or item.get("units") or ""), prop)
    md_value = normalize_property_value(item.get("md_value"), str(item.get("md_units") or item.get("units") or ""), prop)
    if ref_value is None or md_value is None or ref_value == 0:
        return None
    offset = (float(md_value) - float(ref_value)) / float(ref_value)
    source_type = _slug(item.get("source_type") or "literature")
    extraction_confidence = max(0.0, min(1.0, _as_float(item.get("extraction_confidence") or 0.5)))
    return {
        "reference_id": str(item.get("reference_id") or item.get("source_id") or _hash_payload(item)[:12]),
        "substance": str(item.get("substance") or ""),
        "property": prop,
        "ref_value": float(ref_value),
        "ref_units": item.get("ref_units") or item.get("units"),
        "md_value": float(md_value),
        "md_units": item.get("md_units") or item.get("units"),
        "relative_offset": round(offset, 6),
        "source_type": source_type,
        "source_id": str(item.get("source_id") or item.get("reference_id") or ""),
        "independent_key": str(item.get("independent_key") or item.get("source_id") or item.get("reference_id") or ""),
        "extraction_confidence": round(extraction_confidence, 3),
        "source_trust_score": _qualification_source_trust_score(source_type, extraction_confidence),
        "temperature_K": item.get("temperature_K"),
        "pressure_bar": item.get("pressure_bar"),
        "state": item.get("state"),
        "phase_qualification_status": str(item.get("phase_qualification_status") or "unverified_phase"),
        "reference_role": str(item.get("reference_role") or ""),
        "phase_validation_note": item.get("phase_validation_note") or item.get("note"),
        "force_field": item.get("force_field"),
        "engine_scope": item.get("engine_scope"),
        "md_source_type": item.get("md_source_type"),
        "md_source_id": item.get("md_source_id"),
        "profile_kind": str(item.get("profile_kind") or "force_field"),
    }


def propose_ff_qualification_profile(
    context: Mapping[str, Any],
    references: Iterable[Mapping[str, Any]],
    *,
    min_reference_count: int = MIN_REFERENCE_COUNT,
) -> dict[str, Any]:
    prop = canonical_property_name(context.get("property") or "density")
    family = str(context.get("polymer_family") or "")
    force_field = str(context.get("force_field") or "")
    engine_scope = str(context.get("engine_scope") or context.get("target_engine_scope") or "") or None
    matched = [
        _normalise_reference(item, prop)
        for item in references
        if _reference_matches(item, polymer_family=family, force_field=force_field, property_name=prop, engine_scope=engine_scope)
    ]
    normalized = [item for item in matched if item is not None]
    offsets = [float(item["relative_offset"]) for item in normalized]
    mean_offset = sum(offsets) / len(offsets) if offsets else math.nan
    spread = (max(offsets) - min(offsets)) if offsets else math.inf
    uncertainty = _std(offsets) if offsets else math.nan
    confidence = _confidence_score(normalized, spread if math.isfinite(spread) else 1.0, 0.05)
    independent_count = len({item.get("independent_key") for item in normalized})
    high_trust_count = sum(
        1
        for item in normalized
        if str(item.get("source_type")) in QUALIFICATION_HIGH_TRUST_SOURCE_TYPES or float(item.get("source_trust_score", 0.0)) >= 0.82
    )
    phase_validated_count = sum(
        1
        for item in normalized
        if str(item.get("phase_qualification_status") or "").lower() in PHASE_VALIDATED_REFERENCE_STATUSES
    )
    phase_unverified_refs = [
        item.get("reference_id")
        for item in normalized
        if str(item.get("phase_qualification_status") or "").lower() not in PHASE_VALIDATED_REFERENCE_STATUSES
    ]

    activation_status = "eligible_for_machine_freeze"
    if len(normalized) < min_reference_count or independent_count < min_reference_count:
        activation_status = "blocked_insufficient_references"
    elif phase_validated_count < len(normalized):
        activation_status = "blocked_phase_unverified_references"
    elif high_trust_count < 1:
        activation_status = "blocked_low_trust_references"
    elif confidence < MIN_CONFIDENCE_SCORE:
        activation_status = "blocked_low_confidence"
    phase_validation_status = "phase_validated" if normalized and phase_validated_count == len(normalized) else "phase_unverified"

    profile = {
        "schema_version": FF_QUALIFICATION_SCHEMA_VERSION,
        "profile_kind": "force_field_qualification",
        "profile_id": "_".join([_slug(family), _slug(force_field), _slug(engine_scope or "any"), _slug(prop), "ff_qualification"]),
        "polymer_family": family,
        "force_field": force_field,
        "engine_scope": engine_scope or "any",
        "property": prop,
        "systematic_offset": round(mean_offset, 6) if math.isfinite(mean_offset) else None,
        "systematic_uncertainty": round(uncertainty, 6) if math.isfinite(uncertainty) else None,
        "n_references": len(normalized),
        "independent_reference_count": independent_count,
        "phase_validated_reference_count": phase_validated_count,
        "phase_validation_status": phase_validation_status,
        "phase_unverified_reference_ids": phase_unverified_refs,
        "confidence_score": confidence,
        "references": normalized,
        "activation_status": activation_status,
        "activation_checks": {
            "minimum_reference_count_met": len(normalized) >= min_reference_count,
            "independent_reference_count_met": independent_count >= min_reference_count,
            "phase_validation_met": phase_validated_count == len(normalized) and bool(normalized),
            "phase_unverified_references": phase_unverified_refs,
            "high_trust_reference_present": high_trust_count >= 1,
            "confidence_threshold_met": confidence >= MIN_CONFIDENCE_SCORE,
            "engine_offset_kept_separate": True,
        },
        "created_at": utc_now(),
        "frozen_at": None,
        "frozen_by": None,
        "freeze_reason": "",
    }
    profile["profile_hash"] = _hash_payload(profile)
    return profile


def propose_engine_transfer_profile(
    context: Mapping[str, Any],
    references: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    prop = canonical_property_name(context.get("property") or "density")
    family = str(context.get("polymer_family") or "")
    force_field = str(context.get("force_field") or "")
    source_engine = str(context.get("source_engine_scope") or "GROMACS2021.2-production")
    target_engine = str(context.get("target_engine_scope") or context.get("engine_scope") or "GROMACS2024.4-clean")
    matched = [
        dict(item)
        for item in references
        if _engine_transfer_matches(
            item,
            polymer_family=family,
            force_field=force_field,
            property_name=prop,
            source_engine_scope=source_engine,
            target_engine_scope=target_engine,
        )
    ]
    factors: list[float] = []
    weights: list[float] = []
    evidence: list[dict[str, Any]] = []
    for item in matched:
        factor = _as_float(item.get("transfer_factor"))
        source_value = _as_float(item.get("source_value"))
        target_value = _as_float(item.get("target_value"))
        if not math.isfinite(factor) and math.isfinite(source_value) and math.isfinite(target_value) and target_value != 0:
            factor = source_value / target_value
        if not math.isfinite(factor):
            continue
        weight = _as_float(item.get("n_pairs") or item.get("weight") or 1.0)
        if not math.isfinite(weight) or weight <= 0:
            weight = 1.0
        source_type = _slug(item.get("source_type") or "si_p_control")
        extraction_confidence = max(0.0, min(1.0, _as_float(item.get("extraction_confidence") or 0.5)))
        factors.append(float(factor))
        weights.append(float(weight))
        evidence.append(
            {
                "reference_id": str(item.get("reference_id") or item.get("source_id") or ""),
                "substance": str(item.get("substance") or ""),
                "transfer_factor": round(float(factor), 6),
                "source_value": source_value if math.isfinite(source_value) else None,
                "target_value": target_value if math.isfinite(target_value) else None,
                "units": item.get("units"),
                "source_type": source_type,
                "source_id": str(item.get("source_id") or item.get("reference_id") or ""),
                "independent_key": str(item.get("independent_key") or item.get("source_id") or item.get("reference_id") or ""),
                "source_trust_score": _qualification_source_trust_score(source_type, extraction_confidence),
                "extraction_confidence": round(extraction_confidence, 3),
                "weight": round(float(weight), 6),
                "n_pairs": item.get("n_pairs"),
            }
        )
    total_weight = sum(weights)
    mean_factor = sum(factor * weight for factor, weight in zip(factors, weights)) / total_weight if factors and total_weight > 0 else math.nan
    expanded_factors: list[float] = []
    for factor, weight in zip(factors, weights):
        expanded_factors.extend([factor] * max(1, int(round(weight))))
    uncertainty = _std(expanded_factors) if expanded_factors else math.nan
    spread = max(factors) - min(factors) if factors else math.inf
    confidence = _confidence_score(evidence, spread if math.isfinite(spread) else 1.0, 0.08)
    activation_status = "eligible_for_machine_freeze" if evidence else "blocked_insufficient_references"
    profile = {
        "schema_version": ENGINE_TRANSFER_SCHEMA_VERSION,
        "profile_kind": "engine_transfer",
        "profile_id": "_".join([_slug(family), _slug(force_field), _slug(source_engine), "to", _slug(target_engine), _slug(prop), "engine_transfer"]),
        "polymer_family": family,
        "force_field": force_field,
        "property": prop,
        "source_engine_scope": source_engine,
        "target_engine_scope": target_engine,
        "transfer_factor": round(mean_factor, 6) if math.isfinite(mean_factor) else None,
        "transfer_uncertainty": round(uncertainty, 6) if math.isfinite(uncertainty) else None,
        "n_references": len(evidence),
        "total_weight": round(total_weight, 6) if factors else 0,
        "confidence_score": confidence,
        "references": evidence,
        "activation_status": activation_status,
        "created_at": utc_now(),
        "frozen_at": None,
        "frozen_by": None,
        "freeze_reason": "",
    }
    profile["profile_hash"] = _hash_payload(profile)
    return profile


def _profile_path(output_dir: Path, profile: Mapping[str, Any]) -> Path:
    kind = str(profile.get("profile_kind") or "force_field_qualification")
    return output_dir / "profiles" / kind / f"{profile['profile_id']}.json"


def _ledger_path(output_dir: Path) -> Path:
    return output_dir / "ff_qualification_freeze_ledger.jsonl"


def activate_qualification_profile(
    profile: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    actor: str = "codex",
    reason: str = "",
) -> dict[str, Any]:
    prof = dict(profile)
    errors: list[str] = []
    if prof.get("activation_status") != "eligible_for_machine_freeze":
        errors.append(f"profile is not eligible for machine freeze: {prof.get('activation_status')}")
    if float(prof.get("confidence_score") or 0.0) < MIN_CONFIDENCE_SCORE:
        errors.append("confidence_score is below activation threshold 0.65")
    if prof.get("profile_kind") == "force_field_qualification" and int(prof.get("n_references") or 0) < MIN_REFERENCE_COUNT:
        errors.append("force-field qualification requires at least two references")
    if prof.get("profile_kind") == "force_field_qualification" and prof.get("phase_validation_status") != "phase_validated":
        errors.append("force-field qualification references must be phase-validated before activation")
    if errors:
        return {"status": "rejected", "errors": errors, "profile_id": prof.get("profile_id")}

    out_dir = Path(output_dir)
    frozen = dict(prof)
    frozen["frozen_at"] = utc_now()
    frozen["frozen_by"] = "machine_frozen_by_codex" if _slug(actor) == "codex" else str(actor)
    frozen["freeze_reason"] = reason
    frozen["activation_status"] = "active"
    frozen["profile_hash"] = _hash_payload({k: v for k, v in frozen.items() if k != "profile_hash"})
    path = _profile_path(out_dir, frozen)
    _write_json(path, frozen)
    ledger = {
        "timestamp_utc": utc_now(),
        "event": "activate_qualification_profile",
        "status": "ok",
        "profile_id": frozen.get("profile_id"),
        "profile_kind": frozen.get("profile_kind"),
        "actor": actor,
        "reason": reason,
        "profile_path": str(path),
        "profile_hash": frozen.get("profile_hash"),
    }
    with _ledger_path(out_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(ledger), ensure_ascii=False, sort_keys=True) + "\n")
    return {"status": "ok", "profile_id": frozen.get("profile_id"), "affected_path": str(path)}


def find_active_qualification_profile(
    context: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    profile_kind: str = "force_field_qualification",
) -> dict[str, Any] | None:
    out_dir = Path(output_dir)
    prop = canonical_property_name(context.get("property") or "")
    family = _slug(context.get("polymer_family") or "")
    force_field = _slug(context.get("force_field") or "")
    candidates = sorted((out_dir / "profiles" / profile_kind).glob("*.json")) if (out_dir / "profiles" / profile_kind).exists() else []
    for path in candidates:
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
        if profile.get("activation_status") != "active":
            continue
        if canonical_property_name(profile.get("property")) != prop:
            continue
        if _slug(profile.get("force_field")) != force_field:
            continue
        profile_family = _slug(profile.get("polymer_family"))
        if profile_family == family or (family.startswith("pe") and profile_family in {"pe", "pe_lcb100"}):
            return profile
    return None


def build_profiles_from_registry(
    *,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    context: Mapping[str, Any] | None = None,
    execute: bool = False,
    actor: str = "codex",
    reason: str = "build force-field qualification profiles",
) -> dict[str, Any]:
    registry = read_reference_registry(reference_path)
    references = registry.get("references", []) or []
    ctx = {
        "polymer_family": "PE",
        "force_field": "opls-aa",
        "engine_scope": "GROMACS2024.4-clean",
        "property": "density",
    }
    if context:
        ctx.update(dict(context))
    ff_profile = propose_ff_qualification_profile(ctx, references)
    engine_profile = propose_engine_transfer_profile(
        {
            **ctx,
            "polymer_family": "PE-LCB100",
            "source_engine_scope": "GROMACS2021.2-production",
            "target_engine_scope": "GROMACS2024.4-clean",
        },
        references,
    )
    activations = []
    if execute:
        for profile in [ff_profile, engine_profile]:
            activations.append(activate_qualification_profile(profile, output_dir=output_dir, actor=actor, reason=reason))
    return {
        "status": "ok",
        "execute": execute,
        "profiles": [ff_profile, engine_profile],
        "activations": activations,
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reason", default="build force-field qualification profiles")
    args = parser.parse_args(argv)
    result = build_profiles_from_registry(
        reference_path=args.reference_path,
        output_dir=args.output_dir,
        execute=args.execute,
        reason=args.reason,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
