#!/usr/bin/env python3
"""Deterministic similar-structure transfer for missing rule profiles.

This module is deliberately model-free.  It does not set thresholds from a
language-model judgement.  It only compares a target polymer/property context
with structured reference records and returns a candidate transfer packet when
the structural, condition, phase and source constraints are explicit enough.

The PE-LCB100 density rule uses the dedicated full-chain matcher in
``radonpy_density_screening.py``. It does not use this module's composite score.
"""

from __future__ import annotations

import math
from typing import Any

from rule_profile_registry import canonical_engine_scope, canonical_force_field, canonical_property_name

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    _RDKIT_AVAILABLE = True
except (ImportError, OSError):
    Chem = None
    DataStructs = None
    rdFingerprintGenerator = None
    _RDKIT_AVAILABLE = False


SIMILARITY_SCHEMA_VERSION = "similarity_transfer_candidate.v1"
DISTANCE_METRIC_VERSION = "weighted_polymer_similarity.v4"
DEFAULT_MIN_SIMILARITY = 0.80
DEFAULT_HIGH_SIMILARITY = 0.90
DEFAULT_TEMPERATURE_TOLERANCE_K = 25.0
DEFAULT_PRESSURE_TOLERANCE_BAR = 1.0
DEFAULT_TRANSFER_UNCERTAINTY_FLOOR_REL = 0.12
DEFAULT_TRANSFER_UNCERTAINTY_SLOPE_REL = 0.30
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
CONDITION_ANY = {"", "any", "unspecified", "unknown", "not_specified"}
MACHINE_ACTIVATION_POLICY_VERSION = "similarity_transfer_activation.v4"
# Backward-compatible exported alias.
MACHINE_FREEZE_POLICY_VERSION = MACHINE_ACTIVATION_POLICY_VERSION
COMPUTATIONAL_REFERENCE_TYPES = {
    "radonpy",
    "htpmd",
}
CONTROL_REFERENCE_TYPES = {
    "clean_engine_control",
    "internal_clean_control",
    "same_input_clean_control",
}
SEQUENCE_DESCRIPTOR_KEYS = (
    "max_contiguous_carbonate_units",
    "average_segment_length",
    "sequence_block_length",
    "sequence_spacing",
    "branch_spacing",
    "sequence_pattern",
    "placement_code",
    "Lc",
    "lc",
)
CHAIN_SMILES_KEYS = ("canonical_chain_smiles", "chain_smiles", "polymer_chain_smiles", "oligomer_smiles")
SEGMENT_SMILES_KEYS = ("canonical_segment_smiles", "normalized_segment_smiles", "surrogate_dp_smiles")
REPEAT_UNIT_SMILES_KEYS = ("repeat_unit_smiles", "canonical_repeat_unit_smiles", "poly_smiles", "smiles")
MONOMER_SMILES_KEYS = ("monomer_smiles", "mol_smiles")
SMILES_LEVELS = (
    ("chain_smiles", 4, CHAIN_SMILES_KEYS),
    ("normalized_segment_smiles", 3, SEGMENT_SMILES_KEYS),
    ("repeat_unit_plus_dp", 2, REPEAT_UNIT_SMILES_KEYS),
    ("monomer_plus_dp", 1, MONOMER_SMILES_KEYS),
)
ALL_SMILES_KEYS = tuple(dict.fromkeys(CHAIN_SMILES_KEYS + SEGMENT_SMILES_KEYS + REPEAT_UNIT_SMILES_KEYS + MONOMER_SMILES_KEYS))
CHAIN_SIZE_KEYS = (
    "degree_of_polymerization",
    "polymerization_degree",
    "DP",
    "dp",
    "chain_length",
    "target_total_carbons_per_chain",
    "molecular_weight",
    "Mn",
    "mn",
    "Mw",
    "mw",
)
UNIT_GROUPS = {
    "density": {"g_cm3", "g_cc", "g_ml", "kg_m3"},
    "diffusivity": {"m2_s", "m_2_s", "m2_per_s"},
    "thermal_conductivity": {"w_m_k", "w_mk", "w_per_m_k"},
    "dimensionless": {"dimensionless", "none", "unitless"},
    "length": {"nm", "angstrom", "a"},
    "area": {"nm2", "nm_2", "angstrom2", "a2"},
}


def _slug(value: Any) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _tokens(value: Any) -> set[str]:
    slug = _slug(value)
    if not slug:
        return set()
    return {item for item in slug.split("_") if item and item not in {"poly", "polymer", "candidate", "new"}}


def _token_set_from_values(values: list[Any] | tuple[Any, ...] | set[Any]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens |= _tokens(value)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float | None:
    if not left or not right:
        return None
    return len(left & right) / len(left | right)


def _first_available(payload: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, str | None]:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value), key
    return None, None


def _select_smiles_pair(
    target: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None, str, int]:
    for level, rank, keys in SMILES_LEVELS:
        target_smiles, target_field = _first_available(target, keys)
        reference_smiles, reference_field = _first_available(reference, keys)
        if target_smiles and reference_smiles:
            return target_smiles, reference_smiles, target_field, reference_field, level, rank
    return None, None, None, None, "token_fallback", 0


def _smiles_structure_similarity(
    target: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Use the highest shared chain/segment/repeat/monomer SMILES level."""

    target_smiles, reference_smiles, target_field, reference_field, level, rank = _select_smiles_pair(target, reference)
    fallback_reason = None

    if _RDKIT_AVAILABLE and target_smiles and reference_smiles:
        target_mol = Chem.MolFromSmiles(target_smiles) if Chem is not None else None
        reference_mol = Chem.MolFromSmiles(reference_smiles) if Chem is not None else None
        if target_mol is not None and reference_mol is not None:
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)
            target_fp = generator.GetFingerprint(target_mol)
            reference_fp = generator.GetFingerprint(reference_mol)
            score = float(DataStructs.TanimotoSimilarity(target_fp, reference_fp))
            return score, {
                "method": "morgan_tanimoto",
                "fingerprint": "Morgan/ECFP",
                "structure_comparison_level": level,
                "structure_evidence_rank": rank,
                "radius": MORGAN_RADIUS,
                "n_bits": MORGAN_NBITS,
                "target_smiles_field": target_field,
                "reference_smiles_field": reference_field,
                "smiles_fields_used": {
                    "target": target_field,
                    "reference": reference_field,
                },
                "rdkit_available": True,
                "fallback_reason": None,
            }
        if target_mol is None and reference_mol is None:
            fallback_reason = "invalid_target_and_reference_smiles"
        elif target_mol is None:
            fallback_reason = "invalid_target_smiles"
        else:
            fallback_reason = "invalid_reference_smiles"
    elif not _RDKIT_AVAILABLE:
        fallback_reason = "rdkit_unavailable"
    elif not target_smiles and not reference_smiles:
        fallback_reason = "no_shared_smiles_level"
    elif not target_smiles:
        fallback_reason = "target_smiles_missing"
    else:
        fallback_reason = "reference_smiles_missing"

    target_tokens = _token_set_from_values(
        [target.get(key) for key in ALL_SMILES_KEYS]
    )
    reference_tokens = _token_set_from_values(
        [reference.get(key) for key in ALL_SMILES_KEYS]
    )
    return _jaccard(target_tokens, reference_tokens), {
        "method": "token_fallback",
        "fingerprint": None,
        "structure_comparison_level": "token_fallback",
        "structure_evidence_rank": 0,
        "radius": None,
        "n_bits": None,
        "target_smiles_field": target_field,
        "reference_smiles_field": reference_field,
        "smiles_fields_used": {
            "target": target_field,
            "reference": reference_field,
        },
        "rdkit_available": _RDKIT_AVAILABLE,
        "fallback_reason": fallback_reason,
    }


def _overlap(left: set[str], right: set[str]) -> float | None:
    if not left or not right:
        return None
    return len(left & right) / min(len(left), len(right))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _raw_context(context: dict[str, Any]) -> dict[str, Any]:
    raw = context.get("raw_context")
    return dict(raw) if isinstance(raw, dict) else dict(context)


def _reference_fields(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("raw_fields")
    if isinstance(raw, dict):
        merged = dict(raw)
        merged.update({k: v for k, v in record.items() if k != "raw_fields"})
        return merged
    return dict(record)


def _aliases(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("aliases", "synonyms", "polymer_synonyms"):
        value = payload.get(key)
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, (list, tuple, set)):
            out.extend(str(item) for item in value if item not in (None, ""))
    for key in (
        "polymer_family",
        "polymer_type",
        "polymer_name",
        "repeat_unit_name",
        "repeat_unit_smiles",
        "monomer_smiles",
        "smiles",
        "poly_smiles",
        "mol_smiles",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            out.append(str(value))
    return out


def _infer_functional_groups(payload: dict[str, Any]) -> set[str]:
    explicit = payload.get("functional_groups")
    groups: set[str] = set()
    if isinstance(explicit, str):
        groups |= _token_set_from_values([explicit])
    elif isinstance(explicit, (list, tuple, set)):
        groups |= _token_set_from_values(list(explicit))
    haystack = " ".join(_aliases(payload)).lower()
    checks = {
        "carbonate": ["carbonate", "co3", "o=c(o)o"],
        "ether": ["ether", "coc", "polyether"],
        "ester": ["ester", "c(=o)o"],
        "urethane": ["urethane", "carbamate"],
        "olefin": ["olefin", "octene", "alpha_olefin", "alpha olefin"],
        "hydrocarbon": ["polyethylene", "pe", "alkane", "olefin"],
        "aromatic": ["phenyl", "benzene", "aromatic"],
        "branched": ["branched", "branch", "lcb"],
    }
    for group, needles in checks.items():
        if any(needle in haystack for needle in needles):
            groups.add(group)
    return groups


def _numeric_similarity(target: dict[str, Any], reference: dict[str, Any], keys: tuple[str, ...], tolerance: float) -> tuple[float | None, list[str]]:
    scores: list[float] = []
    used: list[str] = []
    for key in keys:
        left = _as_float(target.get(key))
        right = _as_float(reference.get(key))
        if left is None or right is None:
            continue
        scale = max(abs(left), abs(right), tolerance, 1e-12)
        score = max(0.0, 1.0 - abs(left - right) / scale / tolerance)
        scores.append(score)
        used.append(key)
    if not scores:
        return None, []
    return sum(scores) / len(scores), used


def _unit_group(units: Any) -> str | None:
    slug = _slug(units)
    if not slug:
        return None
    for group, aliases in UNIT_GROUPS.items():
        if slug in aliases:
            return group
    return None


def _unit_compatibility(target: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    ref_units = reference.get("units") or reference.get("unit") or reference.get("reference_units")
    target_units = target.get("computed_units") or target.get("units") or target.get("unit") or target.get("value_units")
    ref_value = reference.get("value")
    reasons: list[str] = []
    ref_group = _unit_group(ref_units)
    target_group = _unit_group(target_units)
    if _as_float(ref_value) is None:
        reasons.append("reference_value_missing")
    if not ref_units:
        reasons.append("reference_units_missing")
    elif ref_group is None:
        reasons.append("reference_units_unrecognized")
    if target_units and target_group is None:
        reasons.append("target_units_unrecognized")
    if target_group and ref_group and target_group != ref_group:
        reasons.append("unit_group_mismatch")
    score = 1.0 if not reasons else 0.0
    if target_units in (None, "") and ref_group is not None and _as_float(ref_value) is not None:
        score = 1.0
    return {
        "score": score,
        "reasons": reasons,
        "target_units": target_units,
        "reference_units": ref_units,
        "target_unit_group": target_group,
        "reference_unit_group": ref_group,
    }


def _categorical_compatibility(target_value: Any, reference_value: Any, *, missing_score: float = 0.5) -> dict[str, Any]:
    target_slug = _slug(target_value)
    reference_slug = _slug(reference_value)
    if not target_slug or target_slug in CONDITION_ANY or not reference_slug or reference_slug in CONDITION_ANY:
        return {
            "score": missing_score,
            "reason": "missing_or_any",
            "target": target_slug or None,
            "reference": reference_slug or None,
        }
    if target_slug == reference_slug:
        return {"score": 1.0, "reason": "matched", "target": target_slug, "reference": reference_slug}
    return {"score": 0.0, "reason": "mismatch", "target": target_slug, "reference": reference_slug}


def _range_compatibility(target_value: Any, reference_value: Any, tolerance: float) -> dict[str, Any]:
    target_float = _as_float(target_value)
    reference_float = _as_float(reference_value)
    if target_float is None or reference_float is None:
        return {
            "score": 0.5,
            "reason": "missing",
            "target": target_float,
            "reference": reference_float,
            "delta": None,
            "tolerance": tolerance,
        }
    delta = abs(target_float - reference_float)
    score = max(0.0, 1.0 - delta / max(tolerance, 1e-12))
    return {
        "score": score,
        "reason": "matched" if delta <= tolerance else "outside_tolerance",
        "target": target_float,
        "reference": reference_float,
        "delta": delta,
        "tolerance": tolerance,
    }


def condition_compatibility(
    target_context: dict[str, Any],
    reference: dict[str, Any],
    *,
    temperature_tolerance_K: float = DEFAULT_TEMPERATURE_TOLERANCE_K,
    pressure_tolerance_bar: float = DEFAULT_PRESSURE_TOLERANCE_BAR,
) -> dict[str, Any]:
    target = _raw_context(target_context)
    ref = _reference_fields(reference)
    reasons: list[str] = []
    weighted_reasons: list[str] = []
    target_state = _slug(target_context.get("state") or target.get("state") or "melt")
    ref_state = _slug(ref.get("state"))
    state_check_status = "melt_required"
    if target_state != "melt":
        reasons.append("unsupported_target_state")
    if not ref_state or ref_state in CONDITION_ANY:
        reasons.append("reference_state_missing")
    elif ref_state != "melt":
        reasons.append("reference_state_not_melt")

    target_temp = _as_float(target_context.get("temperature_K") if target_context.get("temperature_K") is not None else target.get("temperature_K"))
    ref_temp = _as_float(ref.get("temperature_K"))
    temp_delta = None
    if target_temp is not None and ref_temp is not None:
        temp_delta = abs(target_temp - ref_temp)
        if temp_delta > temperature_tolerance_K:
            reasons.append("temperature_mismatch")

    target_pressure = _as_float(target_context.get("pressure_bar") if target_context.get("pressure_bar") is not None else target.get("pressure_bar"))
    ref_pressure = _as_float(ref.get("pressure_bar"))
    pressure_delta = None
    if target_pressure is not None and ref_pressure is not None:
        pressure_delta = abs(target_pressure - ref_pressure)
        if pressure_delta > pressure_tolerance_bar:
            reasons.append("pressure_mismatch")

    target_ff = canonical_force_field(target_context.get("force_field") or target.get("force_field"))
    ref_ff = canonical_force_field(ref.get("force_field"))
    if target_ff not in CONDITION_ANY and ref_ff not in CONDITION_ANY and target_ff and ref_ff and target_ff != ref_ff:
        reasons.append("force_field_mismatch")

    target_engine_info = canonical_engine_scope(
        target_context.get("engine_scope") or target.get("engine_scope") or target.get("engine")
    )
    ref_engine_info = canonical_engine_scope(ref.get("engine_scope") or ref.get("engine"))
    target_engine = str(target_engine_info.get("family") or "")
    ref_engine = str(ref_engine_info.get("family") or "")
    if target_engine not in CONDITION_ANY and ref_engine not in CONDITION_ANY and target_engine and ref_engine and target_engine != ref_engine:
        reasons.append("engine_mismatch")

    target_phase = _slug(target.get("phase_state") or target.get("phase_family"))
    ref_phase = _slug(ref.get("phase_state") or ref.get("phase_family"))
    phase_check_status = "melt_boundary_only"
    non_melt_phase_tokens = {"solid", "solid_like", "glass", "glassy", "glass_like"}
    if target_phase in non_melt_phase_tokens:
        reasons.append("unsupported_target_phase")
    if ref_phase in non_melt_phase_tokens:
        reasons.append("reference_phase_not_melt")
    elif target_phase and ref_phase and target_phase not in CONDITION_ANY and ref_phase not in CONDITION_ANY and target_phase != ref_phase:
        weighted_reasons.append("phase_mismatch")

    return {
        "status": "matched" if not reasons else "condition_mismatch",
        "reasons": reasons,
        "weighted_condition_reasons": list(dict.fromkeys(weighted_reasons)),
        "temperature_delta_K": temp_delta,
        "temperature_tolerance_K": temperature_tolerance_K,
        "pressure_delta_bar": pressure_delta,
        "pressure_tolerance_bar": pressure_tolerance_bar,
        "target_pressure_bar": target_pressure,
        "reference_pressure_bar": ref_pressure,
        "target_state": target_state or None,
        "reference_state": ref_state or None,
        "state_check_status": state_check_status,
        "target_force_field": target_ff or None,
        "reference_force_field": ref_ff or None,
        "target_engine": target_engine or None,
        "reference_engine": ref_engine or None,
        "target_phase": target_phase or None,
        "reference_phase": ref_phase or None,
        "phase_check_status": phase_check_status,
    }


def _machine_activation_assessment(
    context: dict[str, Any],
    record: dict[str, Any],
    similarity: dict[str, Any],
    *,
    high_similarity: float,
) -> dict[str, Any]:
    """Apply the strict, auditable gate used only for automatic activation.

    The looser ``min_similarity`` threshold still exposes useful analogues for
    MED/review.  Automatic rule creation additionally requires a real chemical
    fingerprint, repeat-unit-or-stronger structure evidence and complete
    numeric/unit provenance, preventing labels or matched temperature alone
    from manufacturing a transferable rule.
    """

    target = _raw_context(context)
    ref = _reference_fields(record)
    reasons: list[str] = []
    score = float(similarity.get("similarity_score", 0.0) or 0.0)
    if score < high_similarity:
        reasons.append("similarity_below_machine_activation_threshold")
    if similarity.get("repeat_unit_structure_method") != "morgan_tanimoto":
        reasons.append("chemical_fingerprint_required")
    if int(similarity.get("structure_evidence_rank", 0) or 0) < 2:
        reasons.append("repeat_unit_or_higher_structure_required")

    missing = set(similarity.get("missing_descriptors") or [])
    if "degree_or_chain_size" in missing:
        reasons.append("degree_or_chain_size_required")
    if "reference_value_missing" in missing:
        reasons.append("reference_value_required")
    if "reference_units_missing" in missing or "reference_units_unrecognized" in missing:
        reasons.append("reference_units_required")

    compatibility = similarity.get("condition_compatibility") or {}
    if compatibility.get("status") != "matched":
        reasons.extend(str(item) for item in compatibility.get("reasons") or ["condition_mismatch"])
    reasons.extend(str(item) for item in compatibility.get("weighted_condition_reasons") or [])
    for key, reason in (
        ("target_pressure_bar", "target_pressure_required"),
        ("reference_pressure_bar", "reference_pressure_required"),
    ):
        if compatibility.get(key) in (None, ""):
            reasons.append(reason)
    if _as_float(target.get("temperature_K")) is None:
        reasons.append("target_temperature_required")
    if _as_float(ref.get("temperature_K")) is None:
        reasons.append("reference_temperature_required")

    source_type = _slug(ref.get("source_type"))
    if source_type in CONTROL_REFERENCE_TYPES:
        reasons.append("control_reference_not_eligible_for_profile_activation")
    if source_type in COMPUTATIONAL_REFERENCE_TYPES:
        target_ff = _slug(target.get("force_field"))
        reference_ff = _slug(ref.get("force_field"))
        target_engine = _slug(target.get("engine_scope") or target.get("engine"))
        reference_engine = _slug(ref.get("engine_scope") or ref.get("engine"))
        if not target_ff or target_ff in CONDITION_ANY:
            reasons.append("target_force_field_required")
        if not reference_ff or reference_ff in CONDITION_ANY:
            reasons.append("reference_force_field_required")
        if not target_engine or target_engine in CONDITION_ANY:
            reasons.append("target_engine_required")
        if not reference_engine or reference_engine in CONDITION_ANY:
            reasons.append("reference_engine_required")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "policy_version": MACHINE_ACTIVATION_POLICY_VERSION,
        "eligible": not unique_reasons,
        "reasons": unique_reasons,
        "required_similarity": high_similarity,
        "repeat_unit_structure_method": similarity.get("repeat_unit_structure_method"),
    }


def compute_structure_similarity(
    target_context: dict[str, Any],
    reference: dict[str, Any],
    *,
    temperature_tolerance_K: float = DEFAULT_TEMPERATURE_TOLERANCE_K,
) -> dict[str, Any]:
    """Return a deterministic weighted similarity score and audit details."""

    target = _raw_context(target_context)
    ref = _reference_fields(reference)
    components: list[dict[str, Any]] = []
    missing: list[str] = []
    review_hints: dict[str, Any] = {}

    def add_component(name: str, weight: float, score: float, **provenance: Any) -> None:
        component = {"name": name, "weight": weight, "score": round(score, 4)}
        component.update({key: value for key, value in provenance.items() if value is not None})
        components.append(component)

    family_score = _overlap(_token_set_from_values(_aliases(target_context) + _aliases(target)), _token_set_from_values(_aliases(ref)))
    review_hints["family_alias_score"] = round(family_score, 4) if family_score is not None else None
    review_hints["target_aliases"] = _aliases(target_context) + _aliases(target)
    review_hints["reference_aliases"] = _aliases(ref)

    structure_score, structure_provenance = _smiles_structure_similarity(target, ref)
    if structure_score is None:
        missing.append("smiles_structure")
        add_component(
            "smiles_structure",
            0.36,
            0.0,
            **structure_provenance,
        )
    else:
        add_component("smiles_structure", 0.36, structure_score, **structure_provenance)

    group_score = _jaccard(_infer_functional_groups(target), _infer_functional_groups(ref))
    review_hints["functional_group_score"] = round(group_score, 4) if group_score is not None else None

    composition_score, composition_keys = _numeric_similarity(
        target,
        ref,
        CHAIN_SIZE_KEYS,
        0.25,
    )
    if composition_score is None:
        missing.append("degree_or_chain_size")
        add_component("degree_or_chain_size", 0.20, 0.0, method="normalized_numeric_distance")
    else:
        add_component("degree_or_chain_size", 0.20, composition_score, method="normalized_numeric_distance")

    sequence_score, sequence_keys = _numeric_similarity(
        target,
        ref,
        (
            "max_contiguous_carbonate_units",
            "average_segment_length",
            "sequence_block_length",
            "sequence_spacing",
            "branch_spacing",
            "Lc",
            "lc",
        ),
        0.30,
    )
    sequence_tokens = _jaccard(_token_set_from_values([target.get("sequence_pattern"), target.get("placement_code")]), _token_set_from_values([ref.get("sequence_pattern"), ref.get("placement_code")]))
    review_hints["sequence_topology_score"] = (
        round(sequence_score if sequence_score is not None else float(sequence_tokens or 0.0), 4)
        if sequence_score is not None or sequence_tokens is not None
        else None
    )

    compatibility = condition_compatibility(target_context, reference, temperature_tolerance_K=temperature_tolerance_K)
    state_score = 1.0 if compatibility.get("state_check_status") == "melt_required" and not any(
        reason in {"unsupported_target_state", "reference_state_missing", "reference_state_not_melt"}
        for reason in compatibility.get("reasons", [])
    ) else 0.0
    add_component(
        "melt_state_compatibility",
        0.12,
        state_score,
        method="hard_melt_state_check",
        reasons=[
            reason
            for reason in compatibility.get("reasons", [])
            if reason in {"unsupported_target_state", "reference_state_missing", "reference_state_not_melt"}
        ],
    )
    unit = _unit_compatibility(target, ref)
    if unit["reasons"]:
        missing.extend(str(reason) for reason in unit["reasons"])
    add_component(
        "unit_compatibility",
        0.12,
        float(unit["score"]),
        method="deterministic_unit_check",
        reasons=unit["reasons"],
        target_units=unit["target_units"],
        reference_units=unit["reference_units"],
        target_unit_group=unit["target_unit_group"],
        reference_unit_group=unit["reference_unit_group"],
    )

    target_engine_info = canonical_engine_scope(
        target_context.get("engine_scope") or target.get("engine_scope") or target.get("engine")
    )
    reference_engine_info = canonical_engine_scope(ref.get("engine_scope") or ref.get("engine"))
    engine = _categorical_compatibility(
        target_engine_info.get("family"),
        reference_engine_info.get("family"),
    )
    add_component("engine_compatibility", 0.12, float(engine["score"]), method="deterministic_engine_check", reason=engine["reason"])

    temperature = _range_compatibility(
        target_context.get("temperature_K") if target_context.get("temperature_K") is not None else target.get("temperature_K"),
        ref.get("temperature_K"),
        DEFAULT_TEMPERATURE_TOLERANCE_K,
    )
    add_component(
        "temperature_compatibility",
        0.09,
        float(temperature["score"]),
        method="deterministic_temperature_window",
        reason=temperature["reason"],
        delta_K=temperature["delta"],
        tolerance_K=temperature["tolerance"],
    )

    pressure = _range_compatibility(
        target_context.get("pressure_bar") if target_context.get("pressure_bar") is not None else target.get("pressure_bar"),
        ref.get("pressure_bar"),
        DEFAULT_PRESSURE_TOLERANCE_BAR,
    )
    add_component(
        "pressure_compatibility",
        0.06,
        float(pressure["score"]),
        method="deterministic_pressure_window",
        reason=pressure["reason"],
        delta_bar=pressure["delta"],
        tolerance_bar=pressure["tolerance"],
    )

    force_field = _categorical_compatibility(
        canonical_force_field(target_context.get("force_field") or target.get("force_field")),
        canonical_force_field(ref.get("force_field")),
    )
    add_component(
        "force_field_compatibility",
        0.05,
        float(force_field["score"]),
        method="deterministic_force_field_check",
        reason=force_field["reason"],
    )

    total_weight = sum(float(component["weight"]) for component in components)
    score = (
        sum(float(component["weight"]) * float(component["score"]) for component in components) / total_weight
        if total_weight
        else 0.0
    )
    nonzero_coverage_weight = sum(
        float(component["weight"])
        for component in components
        if component["name"] not in {"smiles_structure", "degree_or_chain_size"}
        or float(component["score"]) > 0.0
    )
    return {
        "schema_version": SIMILARITY_SCHEMA_VERSION,
        "distance_metric_version": DISTANCE_METRIC_VERSION,
        "similarity_score": round(score, 4),
        "components": components,
        "repeat_unit_structure_method": structure_provenance["method"],
        "structure_comparison_level": structure_provenance["structure_comparison_level"],
        "structure_evidence_rank": structure_provenance["structure_evidence_rank"],
        "smiles_fields_used": structure_provenance["smiles_fields_used"],
        "repeat_unit_structure_provenance": structure_provenance,
        "missing_descriptors": list(dict.fromkeys(missing)),
        "descriptor_coverage_fraction": round(nonzero_coverage_weight / total_weight, 4) if total_weight else 0.0,
        "critical_structure_complete": "smiles_structure" not in missing,
        "critical_chain_size_complete": "degree_or_chain_size" not in missing,
        "composition_keys_used": composition_keys,
        "sequence_keys_used": sequence_keys,
        "condition_compatibility": compatibility,
        "unit_compatibility": unit,
        "review_hints": review_hints,
    }


def _distance_reason(score: float, compatibility: dict[str, Any], missing: list[str], min_similarity: float) -> str:
    if "reference_value_missing" in missing or "reference_units_missing" in missing or "reference_units_unrecognized" in missing:
        return "reference_value_or_units_missing"
    if "smiles_structure" in missing:
        return "insufficient_structural_descriptors"
    if "degree_or_chain_size" in missing:
        return "insufficient_chain_size_descriptors"
    if compatibility.get("status") != "matched":
        return ",".join(compatibility.get("reasons") or ["condition_mismatch"])
    if score < min_similarity:
        return "similarity_below_threshold"
    return "matched"


def _transfer_uncertainty_rel(score: float) -> float:
    penalty = max(0.0, 1.0 - score) * DEFAULT_TRANSFER_UNCERTAINTY_SLOPE_REL
    return round(DEFAULT_TRANSFER_UNCERTAINTY_FLOOR_REL + penalty, 4)


def _similarity_tier(score: float | None, min_similarity: float, high_similarity: float) -> str:
    if score is None or score < min_similarity:
        return "REVIEW"
    if score >= high_similarity:
        return "HIGH"
    return "OK"


def discover_similarity_transfer(
    context: dict[str, Any],
    property_name: str,
    candidate_records: list[dict[str, Any]],
    *,
    computed_value: float | None = None,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    high_similarity: float = DEFAULT_HIGH_SIMILARITY,
) -> dict[str, Any]:
    """Find a deterministic similar-structure transfer candidate.

    ``min_similarity`` and ``high_similarity`` apply to the weighted composite
    structural/condition score. A score of 0.90 does not mean that reference
    and target property values differ by 10%; property tolerance remains a
    separate axis handled by profile uncertainty and rule-range checks.
    """

    if not 0.0 <= min_similarity <= high_similarity <= 1.0:
        raise ValueError("similarity thresholds must satisfy 0 <= min_similarity <= high_similarity <= 1")

    prop = canonical_property_name(property_name)
    candidates: list[dict[str, Any]] = []
    for record in candidate_records:
        if canonical_property_name(record.get("property")) != prop:
            continue
        similarity = compute_structure_similarity(context, record)
        score = float(similarity["similarity_score"])
        compatibility = similarity["condition_compatibility"]
        reason = _distance_reason(score, compatibility, list(similarity.get("missing_descriptors") or []), min_similarity)
        tier = _similarity_tier(score, min_similarity, high_similarity)
        activation_assessment = _machine_activation_assessment(
            context,
            record,
            similarity,
            high_similarity=high_similarity,
        )
        ref_value = record.get("value")
        u_rel = _transfer_uncertainty_rel(score)
        u_abs = abs(float(computed_value) * u_rel) if computed_value not in (None, "") and _as_float(computed_value) is not None else None
        candidates.append(
            {
                "reference_id": str(record.get("source_id") or record.get("independent_key") or ""),
                "source_type": record.get("source_type"),
                "source_provider_kind": record.get("source_provider_kind"),
                "independent_key": record.get("independent_key"),
                "extraction_confidence": record.get("extraction_confidence"),
                "reference_value": ref_value,
                "reference_units": record.get("units"),
                "reference_state": record.get("state"),
                "reference_temperature_K": record.get("temperature_K"),
                "reference_force_field": record.get("force_field"),
                "similarity_score": score,
                "similarity_tier": tier,
                "similarity_distance_reason": reason,
                "structure_comparison_level": similarity.get("structure_comparison_level"),
                "structure_evidence_rank": similarity.get("structure_evidence_rank"),
                "smiles_fields_used": similarity.get("smiles_fields_used"),
                "phase_check_status": compatibility.get("phase_check_status"),
                "state_check_status": compatibility.get("state_check_status"),
                "u_similarity_rel": u_rel,
                "u_similarity": u_abs,
                "status": (
                    "similarity_transfer_candidate"
                    if reason == "matched"
                    else "condition_mismatch"
                    if compatibility.get("status") != "matched"
                    and reason
                    not in {
                        "reference_value_or_units_missing",
                        "insufficient_structural_descriptors",
                        "insufficient_chain_size_descriptors",
                        "similarity_below_threshold",
                    }
                    else "similarity_distance_too_large"
                ),
                "machine_activation_eligible": activation_assessment["eligible"],
                "machine_activation_reasons": activation_assessment["reasons"],
                "machine_activation_policy_version": activation_assessment["policy_version"],
                "machine_freeze_eligible": activation_assessment["eligible"],
                "machine_freeze_reasons": activation_assessment["reasons"],
                "machine_freeze_policy_version": activation_assessment["policy_version"],
                "similarity": similarity,
                "source_record": record,
            }
        )

    candidates.sort(key=lambda item: float(item.get("similarity_score", 0.0)), reverse=True)
    best = candidates[0] if candidates else None
    if best is None:
        status = "no_reference_records"
        reason = "no_reference_records"
    elif best["status"] == "similarity_transfer_candidate":
        status = "similarity_transfer_candidate"
        reason = "matched"
    elif best["status"] == "condition_mismatch":
        status = "condition_mismatch"
        reason = str(best.get("similarity_distance_reason") or "condition_mismatch")
    else:
        status = "similarity_distance_too_large"
        reason = str(best.get("similarity_distance_reason") or "similarity_below_threshold")

    eligible_candidates = [
        item
        for item in candidates
        if item.get("status") == "similarity_transfer_candidate" and bool(item.get("machine_activation_eligible"))
    ]
    eligible_independent_keys = {
        str(item.get("independent_key") or item.get("reference_id") or "")
        for item in eligible_candidates
        if str(item.get("independent_key") or item.get("reference_id") or "")
    }
    machine_activation_reasons: list[str] = []
    for item in candidates:
        machine_activation_reasons.extend(str(reason) for reason in item.get("machine_activation_reasons") or [])
    if len(eligible_independent_keys) < 2:
        machine_activation_reasons.append("at_least_two_independent_high_quality_references_required")
    machine_activation_reasons = list(dict.fromkeys(machine_activation_reasons))
    machine_activation_eligible = status == "similarity_transfer_candidate" and len(eligible_independent_keys) >= 2

    return {
        "status": status,
        "legacy_status": "profile_unfreezable" if status == "no_reference_records" else status,
        "schema_version": SIMILARITY_SCHEMA_VERSION,
        "distance_metric_version": DISTANCE_METRIC_VERSION,
        "property": prop,
        "min_similarity": min_similarity,
        "high_similarity": high_similarity,
        "computed_value": computed_value,
        "similarity_reference_id": best.get("reference_id") if best else None,
        "similarity_score": best.get("similarity_score") if best else None,
        "similarity_tier": best.get("similarity_tier") if best else "NO_MATCH",
        "similarity_distance_reason": reason,
        "structure_comparison_level": best.get("structure_comparison_level") if best else None,
        "structure_evidence_rank": best.get("structure_evidence_rank") if best else None,
        "smiles_fields_used": best.get("smiles_fields_used") if best else None,
        "phase_check_status": best.get("phase_check_status") if best else "melt_boundary_only",
        "u_similarity_rel": best.get("u_similarity_rel") if best else None,
        "u_similarity": best.get("u_similarity") if best else None,
        "machine_activation_policy_version": MACHINE_ACTIVATION_POLICY_VERSION,
        "machine_activation_eligible": machine_activation_eligible,
        "machine_activation_eligible_reference_count": len(eligible_independent_keys),
        "machine_activation_reasons": [] if machine_activation_eligible else machine_activation_reasons,
        # Compatibility aliases for existing callers and archived packets.
        "machine_freeze_policy_version": MACHINE_ACTIVATION_POLICY_VERSION,
        "machine_freeze_eligible": machine_activation_eligible,
        "machine_freeze_eligible_reference_count": len(eligible_independent_keys),
        "machine_freeze_reasons": [] if machine_activation_eligible else machine_activation_reasons,
        "best_candidate": _json_safe({k: v for k, v in best.items() if k != "source_record"}) if best else None,
        "candidates": _json_safe([{k: v for k, v in item.items() if k != "source_record"} for item in candidates[:10]]),
        "audit_required": status != "similarity_transfer_candidate" or not machine_activation_eligible,
        # Compatibility field retained for archived callers.
        "review_flag": status != "similarity_transfer_candidate" or not machine_activation_eligible,
        "notes": [
            "Similarity transfer is an estimated rule-calibration branch for missing direct profiles.",
            "LLM routing may trigger this branch, but similarity, compatibility and activation checks are deterministic.",
            "Similarity tiers describe a composite structural/condition score, not a percent difference between property values.",
        ],
    }


def similarity_source_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Convert a transfer candidate into a rule-profile reference record."""

    source = dict(candidate.get("source_record") or {})
    if not source:
        raise ValueError("similarity candidate has no source_record")
    source["source_type"] = source.get("source_type") or "similarity_transfer"
    source["similarity_transfer_source"] = True
    source["similarity_reference_id"] = candidate.get("reference_id")
    source["similarity_score"] = candidate.get("similarity_score")
    source["similarity_tier"] = candidate.get("similarity_tier")
    source["similarity_distance_reason"] = candidate.get("similarity_distance_reason")
    source["structure_comparison_level"] = candidate.get("structure_comparison_level")
    source["structure_evidence_rank"] = candidate.get("structure_evidence_rank")
    source["smiles_fields_used"] = candidate.get("smiles_fields_used")
    source["phase_check_status"] = candidate.get("phase_check_status")
    source["distance_metric_version"] = DISTANCE_METRIC_VERSION
    source["transfer_uncertainty_rel"] = candidate.get("u_similarity_rel")
    return source


__all__ = [
    "DEFAULT_HIGH_SIMILARITY",
    "DEFAULT_MIN_SIMILARITY",
    "DISTANCE_METRIC_VERSION",
    "MACHINE_FREEZE_POLICY_VERSION",
    "MACHINE_ACTIVATION_POLICY_VERSION",
    "MORGAN_NBITS",
    "MORGAN_RADIUS",
    "SIMILARITY_SCHEMA_VERSION",
    "compute_structure_similarity",
    "condition_compatibility",
    "discover_similarity_transfer",
    "similarity_source_from_candidate",
]
