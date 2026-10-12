#!/usr/bin/env python3
"""Versioned polymer rule profiles with reference-record activation.

This module keeps rule calibration separate from runtime rule application.
Codex or another agent may assemble reference records, but a rule is only used after it
has been normalized, conflict-checked, written as a versioned profile, and
registered through a ledgered activation step.

Legacy ``freeze`` fields remain readable for backward compatibility. New
outputs also expose activation terminology, which reflects the actual behavior:
an activated profile is versioned and replaceable rather than permanently
immutable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from radonpy_density_screening import (
    DEFAULT_CASE_MATCH_PACK,
    DEFAULT_REFERENCE_PACK,
    load_case_match_pack,
    load_reference_pack,
    match_case_to_references,
)

try:
    from rdkit import Chem

    _RDKIT_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - optional dependency
    Chem = None
    _RDKIT_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RULE_PROFILE_DIR = SCRIPT_DIR / "output" / "rule_profiles"
RULE_PROFILE_SCHEMA_VERSION = "polymer_rule_profile.v1"
DEFAULT_DENSITY_AGREEMENT_TOLERANCE_GCC = 0.08
DEFAULT_RELATIVE_AGREEMENT_TOLERANCE = 0.25
STRUCTURE_SCOPE_SCHEMA_VERSION = "rule_profile_structure_scope.v1"
HIGH_TRUST_SCORE_THRESHOLD = 0.82
STRUCTURE_SCOPE_KEYS: dict[str, tuple[str, ...]] = {
    "structure_hash": ("structure_hash", "canonical_structure_hash", "polymer_structure_hash"),
    "inchikey": ("inchikey", "inchi_key", "polymer_inchikey", "monomer_inchikey"),
    "chain_smiles": ("canonical_chain_smiles", "chain_smiles", "polymer_chain_smiles", "oligomer_smiles"),
    "normalized_segment_smiles": ("canonical_segment_smiles", "normalized_segment_smiles", "surrogate_dp_smiles"),
    "repeat_unit_smiles": ("repeat_unit_smiles", "canonical_repeat_unit_smiles", "poly_smiles", "smiles"),
    "monomer_smiles": ("monomer_smiles", "mol_smiles"),
    "degree_of_polymerization": ("degree_of_polymerization", "polymerization_degree", "DP", "dp", "repeat_units_per_chain"),
    "chain_length": ("chain_length", "target_total_carbons_per_chain", "actual_total_carbons_per_chain"),
    "molecular_weight": ("molecular_weight", "Mn", "mn", "Mw", "mw"),
}

HIGH_TRUST_SOURCE_TYPES = {
    "radonpy",
    "htpmd",
    "polyinfo",
    "authoritative_database",
    "handbook",
    "crc",
    "nist",
    "benchmark_paper",
    "peer_reviewed_literature",
}
CONTROL_SOURCE_TYPES = {
    "clean_engine_control",
    "same_input_clean_control",
    "internal_clean_control",
}
PROFILE_EXCLUDED_PROVIDER_KINDS = {
    "repair_result",
    "clean_repair_result",
    "current_case_result",
    "current_case_output",
    "case_result",
    "postprocess_result",
    "regate_result",
    "clean_overlay_regate",
    "closed_loop_result",
    "runtime_result",
    "md_result",
}
PROFILE_EXCLUDED_SOURCE_ROLES = {
    "repair_result",
    "current_case_result",
    "regate_result",
    "control_source_excluded_from_profile",
}
SUPPORTING_SOURCE_TYPES = {
    "literature",
    "paper",
    "preprint",
    "database",
    "documentation",
    "open_documentation",
}

PROPERTY_SPECS: dict[str, dict[str, Any]] = {
    "density": {
        "units": "g/cm3",
        "value_keys": ["rho_final_gcc_si_o", "rho_final_gcc", "final_density_g_per_cc", "density_g_per_cc", "density_gcc", "density"],
        "conflict_abs": 0.08,
        "padding_abs": 0.015,
        "single_source_screening_padding_abs": 0.08,
        "single_source_screening_min_replicates": 3,
        "profile_kind": "public_data_profile",
    },
    "self_diffusion": {
        "units": "m2/s",
        "value_keys": ["self_diffusion", "self-diffusion", "D_mean_m2_s", "diffusion_m2_s"],
        "conflict_rel": 0.45,
        "padding_rel": 0.12,
        "profile_kind": "public_data_profile",
    },
    "rg": {
        "units": "nm",
        "value_keys": ["rg_mean_nm", "Rg", "rg"],
        "conflict_rel": 0.35,
        "padding_rel": 0.10,
        "requires_protocol_metadata": True,
        "profile_kind": "public_data_profile_chain_sensitive",
    },
    "r2": {
        "units": "nm2",
        "value_keys": ["r2", "rg2_mean_nm2", "Rg2", "r2_mean_nm2"],
        "conflict_rel": 0.45,
        "padding_rel": 0.12,
        "requires_protocol_metadata": True,
        "profile_kind": "public_data_profile_chain_sensitive",
    },
    "cp": {"units": "J/kg/K", "value_keys": ["Cp", "cp"], "conflict_rel": 0.30, "padding_rel": 0.10, "profile_kind": "public_data_profile"},
    "cv": {"units": "J/kg/K", "value_keys": ["Cv", "cv"], "conflict_rel": 0.30, "padding_rel": 0.10, "profile_kind": "public_data_profile"},
    "compressibility": {
        "units": "1/Pa",
        "value_keys": ["compressibility"],
        "conflict_rel": 0.40,
        "padding_rel": 0.15,
        "profile_kind": "public_data_profile",
    },
    "bulk_modulus": {
        "units": "Pa",
        "value_keys": ["bulk_modulus"],
        "conflict_rel": 0.40,
        "padding_rel": 0.15,
        "profile_kind": "public_data_profile",
    },
    "thermal_expansion": {
        "units": "1/K",
        "value_keys": ["volume_expansion", "thermal_expansion", "linear_expansion"],
        "conflict_rel": 0.40,
        "padding_rel": 0.15,
        "profile_kind": "public_data_profile",
    },
    "dielectric_constant": {
        "units": "dimensionless",
        "value_keys": ["static_dielectric_const", "dielectric_const_dc", "dielectric_constant"],
        "conflict_rel": 0.30,
        "padding_rel": 0.10,
        "profile_kind": "public_data_profile",
    },
    "refractive_index": {
        "units": "dimensionless",
        "value_keys": ["refractive_index"],
        "conflict_abs": 0.08,
        "padding_abs": 0.02,
        "profile_kind": "public_data_profile",
    },
    "thermal_conductivity": {
        "units": "W/m/K",
        "value_keys": ["thermal_conductivity"],
        "conflict_rel": 0.35,
        "padding_rel": 0.10,
        "profile_kind": "public_data_profile",
    },
    "thermal_diffusivity": {
        "units": "m2/s",
        "value_keys": ["thermal_diffusivity"],
        "conflict_rel": 0.40,
        "padding_rel": 0.15,
        "profile_kind": "public_data_profile",
    },
    "conductivity": {
        "units": "S/m",
        "value_keys": ["conductivity"],
        "conflict_rel": 0.45,
        "padding_rel": 0.15,
        "profile_kind": "public_data_profile",
    },
    "ion_diffusivity": {
        "units": "m2/s",
        "value_keys": ["li_diffusivity", "tfsi_diffusivity", "ion_diffusivity"],
        "conflict_rel": 0.45,
        "padding_rel": 0.15,
        "profile_kind": "public_data_profile",
    },
    "li_diffusivity": {
        "units": "m2/s",
        "value_keys": ["li_diffusivity"],
        "conflict_rel": 0.45,
        "padding_rel": 0.15,
        "profile_kind": "public_data_profile",
    },
    "tfsi_diffusivity": {
        "units": "m2/s",
        "value_keys": ["tfsi_diffusivity"],
        "conflict_rel": 0.45,
        "padding_rel": 0.15,
        "profile_kind": "public_data_profile",
    },
    "poly_diffusivity": {
        "units": "m2/s",
        "value_keys": ["poly_diffusivity"],
        "conflict_rel": 0.45,
        "padding_rel": 0.15,
        "requires_protocol_metadata": True,
        "profile_kind": "public_data_profile",
    },
    "transference_number": {
        "units": "dimensionless",
        "value_keys": ["transference_number"],
        "conflict_abs": 0.12,
        "padding_abs": 0.04,
        "profile_kind": "public_data_profile",
    },
    "molarity": {
        "units": "mol/L",
        "value_keys": ["molarity"],
        "conflict_rel": 0.30,
        "padding_rel": 0.10,
        "profile_kind": "public_data_profile",
    },
}

# Compatibility is property-scoped even where the current conservative policy
# is strict. Future broad sanity profiles must opt in explicitly per property.
for _property_spec in PROPERTY_SPECS.values():
    _property_spec.setdefault("force_field_policy", "strict")
    _property_spec.setdefault("engine_scope_policy", "strict")

PROPERTY_ALIASES = {
    "diffusivity": "self_diffusion",
    "self-diffusion": "self_diffusion",
    "self_diffusion_coefficient": "self_diffusion",
    "self_diffusion": "self_diffusion",
    "rg": "rg",
    "Rg": "rg",
    "rg_mean_nm": "rg",
    "r2": "r2",
    "Rg2": "r2",
    "rg2": "r2",
    "rg2_mean_nm2": "r2",
    "thermal_expansion": "thermal_expansion",
    "volume_expansion": "thermal_expansion",
    "linear_expansion": "thermal_expansion",
    "static_dielectric_const": "dielectric_constant",
    "dielectric_const_dc": "dielectric_constant",
    "lithium_diffusivity": "li_diffusivity",
    "anion_diffusivity": "tfsi_diffusivity",
}

GENERIC_QC_PROPERTIES = [
    {
        "property": "rg_qc",
        "warn_sources": ["rg_warn"],
        "note": "Generic radius-of-gyration completeness/range checks; external profile freeze requires chain/protocol metadata.",
    },
    {"property": "shape", "warn_sources": ["shape_warn"], "note": "Shape-tensor schema and self-consistency checks."},
    {"property": "rdf", "warn_sources": ["rdf_warn"], "note": "RDF normalization/schema/currentness checks."},
    {"property": "contact", "warn_sources": ["contact_warn"], "note": "Contact-map completeness and finite-value checks."},
    {"property": "packing", "warn_sources": ["packing_warn"], "note": "Void/local-density/packing red-flag checks."},
    {"property": "msd", "warn_sources": ["msd_warn"], "note": "MSD fit quality, slope, and window diagnostics."},
    {"property": "fsqt", "warn_sources": ["fsqt_warn"], "note": "Fs(q,t) decay/window diagnostics."},
    {
        "property": "runtime_provenance",
        "warn_sources": ["runtime_warn", "provenance_warn", "evidence_incomplete_warn"],
        "note": "Engine, force-field, missingness, and runtime-integrity checks.",
    },
]

AUTO_FREEZE_PROPERTIES = frozenset(
    prop
    for prop, spec in PROPERTY_SPECS.items()
    if not bool(spec.get("requires_protocol_metadata"))
)
CONDITIONAL_AUTO_FREEZE_PROPERTIES = frozenset(
    prop
    for prop, spec in PROPERTY_SPECS.items()
    if bool(spec.get("requires_protocol_metadata"))
)
QC_ONLY_RULE_PROPERTIES = frozenset(item["property"] for item in GENERIC_QC_PROPERTIES) | {
    "rdf",
    "shape",
    "contact",
    "packing",
    "packing_void",
    "fsqt",
    "msd",
}


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
    return value


def _hash_payload(payload: Any) -> str:
    blob = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _slug(text: Any) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unspecified"


FORCE_FIELD_ALIASES = {
    "opls": "opls-aa",
    "oplsaa": "opls-aa",
    "opls_aa": "opls-aa",
    "opls_all_atom": "opls-aa",
    "oplsaa_l": "opls-aa",
    "gaff": "gaff",
    "gaff2": "gaff2",
    "gaff_2": "gaff2",
    "charmm": "charmm",
    "charmm36": "charmm36",
    "charmm_36": "charmm36",
    "amber": "amber",
    "dreiding": "dreiding",
}

ENGINE_FAMILY_ALIASES = {
    "gromacs": "gromacs",
    "gmx": "gromacs",
    "lammps": "lammps",
    "openmm": "openmm",
    "namd": "namd",
    "amber": "amber",
}


def _canonical_force_field(value: Any) -> str:
    slug = _slug(value)
    if slug == "unspecified":
        return ""
    compact = slug.replace("_", "")
    if "opls" in compact and ("aa" in compact or compact == "opls"):
        return "opls-aa"
    return FORCE_FIELD_ALIASES.get(slug) or FORCE_FIELD_ALIASES.get(compact) or slug.replace("_", "-")


def canonical_force_field(value: Any) -> str:
    """Return the registered force-field name used for profile matching."""

    return _canonical_force_field(value)


def _canonical_engine_scope(value: Any) -> dict[str, str | bool]:
    raw = str(value or "").strip()
    slug = _slug(raw)
    if slug in {"unspecified", "none", "missing", ""}:
        return {"raw": raw, "scope": "missing", "family": "", "version": "", "specific": False}
    if slug in {"any", "all", "all_engines", "engine_agnostic"}:
        return {"raw": raw, "scope": "any", "family": "any", "version": "", "specific": False}

    family = ""
    parts = slug.split("_")
    for token in parts:
        if token in ENGINE_FAMILY_ALIASES:
            family = ENGINE_FAMILY_ALIASES[token]
            break
    if not family:
        for alias, canonical in ENGINE_FAMILY_ALIASES.items():
            if alias in slug:
                family = canonical
                break
    if not family:
        family = parts[0] if parts else slug

    version = ""
    version_match = re.search(r"(?<!\d)(\d{4}(?:[._-]\d+){0,3}|\d+(?:[._-]\d+){1,3})(?!\d)", raw.lower())
    if version_match:
        version = version_match.group(1).replace("_", ".").replace("-", ".")
    return {"raw": raw, "scope": "engine", "family": family, "version": version, "specific": bool(version)}


def canonical_engine_scope(value: Any) -> dict[str, str | bool]:
    """Return structured engine provenance used for compatibility checks."""

    return _canonical_engine_scope(value)


def _engine_scope_compatible(profile_ref: dict[str, Any], context: dict[str, Any], property_name: str = "") -> bool:
    profile_engine = _canonical_engine_scope(profile_ref.get("engine_scope"))
    context_engine = _canonical_engine_scope(context.get("engine_scope") or context.get("engine"))
    policy = _slug(
        profile_ref.get("engine_scope_policy")
        or profile_ref.get("engine_compatibility_policy")
        or (PROPERTY_SPECS.get(canonical_property_name(property_name)) or {}).get("engine_scope_policy")
        or "strict"
    )
    if profile_engine["scope"] == "missing" or context_engine["scope"] == "missing":
        return False
    if policy in {"any", "engine_agnostic", "broad_sanity_check"}:
        return True
    if profile_engine["scope"] == "any":
        if context_engine["scope"] == "any":
            return True
        applicability = profile_ref.get("applicability_conditions") if isinstance(profile_ref.get("applicability_conditions"), dict) else {}
        allowed_family = _slug(profile_ref.get("engine_family") or applicability.get("engine_family"))
        if policy in {"engine_family_agnostic", "engine_family_scoped", "family_agnostic"}:
            return allowed_family in {"unspecified", str(context_engine["family"])}
        if profile_ref.get("profile_id") == "pe_lcb_c100_oplsaa_density_lower_bound":
            return context_engine["family"] == "gromacs"
        return False
    if context_engine["scope"] == "any":
        return False
    if profile_engine["family"] != context_engine["family"]:
        return False
    if policy in {"family", "engine_family_scoped", "compatibility_group"}:
        return True
    profile_version = str(profile_engine["version"])
    context_version = str(context_engine["version"])
    if profile_version and context_version and profile_version != context_version:
        return False
    if profile_version and not context_version:
        return False
    return True


def _force_field_family(value: Any) -> str:
    canonical = _canonical_force_field(value)
    if canonical.startswith("opls"):
        return "opls"
    if canonical.startswith("charmm"):
        return "charmm"
    if canonical.startswith("gaff") or canonical.startswith("amber"):
        return "amber-gaff"
    return canonical


def _force_field_compatible(profile_ref: dict[str, Any], context: dict[str, Any], property_name: str) -> bool:
    policy = _slug(
        profile_ref.get("force_field_policy")
        or (PROPERTY_SPECS.get(canonical_property_name(property_name)) or {}).get("force_field_policy")
        or "strict"
    )
    profile_force_field = _canonical_force_field(profile_ref.get("force_field"))
    context_force_field = _canonical_force_field(context.get("force_field"))
    if policy in {"any", "force_field_agnostic", "broad_sanity_check"}:
        return True
    if not profile_force_field or not context_force_field:
        return False
    if policy in {"family", "method_family", "compatibility_group"}:
        return _force_field_family(profile_force_field) == _force_field_family(context_force_field)
    return profile_force_field == context_force_field


def _scope_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isfinite(numeric):
        return f"{numeric:.8g}"
    return ""


def _canonical_structure_scope_value(canonical_key: str, value: Any) -> str:
    raw = _scope_value(value)
    if not raw:
        return ""
    if canonical_key == "inchikey":
        return raw.replace(" ", "").upper()
    if canonical_key == "structure_hash":
        return raw.lower()
    if canonical_key.endswith("smiles") and _RDKIT_AVAILABLE and Chem is not None:
        try:
            mol = Chem.MolFromSmiles(raw)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception:  # pragma: no cover - defensive against RDKit edge cases
            pass
    return raw


def _first_scope_value(payloads: list[dict[str, Any]], keys: tuple[str, ...]) -> tuple[str, str] | tuple[None, None]:
    for payload in payloads:
        for key in keys:
            value = _scope_value(payload.get(key))
            if value:
                return value, key
    return None, None


def _structure_scope(context: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic target-structure scope used for profile matching.

    Direct profile lookup remains family-based for legacy profiles.  New
    profiles that carry SMILES/chain-size fields are keyed to those fields so a
    broad ``polymer_family`` label cannot silently match a different structure.
    """

    raw = context.get("raw_context") if isinstance(context.get("raw_context"), dict) else {}
    applicability = context.get("applicability_conditions") if isinstance(context.get("applicability_conditions"), dict) else {}
    payloads = [dict(context), dict(raw), dict(applicability)]
    fields: dict[str, str] = {}
    source_fields: dict[str, str] = {}
    for canonical_key, aliases in STRUCTURE_SCOPE_KEYS.items():
        value, source_key = _first_scope_value(payloads, aliases)
        if value:
            fields[canonical_key] = _canonical_structure_scope_value(canonical_key, value)
            source_fields[canonical_key] = str(source_key)
    if not fields:
        return {
            "schema_version": STRUCTURE_SCOPE_SCHEMA_VERSION,
            "available": False,
            "fields": {},
            "source_fields": {},
            "scope_hash": None,
        }
    return {
        "schema_version": STRUCTURE_SCOPE_SCHEMA_VERSION,
        "available": True,
        "fields": fields,
        "source_fields": source_fields,
        "scope_hash": _hash_payload(fields),
    }


def canonical_property_name(property_name: Any) -> str:
    """Return the canonical registry property name for aliases used by providers."""

    raw = str(property_name or "").strip()
    slug = _slug(raw)
    if slug in PROPERTY_SPECS:
        return slug
    if raw in PROPERTY_ALIASES:
        return PROPERTY_ALIASES[raw]
    if slug in PROPERTY_ALIASES:
        return PROPERTY_ALIASES[slug]
    return slug


def is_qc_only_rule_property(property_name: Any) -> bool:
    """Return True for rule/QC diagnostics that should not use public-data thresholds."""

    prop = canonical_property_name(property_name)
    return prop in QC_ONLY_RULE_PROPERTIES and prop not in PROPERTY_SPECS


def rule_profile_missing_warn_source(property_name: Any) -> str:
    """Return the property-specific WARN source for a missing active profile."""

    prop = canonical_property_name(property_name)
    return f"{prop}_rule_profile_missing"


def rule_profile_automation_policy(property_name: Any) -> dict[str, Any]:
    """Classify how a property may enter automatic rule-profile calibration."""

    prop = canonical_property_name(property_name)
    if prop in AUTO_FREEZE_PROPERTIES:
        return {
            "property": prop,
            "automation_level": "auto_activation",
            "profile_activatable": True,
            "profile_freezable": True,
            "missing_warn_source": rule_profile_missing_warn_source(prop),
            "fallback_outcome": "profile_not_activatable",
            "fallback_status": "profile_unfreezable",
            "reason": "structured independent reference records can activate this property profile when compatibility checks pass",
        }
    if prop in CONDITIONAL_AUTO_FREEZE_PROPERTIES:
        return {
            "property": prop,
            "automation_level": "conditional_auto_activation",
            "profile_activatable": True,
            "profile_freezable": True,
            "missing_warn_source": rule_profile_missing_warn_source(prop),
            "fallback_outcome": "profile_not_activatable",
            "fallback_status": "profile_unfreezable",
            "reason": "this property is chain/protocol sensitive and requires matching structure and protocol metadata",
        }
    if is_qc_only_rule_property(prop):
        return {
            "property": prop,
            "automation_level": "qc_only",
            "profile_activatable": False,
            "profile_freezable": False,
            "missing_warn_source": rule_profile_missing_warn_source(prop),
            "fallback_outcome": "profile_not_activatable",
            "fallback_status": "profile_unfreezable",
            "reason": "this observable is governed by schema/quality checks rather than an external public-data pass threshold",
        }
    return {
        "property": prop,
        "automation_level": "unsupported",
        "profile_activatable": False,
        "profile_freezable": False,
        "missing_warn_source": rule_profile_missing_warn_source(prop),
        "fallback_outcome": "unsupported_property",
        "fallback_status": "unsupported_property",
        "reason": "property is not registered for rule-profile calibration",
    }


def _activation_readiness_label(legacy_status: str) -> str:
    """Use activation terminology while retaining legacy status values."""

    if legacy_status == "eligible_for_machine_freeze":
        return "eligible_for_activation"
    return legacy_status


def get_property_spec(property_name: Any) -> dict[str, Any] | None:
    prop = canonical_property_name(property_name)
    spec = PROPERTY_SPECS.get(prop)
    return dict(spec) if spec else None


def list_supported_rule_properties() -> dict[str, Any]:
    """List public-data profile properties and generic QC-only rule sources."""

    public_props = []
    for prop, spec in sorted(PROPERTY_SPECS.items()):
        public_props.append(
            {
                "property": prop,
                "units": spec.get("units"),
                "profile_kind": spec.get("profile_kind", "public_data_profile"),
                "automation_level": rule_profile_automation_policy(prop)["automation_level"],
                "missing_warn_source": rule_profile_missing_warn_source(prop),
                "requires_protocol_metadata": bool(spec.get("requires_protocol_metadata")),
                "value_keys": list(spec.get("value_keys", [])),
            }
        )
    return {
        "status": "ok",
        "schema_version": RULE_PROFILE_SCHEMA_VERSION,
        "public_data_profile_properties": public_props,
        "generic_qc_properties": [
            item
            | {
                "automation_level": rule_profile_automation_policy(item.get("property"))["automation_level"],
                "missing_warn_source": rule_profile_missing_warn_source(item.get("property")),
                "fallback_status": rule_profile_automation_policy(item.get("property"))["fallback_status"],
            }
            for item in GENERIC_QC_PROPERTIES
        ],
        "notes": [
            "Public-data profiles are activated from independently matched reference records.",
            "Automatically activatable properties still require deterministic compatibility checks.",
            "Conditional properties require matching structure, chain and protocol metadata.",
            "Generic QC properties are rule-source diagnostics and do not require external public-data thresholds.",
        ],
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _parse_jsonish(value: str | dict[str, Any] | list[Any]) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_density_value(value: Any, units: str) -> float:
    """Return density in g/cm3."""

    density = float(value)
    unit = str(units or "").strip().lower().replace(" ", "")
    aliases = {
        "g/cm^3": "g/cm3",
        "g/cc": "g/cm3",
        "gcc": "g/cm3",
        "gml": "g/cm3",
        "g/ml": "g/cm3",
        "kg/m^3": "kg/m3",
        "kg/m**3": "kg/m3",
    }
    unit = aliases.get(unit, unit)
    if unit == "g/cm3":
        return density
    if unit == "kg/m3":
        return density / 1000.0
    raise ValueError(f"Unsupported density unit: {units!r}")


def normalize_property_value(value: Any, units: str, property_name: str) -> float:
    """Normalize a public-data property value to the registry's canonical units."""

    prop = canonical_property_name(property_name)
    spec = PROPERTY_SPECS.get(prop)
    if spec is None:
        raise ValueError(f"Unsupported rule-profile property: {property_name!r}")
    target_units = str(spec.get("units") or "").strip()
    if prop == "density":
        return normalize_density_value(value, units)

    numeric = float(value)
    unit = str(units or target_units).strip().lower().replace(" ", "")
    unit = unit.replace("−", "-")
    unit_aliases = {
        "": "dimensionless",
        "1": "dimensionless",
        "unitless": "dimensionless",
        "dimensionless": "dimensionless",
        "m^2/s": "m2/s",
        "m**2/s": "m2/s",
        "m2s-1": "m2/s",
        "cm^2/s": "cm2/s",
        "cm**2/s": "cm2/s",
        "cm2s-1": "cm2/s",
        "w/mk": "w/m/k",
        "w/(m*k)": "w/m/k",
        "w/m/k": "w/m/k",
        "w/m/k.": "w/m/k",
        "j/kg/k": "j/kg/k",
        "j/(kg*k)": "j/kg/k",
        "j/g/k": "j/g/k",
        "pa^-1": "1/pa",
        "1/pa": "1/pa",
        "1/gpa": "1/gpa",
        "k^-1": "1/k",
        "1/k": "1/k",
        "s/m": "s/m",
        "ms/cm": "ms/cm",
        "s/cm": "s/cm",
        "å": "angstrom",
        "a": "angstrom",
        "ang": "angstrom",
        "angstrom": "angstrom",
        "nm^2": "nm2",
        "a^2": "angstrom2",
        "angstrom^2": "angstrom2",
        "ang2": "angstrom2",
    }
    unit = unit_aliases.get(unit, unit)
    target = target_units.lower().replace(" ", "")

    if target in {"dimensionless"}:
        if unit not in {"dimensionless"}:
            raise ValueError(f"Unsupported unit for {prop}: {units!r}")
        return numeric
    if target == "m2/s":
        if unit == "m2/s":
            return numeric
        if unit == "cm2/s":
            return numeric * 1e-4
    if target == "w/m/k":
        if unit == "w/m/k":
            return numeric
    if target == "s/m":
        if unit == "s/m":
            return numeric
        if unit == "ms/cm":
            return numeric * 0.1
        if unit == "s/cm":
            return numeric * 100.0
    if target == "nm":
        if unit == "nm":
            return numeric
        if unit == "angstrom":
            return numeric * 0.1
    if target == "nm2":
        if unit == "nm2":
            return numeric
        if unit == "angstrom2":
            return numeric * 0.01
    if target == "j/kg/k":
        if unit == "j/kg/k":
            return numeric
        if unit == "j/g/k":
            return numeric * 1000.0
    if target == "pa":
        if unit == "pa":
            return numeric
        if unit == "mpa":
            return numeric * 1e6
        if unit == "gpa":
            return numeric * 1e9
    if target == "1/pa":
        if unit == "1/pa":
            return numeric
        if unit == "1/gpa":
            return numeric / 1e9
    if target == "1/k":
        if unit == "1/k":
            return numeric
    if target == "mol/l":
        if unit in {"mol/l", "m", "molar"}:
            return numeric
        if unit == "mol/m3":
            return numeric / 1000.0

    raise ValueError(f"Unsupported unit for {prop}: {units!r}; expected {target_units!r}")


def _source_trust_score(source_type: str, confidence: float) -> float:
    stype = _slug(source_type)
    base = 0.45
    if stype in HIGH_TRUST_SOURCE_TYPES:
        base = 0.9
    elif stype in CONTROL_SOURCE_TYPES:
        # Controls are useful for repair/regate diagnostics, but they are not
        # independent standard-setting reference records for activating a rule profile.
        base = 0.55
    elif stype in SUPPORTING_SOURCE_TYPES:
        base = 0.65
    return round(max(0.0, min(1.0, 0.7 * base + 0.3 * float(confidence))), 3)


def normalize_source_evidence(source: dict[str, Any], property_name: str) -> dict[str, Any]:
    property_slug = canonical_property_name(property_name)
    spec = PROPERTY_SPECS.get(property_slug)
    if spec is None:
        raise ValueError(f"Unsupported rule-profile property: {property_name!r}")
    if "value" not in source:
        raise ValueError("reference record is missing value")
    if "units" not in source:
        raise ValueError("reference record is missing units")
    source_property = canonical_property_name(source.get("property") or property_slug)
    if source_property != property_slug:
        raise ValueError(f"reference record property mismatch: {source.get('property')!r} != {property_slug!r}")

    if source.get("extraction_confidence") not in (None, ""):
        confidence = float(source["extraction_confidence"])
        confidence_origin = str(source.get("extraction_confidence_origin") or "source_record.extraction_confidence")
    elif source.get("confidence") not in (None, ""):
        confidence = float(source["confidence"])
        confidence_origin = str(source.get("extraction_confidence_origin") or "source_record.confidence")
    else:
        confidence = 0.5
        confidence_origin = "default_0.5"
    source_type = str(source.get("source_type") or source.get("type") or "literature")
    normalized = dict(source)
    normalized["property"] = property_slug
    normalized["normalized_value"] = normalize_property_value(source["value"], str(source["units"]), property_slug)
    normalized["normalized_units"] = spec.get("units")
    for source_key, normalized_key in (
        ("value_min", "normalized_value_min"),
        ("value_max", "normalized_value_max"),
        ("value_std", "normalized_value_std"),
    ):
        if source.get(source_key) not in (None, ""):
            normalized[normalized_key] = normalize_property_value(source[source_key], str(source["units"]), property_slug)
    normalized["source_id"] = str(source.get("source_id") or source.get("doi") or source.get("url") or _hash_payload(source)[:12])
    normalized["independent_key"] = str(source.get("independent_key") or source.get("doi") or source.get("url") or normalized["source_id"])
    normalized["source_type"] = _slug(source_type)
    normalized["extraction_confidence"] = round(max(0.0, min(1.0, confidence)), 3)
    normalized["extraction_confidence_origin"] = confidence_origin
    normalized["source_trust_score"] = _source_trust_score(source_type, normalized["extraction_confidence"])
    normalized["source_trust_interpretation"] = "heuristic_source_tier_not_statistical_confidence"
    normalized["high_trust_by_type"] = normalized["source_type"] in HIGH_TRUST_SOURCE_TYPES
    normalized["high_trust_by_score"] = normalized["source_trust_score"] >= HIGH_TRUST_SCORE_THRESHOLD
    return normalized


def _profile_id(context: dict[str, Any], property_name: str) -> str:
    parts = [
        _slug(context.get("polymer_family")),
        _slug(context.get("force_field")),
        _slug(context.get("engine_scope")),
        _slug(property_name),
    ]
    scope = _structure_scope(context)
    if scope.get("available") and scope.get("scope_hash"):
        parts.append(str(scope["scope_hash"])[:12])
    return "_".join(parts)


def _confidence_score(sources: list[dict[str, Any]], spread: float, tolerance: float) -> float:
    if not sources:
        return 0.0
    trust = sum(float(src.get("source_trust_score", 0.0)) for src in sources) / len(sources)
    agreement = max(0.0, 1.0 - min(1.0, spread / max(tolerance, 1e-9)))
    diversity = min(1.0, len({src.get("independent_key") for src in sources}) / 3.0)
    return round(0.55 * trust + 0.3 * agreement + 0.15 * diversity, 3)


def _is_control_source(source: dict[str, Any]) -> bool:
    return str(source.get("source_type")) in CONTROL_SOURCE_TYPES


def _case_reference_keys(payload: dict[str, Any] | None) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    keys = {
        "case_id",
        "clean_case_id",
        "target_case_id",
        "source_case_id",
        "original_case_id",
        "repair_case_id",
        "simulation_id",
        "run_id",
    }
    return {str(payload.get(key)).strip() for key in keys if payload.get(key) not in (None, "")}


def _profile_exclusion_reason(source: dict[str, Any], context: dict[str, Any] | None = None) -> str | None:
    provider_kind = _slug(source.get("source_provider_kind") or source.get("provider_kind"))
    source_role = _slug(source.get("source_role") or source.get("role"))
    if provider_kind in PROFILE_EXCLUDED_PROVIDER_KINDS or "repair" in provider_kind or "current_case" in provider_kind:
        return "current_case_or_repair_result"
    if source_role in PROFILE_EXCLUDED_SOURCE_ROLES or "repair" in source_role or "current_case" in source_role:
        return "current_case_or_repair_result"
    source_case_keys = _case_reference_keys(source)
    context_case_keys = _case_reference_keys(context)
    if source_case_keys and context_case_keys and source_case_keys & context_case_keys:
        return "current_case_reference"
    if _is_control_source(source):
        return "control_source"
    return None


def _profile_standard_sources(sources: list[dict[str, Any]], context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [src for src in sources if _profile_exclusion_reason(src, context) is None]


def _profile_excluded_sources(sources: list[dict[str, Any]], context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = []
    for src in sources:
        reason = _profile_exclusion_reason(src, context)
        if reason is None:
            continue
        annotated = dict(src)
        annotated["profile_exclusion_reason"] = reason
        excluded.append(annotated)
    return excluded


def _evidence_condition_met(sources: list[dict[str, Any]], context: dict[str, Any] | None = None) -> bool:
    standard_sources = _profile_standard_sources(sources, context)
    independent = {src.get("independent_key") for src in standard_sources if src.get("independent_key") not in (None, "")}
    high_trust = [
        src
        for src in standard_sources
        if str(src.get("source_type")) in HIGH_TRUST_SOURCE_TYPES
        or float(src.get("source_trust_score", 0.0)) >= HIGH_TRUST_SCORE_THRESHOLD
    ]
    return (len(independent) >= 2 and len(high_trust) >= 1) or _single_source_screening_eligible(standard_sources)


def _single_source_screening_eligible(sources: list[dict[str, Any]]) -> bool:
    if len(sources) != 1:
        return False
    source = sources[0]
    if canonical_property_name(source.get("property")) != "density":
        return False
    if not bool(source.get("single_source_screening_eligible")) or not bool(source.get("screening_only")):
        return False
    replicate_count = int(float(source.get("replicate_count") or 0))
    minimum_replicates = int(PROPERTY_SPECS["density"].get("single_source_screening_min_replicates", 3))
    has_bounds = source.get("normalized_value_min") is not None and source.get("normalized_value_max") is not None
    high_trust = bool(source.get("high_trust_by_type")) or float(source.get("source_trust_score", 0.0)) >= HIGH_TRUST_SCORE_THRESHOLD
    return replicate_count >= minimum_replicates and has_bounds and high_trust


def _evidence_reference_status(sources: list[dict[str, Any]], context: dict[str, Any] | None = None) -> str:
    standard_sources = _profile_standard_sources(sources, context)
    independent = {src.get("independent_key") for src in standard_sources if src.get("independent_key") not in (None, "")}
    if not sources:
        return "no_source_reference"
    if not standard_sources or not independent:
        excluded_reasons = {_profile_exclusion_reason(src, context) for src in sources}
        if excluded_reasons - {"control_source", None}:
            return "profile_excluded_reference"
        return "control_only_reference"
    if _single_source_screening_eligible(standard_sources):
        return "single_source_public_screening_reference"
    if len(independent) == 1:
        return "single_source_reference"
    high_trust = [
        src
        for src in standard_sources
        if str(src.get("source_type")) in HIGH_TRUST_SOURCE_TYPES
        or float(src.get("source_trust_score", 0.0)) >= HIGH_TRUST_SCORE_THRESHOLD
    ]
    if not high_trust:
        return "multiple_sources_no_trusted_anchor"
    return "multiple_source_references"


def _reference_record_counts(
    sources: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> dict[str, int]:
    standard_sources = _profile_standard_sources(sources, context)
    excluded_sources = _profile_excluded_sources(sources, context)
    control_sources = [src for src in excluded_sources if src.get("profile_exclusion_reason") == "control_source"]
    return {
        "source_count": len(standard_sources),
        "independent_source_count": len(
            {src.get("independent_key") for src in standard_sources if src.get("independent_key") not in (None, "")}
        ),
        "high_trust_source_count": sum(
            1
            for src in standard_sources
            if str(src.get("source_type")) in HIGH_TRUST_SOURCE_TYPES
            or float(src.get("source_trust_score", 0.0)) >= HIGH_TRUST_SCORE_THRESHOLD
        ),
        "high_trust_by_type_count": sum(bool(src.get("high_trust_by_type")) for src in standard_sources),
        "high_trust_by_score_count": sum(bool(src.get("high_trust_by_score")) for src in standard_sources),
        "control_source_count": len(control_sources),
        "clean_or_internal_source_count": len(control_sources),
    }


def _insufficient_reason(reference_status: str) -> str | None:
    return {
        "no_source_reference": "no_source_records",
        "single_source_reference": "single_independent_source",
        "multiple_sources_no_trusted_anchor": "missing_high_trust_anchor",
        "control_only_reference": "control_records_excluded_from_profile",
        "profile_excluded_reference": "current_case_or_repair_records_excluded",
    }.get(reference_status)


def _protocol_metadata_condition_met(context: dict[str, Any], sources: list[dict[str, Any]], spec: dict[str, Any]) -> bool:
    if not spec.get("requires_protocol_metadata"):
        return True
    applicability = context.get("applicability_conditions") or {}
    if not isinstance(applicability, dict):
        applicability = {}
    ctx_keys = {_slug(key) for key, value in {**context, **applicability}.items() if value not in (None, "")}
    has_chain_length = bool({"dp", "degree_of_polymerization", "mn", "repeat_units_per_chain", "target_total_carbons_per_chain"} & ctx_keys)
    has_protocol = bool({"trajectory_protocol", "chain_count", "force_field", "engine_scope", "simulation_protocol"} & ctx_keys)
    if has_chain_length and has_protocol:
        return True
    for source in sources:
        source_keys = {_slug(key) for key, value in source.items() if value not in (None, "")}
        if {"dp", "degree_of_polymerization", "mn"} & source_keys and {"trajectory_protocol", "method", "source_type"} & source_keys:
            return True
    return False


def _conflict_limit(values: list[float], spec: dict[str, Any]) -> float:
    mean_abs = abs(sum(values) / max(len(values), 1))
    limits = []
    if spec.get("conflict_abs") is not None:
        limits.append(float(spec["conflict_abs"]))
    if spec.get("conflict_rel") is not None:
        limits.append(mean_abs * float(spec["conflict_rel"]))
    if not limits:
        limits.append(mean_abs * DEFAULT_RELATIVE_AGREEMENT_TOLERANCE)
    return max(limit for limit in limits if limit >= 0.0)


def _pass_padding(values: list[float], spec: dict[str, Any], conflict_limit: float) -> float:
    mean_abs = abs(sum(values) / max(len(values), 1))
    pads = []
    if spec.get("padding_abs") is not None:
        pads.append(float(spec["padding_abs"]))
    if spec.get("padding_rel") is not None:
        pads.append(mean_abs * float(spec["padding_rel"]))
    if not pads:
        pads.append(conflict_limit / 4.0)
    return max(pads)


def _round_profile_value(value: float) -> float:
    if value == 0:
        return 0.0
    if abs(value) < 1e-3:
        return float(f"{value:.6g}")
    if abs(value) < 1:
        return round(value, 6)
    if abs(value) < 100:
        return round(value, 4)
    return float(f"{value:.6g}")


def _automatic_density_metadata(
    sources: list[dict[str, Any]],
    pass_range: list[float],
) -> dict[str, Any]:
    matched = [
        source
        for source in sources
        if source.get("structure_similarity_score") is not None
        and str(source.get("source_type") or "") == "radonpy"
    ]
    if not matched:
        return {}
    nearest = max(
        matched,
        key=lambda source: float(source.get("structure_similarity_score") or 0.0),
    )
    return {
        "density_screening_range": list(pass_range),
        "density_range_source": "automatic_radonpy_structure_matched_records",
        "density_range_provenance": {
            "reference_count": len(matched),
            "reference_ids": [source.get("source_id") for source in matched],
            "target_temperature_K": nearest.get("temperature_K"),
            "range_generation": (
                "structure-matched RadonPy density bounds with the registered "
                "public-data screening padding"
            ),
        },
        "structure_similarity_score": nearest.get("structure_similarity_score"),
        "structure_similarity_tier": nearest.get("structure_similarity_tier"),
        "nearest_radonpy_reference": (
            nearest.get("nearest_radonpy_reference") or nearest.get("source_id")
        ),
        "structure_similarity_policy": {
            "algorithm": (
                "unhashed sparse atom-pair count fingerprint and count Tanimoto"
            ),
            "high": ">=0.90",
            "medium": "0.80-0.90",
            "distant": "<0.80",
            "condition_checks": "temperature, pressure, melt state and record completeness",
        },
    }


def propose_rule_profile_from_sources(
    context: str | dict[str, Any],
    property_name: str,
    source_evidence: str | list[dict[str, Any]],
    *,
    output_dir: str | Path | None = None,
    agreement_tolerance_g_per_cc: float = DEFAULT_DENSITY_AGREEMENT_TOLERANCE_GCC,
) -> dict[str, Any]:
    """Create a candidate rule profile from independent reference evidence.

    Clean/control records may be attached for repair, diagnosis and re-gating
    provenance, but they are excluded from profile-standard range generation.
    """

    del output_dir  # Proposal is pure; output_dir is accepted for API symmetry.
    # Retained only for API compatibility. Conflict limits now come from the
    # property registry rather than being inferred from a g/cm3 unit string.
    del agreement_tolerance_g_per_cc
    ctx = dict(_parse_jsonish(context))
    prop = canonical_property_name(property_name)
    spec = PROPERTY_SPECS.get(prop)
    if spec is None:
        raise ValueError(f"Unsupported rule-profile property: {property_name!r}")
    sources_raw = _parse_jsonish(source_evidence)
    if not isinstance(sources_raw, list):
        raise ValueError("source_evidence must be a list")
    sources = [normalize_source_evidence(dict(src), prop) for src in sources_raw]
    profile_sources = _profile_standard_sources(sources, ctx)
    excluded_sources = _profile_excluded_sources(sources, ctx)
    control_sources = [src for src in excluded_sources if src.get("profile_exclusion_reason") == "control_source"]
    if not profile_sources:
        evidence_reference_status = _evidence_reference_status(sources, ctx)
        record_counts = _reference_record_counts(sources, ctx)
        insufficient_reason = _insufficient_reason(evidence_reference_status)
        if evidence_reference_status == "control_only_reference":
            activation_status = "blocked_control_only_reference"
        elif evidence_reference_status == "profile_excluded_reference":
            activation_status = "blocked_profile_excluded_reference"
        else:
            activation_status = "blocked_no_source_reference"
        source_manifest = {
            **record_counts,
            "total_record_count": len(sources),
            "profile_source_count": 0,
            "profile_excluded_source_count": len(excluded_sources),
            "current_case_source_excluded_count": sum(
                1
                for src in excluded_sources
                if src.get("profile_exclusion_reason") in {"current_case_or_repair_result", "current_case_reference"}
            ),
            "control_independent_source_count": len({src.get("independent_key") for src in control_sources}),
            "evidence_reference_status": evidence_reference_status,
            "reference_record_status": evidence_reference_status,
            "insufficient_reason": insufficient_reason,
            "minimum_independent_sources_required": 2,
            "high_trust_score_threshold": HIGH_TRUST_SCORE_THRESHOLD,
            "source_trust_interpretation": "heuristic_source_tier_not_statistical_confidence",
            "preferred_sources": sorted(HIGH_TRUST_SOURCE_TYPES),
            "control_source_types": sorted(CONTROL_SOURCE_TYPES),
            "control_source_role": "repair_diagnosis_regate_only",
        }
        profile = {
            "schema_version": RULE_PROFILE_SCHEMA_VERSION,
            "profile_id": str(ctx.get("profile_id") or _profile_id(ctx, prop)),
            "property": prop,
            "profile_kind": spec.get("profile_kind", "public_data_profile"),
            "polymer_family": str(ctx.get("polymer_family") or ""),
            "target_structure_scope": _structure_scope(ctx),
            "force_field": str(ctx.get("force_field") or ""),
            "force_field_policy": str(spec.get("force_field_policy") or "strict"),
            "engine_scope": str(ctx.get("engine_scope") or "any"),
            "engine_scope_policy": str(spec.get("engine_scope_policy") or "strict"),
            "temperature_K": ctx.get("temperature_K"),
            "pressure_bar": ctx.get("pressure_bar"),
            "state": ctx.get("state"),
            "applicability_conditions": ctx.get("applicability_conditions", {}),
            "pass_range": [],
            "warn_range": [],
            "fail_conditions": ["unsupported_topology", "unsupported_force_field", "unsupported_backend"],
            "units": spec.get("units"),
            "source_evidence": [],
            "reference_records": [],
            "control_evidence": control_sources,
            "control_records": control_sources,
            "excluded_evidence": excluded_sources,
            "excluded_reference_records": excluded_sources,
            "source_manifest": source_manifest,
            "confidence_score": 0.0,
            "version": "v1",
            "created_at": _utc_now(),
            "activated_at": None,
            "activated_by": None,
            "frozen_at": None,
            "frozen_by": None,
            "activation_status": activation_status,
            "activation_readiness": _activation_readiness_label(activation_status),
            "profile_activatable": False,
            "conflict_detected": False,
            "activation_checks": {
                "schema_valid": True,
                "units_normalized": True,
                "evidence_condition_met": False,
                "reference_condition_met": False,
                "protocol_metadata_condition_met": _protocol_metadata_condition_met(ctx, [], spec),
                "conflict_detected": False,
                "evidence_reference_status": evidence_reference_status,
                "reference_record_status": evidence_reference_status,
                "insufficient_reason": insufficient_reason,
                "minimum_independent_sources_required": 2,
                "control_sources_excluded_from_profile": len(control_sources),
                "profile_sources_excluded_from_profile": len(excluded_sources),
            },
        }
        profile["profile_hash"] = _hash_payload(profile)
        return profile
    values = [float(src["normalized_value"]) for src in profile_sources]
    single_source_screening = _single_source_screening_eligible(profile_sources)
    spread = max(values) - min(values)
    agreement_tolerance = _conflict_limit(values, spec)
    conflict_detected = spread > agreement_tolerance
    enough_evidence = _evidence_condition_met(profile_sources, ctx)
    evidence_reference_status = _evidence_reference_status(profile_sources, ctx)
    record_counts = _reference_record_counts(sources, ctx)
    insufficient_reason = _insufficient_reason(evidence_reference_status)
    protocol_metadata_ok = _protocol_metadata_condition_met(ctx, profile_sources, spec)
    mean_value = sum(values) / len(values)
    confidence = _confidence_score(profile_sources, spread, agreement_tolerance)
    padding = _pass_padding(values, spec, agreement_tolerance)
    range_values_low = values
    range_values_high = values
    if single_source_screening:
        padding = max(padding, float(spec.get("single_source_screening_padding_abs", agreement_tolerance)))
        range_values_low = [float(profile_sources[0]["normalized_value_min"])]
        range_values_high = [float(profile_sources[0]["normalized_value_max"])]
    pass_low = _round_profile_value(max(0.0, min(range_values_low) - padding))
    pass_high = _round_profile_value(max(range_values_high) + padding)
    warn_padding = max(agreement_tolerance, padding * 2.0)

    activation_status = "eligible_for_machine_freeze"
    if conflict_detected:
        activation_status = "blocked_conflict"
    elif not enough_evidence:
        if evidence_reference_status == "single_source_reference":
            activation_status = "blocked_single_source_reference"
        elif evidence_reference_status == "no_source_reference":
            activation_status = "blocked_no_source_reference"
        elif evidence_reference_status == "multiple_sources_no_trusted_anchor":
            activation_status = "blocked_no_trusted_anchor"
        else:
            activation_status = "blocked_insufficient_evidence"
    elif not protocol_metadata_ok:
        activation_status = "blocked_insufficient_protocol_metadata"

    profile = {
        "schema_version": RULE_PROFILE_SCHEMA_VERSION,
        "profile_id": str(ctx.get("profile_id") or _profile_id(ctx, prop)),
        "property": prop,
        "profile_kind": "public_data_screening_profile" if single_source_screening else spec.get("profile_kind", "public_data_profile"),
        "polymer_family": str(ctx.get("polymer_family") or ""),
        "target_structure_scope": _structure_scope(ctx),
        "force_field": str(ctx.get("force_field") or ""),
        "force_field_policy": "broad_sanity_check" if single_source_screening else str(spec.get("force_field_policy") or "strict"),
        "engine_scope": str(ctx.get("engine_scope") or "any"),
        "engine_scope_policy": "broad_sanity_check" if single_source_screening else str(spec.get("engine_scope_policy") or "strict"),
        "temperature_K": ctx.get("temperature_K"),
        "pressure_bar": ctx.get("pressure_bar"),
        "state": ctx.get("state"),
        "applicability_conditions": ctx.get("applicability_conditions", {}),
        "pass_range": [pass_low, pass_high],
        "warn_range": [_round_profile_value(max(0.0, pass_low - warn_padding)), _round_profile_value(pass_high + warn_padding)],
        "fail_conditions": ["unsupported_topology", "unsupported_force_field", "unsupported_backend"],
        "units": spec.get("units"),
        "source_evidence": profile_sources,
        "reference_records": profile_sources,
        "control_evidence": control_sources,
        "control_records": control_sources,
        "excluded_evidence": excluded_sources,
        "excluded_reference_records": excluded_sources,
        "source_manifest": {
            **record_counts,
            "total_record_count": len(sources),
            "profile_source_count": len(profile_sources),
            "profile_excluded_source_count": len(excluded_sources),
            "current_case_source_excluded_count": sum(
                1
                for src in excluded_sources
                if src.get("profile_exclusion_reason") in {"current_case_or_repair_result", "current_case_reference"}
            ),
            "control_independent_source_count": len({src.get("independent_key") for src in control_sources}),
            "mean_normalized_value": round(mean_value, 5),
            "spread": round(spread, 8),
            "agreement_tolerance": round(agreement_tolerance, 8),
            "agreement_tolerance_units": spec.get("units"),
            "evidence_reference_status": evidence_reference_status,
            "reference_record_status": evidence_reference_status,
            "insufficient_reason": insufficient_reason,
            "minimum_independent_sources_required": 1 if single_source_screening else 2,
            "single_source_screening_fallback": single_source_screening,
            "single_source_screening_padding": padding if single_source_screening else None,
            "high_trust_score_threshold": HIGH_TRUST_SCORE_THRESHOLD,
            "source_trust_interpretation": "heuristic_source_tier_not_statistical_confidence",
            "preferred_sources": sorted(HIGH_TRUST_SOURCE_TYPES),
            "control_source_types": sorted(CONTROL_SOURCE_TYPES),
            "control_source_role": "repair_diagnosis_regate_only",
        },
        "confidence_score": min(confidence, 0.70) if single_source_screening else confidence,
        "version": "v1",
        "created_at": _utc_now(),
        "activated_at": None,
        "activated_by": None,
        "frozen_at": None,
        "frozen_by": None,
        "activation_status": activation_status,
        "activation_readiness": _activation_readiness_label(activation_status),
        "profile_activatable": activation_status == "eligible_for_machine_freeze",
        "conflict_detected": conflict_detected,
        "activation_checks": {
            "schema_valid": True,
            "units_normalized": True,
            "evidence_condition_met": enough_evidence,
            "reference_condition_met": enough_evidence,
            "protocol_metadata_condition_met": protocol_metadata_ok,
            "conflict_detected": conflict_detected,
            "evidence_reference_status": evidence_reference_status,
            "reference_record_status": evidence_reference_status,
            "insufficient_reason": insufficient_reason,
            "minimum_independent_sources_required": 2,
            "control_sources_excluded_from_profile": len(control_sources),
            "profile_sources_excluded_from_profile": len(excluded_sources),
        },
    }
    if prop == "density":
        profile.update(
            _automatic_density_metadata(
                profile_sources,
                [pass_low, pass_high],
            )
        )
    profile["profile_hash"] = _hash_payload(profile)
    return profile


def propose_similarity_transfer_profile(
    context: str | dict[str, Any],
    property_name: str,
    similarity_result: str | dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    agreement_tolerance_g_per_cc: float = DEFAULT_DENSITY_AGREEMENT_TOLERANCE_GCC,
) -> dict[str, Any]:
    """Create an estimated rule profile from deterministic similar references.

    The transfer packet must come from ``structure_similarity_transfer``.  This
    helper only reshapes accepted candidates into ordinary reference records; it
    does not relax conflict, independence, compatibility or confidence checks.
    """

    result = dict(_parse_jsonish(similarity_result))
    prop = canonical_property_name(property_name)
    if result.get("status") != "similarity_transfer_candidate":
        blocked_status = f"blocked_{result.get('status') or 'no_similarity_candidate'}"
        profile = {
            "schema_version": RULE_PROFILE_SCHEMA_VERSION,
            "profile_id": _profile_id(dict(_parse_jsonish(context)), prop),
            "property": prop,
            "profile_kind": "similarity_transfer_profile",
            "rule_basis": "similarity_transfer",
            "activation_status": blocked_status,
            "activation_readiness": blocked_status,
            "profile_activatable": False,
            "conflict_detected": False,
            "confidence_score": 0.0,
            "similarity_transfer": {
                "status": result.get("status"),
                "distance_metric_version": result.get("distance_metric_version"),
                "min_similarity": result.get("min_similarity"),
                "high_similarity": result.get("high_similarity"),
                "similarity_reference_id": result.get("similarity_reference_id"),
                "similarity_score": result.get("similarity_score"),
                "similarity_tier": result.get("similarity_tier"),
                "similarity_distance_reason": result.get("similarity_distance_reason"),
                "review_flag": True,
            },
        }
        profile["profile_hash"] = _hash_payload(profile)
        return profile

    sources: list[dict[str, Any]] = []
    accepted_candidates = [
        dict(candidate)
        for candidate in result.get("candidates", []) or []
        if candidate.get("status") == "similarity_transfer_candidate"
    ]
    if not accepted_candidates and isinstance(result.get("best_candidate"), dict):
        accepted_candidates = [dict(result["best_candidate"])]
    for candidate in accepted_candidates:
        if candidate.get("reference_value") in (None, "") or not candidate.get("reference_units"):
            continue
        sources.append(
            {
                "source_id": candidate.get("reference_id"),
                "independent_key": candidate.get("independent_key") or candidate.get("reference_id"),
                "property": prop,
                "value": candidate.get("reference_value"),
                "units": candidate.get("reference_units"),
                "source_type": candidate.get("source_type") or "literature",
                "source_provider_kind": candidate.get("source_provider_kind") or "similarity_transfer",
                "state": candidate.get("reference_state"),
                "temperature_K": candidate.get("reference_temperature_K"),
                "force_field": candidate.get("reference_force_field"),
                "extraction_confidence": candidate.get("extraction_confidence", 0.85),
                "similarity_transfer_source": True,
                "similarity_reference_id": candidate.get("reference_id"),
                "similarity_score": candidate.get("similarity_score"),
                "similarity_tier": candidate.get("similarity_tier"),
                "similarity_distance_reason": candidate.get("similarity_distance_reason"),
                "structure_comparison_level": candidate.get("structure_comparison_level"),
                "structure_evidence_rank": candidate.get("structure_evidence_rank"),
                "smiles_fields_used": candidate.get("smiles_fields_used"),
                "phase_check_status": candidate.get("phase_check_status"),
                "distance_metric_version": result.get("distance_metric_version"),
                "transfer_uncertainty_rel": candidate.get("u_similarity_rel"),
                "repeat_unit_structure_method": (candidate.get("similarity") or {}).get("repeat_unit_structure_method"),
                "machine_activation_eligible": bool(
                    candidate.get("machine_activation_eligible", candidate.get("machine_freeze_eligible"))
                ),
                "machine_activation_reasons": list(
                    candidate.get("machine_activation_reasons") or candidate.get("machine_freeze_reasons") or []
                ),
                "machine_activation_policy_version": candidate.get("machine_activation_policy_version")
                or candidate.get("machine_freeze_policy_version"),
                "machine_freeze_eligible": bool(
                    candidate.get("machine_activation_eligible", candidate.get("machine_freeze_eligible"))
                ),
                "machine_freeze_reasons": list(
                    candidate.get("machine_activation_reasons") or candidate.get("machine_freeze_reasons") or []
                ),
                "machine_freeze_policy_version": candidate.get("machine_activation_policy_version")
                or candidate.get("machine_freeze_policy_version"),
            }
        )

    if not sources:
        profile = {
            "schema_version": RULE_PROFILE_SCHEMA_VERSION,
            "profile_id": _profile_id(dict(_parse_jsonish(context)), prop),
            "property": prop,
            "profile_kind": "similarity_transfer_profile",
            "rule_basis": "similarity_transfer",
            "activation_status": "blocked_similarity_missing_reference_value",
            "activation_readiness": "blocked_similarity_missing_reference_value",
            "profile_activatable": False,
            "conflict_detected": False,
            "confidence_score": 0.0,
            "similarity_transfer": {
                "status": result.get("status"),
                "distance_metric_version": result.get("distance_metric_version"),
                "min_similarity": result.get("min_similarity"),
                "high_similarity": result.get("high_similarity"),
                "similarity_reference_id": result.get("similarity_reference_id"),
                "similarity_score": result.get("similarity_score"),
                "similarity_tier": result.get("similarity_tier"),
                "similarity_distance_reason": "missing_reference_value",
                "review_flag": True,
            },
        }
        profile["profile_hash"] = _hash_payload(profile)
        return profile

    profile = propose_rule_profile_from_sources(
        context,
        prop,
        sources,
        output_dir=output_dir,
        agreement_tolerance_g_per_cc=agreement_tolerance_g_per_cc,
    )
    scores = [float(src.get("similarity_score", 0.0)) for src in sources]
    score_min = min(scores) if scores else 0.0
    score_mean = sum(scores) / len(scores) if scores else 0.0
    u_rel_values = [float(src.get("transfer_uncertainty_rel", 0.0)) for src in sources if src.get("transfer_uncertainty_rel") not in (None, "")]
    profile["profile_kind"] = "similarity_transfer_profile"
    profile["rule_basis"] = "similarity_transfer"
    profile["similarity_transfer"] = {
        "status": result.get("status"),
        "schema_version": result.get("schema_version"),
        "distance_metric_version": result.get("distance_metric_version"),
        "min_similarity": result.get("min_similarity"),
        "high_similarity": result.get("high_similarity"),
        "similarity_reference_id": result.get("similarity_reference_id"),
        "similarity_reference_ids": [src.get("similarity_reference_id") for src in sources],
        "similarity_score": round(score_mean, 4),
        "similarity_score_min": round(score_min, 4),
        "similarity_tier": result.get("similarity_tier"),
        "similarity_distance_reason": result.get("similarity_distance_reason"),
        "u_similarity_rel": round(max(u_rel_values), 4) if u_rel_values else result.get("u_similarity_rel"),
        "condition_compatibility": (result.get("best_candidate") or {}).get("similarity", {}).get("condition_compatibility"),
        "structure_comparison_level": result.get("structure_comparison_level"),
        "structure_evidence_rank": result.get("structure_evidence_rank"),
        "smiles_fields_used": result.get("smiles_fields_used"),
        "phase_check_status": result.get("phase_check_status"),
        "reference_count": len(sources),
        "machine_activation_policy_version": result.get("machine_activation_policy_version")
        or result.get("machine_freeze_policy_version"),
        "machine_activation_eligible": bool(
            result.get("machine_activation_eligible", result.get("machine_freeze_eligible"))
        ),
        "machine_activation_eligible_reference_count": int(
            result.get("machine_activation_eligible_reference_count", result.get("machine_freeze_eligible_reference_count", 0))
            or 0
        ),
        "machine_activation_reasons": list(
            result.get("machine_activation_reasons") or result.get("machine_freeze_reasons") or []
        ),
        "machine_freeze_policy_version": result.get("machine_activation_policy_version")
        or result.get("machine_freeze_policy_version"),
        "machine_freeze_eligible": bool(
            result.get("machine_activation_eligible", result.get("machine_freeze_eligible"))
        ),
        "machine_freeze_eligible_reference_count": int(
            result.get("machine_activation_eligible_reference_count", result.get("machine_freeze_eligible_reference_count", 0))
            or 0
        ),
        "machine_freeze_reasons": list(
            result.get("machine_activation_reasons") or result.get("machine_freeze_reasons") or []
        ),
        "repeat_unit_structure_methods": sorted(
            {str(src.get("repeat_unit_structure_method") or "") for src in sources if src.get("repeat_unit_structure_method")}
        ),
        "review_flag": not bool(result.get("machine_activation_eligible", result.get("machine_freeze_eligible"))),
    }
    profile["source_manifest"]["similarity_transfer_source_count"] = len(sources)
    profile["source_manifest"]["similarity_score_min"] = round(score_min, 4)
    profile["source_manifest"]["similarity_score_mean"] = round(score_mean, 4)
    profile["source_manifest"]["distance_metric_version"] = result.get("distance_metric_version")
    profile["source_manifest"]["machine_activation_policy_version"] = result.get("machine_activation_policy_version") or result.get(
        "machine_freeze_policy_version"
    )
    profile["source_manifest"]["machine_activation_eligible_reference_count"] = int(
        result.get("machine_activation_eligible_reference_count", result.get("machine_freeze_eligible_reference_count", 0))
        or 0
    )
    profile["source_manifest"]["machine_freeze_policy_version"] = profile["source_manifest"][
        "machine_activation_policy_version"
    ]
    profile["source_manifest"]["machine_freeze_eligible_reference_count"] = profile["source_manifest"][
        "machine_activation_eligible_reference_count"
    ]
    min_similarity = float(result.get("min_similarity", 0.80))
    high_similarity = float(result.get("high_similarity", 0.90))
    machine_activation_eligible = bool(result.get("machine_activation_eligible", result.get("machine_freeze_eligible")))
    if profile["activation_status"] == "eligible_for_machine_freeze" and not machine_activation_eligible:
        profile["activation_status"] = "blocked_similarity_warn"
        profile["activation_checks"]["similarity_machine_freeze_policy_met"] = False
    else:
        profile["activation_checks"]["similarity_machine_freeze_policy_met"] = machine_activation_eligible
    profile["activation_checks"]["similarity_machine_activation_policy_met"] = machine_activation_eligible
    if profile["activation_status"] == "eligible_for_machine_freeze" and score_min < high_similarity:
        profile["activation_status"] = "blocked_similarity_not_high_confidence"
        profile["activation_checks"]["similarity_high_threshold_met"] = False
    else:
        profile["activation_checks"]["similarity_high_threshold_met"] = score_min >= high_similarity
    if profile["activation_status"] == "eligible_for_machine_freeze" and score_min < min_similarity:
        profile["activation_status"] = "blocked_similarity_distance_too_large"
        profile["activation_checks"]["similarity_distance_met"] = False
    else:
        profile["activation_checks"]["similarity_distance_met"] = score_min >= min_similarity
    profile["activation_checks"]["similarity_condition_met"] = result.get("similarity_distance_reason") == "matched"
    profile["activation_readiness"] = _activation_readiness_label(str(profile["activation_status"]))
    profile["profile_activatable"] = profile["activation_status"] == "eligible_for_machine_freeze"
    profile["profile_hash"] = _hash_payload(profile)
    return profile


def _active_profiles_path(output_dir: Path) -> Path:
    return output_dir / "active_rule_profile.json"


def _profile_path(output_dir: Path, profile_id: str) -> Path:
    return output_dir / "profiles" / f"{profile_id}.json"


def _freeze_ledger_path(output_dir: Path) -> Path:
    """Legacy ledger path retained for backward-compatible readers."""

    return output_dir / "rule_freeze_ledger.jsonl"


def _activation_ledger_path(output_dir: Path) -> Path:
    return output_dir / "rule_activation_ledger.jsonl"


def _source_pack_path(output_dir: Path, profile_id: str) -> Path:
    return output_dir / "reference_records" / f"{profile_id}_reference_record_pack.jsonl"


def _validate_profile_for_activation(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("schema_version", "profile_id", "property", "polymer_family", "force_field", "pass_range", "warn_range", "units"):
        if key not in profile or profile.get(key) in (None, ""):
            errors.append(f"missing required profile field: {key}")
    if profile.get("schema_version") != RULE_PROFILE_SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {profile.get('schema_version')!r}")
    activatable = bool(
        profile.get("profile_activatable", profile.get("activation_status") == "eligible_for_machine_freeze")
    )
    if not activatable:
        errors.append(
            f"profile is not eligible for activation: {profile.get('activation_readiness', profile.get('activation_status'))}"
        )
    if profile.get("conflict_detected"):
        errors.append("conflicting reference records are present")
    if float(profile.get("confidence_score", 0.0)) < 0.65:
        errors.append("confidence_score is below activation threshold 0.65")
    if profile.get("profile_kind") == "similarity_transfer_profile":
        transfer = profile.get("similarity_transfer") or {}
        if transfer.get("distance_metric_version") in (None, ""):
            errors.append("similarity_transfer_profile is missing distance_metric_version")
        if transfer.get("similarity_reference_id") in (None, ""):
            errors.append("similarity_transfer_profile is missing similarity_reference_id")
        high_similarity = float(transfer.get("high_similarity", 0.90) or 0.90)
        if float(transfer.get("similarity_score_min", transfer.get("similarity_score", 0.0)) or 0.0) < high_similarity:
            errors.append(f"similarity score is below machine-activation threshold {high_similarity:.2f}")
        if transfer.get("similarity_distance_reason") != "matched":
            errors.append(f"similarity transfer compatibility failed: {transfer.get('similarity_distance_reason')}")
        activation_policy_version = transfer.get("machine_activation_policy_version") or transfer.get(
            "machine_freeze_policy_version"
        )
        if activation_policy_version in (None, ""):
            errors.append("similarity transfer is missing machine-activation policy provenance")
        activation_eligible = bool(
            transfer.get("machine_activation_eligible", transfer.get("machine_freeze_eligible"))
        )
        if not activation_eligible:
            errors.append("similarity transfer did not pass the strict machine-activation policy")
        activation_reference_count = int(
            transfer.get(
                "machine_activation_eligible_reference_count",
                transfer.get("machine_freeze_eligible_reference_count", 0),
            )
            or 0
        )
        if activation_reference_count < 2:
            errors.append("similarity transfer requires at least two independent eligible references")
        if set(transfer.get("repeat_unit_structure_methods") or []) != {"morgan_tanimoto"}:
            errors.append("all transferred references must use Morgan/Tanimoto chemical fingerprints")
        if bool(transfer.get("review_flag")):
            errors.append("similarity transfer candidate is marked for review")
    return errors


def activate_rule_profile(
    profile: str | dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    actor: str = "codex",
    reason: str = "",
) -> dict[str, Any]:
    """Persist and register an eligible rule profile as active."""

    out_dir = Path(output_dir) if output_dir else DEFAULT_RULE_PROFILE_DIR
    prof = dict(_parse_jsonish(profile))
    errors = _validate_profile_for_activation(prof)
    if errors:
        return {"status": "rejected", "errors": errors, "profile_id": prof.get("profile_id")}

    frozen = dict(prof)
    activated_at = _utc_now()
    activated_by = "machine_activated_by_codex" if _slug(actor) == "codex" else str(actor)
    frozen["activated_at"] = activated_at
    frozen["activated_by"] = activated_by
    frozen["activation_reason"] = reason
    # Compatibility aliases for historical profiles and downstream readers.
    frozen["frozen_at"] = activated_at
    frozen["frozen_by"] = activated_by
    frozen["freeze_reason"] = reason
    frozen["activation_status"] = "active"
    frozen["activation_readiness"] = "active"
    frozen["profile_activatable"] = True
    frozen["profile_hash"] = _hash_payload({k: v for k, v in frozen.items() if k != "profile_hash"})

    profile_id = str(frozen["profile_id"])
    profile_path = _profile_path(out_dir, profile_id)
    _write_json(profile_path, frozen)

    active = _read_json(_active_profiles_path(out_dir), {"schema_version": RULE_PROFILE_SCHEMA_VERSION, "profiles": []})
    profiles = [item for item in active.get("profiles", []) if item.get("profile_id") != profile_id]
    profiles.append(
        {
            "profile_id": profile_id,
            "property": frozen.get("property"),
            "profile_kind": frozen.get("profile_kind"),
            "polymer_family": frozen.get("polymer_family"),
            "target_structure_scope": frozen.get("target_structure_scope"),
            "force_field": frozen.get("force_field"),
            "force_field_policy": frozen.get("force_field_policy", "strict"),
            "engine_scope": frozen.get("engine_scope"),
            "engine_scope_policy": frozen.get("engine_scope_policy"),
            "state": frozen.get("state", "melt"),
            "temperature_K": frozen.get("temperature_K"),
            "temperature_tolerance_K": frozen.get("temperature_tolerance_K", 25.0),
            "pressure_bar": frozen.get("pressure_bar"),
            "pressure_tolerance_bar": frozen.get("pressure_tolerance_bar", 1.0),
            "applicability_conditions": frozen.get("applicability_conditions"),
            "path": str(profile_path),
            "profile_hash": frozen.get("profile_hash"),
            "activated_at": frozen.get("activated_at"),
            "status": "active",
        }
    )
    active = {"schema_version": RULE_PROFILE_SCHEMA_VERSION, "updated_at": _utc_now(), "profiles": profiles}
    _write_json(_active_profiles_path(out_dir), active)

    reference_record_pack = _source_pack_path(out_dir, profile_id)
    for source in frozen.get("reference_records", frozen.get("source_evidence", [])) or []:
        _append_jsonl(
            reference_record_pack,
            {"profile_id": profile_id, "record_role": "profile_standard_reference", "reference_record": source},
        )
    for source in frozen.get("excluded_reference_records", frozen.get("excluded_evidence", [])) or []:
        reason = source.get("profile_exclusion_reason")
        role = (
            "control_source_excluded_from_profile"
            if reason == "control_source"
            else "current_case_source_excluded_from_profile"
        )
        _append_jsonl(
            reference_record_pack,
            {
                "profile_id": profile_id,
                "record_role": role,
                "profile_exclusion_reason": reason,
                "reference_record": source,
            },
        )

    ledger_row = {
        "timestamp_utc": _utc_now(),
        "event": "activate_rule_profile",
        "status": "ok",
        "profile_id": profile_id,
        "actor": actor,
        "activated_by": frozen["activated_by"],
        "frozen_by": frozen["frozen_by"],
        "reason": reason,
        "profile_path": str(profile_path),
        "active_profile_path": str(_active_profiles_path(out_dir)),
        "reference_record_pack_path": str(reference_record_pack),
        # Backward-compatible ledger alias for historical readers.
        "evidence_pack_path": str(reference_record_pack),
        "profile_hash": frozen.get("profile_hash"),
    }
    _append_jsonl(_activation_ledger_path(out_dir), ledger_row)
    _append_jsonl(_freeze_ledger_path(out_dir), ledger_row)
    return {
        "status": "ok",
        "profile_id": profile_id,
        "affected_path": str(profile_path),
        "active_profile_path": str(_active_profiles_path(out_dir)),
        "activation_ledger_path": str(_activation_ledger_path(out_dir)),
    }


def radonpy_pe_lcb_density_profile(
    reference_pack_path: str | Path = DEFAULT_REFERENCE_PACK,
) -> dict[str, Any]:
    """Build the PE-LCB100 density profile from the public RadonPy pack."""

    reference_pack = load_reference_pack(reference_pack_path)
    references = list(reference_pack["records"])
    density_range = reference_pack["density_range"]
    pass_range = [float(value) for value in density_range["pass_range_g_cm3"]]
    source_evidence = [
        {
            "source_id": f"radonpy_pi1070_{record['reference_id']}_400K",
            "source_type": "radonpy",
            "independent_key": f"radonpy_pi1070_{record['reference_id']}",
            "normalized_value": record["density_400K_g_cm3"],
            "normalized_range": [
                record["density_400K_low_g_cm3"],
                record["density_400K_high_g_cm3"],
            ],
            "normalized_units": "g/cm3",
            "temperature_K": 400.0,
            "repeat_unit_smiles": record["repeat_unit_smiles"],
            "force_field": record["force_field"],
            "engine_scope": record["calculation_software"],
            "source_trust_score": 0.9,
            "extraction_confidence": 1.0,
            "role": "public_polyolefin_density_screening_record",
        }
        for record in references
    ]
    profile = {
        "schema_version": RULE_PROFILE_SCHEMA_VERSION,
        # The historical identifier is retained so archived readers keep working.
        "profile_id": "pe_lcb_c100_oplsaa_density_lower_bound",
        "property": "density",
        "profile_kind": "radonpy_public_density_screening_profile",
        "polymer_family": "PE-LCB100",
        "force_field": "opls-aa",
        "force_field_policy": "strict",
        "engine_scope": "any",
        "engine_scope_policy": "engine_family_scoped",
        "temperature_K": 400.0,
        "temperature_tolerance_K": 25.0,
        "pressure_bar": 1.0,
        "pressure_tolerance_bar": 1.0,
        "state": "melt",
        "applicability_conditions": {
            "carbon_budget": "approximately C100",
            "architecture": "long-chain-branched polyethylene",
            "note": (
                "The interval screens PE-LCB melt densities near 400 K and 1 bar. "
                "It is derived only from eligible RadonPy PI1070 records."
            ),
        },
        "pass_range": pass_range,
        "warn_range": [0.0, 10.0],
        "density_screening_range": pass_range,
        "density_range_source": "radonpy_pi1070_saturated_acyclic_40_record_envelope_400K",
        "density_range_provenance": {
            "name": "RadonPy PI1070 saturated acyclic hydrocarbon records",
            "reference_count": len(references),
            "source_url": reference_pack["source"]["url"],
            "source_sha256": reference_pack["source"]["sha256"],
            "raw_range_g_cm3": density_range["raw_range_g_cm3"],
            "rounding": density_range["rounding"],
            "target_temperature_K": 400.0,
        },
        "structure_similarity_policy": {
            "algorithm": "unhashed sparse atom-pair count fingerprint and count Tanimoto",
            "reference_expansion": "expand each repeat unit to the closest target carbon count",
            "high": ">=0.90",
            "medium": "0.80-0.90",
            "distant": "<0.80",
            "effect_on_density_range": "none",
        },
        "fail_conditions": ["unsupported_topology", "unsupported_force_field", "unsupported_backend"],
        "units": "g/cm3",
        "source_evidence": source_evidence,
        "reference_records": references,
        "control_evidence": [],
        "control_records": [],
        "source_manifest": {
            "source_count": len(references),
            "total_record_count": len(references),
            "profile_source_count": len(references),
            "control_source_count": 0,
            "independent_source_count": len(references),
            "control_independent_source_count": 0,
            "high_trust_source_count": len(references),
            "evidence_reference_status": "radonpy_public_dataset_40_records",
            "reference_record_status": "radonpy_public_dataset_40_records",
            "insufficient_reason": None,
            "preferred_sources": ["radonpy"],
            "control_source_types": [],
            "control_source_role": "excluded_from_density_range_derivation",
        },
        "confidence_score": 0.95,
        "version": "v3_radonpy_40_record_envelope",
        "created_at": _utc_now(),
        "activated_at": None,
        "activated_by": None,
        "frozen_at": None,
        "frozen_by": None,
        "activation_status": "eligible_for_machine_freeze",
        "activation_readiness": "eligible_for_activation",
        "profile_activatable": True,
        "conflict_detected": False,
        "activation_checks": {
            "schema_valid": True,
            "units_normalized": True,
            "evidence_condition_met": True,
            "evidence_reference_status": "radonpy_public_dataset_40_records",
            "control_reference_exception": False,
            "conflict_detected": False,
        },
    }
    profile["profile_hash"] = _hash_payload(profile)
    return profile


def legacy_pe_lcb_density_profile() -> dict[str, Any]:
    """Compatibility alias for historical callers."""

    return radonpy_pe_lcb_density_profile()


def install_radonpy_pe_lcb_density_profile(
    *,
    output_dir: str | Path | None = None,
    reason: str = "install RadonPy PE-LCB100 density screening profile",
) -> dict[str, Any]:
    return activate_rule_profile(
        radonpy_pe_lcb_density_profile(),
        output_dir=output_dir,
        actor="codex",
        reason=reason,
    )


def install_legacy_pe_lcb_density_profile(
    *,
    output_dir: str | Path | None = None,
    reason: str = "install PE-LCB100 density rule profile",
) -> dict[str, Any]:
    """Compatibility alias that installs the current RadonPy-only profile."""

    return install_radonpy_pe_lcb_density_profile(output_dir=output_dir, reason=reason)


def _profile_matches(profile_ref: dict[str, Any], context: dict[str, Any], property_name: str) -> bool:
    if canonical_property_name(profile_ref.get("property")) != canonical_property_name(property_name):
        return False
    profile_scope = profile_ref.get("target_structure_scope") if isinstance(profile_ref.get("target_structure_scope"), dict) else {}
    profile_scope_fields = profile_scope.get("fields") if isinstance(profile_scope.get("fields"), dict) else {}
    if profile_scope_fields:
        context_scope = _structure_scope(context)
        context_scope_fields = context_scope.get("fields") if isinstance(context_scope.get("fields"), dict) else {}
        if not context_scope_fields:
            return False
        for key, expected in profile_scope_fields.items():
            if context_scope_fields.get(key) != expected:
                return False
    else:
        context_family = _slug(context.get("polymer_family"))
        profile_family = _slug(profile_ref.get("polymer_family"))
        if context_family != profile_family:
            return False
    context_state = _slug(context.get("state") or "melt")
    profile_state = _slug(profile_ref.get("state") or "melt")
    if context_state != "melt" or profile_state != "melt":
        return False
    profile_temperature = profile_ref.get("temperature_K")
    context_temperature = context.get("temperature_K")
    if profile_temperature not in (None, "") and context_temperature not in (None, ""):
        temperature_tolerance = float(profile_ref.get("temperature_tolerance_K", 25.0) or 25.0)
        if abs(float(profile_temperature) - float(context_temperature)) > temperature_tolerance:
            return False
    profile_pressure = profile_ref.get("pressure_bar")
    context_pressure = context.get("pressure_bar")
    if profile_pressure not in (None, "") and context_pressure not in (None, ""):
        pressure_tolerance = float(profile_ref.get("pressure_tolerance_bar", 1.0) or 1.0)
        if abs(float(profile_pressure) - float(context_pressure)) > pressure_tolerance:
            return False
    if not _force_field_compatible(profile_ref, context, property_name):
        return False
    if not _engine_scope_compatible(profile_ref, context, property_name):
        return False
    return True


def select_rule_profile(
    context: str | dict[str, Any],
    property_name: str,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    out_dir = Path(output_dir) if output_dir else DEFAULT_RULE_PROFILE_DIR
    ctx = dict(_parse_jsonish(context))
    active = _read_json(_active_profiles_path(out_dir), {"profiles": []})
    for ref in active.get("profiles", []) or []:
        if not _profile_matches(ref, ctx, property_name):
            continue
        path = Path(str(ref.get("path") or ""))
        if not path.is_absolute():
            path = out_dir / path
        profile = _read_json(path, None)
        if isinstance(profile, dict):
            return profile
    return None


def _extract_density_gcc(evidence: dict[str, Any]) -> float | None:
    for key in ("rho_final_gcc_si_o", "rho_final_gcc", "final_density_g_per_cc", "density_g_per_cc", "density_gcc"):
        if key in evidence and evidence.get(key) not in (None, ""):
            return float(evidence[key])
    return None


def _extract_property_value(evidence: dict[str, Any], property_name: str) -> float | None:
    prop = canonical_property_name(property_name)
    spec = PROPERTY_SPECS.get(prop)
    if spec is None:
        return None
    for key in spec.get("value_keys", []):
        if key not in evidence or evidence.get(key) in (None, ""):
            continue
        units = (
            evidence.get("units")
            or evidence.get(f"{key}_units")
            or evidence.get(f"{prop}_units")
            or spec.get("units")
        )
        return normalize_property_value(evidence[key], str(units), prop)
    if prop == "density":
        return _extract_density_gcc(evidence)
    if "value" in evidence and evidence.get("value") not in (None, ""):
        units = evidence.get("units") or spec.get("units")
        return normalize_property_value(evidence["value"], str(units), prop)
    return None


def _density_screening_metadata(profile: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if not profile.get("density_screening_range"):
        return {}
    metadata: dict[str, Any] = {
        "structure_similarity_score": (
            context.get("structure_similarity_score")
            if context.get("structure_similarity_score") is not None
            else profile.get("structure_similarity_score")
        ),
        "structure_similarity_tier": (
            context.get("structure_similarity_tier")
            or profile.get("structure_similarity_tier")
        ),
        "nearest_radonpy_reference": (
            context.get("nearest_radonpy_reference")
            or profile.get("nearest_radonpy_reference")
        ),
        "density_screening_range": profile.get("density_screening_range", profile.get("pass_range")),
        "density_range_source": profile.get("density_range_source"),
        "density_range_provenance": profile.get("density_range_provenance"),
    }
    if metadata["structure_similarity_score"] is not None:
        return metadata
    if profile.get("profile_kind") != "radonpy_public_density_screening_profile":
        return metadata
    structure = context.get("monomer_smiles") or context.get("chain_smiles")
    if structure:
        try:
            match = match_case_to_references(
                context,
                profile.get("reference_records") or [],
                density_range=profile.get("density_screening_range", profile.get("pass_range")),
            )
            metadata.update({key: match.get(key) for key in (
                "structure_similarity_score",
                "structure_similarity_tier",
                "nearest_radonpy_reference",
                "density_screening_range",
                "density_range_source",
            )})
            return metadata
        except (TypeError, ValueError):
            pass
    case_id = str(context.get("case_id") or "")
    if case_id and DEFAULT_CASE_MATCH_PACK.is_file():
        try:
            match_pack = load_case_match_pack()
            match = next(
                (item for item in match_pack.get("matches", []) if item.get("case_id") == case_id),
                None,
            )
            if match:
                metadata.update({key: match.get(key) for key in (
                    "structure_similarity_score",
                    "structure_similarity_tier",
                    "nearest_radonpy_reference",
                    "density_screening_range",
                    "density_range_source",
                )})
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return metadata


def apply_rule_gate(
    evidence: str | dict[str, Any],
    *,
    context: str | dict[str, Any],
    property_name: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Apply an active profile to a result record and return PASS/WARN/FAIL."""

    ctx = dict(_parse_jsonish(context))
    ev = dict(_parse_jsonish(evidence))
    prop = canonical_property_name(property_name)
    policy = rule_profile_automation_policy(prop)
    computed_value = _extract_property_value(ev, prop)
    profile = select_rule_profile(ctx, prop, output_dir=output_dir)
    if profile is None:
        return {
            "status": "ok",
            "rule_verdict": "WARN",
            "audit_required": True,
            "warn_source": policy["missing_warn_source"],
            "profile_id": None,
            "property": prop,
            "polymer_type": str(ctx.get("polymer_type") or ctx.get("polymer_family") or ""),
            "condition": {
                "temperature_K": ctx.get("temperature_K"),
                "pressure_bar": ctx.get("pressure_bar"),
                "state": ctx.get("state"),
            },
            "force_field": str(ctx.get("force_field") or ""),
            "engine": str(ctx.get("engine") or ctx.get("engine_scope") or ""),
            "computed_value": computed_value,
            "units": (PROPERTY_SPECS.get(prop) or {}).get("units"),
            "rule_basis": "none",
            "missing_reason": "no_frozen_profile_match"
            if policy["profile_freezable"]
            else policy["fallback_status"],
            "automation_level": policy["automation_level"],
            "calibration_route": "auto_rule_profile_calibration"
            if policy["profile_freezable"]
            else "quality_or_audit_only",
            "fallback_status": policy["fallback_status"],
            "reason": (
                "No active rule profile matched polymer_family/property/force_field/engine_scope; "
                "automatic calibration may run only through structured reference records and deterministic activation checks."
            ),
        }
    density_metadata = _density_screening_metadata(profile, ctx) if prop == "density" else {}
    value = computed_value
    if value is None:
        return {
            "status": "ok",
            "rule_verdict": "WARN",
            "audit_required": True,
            "warn_source": "evidence_incomplete_warn",
            "profile_id": profile.get("profile_id"),
            "profile_kind": profile.get("profile_kind"),
            "rule_basis": "similarity_transfer"
            if profile.get("profile_kind") == "similarity_transfer_profile"
            else "direct_profile",
            "property": prop,
            "reason": f"{prop} result data are missing.",
            **density_metadata,
        }

    pass_low, pass_high = [float(v) for v in profile["pass_range"]]
    warn_low, warn_high = [float(v) for v in profile["warn_range"]]
    if pass_low <= value <= pass_high:
        verdict = "PASS"
        warn_source = None
        reason = f"{prop} value falls inside the active profile pass_range"
    elif warn_low <= value <= warn_high:
        verdict = "WARN"
        warn_source = f"{prop}_warn"
        reason = f"{prop} value falls outside pass_range but inside the active profile warn_range"
    else:
        verdict = "WARN"
        warn_source = f"{prop}_warn"
        reason = f"{prop} value falls outside the active profile warn_range; route to audit diagnosis before any action"
    transfer = profile.get("similarity_transfer") or {}
    rule_basis = "similarity_transfer" if profile.get("profile_kind") == "similarity_transfer_profile" else "direct_profile"
    return {
        "status": "ok",
        "rule_verdict": verdict,
        "audit_required": verdict == "WARN",
        "warn_source": warn_source,
        "profile_id": profile.get("profile_id"),
        "profile_hash": profile.get("profile_hash"),
        "profile_kind": profile.get("profile_kind"),
        "rule_basis": rule_basis,
        "property": prop,
        "value": value,
        "units": profile.get("units"),
        "pass_range": profile.get("pass_range"),
        "warn_range": profile.get("warn_range"),
        "similarity_reference_id": transfer.get("similarity_reference_id"),
        "similarity_score": transfer.get("similarity_score"),
        "similarity_score_min": transfer.get("similarity_score_min"),
        "similarity_distance_reason": transfer.get("similarity_distance_reason"),
        "u_similarity_rel": transfer.get("u_similarity_rel"),
        "similarity_distance_metric_version": transfer.get("distance_metric_version"),
        "reason": reason,
        **density_metadata,
    }


def get_rule_profile(
    context: str | dict[str, Any],
    property_name: str,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    profile = select_rule_profile(context, property_name, output_dir=output_dir)
    if profile is None:
        return {"status": "not_found", "profile": None}
    return {"status": "ok", "profile": profile}


def rule_profile_source_taxonomy() -> dict[str, Any]:
    return {
        "status": "ok",
        "schema_version": RULE_PROFILE_SCHEMA_VERSION,
        "high_trust_source_types": sorted(HIGH_TRUST_SOURCE_TYPES),
        "control_source_types": sorted(CONTROL_SOURCE_TYPES),
        "supporting_source_types": sorted(SUPPORTING_SOURCE_TYPES),
        "supported_rule_properties": list_supported_rule_properties(),
        "discovery_modes": {
            "offline_structured_seed": (
                "Reproducible registered source records used for tests and smoke runs. "
                "These records are provider seeds, not live database queries."
            ),
            "live_public_csv": "RadonPy/HTPMD structured public CSV connectors using the reference-record schema.",
            "polyinfo_export": "PoLyInfo is read only from official exports or user-provided local caches; bulk scraping is disabled.",
            "same_input_control": "Local clean/control simulation records retained for repair and re-gating only.",
        },
        "auto_activation_rule": (
            "At least two independent sources are required, with at least one high-trust "
            "source such as RadonPy, HTPMD, PoLyInfo, a handbook/NIST/CRC source, a benchmark paper, "
            "or peer-reviewed literature. Clean/control simulation records are retained for repair, "
            "diagnosis and re-gating provenance, but they do not set active profile standard ranges "
            "by default. Conflicting sources block activation."
        ),
    }
