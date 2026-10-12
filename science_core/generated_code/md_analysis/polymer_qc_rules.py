#!/usr/bin/env python3
"""Polymer-general evidence-integrity QC rules.

These checks are intentionally not family-specific physical thresholds.  They
cover completion, finite numeric evidence, schema consistency, probability
bounds, and unsupported module contracts.  Property-specific plausibility
thresholds remain in frozen rule profiles.

The MSD diffusive-regime gate invoked here is an internal convergence and
evidence-quality check (diffusive slope band, fit R^2, and MSD-over-Rg^2
self-consistency), not an external property threshold; external property
acceptance such as density tiers stays in the frozen rule profiles.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from pelcb_msd_publication_gate import evaluate_msd_publication_gate
from polymer_runtime_registry import (  # noqa: E402
    RUNTIME_CAPABILITY_SCHEMA_VERSION,
    assess_preparation_capability,
)


QC_SCHEMA_VERSION = "polymer_general_qc_rules.v1"
MISSING_STATUSES = {"", "missing", "nan", "none", "null"}
CONTACT_MONOTONIC_ABS_TOL = 0.002
CONTACT_MONOTONIC_REL_TOL = 0.05
RUNTIME_SUPPORT_PROFILE_ID = RUNTIME_CAPABILITY_SCHEMA_VERSION


def _parse_jsonish(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return dict(json.loads(value))
    return dict(value)


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


def _value(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def _text(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = _value(row, key, default)
    return str(value if value is not None else default).strip()


def _status(row: Mapping[str, Any], key: str) -> str:
    return _text(row, key).upper()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_STATUSES


def _finite_positive(row: Mapping[str, Any], key: str, issues: list[str], *, allow_zero: bool = False) -> float | None:
    value = _value(row, key)
    if _is_missing(value):
        issues.append(f"{key} missing")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        issues.append(f"{key} non_numeric")
        return None
    if not math.isfinite(number):
        issues.append(f"{key} non_finite")
        return None
    if allow_zero:
        if number < 0:
            issues.append(f"{key} negative")
    elif number <= 0:
        issues.append(f"{key} non_positive")
    return number


def _finite_number(row: Mapping[str, Any], key: str, issues: list[str]) -> float | None:
    value = _value(row, key)
    if _is_missing(value):
        issues.append(f"{key} missing")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        issues.append(f"{key} non_numeric")
        return None
    if not math.isfinite(number):
        issues.append(f"{key} non_finite")
        return None
    return number


def _finite_fraction(row: Mapping[str, Any], key: str, issues: list[str]) -> float | None:
    number = _finite_positive(row, key, issues, allow_zero=True)
    if number is not None and not (0.0 <= number <= 1.0):
        issues.append(f"{key} outside_[0,1]")
    return number


def _bool_false(value: Any) -> bool:
    return str(value).strip().lower() in {"0", "false", "no", "unsupported", "not_supported"}


def _bool_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "required", "require", "ok"}


def _check(
    domain: str,
    verdict: str,
    warn_source: str | None,
    issues: list[str],
    *,
    failure_source: str | None = None,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "verdict": verdict,
        "warn_source": warn_source,
        "failure_source": failure_source,
        "issues": issues,
    }


def assess_registered_runtime_support(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive support flags from explicit runtime configuration values."""

    assessment = assess_preparation_capability(
        polymer_family=_value(row, "polymer_family"),
        architecture=_value(row, "architecture"),
        builder_backend=_value(row, "builder_backend"),
        force_field=_value(row, "force_field"),
        elements=_value(row, "elements") or _value(row, "chemical_elements"),
        topology_format=_value(row, "topology_format"),
    )
    return {
        "force_field_supported": assessment["force_field_supported"],
        "builder_backend_supported": assessment["builder_backend_supported"],
        "topology_supported": assessment["topology_supported"],
        "support_assessment_source": assessment["support_assessment_source"],
        "support_assessment": assessment,
    }


def _status_check(row: Mapping[str, Any], status_key: str, domain: str, warn_source: str) -> tuple[str, str | None, list[str]]:
    status = _status(row, status_key)
    issues: list[str] = []
    if status in {"OK", "PASS", "CURRENT", "VALIDATED"}:
        return "PASS", None, issues
    if status in {"WARN", "CHECK", "EXCLUDED"}:
        issues.append(f"{status_key}={status}")
        return "WARN", warn_source, issues
    if status in {"MISSING", ""}:
        issues.append(f"{status_key}=MISSING")
        return "WARN", "evidence_incomplete_warn", issues
    issues.append(f"{status_key}={status or 'unknown'}")
    return "WARN", warn_source, issues


def check_runtime_provenance(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    provenance_issues: list[str] = []
    runtime_issues: list[str] = []
    unsupported_issues: list[str] = []
    qualification_issues: list[str] = []

    if _is_missing(_value(row, "force_field")):
        provenance_issues.append("force_field missing")
    if _is_missing(_value(row, "engine_version")):
        provenance_issues.append("engine_version missing")
    support_fields = ("force_field_supported", "builder_backend_supported", "topology_supported")
    support_inputs = {
        "force_field_supported": "force_field",
        "builder_backend_supported": "builder_backend",
        "topology_supported": "topology_format",
    }
    derived_support = assess_registered_runtime_support(row)
    missing_support_fields = [key for key in support_fields if _is_missing(_value(row, key))]
    if missing_support_fields:
        provenance_issues.append("support declaration missing: " + ",".join(missing_support_fields))
    if not _value(row, "support_assessment_source"):
        provenance_issues.append("support_assessment_source missing")
    unsupported_labels = {
        "force_field_supported": "force_field unsupported",
        "builder_backend_supported": "builder_backend unsupported",
        "topology_supported": "topology unsupported",
    }
    for support_field, input_field in support_inputs.items():
        declared = _value(row, support_field)
        derived = bool(derived_support[support_field])
        if _is_missing(_value(row, input_field)):
            provenance_issues.append(f"{input_field} missing for registered support assessment")
            continue
        if not derived or (not _is_missing(declared) and _bool_false(declared)):
            unsupported_issues.append(unsupported_labels[support_field])
        if not _is_missing(declared) and (_bool_true(declared) != derived):
            provenance_issues.append(f"{support_field} disagrees with registered support assessment")

    force_field_qualification = _text(row, "force_field_qualification_status").lower()
    if force_field_qualification == "external_reference_qualification_required":
        qualification_issues.append("external force-field qualification is required")

    fatal_markers = _value(row, "fatal_log_marker_count", 0)
    try:
        fatal_count = int(float(fatal_markers or 0))
    except (TypeError, ValueError):
        fatal_count = 0
    if fatal_count > 0:
        runtime_issues.append("fatal_log_marker_count > 0")
    if _bool_false(_value(row, "gromacs_finished")):
        runtime_issues.append("gromacs_finished false")

    if unsupported_issues:
        checks.append(
            _check(
                "runtime_provenance",
                "FAIL",
                None,
                unsupported_issues,
                failure_source="unsupported_configuration_fail",
            )
        )
    elif provenance_issues:
        checks.append(_check("runtime_provenance", "WARN", "provenance_warn", provenance_issues))
    else:
        checks.append(_check("runtime_provenance", "PASS", None, []))
    if qualification_issues:
        checks.append(
            _check(
                "force_field_qualification",
                "WARN",
                "force_field_qualification_warn",
                qualification_issues,
            )
        )
    else:
        checks.append(_check("force_field_qualification", "PASS", None, []))
    if runtime_issues:
        checks.append(_check("runtime", "WARN", "runtime_warn", runtime_issues))
    else:
        checks.append(_check("runtime", "PASS", None, []))
    return checks


def check_rg(row: Mapping[str, Any]) -> dict[str, Any]:
    verdict, warn_source, issues = _status_check(row, "status_Rg", "rg", "rg_warn")
    _finite_positive(row, "rg_mean_nm", issues)
    _finite_positive(row, "rg2_mean_nm2", issues)
    if issues and verdict == "PASS":
        verdict, warn_source = "WARN", "rg_warn"
    return _check("rg", verdict, warn_source, issues)


def check_shape(row: Mapping[str, Any]) -> dict[str, Any]:
    verdict, warn_source, issues = _status_check(row, "status_shape", "shape", "shape_warn")
    rg2 = _finite_positive(row, "shape_rg2_carbon_mean_nm2", issues)
    asph = _finite_positive(row, "shape_asphericity_mean_nm2", issues, allow_zero=True)
    acyl = _finite_positive(row, "shape_acylindricity_mean_nm2", issues, allow_zero=True)
    rel = _finite_positive(row, "shape_relative_anisotropy_mean", issues, allow_zero=True)
    if rg2 is not None and asph is not None and asph > rg2 + 1e-12:
        issues.append("shape_asphericity_mean_nm2 exceeds shape_rg2_carbon_mean_nm2")
    if rg2 is not None and acyl is not None and acyl > rg2 + 1e-12:
        issues.append("shape_acylindricity_mean_nm2 exceeds shape_rg2_carbon_mean_nm2")
    if rel is not None and not (0.0 <= rel <= 1.0):
        issues.append("shape_relative_anisotropy_mean outside_[0,1]")
    if issues and verdict == "PASS":
        verdict, warn_source = "WARN", "shape_warn"
    return _check("shape", verdict, warn_source, issues)


def check_rdf(row: Mapping[str, Any]) -> dict[str, Any]:
    verdict, warn_source, issues = _status_check(row, "status_RDF", "rdf", "rdf_warn")
    validation = _text(row, "rdf_validation_status").lower()
    if validation in {"excluded", "warn"}:
        issues.append(f"rdf_validation_status={validation}")
        verdict, warn_source = "WARN", "rdf_warn"
    for key in (
        "rdf_all_first_peak_g",
        "rdf_excl_first_peak_g",
        "rdf_inter_chain_cc_first_peak_g",
        "rdf_intra_chain_nonbonded_cc_first_peak_g",
    ):
        _finite_positive(row, key, issues)
    if issues and verdict == "PASS":
        verdict, warn_source = "WARN", "rdf_warn"
    return _check("rdf", verdict, warn_source, issues)


def check_contact(row: Mapping[str, Any]) -> dict[str, Any]:
    verdict, warn_source, issues = _status_check(row, "status_contact", "contact", "contact_warn")
    contacts = [_finite_fraction(row, key, issues) for key in ("P_contact_rc0p8_s10", "P_contact_rc0p8_s20", "P_contact_rc0p8_s50")]
    if all(value is not None for value in contacts):
        s10, s20, s50 = contacts
        assert s10 is not None and s20 is not None and s50 is not None
        # Contact probabilities are estimated from finite trajectory samples.
        # Treat tiny non-monotonic inversions as numerical/statistical noise,
        # while retaining the QC warning for material violations.
        tol = max(CONTACT_MONOTONIC_ABS_TOL, CONTACT_MONOTONIC_REL_TOL * max(s10, s20, s50))
        if not (s10 + tol >= s20 and s20 + tol >= s50):
            issues.append("contact probabilities are not monotonic with sequence separation")
    if issues and verdict == "PASS":
        verdict, warn_source = "WARN", "contact_warn"
    return _check("contact", verdict, warn_source, issues)


def check_packing(row: Mapping[str, Any]) -> dict[str, Any]:
    verdict, warn_source, issues = _status_check(row, "status_packing", "packing", "packing_warn")
    if verdict == "WARN" and warn_source == "evidence_incomplete_warn":
        # Packing evidence is optional unless a producer explicitly reports a
        # packing status or metrics.  Missing status alone should not penalize
        # legacy aggregate rows.
        verdict, warn_source, issues = "PASS", None, []

    present_keys = [
        key
        for key in (
            "void_fraction",
            "largest_void_radius_nm",
            "low_density_voxel_fraction",
            "min_nonbonded_distance_nm",
            "packing_uniformity_score",
            "box_occupancy_ratio",
        )
        if not _is_missing(_value(row, key))
    ]
    if not present_keys:
        return _check("packing", verdict, warn_source, issues)

    void_fraction = _finite_fraction(row, "void_fraction", issues) if "void_fraction" in present_keys else None
    low_density = (
        _finite_fraction(row, "low_density_voxel_fraction", issues) if "low_density_voxel_fraction" in present_keys else None
    )
    largest_void = (
        _finite_positive(row, "largest_void_radius_nm", issues, allow_zero=True)
        if "largest_void_radius_nm" in present_keys
        else None
    )
    min_dist = (
        _finite_positive(row, "min_nonbonded_distance_nm", issues, allow_zero=True)
        if "min_nonbonded_distance_nm" in present_keys
        else None
    )
    packing_uniformity = (
        _finite_fraction(row, "packing_uniformity_score", issues) if "packing_uniformity_score" in present_keys else None
    )
    occupancy = _finite_fraction(row, "box_occupancy_ratio", issues) if "box_occupancy_ratio" in present_keys else None

    if void_fraction is not None and void_fraction > 0.10:
        issues.append("void_fraction > 0.10")
    if low_density is not None and low_density > 0.15:
        issues.append("low_density_voxel_fraction > 0.15")
    if largest_void is not None and largest_void > 1.0:
        issues.append("largest_void_radius_nm > 1.0")
    if min_dist is not None and min_dist < 0.08:
        issues.append("min_nonbonded_distance_nm < 0.08")
    if packing_uniformity is not None and packing_uniformity < 0.5:
        issues.append("packing_uniformity_score < 0.5")
    if occupancy is not None and occupancy < 0.5:
        issues.append("box_occupancy_ratio < 0.5")

    if issues and verdict == "PASS":
        verdict, warn_source = "WARN", "packing_warn"
    return _check("packing", verdict, warn_source, issues)


def check_msd(row: Mapping[str, Any]) -> dict[str, Any]:
    verdict, warn_source, issues = _status_check(row, "status_MSD", "msd", "msd_warn")
    gate = evaluate_msd_publication_gate(row, require_clean=True)
    if not gate["msd_publication_grade"]:
        issues.extend(gate["issues"])
        verdict, warn_source = "WARN", "msd_warn"
    _finite_positive(row, "D_mean_m2_s", issues)
    if issues and verdict == "PASS":
        verdict, warn_source = "WARN", "msd_warn"
    return _check("msd", verdict, warn_source, issues)


def check_intrachain_dynamics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Require clean g3 chain-COM evidence when a clean overlay declares it.

    The publication gate is carried by clean chain-COM dynamics.  The g1/g3
    ratio is retained as a mechanistic sidecar and must not block release when
    the clean g3 evidence is present and finite.
    """

    evidence_keys = (
        "status_intrachain_g3",
        "intrachain_g3_alpha_100_300",
        "intrachain_g3_time_last_ps",
        "g1_monomer_over_g3_10ns",
        "intrachain_g3_source",
    )
    required = _bool_true(_value(row, "require_intrachain_g3")) or any(
        not _is_missing(_value(row, key)) for key in evidence_keys
    )
    if not required:
        return _check("intrachain_dynamics", "PASS", None, [])

    verdict, warn_source, issues = _status_check(
        row,
        "status_intrachain_g3",
        "intrachain_dynamics",
        "evidence_incomplete_warn",
    )
    _finite_number(row, "intrachain_g3_alpha_100_300", issues)
    _finite_positive(row, "intrachain_g3_time_last_ps", issues)
    if not _is_missing(_value(row, "g1_monomer_over_g3_10ns")):
        _finite_positive(row, "g1_monomer_over_g3_10ns", issues)
    if _is_missing(_value(row, "intrachain_g3_source")):
        issues.append("intrachain_g3_source missing")
    if issues and verdict == "PASS":
        verdict, warn_source = "WARN", "evidence_incomplete_warn"
    return _check("intrachain_dynamics", verdict, warn_source, issues)


def check_fsqt(row: Mapping[str, Any]) -> dict[str, Any]:
    verdict, warn_source, issues = _status_check(row, "status_fsqt_tau_alpha", "fsqt", "fsqt_warn")
    _finite_positive(row, "tau_alpha_ps", issues)
    if issues and verdict == "PASS":
        verdict, warn_source = "WARN", "fsqt_warn"
    return _check("fsqt", verdict, warn_source, issues)


def check_case_completion(row: Mapping[str, Any]) -> dict[str, Any]:
    case_status = _text(row, "case_status").lower()
    if case_status in {"incomplete_md", "missing", "missing_all"}:
        return _check("case_completion", "WARN", "evidence_incomplete_warn", [f"case_status={case_status}"])
    warning = _text(row, "warning_reason")
    if warning == "MISSING_ALL":
        return _check("case_completion", "WARN", "evidence_incomplete_warn", ["warning_reason=MISSING_ALL"])
    return _check("case_completion", "PASS", None, [])


def apply_polymer_qc_rules(evidence: str | Mapping[str, Any]) -> dict[str, Any]:
    """Apply polymer-general QC checks to one evidence row."""

    row = _parse_jsonish(evidence)
    checks: list[dict[str, Any]] = []
    checks.append(check_case_completion(row))
    checks.extend(check_runtime_provenance(row))
    checks.extend(
        [
            check_rg(row),
            check_shape(row),
            check_rdf(row),
            check_contact(row),
            check_packing(row),
            check_msd(row),
            check_intrachain_dynamics(row),
            check_fsqt(row),
        ]
    )

    warn_sources: list[str] = []
    failure_sources: list[str] = []
    terminal_reason: str | None = None
    for check in checks:
        source = check.get("warn_source")
        if check.get("verdict") == "WARN" and source and source not in warn_sources:
            warn_sources.append(str(source))
        if check.get("verdict") == "FAIL":
            terminal_reason = "unsupported_by_current_system"
            failure_source = check.get("failure_source")
            if failure_source and failure_source not in failure_sources:
                failure_sources.append(str(failure_source))

    if terminal_reason:
        verdict = "FAIL"
    elif warn_sources:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "schema_version": QC_SCHEMA_VERSION,
        "status": "ok",
        "case_id": _text(row, "case_id"),
        "rule_verdict": verdict,
        "warn_sources": warn_sources,
        "failure_sources": failure_sources,
        "terminal_reason": terminal_reason,
        "audit_required": verdict == "WARN",
        "domain_verdicts": {check["domain"]: check["verdict"] for check in checks},
        "checks": _json_safe(checks),
    }


__all__ = ["apply_polymer_qc_rules", "assess_registered_runtime_support"]
