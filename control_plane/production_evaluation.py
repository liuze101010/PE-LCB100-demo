from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .science_core_runtime import analysis_path


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def status(value: Any) -> str:
    return "OK" if str(value or "").strip().lower() in {"ok", "pass"} else "WARN"


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "supported", "pass", "ok"}:
        return True
    if text in {"0", "false", "no", "unsupported", "fail"}:
        return False
    return None


def normalize_case_evidence(raw: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    rg = dict(raw.get("rg_summary") or {})
    msd = dict(raw.get("msd_summary") or {})
    shape = dict(raw.get("shape_summary") or {})
    dynamics = dict(raw.get("dynamics_summary") or {})
    contact = dict(raw.get("contact_summary") or {})
    intrachain = dict(raw.get("intrachain_summary") or {})
    density = dict(raw.get("density") or {})
    runtime = dict(raw.get("runtime") or {})
    rdf_payload = dict(raw.get("rdf_summary") or {})
    rdf_entries = {
        str(item.get("rdf_kind")): dict(item)
        for item in rdf_payload.get("entries", [])
        if isinstance(item, dict) and item.get("rdf_kind")
    }
    builder_spec = dict(case.get("builder_spec") or {})
    automatic_setup = dict(builder_spec.get("automatic_setup") or case.get("automatic_setup") or {})
    preparation_check = dict(
        builder_spec.get("preparation_check")
        or case.get("preparation_check")
        or automatic_setup.get("preparation_check")
        or {}
    )
    builder_selection = dict(automatic_setup.get("builder_selection") or {})
    force_field_selection = dict(automatic_setup.get("force_field_selection") or {})
    setup_needs_review = str(automatic_setup.get("status") or "").strip().lower() == "review_required"
    force_field = first_value(
        case.get("force_field"),
        builder_spec.get("force_field"),
        force_field_selection.get("force_field"),
    )
    builder_backend = first_value(
        case.get("builder_backend"),
        builder_spec.get("builder_backend"),
        builder_selection.get("builder_backend"),
    )
    topology_format = first_value(case.get("topology_format"), builder_spec.get("topology_format"))
    if force_field is None:
        force_field = "unsupported" if setup_needs_review else "opls/2024/opls-aa"
    if builder_backend is None:
        builder_backend = "unsupported" if setup_needs_review else "emc_alpha_olefin_builder"
    if topology_format is None:
        topology_format = "unsupported" if preparation_check.get("topology_complete") is False else "gromacs_atomistic"
    engine_text = "\n".join(str(item) for item in runtime.get("engine_version", []))
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "case_status": "completed",
        "force_field": force_field,
        "builder_backend": builder_backend,
        "topology_format": topology_format,
        "engine_version": engine_text,
        "force_field_supported": optional_bool(
            first_value(
                preparation_check.get("force_field_supported"),
                builder_spec.get("force_field_supported"),
                case.get("force_field_supported"),
            )
        ),
        "force_field_qualification_status": (
            case.get("force_field_qualification_status")
            or force_field_selection.get("scientific_qualification_status")
            or builder_spec.get("force_field_qualification_status")
            or ""
        ),
        "builder_backend_supported": optional_bool(
            first_value(
                preparation_check.get("builder_backend_supported"),
                builder_spec.get("builder_backend_supported"),
                case.get("builder_backend_supported"),
            )
        ),
        "topology_supported": optional_bool(
            first_value(
                preparation_check.get("topology_supported"),
                builder_spec.get("topology_supported"),
                case.get("topology_supported"),
            )
        ),
        "support_assessment_source": first_value(
            preparation_check.get("source"),
            builder_spec.get("support_assessment_source"),
            case.get("support_assessment_source"),
        ),
        "preparation_check": preparation_check,
        "gromacs_finished": bool(runtime.get("gromacs_finished")),
        "fatal_log_marker_count": int(runtime.get("fatal_log_marker_count") or 0),
        "status_density": density.get("status_density", "MISSING"),
        "density_g_cm3": density.get("density_g_cm3"),
        "density_stage": density.get("density_stage"),
        "density_frame_count": density.get("density_frame_count"),
        "status_Rg": status(rg.get("status")),
        "rg_mean_nm": rg.get("rg_mean_nm"),
        "rg2_mean_nm2": rg.get("rg2_mean_nm2"),
        "status_shape": status(shape.get("status")),
        "shape_rg2_carbon_mean_nm2": shape.get("rg2_carbon_mean_nm2"),
        "shape_asphericity_mean_nm2": shape.get("asphericity_mean_nm2"),
        "shape_acylindricity_mean_nm2": shape.get("acylindricity_mean_nm2"),
        "shape_relative_anisotropy_mean": shape.get("relative_shape_anisotropy_mean"),
        "status_RDF": "OK" if set(rdf_entries) >= {"all", "excl", "inter_chain_cc", "intra_chain_nonbonded_cc"} else "MISSING",
        "rdf_validation_status": "ok" if set(rdf_entries) >= {"all", "excl", "inter_chain_cc", "intra_chain_nonbonded_cc"} else "warn",
        "status_contact": status(contact.get("status")),
        "P_contact_rc0p8_s10": contact.get("P_contact_rc0p8_s10"),
        "P_contact_rc0p8_s20": contact.get("P_contact_rc0p8_s20"),
        "P_contact_rc0p8_s50": contact.get("P_contact_rc0p8_s50"),
        "status_MSD": status(msd.get("status")),
        "msd_clean_engine_source": "2024.4" in engine_text,
        "D_mean_m2_s": msd.get("D_mean_m2_s"),
        "D_outlier_fraction": msd.get("D_outlier_fraction"),
        "diffusive_regime_ok": msd.get("diffusive_regime_ok"),
        "fit_quality": msd.get("fit_quality"),
        "fit_begin_ps": msd.get("fit_begin_ps"),
        "fit_end_ps": msd.get("fit_end_ps"),
        "time_first_ps": msd.get("time_first_ps"),
        "time_last_ps": msd.get("time_last_ps"),
        "time_final_ps": msd.get("time_final_ps"),
        "loglog_slope": msd.get("loglog_slope"),
        "python_fit": msd.get("python_fit"),
        "window_D_cv_pct": msd.get("window_D_cv_pct"),
        "window_sensitive": msd.get("window_sensitive"),
        "final_MSD_nm2": msd.get("final_MSD_nm2"),
        "status_fsqt_tau_alpha": "OK"
        if status(dynamics.get("status")) == "OK" and number(dynamics.get("tau_alpha_ps"))
        else "WARN",
        "tau_alpha_ps": dynamics.get("tau_alpha_ps"),
        "tau_alpha_censored": dynamics.get("tau_alpha_censored"),
        "require_intrachain_g3": True,
        "status_intrachain_g3": "OK"
        if number(intrachain.get("g3_alpha_100_300")) and not intrachain.get("g3_error")
        else "WARN",
        "intrachain_g3_alpha_100_300": intrachain.get("g3_alpha_100_300"),
        "intrachain_g3_time_last_ps": intrachain.get("time_last_ps"),
        "g1_monomer_over_g3_10ns": intrachain.get("g1_monomer_over_g3_10ns"),
        "intrachain_g3_source": "canonical_clean_g3" if not intrachain.get("g3_error") else "",
    }
    for kind in ("all", "excl", "inter_chain_cc", "intra_chain_nonbonded_cc"):
        row[f"rdf_{kind}_first_peak_g"] = rdf_entries.get(kind, {}).get("first_peak_g")
    return row


def evaluate_preparation_support(case: dict[str, Any]) -> dict[str, Any]:
    """Let the registered rule layer assign PASS, WARN or FAIL to preparation records."""

    analysis_path()
    from polymer_qc_rules import apply_polymer_qc_rules, assess_registered_runtime_support  # type: ignore

    row = normalize_case_evidence({}, case)
    row["engine_version"] = "preparation_preflight"
    registered_support = assess_registered_runtime_support(row)
    for key in ("force_field_supported", "builder_backend_supported", "topology_supported"):
        if row.get(key) is None:
            row[key] = registered_support[key]
    if not row.get("support_assessment_source"):
        row["support_assessment_source"] = registered_support["support_assessment_source"]
    row["support_assessment"] = registered_support["support_assessment"]

    generic = apply_polymer_qc_rules(row)
    checks = {str(item.get("domain")): item for item in generic.get("checks", [])}
    support_check = checks.get("runtime_provenance") or {}
    qualification_check = checks.get("force_field_qualification") or {}
    if support_check.get("verdict") == "FAIL":
        verdict = "FAIL"
    elif qualification_check.get("verdict") == "WARN":
        verdict = "WARN"
    else:
        verdict = "PASS"
    return {
        "rule_verdict": verdict,
        "audit_required": verdict == "WARN",
        "failure_sources": list(generic.get("failure_sources") or []) if verdict == "FAIL" else [],
        "warn_sources": ["force_field_qualification_warn"] if verdict == "WARN" else [],
        "support_check": support_check,
        "force_field_qualification_check": qualification_check,
        "support_assessment": registered_support["support_assessment"],
        "support_assessment_source": row["support_assessment_source"],
    }


def evaluate_case(raw: dict[str, Any], case: dict[str, Any], profile_dir: Path) -> dict[str, Any]:
    analysis_path()
    from polymer_qc_rules import apply_polymer_qc_rules, assess_registered_runtime_support  # type: ignore
    from rule_profile_registry import apply_rule_gate, install_radonpy_pe_lcb_density_profile  # type: ignore

    profile_dir.mkdir(parents=True, exist_ok=True)
    if not (profile_dir / "active_rule_profile.json").is_file():
        install_radonpy_pe_lcb_density_profile(
            output_dir=profile_dir,
            reason="install bundled RadonPy PE-LCB100 density profile",
        )
    row = normalize_case_evidence(raw, case)
    registered_support = assess_registered_runtime_support(row)
    for key in ("force_field_supported", "builder_backend_supported", "topology_supported"):
        if row.get(key) is None:
            row[key] = registered_support[key]
    if not row.get("support_assessment_source"):
        row["support_assessment_source"] = registered_support["support_assessment_source"]
    row["support_assessment"] = registered_support["support_assessment"]
    generic = apply_polymer_qc_rules(row)
    density = apply_rule_gate(
        {"density_gcc": row.get("density_g_cm3")},
        context={
            "polymer_family": "PE-LCB100",
            "force_field": "opls-aa",
            "engine": row.get("engine_version"),
            "temperature_K": case.get("temperature_k", 400),
            "pressure_bar": float(case.get("pressure_atm", 1)) * 1.01325,
            "state": "melt",
            "case_id": case.get("case_id"),
            "monomer_smiles": case.get("monomer_smiles"),
            "actual_total_carbons_per_chain": case.get("actual_total_carbons_per_chain"),
            "target_total_carbons_per_chain": case.get("target_total_carbons_per_chain"),
        },
        property_name="density",
        output_dir=profile_dir,
    )
    warn_sources = list(generic.get("warn_sources") or [])
    if density.get("warn_source") and density["warn_source"] not in warn_sources:
        warn_sources.append(density["warn_source"])
    verdict = "FAIL" if generic.get("rule_verdict") == "FAIL" else "WARN" if warn_sources else "PASS"
    return {
        "case_id": case["case_id"],
        "rule_verdict": verdict,
        "warn_sources": warn_sources,
        "generic_qc": generic,
        "density_rule": density,
        "evidence": row,
    }
