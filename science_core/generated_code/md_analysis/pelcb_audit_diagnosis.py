#!/usr/bin/env python3
"""General WARN diagnosis layer for polymer-MD repair routing.

Public rule verdicts are tri-state: PASS, WARN, or FAIL.  WARN is a trigger,
not a cause.  This module collects warning sources from frozen evidence,
infers candidate diagnoses, selects a primary diagnosis, and returns the
deterministic tool and parameters the repair router should use.  It does not
execute shell commands, edit mdp physics, or change thresholds.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping

from diagnosis_candidate_registry import DEFAULT_DIAGNOSIS_PROFILE_DIR, list_active_diagnosis_profiles, route_from_profile
from polymer_design_validator import list_registered_adapters


DIAGNOSIS_SCHEMA_VERSION = "pelcb_audit_diagnosis.v2"
DIAGNOSIS_PROFILE_DIR = Path(os.environ.get("PELCB_DIAGNOSIS_PROFILE_DIR", DEFAULT_DIAGNOSIS_PROFILE_DIR))

RULE_WARN_SOURCES = [
    "density_warn",
    "msd_warn",
    "rg_warn",
    "rdf_warn",
    "shape_warn",
    "contact_warn",
    "packing_warn",
    "fsqt_warn",
    "equilibration_warn",
    "runtime_warn",
    "topology_warn",
    "force_field_warn",
    "provenance_warn",
    "evidence_incomplete_warn",
    "unknown_warn",
    "rule_profile_missing",
    "density_rule_profile_missing",
    "self_diffusion_rule_profile_missing",
    "rg_rule_profile_missing",
    "r2_rule_profile_missing",
    "thermal_expansion_rule_profile_missing",
    "dielectric_constant_rule_profile_missing",
    "refractive_index_rule_profile_missing",
    "thermal_conductivity_rule_profile_missing",
    "conductivity_rule_profile_missing",
    "ion_diffusivity_rule_profile_missing",
    "rdf_rule_profile_missing",
    "shape_rule_profile_missing",
    "contact_rule_profile_missing",
    "packing_rule_profile_missing",
    "fsqt_rule_profile_missing",
]

# Audit disagreement is produced after the audit models vote.  It is not a
# rule-layer warning and must never be used to admit an otherwise passing case
# into the audit layer.  The legacy name remains explicit so archived ledgers
# can be interpreted without treating it as a current rule source.
LEGACY_AUDIT_OUTCOME_SOURCES = {"audit_disagreement_warn"}
WARN_SOURCES = RULE_WARN_SOURCES


def is_rule_profile_missing_warn_source(warn_source: str) -> bool:
    return warn_source == "rule_profile_missing" or str(warn_source).endswith("_rule_profile_missing")


def property_from_rule_profile_missing_warn_source(warn_source: str) -> str:
    source = str(warn_source or "")
    if source == "rule_profile_missing":
        return "unknown"
    suffix = "_rule_profile_missing"
    if source.endswith(suffix):
        return source[: -len(suffix)]
    return "unknown"

DIAGNOSIS_TAXONOMY: dict[str, list[str]] = {
    "construction_topology": [
        "builder_input_incomplete",
        "topology_unsupported",
        "emc_builder_failed",
        "force_field_assignment_unsupported",
    ],
    "md_execution": [
        "missing_md_outputs",
        "checkpoint_resume_failure",
        "fatal_runtime_error",
        "trajectory_missing_or_corrupt",
        "engine_or_protocol_sensitive_result",
        "insufficient_equilibration",
        "pressure_or_volume_not_relaxed",
    ],
    "postprocessing": [
        "postprocess_input_missing",
        "fit_window_or_method_issue",
        "rdf_cutoff_or_normalization_issue",
        "shape_tensor_schema_issue",
        "contact_definition_issue",
        "packing_void_or_local_density_issue",
        "fsqt_decay_window_issue",
    ],
    "physical_protocol_interpretation": [
        "force_field_systematic_offset",
        "out_of_protocol_scope",
        "true_borderline_physical_behavior",
        "evidence_conflict_between_observables",
    ],
    "audit_review": [
        "cross_model_disagreement",
        "low_confidence_audit",
        "insufficient_evidence_for_diagnosis",
        "automatic_review_required",
    ],
    "rule_calibration": [
        "rule_profile_missing",
        "profile_unfreezable",
        "profile_conflict",
    ],
    "terminal": [
        "unsupported_by_current_system",
    ],
}

DETERMINISTIC_MODULES: dict[str, dict[str, str]] = {
    "emc": {
        "module": "emc",
        "tool": "validate_or_build_emc_input",
        "responsibility": "candidate validation, builder-spec generation, EMC input/topology/coordinate materialization",
    },
    "md": {
        "module": "md",
        "tool": "submit_md_clean_engine",
        "responsibility": "restart-aware four-stage GROMACS execution, resume, clean-engine rerun, and MD extension when pre-frozen",
    },
    "postprocess": {
        "module": "postprocess",
        "tool": "run_postprocess_repair",
        "responsibility": "density/MSD/Rg/RDF/shape/contact/Fs(q,t) postprocessing repair and sidecar overlays",
    },
    "rule_review": {
        "module": "rule_review",
        "tool": "open_rule_calibration_review",
        "responsibility": "review family-specific thresholds, protocol scope, and force-field qualification before any new freeze",
    },
    "rule_calibration": {
        "module": "rule_calibration",
        "tool": "build_rule_profile_from_online_evidence",
        "responsibility": "discover structured evidence, propose a candidate rule_profile, machine-freeze if deterministic checks pass, and re-enter the rule gate",
    },
    "audit_review": {
        "module": "audit_review",
        "tool": "open_review_packet",
        "responsibility": "bounded audit review packet when deterministic re-entry is not justified",
    },
    "unsupported": {
        "module": "unsupported",
        "tool": "respond_unsupported",
        "responsibility": "return a current-system unsupported response without launching EMC, MD, or postprocessing",
    },
}

DIAGNOSIS_TO_ROUTE: dict[str, dict[str, Any]] = {
    "missing_md_outputs": {
        "repair_route": "repair_md_clean_engine",
        "repair_action": "md_clean_engine_rerun",
        "automation_level": "sidecar_md_execution",
        "executor": "o7_clean_supervisor_queue.py",
        "module": "md",
        "repair_tool": "submit_md_clean_engine",
        "tool_parameters": {
            "engine_profile": "gromacs_2024_4_clean",
            "reentry_stage": "md",
            "input_source": "archived_source_inputs",
        },
    },
    "checkpoint_resume_failure": {
        "repair_route": "repair_md_clean_engine",
        "repair_action": "md_clean_engine_rerun",
        "automation_level": "sidecar_md_execution",
        "executor": "o7_clean_supervisor_queue.py",
        "module": "md",
        "repair_tool": "submit_md_clean_engine",
        "tool_parameters": {"engine_profile": "gromacs_2024_4_clean", "reentry_stage": "md"},
    },
    "fatal_runtime_error": {
        "repair_route": "repair_md_clean_engine",
        "repair_action": "md_clean_engine_rerun",
        "automation_level": "sidecar_md_execution",
        "executor": "o7_clean_supervisor_queue.py",
        "module": "md",
        "repair_tool": "submit_md_clean_engine",
        "tool_parameters": {"engine_profile": "gromacs_2024_4_clean", "reentry_stage": "md"},
    },
    "trajectory_missing_or_corrupt": {
        "repair_route": "repair_md_clean_engine",
        "repair_action": "md_clean_engine_rerun",
        "automation_level": "sidecar_md_execution",
        "executor": "o7_clean_supervisor_queue.py",
        "module": "md",
        "repair_tool": "submit_md_clean_engine",
        "tool_parameters": {"engine_profile": "gromacs_2024_4_clean", "reentry_stage": "md"},
    },
    "engine_or_protocol_sensitive_result": {
        "repair_route": "optional_clean_engine_validation",
        "repair_action": "md_clean_engine_rerun",
        "automation_level": "optional_sidecar_validation",
        "executor": "o7_clean_supervisor_queue.py",
        "module": "md",
        "repair_tool": "submit_md_clean_engine",
        "tool_parameters": {
            "engine_profile": "gromacs_2024_4_clean",
            "reentry_stage": "md",
            "rerun_scope": "same_input_clean_engine_validation",
        },
    },
    "insufficient_equilibration": {
        "repair_route": "repair_md_clean_engine",
        "repair_action": "md_clean_engine_rerun_or_extension",
        "automation_level": "sidecar_md_execution",
        "executor": "o7_clean_supervisor_queue.py",
        "module": "md",
        "repair_tool": "submit_md_clean_engine",
        "tool_parameters": {"engine_profile": "gromacs_2024_4_clean", "reentry_stage": "md"},
    },
    "pressure_or_volume_not_relaxed": {
        "repair_route": "repair_md_clean_engine",
        "repair_action": "md_clean_engine_rerun_or_extension",
        "automation_level": "sidecar_md_execution",
        "executor": "o7_clean_supervisor_queue.py",
        "module": "md",
        "repair_tool": "submit_md_clean_engine",
        "tool_parameters": {"engine_profile": "gromacs_2024_4_clean", "reentry_stage": "md"},
    },
    "fit_window_or_method_issue": {
        "repair_route": "repair_postprocess_msd_window",
        "repair_action": "postprocess_msd_window_recompute",
        "automation_level": "postprocess_only_sidecar",
        "executor": "pelcb_msd_window_repair.py",
        "module": "postprocess",
        "repair_tool": "run_postprocess_repair",
        "tool_parameters": {
            "repair_kind": "msd_window",
            "window_strategy": "c100_aware",
        },
    },
    "rdf_cutoff_or_normalization_issue": {
        "repair_route": "repair_postprocess_rdf",
        "repair_action": "postprocess_rdf_recompute",
        "automation_level": "postprocess_only_sidecar",
        "executor": "compute_full_rdf_suite.py",
        "module": "postprocess",
        "repair_tool": "run_postprocess_repair",
        "tool_parameters": {"repair_kind": "rdf_recompute"},
    },
    "shape_tensor_schema_issue": {
        "repair_route": "repair_postprocess_shape",
        "repair_action": "postprocess_shape_tensor_recompute",
        "automation_level": "postprocess_only_sidecar",
        "executor": "compute_shape_tensor.py",
        "module": "postprocess",
        "repair_tool": "run_postprocess_repair",
        "tool_parameters": {"repair_kind": "shape_tensor"},
    },
    "contact_definition_issue": {
        "repair_route": "repair_postprocess_contact",
        "repair_action": "postprocess_contact_recompute",
        "automation_level": "postprocess_only_sidecar",
        "executor": "contact postprocess suite",
        "module": "postprocess",
        "repair_tool": "run_postprocess_repair",
        "tool_parameters": {"repair_kind": "contact"},
    },
    "packing_void_or_local_density_issue": {
        "repair_route": "repair_postprocess_packing",
        "repair_action": "postprocess_packing_void_recompute",
        "automation_level": "postprocess_only_sidecar",
        "executor": "packing/void postprocess suite",
        "module": "postprocess",
        "repair_tool": "run_postprocess_repair",
        "tool_parameters": {"repair_kind": "packing_void"},
    },
    "fsqt_decay_window_issue": {
        "repair_route": "repair_postprocess_fsqt",
        "repair_action": "postprocess_fsqt_recompute",
        "automation_level": "postprocess_only_sidecar",
        "executor": "Fs(q,t) postprocess suite",
        "module": "postprocess",
        "repair_tool": "run_postprocess_repair",
        "tool_parameters": {"repair_kind": "fsqt"},
    },
    "postprocess_input_missing": {
        "repair_route": "repair_postprocess_input_missing",
        "repair_action": "postprocess_input_repair",
        "automation_level": "postprocess_only_sidecar",
        "executor": "postprocess orchestrator",
        "module": "postprocess",
        "repair_tool": "run_postprocess_repair",
        "tool_parameters": {"repair_kind": "input_repair"},
    },
    "builder_input_incomplete": {
        "repair_route": "repair_emc_rebuild",
        "repair_action": "emc_rebuild",
        "automation_level": "sidecar_emc_rebuild",
        "executor": "EMC builder adapter",
        "module": "emc",
        "repair_tool": "validate_or_build_emc_input",
        "tool_parameters": {"repair_kind": "builder_input_completion"},
    },
    "emc_builder_failed": {
        "repair_route": "repair_emc_rebuild",
        "repair_action": "emc_rebuild",
        "automation_level": "sidecar_emc_rebuild",
        "executor": "EMC builder adapter",
        "module": "emc",
        "repair_tool": "validate_or_build_emc_input",
        "tool_parameters": {"repair_kind": "emc_rebuild"},
    },
    "topology_unsupported": {
        "repair_route": "unsupported_user_response",
        "repair_action": "respond_unsupported",
        "automation_level": "terminal_unsupported",
        "executor": "",
        "module": "unsupported",
        "repair_tool": "respond_unsupported",
        "tool_parameters": {"reason": "topology_unsupported"},
    },
    "force_field_assignment_unsupported": {
        "repair_route": "unsupported_user_response",
        "repair_action": "respond_unsupported",
        "automation_level": "terminal_unsupported",
        "executor": "",
        "module": "unsupported",
        "repair_tool": "respond_unsupported",
        "tool_parameters": {"reason": "force_field_assignment_unsupported"},
    },
    "unsupported_by_current_system": {
        "repair_route": "unsupported_user_response",
        "repair_action": "respond_unsupported",
        "automation_level": "terminal_unsupported",
        "executor": "",
        "module": "unsupported",
        "repair_tool": "respond_unsupported",
        "tool_parameters": {"reason": "unsupported_by_current_system"},
    },
    "force_field_systematic_offset": {
        "repair_route": "rule_calibration_review",
        "repair_action": "open_rule_calibration_review",
        "automation_level": "automatic_rule_review_packet",
        "executor": "",
        "module": "rule_review",
        "repair_tool": "open_rule_calibration_review",
        "tool_parameters": {"review_kind": "force_field_offset"},
    },
    "out_of_protocol_scope": {
        "repair_route": "rule_calibration_review",
        "repair_action": "open_protocol_scope_review",
        "automation_level": "automatic_rule_review_packet",
        "executor": "",
        "module": "rule_review",
        "repair_tool": "open_rule_calibration_review",
        "tool_parameters": {"review_kind": "protocol_scope"},
    },
    "true_borderline_physical_behavior": {
        "repair_route": "rule_calibration_review",
        "repair_action": "retain_qualified_warn_or_review",
        "automation_level": "automatic_rule_review_packet",
        "executor": "",
        "module": "rule_review",
        "repair_tool": "open_rule_calibration_review",
        "tool_parameters": {"review_kind": "borderline_physical_behavior"},
    },
    "rule_profile_missing": {
        "repair_route": "auto_rule_profile_calibration",
        "repair_action": "build_rule_profile_from_structured_evidence",
        "automation_level": "machine_freeze_if_evidence_sufficient",
        "executor": "mcp/pelcb_mcp_server.py::build_rule_profile_from_online_evidence",
        "module": "rule_calibration",
        "repair_tool": "build_rule_profile_from_online_evidence",
        "tool_parameters": {
            "repair_kind": "rule_profile_calibration",
            "use_live": True,
            "include_seed": True,
            "expected_success_condition": "candidate rule_profile is machine-frozen from structured evidence and original case re-enters the rule gate",
        },
    },
    "profile_unfreezable": {
        "repair_route": "rule_profile_unfreezable_review",
        "repair_action": "retain_profile_unfreezable_review",
        "automation_level": "automatic_review_packet",
        "executor": "MCP rule evidence ledger",
        "module": "rule_review",
        "repair_tool": "open_rule_calibration_review",
        "tool_parameters": {"review_kind": "rule_profile_missing"},
    },
    "profile_conflict": {
        "repair_route": "rule_profile_conflict_review",
        "repair_action": "open_rule_profile_conflict_review",
        "automation_level": "automatic_review_packet",
        "executor": "MCP rule evidence ledger",
        "module": "rule_review",
        "repair_tool": "open_rule_calibration_review",
        "tool_parameters": {"review_kind": "rule_profile_missing"},
    },
    "evidence_conflict_between_observables": {
        "repair_route": "review_evidence_conflict",
        "repair_action": "open_review_packet",
        "automation_level": "automatic_review_packet",
        "executor": "MCP get_audit_record / evidence ledger",
        "module": "audit_review",
        "repair_tool": "open_review_packet",
        "tool_parameters": {"review_kind": "evidence_conflict"},
    },
    "cross_model_disagreement": {
        "repair_route": "review_audit_disagreement",
        "repair_action": "open_audit_review_packet",
        "automation_level": "automatic_review_packet",
        "executor": "MCP get_audit_record / A3 ledger",
        "module": "audit_review",
        "repair_tool": "open_review_packet",
        "tool_parameters": {"review_kind": "audit_disagreement"},
    },
    "low_confidence_audit": {
        "repair_route": "review_low_confidence_audit",
        "repair_action": "open_review_packet",
        "automation_level": "automatic_review_packet",
        "executor": "MCP get_audit_record / A3 ledger",
        "module": "audit_review",
        "repair_tool": "open_review_packet",
        "tool_parameters": {"review_kind": "low_confidence_audit"},
    },
    "insufficient_evidence_for_diagnosis": {
        "repair_route": "review_insufficient_evidence",
        "repair_action": "open_review_packet",
        "automation_level": "automatic_review_packet",
        "executor": "MCP get_case_summary / evidence ledger",
        "module": "audit_review",
        "repair_tool": "open_review_packet",
        "tool_parameters": {"review_kind": "insufficient_evidence_for_diagnosis"},
    },
    "automatic_review_required": {
        "repair_route": "review_automatic",
        "repair_action": "open_review_packet",
        "automation_level": "automatic_review_packet",
        "executor": "MCP get_case_summary / evidence ledger",
        "module": "audit_review",
        "repair_tool": "open_review_packet",
        "tool_parameters": {"review_kind": "automatic_review_required"},
    },
    # Backward-compatible alias for archived diagnosis records. New runs emit
    # automatic_review_required and do not require a person to assign REVIEW.
    "manual_review_required": {
        "repair_route": "review_automatic",
        "repair_action": "open_review_packet",
        "automation_level": "automatic_review_packet",
        "executor": "MCP get_case_summary / evidence ledger",
        "module": "audit_review",
        "repair_tool": "open_review_packet",
        "tool_parameters": {"review_kind": "automatic_review_required"},
    },
    "none": {
        "repair_route": "no_action",
        "repair_action": "none",
        "automation_level": "none",
        "executor": "",
        "module": "audit_review",
        "repair_tool": "none",
        "tool_parameters": {},
    },
}

DIAGNOSIS_PRIORITY = {
    "unsupported_by_current_system": 0,
    "topology_unsupported": 1,
    "force_field_assignment_unsupported": 1,
    "missing_md_outputs": 10,
    "checkpoint_resume_failure": 12,
    "fatal_runtime_error": 12,
    "trajectory_missing_or_corrupt": 12,
    "emc_builder_failed": 15,
    "builder_input_incomplete": 15,
    "fit_window_or_method_issue": 20,
    "rdf_cutoff_or_normalization_issue": 25,
    "shape_tensor_schema_issue": 25,
    "contact_definition_issue": 25,
    "packing_void_or_local_density_issue": 25,
    "fsqt_decay_window_issue": 25,
    "postprocess_input_missing": 25,
    "engine_or_protocol_sensitive_result": 30,
    "insufficient_equilibration": 35,
    "pressure_or_volume_not_relaxed": 35,
    "force_field_systematic_offset": 50,
    "out_of_protocol_scope": 50,
    "true_borderline_physical_behavior": 55,
    "evidence_conflict_between_observables": 60,
    "cross_model_disagreement": 90,
    "low_confidence_audit": 90,
    "automatic_review_required": 95,
    "manual_review_required": 95,
    "insufficient_evidence_for_diagnosis": 100,
    "none": 999,
}

TOOL_PARAMETER_ALLOWLIST: dict[str, dict[str, set[Any]]] = {
    "submit_md_clean_engine": {
        "engine_profile": {"gromacs_2024_4_clean", "gromacs_2026_clean"},
        "clean_md_profile": {"pressure_relaxation_clean"},
        "reentry_stage": {"md", "pr", "nvt_prod"},
        "input_source": {"archived_source_inputs"},
        "rerun_scope": {"same_input_clean_engine_validation"},
    },
    "extend_clean_nvt_prod": {
        "repair_kind": {"clean_nvt_extension"},
        "reentry_stage": {"nvt_prod"},
        "expected_success_condition": {
            "clean tail-window diffusion passes the strict MSD publication gate"
        },
    },
    "run_postprocess_repair": {
        "repair_kind": {
            "msd_window",
            "rdf_recompute",
            "shape_tensor",
            "contact",
            "packing_void",
            "fsqt",
            "input_repair",
        },
        "grid_spacing_nm": {0.25, 0.5},
        "window_strategy": {"c100_aware", "strict_alternative", "default"},
        "window_policy": {"c100_aware", "strict_alternative", "default"},
    },
    "validate_or_build_emc_input": {
        "repair_kind": {"builder_input_completion", "emc_rebuild"},
        "adapter_id": set(list_registered_adapters()),
    },
    "open_review_packet": {
        "review_kind": {
            "evidence_conflict",
            "audit_disagreement",
            "low_confidence_audit",
            "insufficient_evidence_for_diagnosis",
            "automatic_review_required",
            "manual_review_required",
            "guard_rejected_tool_call",
            "candidate_route_review",
            "persistent_warn_review",
        },
    },
    "open_rule_calibration_review": {
        "review_kind": {
            "force_field_offset",
            "protocol_scope",
            "borderline_physical_behavior",
            "rule_profile_missing",
        },
        "property_name": {
            "density",
            "self_diffusion",
            "rg",
            "rdf",
            "msd",
            "packing_void",
            "contact",
            "fsqt",
            "shape",
        },
    },
    "build_rule_profile_from_online_evidence": {
        "repair_kind": {"rule_profile_calibration"},
        "property_name": {
            "density",
            "self_diffusion",
            "rg",
            "r2",
            "cp",
            "cv",
            "compressibility",
            "bulk_modulus",
            "thermal_expansion",
            "dielectric_constant",
            "refractive_index",
            "thermal_conductivity",
            "thermal_diffusivity",
            "conductivity",
            "ion_diffusivity",
            "li_diffusivity",
            "tfsi_diffusivity",
            "poly_diffusivity",
            "transference_number",
            "molarity",
        },
        "use_live": {True, False},
        "include_seed": {True, False},
    },
    "respond_unsupported": {
        "reason": {"topology_unsupported", "force_field_assignment_unsupported", "unsupported_by_current_system"},
    },
    "none": {},
}

FORBIDDEN_TOOL_PARAMETER_KEYS = {
    "cmd",
    "command",
    "shell",
    "shell_command",
    "arbitrary_shell",
    "mdp_override",
    "mdp_edits",
    "mdp_patch",
    "rule_threshold",
    "threshold",
    "density_threshold",
    "locked_label",
    "canonical_release",
    "canonical_path",
    "expert_label",
}


def repair_tool_affordances() -> dict[str, dict[str, Any]]:
    """Return bounded tool capabilities available to LLM repair routing."""
    return {
        "submit_md_clean_engine": {
            "tool": "submit_md_clean_engine",
            "module": "md",
            "warn_sources": ["density_warn", "runtime_warn", "evidence_incomplete_warn", "provenance_warn"],
            "allowed_parameters": {
                key: sorted(values, key=lambda item: str(item))
                for key, values in TOOL_PARAMETER_ALLOWLIST["submit_md_clean_engine"].items()
            },
            "forbidden_parameters": sorted(FORBIDDEN_TOOL_PARAMETER_KEYS),
            "max_repair_attempts": 2,
            "requires_evidence": ["mdout.mdp/provenance or missing-output evidence"],
        },
        "extend_clean_nvt_prod": {
            "tool": "extend_clean_nvt_prod",
            "module": "md",
            "warn_sources": ["msd_warn", "provenance_warn"],
            "allowed_parameters": {
                key: sorted(values, key=lambda item: str(item))
                for key, values in TOOL_PARAMETER_ALLOWLIST["extend_clean_nvt_prod"].items()
            },
            "forbidden_parameters": sorted(FORBIDDEN_TOOL_PARAMETER_KEYS),
            "max_repair_attempts": 2,
            "requires_evidence": [
                "clean checkpoint/provenance plus strict MSD publication-gate evidence"
            ],
        },
        "run_postprocess_repair": {
            "tool": "run_postprocess_repair",
            "module": "postprocess",
            "warn_sources": ["msd_warn", "rdf_warn", "shape_warn", "contact_warn", "packing_warn", "fsqt_warn", "rg_warn"],
            "allowed_parameters": {
                key: sorted(values, key=lambda item: str(item))
                for key, values in TOOL_PARAMETER_ALLOWLIST["run_postprocess_repair"].items()
            },
            "forbidden_parameters": sorted(FORBIDDEN_TOOL_PARAMETER_KEYS),
            "max_repair_attempts": 2,
            "requires_evidence": ["postprocess status, numeric evidence, or input-missing evidence"],
        },
        "validate_or_build_emc_input": {
            "tool": "validate_or_build_emc_input",
            "module": "emc",
            "warn_sources": ["topology_warn", "force_field_warn"],
            "allowed_parameters": {
                key: sorted(values, key=lambda item: str(item))
                for key, values in TOOL_PARAMETER_ALLOWLIST["validate_or_build_emc_input"].items()
            },
            "forbidden_parameters": sorted(FORBIDDEN_TOOL_PARAMETER_KEYS),
            "max_repair_attempts": 1,
            "requires_evidence": ["candidate schema, topology, or builder evidence"],
        },
        "open_rule_calibration_review": {
            "tool": "open_rule_calibration_review",
            "module": "rule_review",
            "warn_sources": ["unknown_warn", "provenance_warn", "rule_profile_missing", "*_rule_profile_missing"],
            "allowed_parameters": {
                key: sorted(values, key=lambda item: str(item))
                for key, values in TOOL_PARAMETER_ALLOWLIST["open_rule_calibration_review"].items()
            },
            "forbidden_parameters": sorted(FORBIDDEN_TOOL_PARAMETER_KEYS),
            "max_repair_attempts": 0,
            "requires_evidence": ["rule-profile or public-source calibration evidence"],
        },
        "build_rule_profile_from_online_evidence": {
            "tool": "build_rule_profile_from_online_evidence",
            "module": "rule_calibration",
            "warn_sources": ["rule_profile_missing", "*_rule_profile_missing"],
            "allowed_parameters": {
                key: sorted(values, key=lambda item: str(item))
                for key, values in TOOL_PARAMETER_ALLOWLIST["build_rule_profile_from_online_evidence"].items()
            },
            "forbidden_parameters": sorted(FORBIDDEN_TOOL_PARAMETER_KEYS),
            "max_repair_attempts": 1,
            "requires_evidence": ["structured public/control rule evidence; deterministic freeze checks"],
        },
        "open_review_packet": {
            "tool": "open_review_packet",
            "module": "audit_review",
            "warn_sources": ["*"],
            "allowed_parameters": {
                key: sorted(values, key=lambda item: str(item))
                for key, values in TOOL_PARAMETER_ALLOWLIST["open_review_packet"].items()
            },
            "forbidden_parameters": sorted(FORBIDDEN_TOOL_PARAMETER_KEYS),
            "max_repair_attempts": 0,
            "requires_evidence": ["audit, evidence conflict, or guard rejection record"],
        },
    }


def _coerce_allowed_parameter(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped) if "." in stripped else stripped
        except Exception:
            return stripped
    return value


def _proposal_for_review(reason: str) -> dict[str, Any]:
    return {
        "tool": "open_review_packet",
        "parameters": {"review_kind": "guard_rejected_tool_call"},
        "rationale": reason,
        "expected_success_condition": "review packet records the rejected repair proposal",
    }


def _warn_source_supported(warn_source: str, supported_sources: set[str]) -> bool:
    if "*" in supported_sources or warn_source in supported_sources:
        return True
    for pattern in supported_sources:
        if pattern.startswith("*") and warn_source.endswith(pattern[1:]):
            return True
        if pattern.endswith("*") and warn_source.startswith(pattern[:-1]):
            return True
    return False


def validate_repair_tool_call(
    proposal: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an LLM repair proposal against MCP tool affordances.

    The guard accepts natural-language diagnoses only when the resulting tool
    call is a registered tool with allowlisted parameters. It rejects attempts
    to edit thresholds, mdp physics, labels, canonical release, or shell.
    """
    proposal = dict(proposal or {})
    context = dict(context or {})
    tool = str(proposal.get("tool") or proposal.get("repair_tool") or "").strip()
    parameters = proposal.get("parameters") or proposal.get("tool_parameters") or {}
    if not isinstance(parameters, Mapping):
        parameters = {}
    affordances = repair_tool_affordances()
    if tool not in affordances:
        reason = f"tool {tool or '<missing>'} is not a registered repair tool"
        review = _proposal_for_review(reason)
        return {
            "guard_verdict": "reject",
            "guard_rejection_reason": reason,
            "tool_affordance_match": "",
            "proposal": review,
        }

    allowed = TOOL_PARAMETER_ALLOWLIST.get(tool, {})
    normalized: dict[str, Any] = {}
    for key, value in parameters.items():
        key_text = str(key)
        key_lower = key_text.lower()
        if key_lower in FORBIDDEN_TOOL_PARAMETER_KEYS or any(fragment in key_lower for fragment in ("threshold", "mdp", "shell", "canonical", "label")):
            reason = f"forbidden parameter {key_text} cannot be set by LLM repair routing"
            review = _proposal_for_review(reason)
            return {
                "guard_verdict": "reject",
                "guard_rejection_reason": reason,
                "tool_affordance_match": tool,
                "proposal": review,
            }
        if key_text not in allowed:
            reason = f"parameter {key_text} is not allowlisted for {tool}"
            review = _proposal_for_review(reason)
            return {
                "guard_verdict": "reject",
                "guard_rejection_reason": reason,
                "tool_affordance_match": tool,
                "proposal": review,
            }
        candidate = _coerce_allowed_parameter(value)
        if candidate not in allowed[key_text]:
            reason = f"value {candidate!r} for {key_text} is not allowlisted for {tool}"
            review = _proposal_for_review(reason)
            return {
                "guard_verdict": "reject",
                "guard_rejection_reason": reason,
                "tool_affordance_match": tool,
                "proposal": review,
            }
        normalized[key_text] = candidate

    warn_sources = [str(item) for item in (proposal.get("warn_sources") or context.get("warn_sources") or [])]
    supported_sources = set(str(item) for item in affordances[tool].get("warn_sources", []))
    if warn_sources and not any(_warn_source_supported(source, supported_sources) for source in warn_sources):
        reason = f"tool {tool} does not declare support for warn sources {warn_sources}"
        review = _proposal_for_review(reason)
        return {
            "guard_verdict": "review",
            "guard_rejection_reason": reason,
            "tool_affordance_match": tool,
            "proposal": review,
        }

    return {
        "guard_verdict": "pass",
        "guard_rejection_reason": "",
        "tool_affordance_match": tool,
        "proposal": {
            "tool": tool,
            "parameters": normalized,
            "rationale": str(proposal.get("rationale") or ""),
            "expected_success_condition": str(
                proposal.get("expected_success_condition")
                or "selected deterministic module writes overlay evidence and re-enters rule gate"
            ),
        },
    }


def _value(row: Mapping[str, Any], key: str, default: Any = "") -> Any:
    try:
        value = row.get(key, default)  # type: ignore[attr-defined]
    except AttributeError:
        value = row[key] if key in row else default
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def _text(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = _value(row, key, default)
    return str(value if value is not None else default).strip()


def _upper(row: Mapping[str, Any], key: str) -> str:
    return _text(row, key).upper()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _falsey(value: Any) -> bool:
    return str(value).strip().lower() in {"0", "false", "no", "unsupported", "not_supported"}


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _has_value(row: Mapping[str, Any], key: str) -> bool:
    value = _value(row, key, "")
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none", "null", "missing"}


def _present_values(row: Mapping[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: _value(row, key) for key in keys if _has_value(row, key)}


MDP_EVIDENCE_PARSER_VERSION = "gromacs_mdp_effective_parameters.v1"
MDP_EVIDENCE_KEYS = {
    "integrator",
    "dt",
    "nsteps",
    "continuation",
    "cutoff-scheme",
    "nstlist",
    "rlist",
    "rcoulomb",
    "rvdw",
    "verlet-buffer-tolerance",
    "tcoupl",
    "tau-t",
    "ref-t",
    "pcoupl",
    "pcoupltype",
    "tau-p",
    "ref-p",
    "compressibility",
    "constraints",
}


def _resolve_evidence_path(raw_path: Any) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, Path(__file__).resolve().parent / path])
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _parse_mdp_effective_parameters(path: Path) -> dict[str, Any]:
    """Parse a small allowlist of effective run parameters from frozen mdout.mdp text."""

    if path.stat().st_size > 2 * 1024 * 1024:
        return {"status": "too_large", "source_path": str(path)}
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    parameters: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip().lower().replace("_", "-")
        if normalized_key in MDP_EVIDENCE_KEYS:
            parameters[normalized_key] = value.strip()
    return {
        "status": "parsed",
        "source_path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "parser_version": MDP_EVIDENCE_PARSER_VERSION,
        "parameters": parameters,
    }


def extract_mdp_parameter_evidence(row: Mapping[str, Any], path_keys: list[str]) -> dict[str, Any]:
    """Return hashed effective parameters and cross-file differences for listed frozen MDP files."""

    sources: dict[str, Any] = {}
    for field_name in path_keys:
        if "mdout_mdp" not in field_name.lower() or not _has_value(row, field_name):
            continue
        path = _resolve_evidence_path(_value(row, field_name))
        if path is None:
            sources[field_name] = {
                "status": "path_unavailable",
                "source_path": str(_value(row, field_name)),
                "parser_version": MDP_EVIDENCE_PARSER_VERSION,
            }
            continue
        try:
            sources[field_name] = _parse_mdp_effective_parameters(path)
        except OSError as exc:
            sources[field_name] = {
                "status": "unreadable",
                "source_path": str(path),
                "parser_version": MDP_EVIDENCE_PARSER_VERSION,
                "error_type": type(exc).__name__,
            }

    parsed = {
        field_name: record.get("parameters", {})
        for field_name, record in sources.items()
        if isinstance(record, Mapping) and record.get("status") == "parsed"
    }
    differences: dict[str, dict[str, str]] = {}
    if len(parsed) >= 2:
        all_keys = sorted({key for parameters in parsed.values() for key in parameters})
        for key in all_keys:
            values = {field_name: str(parameters.get(key, "MISSING")) for field_name, parameters in parsed.items()}
            if len(set(values.values())) > 1:
                differences[key] = values
    return {
        "parser_version": MDP_EVIDENCE_PARSER_VERSION,
        "sources": sources,
        "effective_parameter_differences": differences,
    }


def _ordered_unique(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _missing_values(row: Mapping[str, Any], keys: list[str]) -> list[str]:
    return [key for key in keys if not _has_value(row, key)]


def _add_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _split_warn_source_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_split_warn_source_tokens(item))
        return tokens
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    for sep in [";", "|", "\n", "\t"]:
        text = text.replace(sep, ",")
    raw_tokens = []
    for part in text.split(","):
        raw_tokens.extend(part.strip().split())
    return [token.strip("[]'\"") for token in raw_tokens if token.strip("[]'\"")]


def collect_warn_sources(row: Mapping[str, Any]) -> list[str]:
    """Collect registered warning sources from a frozen evidence row."""
    case_status = _text(row, "case_status")
    case_status_lower = case_status.lower()
    warning_reason = _text(row, "warning_reason")
    warning_lower = warning_reason.lower()
    sources: list[str] = []

    for key in ["warn_source", "warn_sources", "rule_warn_source", "rule_profile_warn_source"]:
        for token in _split_warn_source_tokens(_value(row, key, "")):
            if token in LEGACY_AUDIT_OUTCOME_SOURCES:
                continue
            if is_rule_profile_missing_warn_source(token):
                _add_unique(sources, token)
            elif token in WARN_SOURCES:
                _add_unique(sources, token)

    for token in _split_warn_source_tokens(warning_reason):
        if is_rule_profile_missing_warn_source(token):
            _add_unique(sources, token)

    if (
        _text(row, "missing_reason") == "no_frozen_profile_match"
        or _text(row, "fallback_status") in {"profile_unfreezable", "profile_conflict"}
        or "rule_profile_missing" in warning_lower
    ):
        prop = _text(row, "property") or _text(row, "property_name") or "unknown"
        if prop and prop.lower() != "unknown":
            _add_unique(sources, f"{prop.lower()}_rule_profile_missing")
        else:
            _add_unique(sources, "rule_profile_missing")

    if (
        case_status == "incomplete_md"
        or _upper(row, "density_tier_si_o") == "MISSING"
        or warning_reason == "MISSING_ALL"
        or "missing" in warning_lower
        or "incomplete" in case_status_lower
    ):
        _add_unique(sources, "evidence_incomplete_warn")

    if _upper(row, "density_tier_si_o") == "WARN" or _upper(row, "status_density") == "WARN":
        _add_unique(sources, "density_warn")

    if "MSD:WARN" in warning_reason or _upper(row, "status_MSD") == "WARN" or case_status == "completed_msd_warn":
        _add_unique(sources, "msd_warn")

    if _upper(row, "status_Rg") == "WARN" or _upper(row, "status_rg") == "WARN":
        _add_unique(sources, "rg_warn")
    if _upper(row, "status_RDF") == "WARN" or _upper(row, "rdf_validation_status") == "WARN":
        _add_unique(sources, "rdf_warn")
    if _upper(row, "status_shape") == "WARN" or _upper(row, "status_ShapeTensor") == "WARN":
        _add_unique(sources, "shape_warn")
    if _upper(row, "status_contact") == "WARN":
        _add_unique(sources, "contact_warn")
    if _upper(row, "status_packing") == "WARN" or _upper(row, "packing_qc_status") == "WARN":
        _add_unique(sources, "packing_warn")
    if _upper(row, "status_FsQ") == "WARN" or _upper(row, "status_fsqt") == "WARN":
        _add_unique(sources, "fsqt_warn")

    if "equilibration" in warning_lower or _upper(row, "status_equilibration") == "WARN":
        _add_unique(sources, "equilibration_warn")
    if "runtime" in warning_lower or "checkpoint" in warning_lower or "fatal" in warning_lower:
        _add_unique(sources, "runtime_warn")
    if "topology" in warning_lower or "builder" in warning_lower or "emc" in warning_lower:
        _add_unique(sources, "topology_warn")
    if "force field" in warning_lower or _falsey(_value(row, "force_field_supported", "")):
        _add_unique(sources, "force_field_warn")
    if "provenance" in warning_lower or _upper(row, "engine_qualified") == "WARN":
        _add_unique(sources, "provenance_warn")

    if not sources and ("warn" in warning_lower or "warn" in case_status_lower or not _truthy(_value(row, "ml_usable", True))):
        _add_unique(sources, "unknown_warn")
    return sources


def _candidate(
    diagnosis: str,
    *,
    basis: list[str],
    confidence: float,
    rationale: str,
    requires_freeze: bool = False,
) -> dict[str, Any]:
    route = DIAGNOSIS_TO_ROUTE[diagnosis]
    return {
        "diagnosis": diagnosis,
        "diagnosis_basis": basis,
        "confidence": round(float(confidence), 3),
        "rationale": rationale,
        "requires_freeze": bool(requires_freeze),
        "recommended_reentry_module": route["module"],
        "recommended_action": route["repair_action"],
        "repair_tool": route["repair_tool"],
        "tool_parameters": dict(route["tool_parameters"]),
    }


def normalize_tool_parameters(diagnosis: str, requested: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge model-requested parameters with the registered route, keeping only allowlisted values."""
    route = DIAGNOSIS_TO_ROUTE.get(diagnosis, DIAGNOSIS_TO_ROUTE["insufficient_evidence_for_diagnosis"])
    tool = str(route.get("repair_tool", ""))
    normalized = dict(route.get("tool_parameters", {}))
    allowed = TOOL_PARAMETER_ALLOWLIST.get(tool, {})
    for key, value in (requested or {}).items():
        if key not in allowed:
            continue
        allowed_values = allowed[key]
        candidate = _coerce_allowed_parameter(value)
        if candidate in allowed_values:
            normalized[key] = candidate
    return normalized


def _packet(
    row: Mapping[str, Any],
    warn_source: str,
    *,
    status_keys: list[str],
    numeric_keys: list[str],
    source_path_keys: list[str] | None = None,
    protocol_log_keys: list[str] | None = None,
    structure_trajectory_keys: list[str] | None = None,
    diagnostic_plot_keys: list[str] | None = None,
    required_keys: list[str] | None = None,
    sufficient: bool,
    reasons: list[str],
) -> dict[str, Any]:
    route_hints = {
        source: DIAGNOSIS_TO_ROUTE[diagnosis]
        for source, diagnosis in {
            "density_warn": "engine_or_protocol_sensitive_result",
            "msd_warn": "fit_window_or_method_issue",
            "packing_warn": "packing_void_or_local_density_issue",
            "runtime_warn": "fatal_runtime_error",
            "rdf_warn": "rdf_cutoff_or_normalization_issue",
            "shape_warn": "shape_tensor_schema_issue",
            "contact_warn": "contact_definition_issue",
            "fsqt_warn": "fsqt_decay_window_issue",
            "rg_warn": "postprocess_input_missing",
            "evidence_incomplete_warn": "missing_md_outputs",
        }.items()
    }
    if is_rule_profile_missing_warn_source(warn_source):
        route = DIAGNOSIS_TO_ROUTE["rule_profile_missing"]
    else:
        route = route_hints.get(warn_source, DIAGNOSIS_TO_ROUTE["insufficient_evidence_for_diagnosis"])
    repair_affordances = [
        {
            "diagnosis": next(
                (diagnosis for diagnosis, candidate_route in DIAGNOSIS_TO_ROUTE.items() if candidate_route is route),
                "insufficient_evidence_for_diagnosis",
            ),
            "module": route["module"],
            "repair_tool": route["repair_tool"],
            "tool_parameters": dict(route["tool_parameters"]),
        }
    ]
    source_keys = _ordered_unique((source_path_keys or []) + (structure_trajectory_keys or []) + (diagnostic_plot_keys or []))
    mdp_parameter_evidence = extract_mdp_parameter_evidence(row, protocol_log_keys or [])
    return {
        "warn_source": warn_source,
        "sufficient": bool(sufficient),
        "status_fields": _present_values(row, status_keys),
        "numeric_fields": _present_values(row, numeric_keys),
        "source_paths": _present_values(row, source_keys),
        "evidence_modalities": {
            "scalar_descriptors": _present_values(row, numeric_keys),
            "protocol_and_logs": _present_values(row, protocol_log_keys or []),
            "effective_mdp_parameters": mdp_parameter_evidence,
            "structure_and_trajectory": _present_values(row, structure_trajectory_keys or []),
            "diagnostic_plots": _present_values(row, diagnostic_plot_keys or []),
            "repair_affordances": repair_affordances,
        },
        "missing_fields": _missing_values(row, required_keys or []),
        "sufficiency_reasons": reasons,
        "available_repair_tools": repair_affordances,
    }


def build_warn_evidence_packets(
    row: Mapping[str, Any],
    warn_sources: list[str],
    clean_state: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build per-WARN evidence packets used by deterministic and LLM audit."""
    packets: dict[str, dict[str, Any]] = {}
    for warn_source in warn_sources:
        if is_rule_profile_missing_warn_source(warn_source):
            prop = property_from_rule_profile_missing_warn_source(warn_source)
            value_keys = [
                "computed_value",
                "value",
                prop,
                f"{prop}_value",
                f"{prop}_mean",
                "rho_final_gcc_si_o",
                "D_mean_m2_s",
                "rg_mean_nm",
                "rg2_mean_nm2",
            ]
            has_context = any(
                _has_value(row, key)
                for key in [
                    "polymer_family",
                    "polymer_type",
                    "force_field",
                    "engine_scope",
                    "engine_version",
                    "temperature_K",
                    "pressure_bar",
                    "state",
                ]
            )
            packets[warn_source] = _packet(
                row,
                warn_source,
                status_keys=[
                    "property",
                    "polymer_family",
                    "polymer_type",
                    "force_field",
                    "engine_scope",
                    "engine_version",
                    "missing_reason",
                    "automation_level",
                    "fallback_status",
                ],
                numeric_keys=value_keys,
                source_path_keys=[
                    "rule_profile_discovery_result_json",
                    "similarity_transfer_result_json",
                    "rule_evidence_pack_jsonl",
                    "candidate_rule_profile_json",
                    "similarity_transfer_profile_json",
                    "zotero_evidence_path",
                    "polyinfo_export_path",
                    "control_evidence_path",
                ],
                protocol_log_keys=[
                    "temperature_K",
                    "pressure_bar",
                    "state",
                    "force_field",
                    "engine_scope",
                    "engine_version",
                    "missing_reason",
                    "automation_level",
                    "fallback_status",
                    "rule_basis",
                    "similarity_reference_id",
                    "similarity_score",
                    "similarity_distance_reason",
                    "u_similarity",
                ],
                structure_trajectory_keys=[],
                diagnostic_plot_keys=[],
                required_keys=[],
                sufficient=has_context,
                reasons=["rule_profile_missing_context_available"] if has_context else [],
            )
        elif warn_source == "density_warn":
            required = ["density_tier_si_o", "rho_final_gcc_si_o", "engine_version"]
            provenance_keys = [
                "same_input_clean_engine_control",
                "clean_engine_control_status",
                "clean_engine_density_overlay_path",
                "provenance_qualification_note",
            ]
            clean_repair_state = _text(clean_state or {}, "clean_repair_state")
            has_existing_clean_state = clean_repair_state in {
                "active_clean_queue",
                "clean_queue_waiting_for_slots",
                "clean_nvt_done",
                "clean_postprocess_closed",
                "clean_runtime_fatal",
            }
            has_density = _has_value(row, "density_tier_si_o") and (
                _has_value(row, "rho_final_gcc_si_o") or _has_value(row, "status_density") or has_existing_clean_state
            )
            has_engine = _has_value(row, "engine_version") or _has_value(row, "engine_qualified") or has_existing_clean_state
            has_provenance = any(_has_value(row, key) for key in provenance_keys) or has_existing_clean_state
            sufficient = has_density and has_engine and has_provenance
            reasons = []
            if has_density:
                reasons.append("density_tier_or_rho_available")
            if has_engine:
                reasons.append("engine_or_engine_qualification_available")
            if has_provenance:
                reasons.append("provenance_or_clean_control_available")
            packets[warn_source] = _packet(
                row,
                warn_source,
                status_keys=["status_density", "density_tier_si_o", "engine_version", "engine_qualified"],
                numeric_keys=["rho_final_gcc_si_o", "rho_final_gcc", "initial_density_g_per_cc"],
                source_path_keys=[
                    "clean_engine_density_overlay_path",
                    "density_source_path",
                    "pressure_relaxation_evidence_path",
                    "mdout_mdp_path",
                    "gromacs_log_path",
                    "trajectory_source_path",
                    "final_structure_gro_path",
                    "remote_run_dir",
                    "density_timeseries_plot_path",
                    "density_time_series_plot_path",
                    "density_plot_path",
                    "clean_engine_density_overlay_plot_path",
                    "pressure_relaxation_plot_path",
                ],
                protocol_log_keys=[
                    "engine_version",
                    "engine_qualified",
                    "cluster_profile",
                    "mdout_mdp_path",
                    "mdout_mdp",
                    "gromacs_log_path",
                    "md_log_path",
                    "pr_mdout_mdp_path",
                    "nvt_mdout_mdp_path",
                    "provenance_qualification_note",
                    "same_input_clean_engine_control",
                    "clean_engine_control_status",
                ],
                structure_trajectory_keys=[
                    "trajectory_source_path",
                    "final_structure_gro_path",
                    "structure_gro_path",
                    "source_case_dir",
                    "remote_run_dir",
                ],
                diagnostic_plot_keys=[
                    "density_timeseries_plot_path",
                    "density_time_series_plot_path",
                    "density_plot_path",
                    "clean_engine_density_overlay_path",
                    "clean_engine_density_overlay_plot_path",
                    "pressure_relaxation_plot_path",
                    "pressure_relaxation_evidence_path",
                ],
                required_keys=required,
                sufficient=sufficient,
                reasons=reasons,
            )
        elif warn_source == "msd_warn":
            has_msd = any(
                _has_value(row, key)
                for key in [
                    "loglog_slope",
                    "fit_quality",
                    "regime_warning",
                    "msd_window_begin_ps",
                    "msd_window_end_ps",
                    "warning_reason",
                ]
            )
            packets[warn_source] = _packet(
                row,
                warn_source,
                status_keys=["status_MSD", "warning_reason", "fit_quality", "regime_warning", "msd_window_source"],
                numeric_keys=["D_mean_m2_s", "loglog_slope", "fit_begin_ps", "fit_end_ps", "msd_window_begin_ps", "msd_window_end_ps"],
                source_path_keys=[
                    "msd_source_path",
                    "trajectory_source_path",
                    "final_structure_gro_path",
                    "msd_fit_plot_path",
                    "msd_window_plot_path",
                    "msd_window_overlay_path",
                    "msd_overlay_plot_path",
                ],
                protocol_log_keys=["warning_reason", "fit_quality", "regime_warning", "msd_window_source", "msd_window_policy"],
                structure_trajectory_keys=["trajectory_source_path", "final_structure_gro_path", "structure_gro_path", "msd_source_path"],
                diagnostic_plot_keys=["msd_fit_plot_path", "msd_window_plot_path", "msd_window_overlay_path", "msd_overlay_plot_path"],
                required_keys=["status_MSD", "loglog_slope"],
                sufficient=has_msd,
                reasons=["msd_fit_or_warning_evidence_available"] if has_msd else [],
            )
        elif warn_source == "rdf_warn":
            packets[warn_source] = _packet(
                row,
                warn_source,
                status_keys=["status_RDF", "rdf_validation_status", "rdf_source_status"],
                numeric_keys=[
                    "trajectory_RDF_peak_r_nm",
                    "trajectory_RDF_peak_g_r",
                    "rdf_first_peak_r_nm",
                    "rdf_first_peak_g",
                    "rdf_cutoff_nm",
                ],
                source_path_keys=[
                    "rdf_source_path",
                    "rdf_summary_path",
                    "trajectory_source_path",
                    "final_structure_gro_path",
                    "rdf_plot_path",
                    "rdf_validation_plot_path",
                ],
                protocol_log_keys=["status_RDF", "rdf_validation_status", "rdf_source_status"],
                structure_trajectory_keys=["trajectory_source_path", "final_structure_gro_path", "structure_gro_path", "rdf_source_path"],
                diagnostic_plot_keys=["rdf_plot_path", "rdf_validation_plot_path"],
                required_keys=[],
                sufficient=True,
                reasons=["registered_rdf_warn_source_present"],
            )
        elif warn_source == "packing_warn":
            has_packing = any(
                _has_value(row, key)
                for key in [
                    "void_fraction",
                    "low_density_voxel_fraction",
                    "largest_void_radius_nm",
                    "min_nonbonded_distance_nm",
                    "packing_uniformity_score",
                ]
            )
            packets[warn_source] = _packet(
                row,
                warn_source,
                status_keys=["status_packing", "void_flag", "packing_method"],
                numeric_keys=[
                    "void_fraction",
                    "low_density_voxel_fraction",
                    "largest_connected_low_density_fraction",
                    "largest_void_radius_nm",
                    "min_nonbonded_distance_nm",
                    "packing_uniformity_score",
                    "box_occupancy_ratio",
                    "packing_grid_spacing_nm",
                ],
                source_path_keys=[
                    "packing_source_path",
                    "packing_summary_path",
                    "trajectory_source_path",
                    "final_structure_gro_path",
                    "packing_plot_path",
                    "void_voxel_plot_path",
                ],
                protocol_log_keys=["status_packing", "packing_method", "packing_grid_spacing_nm", "packing_missing_reason"],
                structure_trajectory_keys=[
                    "packing_source_path",
                    "trajectory_source_path",
                    "final_structure_gro_path",
                    "structure_gro_path",
                ],
                diagnostic_plot_keys=["packing_plot_path", "void_voxel_plot_path", "packing_summary_path"],
                required_keys=["void_fraction", "packing_source_path"],
                sufficient=has_packing,
                reasons=["packing_void_numeric_evidence_available"] if has_packing else [],
            )
        elif warn_source == "fsqt_warn":
            packets[warn_source] = _packet(
                row,
                warn_source,
                status_keys=["status_FsQ", "status_fsqt", "fsqt_status"],
                numeric_keys=["tau_alpha_ps", "fsqt_tau_alpha_ps", "fsqt_q_nm_inv", "fsqt_decay_fraction"],
                source_path_keys=[
                    "fsqt_source_path",
                    "fsqt_summary_path",
                    "trajectory_source_path",
                    "final_structure_gro_path",
                    "fsqt_plot_path",
                    "fsqt_decay_plot_path",
                ],
                protocol_log_keys=["status_FsQ", "status_fsqt", "fsqt_status", "fsqt_window_policy"],
                structure_trajectory_keys=["trajectory_source_path", "final_structure_gro_path", "structure_gro_path", "fsqt_source_path"],
                diagnostic_plot_keys=["fsqt_plot_path", "fsqt_decay_plot_path", "fsqt_summary_path"],
                required_keys=[],
                sufficient=True,
                reasons=["registered_fsqt_warn_source_present"],
            )
        elif warn_source == "evidence_incomplete_warn":
            packets[warn_source] = _packet(
                row,
                warn_source,
                status_keys=["case_status", "warning_reason", "density_tier_si_o"],
                numeric_keys=[],
                source_path_keys=[
                    "source_case_dir",
                    "source_input_archive",
                    "trajectory_source_path",
                    "final_structure_gro_path",
                    "checkpoint_path",
                    "remote_run_dir",
                    "gromacs_log_path",
                ],
                protocol_log_keys=["case_status", "warning_reason", "checkpoint_path", "gromacs_log_path"],
                structure_trajectory_keys=["source_case_dir", "source_input_archive", "trajectory_source_path", "final_structure_gro_path", "remote_run_dir"],
                diagnostic_plot_keys=[],
                required_keys=[],
                sufficient=True,
                reasons=["missing_outputs_are_the_evidence"],
            )
        elif warn_source == "runtime_warn":
            runtime_keys = ["warning_reason", "clean_fatal_stage", "clean_fatal_log", "clean_fatal_reason"]
            runtime_row = {**dict(row), **dict(clean_state or {})}
            packets[warn_source] = _packet(
                runtime_row,
                warn_source,
                status_keys=runtime_keys,
                numeric_keys=[],
                source_path_keys=["clean_fatal_log", "remote_run_dir", "checkpoint_path", "gromacs_log_path"],
                protocol_log_keys=["warning_reason", "clean_fatal_stage", "clean_fatal_log", "clean_fatal_reason", "checkpoint_path", "gromacs_log_path"],
                structure_trajectory_keys=["remote_run_dir"],
                diagnostic_plot_keys=[],
                required_keys=["warning_reason"],
                sufficient=any(_has_value(runtime_row, key) for key in runtime_keys),
                reasons=["runtime_or_clean_fatal_evidence_available"],
            )
        else:
            packets[warn_source] = _packet(
                row,
                warn_source,
                status_keys=["case_status", "warning_reason", f"status_{warn_source.split('_warn')[0]}"],
                numeric_keys=[],
                source_path_keys=[],
                required_keys=[],
                sufficient=True,
                reasons=["registered_warn_source_present"],
            )
    return packets


def evidence_sufficiency_summary(packets: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        warn_source: {
            "sufficient": bool(packet.get("sufficient", False)),
            "missing_fields": list(packet.get("missing_fields", [])),
            "sufficiency_reasons": list(packet.get("sufficiency_reasons", [])),
        }
        for warn_source, packet in packets.items()
    }


def infer_diagnoses(
    row: Mapping[str, Any],
    warn_sources: list[str],
    clean_state: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Infer candidate diagnoses from warning sources and frozen evidence."""
    evidence_packets = build_warn_evidence_packets(row, warn_sources, clean_state)

    def evidence_ok(source: str) -> bool:
        return bool(evidence_packets.get(source, {}).get("sufficient", False))

    def insufficient_candidate(source: str) -> dict[str, Any]:
        packet = evidence_packets.get(source, {})
        missing = list(packet.get("missing_fields", []))
        return _candidate(
            "insufficient_evidence_for_diagnosis",
            basis=[f"{source}_evidence_packet"] + [f"missing:{field}" for field in missing],
            confidence=0.62,
            rationale=f"{source} was triggered, but the evidence packet is insufficient for a narrower repair diagnosis",
            requires_freeze=True,
        )

    warning_lower = _text(row, "warning_reason").lower()
    clean_reason_lower = " ".join(
        [
            _text(clean_state or {}, "clean_repair_state"),
            _text(clean_state or {}, "clean_fatal_stage"),
            _text(clean_state or {}, "clean_fatal_log"),
            _text(clean_state or {}, "clean_fatal_reason"),
        ]
    ).lower()
    runtime_lower = f"{warning_lower} {clean_reason_lower}".strip()
    candidates: list[dict[str, Any]] = []

    if _falsey(_value(row, "force_field_supported", "")) or "force field backend unsupported" in warning_lower:
        candidates.append(
            _candidate(
                "unsupported_by_current_system",
                basis=["force_field_or_backend_allowlist"],
                confidence=0.98,
                rationale="force-field/backend evidence says the candidate is unsupported by the current workflow",
            )
        )
    elif "force_field_warn" in warn_sources:
        candidates.append(
            _candidate(
                "force_field_assignment_unsupported",
                basis=["force_field_assignment_evidence"],
                confidence=0.9,
                rationale="force-field assignment is unsupported for the current candidate or backend",
            )
        )

    if "topology_warn" in warn_sources:
        if "unsupported" in warning_lower:
            diagnosis = "topology_unsupported"
            confidence = 0.92
        else:
            diagnosis = "emc_builder_failed"
            confidence = 0.82
        candidates.append(
            _candidate(
                diagnosis,
                basis=["builder_or_topology_log", "registered_builder_adapter"],
                confidence=confidence,
                rationale="construction/topology evidence should be handled by the EMC module or marked unsupported",
            )
        )

    if "evidence_incomplete_warn" in warn_sources:
        candidates.append(
            _candidate(
                "missing_md_outputs",
                basis=["quality_mask_case_status", "source_input_archive"],
                confidence=0.95,
                rationale="missing MD/postprocessing outputs are recoverable through the MD module when archived inputs exist",
            )
        )

    if "runtime_warn" in warn_sources:
        if "checkpoint" in runtime_lower:
            diagnosis = "checkpoint_resume_failure"
        elif any(marker in runtime_lower for marker in ("fatal", "cuda error", "illegaladdress", "illegal address", "lincs")):
            diagnosis = "fatal_runtime_error"
        else:
            diagnosis = "trajectory_missing_or_corrupt"
        candidates.append(
            _candidate(
                diagnosis,
                basis=["runtime_log", "checkpoint_state"],
                confidence=0.84,
                rationale="runtime evidence points to MD execution recovery rather than postprocessing or EMC rebuild",
            )
        )

    profile_missing_sources = [source for source in warn_sources if is_rule_profile_missing_warn_source(source)]
    for source in profile_missing_sources:
        prop = property_from_rule_profile_missing_warn_source(source)
        if not evidence_ok(source):
            candidates.append(insufficient_candidate(source))
        else:
            candidate = _candidate(
                "rule_profile_missing",
                basis=[source, f"property:{prop}", "no_frozen_rule_profile_match"],
                confidence=0.9,
                rationale=(
                    f"{source} means the computed evidence exists or may exist, but no frozen {prop} "
                    "rule_profile matched the case context; route to automatic structured-evidence or similar-structure "
                    "calibration, not MD repair"
                ),
                requires_freeze=True,
            )
            if prop != "unknown":
                candidate["tool_parameters"]["property_name"] = prop
            candidates.append(candidate)

    if "msd_warn" in warn_sources:
        if not evidence_ok("msd_warn"):
            candidates.append(insufficient_candidate("msd_warn"))
        else:
            candidates.append(
                _candidate(
                    "fit_window_or_method_issue",
                    basis=["c100_aware_msd_window_scan", "strict_alternative_fit_windows"],
                    confidence=0.9,
                    rationale="MSD warning is best handled by method/window repair unless trajectory evidence is missing or corrupt",
                )
            )

    if "rdf_warn" in warn_sources:
        candidates.append(
            _candidate(
                "rdf_cutoff_or_normalization_issue",
                basis=["rdf_suite_status", "cutoff_or_normalization_schema"],
                confidence=0.8,
                rationale="RDF warning is a postprocessing/cutoff provenance issue before it is an MD rerun issue",
            )
        )

    if "shape_warn" in warn_sources:
        candidates.append(
            _candidate(
                "shape_tensor_schema_issue",
                basis=["shape_tensor_schema", "shape_summary_status"],
                confidence=0.78,
                rationale="shape tensor warning should re-enter the shape postprocessing module",
            )
        )

    if "contact_warn" in warn_sources:
        candidates.append(
            _candidate(
                "contact_definition_issue",
                basis=["contact_cutoff_provenance", "contact_postprocess_status"],
                confidence=0.78,
                rationale="contact warning should re-enter contact postprocessing or cutoff provenance review",
            )
        )

    if "packing_warn" in warn_sources:
        if not evidence_ok("packing_warn"):
            candidates.append(insufficient_candidate("packing_warn"))
        else:
            candidates.append(
                _candidate(
                    "packing_void_or_local_density_issue",
                    basis=["packing_void_metrics", "local_density_grid", "final_coordinate_or_trajectory_evidence"],
                    confidence=0.8,
                    rationale="packing/void warning should re-enter structure-level postprocessing before MD rerun",
                )
            )

    if "fsqt_warn" in warn_sources:
        candidates.append(
            _candidate(
                "fsqt_decay_window_issue",
                basis=["fsqt_decay_status", "tau_alpha_window"],
                confidence=0.78,
                rationale="Fs(q,t) warning should re-enter the Fs(q,t) postprocessing module",
            )
        )

    if "rg_warn" in warn_sources:
        candidates.append(
            _candidate(
                "postprocess_input_missing",
                basis=["rg_postprocess_status"],
                confidence=0.7,
                rationale="Rg warning requires postprocess input/status inspection before MD rerun",
            )
        )

    if "density_warn" in warn_sources:
        if not evidence_ok("density_warn"):
            candidates.append(insufficient_candidate("density_warn"))
        else:
            candidates.append(
                _candidate(
                    "engine_or_protocol_sensitive_result",
                    basis=[
                        "same_input_clean_engine_control",
                        "effective_run_parameter_provenance",
                        "mdout_mdp_effective_parameter_differences",
                        "density_tier_comparison",
                    ],
                    confidence=0.85,
                    rationale="PE-LCB100 density WARN is supported by same-input clean-control and effective-run-parameter evidence as protocol-sensitive",
                    requires_freeze=True,
                )
            )

    if "equilibration_warn" in warn_sources:
        candidates.append(
            _candidate(
                "insufficient_equilibration",
                basis=["equilibration_status", "thermal_or_pressure_stability_evidence"],
                confidence=0.75,
                rationale="equilibration warning should re-enter MD only under a frozen protocol action",
                requires_freeze=True,
            )
        )

    if "provenance_warn" in warn_sources:
        candidates.append(
            _candidate(
                "force_field_systematic_offset",
                basis=["engine_force_field_provenance"],
                confidence=0.65,
                rationale="provenance warning may require rule/protocol qualification rather than rerun",
                requires_freeze=True,
            )
        )

    if "unknown_warn" in warn_sources or not candidates:
        candidates.append(
            _candidate(
                "insufficient_evidence_for_diagnosis",
                basis=["unclassified_warning_state"],
                confidence=0.5,
                rationale="warning exists but no registered evidence pattern supports a narrower deterministic repair",
                requires_freeze=True,
            )
        )
    return candidates


def select_primary_diagnosis(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the primary diagnosis by deterministic priority, then confidence."""
    return sorted(
        candidates,
        key=lambda item: (
            DIAGNOSIS_PRIORITY.get(str(item["diagnosis"]), 999),
            -float(item.get("confidence", 0.0)),
            str(item["diagnosis"]),
        ),
    )[0]


def _active_profile_route(diagnosis: str, profile_dir: str | Path | None = None) -> dict[str, Any] | None:
    for profile in list_active_diagnosis_profiles(profile_dir or DIAGNOSIS_PROFILE_DIR):
        if str(profile.get("diagnosis")) == str(diagnosis):
            return route_from_profile(profile)
    return None


def route_from_primary_diagnosis(diagnosis: str, profile_dir: str | Path | None = None) -> dict[str, Any]:
    """Return route metadata for a primary diagnosis name."""
    if diagnosis in DIAGNOSIS_TO_ROUTE:
        return dict(DIAGNOSIS_TO_ROUTE[diagnosis])
    profile_route = _active_profile_route(diagnosis, profile_dir)
    if profile_route:
        return dict(profile_route)
    return dict(DIAGNOSIS_TO_ROUTE["insufficient_evidence_for_diagnosis"])


def _route_from_guarded_proposal(diagnosis: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(diagnosis.get("guard_verdict", "")) != "pass":
        return None
    proposal = diagnosis.get("llm_tool_call_proposal")
    if not isinstance(proposal, Mapping):
        return None
    tool = str(proposal.get("tool") or "").strip()
    affordances = repair_tool_affordances()
    if tool not in affordances:
        return None
    module_name = str(affordances[tool]["module"])
    module = dict(DETERMINISTIC_MODULES.get(module_name, DETERMINISTIC_MODULES["audit_review"]))
    module["tool"] = tool
    return {
        "repair_route": "guarded_llm_tool_affordance",
        "repair_action": "guarded_tool_call",
        "automation_level": "guarded_sidecar_tool_plan",
        "executor": "MCP bounded tool interface",
        "module": module_name,
        "repair_tool": tool,
        "tool_parameters": dict(proposal.get("parameters") or {}),
        "deterministic_module": module,
        "tool_plan": {
            "tool": tool,
            "parameters": dict(proposal.get("parameters") or {}),
        },
    }


def _base_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "case_id": _text(row, "case_id"),
        "case_status": _text(row, "case_status"),
        "density_tier_si_o": _text(row, "density_tier_si_o"),
        "warning_reason": _text(row, "warning_reason"),
        "ml_usable": _truthy(_value(row, "ml_usable", False)),
        "audit_disagreement_count": _int_value(_value(row, "audit_disagreement_count", 0)),
    }


def diagnose_case(row: Mapping[str, Any], clean_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify one release row into a generalized WARN diagnosis record."""
    warn_sources = collect_warn_sources(row)
    if _text(clean_state or {}, "clean_repair_state") == "missing_md_outputs":
        _add_unique(warn_sources, "evidence_incomplete_warn")
    if _text(clean_state or {}, "clean_repair_state") == "clean_runtime_fatal":
        _add_unique(warn_sources, "runtime_warn")

    payload = _base_payload(row)
    evidence_packets = build_warn_evidence_packets(row, warn_sources, clean_state)
    sufficiency = evidence_sufficiency_summary(evidence_packets)
    if not warn_sources and (_truthy(_value(row, "ml_usable", False)) or _text(row, "case_status") == "completed_quality_passed"):
        payload.update(
            {
                "rule_verdict": "PASS",
                "warn_sources": [],
                "warn_source": "none",
                "primary_audit_diagnosis": "none",
                "cause_diagnosis": "none",
                "secondary_audit_diagnoses": [],
                "audit_diagnosis": "none",
                "diagnosis_basis": [],
                "recommended_reentry_module": "audit_review",
                "recommended_action": "none",
                "repair_tool": "none",
                "tool_parameters": {},
                "repair_instruction": {},
                "confidence": 0.7,
                "requires_freeze": False,
                "requires_profile_freeze": False,
                "rationale": "case has no registered warning source and can be released without repair routing",
                "max_repair_attempts": 0,
                "warn_evidence_packets": {},
                "evidence_sufficiency": {},
            }
        )
        return payload

    candidates = infer_diagnoses(row, warn_sources, clean_state)
    primary = select_primary_diagnosis(candidates)
    secondary = [candidate["diagnosis"] for candidate in candidates if candidate["diagnosis"] != primary["diagnosis"]]
    rule_verdict = "FAIL" if primary["diagnosis"] in {"unsupported_by_current_system", "topology_unsupported", "force_field_assignment_unsupported"} else "WARN"

    payload.update(
        {
            "rule_verdict": rule_verdict,
            "warn_sources": warn_sources,
            "warn_source": ",".join(warn_sources) if warn_sources else "none",
            "primary_audit_diagnosis": primary["diagnosis"],
            "cause_diagnosis": primary["diagnosis"],
            "secondary_audit_diagnoses": secondary,
            "audit_diagnosis": primary["diagnosis"],
            "diagnosis_basis": primary["diagnosis_basis"],
            "recommended_reentry_module": primary["recommended_reentry_module"],
            "recommended_action": primary["recommended_action"],
            "repair_tool": primary["repair_tool"],
            "tool_parameters": primary["tool_parameters"],
            "repair_instruction": {
                "repair_tool": primary["repair_tool"],
                "tool_parameters": primary["tool_parameters"],
                "expected_success_condition": "run the selected deterministic module, write sidecar overlay evidence, and re-enter the rule gate",
            },
            "confidence": primary["confidence"],
            "requires_freeze": primary["requires_freeze"],
            "requires_profile_freeze": primary["requires_freeze"],
            "rationale": primary["rationale"],
            "max_repair_attempts": 0 if rule_verdict == "FAIL" or primary["recommended_reentry_module"].endswith("_review") else 2,
            "warn_evidence_packets": evidence_packets,
            "evidence_sufficiency": sufficiency,
        }
    )
    return payload


def route_from_diagnosis(diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    """Return the repair-route metadata for a diagnosis record."""
    publication_guard = diagnosis.get("publication_repair_guard")
    guard_applied = isinstance(publication_guard, Mapping) and bool(publication_guard.get("applied"))
    guarded_route = None if guard_applied else _route_from_guarded_proposal(diagnosis)
    if guarded_route is not None:
        return guarded_route
    primary = str(
        diagnosis.get("primary_audit_diagnosis")
        or diagnosis.get("audit_diagnosis")
        or "none"
    )
    route = route_from_primary_diagnosis(primary)
    if guard_applied:
        route["repair_route"] = str(diagnosis.get("repair_route") or route["repair_route"])
        route["repair_action"] = str(
            diagnosis.get("recommended_action") or route["repair_action"]
        )
        route["automation_level"] = str(
            diagnosis.get("recommended_automation_level") or route["automation_level"]
        )
        route["executor"] = str(diagnosis.get("recommended_executor") or route["executor"])
    module_name = str(diagnosis.get("recommended_reentry_module") or route.get("module") or "audit_review")
    module = dict(DETERMINISTIC_MODULES.get(module_name, DETERMINISTIC_MODULES["audit_review"]))
    selected_tool = str(diagnosis.get("repair_tool") or route.get("repair_tool") or module["tool"])
    if selected_tool != module.get("tool"):
        module["tool"] = selected_tool
    route["deterministic_module"] = module
    route["tool_plan"] = {
        "tool": selected_tool,
        "parameters": dict(diagnosis.get("tool_parameters") or route.get("tool_parameters") or {}),
    }
    return route


def registered_module_specs() -> dict[str, dict[str, str]]:
    """Return deterministic module contracts exposed by the repair router."""
    return {key: dict(value) for key, value in DETERMINISTIC_MODULES.items()}


def warn_diagnosis_taxonomy(profile_dir: str | Path | None = None) -> dict[str, Any]:
    """Return public WARN source and diagnosis taxonomy for MCP clients."""
    active_profiles = list_active_diagnosis_profiles(profile_dir or DIAGNOSIS_PROFILE_DIR)
    frozen_routes = {str(profile["diagnosis"]): route_from_profile(profile) for profile in active_profiles}
    static_routes = {
        diagnosis: {
            "repair_route": route["repair_route"],
            "module": route["module"],
            "repair_tool": route["repair_tool"],
            "tool_parameters": dict(route["tool_parameters"]),
        }
        for diagnosis, route in DIAGNOSIS_TO_ROUTE.items()
        if diagnosis != "none"
    }
    profile_routes = {
        diagnosis: {
            "repair_route": route["repair_route"],
            "module": route["module"],
            "repair_tool": route["repair_tool"],
            "tool_parameters": dict(route["tool_parameters"]),
        }
        for diagnosis, route in frozen_routes.items()
    }
    return {
        "warn_sources": list(WARN_SOURCES),
        "taxonomy": {
            **{category: list(values) for category, values in DIAGNOSIS_TAXONOMY.items()},
            "frozen_open_diagnosis": list(frozen_routes),
        },
        "frozen_open_diagnoses": list(frozen_routes),
        "routes": {**profile_routes, **static_routes},
    }
