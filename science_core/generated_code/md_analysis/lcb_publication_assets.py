#!/usr/bin/env python3
"""Build publication-oriented provenance, quality, metrics, and ML assets.

This module is intentionally local-only.  It reads the configured LCB project
outputs and writes auditable CSV/JSON/Markdown products under the integrated
aggregate directory.  It never submits jobs or modifies remote state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from lcb_project_config import LCBProjectConfig, load_lcb_config
from pelcb_msd_publication_gate import evaluate_msd_publication_gate
from result_confidence import build_result_confidence_table, confidence_wide_columns

try:
    from agentic.tri_llm_judge import compute_rule_verdict
except ImportError:  # pragma: no cover - keeps generic asset generation usable
    compute_rule_verdict = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FILES = [
    "lcb_project_config.py",
    "lcb300_runtime_dashboard.py",
    "lcb300_supervisor.py",
    "lcb300_orchestrator.py",
    "lcb300_postprocess_closure.py",
    "build_lcb300_overview.py",
    "lcb_publication_assets.py",
]
NON_VISC_STATUS = ["Rg", "RDF", "shape", "MSD", "dynamics", "contact", "density"]
# RDF is included in the automated postprocessing package. After external
# review, OK RDF defaults to "current"; rdf_validation_status remains an
# explicit override for future audits.
QUALITY_GATE_STATUS = ["Rg", "shape", "MSD", "dynamics", "contact", "density"]
STRUCTURAL_PUBLICATION_REQUIRED = ["Rg"]
RDF_VALIDATION_ALLOWED = {"current", "under_external_review", "validated", "excluded"}
RDF_USABLE_STATUSES = {"current", "validated"}
FEATURE_CANDIDATES = [
    "branch_points",
    "branch_carbons",
    "actual_total_carbons_per_chain",
    "backbone_repeat_units",
    "rg_mean_nm",
    "rg2_mean_nm2",
    "graph_diameter2_over_6rg2",
    "graph_diameter_topological_carbons",
    "shape_rg_carbon_mean_nm",
    "shape_rg2_carbon_mean_nm2",
    "shape_asphericity_mean_nm2",
    "shape_acylindricity_mean_nm2",
    "shape_relative_anisotropy_mean",
    "P_contact_rc0p8_s10",
    "P_contact_rc0p8_s20",
    "P_contact_rc0p8_s50",
    "rdf_all_first_peak_g",
    "rdf_excl_first_peak_g",
    "rdf_inter_chain_cc_first_peak_g",
    "rdf_intra_chain_nonbonded_cc_first_peak_g",
    "Density_mean",
]
TARGET_CANDIDATES = [
    "D_mean_m2_s",
    "D_mean_1e-5_cm2_s",
    "loglog_slope",
    "tau_fs_ps",
    "tau_alpha_ps",
    "g1_over_g3_10ns",
    "g1_monomer_over_g3_10ns",
]
CATEGORICAL_FEATURES = [
    "topology_family",
    "branch_topology",
    "branch_position_pattern",
    "regime_tag",
]
DECISION_TYPES = ("equilibration", "fit_quality", "anomaly")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def portable_path(path: Path | str, base: Path) -> str:
    """Return a release-portable path when an artifact is inside ``base``."""
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return str(path)


def relative_portable_path(path: Path | str, base: Path) -> str:
    """Return a relative path without leaking workstation-specific roots."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()
    except Exception:
        return str(path)


def stable_row_hash(row: pd.Series | dict[str, Any]) -> str:
    data = dict(row)
    normalized = {}
    for key, value in sorted(data.items(), key=lambda item: str(item[0])):
        try:
            missing = bool(pd.isna(value))
        except Exception:
            missing = False
        normalized[str(key)] = "" if missing else value
    text = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def present(value: Any) -> bool:
    text = str(value if value is not None else "").strip()
    return text != "" and text.lower() not in {"nan", "none", "missing"}


def number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def status_value(row: pd.Series, name: str) -> str:
    value = row.get(f"status_{name}", "MISSING")
    if not present(value):
        return "MISSING"
    return str(value).strip().upper()


def rdf_validation_status(row: pd.Series | dict[str, Any]) -> str:
    existing = str(row.get("rdf_validation_status", "") or "").strip().lower()
    if existing in RDF_VALIDATION_ALLOWED:
        return existing
    return "current" if status_value(row, "RDF") == "OK" else "excluded"


def non_visc_warn_fail(row: pd.Series) -> list[str]:
    bad = []
    for name in QUALITY_GATE_STATUS:
        status = status_value(row, name)
        if status in {"WARN", "FAIL", "MISSING"}:
            bad.append(f"{name}:{status}")
    return bad


def missing_statuses(row: pd.Series) -> list[str]:
    return [name for name in QUALITY_GATE_STATUS if status_value(row, name) == "MISSING"]


def publication_grade(row: pd.Series) -> bool:
    if non_visc_warn_fail(row):
        return False
    return all(status_value(row, name) == "OK" for name in QUALITY_GATE_STATUS)


def ml_usable(row: pd.Series) -> bool:
    return bool_value(row.get("any_analyzed")) and not non_visc_warn_fail(row)


def warning_reason(row: pd.Series) -> str:
    reasons = non_visc_warn_fail(row)
    if status_value(row, "RDF") == "OK" and rdf_validation_status(row) == "under_external_review":
        reasons.append("RDF_UNDER_EXTERNAL_REVIEW")
    if not reasons and not bool_value(row.get("any_analyzed")):
        reasons.append("MISSING_ALL")
    if status_value(row, "MSD") == "WARN":
        for col in ["fit_quality", "regime_warning", "diffusion_confidence", "msd_publication_issues"]:
            if present(row.get(col)):
                reasons.append(f"MSD_{col}={row.get(col)}")
    return ";".join(dict.fromkeys(reasons)) if reasons else "OK"


def script_versions() -> dict[str, str]:
    return {
        filename: file_sha256(SCRIPT_DIR / filename)
        for filename in SOURCE_FILES
        if (SCRIPT_DIR / filename).exists()
    }


def first_existing_case_json(config: LCBProjectConfig, case_id: str) -> Path | None:
    candidates: list[Path] = []
    for root in config.case_roots:
        candidates.extend([root.path / case_id / "case.json", root.path / f"{case_id}.json"])
    for root in config.bridge_roots:
        candidates.extend([root / case_id / "case.json", root / f"{case_id}.json"])
    for path in candidates:
        if path.exists():
            return path
    return None


def first_existing_dir(roots: list[tuple[str, Path]], case_id: str) -> tuple[str, Path] | tuple[str, None]:
    for label, root in roots:
        exact = root / case_id
        if exact.exists():
            return label, exact
        matches = sorted(root.glob(f"{case_id}*")) if root.exists() else []
        for match in matches:
            if match.is_dir():
                return label, match
    return "", None


def build_quality_columns(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    msd_results = out.apply(lambda row: evaluate_msd_publication_gate(row, require_clean=True), axis=1)
    out["status_MSD"] = [result["status_MSD"] for result in msd_results]
    out["msd_publication_grade"] = [result["msd_publication_grade"] for result in msd_results]
    out["msd_publication_issues"] = [result["issue_string"] for result in msd_results]
    out["msd_publication_diagnostic_issues"] = [result["diagnostic_issue_string"] for result in msd_results]
    out["msd_displacement_amplitude_check"] = [result["displacement_amplitude_check"] for result in msd_results]
    out["msd_diffusion_confidence"] = [result["msd_diffusion_confidence"] for result in msd_results]
    out["msd_clean_engine_source"] = [result["clean_engine_source"] for result in msd_results]
    out["msd_tail_window_ok"] = [result["tail_window_ok"] for result in msd_results]
    out["msd_fit_R2_publication"] = [result["fit_R2"] for result in msd_results]
    out["rdf_validation_status"] = out.apply(rdf_validation_status, axis=1)
    out["rdf_publication_usable"] = out["rdf_validation_status"].isin(RDF_USABLE_STATUSES)
    out["publication_grade"] = out.apply(publication_grade, axis=1)
    out["ml_usable"] = out.apply(ml_usable, axis=1)
    out["warning_reason"] = out.apply(warning_reason, axis=1)
    out["missing_nonvisc_observables"] = out.apply(lambda row: ";".join(missing_statuses(row)), axis=1)
    return out


def merge_shape_tensor_features(config: LCBProjectConfig, wide: pd.DataFrame) -> pd.DataFrame:
    """Add self-consistent carbon-skeleton shape-tensor fields for ML export."""
    shape_path = config.integrated_root / "ShapeTensor" / "summary" / "shape_tensor_summary.csv"
    shape = read_csv(shape_path)
    if shape.empty or "case_id" not in shape.columns:
        return wide
    rename = {
        "rg_carbon_mean_nm": "shape_rg_carbon_mean_nm",
        "rg2_carbon_mean_nm2": "shape_rg2_carbon_mean_nm2",
        "asphericity_mean_nm2": "shape_asphericity_mean_nm2",
        "acylindricity_mean_nm2": "shape_acylindricity_mean_nm2",
        "relative_shape_anisotropy_mean": "shape_relative_anisotropy_mean",
    }
    keep = ["case_id", *[col for col in rename if col in shape.columns]]
    if len(keep) == 1:
        return wide
    shape = shape[keep].rename(columns=rename)
    wide = wide.drop(columns=[col for col in rename.values() if col in wide.columns], errors="ignore")
    return wide.merge(shape, on="case_id", how="left")


def latest_existing_file(root: Path, glob_pattern: str) -> Path | None:
    if not root.exists():
        return None
    matches = [path for path in root.glob(glob_pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def a3_group_results_path(config: LCBProjectConfig) -> Path | None:
    root = SCRIPT_DIR / "agentic" / "results" / config.project_id
    return latest_existing_file(root, "a3_locked_eval_*/a3_group_results.csv")


def t1_labels_path(config: LCBProjectConfig) -> Path | None:
    root = SCRIPT_DIR / "agentic" / "groundtruth" / config.project_id
    return latest_existing_file(root, "locked_eval_*/labels_human_t1_expert_normalized_*.csv")


def raw_judgments_path(config: LCBProjectConfig) -> Path | None:
    root = SCRIPT_DIR / "agentic" / "results" / config.project_id
    return latest_existing_file(root, "a3_locked_eval_*/raw_judgments.jsonl")


def release_rule_evidence(row: pd.Series) -> dict[str, Any]:
    """Build the minimal frozen-evidence view needed for release rule columns.

    The public ML table records release-facing rule summaries. Missing MD rows
    are represented as ``missing`` before this evidence is evaluated; analyzed
    rows use the same rule implementation as the A3 audit benchmark.
    """
    return {
        "case_id": row.get("case_id", ""),
        "thermal": {
            "temperature_mean_K": number_or_none(row.get("temperature_k")),
            "early_late_mean_delta_K": number_or_none(row.get("early_late_mean_delta_K")),
            "late_linear_slope_K_per_sample": number_or_none(row.get("late_linear_slope_K_per_sample")),
        },
        "energetic": {
            "total_energy_late_slope_over_abs_mean": number_or_none(row.get("energy_late_slope_normalized")),
        },
        "compaction": {
            "final_density_g_per_cc": number_or_none(row.get("rho_final_gcc"))
            or number_or_none(row.get("Density_mean")),
            "box_volume_late_slope_over_abs_mean": number_or_none(row.get("box_volume_late_slope_over_abs_mean")),
        },
        "void": {
            "void_flag": row.get("void_flag") if isinstance(row.get("void_flag"), bool) else None,
            "low_density_voxel_fraction": number_or_none(row.get("low_density_voxel_fraction")),
            "largest_connected_low_density_fraction": number_or_none(row.get("largest_connected_low_density_fraction")),
        },
        "fit_quality": {
            "msd_loglog_slope": number_or_none(row.get("loglog_slope")),
            "fit_window_frames": number_or_none(row.get("fit_window_frames")),
            "window_sensitivity": number_or_none(row.get("window_sensitivity")),
        },
        "runtime": {
            "fatal_log_marker_count": number_or_none(row.get("fatal_log_marker_count")) or 0,
            "non_finite_energy_count": number_or_none(row.get("non_finite_energy_count")) or 0,
            "checkpoint_integrity": None,
        },
    }


def fallback_rule_verdict(row: pd.Series, decision_type: str) -> str:
    """Conservative fallback used only if the A3 rule module cannot import."""
    if not bool_value(row.get("any_analyzed")):
        return "missing"
    if decision_type == "fit_quality":
        slope = number_or_none(row.get("loglog_slope"))
        if slope is None:
            return "missing"
        return "fail" if slope < 0.85 or slope > 1.05 else "warn"
    return "warn"


def build_rule_verdict_columns(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in wide.iterrows():
        out: dict[str, Any] = {"case_id": row["case_id"]}
        if not bool_value(row.get("any_analyzed")):
            for decision_type in DECISION_TYPES:
                out[f"rule_{decision_type}"] = "missing"
            rows.append(out)
            continue
        evidence = release_rule_evidence(row)
        for decision_type in DECISION_TYPES:
            if compute_rule_verdict is None:
                verdict = fallback_rule_verdict(row, decision_type)
            else:
                verdict = compute_rule_verdict(evidence, decision_type).get("verdict", "")
            out[f"rule_{decision_type}"] = verdict or "missing"
        rows.append(out)
    return pd.DataFrame(rows)


def build_expert_label_columns(config: LCBProjectConfig) -> tuple[pd.DataFrame, Path | None]:
    path = t1_labels_path(config)
    if path is None:
        return pd.DataFrame(), None
    labels = read_csv(path)
    required = {"case_id", "decision_type", "label"}
    if labels.empty or not required.issubset(labels.columns):
        return pd.DataFrame(), path
    labels = labels[labels["decision_type"].isin(DECISION_TYPES)].copy()
    pivot = labels.pivot_table(
        index="case_id",
        columns="decision_type",
        values="label",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(columns={key: f"expert_label_{key}" for key in DECISION_TYPES})
    return pivot, path


def build_audit_summary_columns(config: LCBProjectConfig) -> tuple[pd.DataFrame, Path | None, Path | None]:
    group_path = a3_group_results_path(config)
    raw_path = raw_judgments_path(config)
    if group_path is None:
        return pd.DataFrame(), None, raw_path
    groups = read_csv(group_path)
    required = {"case_id", "decision_type", "mode", "majority_verdict", "cross_model_disagreement"}
    if groups.empty or not required.issubset(groups.columns):
        return pd.DataFrame(), group_path, raw_path

    audit = groups[(groups["mode"] == "audit") & groups["decision_type"].isin(DECISION_TYPES)].copy()
    pivot = audit.pivot_table(
        index="case_id",
        columns="decision_type",
        values="majority_verdict",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(columns={key: f"audit_majority_{key}" for key in DECISION_TYPES})

    disagreement = groups.copy()
    disagreement["cross_model_disagreement"] = disagreement["cross_model_disagreement"].map(bool_value)
    by_case = disagreement.groupby("case_id", as_index=False).agg(
        audit_disagreement_flag=("cross_model_disagreement", "any"),
        audit_disagreement_count=("cross_model_disagreement", "sum"),
    )
    by_case["a3_locked_eval_member"] = True
    out = pivot.merge(by_case, on="case_id", how="outer")
    return out, group_path, raw_path


def build_case_status(row: pd.Series) -> str:
    if not bool_value(row.get("any_analyzed")):
        return "incomplete_md"
    if not bool_value(row.get("ml_usable")):
        if status_value(row, "MSD") == "WARN":
            return "completed_msd_warn"
        if row.get("warning_reason") not in {"", "OK", None}:
            return "completed_quality_warned"
        return "completed_not_ml_usable"
    tier = str(row.get("compaction_tier_preview", "") or "").upper()
    if tier == "WARN":
        return "completed_engine_qualified_warn"
    return "completed_quality_passed"


def build_provenance_cases(config: LCBProjectConfig, wide: pd.DataFrame) -> pd.DataFrame:
    manifest = read_csv(config.manifest_csv)
    manifest_by_case = manifest.set_index("case_id") if "case_id" in manifest.columns and not manifest.empty else pd.DataFrame()
    runtime = read_csv(config.runtime_output_dir / "runtime_cases.csv")
    runtime = runtime.drop_duplicates("case_id", keep="last") if "case_id" in runtime.columns else pd.DataFrame()
    runtime_by_case = runtime.set_index("case_id") if not runtime.empty else pd.DataFrame()
    active = read_csv(config.runtime_output_dir / "runtime_active_processes.csv")
    active_by_case = active.groupby("case_id").first() if "case_id" in active.columns and not active.empty else pd.DataFrame()
    case_roots = [(root.label, root.path) for root in config.case_roots]
    bridge_roots = [(f"bridge_{idx}", root) for idx, root in enumerate(config.bridge_roots, start=1)]
    versions = script_versions()

    rows: list[dict[str, Any]] = []
    for _, row in wide.sort_values("case_seq").iterrows():
        case_id = str(row["case_id"])
        manifest_row = manifest_by_case.loc[case_id] if not manifest_by_case.empty and case_id in manifest_by_case.index else row
        runtime_row = runtime_by_case.loc[case_id] if not runtime_by_case.empty and case_id in runtime_by_case.index else {}
        active_row = active_by_case.loc[case_id] if not active_by_case.empty and case_id in active_by_case.index else {}
        emc_label, emc_dir = first_existing_dir(case_roots, case_id)
        bridge_label, bridge_dir = first_existing_dir(bridge_roots, case_id)
        case_json = first_existing_case_json(config, case_id)
        source_case_dir = str(row.get("source_case_dir", "") or "")
        rows.append(
            {
                "case_id": case_id,
                "project_id": config.project_id,
                "case_seq": row.get("case_seq", ""),
                "manifest_row_hash": stable_row_hash(manifest_row),
                "case_json_path": str(case_json) if case_json else "",
                "case_json_sha256": file_sha256(case_json) if case_json else "",
                "emc_case_root_label": emc_label,
                "emc_case_dir": str(emc_dir) if emc_dir else "",
                "bridge_root_label": bridge_label,
                "bridge_case_dir": str(bridge_dir) if bridge_dir else "",
                "runtime_status": getattr(runtime_row, "runtime_status", row.get("runtime_status", "")),
                "active": getattr(runtime_row, "active", row.get("active", "")),
                "active_locations": getattr(runtime_row, "active_locations", row.get("active_locations", "")),
                "active_host_label": getattr(active_row, "host_label", ""),
                "active_node": getattr(active_row, "node", ""),
                "active_gpu_id": getattr(active_row, "gpu_id", ""),
                "active_pid": getattr(active_row, "pid", ""),
                "strict_done": getattr(runtime_row, "strict_done", row.get("strict_done", "")),
                "deployed_paths": getattr(runtime_row, "deployed_paths", row.get("deployed_paths", "")),
                "latest_nvt_logs": getattr(runtime_row, "latest_nvt_logs", row.get("latest_nvt_logs", "")),
                "source_case_dir": source_case_dir,
                "status_Rg": row.get("status_Rg", ""),
                "status_RDF": row.get("status_RDF", ""),
                "rdf_validation_status": row.get("rdf_validation_status", ""),
                "rdf_publication_usable": row.get("rdf_publication_usable", False),
                "status_MSD": row.get("status_MSD", ""),
                "status_shape": row.get("status_shape", ""),
                "status_dynamics": row.get("status_dynamics", ""),
                "status_contact": row.get("status_contact", ""),
                "status_density": row.get("status_density", ""),
                "status_eta": row.get("status_eta", ""),
                "quality_flags": row.get("quality_flags", ""),
                "warning_reason": row.get("warning_reason", ""),
                "publication_grade": row.get("publication_grade", False),
                "ml_usable": row.get("ml_usable", False),
                "script_versions_json": json.dumps(versions, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def build_provenance_graph(config: LCBProjectConfig, provenance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, case_id: str, label: str, path: str = "", status: str = "") -> None:
        nodes.append({"node_id": node_id, "node_type": node_type, "case_id": case_id, "label": label, "path": path, "status": status})

    def add_edge(source: str, target: str, edge_type: str, case_id: str) -> None:
        edges.append({"source": source, "target": target, "edge_type": edge_type, "case_id": case_id})

    for _, row in provenance.iterrows():
        case_id = str(row["case_id"])
        case_node = f"case:{case_id}"
        manifest_node = f"manifest:{case_id}"
        add_node(case_node, "case", case_id, case_id, status=str(row.get("runtime_status", "")))
        add_node(manifest_node, "manifest_row", case_id, f"manifest row {case_id}", path=str(config.manifest_csv))
        add_edge(case_node, manifest_node, "generated_from", case_id)
        if present(row.get("emc_case_dir")):
            emc_node = f"emc:{case_id}"
            add_node(emc_node, "emc_artifact", case_id, "EMC artifact", path=str(row.get("emc_case_dir", "")))
            add_edge(emc_node, case_node, "generated_from", case_id)
        if present(row.get("bridge_case_dir")):
            bridge_node = f"bridge:{case_id}"
            add_node(bridge_node, "bridge_artifact", case_id, "bridge artifact", path=str(row.get("bridge_case_dir", "")))
            add_edge(bridge_node, f"emc:{case_id}" if present(row.get("emc_case_dir")) else case_node, "generated_from", case_id)
        if present(row.get("deployed_paths")) or present(row.get("active_locations")):
            md_node = f"md:{case_id}"
            add_node(md_node, "md_run", case_id, "MD run", path=str(row.get("deployed_paths", "")), status=str(row.get("runtime_status", "")))
            add_edge(md_node, f"bridge:{case_id}" if present(row.get("bridge_case_dir")) else case_node, "simulated_by", case_id)
        if str(row.get("publication_grade", "")).lower() in {"true", "1"} or str(row.get("ml_usable", "")).lower() in {"true", "1"}:
            integrated_node = f"integrated:{case_id}"
            add_node(integrated_node, "integrated_row", case_id, "integrated row", path=str(config.integrated_summary_csv), status=str(row.get("quality_flags", "")))
            add_edge(integrated_node, f"md:{case_id}" if (present(row.get("deployed_paths")) or present(row.get("active_locations"))) else case_node, "merged_into", case_id)
        for observable in NON_VISC_STATUS + ["eta"]:
            status = str(row.get(f"status_{observable}", "MISSING")).upper()
            if status in {"OK", "WARN"}:
                obs_node = f"observable:{observable}:{case_id}"
                add_node(obs_node, "observable_output", case_id, observable, path=str(config.integrated_root), status=status)
                add_edge(obs_node, f"md:{case_id}" if (present(row.get("deployed_paths")) or present(row.get("active_locations"))) else case_node, "analyzed_by", case_id)
                if str(row.get("publication_grade", "")).lower() in {"true", "1"} or str(row.get("ml_usable", "")).lower() in {"true", "1"}:
                    add_edge(f"integrated:{case_id}", obs_node, "merged_from", case_id)
    node_df = pd.DataFrame(nodes).drop_duplicates("node_id", keep="last")
    edge_df = pd.DataFrame(edges).drop_duplicates()
    return node_df, edge_df


def build_automation_metrics(config: LCBProjectConfig, wide: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    runtime = read_csv(config.runtime_output_dir / "runtime_cases.csv")
    active = read_csv(config.runtime_output_dir / "runtime_active_processes.csv")
    gpus = read_csv(config.runtime_output_dir / "runtime_gpu_snapshot.csv")
    duplicates = read_csv(config.runtime_output_dir / "runtime_duplicates.csv")
    events = read_csv(config.supervisor_output_dir / "supervisor_events.csv")

    rows: list[dict[str, Any]] = []

    def metric(name: str, value: Any, unit: str = "count", source: str = "") -> None:
        rows.append({"metric": name, "value": value, "unit": unit, "source": source})

    metric("total_design_cases", len(wide), source="integrated")
    metric("any_analyzed_cases", int(wide.get("any_analyzed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()), source="integrated")
    for status_col in [col for col in wide.columns if col.startswith("status_")]:
        counts = Counter(wide[status_col].fillna("MISSING").astype(str))
        for key, value in sorted(counts.items()):
            metric(f"{status_col}_{key}", int(value), source="integrated")
    if not runtime.empty:
        metric("strict_done_cases", int(runtime.get("strict_done", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"]).sum()), source="runtime")
        metric("ready_for_analysis_cases", int(runtime.get("ready_for_analysis", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"]).sum()), source="runtime")
        metric("backlog_submit_candidates", int(runtime.get("backlog_submit_candidate", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1", "yes"]).sum()), source="runtime")
    metric("active_processes", len(active), source="runtime")
    metric("duplicate_rows", len(duplicates), source="runtime")
    if not gpus.empty and "util_percent" in gpus.columns:
        util = pd.to_numeric(gpus["util_percent"], errors="coerce")
        metric("gpu_util_mean", round(float(util.mean()), 3) if util.notna().any() else "", unit="percent", source="runtime")
        metric("gpu_count_scanned", int(len(gpus)), source="runtime")
    if not events.empty:
        for (action, result), sub in events.groupby(["action", "result"], dropna=False):
            metric(f"event_{action}_{result}", int(len(sub)), source="supervisor_events")

    lines = [
        f"# {config.project_id.upper()} Automation Metrics",
        "",
        f"- Generated: `{now_iso()}`",
        f"- Source integrated: `{config.integrated_summary_csv}`",
        f"- Runtime output: `{config.runtime_output_dir}`",
        f"- Supervisor output: `{config.supervisor_output_dir}`",
        "",
        "## Metrics",
        "",
    ]
    for row in rows:
        unit = f" {row['unit']}" if row.get("unit") else ""
        lines.append(f"- `{row['metric']}`: {row['value']}{unit} ({row.get('source', '')})")
    return pd.DataFrame(rows), "\n".join(lines) + "\n"


def plot_automation_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    selected = metrics[metrics["metric"].isin(["total_design_cases", "any_analyzed_cases", "strict_done_cases", "active_processes", "ready_for_analysis_cases", "backlog_submit_candidates"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if selected.empty:
        ax.text(0.5, 0.5, "No automation metrics available", ha="center", va="center")
        ax.axis("off")
    else:
        values = pd.to_numeric(selected["value"], errors="coerce").fillna(0)
        ax.bar(selected["metric"], values, color="#607d8b")
        ax.set_ylabel("count")
        ax.set_title("Automation state summary")
        ax.tick_params(axis="x", rotation=35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def build_ml_dataset(config: LCBProjectConfig, wide: pd.DataFrame) -> dict[str, Any]:
    agg = config.integrated_output_dir
    wide = merge_shape_tensor_features(config, wide)
    tier_path = agg / "compaction_tier_preview_20260523.csv"
    if tier_path.exists() and "case_id" in wide.columns:
        tiers = read_csv(tier_path)
        keep = [col for col in ["case_id", "rho_final_gcc", "compaction_tier_preview"] if col in tiers.columns]
        if keep:
            wide = wide.merge(tiers[keep], on="case_id", how="left")
    rule_columns = build_rule_verdict_columns(wide)
    expert_columns, expert_source_path = build_expert_label_columns(config)
    audit_columns, audit_group_path, raw_audit_path = build_audit_summary_columns(config)
    base = wide[["case_id", "case_seq"]].copy()
    feature_cols = [col for col in CATEGORICAL_FEATURES + FEATURE_CANDIDATES if col in wide.columns]
    target_cols = [col for col in TARGET_CANDIDATES if col in wide.columns]
    features = pd.concat([base, wide[feature_cols]], axis=1)
    targets = pd.concat([base, wide[target_cols]], axis=1)
    mask = base.copy()
    for col in feature_cols:
        usable = wide[col].map(present)
        if col.startswith("rdf_"):
            usable = usable & wide["rdf_publication_usable"].astype(bool)
        mask[f"feature_{col}_usable"] = usable
    target_status = {
        "D_mean_m2_s": "MSD",
        "D_mean_1e-5_cm2_s": "MSD",
        "loglog_slope": "MSD",
        "tau_fs_ps": "fsqt_tau_alpha",
        "tau_alpha_ps": "fsqt_tau_alpha",
        "g1_over_g3_10ns": "dynamics",
        "g1_monomer_over_g3_10ns": "dynamics",
    }
    for col in target_cols:
        status = target_status.get(col, "")
        usable = wide[col].map(present)
        if status:
            usable = usable & wide.get(f"status_{status}", pd.Series(["MISSING"] * len(wide))).astype(str).str.upper().eq("OK")
        mask[f"target_{col}_usable"] = usable
    mask["publication_grade"] = wide["publication_grade"]
    mask["ml_usable"] = wide["ml_usable"]
    mask["warning_reason"] = wide["warning_reason"]
    mask["status_MSD"] = wide.get("status_MSD", "")
    mask["msd_publication_grade"] = wide.get("msd_publication_grade", False)
    mask["msd_publication_issues"] = wide.get("msd_publication_issues", "")
    mask["msd_clean_engine_source"] = wide.get("msd_clean_engine_source", False)
    mask["msd_tail_window_ok"] = wide.get("msd_tail_window_ok", False)
    mask["msd_fit_R2_publication"] = wide.get("msd_fit_R2_publication", "")
    mask["rdf_validation_status"] = wide["rdf_validation_status"]
    mask["rdf_publication_usable"] = wide["rdf_publication_usable"]
    mask["status_density"] = wide.get("status_density", "")
    mask["density_tier_si_o"] = wide.get("compaction_tier_preview", "")
    mask["rho_final_gcc_si_o"] = wide.get("rho_final_gcc", "")
    mask["force_field"] = wide.get("force_field", "")
    mask["force_field_qualified"] = True
    mask["engine_version"] = "GROMACS2021.2-production"
    mask["engine_qualified"] = wide.get("status_density", pd.Series([""] * len(wide))).map(present)
    mask["case_status"] = wide.apply(build_case_status, axis=1)
    mask["provenance_qualification_note"] = (
        "density values are qualified by production-engine provenance and standard OPLS-AA force-field offset. "
        "Before supervised learning, consult: density_tier_si_o, rho_final_gcc_si_o, warning_reason, "
        "case_status, engine_version, rule_* (release-time gate), and audit_majority_* + expert_label_* "
        "(40-case locked-evaluation subset only)."
    )
    confidence_long = build_result_confidence_table(wide)
    confidence_wide = confidence_wide_columns(confidence_long)
    if not confidence_wide.empty and "case_id" in confidence_wide.columns:
        mask = mask.merge(confidence_wide, on="case_id", how="left")
    for extra in (rule_columns, audit_columns, expert_columns):
        if not extra.empty and "case_id" in extra.columns:
            mask = mask.merge(extra, on="case_id", how="left")
    if "a3_locked_eval_member" not in mask.columns:
        mask["a3_locked_eval_member"] = False
    else:
        mask["a3_locked_eval_member"] = mask["a3_locked_eval_member"].fillna(False).astype(bool)
    if "audit_disagreement_flag" not in mask.columns:
        mask["audit_disagreement_flag"] = False
    else:
        mask["audit_disagreement_flag"] = mask["audit_disagreement_flag"].fillna(False).astype(bool)
    if "audit_disagreement_count" not in mask.columns:
        mask["audit_disagreement_count"] = 0
    else:
        mask["audit_disagreement_count"] = pd.to_numeric(mask["audit_disagreement_count"], errors="coerce").fillna(0).astype(int)

    features_path = agg / "ml_features.csv"
    targets_path = agg / "ml_targets.csv"
    mask_path = agg / "ml_quality_mask.csv"
    confidence_path = agg / "result_confidence.csv"
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    targets.to_csv(targets_path, index=False, encoding="utf-8-sig")
    mask.to_csv(mask_path, index=False, encoding="utf-8-sig")
    confidence_long.to_csv(confidence_path, index=False, encoding="utf-8-sig")
    manifest = {
        "generated_at": now_iso(),
        "project_id": config.project_id,
        "schema_version": 2,
        "case_count": int(len(wide)),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "path_base": "this manifest directory",
        "features_csv": portable_path(features_path, agg),
        "targets_csv": portable_path(targets_path, agg),
        "quality_mask_csv": portable_path(mask_path, agg),
        "result_confidence_csv": portable_path(confidence_path, agg),
        "joined_release_columns": {
            "rule_verdict_columns": [f"rule_{decision_type}" for decision_type in DECISION_TYPES],
            "audit_majority_columns": [f"audit_majority_{decision_type}" for decision_type in DECISION_TYPES],
            "expert_label_columns": [f"expert_label_{decision_type}" for decision_type in DECISION_TYPES],
            "availability": (
                "rule columns are populated for completed evidence rows and marked missing for no-analysis rows; "
                "audit and expert columns are populated only for the 40-case locked A3 evaluation subset; "
                "audit_disagreement_flag is true when any blind or rule-aware A3 tri-model group split for that case "
                "(matching the paper's 34/240 blind-or-audit-group disagreement count, not the 31/120 "
                "case-decision count that collapses blind/audit mode records before counting)"
            ),
        },
        "release_input_artifacts": {
            "a3_group_results_csv": relative_portable_path(audit_group_path, SCRIPT_DIR) if audit_group_path else "",
            "a3_raw_judgments_jsonl": relative_portable_path(raw_audit_path, SCRIPT_DIR) if raw_audit_path else "",
            "t1_expert_labels_csv": relative_portable_path(expert_source_path, SCRIPT_DIR) if expert_source_path else "",
        },
        "notes": [
            "WARN/FAIL values are retained in raw integrated tables but controlled by ml_quality_mask.",
            "RDF is included in the automated postprocessing package and defaults to current when status_RDF is OK; use rdf_validation_status to exclude it if a later audit requires.",
            "The exported ML target table excludes all-empty and legacy exploratory target fields; raw integrated tables retain them for provenance.",
            "The rg_mean_nm and rg2_mean_nm2 fields come from production Rg postprocessing; shape_* fields come from the carbon-skeleton gyration tensor and absolute shape components should be compared against shape_rg2_carbon_mean_nm2, not rg2_mean_nm2.",
            "Density is released as engine-qualified and force-field-qualified; use density_tier_si_o, rho_final_gcc_si_o, and warning_reason before supervised learning.",
            "rule_* columns reflect the rule baseline applied to canonical-aggregate scalar evidence in the ML release table. The A3 locked-benchmark evaluation used a richer per-case evidence bundle, including thermal/energetic/box-volume drift time-series evidence, runtime markers, and checkpoint integrity, and its rule verdicts are reported in Fig. 4 and ablation_metrics.json. The two are intentionally separate: ML-mask rule columns represent the deterministic release-time gate, whereas A3 rule columns represent the evaluation-time gate.",
            "Rule verdict, audit-majority, expert-label, engine-version, and case-status columns in ml_quality_mask make the released ML tables a reliability map rather than a flat descriptor table.",
            "Full model rationales remain in raw_judgments.jsonl; ml_quality_mask intentionally stores only joinable audit summaries to keep the release table ML-usable.",
            "This is a quality-masked descriptor-screening dataset, not a completed autonomous design result.",
        ],
    }
    manifest_path = agg / "ml_dataset_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "ml_features_csv": features_path,
        "ml_targets_csv": targets_path,
        "ml_quality_mask_csv": mask_path,
        "result_confidence_csv": confidence_path,
        "ml_dataset_manifest_json": manifest_path,
    }


def write_candidate_manifest_scaffold(config: LCBProjectConfig, wide: pd.DataFrame) -> Path:
    path = config.integrated_output_dir / "candidate_manifest_dryrun.csv"
    fields = [
        "candidate_id",
        "source_model",
        "parent_project_id",
        "branch_points",
        "branch_carbons",
        "branch_topology",
        "branch_position_pattern",
        "expected_target",
        "human_approved",
        "notes",
    ]
    if not path.exists():
        write_csv(path, [], fields)
    return path


def update_overview_manifest(config: LCBProjectConfig, extra_outputs: dict[str, Path]) -> None:
    manifest_path = config.integrated_output_dir / f"{config.project_id}_overview_manifest.json"
    payload: dict[str, Any] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.setdefault("outputs", {})
    payload["outputs"].update({key: portable_path(path, config.integrated_output_dir) for key, path in extra_outputs.items()})
    payload["outputs"] = {
        key: portable_path(value, config.integrated_output_dir)
        for key, value in payload["outputs"].items()
    }
    payload["publication_assets_generated_at"] = now_iso()
    payload["quality_policy"] = {
        "publication_grade": "all required release observables in QUALITY_GATE_STATUS are OK; WARN, FAIL, MISSING, and NaN statuses block publication_grade; RDF is tracked separately by rdf_validation_status",
        "ml_usable": "any analyzed data and no quality-gated WARN/FAIL/MISSING; per-target masks still apply",
        "rdf": "included in automated postprocessing and current by default when status_RDF is OK; use rdf_validation_status for later validated/excluded overrides",
        "legacy_exploratory_targets": "retained only in raw integrated tables; excluded from the released ML target table",
        "density": "engine-qualified and force-field-qualified; SI-O density tier and warning_reason control supervised-learning use",
    }
    write_json(manifest_path, payload)


def build_publication_assets(config: LCBProjectConfig, wide: pd.DataFrame | None = None) -> dict[str, Path]:
    agg = config.integrated_output_dir
    if wide is None:
        wide = read_csv(config.integrated_summary_csv)
    if wide.empty:
        raise FileNotFoundError(f"Integrated summary missing or empty: {config.integrated_summary_csv}")
    wide = build_quality_columns(wide)
    wide.to_csv(config.integrated_summary_csv, index=False, encoding="utf-8-sig")
    json_path = agg / f"{config.project_id}_summary_wide.json"
    if json_path.exists():
        json_path.write_text(wide.to_json(orient="records", indent=2, force_ascii=False) + "\n", encoding="utf-8")

    provenance = build_provenance_cases(config, wide)
    prov_csv = agg / "provenance_cases.csv"
    prov_json = agg / "provenance_cases.json"
    provenance.to_csv(prov_csv, index=False, encoding="utf-8-sig")
    prov_json.write_text(provenance.to_json(orient="records", indent=2, force_ascii=False) + "\n", encoding="utf-8")

    nodes, edges = build_provenance_graph(config, provenance)
    nodes_csv = agg / "provenance_nodes.csv"
    edges_csv = agg / "provenance_edges.csv"
    nodes.to_csv(nodes_csv, index=False, encoding="utf-8-sig")
    edges.to_csv(edges_csv, index=False, encoding="utf-8-sig")

    metrics, metrics_md = build_automation_metrics(config, wide)
    metrics_csv = agg / "automation_metrics.csv"
    metrics_md_path = agg / "automation_metrics.md"
    metrics_png = agg / "plots" / "automation_metrics.png"
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    metrics_md_path.write_text(metrics_md, encoding="utf-8")
    plot_automation_metrics(metrics, metrics_png)

    ml_paths = build_ml_dataset(config, wide)
    candidate_path = write_candidate_manifest_scaffold(config, wide)
    output_paths = {
        "provenance_cases_csv": prov_csv,
        "provenance_cases_json": prov_json,
        "provenance_nodes_csv": nodes_csv,
        "provenance_edges_csv": edges_csv,
        "automation_metrics_csv": metrics_csv,
        "automation_metrics_md": metrics_md_path,
        "automation_metrics_png": metrics_png,
        "candidate_manifest_dryrun_csv": candidate_path,
        **ml_paths,
    }
    update_overview_manifest(config, output_paths)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "lcb300.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_lcb_config(args.config)
    outputs = build_publication_assets(config)
    print(f"Wrote publication assets for {config.project_id}:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
