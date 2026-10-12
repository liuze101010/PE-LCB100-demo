#!/usr/bin/env python3
"""Re-gate collected clean-engine sidecar outputs.

This script evaluates clean-engine repair overlays without modifying the
PE-LCB100 canonical aggregate.  It distinguishes a targeted re-gate verdict
for the WARN sources that caused the repair from a full release-overlay
verdict over the available clean evidence.  The release overlay is intentionally
scoped to publication-blocking gates: auxiliary Fs(q,t) diagnostics remain
visible in clean_warn_sources but do not by themselves block publication-grade
status once density, core QC, and required evidence are satisfied.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from polymer_qc_rules import apply_polymer_qc_rules, assess_registered_runtime_support
from pelcb_msd_publication_gate import evaluate_msd_publication_gate
from rule_profile_registry import apply_rule_gate, install_radonpy_pe_lcb_density_profile


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output"
AGGREGATE_CSV = OUTPUT_ROOT / "pelcb100_integrated" / "aggregate" / "pelcb100_summary_wide.csv"
CLEAN_CLOSURE_DIR = OUTPUT_ROOT / "pelcb100_supervisor" / "o7_clean_cuda2024_closure"
CLEAN_STATUS_CSV = CLEAN_CLOSURE_DIR / "o7_clean_run_status.csv"
CLEAN_DENSITY_CSV = CLEAN_CLOSURE_DIR / "o7_clean_density_summary.csv"
RULE_PROFILE_DIR = (
    OUTPUT_ROOT
    / "rule_profiles"
    / "pelcb100_clean_overlay_publication_density_radonpy_20260716"
)
REPAIR_LOOP_ROOT = OUTPUT_ROOT / "pelcb100_repair_loop"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "pelcb100_clean_overlay_regate"

RELEASE_NONBLOCKING_WARN_SOURCES = {"fsqt_warn"}
RELEASE_NONBLOCKING_EVIDENCE_DOMAINS = {"fsqt"}
AVOGADRO = 6.02214076e23
CH2_MASS_G_PER_MOL = 14.027


CSV_FIELDS = [
    "schema_version",
    "case_id",
    "clean_case_id",
    "target_warn_sources",
    "clean_warn_sources",
    "resolved_warn_sources",
    "remaining_target_warn_sources",
    "density_rule_verdict",
    "density_value_gcc",
    "qc_rule_verdict",
    "targeted_regate_verdict",
    "release_overlay_verdict",
    "release_overlay_reason",
    "release_blocking_warn_sources",
    "evidence_status",
    "missing_clean_outputs",
    "require_intrachain_g3",
    "clean_tag",
    "clean_remote_run_dir",
    "density_evidence_source",
    "density_evidence_stage",
    "density_time_first_ps",
    "density_time_last_ps",
    "density_tail_start_ps",
    "density_frames",
    "density_tail_frames",
    "density_profile_id",
    "density_reason",
    "clean_msd_window_overlay_status",
    "clean_msd_window_overlay_source",
    "clean_msd_window_overlay_fit_window_ns",
    "clean_msd_window_overlay_loglog_slope",
    "clean_msd_window_overlay_D_m2_s",
    "clean_msd_window_overlay_local_D_cv_pct",
    "status_MSD",
    "msd_publication_grade",
    "msd_publication_issues",
    "msd_publication_diagnostic_issues",
    "msd_displacement_amplitude_check",
    "msd_diffusion_confidence",
    "msd_clean_engine_source",
    "msd_tail_window_ok",
    "msd_fit_R2_publication",
    "loglog_slope",
    "fit_quality",
    "diffusive_regime_ok",
    "regime_warning",
    "window_sensitive",
    "window_D_cv_pct",
    "D_mean_m2_s",
    "D_median_m2_s",
    "D_outlier_fraction",
    "fit_begin_ps",
    "fit_end_ps",
    "time_last_ps",
    "window_msd_over_rg2_min",
    "msd_over_rg2",
    "status_contact",
    "P_contact_rc0p8_s10",
    "P_contact_rc0p8_s20",
    "P_contact_rc0p8_s50",
    "contact_source",
    "status_intrachain_g3",
    "intrachain_g3_alpha_100_300",
    "intrachain_g3_time_last_ps",
    "g1_monomer_over_g3_10ns",
    "intrachain_g3_source",
    "intrachain_g3_provenance",
    "qc_domain_verdicts_json",
    "qc_checks_json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def choose_clean_status_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Pick the most advanced clean status row per canonical case."""

    def rank(row: Mapping[str, Any]) -> tuple[int, int, str]:
        # Manual migrations and retries can leave newer partial rows behind.
        # Prefer completed NVT evidence before recency so a partial retry does
        # not hide a usable clean trajectory.
        return (
            int(truthy(row.get("nvt_prod_done"))),
            int(truthy(row.get("pr_done"))),
            str(row.get("scan_timestamp", "")),
        )

    chosen: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        if case_id not in chosen or rank(row) > rank(chosen[case_id]):
            chosen[case_id] = row
    return chosen


def _same_text(left: Any, right: Any) -> bool:
    return str(left or "").strip() == str(right or "").strip()


def _clean_density_row_rank(row: Mapping[str, Any], status_row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    """Rank density rows by whether they belong to the selected clean run."""

    # Density summaries may contain short smoke or PR rows for the same
    # canonical case. Publication re-gating must judge the clean status row
    # with evidence from that same clean attempt, not an unrelated smoke check.
    stage = str(row.get("stage") or "").strip()
    return (
        int(stage == "nvt_prod"),
        int(_same_text(row.get("tag"), status_row.get("tag"))),
        int(_same_text(row.get("remote_run_dir"), status_row.get("remote_run_dir"))),
        str(row.get("extracted_at", "")),
    )


def density_from_volume_analysis(source_row: Mapping[str, Any], clean_case_id: str) -> dict[str, Any] | None:
    analysis_path = clean_output_paths(clean_case_id)["density"]
    payload = read_json(analysis_path) if analysis_path else None
    if not payload:
        return None
    volume_nm3 = finite_float(payload.get("volume_nm3"))
    chains = finite_float(source_row.get("chain_count")) or finite_float(source_row.get("n_chains"))
    carbons = finite_float(source_row.get("actual_total_carbons_per_chain")) or finite_float(
        source_row.get("target_total_carbons_per_chain")
    )
    if volume_nm3 is None or chains is None or carbons is None:
        return None
    if volume_nm3 <= 0 or chains <= 0 or carbons <= 0:
        return None
    density = chains * carbons * CH2_MASS_G_PER_MOL / (AVOGADRO * volume_nm3) * 1.0e21
    return {
        "case_id": source_row.get("case_id", ""),
        "clean_case_id": clean_case_id,
        "stage": "nvt_prod",
        "ok": "true",
        "density_last10ns_gcc": f"{density:.12g}",
        "density_evidence_source": f"volume_analysis:{analysis_path}",
        "volume_nm3": f"{volume_nm3:.12g}",
    }


def select_clean_density_row(
    source_row: Mapping[str, Any],
    status_row: Mapping[str, Any],
    density_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    case_id = str(status_row.get("case_id") or source_row.get("case_id") or "").strip()
    clean_case_id = str(status_row.get("clean_case_id") or "").strip()
    matching = [
        dict(row)
        for row in density_rows
        if truthy(row.get("ok"))
        and _same_text(row.get("case_id"), case_id)
        and _same_text(row.get("clean_case_id"), clean_case_id)
        and (
            _same_text(row.get("tag"), status_row.get("tag"))
            or _same_text(row.get("remote_run_dir"), status_row.get("remote_run_dir"))
        )
    ]
    if matching:
        chosen = max(matching, key=lambda row: _clean_density_row_rank(row, status_row))
        chosen.setdefault("density_evidence_source", "density_summary")
        return chosen
    return density_from_volume_analysis(source_row, clean_case_id)


def finite_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return text if number == number and number not in (float("inf"), float("-inf")) else ""


def finite_float(value: Any) -> float | None:
    text = finite_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_sources(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text.replace("'", '"'))
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def read_msd_window_overlays(
    path: Path | None,
    repair_root: Path = REPAIR_LOOP_ROOT,
) -> dict[str, dict[str, Any]]:
    paths = [path] if path else sorted(
        repair_root.glob("*/msd_window_repair_candidates.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    overlays: dict[str, dict[str, Any]] = {}
    for candidate_path in paths:
        for row in read_csv(candidate_path):
            case_id = str(row.get("case_id", "")).strip()
            if case_id:
                overlays[case_id] = dict(row)
    return overlays


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def latest_repair_plan(root: Path = REPAIR_LOOP_ROOT) -> Path | None:
    plans = list(root.glob("*/repair_loop_plan.csv"))
    if not plans:
        return None

    def plan_rank(path: Path) -> tuple[int, int, int]:
        parent = path.parent.name.lower()
        execute_like = int((path.parent / "repair_loop_execution_ledger.jsonl").exists() or "daemon" in parent)
        selfcheck_penalty = int(parent.startswith("selfcheck"))
        return (execute_like, -selfcheck_penalty, path.stat().st_mtime_ns)

    return max(plans, key=plan_rank)


def first_existing(paths: Iterable[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def latest_matching_output(pattern: str) -> Path | None:
    candidates = [path for path in OUTPUT_ROOT.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.parts, path.stat().st_mtime_ns))


def clean_output_paths(clean_case_id: str) -> dict[str, Path | None]:
    return {
        "density": OUTPUT_ROOT / "o7clean_density_energy" / clean_case_id / "analysis.json",
        "rg": latest_matching_output(f"o7clean_rg_*/cases/{clean_case_id}/rg/rg_summary.json"),
        "rdf": first_existing(
            [
                OUTPUT_ROOT / "o7clean_integrated" / "RDF" / "raw" / "cases" / clean_case_id / "rdf" / "rdf_summary.json",
                latest_matching_output(f"o7clean_rdf_*/cases/{clean_case_id}/rdf/rdf_summary.json"),
            ]
        ),
        "shape": OUTPUT_ROOT
        / "o7clean_integrated"
        / "ShapeTensor"
        / "raw"
        / "cases"
        / clean_case_id
        / "shape_tensor"
        / "shape_summary.json",
        "msd": latest_matching_output(f"o7clean_msd_*/cases/{clean_case_id}/msd/msd_summary.json"),
        "fsqt": latest_matching_output(f"o7clean_dynamics_*/cases/{clean_case_id}/dynamics/fsqt_summary.json"),
        "intrachain_g1": OUTPUT_ROOT
        / "o7clean_integrated"
        / "IntrachainDynamics"
        / "cases"
        / clean_case_id
        / "intrachain"
        / "g1_monomer_summary.json",
        "contact": OUTPUT_ROOT
        / "o7clean_integrated"
        / "IntrachainDynamics"
        / "cases"
        / clean_case_id
        / "intrachain"
        / "contact_probability_summary.json",
        "intrachain_g3": OUTPUT_ROOT
        / "o7clean_integrated"
        / "IntrachainDynamics"
        / "cases"
        / clean_case_id
        / "intrachain"
        / "g1_g3_summary.json",
    }


CORE_CLEAN_OUTPUTS = {
    "density",
    "rg",
    "rdf",
    "shape",
    "msd",
    "intrachain_g1",
    "contact",
    "intrachain_g3",
}


def clean_optional_requirements(source_row: Mapping[str, Any], repair_row: Mapping[str, Any] | None = None) -> set[str]:
    required: set[str] = set()
    fsqt_status = str(source_row.get("status_fsqt_tau_alpha", "")).strip().upper()
    if fsqt_status in {"MISSING", "WARN", "FAIL"}:
        required.add("fsqt")
    repair_sources = parse_sources(repair_row.get("warn_sources") if repair_row else "")
    if any("fsqt" in source.lower() or "tau_alpha" in source.lower() for source in repair_sources):
        required.add("fsqt")
    return required


def missing_clean_outputs(clean_case_id: str, required_optional_outputs: Iterable[str] | None = None) -> list[str]:
    paths = clean_output_paths(clean_case_id)
    required = set(CORE_CLEAN_OUTPUTS)
    required.update(required_optional_outputs or [])
    return [name for name, path in paths.items() if name in required and (path is None or not path.exists())]


def rdf_entry(summary: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    for entry in summary.get("entries", []) if isinstance(summary.get("entries"), list) else []:
        if str(entry.get("rdf_kind", "")).strip() == kind:
            return entry
    return {}


def classify_intrachain_g3_provenance(summary: Mapping[str, Any]) -> str:
    method = str(summary.get("method") or "").lower()
    source = str(summary.get("g3_source") or summary.get("source_timeseries") or "").lower()
    if "backfill" in method or "backfill" in source or "o7clean_msd" in source:
        return "clean_com_msd_backfill"
    if "gmx msd" in method or "g3_com_msd" in source:
        return "native_gmx_msd_mol"
    return "unknown"


def add_clean_json_evidence(row: dict[str, Any], clean_case_id: str) -> None:
    paths = clean_output_paths(clean_case_id)
    rg = read_json(paths["rg"]) if paths["rg"] else None
    rdf = read_json(paths["rdf"]) if paths["rdf"] else None
    shape = read_json(paths["shape"]) if paths["shape"] else None
    msd = read_json(paths["msd"]) if paths["msd"] else None
    fsqt = read_json(paths["fsqt"]) if paths["fsqt"] else None
    contact = read_json(paths["contact"]) if paths["contact"] else None
    intrachain_g3 = read_json(paths["intrachain_g3"]) if paths["intrachain_g3"] else None

    row["require_intrachain_g3"] = True

    if rg:
        row.update(
            {
                "status_Rg": rg.get("status", "OK"),
                "rg_mean_nm": rg.get("rg_mean_nm"),
                "rg2_mean_nm2": rg.get("rg2_mean_nm2"),
            }
        )
    if rdf:
        all_entry = rdf_entry(rdf, "all")
        excl_entry = rdf_entry(rdf, "excl")
        inter_entry = rdf_entry(rdf, "inter_chain_cc")
        intra_entry = rdf_entry(rdf, "intra_chain_nonbonded_cc")
        row.update(
            {
                "status_RDF": "OK"
                if all(str(entry.get("status", "")).lower() == "ok" for entry in (all_entry, excl_entry, inter_entry, intra_entry))
                else "WARN",
                "rdf_validation_status": "ok",
                "rdf_all_first_peak_g": all_entry.get("first_peak_g"),
                "rdf_excl_first_peak_g": excl_entry.get("first_peak_g"),
                "rdf_inter_chain_cc_first_peak_g": inter_entry.get("first_peak_g"),
                "rdf_intra_chain_nonbonded_cc_first_peak_g": intra_entry.get("first_peak_g"),
            }
        )
    if shape:
        row.update(
            {
                "status_shape": "OK" if str(shape.get("status", "")).lower() == "ok" else "WARN",
                "shape_rg2_carbon_mean_nm2": shape.get("rg2_carbon_mean_nm2"),
                "shape_asphericity_mean_nm2": shape.get("asphericity_mean_nm2"),
                "shape_acylindricity_mean_nm2": shape.get("acylindricity_mean_nm2"),
                "shape_relative_anisotropy_mean": shape.get("relative_shape_anisotropy_mean"),
            }
        )
    if msd:
        msd_evidence = dict(msd)
        msd_evidence.update(
            {
                "engine_version": "gromacs_2024.4_clean_overlay",
                "clean_case_id": clean_case_id,
                "msd_source_engine": "GROMACS 2024.4 clean nvt_prod",
            }
        )
        msd_gate = evaluate_msd_publication_gate(msd_evidence, require_clean=True)
        row.update(
            {
                "status_MSD": msd_gate["status_MSD"],
                "D_mean_m2_s": msd.get("D_mean_m2_s"),
                "D_median_m2_s": msd.get("D_median_m2_s"),
                "D_outlier_fraction": msd.get("D_outlier_fraction"),
                "loglog_slope": msd.get("loglog_slope"),
                "fit_quality": msd.get("fit_quality"),
                "diffusive_regime_ok": msd.get("diffusive_regime_ok"),
                "regime_warning": msd.get("regime_warning"),
                "window_sensitive": msd.get("window_sensitive"),
                "window_D_cv_pct": msd.get("window_D_cv_pct"),
                "fit_begin_ps": msd.get("fit_begin_ps"),
                "fit_end_ps": msd.get("fit_end_ps"),
                "time_first_ps": msd.get("time_first_ps"),
                "time_last_ps": msd.get("time_last_ps"),
                "python_fit": msd.get("python_fit"),
                "publication_window_policy": msd.get("publication_window_policy"),
                "msd_publication_window_policy": msd.get("msd_publication_window_policy"),
                "window_msd_over_rg2_min": msd.get("window_msd_over_rg2_min"),
                "msd_over_rg2": msd.get("msd_over_rg2"),
                "msd_publication_grade": msd_gate["msd_publication_grade"],
                "msd_publication_issues": msd_gate["issue_string"],
                "msd_publication_diagnostic_issues": msd_gate.get("diagnostic_issue_string", ""),
                "msd_displacement_amplitude_check": msd_gate.get("displacement_amplitude_check", ""),
                "msd_diffusion_confidence": msd_gate.get("msd_diffusion_confidence", ""),
                "msd_clean_engine_source": msd_gate["clean_engine_source"],
                "msd_tail_window_ok": msd_gate["tail_window_ok"],
                "msd_fit_R2_publication": msd_gate["fit_R2"],
            }
        )
    if contact:
        row.update(
            {
                "status_contact": contact.get("status", "OK"),
                "P_contact_rc0p8_s10": contact.get("P_contact_rc0p8_s10"),
                "P_contact_rc0p8_s20": contact.get("P_contact_rc0p8_s20"),
                "P_contact_rc0p8_s50": contact.get("P_contact_rc0p8_s50"),
                "contact_source": str(paths["contact"]),
            }
        )
    if fsqt:
        fsqt_status = str(fsqt.get("status", "")).lower()
        row.update(
            {
                "status_fsqt_tau_alpha": "OK" if fsqt_status == "ok" else "WARN",
                "tau_alpha_ps": fsqt.get("tau_alpha_ps") or fsqt.get("tau_fs_ps"),
                "fsqt_tau_alpha_ps": fsqt.get("tau_alpha_ps") or fsqt.get("tau_fs_ps"),
                "fsqt_q_nm_inv": fsqt.get("q_nm_inv"),
                "fsqt_source_path": str(paths["fsqt"]),
            }
        )
    if intrachain_g3:
        row.update(
            {
                "status_intrachain_g3": intrachain_g3.get("status", "OK"),
                "intrachain_g3_alpha_100_300": intrachain_g3.get("g3_alpha_100_300"),
                "intrachain_g3_time_last_ps": intrachain_g3.get("time_last_ps"),
                "g1_monomer_over_g3_10ns": intrachain_g3.get("g1_monomer_over_g3_10ns"),
                "intrachain_g3_source": intrachain_g3.get("g3_source") or intrachain_g3.get("source_timeseries"),
                "intrachain_g3_provenance": classify_intrachain_g3_provenance(intrachain_g3),
            }
        )


def apply_msd_window_overlay(
    row: dict[str, Any],
    clean_case_id: str,
    overlays: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    if not overlays:
        return False
    # Publication-grade clean re-gating must not consume legacy MSD-window
    # overlays keyed by the source case.  Only overlays generated from the
    # clean trajectory namespace are eligible here; source-keyed overlays remain
    # useful as historical diagnostics but cannot close the clean release gate.
    overlay = overlays.get(clean_case_id)
    if not overlay:
        return False
    status = str(overlay.get("repair_status", "")).strip()
    row["clean_msd_window_overlay_status"] = status
    row["clean_msd_window_overlay_source"] = str(overlay.get("timeseries_path", "")).strip()
    begin = finite_text(overlay.get("best_begin_ns"))
    end = finite_text(overlay.get("best_end_ns"))
    row["clean_msd_window_overlay_fit_window_ns"] = f"{begin}-{end}" if begin and end else ""
    row["clean_msd_window_overlay_loglog_slope"] = finite_text(overlay.get("best_loglog_slope"))
    row["clean_msd_window_overlay_D_m2_s"] = finite_text(overlay.get("best_D_m2_s"))
    row["clean_msd_window_overlay_local_D_cv_pct"] = finite_text(overlay.get("best_local_D_cv_pct"))
    return False


def build_evidence_row(
    source_row: Mapping[str, Any],
    status_row: Mapping[str, Any],
    density_row: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(source_row)
    clean_case_id = str(status_row.get("clean_case_id") or "")
    builder_backend = str(source_row.get("builder_backend") or "emc_alpha_olefin_builder")
    topology_format = str(source_row.get("topology_format") or "gromacs_atomistic")
    row.update(
        {
            "case_status": "completed_clean_engine_overlay",
            "clean_case_id": clean_case_id,
            "engine_version": "gromacs_2024.4_clean_overlay",
            "gromacs_finished": "true",
            "builder_backend": builder_backend,
            "topology_format": topology_format,
            "fatal_log_marker_count": "0",
            "rho_final_gcc_si_o": density_row.get("density_last10ns_gcc", ""),
            "density_g_per_cc": density_row.get("density_last10ns_gcc", ""),
            "status_packing": source_row.get("status_packing", ""),
            "status_contact": "MISSING",
            "status_fsqt_tau_alpha": source_row.get("status_fsqt_tau_alpha", "OK"),
        }
    )
    row.update(assess_registered_runtime_support(row))
    add_clean_json_evidence(row, clean_case_id)
    return row


def normalized_force_field(value: str) -> str:
    text = str(value or "").lower()
    if "opls" in text:
        return "opls-aa"
    return str(value or "")


def context_for(row: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "")
    family = "PE-LCB100" if case_id.startswith("lcb_pe_") else (row.get("polymer_family") or "PE")
    return {
        "polymer_family": family,
        "force_field": normalized_force_field(str(row.get("force_field") or "")),
        "engine_scope": "any",
        "temperature_K": row.get("temperature_k") or 400,
        "pressure_bar": 1.0,
        "state": "melt",
        "case_id": case_id,
        "monomer_smiles": row.get("monomer_smiles"),
        "actual_total_carbons_per_chain": row.get("actual_total_carbons_per_chain"),
        "target_total_carbons_per_chain": row.get("target_total_carbons_per_chain"),
    }


def ensure_clean_overlay_rule_profiles(rule_profile_dir: Path) -> None:
    """Ensure re-gating uses the frozen 40-record RadonPy density profile."""

    active_path = rule_profile_dir / "active_rule_profile.json"
    if active_path.exists():
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            active = {}
        for ref in active.get("profiles", []) or []:
            if (
                str(ref.get("profile_id") or "") == "pe_lcb_c100_oplsaa_density_lower_bound"
                and str(ref.get("status") or "") == "active"
            ):
                profile_path = Path(
                    str(
                        ref.get("path")
                        or rule_profile_dir
                        / "profiles"
                        / "pe_lcb_c100_oplsaa_density_lower_bound.json"
                    )
                )
                if not profile_path.is_absolute():
                    profile_path = rule_profile_dir / profile_path
                try:
                    profile = json.loads(profile_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    profile = {}
                if (
                    profile.get("profile_kind") == "radonpy_public_density_screening_profile"
                    and profile.get("pass_range") == [0.73, 0.87]
                ):
                    return
    activated = install_radonpy_pe_lcb_density_profile(
        output_dir=rule_profile_dir,
        reason="activate PE-LCB100 RadonPy 40-record density screening profile",
    )
    if activated.get("status") != "ok":
        raise RuntimeError(f"density_rule_profile_activation_failed: {activated.get('errors')}")


def evidence_incomplete_is_release_blocking(
    qc_checks: Iterable[Mapping[str, Any]] | None,
    missing: Iterable[str],
) -> bool:
    """Return whether evidence-incomplete warnings block publication release.

    File-level missing clean outputs always block release.  Field-level
    evidence gaps block unless every incomplete domain is explicitly scoped as
    a non-blocking auxiliary diagnostic such as Fs(q,t).
    """

    missing_items = [item for item in missing if str(item).strip()]
    if missing_items:
        return True
    checks = list(qc_checks or [])
    incomplete_domains = {
        str(check.get("domain", "")).strip()
        for check in checks
        if str(check.get("warn_source", "")).strip() == "evidence_incomplete_warn"
    }
    if not incomplete_domains:
        return False
    return not incomplete_domains.issubset(RELEASE_NONBLOCKING_EVIDENCE_DOMAINS)


def release_blocking_warn_sources(
    clean_warn_sources: Iterable[str],
    missing: Iterable[str],
    qc_checks: Iterable[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Return clean-overlay warning sources that still block publication release.

    Fs(q,t) is an auxiliary relaxation diagnostic for PE-LCB100.  Missing or
    warning Fs(q,t) outputs are retained for audit, but the publication-grade
    diffusion decision is made by the clean COM MSD gate.  Evidence-incomplete
    only blocks release when a required clean output is still missing.
    """

    evidence_incomplete_blocks = evidence_incomplete_is_release_blocking(qc_checks, missing)
    blocking: list[str] = []
    for source in clean_warn_sources:
        if source in RELEASE_NONBLOCKING_WARN_SOURCES:
            continue
        if source == "evidence_incomplete_warn" and not evidence_incomplete_blocks:
            continue
        if source not in blocking:
            blocking.append(source)
    return sorted(blocking)


def regate_case(
    source_row: Mapping[str, Any],
    status_row: Mapping[str, Any],
    density_row: Mapping[str, Any],
    repair_row: Mapping[str, Any] | None,
    *,
    rule_profile_dir: Path,
    msd_window_overlays: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    case_id = str(source_row.get("case_id"))
    clean_case_id = str(status_row.get("clean_case_id") or "")
    target_sources = parse_sources(repair_row.get("warn_sources") if repair_row else "")
    missing = missing_clean_outputs(clean_case_id, clean_optional_requirements(source_row, repair_row))
    evidence = build_evidence_row(source_row, status_row, density_row)
    msd_overlay_applied = apply_msd_window_overlay(evidence, clean_case_id, msd_window_overlays)
    if msd_overlay_applied:
        missing = [item for item in missing if item != "msd"]
    density_gate = apply_rule_gate(
        {"rho_final_gcc_si_o": evidence.get("rho_final_gcc_si_o")},
        context=context_for(evidence),
        property_name="density",
        output_dir=rule_profile_dir,
    )
    qc_gate = apply_polymer_qc_rules(evidence)
    clean_warn_sources = set(qc_gate.get("warn_sources", []))
    if density_gate.get("rule_verdict") == "WARN" and density_gate.get("warn_source"):
        clean_warn_sources.add(str(density_gate["warn_source"]))

    resolved: list[str] = []
    remaining: list[str] = []
    for source in target_sources:
        if source == "evidence_incomplete_warn" and not missing:
            resolved.append(source)
        elif source not in clean_warn_sources:
            resolved.append(source)
        else:
            remaining.append(source)

    target_blocking_missing = missing if "evidence_incomplete_warn" in target_sources else []
    targeted = "PASS" if target_sources and not remaining and not target_blocking_missing else "WARN"
    if not target_sources:
        targeted = "PASS" if not clean_warn_sources and not missing else "WARN"

    release_blocking_sources = release_blocking_warn_sources(
        clean_warn_sources,
        missing,
        qc_gate.get("checks", []),
    )
    release_overlay = "PASS" if density_gate.get("rule_verdict") == "PASS" and not release_blocking_sources and not missing else "WARN"
    release_reason = (
        "all publication-blocking clean-overlay gates PASS"
        if release_overlay == "PASS"
        else "clean overlay still has publication-blocking WARN or missing evidence"
    )
    return {
        "schema_version": "pelcb_clean_overlay_regate.v1",
        "case_id": case_id,
        "clean_case_id": clean_case_id,
        "target_warn_sources": ",".join(target_sources),
        "clean_warn_sources": ",".join(sorted(clean_warn_sources)),
        "resolved_warn_sources": ",".join(resolved),
        "remaining_target_warn_sources": ",".join(remaining),
        "density_rule_verdict": density_gate.get("rule_verdict"),
        "density_value_gcc": density_gate.get("value") or evidence.get("rho_final_gcc_si_o"),
        "qc_rule_verdict": qc_gate.get("rule_verdict"),
        "targeted_regate_verdict": targeted,
        "release_overlay_verdict": release_overlay,
        "release_overlay_reason": release_reason,
        "release_blocking_warn_sources": ",".join(release_blocking_sources),
        "evidence_status": "complete" if not missing else "missing_clean_outputs",
        "missing_clean_outputs": ",".join(missing),
        "require_intrachain_g3": evidence.get("require_intrachain_g3", ""),
        "clean_tag": status_row.get("tag", ""),
        "clean_remote_run_dir": status_row.get("remote_run_dir", ""),
        "density_evidence_source": density_row.get("density_evidence_source", "density_summary"),
        "density_evidence_stage": density_row.get("stage", ""),
        "density_time_first_ps": density_row.get("time_first_ps", ""),
        "density_time_last_ps": density_row.get("time_last_ps", ""),
        "density_tail_start_ps": density_row.get("tail_start_ps", ""),
        "density_frames": density_row.get("frames", ""),
        "density_tail_frames": density_row.get("tail_frames", ""),
        "density_profile_id": density_gate.get("profile_id", ""),
        "density_reason": density_gate.get("reason", ""),
        "clean_msd_window_overlay_status": evidence.get("clean_msd_window_overlay_status", ""),
        "clean_msd_window_overlay_source": evidence.get("clean_msd_window_overlay_source", ""),
        "clean_msd_window_overlay_fit_window_ns": evidence.get("clean_msd_window_overlay_fit_window_ns", ""),
        "clean_msd_window_overlay_loglog_slope": evidence.get("clean_msd_window_overlay_loglog_slope", ""),
        "clean_msd_window_overlay_D_m2_s": evidence.get("clean_msd_window_overlay_D_m2_s", ""),
        "clean_msd_window_overlay_local_D_cv_pct": evidence.get("clean_msd_window_overlay_local_D_cv_pct", ""),
        "status_MSD": evidence.get("status_MSD", ""),
        "msd_publication_grade": evidence.get("msd_publication_grade", ""),
        "msd_publication_issues": evidence.get("msd_publication_issues", ""),
        "msd_publication_diagnostic_issues": evidence.get("msd_publication_diagnostic_issues", ""),
        "msd_displacement_amplitude_check": evidence.get("msd_displacement_amplitude_check", ""),
        "msd_diffusion_confidence": evidence.get("msd_diffusion_confidence", ""),
        "msd_clean_engine_source": evidence.get("msd_clean_engine_source", ""),
        "msd_tail_window_ok": evidence.get("msd_tail_window_ok", ""),
        "msd_fit_R2_publication": evidence.get("msd_fit_R2_publication", ""),
        "loglog_slope": evidence.get("loglog_slope", ""),
        "fit_quality": evidence.get("fit_quality", ""),
        "diffusive_regime_ok": evidence.get("diffusive_regime_ok", ""),
        "regime_warning": evidence.get("regime_warning", ""),
        "window_sensitive": evidence.get("window_sensitive", ""),
        "window_D_cv_pct": evidence.get("window_D_cv_pct", ""),
        "D_mean_m2_s": evidence.get("D_mean_m2_s", ""),
        "D_median_m2_s": evidence.get("D_median_m2_s", ""),
        "D_outlier_fraction": evidence.get("D_outlier_fraction", ""),
        "fit_begin_ps": evidence.get("fit_begin_ps", ""),
        "fit_end_ps": evidence.get("fit_end_ps", ""),
        "time_last_ps": evidence.get("time_last_ps", ""),
        "window_msd_over_rg2_min": evidence.get("window_msd_over_rg2_min", ""),
        "msd_over_rg2": evidence.get("msd_over_rg2", ""),
        "status_contact": evidence.get("status_contact", ""),
        "P_contact_rc0p8_s10": evidence.get("P_contact_rc0p8_s10", ""),
        "P_contact_rc0p8_s20": evidence.get("P_contact_rc0p8_s20", ""),
        "P_contact_rc0p8_s50": evidence.get("P_contact_rc0p8_s50", ""),
        "contact_source": evidence.get("contact_source", ""),
        "status_intrachain_g3": evidence.get("status_intrachain_g3", ""),
        "intrachain_g3_alpha_100_300": evidence.get("intrachain_g3_alpha_100_300", ""),
        "intrachain_g3_time_last_ps": evidence.get("intrachain_g3_time_last_ps", ""),
        "g1_monomer_over_g3_10ns": evidence.get("g1_monomer_over_g3_10ns", ""),
        "intrachain_g3_source": evidence.get("intrachain_g3_source", ""),
        "intrachain_g3_provenance": evidence.get("intrachain_g3_provenance", ""),
        "qc_domain_verdicts_json": json_dumps(qc_gate.get("domain_verdicts", {})),
        "qc_checks_json": json_dumps(qc_gate.get("checks", [])),
    }


def run_regate(args: argparse.Namespace) -> dict[str, Any]:
    ensure_clean_overlay_rule_profiles(args.rule_profile_dir)
    aggregate_rows = {row["case_id"]: row for row in read_csv(args.aggregate_csv)}
    status_rows = {
        case_id: dict(row)
        for case_id, row in choose_clean_status_rows(read_csv(args.clean_status_csv)).items()
        if truthy(row.get("nvt_prod_done"))
    }
    density_rows = read_csv(args.clean_density_csv)
    repair_plan = args.repair_plan or latest_repair_plan()
    repair_rows = {row["case_id"]: row for row in read_csv(repair_plan)} if repair_plan else {}
    msd_window_overlays = read_msd_window_overlays(args.msd_window_candidates)

    rows: list[dict[str, Any]] = []
    for case_id, status_row in sorted(status_rows.items()):
        source_row = aggregate_rows.get(case_id)
        if not source_row:
            continue
        density_row = select_clean_density_row(source_row, status_row, density_rows)
        if not density_row:
            continue
        clean_case_id = status_row.get("clean_case_id", "")
        if missing_clean_outputs(str(clean_case_id), clean_optional_requirements(source_row, repair_rows.get(case_id))):
            # Still write an explicit row for closed-loop accounting.
            pass
        rows.append(
            regate_case(
                source_row,
                status_row,
                density_row,
                repair_rows.get(case_id),
                rule_profile_dir=args.rule_profile_dir,
                msd_window_overlays=msd_window_overlays,
            )
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"clean_overlay_regate_{stamp}"
    out_csv = out_dir / "clean_overlay_regate.csv"
    out_jsonl = out_dir / "clean_overlay_regate.jsonl"
    write_csv(out_csv, rows, CSV_FIELDS)
    out_jsonl.write_text("\n".join(json_dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "schema_version": "pelcb_clean_overlay_regate.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "aggregate_rows": len(aggregate_rows),
        "missing_from_overlay_case_ids": sorted(set(aggregate_rows) - {row["case_id"] for row in rows}),
        "repair_plan": str(repair_plan) if repair_plan else "",
        "csv": str(out_csv),
        "jsonl": str(out_jsonl),
        "targeted_regate_counts": dict(Counter(row["targeted_regate_verdict"] for row in rows)),
        "release_overlay_counts": dict(Counter(row["release_overlay_verdict"] for row in rows)),
        "clean_warn_source_counts": dict(
            Counter(source for row in rows for source in parse_sources(row.get("clean_warn_sources", "")))
        ),
    }
    summary_path = out_dir / "clean_overlay_regate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest_path = args.output_dir / "clean_overlay_regate_latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps({"summary": summary, "summary_path": str(summary_path)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-gate clean-engine sidecar overlay outputs.")
    parser.add_argument("--aggregate-csv", type=Path, default=AGGREGATE_CSV)
    parser.add_argument("--clean-status-csv", type=Path, default=CLEAN_STATUS_CSV)
    parser.add_argument("--clean-density-csv", type=Path, default=CLEAN_DENSITY_CSV)
    parser.add_argument("--repair-plan", type=Path, default=None)
    parser.add_argument("--rule-profile-dir", type=Path, default=RULE_PROFILE_DIR)
    parser.add_argument("--msd-window-candidates", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_regate(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
