"""Canonical v2 paths and field schemas for MD analysis outputs."""

import os
from datetime import datetime
from pathlib import Path


ANALYSIS_VERSION = "canonical_v2"

COMMON_FIELDS = [
    "case_id",
    "status",
    "n_chains",
    "n_frames",
    "time_first_ps",
    "time_last_ps",
    "dt_ps",
    "method",
    "analysis_version",
    "source_case_dir",
]

RG_FIELDS = COMMON_FIELDS + [
    "rg_mean_nm",
    "rg_std_time_nm",
    "rg_sem_time_nm",
    "rg2_mean_nm2",
    "rg2_over_linear_rg2",
    "linear_baseline_case_id",
    "baseline_status",
]

RG_CONTRACTION_FIELDS = [
    "case_id",
    "chain_carbons",
    "rg_mean_nm",
    "rg_std_time_nm",
    "rg2_mean_nm2",
    "rg2_error_approx_nm2",
    "linear_baseline_case_id",
    "rg_linear_baseline_nm",
    "rg2_over_linear_rg2",
    "baseline_status",
    "note",
]

REE_FIELDS = COMMON_FIELDS + [
    "ree_endpoint_strategy",
    "ree_mean_nm",
    "ree_std_time_nm",
    "ree_chain_mean_nm",
    "ree2_mean_nm2",
]

MSD_FIELDS = COMMON_FIELDS + [
    "msd_final_nm2",
    "D_mean_m2_s",
    "D_std_m2_s",
    "D_mean_1e-5_cm2_s",
    "D_median_m2_s",
    "D_mad_m2_s",
    "D_n_outliers",
    "D_outlier_fraction",
    "fit_begin_ps",
    "fit_end_ps",
    "loglog_slope",
    "fit_quality",
    "diffusive_regime_ok",
    "regime_warning",
    "window_sensitive",
]

RDF_FIELDS = COMMON_FIELDS + [
    "rdf_kind",
    "selection",
    "first_peak_r_nm",
    "first_min_r_nm",
    "first_peak_g",
    "tail_g_mean",
    "tail_g_std",
    "normalization",
    "tail_normalization_status",
    "raw_tail_g_mean",
    "raw_tail_g_std",
]

DYNAMICS_FIELDS = COMMON_FIELDS + [
    "alpha_100_300_mean",
    "alpha_100_300_status",
    "tau_ree_ps",
    "tau_ree_censored",
    "tau_fs_ps",
    "tau_fs_censored",
    "tau_alpha_ps",
    "tau_alpha_censored",
    "q_nm_inv",
    "q_source",
]


def default_output_root():
    return Path(__file__).resolve().parent / "output"


def default_date_tag():
    return datetime.now().strftime("%Y%m%d")


def batch_dir(output_root, observable, date_tag):
    project_id = os.environ.get("LCB_PROJECT_ID", "lcb300").strip() or "lcb300"
    return Path(output_root) / f"{project_id}_{observable}_{date_tag}"


FINAL_OBSERVABLE_DIRS = {
    "MSD",
    "Rg",
    "Ree",
    "RDF",
    "ShapeTensor",
    "Dynamics",
    "IntrachainDynamics",
    "Viscosity",
    "Density",
    "StructureViews",
    "StructureDynamics",
}


def is_final_observable_root(path):
    path = Path(path)
    return path.name in FINAL_OBSERVABLE_DIRS or ((path / "raw").exists() and (path / "summary").exists())


def aggregate_dir(batch_root):
    root = Path(batch_root)
    if is_final_observable_root(root):
        return root / "summary"
    return root / "aggregate"


def case_dir(batch_root, case_id, observable):
    root = Path(batch_root)
    if is_final_observable_root(root):
        return root / "raw" / "cases" / str(case_id) / str(observable)
    return root / "cases" / str(case_id) / str(observable)


def manifest_path(batch_root, observable):
    return aggregate_dir(batch_root) / f"{observable}_normalization_manifest.json"
