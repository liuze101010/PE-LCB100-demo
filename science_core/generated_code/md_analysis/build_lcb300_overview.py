"""Build LCB300-wide analysis tables and overview plots.

The script is intentionally tolerant of incomplete analysis coverage. Every
LCB300 design point is kept in the output tables; missing observables are
reported explicitly as MISSING instead of being dropped from figures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from case_labels import clean_int_text, generic_metadata
from lcb_project_config import load_lcb_config

try:
    from plot_style import apply_rc, save_fig
except ImportError:  # pragma: no cover
    from .plot_style import apply_rc, save_fig  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ID = "lcb300"
DEFAULT_MANIFEST = SCRIPT_DIR.parent / "emc_alpha_olefin_builder" / "output" / "screen_lcb300" / "manifest.csv"
LCB300_ROOT = SCRIPT_DIR / "output" / "lcb300_integrated"
DEFAULT_SUMMARY = LCB300_ROOT / "StructureDynamics" / "summary" / "structure_dynamics_summary.csv"
DEFAULT_DENSITY_DIR = LCB300_ROOT / "Density" / "raw" / "series"
DEFAULT_OUTPUT_DIR = LCB300_ROOT / "aggregate"
DEFAULT_ANALYSIS_ROOT = SCRIPT_DIR / "output"

OBSERVABLE_DIR_NAMES = {
    "viscosity": "Viscosity",
    "rg": "Rg",
    "msd": "MSD",
    "ree": "Ree",
    "rdf": "RDF",
    "shape": "ShapeTensor",
    "dynamics": "Dynamics",
    "intrachain_dynamics": "IntrachainDynamics",
    "density": "Density",
    "structure_views": "StructureViews",
    "structure_dynamics": "StructureDynamics",
}

OBSERVABLES: dict[str, dict[str, Any]] = {
    "eta_Pa_s": {"unit": "Pa s", "label": "viscosity eta", "status": "eta", "log10": True},
    "rg_mean_nm": {"unit": "nm", "label": "Rg", "status": "Rg"},
    "graph_diameter2_over_6rg2": {
        "unit": "1",
        "label": "Graph-diameter span^2 / 6Rg^2",
        "status": "shape",
    },
    "relative_shape_anisotropy_mean": {"unit": "1", "label": "shape anisotropy", "status": "shape"},
    "D_mean_m2_s": {"unit": "m2/s", "label": "chain diffusion D", "status": "MSD", "log10": True},
    "loglog_slope": {"unit": "1", "label": "MSD log-log slope", "status": "MSD"},
    "g1_over_g3_10ns": {"unit": "1", "label": "g1/g3 at 10 ns", "status": "dynamics"},
    "g1_monomer_over_g3_10ns": {"unit": "1", "label": "monomer g1/g3 at 10 ns", "status": "dynamics"},
    "tau_fs_ps": {"unit": "ps", "label": "Fs tau", "status": "dynamics", "log10": True},
    "P_contact_rc0p8_s20": {"unit": "1", "label": "Pcontact s20", "status": "contact"},
    "rdf_intra_chain_nonbonded_cc_first_peak_g": {
        "unit": "1",
        "label": "intra RDF peak g",
        "status": "RDF",
    },
    "rdf_inter_chain_cc_first_peak_g": {"unit": "1", "label": "inter RDF peak g", "status": "RDF"},
}

HISTORICAL_STATUS_COLUMNS = ["eta"]
STATUS_COLUMNS = ["Rg", "shape", "MSD", "dynamics", "fsqt_tau_alpha", "contact", "RDF", "density"]
ALL_STATUS_COLUMNS = HISTORICAL_STATUS_COLUMNS + STATUS_COLUMNS
STATUS_SCORE = {"MISSING": 0, "OK": 1, "WARN": 2, "FAIL": 3}
STATUS_COLORS = ["#d9d9d9", "#3f8f6b", "#d97b2b", "#b84a4a"]
FAMILY_COLORS = {
    "single-branch": "#0072B2",
    "terminal-single": "#56B4E9",
    "comb": "#009E73",
    "two-end": "#E69F00",
    "pompom": "#CC79A7",
    "three-point": "#332288",
    "cluster": "#AA4499",
    "gradient": "#44AA99",
    "center-packed": "#999933",
    "terminal-packed": "#D55E00",
    "wide-center": "#88CCEE",
    "split": "#DDCC77",
    "other": "#888888",
}

DEPRECATED_TERMINAL_REE_FIELDS = {
    "raw_backbone_terminal_ree_nm",
    "raw_backbone_terminal_ree2_nm2",
    "ree2_over_rg2",
    "ree_over_sqrt_backbone_carbons",
    "ree2_over_backbone_carbons",
    "ree_definition_warning",
    "tau_ree_ps",
    "tau_ree_censored",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LCB300 overview tables and plots.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--density-dir", type=Path, default=DEFAULT_DENSITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def lcb300_root(analysis_root: Path) -> Path:
    return Path(analysis_root) / f"{PROJECT_ID}_integrated"


def observable_base(analysis_root: Path, observable: str) -> Path:
    """Return the final LCB300 observable root, with legacy fallback."""

    root = lcb300_root(analysis_root)
    final = root / OBSERVABLE_DIR_NAMES[observable]
    legacy = root / "observables" / observable
    if final.exists() or not legacy.exists():
        return final
    return legacy


def observable_summary_file(analysis_root: Path, observable: str, filename: str) -> Path:
    base = observable_base(analysis_root, observable)
    candidates = [
        base / "summary" / filename,
        base / "summary" / "dataset" / filename,
        base / "aggregate" / filename,
        base / "dataset" / filename,
        base / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    batch_observable = {
        "rg": "rg",
        "msd": "msd",
        "ree": "ree",
        "rdf": "rdf",
        "dynamics": "dynamics",
    }.get(observable)
    if batch_observable:
        batch_matches = sorted(
            Path(analysis_root).glob(f"{PROJECT_ID}_{batch_observable}_*/aggregate/{filename}"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        for path in batch_matches:
            if path.exists():
                return path
    return candidates[0]


def case_sequence(case_id: str) -> int:
    match = re.match(r"(?:lcb(\d+)_|lcb_[a-z]+_(\d+)_)", str(case_id), flags=re.IGNORECASE)
    if not match:
        return 999999
    seq = next((group for group in match.groups() if group), None)
    return int(seq) if seq else 999999


def topology_slug(case_id: str) -> str:
    if re.match(r"lcb_[a-z]+_\d+_", str(case_id), flags=re.IGNORECASE):
        return ""
    match = re.match(r"lcb\d+_([^_]+)_", str(case_id))
    return match.group(1) if match else ""


def topology_family(row: pd.Series | dict[str, Any]) -> str:
    slug = str(row.get("topology_slug") or topology_slug(str(row.get("case_id", "")))).lower()
    topology = str(row.get("branch_topology", "")).lower()
    pattern = str(row.get("branch_position_pattern", "")).lower()
    text = " ".join([slug, topology, pattern])
    if slug in {"star3"} or "three-arm-star" in topology:
        return "single-branch"
    if slug == "center" or "center-branched" in topology:
        return "terminal-single"
    if "db" in slug or "htype" in slug or "h-type" in topology or "dumbbell" in topology:
        return "two-end"
    if "pompom" in text:
        return "pompom"
    if "clust" in slug or "cluster" in text:
        return "cluster"
    if "grad" in slug or "gradient" in text:
        return "gradient"
    if "tri" in slug or "three-point" in topology:
        return "three-point"
    if "wide" in text:
        return "wide-center"
    if "split" in text or "isplit" in slug:
        return "split"
    if "start" in text or "end" in text:
        return "terminal-packed"
    if "center" in text:
        return "center-packed"
    if "comb" in text:
        return "comb"
    return "other"


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return text != "" and text.lower() != "nan"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def manifest_rows(manifest_csv: Path) -> pd.DataFrame:
    df = read_csv(manifest_csv)
    if df.empty:
        raise FileNotFoundError(f"No manifest rows found in {manifest_csv}")
    df = df[df["case_id"].astype(str).str.match(r"(?:lcb\d+_|lcb_[a-z]+_\d+_)", case=False)].copy()
    df["case_seq"] = df["case_id"].map(case_sequence)
    df = df[df["case_seq"] >= 1].copy()
    df["topology_slug"] = df["case_id"].map(topology_slug)
    df["topology_family"] = df.apply(topology_family, axis=1)
    for col in ["branch_points", "branch_carbons", "actual_total_carbons_per_chain"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    generic_rows = [generic_metadata(row["case_id"], row) for _, row in df.iterrows()]
    for key in [
        "source_case_id",
        "generic_case_name",
        "generic_plot_label",
        "polymer_family",
        "branch_point_count",
        "branch_arm_carbons",
        "branch_distribution_code",
        "branch_distribution_label",
        "legacy_template_topology",
        "naming_version",
    ]:
        df[key] = [item[key] for item in generic_rows]
    return df.sort_values("case_seq").reset_index(drop=True)


def load_summary(input_summary: Path) -> pd.DataFrame:
    summary = read_csv(input_summary)
    if summary.empty:
        return pd.DataFrame(columns=["case_id"])
    summary = summary.copy()
    summary["case_seq"] = summary["case_id"].map(case_sequence)
    return summary.drop_duplicates("case_id", keep="last")


def density_case_ids(density_dir: Path, analysis_root: Path | None = None) -> set[str]:
    out = set()
    if density_dir.exists():
        for path in density_dir.glob("lcb*.xvg"):
            match = re.match(r"((?:lcb\d+|lcb_[a-z]+_\d+)_[^.]*?)$", path.stem, flags=re.IGNORECASE)
            if match:
                out.add(match.group(1))
    if analysis_root is not None:
        density_energy = Path(analysis_root) / f"{PROJECT_ID}_density_energy"
        if density_energy.exists():
            for path in density_energy.glob("*/analysis.json"):
                out.add(path.parent.name)
    return out


def merge_wide(manifest: pd.DataFrame, summary: pd.DataFrame, density_ids: set[str]) -> pd.DataFrame:
    overlap = [c for c in summary.columns if c in manifest.columns and c != "case_id"]
    summary = summary.drop(columns=overlap, errors="ignore")
    wide = manifest.merge(summary, on="case_id", how="left", suffixes=("", "_analysis"))
    wide = drop_deprecated_terminal_ree(wide)
    wide["has_density_series"] = wide["case_id"].isin(density_ids)
    wide["any_analyzed"] = False
    for column in OBSERVABLES:
        if column in wide.columns:
            wide["any_analyzed"] = wide["any_analyzed"] | wide[column].map(is_present)
    wide["any_analyzed"] = wide["any_analyzed"] | wide["has_density_series"]
    return wide


def drop_deprecated_terminal_ree(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.drop(columns=[col for col in DEPRECATED_TERMINAL_REE_FIELDS if col in wide.columns], errors="ignore")


def merge_case_table(wide: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "case_id" not in table.columns:
        return wide
    table = table.drop_duplicates("case_id", keep="last").copy()
    protected = {
        "case_seq",
        "branch_points",
        "branch_carbons",
        "actual_total_carbons_per_chain",
        "branch_topology",
        "branch_position_pattern",
        "topology_slug",
        "topology_family",
        "source_case_id",
        "generic_case_name",
        "generic_plot_label",
        "polymer_family",
        "branch_point_count",
        "branch_arm_carbons",
        "branch_distribution_code",
        "branch_distribution_label",
        "legacy_template_topology",
        "naming_version",
    }
    overlap = [col for col in table.columns if col in wide.columns and col != "case_id"]
    table = table.drop(columns=[col for col in overlap if col in protected], errors="ignore")
    overlap = [col for col in table.columns if col in wide.columns and col != "case_id"]
    merged = wide.merge(table, on="case_id", how="left", suffixes=("", "__new"))
    for col in overlap:
        new_col = f"{col}__new"
        if new_col in merged.columns:
            # Earlier sources have higher precedence; later tables fill missing values only.
            merged[col] = merged[col].combine_first(merged[new_col])
            merged = merged.drop(columns=[new_col])
    return merged


def with_intrachain_case_json(analysis_root: Path, table: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Append latest per-case intrachain summaries to an aggregate table.

    Postprocess closure may collect updated per-case files before the aggregate
    CSV has been refreshed.  The overview is the canonical user-facing table, so
    it should prefer case-local summaries when they are present instead of
    reporting stale MISSING contact/MSD status.
    """

    base = observable_base(analysis_root, "intrachain_dynamics")
    rows: list[dict[str, Any]] = []
    for cases_root in [base / "cases", base / "raw" / "cases"]:
        if not cases_root.exists():
            continue
        for case_dir in sorted(cases_root.iterdir()):
            if not case_dir.is_dir():
                continue
            path = case_dir / "intrachain" / filename
            if not path.exists():
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            row.setdefault("case_id", case_dir.name)
            rows.append(row)
    if not rows:
        return table
    return pd.concat([table, pd.DataFrame(rows)], ignore_index=True)


RDF_KIND_PREFIX = {
    "all": "rdf_all",
    "excl": "rdf_excl",
    "inter_chain_cc": "rdf_inter_chain_cc",
    "intra_chain_nonbonded_cc": "rdf_intra_chain_nonbonded_cc",
}

RDF_REQUIRED_FEATURE_COLUMNS = [
    f"{prefix}_first_peak_g" for prefix in RDF_KIND_PREFIX.values()
]


def rdf_features(rdf: pd.DataFrame) -> pd.DataFrame:
    if rdf.empty or "case_id" not in rdf.columns:
        return pd.DataFrame(columns=["case_id"])
    rows: list[dict[str, Any]] = []
    for case_id, sub in rdf.groupby("case_id", dropna=False):
        row: dict[str, Any] = {"case_id": case_id}
        for item in sub.to_dict("records"):
            kind = str(item.get("rdf_kind", "")).lower()
            prefix = RDF_KIND_PREFIX.get(kind)
            if not prefix:
                continue
            for source, suffix in [
                ("first_peak_r_nm", "first_peak_r_nm"),
                ("first_min_r_nm", "first_min_r_nm"),
                ("first_peak_g", "first_peak_g"),
                ("tail_g_mean", "tail_g_mean"),
                ("tail_g_std", "tail_g_std"),
            ]:
                row[f"{prefix}_{suffix}"] = item.get(source, "")
        rows.append(row)
    return pd.DataFrame(rows)


def merge_observable_aggregates(wide: pd.DataFrame, analysis_root: Path) -> pd.DataFrame:
    """Left-join canonical observable aggregates directly onto the LCB300 manifest."""
    root = Path(analysis_root)
    for path in [
        observable_summary_file(root, "viscosity", "summary.csv"),
        observable_summary_file(root, "rg", "rg_summary.csv"),
        observable_summary_file(root, "msd", "msd_summary.csv"),
        observable_summary_file(root, "shape", "shape_tensor_summary.csv"),
        observable_summary_file(root, "dynamics", "dynamics_summary.csv"),
    ]:
        wide = merge_case_table(wide, read_csv(path))
    for filename in [
        "contact_probability_summary.json",
        "g1_g3_summary.json",
        "g1_monomer_summary.json",
    ]:
        csv_name = filename.replace(".json", ".csv")
        table = read_csv(observable_summary_file(root, "intrachain_dynamics", csv_name))
        wide = merge_case_table(wide, with_intrachain_case_json(root, table, filename))

    rdf = rdf_features(read_csv(observable_summary_file(root, "rdf", "rdf_summary.csv")))
    wide = merge_case_table(wide, rdf)
    return drop_deprecated_terminal_ree(wide)


def status_for(row: pd.Series, name: str) -> str:
    if name == "eta":
        if not is_present(row.get("eta_Pa_s")):
            return "MISSING"
        eta = to_float(row.get("eta_Pa_s"))
        r2 = to_float(row.get("r2_min"))
        spread = to_float(row.get("component_relative_std"))
        if eta <= 0:
            return "FAIL"
        if (math.isfinite(r2) and r2 < 0.95) or (math.isfinite(spread) and spread > 0.30):
            return "WARN"
        return "OK"
    if name == "Rg":
        return "OK" if is_present(row.get("rg_mean_nm")) else "MISSING"
    if name == "shape":
        if is_present(row.get("graph_diameter2_over_6rg2")):
            return "OK"
        bad_order = to_float(row.get("validation_n_bad_order"))
        bad_negative = to_float(row.get("validation_n_bad_negative"))
        if not is_present(row.get("relative_shape_anisotropy_mean")):
            return "MISSING"
        return "WARN" if bad_order > 0 or bad_negative > 0 else "OK"
    if name == "MSD":
        if not is_present(row.get("D_mean_m2_s")) and not is_present(row.get("loglog_slope")):
            return "MISSING"
        if str(row.get("fit_quality", "")).lower() not in {"", "ok"}:
            return "WARN"
        if str(row.get("diffusive_regime_ok", "")).lower() in {"false", "0", "no"}:
            return "WARN"
        if truthy(row.get("window_sensitive")):
            return "WARN"
        return "OK"
    if name == "dynamics":
        present = any(
            is_present(row.get(col))
            for col in ["tau_fs_ps", "g1_over_g3_10ns", "g1_monomer_over_g3_10ns"]
        )
        if not present:
            return "MISSING"
        if truthy(row.get("tau_fs_censored")):
            return "WARN"
        return "OK"
    if name == "fsqt_tau_alpha":
        if not is_present(row.get("q_source")) and not is_present(row.get("q_nm_inv")):
            return "MISSING"
        if not is_present(row.get("tau_alpha_ps")) and not is_present(row.get("tau_fs_ps")):
            return "MISSING"
        if truthy(row.get("tau_alpha_censored")) or truthy(row.get("tau_fs_censored")):
            return "WARN"
        return "OK"
    if name == "contact":
        return "OK" if is_present(row.get("P_contact_rc0p8_s20")) else "MISSING"
    if name == "RDF":
        return "OK" if all(is_present(row.get(col)) for col in RDF_REQUIRED_FEATURE_COLUMNS) else "MISSING"
    if name == "density":
        return "OK" if bool(row.get("has_density_series")) else "MISSING"
    return "MISSING"


def add_quality_flags(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.copy()
    for name in ALL_STATUS_COLUMNS:
        wide[f"status_{name}"] = wide.apply(lambda row: status_for(row, name), axis=1)
    wide["rdf_validation_status"] = wide["status_RDF"].map(
        lambda status: "current" if str(status).upper() == "OK" else "excluded"
    )
    wide["rdf_publication_usable"] = wide["rdf_validation_status"].isin({"current", "validated"})
    flag_columns = [f"status_{name}" for name in STATUS_COLUMNS]
    flags = []
    for _, row in wide.iterrows():
        case_flags = []
        for name, column in zip(STATUS_COLUMNS, flag_columns):
            status = row[column]
            if status in {"WARN", "FAIL"}:
                case_flags.append(f"{name}:{status}")
        if not case_flags:
            case_flags.append("MISSING_ALL" if not row["any_analyzed"] else "OK")
        flags.append(";".join(case_flags))
    wide["quality_flags"] = flags
    wide["status_ok_count"] = wide[flag_columns].eq("OK").sum(axis=1)
    wide["status_warn_count"] = wide[flag_columns].eq("WARN").sum(axis=1)
    wide["status_fail_count"] = wide[flag_columns].eq("FAIL").sum(axis=1)
    wide["status_missing_count"] = wide[flag_columns].eq("MISSING").sum(axis=1)
    return wide


def design_matrix_audit(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = wide.groupby(["branch_points", "branch_carbons"], dropna=False)
    for (bval, lval), sub in grouped:
        sub = sub.sort_values("case_seq")
        topology_patterns = sorted(
            {
                f"{row.branch_topology}:{row.branch_position_pattern}"
                for row in sub.itertuples(index=False)
            }
        )
        rows.append(
            {
                "branch_points": int(bval),
                "branch_carbons": int(lval),
                "case_count": int(len(sub)),
                "analyzed_count": int(sub["any_analyzed"].sum()),
                "unique_topology_count": int(sub["branch_topology"].nunique()),
                "unique_topology_pattern_count": int(len(topology_patterns)),
                "case_ids": ";".join(sub["case_id"].astype(str)),
                "topologies": ";".join(sorted(sub["branch_topology"].astype(str).unique())),
                "topology_patterns": ";".join(topology_patterns),
            }
        )
    return pd.DataFrame(rows).sort_values(["branch_points", "branch_carbons"])


def analysis_status_table(wide: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "case_id",
        "source_case_id",
        "generic_case_name",
        "generic_plot_label",
        "polymer_family",
        "case_seq",
        "branch_points",
        "branch_carbons",
        "branch_distribution_code",
        "branch_distribution_label",
        "legacy_template_topology",
        "topology_family",
        "topology_slug",
    ]
    rows = []
    for _, row in wide.sort_values("case_seq").iterrows():
        base = {field: row.get(field, "") for field in fields}
        for name in STATUS_COLUMNS:
            base[name] = row[f"status_{name}"]
        base["rdf_validation_status"] = row.get("rdf_validation_status", "")
        base["quality_flags"] = row["quality_flags"]
        rows.append(base)
    return pd.DataFrame(rows)


def long_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_fields = [
        "case_id",
        "source_case_id",
        "generic_case_name",
        "generic_plot_label",
        "polymer_family",
        "case_seq",
        "branch_points",
        "branch_carbons",
        "branch_distribution_code",
        "branch_distribution_label",
        "legacy_template_topology",
        "topology_family",
        "topology_slug",
        "branch_topology",
        "branch_position_pattern",
        "quality_flags",
    ]
    for _, row in wide.sort_values("case_seq").iterrows():
        base = {field: row.get(field, "") for field in base_fields}
        for observable, meta in OBSERVABLES.items():
            value = row.get(observable, "")
            rows.append(
                {
                    **base,
                    "observable": observable,
                    "observable_label": meta["label"],
                    "value": value if is_present(value) else "",
                    "unit": meta["unit"],
                    "status": row.get(f"status_{meta['status']}", "MISSING"),
                }
            )
    return pd.DataFrame(rows)


def save_tables(wide: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "wide_csv": output_dir / f"{PROJECT_ID}_summary_wide.csv",
        "wide_json": output_dir / f"{PROJECT_ID}_summary_wide.json",
        "long_csv": output_dir / f"{PROJECT_ID}_plotting_long.csv",
        "status_csv": output_dir / f"{PROJECT_ID}_analysis_status_matrix.csv",
        "audit_csv": output_dir / f"{PROJECT_ID}_design_matrix_audit.csv",
        "manifest_json": output_dir / f"{PROJECT_ID}_overview_manifest.json",
    }
    wide.to_csv(paths["wide_csv"], index=False, encoding="utf-8-sig")
    paths["wide_json"].write_text(wide.to_json(orient="records", indent=2, force_ascii=False) + "\n", encoding="utf-8")
    label_fields = [
        "case_id",
        "source_case_id",
        "generic_case_name",
        "generic_plot_label",
        "polymer_family",
        "branch_point_count",
        "branch_arm_carbons",
        "branch_distribution_code",
        "branch_distribution_label",
        "legacy_template_topology",
        "naming_version",
        "case_prefix",
        "plot_label",
        "topology_label",
        "branch_count",
        "branch_length",
        "backbone_carbons",
        "chain_carbons",
    ]
    label_table = wide.copy()
    label_table["branch_count"] = label_table.get("branch_point_count", label_table.get("branch_count", ""))
    label_table["branch_length"] = label_table.get("branch_arm_carbons", label_table.get("branch_length", ""))
    label_table["chain_carbons"] = label_table.get(
        "actual_total_carbons_per_chain",
        label_table.get("chain_carbons", ""),
    )
    if "plot_label" not in label_table.columns:
        label_table["plot_label"] = label_table["generic_plot_label"]
    else:
        label_table["plot_label"] = label_table["generic_plot_label"].combine_first(label_table["plot_label"])
    for col in label_fields:
        if col not in label_table.columns:
            label_table[col] = ""
    for col in [
        "branch_point_count",
        "branch_arm_carbons",
        "branch_count",
        "branch_length",
        "backbone_carbons",
        "chain_carbons",
    ]:
        label_table[col] = label_table[col].map(clean_int_text)
    label_table = label_table[label_fields].drop_duplicates("case_id", keep="last")
    for name in ["case_label_map", "generic_name_map"]:
        csv_path = output_dir / f"{name}.csv"
        json_path = output_dir / f"{name}.json"
        label_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        json_path.write_text(label_table.to_json(orient="records", indent=2, force_ascii=False) + "\n", encoding="utf-8")
        paths[f"{name}_csv"] = csv_path
        paths[f"{name}_json"] = json_path
    long = long_table(wide)
    long.to_csv(paths["long_csv"], index=False, encoding="utf-8-sig")
    status = analysis_status_table(wide)
    status.to_csv(paths["status_csv"], index=False, encoding="utf-8-sig")
    audit = design_matrix_audit(wide)
    audit.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    rep = representative_topology_table(wide)
    rep_path = output_dir / f"{PROJECT_ID}_representative_topologies.csv"
    rep.to_csv(rep_path, index=False, encoding="utf-8-sig")
    paths["representatives_csv"] = rep_path
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_design_cases": int(len(wide)),
        "any_analyzed_cases": int(wide["any_analyzed"].sum()),
        "status_counts": {
            name: dict(Counter(wide[f"status_{name}"]))
            for name in STATUS_COLUMNS
        },
        "historical_status_counts": {
            name: dict(Counter(wide[f"status_{name}"]))
            for name in HISTORICAL_STATUS_COLUMNS
        },
        "rdf_validation_status_counts": dict(Counter(wide.get("rdf_validation_status", pd.Series(dtype=str)))),
        "outputs": {key: str(path) for key, path in paths.items() if key != "manifest_json"},
        "project_id": PROJECT_ID,
        "quality_policy": {
            "rdf": "RDF values are included in the automated postprocessing package and default to current when status_RDF is OK; rdf_validation_status can still be overridden to validated/excluded after audits.",
            "viscosity": "historical/exploratory and outside the default completion gate.",
            "entanglement": "outside the default C100 LCB release.",
        },
        "note": f"Single-sample {PROJECT_ID.upper()} overview; component spread is not replicate uncertainty.",
    }
    paths["manifest_json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def representative_topology_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["topology_family", "topology_slug", "branch_position_pattern"]
    for key, sub in wide.sort_values(["branch_carbons", "case_seq"]).groupby(keys, dropna=False):
        first = sub.iloc[0]
        rows.append(
            {
                "topology_family": key[0],
                "topology_slug": key[1],
                "branch_position_pattern": key[2],
                "representative_case_id": first["case_id"],
                "representative_generic_case_name": first.get("generic_case_name", ""),
                "representative_generic_plot_label": first.get("generic_plot_label", ""),
                "branch_distribution_code": first.get("branch_distribution_code", ""),
                "branch_distribution_label": first.get("branch_distribution_label", ""),
                "legacy_template_topology": first.get("legacy_template_topology", ""),
                "branch_points": int(first["branch_points"]),
                "branch_carbons": int(first["branch_carbons"]),
                "n_design_cases": int(len(sub)),
                "n_analyzed_cases": int(sub["any_analyzed"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["topology_family", "topology_slug", "branch_position_pattern"])


def numeric_grid(
    df: pd.DataFrame,
    value_col: str,
    agg: str = "mean",
    index_col: str = "branch_points",
    column_col: str = "branch_carbons",
) -> tuple[np.ndarray, list[int], list[int]]:
    bvals = sorted(int(v) for v in df[index_col].dropna().unique())
    lvals = sorted(int(v) for v in df[column_col].dropna().unique())
    grid = np.full((len(bvals), len(lvals)), np.nan)
    for i, bval in enumerate(bvals):
        for j, lval in enumerate(lvals):
            vals = pd.to_numeric(
                df[(df[index_col] == bval) & (df[column_col] == lval)][value_col],
                errors="coerce",
            ).dropna()
            if vals.empty:
                continue
            grid[i, j] = float(vals.median() if agg == "median" else vals.mean())
    return grid, bvals, lvals


def plot_design_coverage(wide: pd.DataFrame, figures_dir: Path) -> Path:
    audit = design_matrix_audit(wide)
    bvals = sorted(audit["branch_points"].unique())
    lvals = sorted(audit["branch_carbons"].unique())
    design = np.zeros((len(bvals), len(lvals)))
    for _, row in audit.iterrows():
        i = bvals.index(row["branch_points"])
        j = lvals.index(row["branch_carbons"])
        design[i, j] = row["case_count"]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    im = ax.imshow(design, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(lvals)), [f"C{v}" for v in lvals], rotation=45)
    ax.set_yticks(range(len(bvals)), [f"b={v}" for v in bvals])
    ax.set_xlabel("branch length L")
    ax.set_ylabel("branch count b")
    ax.set_title(f"{PROJECT_ID.upper()} design coverage matrix")
    for i in range(design.shape[0]):
        for j in range(design.shape[1]):
            ax.text(j, i, str(int(design[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82, label="case count")
    return save_one(fig, figures_dir / f"{PROJECT_ID.upper()}_design_coverage_matrix")


def plot_status_matrix(wide: pd.DataFrame, figures_dir: Path) -> Path:
    data = np.array(
        [[STATUS_SCORE[row[f"status_{name}"]] for name in STATUS_COLUMNS] for _, row in wide.sort_values("case_seq").iterrows()],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.6, 10.5))
    cmap = ListedColormap(STATUS_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(STATUS_COLUMNS)), STATUS_COLUMNS, rotation=45, ha="right")
    seqs = list(wide.sort_values("case_seq")["case_seq"])
    tick_idx = list(range(0, len(seqs), 20))
    if len(seqs) - 1 not in tick_idx:
        tick_idx.append(len(seqs) - 1)
    ax.set_yticks(tick_idx, [f"lcb{seqs[i]:03d}" for i in tick_idx])
    ax.set_title(f"{PROJECT_ID.upper()} analysis status matrix")
    legend = [
        Patch(facecolor=STATUS_COLORS[0], label="MISSING"),
        Patch(facecolor=STATUS_COLORS[1], label="OK"),
        Patch(facecolor=STATUS_COLORS[2], label="WARN"),
        Patch(facecolor=STATUS_COLORS[3], label="FAIL"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return save_one(fig, figures_dir / f"{PROJECT_ID.upper()}_analysis_status_matrix")


def plot_quality_dashboard(wide: pd.DataFrame, figures_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    ax = axes[0, 0]
    eta_series = pd.to_numeric(
        wide["eta_Pa_s"] if "eta_Pa_s" in wide.columns else pd.Series([pd.NA] * len(wide), index=wide.index),
        errors="coerce",
    )
    eta_rows = wide[eta_series.notna()]
    if eta_rows.empty:
        ax.text(0.5, 0.5, "No eta data", ha="center", va="center")
    else:
        ax.scatter(
            pd.to_numeric(eta_rows["component_relative_std"], errors="coerce"),
            pd.to_numeric(eta_rows["r2_min"], errors="coerce"),
            c=[FAMILY_COLORS.get(x, "#666666") for x in eta_rows["topology_family"]],
            edgecolors="black",
            linewidths=0.3,
        )
        ax.axhline(0.95, color="#b84a4a", linestyle="--", linewidth=0.8)
        ax.axvline(0.30, color="#b84a4a", linestyle="--", linewidth=0.8)
        ax.set_xlabel("component relative std")
        ax.set_ylabel("r2_min")
    ax.set_title(f"Viscosity QC (n={len(eta_rows)})")

    ax = axes[0, 1]
    msd_source = wide["loglog_slope"] if "loglog_slope" in wide.columns else pd.Series([pd.NA] * len(wide), index=wide.index)
    msd = pd.to_numeric(msd_source, errors="coerce").dropna()
    if msd.empty:
        ax.text(0.5, 0.5, "No MSD slope data", ha="center", va="center")
    else:
        ax.hist(msd, bins=min(12, max(4, len(msd))), color="#6f8294", edgecolor="white")
        ax.axvline(1.0, color="#17212b", linestyle="--", linewidth=0.8)
        ax.set_xlabel("MSD log-log slope")
        ax.set_ylabel("case count")
    ax.set_title(f"MSD regime check (n={len(msd)})")

    ax = axes[1, 0]
    counts = [int((wide[f"status_{name}"] == "OK").sum()) for name in STATUS_COLUMNS]
    ax.bar(range(len(STATUS_COLUMNS)), counts, color="#3f8f6b")
    ax.set_xticks(range(len(STATUS_COLUMNS)), STATUS_COLUMNS, rotation=45, ha="right")
    ax.set_ylabel("OK case count")
    ax.set_title("Observable completion")

    ax = axes[1, 1]
    flag_counts = Counter()
    for flags in wide["quality_flags"].astype(str):
        for flag in flags.split(";"):
            flag_counts[flag] += 1
    common = flag_counts.most_common(10)
    labels = [item[0] for item in common]
    vals = [item[1] for item in common]
    ax.barh(range(len(vals)), vals, color="#d97b2b")
    ax.set_yticks(range(len(vals)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("case count")
    ax.set_title("Top quality flags")
    return save_one(fig, figures_dir / f"{PROJECT_ID.upper()}_quality_dashboard")


def metric_values(df: pd.DataFrame, metric: str, log10: bool) -> pd.Series:
    source = df[metric] if metric in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
    values = pd.to_numeric(source, errors="coerce")
    if log10:
        values = values.where(values > 0).map(lambda x: math.log10(x) if pd.notna(x) else math.nan)
    return values


def plot_metric_facets(wide: pd.DataFrame, metric: str, figures_dir: Path) -> Path:
    meta = OBSERVABLES[metric]
    df = wide.copy()
    df["_value"] = metric_values(df, metric, bool(meta.get("log10")))
    families = [family for family, sub in df.groupby("topology_family") if sub["_value"].notna().any()]
    if not families:
        families = sorted(df["topology_family"].dropna().unique())[:1]
    ncols = min(4, max(1, len(families)))
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.55 * nrows), squeeze=False)
    all_values = df["_value"].dropna()
    vmin = float(all_values.min()) if not all_values.empty else 0.0
    vmax = float(all_values.max()) if not all_values.empty else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#d9d9d9")
    image = None
    for ax, family in zip(axes.ravel(), families):
        sub = df[df["topology_family"] == family]
        grid, bvals, lvals = numeric_grid(sub, "_value", agg="median")
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(lvals)), [f"C{v}" for v in lvals], rotation=45)
        ax.set_yticks(range(len(bvals)), [f"b={v}" for v in bvals])
        n = int(sub["_value"].notna().sum())
        ax.set_title(f"{family} (n={n})")
    for ax in axes.ravel()[len(families):]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label=("log10 " if meta.get("log10") else "") + meta["unit"])
    fig.suptitle(f"{PROJECT_ID.upper()} topology-faceted {meta['label']}")
    return save_one(fig, figures_dir / f"{PROJECT_ID.upper()}_facet_heatmap_{metric}")


def plot_scatter_panels(wide: pd.DataFrame, figures_dir: Path) -> Path:
    pairs = [
        ("rg_mean_nm", "D_mean_m2_s", "Rg", "log10 D", False, True),
        ("graph_diameter2_over_6rg2", "D_mean_m2_s", "span^2/6Rg^2", "log10 D", False, True),
        ("relative_shape_anisotropy_mean", "D_mean_m2_s", "shape anisotropy", "log10 D", False, True),
        ("P_contact_rc0p8_s20", "D_mean_m2_s", "Pcontact s20", "log10 D", False, True),
        ("D_mean_m2_s", "eta_Pa_s", "log10 D", "log10 eta", True, True),
        ("relative_shape_anisotropy_mean", "eta_Pa_s", "shape anisotropy", "log10 eta", False, True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    for ax, (xcol, ycol, xlabel, ylabel, xlog, ylog) in zip(axes.ravel(), pairs):
        x = metric_values(wide, xcol, xlog)
        y = metric_values(wide, ycol, ylog)
        mask = x.notna() & y.notna()
        sub = wide[mask]
        if sub.empty:
            ax.text(0.5, 0.5, "n=0", ha="center", va="center")
        else:
            ax.scatter(
                x[mask],
                y[mask],
                c=[FAMILY_COLORS.get(f, "#666666") for f in sub["topology_family"]],
                s=32,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.25,
            )
            ax.set_title(f"n={len(sub)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=family, markersize=5)
        for family, color in FAMILY_COLORS.items()
        if family in set(wide["topology_family"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"{PROJECT_ID.upper()} structure/contact vs dynamics/rheology")
    return save_one(fig, figures_dir / f"{PROJECT_ID.upper()}_structure_dynamics_scatter_panels")


def plot_feature_correlation(wide: pd.DataFrame, figures_dir: Path) -> Path:
    metrics = [
        "eta_Pa_s",
        "rg_mean_nm",
        "graph_diameter2_over_6rg2",
        "relative_shape_anisotropy_mean",
        "D_mean_m2_s",
        "loglog_slope",
        "P_contact_rc0p8_s20",
        "rdf_intra_chain_nonbonded_cc_first_peak_g",
    ]
    data = pd.DataFrame({m: metric_values(wide, m, bool(OBSERVABLES.get(m, {}).get("log10"))) for m in metrics})
    valid = data.dropna(axis=1, thresh=3)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if valid.shape[1] < 2 or len(valid.dropna(how="all")) < 3:
        ax.text(0.5, 0.5, "Not enough complete feature data for correlation", ha="center", va="center")
        ax.axis("off")
    else:
        corr = valid.corr(min_periods=3)
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.82)
    ax.set_title(f"{PROJECT_ID.upper()} feature correlation screen")
    return save_one(fig, figures_dir / f"{PROJECT_ID.upper()}_feature_correlation_heatmap")


def plot_feature_pca(wide: pd.DataFrame, figures_dir: Path) -> Path:
    metrics = [
        "rg_mean_nm",
        "graph_diameter2_over_6rg2",
        "relative_shape_anisotropy_mean",
        "D_mean_m2_s",
        "loglog_slope",
        "P_contact_rc0p8_s20",
        "g1_over_g3_10ns",
    ]
    data = pd.DataFrame({m: metric_values(wide, m, bool(OBSERVABLES.get(m, {}).get("log10"))) for m in metrics})
    data = data.dropna(axis=1, thresh=3)
    complete = data.dropna()
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if complete.shape[0] < 3 or complete.shape[1] < 2:
        ax.text(0.5, 0.5, "Not enough complete feature data for PCA", ha="center", va="center")
        ax.axis("off")
    else:
        x = complete.to_numpy(dtype=float)
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
        x = np.nan_to_num(x)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        scores = x @ vt[:2].T
        denom = np.sum(s**2)
        explained = (s[:2] ** 2 / denom) if denom else np.array([0.0, 0.0])
        sub = wide.loc[complete.index]
        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            c=[FAMILY_COLORS.get(f, "#666666") for f in sub["topology_family"]],
            edgecolors="black",
            linewidths=0.25,
        )
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        ax.set_title(f"Feature PCA screen (n={len(sub)})")
    return save_one(fig, figures_dir / f"{PROJECT_ID.upper()}_feature_pca_screen")


def save_one(fig: plt.Figure, stem: Path) -> Path:
    save_fig(fig, stem, formats=("pdf", "png"))
    return stem.with_suffix(".png")


def build_plots(wide: pd.DataFrame, output_dir: Path) -> list[Path]:
    apply_rc()
    figures_dir = output_dir / "plots"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_design_coverage(wide, figures_dir),
        plot_status_matrix(wide, figures_dir),
        plot_quality_dashboard(wide, figures_dir),
        plot_scatter_panels(wide, figures_dir),
        plot_feature_correlation(wide, figures_dir),
        plot_feature_pca(wide, figures_dir),
    ]
    for metric in [
        "rg_mean_nm",
        "graph_diameter2_over_6rg2",
        "relative_shape_anisotropy_mean",
        "D_mean_m2_s",
        "P_contact_rc0p8_s20",
        "eta_Pa_s",
    ]:
        outputs.append(plot_metric_facets(wide, metric, figures_dir))
    return outputs


def build(
    manifest_csv: Path,
    input_summary: Path,
    analysis_root: Path,
    density_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if output_dir is None:
        if density_dir is None:
            raise TypeError("build() requires either 5 args or legacy 4 args: manifest, summary, density_dir, output_dir")
        output_dir = density_dir
        density_dir = analysis_root
        analysis_root = Path(output_dir).parent / "_empty_observable_root"
    manifest = manifest_rows(manifest_csv)
    summary = load_summary(input_summary)
    wide = merge_wide(manifest, summary, density_case_ids(density_dir, analysis_root))
    wide = merge_observable_aggregates(wide, analysis_root)
    wide["any_analyzed"] = False
    for column in OBSERVABLES:
        if column in wide.columns:
            wide["any_analyzed"] = wide["any_analyzed"] | wide[column].map(is_present)
    wide["any_analyzed"] = wide["any_analyzed"] | wide["has_density_series"]
    wide = add_quality_flags(wide)
    table_paths = save_tables(wide, output_dir)
    plot_paths = build_plots(wide, output_dir)
    return {
        "wide": wide,
        "table_paths": table_paths,
        "plot_paths": plot_paths,
    }


def main() -> int:
    global PROJECT_ID
    args = parse_args()
    config = None
    if args.config:
        config = load_lcb_config(args.config)
        PROJECT_ID = config.project_id
        args.manifest_csv = config.manifest_csv
        args.analysis_root = config.output_root
        args.output_dir = config.integrated_output_dir
        args.input_summary = config.integrated_root / "StructureDynamics" / "summary" / "structure_dynamics_summary.csv"
        args.density_dir = config.integrated_root / "Density" / "raw" / "series"
    result = build(args.manifest_csv, args.input_summary, args.analysis_root, args.density_dir, args.output_dir)
    wide = result["wide"]
    publication_asset_count = 0
    if config is not None:
        try:
            from lcb_publication_assets import build_publication_assets

            publication_assets = build_publication_assets(config, wide)
            publication_asset_count = len(publication_assets)
        except Exception as exc:  # noqa: BLE001
            print(f"Publication assets skipped: {exc}")
    print(f"{PROJECT_ID.upper()} rows: {len(wide)}")
    print(f"Analyzed rows: {int(wide['any_analyzed'].sum())}")
    print(f"Output: {args.output_dir}")
    print(f"Plots: {len(result['plot_paths'])}")
    if publication_asset_count:
        print(f"Publication assets: {publication_asset_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
