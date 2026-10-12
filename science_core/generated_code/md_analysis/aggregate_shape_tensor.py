#!/usr/bin/env python3
"""Aggregate strict shape-tensor per-case outputs into canonical v2 tables."""

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from case_labels import label_for_case, load_case_label_map
except ImportError:  # pragma: no cover
    from .case_labels import label_for_case, load_case_label_map  # type: ignore


def parse_args():
    root = Path(__file__).resolve().parent / "output" / "lcb100_shape_tensor_20260504"
    parser = argparse.ArgumentParser(description="Aggregate strict carbon-skeleton shape tensor outputs.")
    parser.add_argument("--batch-dir", type=Path, default=root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional final observable root with raw/summary/plots subdirectories.",
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def row_case_fields(summary):
    validation = summary.get("validation", {}) or {}
    row = {
        "case_id": summary.get("case_id", ""),
        "status": summary.get("status", ""),
        "n_chains": summary.get("n_chains", ""),
        "n_frames": summary.get("n_frames", ""),
        "time_first_ps": summary.get("time_first_ps", ""),
        "time_last_ps": summary.get("time_last_ps", ""),
        "dt_ps": summary.get("dt_ps", ""),
        "method": summary.get("method", ""),
        "analysis_version": summary.get("analysis_version", ""),
        "source_case_dir": summary.get("source_case_dir", ""),
        "atom_count": summary.get("atom_count", ""),
        "atoms_per_chain": summary.get("atoms_per_chain", ""),
        "carbon_count": summary.get("carbon_count", ""),
        "carbon_per_chain": summary.get("carbon_per_chain", ""),
        "carbon_graph_connected": summary.get("carbon_graph_connected", ""),
        "graph_diameter_topological_edges": summary.get("graph_diameter_topological_edges", ""),
        "graph_diameter_topological_carbons": summary.get("graph_diameter_topological_carbons", ""),
        "graph_diameter_distance_mean_nm": summary.get("graph_diameter_distance_mean_nm", ""),
        "graph_diameter_distance_std_time_nm": summary.get("graph_diameter_distance_std_time_nm", ""),
        "graph_diameter_distance2_mean_nm2": summary.get("graph_diameter_distance2_mean_nm2", ""),
        "graph_diameter_distance2_std_time_nm2": summary.get("graph_diameter_distance2_std_time_nm2", ""),
        "graph_diameter2_over_6rg2": summary.get("graph_diameter2_over_6rg2", ""),
        "graph_diameter_local_atom_indices_0based": json.dumps(
            summary.get("graph_diameter_local_atom_indices_0based", []), ensure_ascii=True
        ),
        "graph_diameter_order_endpoints": json.dumps(
            summary.get("graph_diameter_order_endpoints", []), ensure_ascii=True
        ),
        "rg_carbon_mean_nm": summary.get("rg_carbon_mean_nm", ""),
        "rg_carbon_std_time_nm": summary.get("rg_carbon_std_time_nm", ""),
        "rg2_carbon_mean_nm2": summary.get("rg2_carbon_mean_nm2", ""),
        "rg2_carbon_std_time_nm2": summary.get("rg2_carbon_std_time_nm2", ""),
        "lambda1_mean_nm2": summary.get("lambda1_mean_nm2", ""),
        "lambda2_mean_nm2": summary.get("lambda2_mean_nm2", ""),
        "lambda3_mean_nm2": summary.get("lambda3_mean_nm2", ""),
        "asphericity_mean_nm2": summary.get("asphericity_mean_nm2", ""),
        "acylindricity_mean_nm2": summary.get("acylindricity_mean_nm2", ""),
        "relative_shape_anisotropy_mean": summary.get("relative_shape_anisotropy_mean", ""),
        "validation_n_bad_order": validation.get("n_bad_order", ""),
        "validation_n_bad_negative": validation.get("n_bad_negative", ""),
    }
    return row


def to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def maybe_plot(figures_dir, summary_rows, labels_root):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels_map = load_case_label_map(Path(labels_root))
    cases = [label_for_case(row["case_id"], labels_map) for row in summary_rows]
    outputs = []
    for field, ylabel, stem in [
        ("relative_shape_anisotropy_mean", "Relative shape anisotropy", "shape_tensor_relative_anisotropy_15cases"),
        ("graph_diameter_distance_mean_nm", "Graph diameter distance (nm)", "shape_tensor_graph_diameter_15cases"),
        ("graph_diameter2_over_6rg2", "Graph-diameter span^2 / 6Rg^2", "shape_tensor_graph_diameter2_over_6rg2_15cases"),
    ]:
        values = [to_float(row.get(field)) for row in summary_rows]
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.bar(range(len(cases)), values)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(cases, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " from strict carbon-skeleton tensor")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = figures_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=220 if suffix == "png" else None)
            outputs.append(str(path))
        plt.close(fig)
    return outputs


def main():
    args = parse_args()
    batch = Path(args.batch_dir)
    if args.output_dir is None:
        summary_dir = batch / "aggregate"
        raw_dir = batch / "aggregate"
        plots_dir = batch / "aggregate"
        labels_root = batch.parent
    else:
        output_root = Path(args.output_dir)
        summary_dir = output_root / "summary"
        raw_dir = output_root / "raw"
        plots_dir = output_root / "plots"
        labels_root = output_root.parent
    case_dirs = sorted((batch / "cases").glob("*/shape_tensor"))
    summary_rows = []
    timeseries_rows = []
    chain_rows = []
    for case_dir in case_dirs:
        summary_path = case_dir / "shape_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        case_id = summary.get("case_id") or case_dir.parent.name
        summary_rows.append(row_case_fields(summary))
        for row in read_csv(case_dir / "shape_timeseries.csv"):
            row["case_id"] = case_id
            timeseries_rows.append(row)
        for row in read_csv(case_dir / "shape_chains.csv"):
            row["case_id"] = case_id
            chain_rows.append(row)

    summary_rows.sort(key=lambda r: r.get("case_id", ""))
    timeseries_rows.sort(key=lambda r: (r.get("case_id", ""), float(r.get("time_ps", 0) or 0)))
    chain_rows.sort(key=lambda r: (r.get("case_id", ""), int(float(r.get("chain", 0) or 0))))

    summary_fields = [
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
        "atom_count",
        "atoms_per_chain",
        "carbon_count",
        "carbon_per_chain",
        "carbon_graph_connected",
        "graph_diameter_topological_edges",
        "graph_diameter_topological_carbons",
        "graph_diameter_distance_mean_nm",
        "graph_diameter_distance_std_time_nm",
        "graph_diameter_distance2_mean_nm2",
        "graph_diameter_distance2_std_time_nm2",
        "graph_diameter2_over_6rg2",
        "graph_diameter_local_atom_indices_0based",
        "graph_diameter_order_endpoints",
        "rg_carbon_mean_nm",
        "rg_carbon_std_time_nm",
        "rg2_carbon_mean_nm2",
        "rg2_carbon_std_time_nm2",
        "lambda1_mean_nm2",
        "lambda2_mean_nm2",
        "lambda3_mean_nm2",
        "asphericity_mean_nm2",
        "acylindricity_mean_nm2",
        "relative_shape_anisotropy_mean",
        "validation_n_bad_order",
        "validation_n_bad_negative",
    ]
    time_fields = [
        "case_id",
        "time_ps",
        "rg_carbon_mean_nm",
        "rg2_carbon_mean_nm2",
        "lambda1_mean_nm2",
        "lambda2_mean_nm2",
        "lambda3_mean_nm2",
        "asphericity_mean_nm2",
        "acylindricity_mean_nm2",
        "relative_shape_anisotropy_mean",
        "graph_diameter_distance_mean_nm",
        "graph_diameter_distance2_mean_nm2",
        "graph_diameter2_over_6rg2",
    ]
    chain_fields = [
        "case_id",
        "chain",
        "rg_carbon_mean_nm",
        "rg2_carbon_mean_nm2",
        "lambda1_mean_nm2",
        "lambda2_mean_nm2",
        "lambda3_mean_nm2",
        "asphericity_mean_nm2",
        "acylindricity_mean_nm2",
        "relative_shape_anisotropy_mean",
        "graph_diameter_distance_mean_nm",
        "graph_diameter_distance2_mean_nm2",
        "graph_diameter2_over_6rg2",
    ]

    write_csv(summary_dir / "shape_tensor_summary.csv", summary_rows, summary_fields)
    write_csv(raw_dir / "shape_tensor_timeseries_long.csv", timeseries_rows, time_fields)
    write_csv(raw_dir / "shape_tensor_chains.csv", chain_rows, chain_fields)
    plots = maybe_plot(plots_dir, summary_rows, labels_root)
    write_json(
        summary_dir / "shape_tensor_manifest.json",
        {
            "analysis_version": "canonical_v2",
            "case_count": len(summary_rows),
            "timeseries_rows": len(timeseries_rows),
            "chain_rows": len(chain_rows),
            "method": "strict carbon-skeleton gyration tensor from nojump trajectory coordinates",
            "plots": plots,
            "deprecated_replaces": "output/lcb100_shape_20260504",
        },
    )
    print(f"Aggregated strict shape tensor: {len(summary_rows)} cases")


if __name__ == "__main__":
    main()
