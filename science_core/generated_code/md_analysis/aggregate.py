"""Aggregate LCB100 per-case analysis.json files into ML-ready tables.

This script is intentionally single-sample aware: each LCB topology is treated
as one case, not as a replicate of another case.
"""

import argparse
from pathlib import Path

try:
    from .canonical_outputs import batch_dir
    from .dataset_common import (
        MANIFEST_FIELDS,
        SCALAR_TERMS,
        SUMMARY_FIELDS,
        case_sort_key,
        load_manifest_cases,
        read_json,
        write_csv,
        write_json,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from canonical_outputs import batch_dir  # type: ignore
    from dataset_common import (  # type: ignore
        MANIFEST_FIELDS,
        SCALAR_TERMS,
        SUMMARY_FIELDS,
        case_sort_key,
        load_manifest_cases,
        read_json,
        write_csv,
        write_json,
    )


def parse_args():
    default_root = Path(__file__).resolve().parent / "output"
    default_manifest = (
        Path(__file__).resolve().parents[1]
        / "emc_alpha_olefin_builder"
        / "output"
        / "screen_lcb100"
        / "manifest.json"
    )
    parser = argparse.ArgumentParser(description="Aggregate md_analysis case outputs for LCB100.")
    parser.add_argument("--analysis-root", type=Path, default=default_root)
    parser.add_argument("--manifest-json", type=Path, default=default_manifest)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <analysis-root>/dataset",
    )
    return parser.parse_args()


def find_analysis_files(analysis_root):
    analysis_root = Path(analysis_root)
    files = []
    for path in analysis_root.glob("*/analysis.json"):
        if path.parent.name == "dataset":
            continue
        files.append(path)
    return sorted(files, key=lambda p: case_sort_key(p.parent.name))


def flatten_analysis(analysis, manifest_case=None, trajectory=None):
    case_id = str(analysis.get("case_id", ""))
    viscosity = analysis.get("viscosity", {}) or {}
    scalars = analysis.get("scalars", {}) or {}
    equilibration = analysis.get("equilibration", {}) or {}
    warnings = analysis.get("warnings", []) or []

    row = {
        "case_id": case_id,
        "eta_Pa_s": viscosity.get("eta_Pa_s", ""),
        "eta_mPa_s": _scale(viscosity.get("eta_Pa_s", ""), 1000.0),
        "sigma_eta_Pa_s": viscosity.get("sigma_eta_Pa_s", ""),
        "sigma_eta_mPa_s": _scale(viscosity.get("sigma_eta_Pa_s", ""), 1000.0),
        "r2_min": viscosity.get("r2_min", ""),
        "r2_mean": viscosity.get("r2_mean", ""),
        "component_relative_std": viscosity.get("component_relative_std", ""),
        "component_consistency_ok": viscosity.get("component_consistency_ok", ""),
        "high_fit_quality": viscosity.get("high_fit_quality", ""),
        "low_fit_quality": viscosity.get("low_fit_quality", ""),
        "equilibrated": equilibration.get("equilibrated", ""),
        "temperature_K_for_viscosity": analysis.get("temperature_K_for_viscosity", ""),
        "volume_nm3_for_viscosity": analysis.get("volume_nm3_for_viscosity", ""),
        "warnings": ";".join(str(item) for item in warnings),
    }
    row.update(flatten_trajectory_summary(trajectory or {}))

    for term in SCALAR_TERMS:
        stats = scalars.get(term, {}) or {}
        for stat in ("mean", "std", "min", "max"):
            row[f"{term}_{stat}"] = stats.get(stat, "")

    manifest_case = manifest_case or {}
    for field in MANIFEST_FIELDS:
        value = manifest_case.get(field, "")
        if isinstance(value, (list, dict)):
            value = str(value)
        row[field] = value

    return row


def load_trajectory_summary(case_output_dir, analysis_root=None, case_id=None):
    path = Path(case_output_dir) / "trajectory" / "trajectory_analysis.json"
    legacy = {}
    if not path.exists():
        legacy = {}
    else:
        try:
            legacy = read_json(path)
        except Exception:
            legacy = {}
    if analysis_root is not None and case_id:
        canonical = load_canonical_trajectory_summary(analysis_root, case_id)
        if canonical:
            merged = legacy if isinstance(legacy, dict) else {}
            results = dict(merged.get("results", {}) or {})
            results.update(canonical)
            merged["case_id"] = case_id
            merged["results"] = results
            merged["analysis_version"] = "canonical_v2"
            return merged
    return legacy


def load_canonical_trajectory_summary(analysis_root, case_id):
    analysis_root = Path(analysis_root)
    search_roots = [analysis_root]
    if analysis_root.name.startswith("lcb100_") and analysis_root.parent not in search_roots:
        search_roots.append(analysis_root.parent)
    specs = [
        ("rg", "rg", "rg_summary.json"),
        ("msd", "msd", "msd_summary.json"),
        ("ree", "ree", "ree_summary.json"),
        ("rdf", "rdf", "rdf_summary.json"),
    ]
    results = {}
    for batch_name, obs_name, filename in specs:
        matches = []
        for root in search_roots:
            matches.extend(root.glob(f"lcb100_{batch_name}_*/cases/{case_id}/{obs_name}/{filename}"))
        matches = sorted(matches, reverse=True)
        if not matches:
            continue
        try:
            key = "rdf" if batch_name == "interrdf" and "rdf" not in results else batch_name
            results[key] = read_json(matches[0])
        except Exception:
            continue
    return results


def flatten_trajectory_summary(trajectory):
    results = trajectory.get("results", {}) if isinstance(trajectory, dict) else {}
    rg = results.get("rg", {}) or {}
    msd = results.get("msd", {}) or {}
    ree = results.get("ree", {}) or {}
    rdf = results.get("rdf", {}) or {}
    rdf_entry = _rdf_entry(rdf)
    return {
        "trajectory_Rg_mean_nm": rg.get("rg_mean_nm", _nested(rg, "mean_Rg_nm", "mean")),
        "trajectory_Rg_chain_mean_nm": rg.get("rg_mean_nm", _nested(rg, "chain_mean_Rg_nm", "mean")),
        "trajectory_MSD_final_nm2": msd.get("msd_final_nm2", msd.get("mean_final_MSD_nm2", msd.get("final_MSD_nm2", ""))),
        "trajectory_D_mean_1e-5_cm2_s": msd.get(
            "D_mean_1e-5_cm2_s",
            _nested(msd.get("diffusion_per_molecule", {}) or {}, "D_1e-5_cm2_s", "mean"),
        ),
        "trajectory_D_mean_m2_s": msd.get(
            "D_mean_m2_s",
            _nested(msd.get("diffusion_per_molecule", {}) or {}, "D_m2_s", "mean"),
        ),
        "trajectory_Ree_mean_nm": ree.get("ree_mean_nm", _nested(ree, "mean_Ree_nm", "mean")),
        "trajectory_Ree2_mean_nm2": ree.get("ree2_mean_nm2", _nested(ree, "mean_Ree2_nm2", "mean")),
        "trajectory_RDF_peak_r_nm": rdf_entry.get("first_peak_r_nm", rdf.get("peak_r_nm", "")),
        "trajectory_RDF_peak_g_r": rdf_entry.get("first_peak_g", rdf.get("peak_g_r", "")),
    }


def _nested(payload, key, subkey):
    value = payload.get(key, {}) if isinstance(payload, dict) else {}
    if isinstance(value, dict):
        return value.get(subkey, "")
    return ""


def _rdf_entry(rdf):
    entries = rdf.get("entries", []) if isinstance(rdf, dict) else []
    if not isinstance(entries, list) or not entries:
        return rdf if isinstance(rdf, dict) else {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("rdf_kind") == "all":
            return entry
    return entries[0] if isinstance(entries[0], dict) else {}


def _scale(value, factor):
    try:
        return float(value) * factor
    except (TypeError, ValueError):
        return ""


def aggregate(analysis_root, manifest_json=None, output_dir=None):
    analysis_root = Path(analysis_root)
    output_dir = Path(output_dir) if output_dir else analysis_root / "dataset"
    manifest_cases = load_manifest_cases(manifest_json) if manifest_json else {}

    rows = []
    skipped = []
    for analysis_path in find_analysis_files(analysis_root):
        try:
            analysis = read_json(analysis_path)
        except Exception as exc:  # pragma: no cover - defensive path
            skipped.append({"path": str(analysis_path), "reason": str(exc)})
            continue
        case_id = str(analysis.get("case_id") or analysis_path.parent.name)
        trajectory = load_trajectory_summary(analysis_path.parent, analysis_root=analysis_root, case_id=case_id)
        rows.append(flatten_analysis(analysis, manifest_cases.get(case_id), trajectory=trajectory))

    rows = sorted(rows, key=lambda row: case_sort_key(row.get("case_id", "")))
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    write_csv(summary_csv, rows, SUMMARY_FIELDS)
    write_json(
        summary_json,
        {
            "analysis_root": str(analysis_root.resolve()),
            "manifest_json": str(Path(manifest_json).resolve()) if manifest_json else "",
            "case_count": len(rows),
            "skipped": skipped,
            "fields": SUMMARY_FIELDS,
            "rows": rows,
        },
    )
    return {
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "case_count": len(rows),
        "skipped": skipped,
    }


def main():
    args = parse_args()
    result = aggregate(args.analysis_root, args.manifest_json, args.output_dir)
    print(f"Wrote {result['summary_csv']} ({result['case_count']} cases)")
    print(f"Wrote {result['summary_json']}")
    if result["skipped"]:
        print(f"Skipped {len(result['skipped'])} malformed analysis file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
