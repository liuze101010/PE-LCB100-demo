"""Dataset-level helpers for LCB100 single-sample analysis outputs."""

import csv
import json
import math
import re
from pathlib import Path


MANIFEST_FIELDS = [
    "architecture",
    "backbone_monomer",
    "backbone_repeat_units",
    "branch_points",
    "branch_carbons",
    "branch_topology",
    "branch_position_pattern",
    "actual_total_carbons_per_chain",
    "carbon_delta",
    "chain_count",
    "ml_feature_coord",
    "regime_tag",
]

SCALAR_TERMS = [
    "Temperature",
    "Pressure",
    "Volume",
    "Density",
    "Total-Energy",
    "Conserved-Energy",
]

SUMMARY_FIELDS = [
    "case_id",
    "eta_Pa_s",
    "eta_mPa_s",
    "sigma_eta_Pa_s",
    "sigma_eta_mPa_s",
    "r2_min",
    "r2_mean",
    "component_relative_std",
    "component_consistency_ok",
    "high_fit_quality",
    "low_fit_quality",
    "equilibrated",
    "temperature_K_for_viscosity",
    "volume_nm3_for_viscosity",
    "warnings",
    "trajectory_Rg_mean_nm",
    "trajectory_Rg_chain_mean_nm",
    "trajectory_MSD_final_nm2",
    "trajectory_D_mean_1e-5_cm2_s",
    "trajectory_D_mean_m2_s",
    "trajectory_Ree_mean_nm",
    "trajectory_Ree2_mean_nm2",
    "trajectory_RDF_peak_r_nm",
    "trajectory_RDF_peak_g_r",
] + [f"{term}_{stat}" for term in SCALAR_TERMS for stat in ("mean", "std", "min", "max")] + MANIFEST_FIELDS


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=float("nan")):
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_warnings(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [item for item in re.split(r"[;,]", str(value)) if item]


def case_sort_key(case_id):
    match = re.search(r"lcb(\d+)", str(case_id))
    if match:
        return int(match.group(1)), str(case_id)
    return 10**9, str(case_id)


def load_manifest_cases(manifest_json):
    if not manifest_json:
        return {}
    path = Path(manifest_json)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    payload = read_json(path)
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    return {str(case["case_id"]): case for case in cases if "case_id" in case}


def tukey_outlier_flags(values, iqr_coeff=1.5):
    """Return Tukey-fence outlier flags for finite values.

    This is a cross-topology screen for unusual LCB cases, not a replicate
    analysis. With fewer than four finite values, all flags are False.
    """
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 4:
        return [False for _ in values], {
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "n_finite": len(finite),
        }

    sorted_vals = sorted(finite)
    q1 = percentile(sorted_vals, 25.0)
    q3 = percentile(sorted_vals, 75.0)
    iqr = q3 - q1
    lower = q1 - iqr_coeff * iqr
    upper = q3 + iqr_coeff * iqr
    flags = [
        bool(math.isfinite(float(v)) and (float(v) < lower or float(v) > upper))
        for v in values
    ]
    return flags, {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "n_finite": len(finite),
    }


def percentile(sorted_values, pct):
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)
