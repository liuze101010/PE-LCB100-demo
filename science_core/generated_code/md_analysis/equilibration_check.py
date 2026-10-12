"""NVT-aware equilibrium diagnostics for production MD energy traces."""

import math

import numpy as np


def _finite_array(name, values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < 3:
        raise ValueError(f"{name} must contain at least 3 samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _linear_slope(time_ps, values):
    time_ps = _finite_array("time_ps", time_ps)
    values = _finite_array("values", values)
    if time_ps.size != values.size:
        raise ValueError("time and values must have the same length")
    slope, intercept = np.polyfit(time_ps, values, 1)
    return float(slope), float(intercept)


def check_temperature(time_ps, temperature_K, target_K=400.0, max_rel_offset=0.01, max_drift_fraction=0.01):
    temperature_K = _finite_array("temperature_K", temperature_K)
    slope, _ = _linear_slope(time_ps, temperature_K)
    span_ps = float(np.max(time_ps) - np.min(time_ps))
    mean_T = float(np.mean(temperature_K))
    rel_offset = abs(mean_T - target_K) / target_K
    drift_fraction = abs(slope * span_ps) / target_K
    return {
        "passed": bool(rel_offset < max_rel_offset and drift_fraction < max_drift_fraction),
        "mean_K": mean_T,
        "relative_offset": rel_offset,
        "drift_fraction": drift_fraction,
        "slope_K_per_ps": float(slope),
    }


def check_conserved_energy_drift(time_ps, conserved_energy, max_drift_fraction=0.01):
    conserved_energy = _finite_array("conserved_energy", conserved_energy)
    slope, _ = _linear_slope(time_ps, conserved_energy)
    span_ps = float(np.max(time_ps) - np.min(time_ps))
    mean_abs = float(np.mean(np.abs(conserved_energy)))
    drift_fraction = float("inf") if mean_abs == 0.0 else abs(slope * span_ps) / mean_abs
    return {
        "passed": bool(drift_fraction < max_drift_fraction),
        "mean": float(np.mean(conserved_energy)),
        "drift_fraction": drift_fraction,
        "slope_per_ps": float(slope),
    }


def check_energy_drift(time_ps, total_energy, max_drift_fraction=0.01):
    """Backward-compatible alias for non-thermostatted traces."""
    return check_conserved_energy_drift(time_ps, total_energy, max_drift_fraction=max_drift_fraction)


def check_stress_mean_blocks(stress_components_bar, component_names, block_count=20):
    stress = np.asarray(stress_components_bar, dtype=float)
    if stress.ndim == 1:
        stress = stress.reshape((-1, 1))
    if stress.ndim != 2:
        raise ValueError("stress_components_bar must be one- or two-dimensional")
    if len(component_names) != stress.shape[1]:
        raise ValueError("component_names length must match stress component count")
    if stress.shape[0] < block_count:
        block_count = max(2, stress.shape[0] // 2)

    component_results = {}
    all_passed = True
    for idx, name in enumerate(component_names):
        values = stress[:, idx]
        blocks = np.array_split(values, block_count)
        block_means = np.asarray([float(np.mean(block)) for block in blocks if len(block) > 0])
        mean_value = float(np.mean(values))
        if block_means.size < 2:
            sem = float("inf")
        else:
            sem = float(np.std(block_means, ddof=1) / math.sqrt(block_means.size))
        passed = bool(abs(mean_value) <= 2.0 * sem) if sem > 0 else bool(abs(mean_value) < 1e-12)
        all_passed = all_passed and passed
        component_results[name] = {
            "passed": passed,
            "mean_bar": mean_value,
            "block_sem_bar": sem,
            "n_blocks": int(block_means.size),
            "threshold_bar": float(2.0 * sem),
        }

    return {"passed": bool(all_passed), "components": component_results}


def check_eta_quality(viscosity_result, min_r2=0.95):
    eta = float(viscosity_result.get("eta_Pa_s", float("nan")))
    r2_min = float(viscosity_result.get("r2_min", float("nan")))
    rel_std = float(viscosity_result.get("component_relative_std", float("inf")))
    passed = bool(np.isfinite(eta) and eta > 0.0 and r2_min >= min_r2 and rel_std < 0.30)
    return {
        "passed": passed,
        "eta_positive": bool(np.isfinite(eta) and eta > 0.0),
        "r2_min": r2_min,
        "component_relative_std": rel_std,
        "high_fit_quality": bool(r2_min >= 0.99),
        "low_fit_quality": bool(r2_min < min_r2),
    }


def run_equilibration_checks(time_ps, scalars, stress_components_bar, component_names, viscosity_result, target_K=400.0):
    checks = {
        "temperature": check_temperature(time_ps, scalars["Temperature"], target_K=target_K),
        "conserved_energy": check_conserved_energy_drift(time_ps, scalars["Conserved-Energy"]),
        "stress_mean": check_stress_mean_blocks(stress_components_bar, component_names),
        "eta_linear": check_eta_quality(viscosity_result),
    }
    equilibrated = all(check["passed"] for check in checks.values())
    return {"equilibrated": bool(equilibrated), "checks": checks}
