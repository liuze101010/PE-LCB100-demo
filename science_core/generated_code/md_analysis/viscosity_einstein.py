"""Einstein-Helfand shear viscosity from off-diagonal pressure tensors."""

import math

import numpy as np

try:
    from .units import (
        BAR_TO_PA,
        PS_TO_S,
        viscosity_from_helfand_slope_pa2s,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from units import (  # type: ignore
        BAR_TO_PA,
        PS_TO_S,
        viscosity_from_helfand_slope_pa2s,
    )


DEFAULT_COMPONENTS = ("Pres-XY", "Pres-XZ", "Pres-YZ")


def _as_1d_float(name, values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < 3:
        raise ValueError(f"{name} must contain at least 3 samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_2d_stress(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape((-1, 1))
    if arr.ndim != 2:
        raise ValueError("stress_bar_components must be one- or two-dimensional")
    if arr.shape[1] < 1:
        raise ValueError("stress_bar_components must contain at least one component")
    if not np.all(np.isfinite(arr)):
        raise ValueError("stress_bar_components contains non-finite values")
    return arr


def discard_initial(time_ps, stress_bar, fraction):
    if fraction < 0 or fraction >= 1:
        raise ValueError("discard fraction must be in [0, 1)")
    n_drop = int(math.floor(len(time_ps) * fraction))
    if len(time_ps) - n_drop < 3:
        raise ValueError("discard fraction leaves fewer than 3 frames")
    return time_ps[n_drop:], stress_bar[n_drop:]


def cumulative_trapezoid(time_s, values_pa):
    h = np.zeros_like(values_pa, dtype=float)
    dt = np.diff(time_s)
    increments = 0.5 * (values_pa[1:] + values_pa[:-1]) * dt
    h[1:] = np.cumsum(increments)
    return h


def choose_lags(n_samples, max_lag_fraction=0.25, max_lags=400):
    max_lag = max(1, int(math.floor((n_samples - 1) * max_lag_fraction)))
    if max_lag <= max_lags:
        return np.arange(1, max_lag + 1, dtype=int)
    raw = np.geomspace(1, max_lag, num=max_lags)
    return np.unique(np.rint(raw).astype(int))


def compute_helfand_msd(helfand, lags):
    msd = np.empty(len(lags), dtype=float)
    counts = np.empty(len(lags), dtype=int)
    for idx, lag in enumerate(lags):
        diffs = helfand[lag:] - helfand[:-lag]
        msd[idx] = float(np.mean(diffs * diffs))
        counts[idx] = int(diffs.size)
    return msd, counts


def linear_fit(x, y):
    x = _as_1d_float("x", x)
    y = _as_1d_float("y", y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        raise ValueError("at least two points are required for a linear fit")
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_points": int(x.size),
    }


def _fit_mask(lag_times_ps, fit_min_ps, fit_max_ps):
    if fit_min_ps is None:
        fit_min_ps = max(100.0, 5.0 * float(lag_times_ps[0]))
    if fit_max_ps is None:
        fit_max_ps = min(float(lag_times_ps[-1]), max(float(lag_times_ps[-1]) * 0.4, fit_min_ps))
    mask = (lag_times_ps >= fit_min_ps) & (lag_times_ps <= fit_max_ps)
    if int(np.sum(mask)) >= 3:
        return mask

    lo = np.percentile(lag_times_ps, 20)
    hi = np.percentile(lag_times_ps, 70)
    mask = (lag_times_ps >= lo) & (lag_times_ps <= hi)
    if int(np.sum(mask)) < 3:
        raise ValueError("not enough lag points for fitting")
    return mask


def component_viscosity(
    time_ps,
    stress_bar,
    volume_nm3,
    temperature_K,
    fit_min_ps=None,
    fit_max_ps=None,
    max_lag_fraction=0.25,
    max_msd_lags=400,
):
    time_ps = _as_1d_float("time_ps", time_ps)
    stress_bar = _as_1d_float("stress_bar", stress_bar)
    if time_ps.size != stress_bar.size:
        raise ValueError("time_ps and stress_bar must have the same length")
    if not np.all(np.diff(time_ps) > 0):
        raise ValueError("time_ps must be strictly increasing")

    stress_mean_bar = float(np.mean(stress_bar))
    stress_pa = (stress_bar - stress_mean_bar) * BAR_TO_PA
    time_s = time_ps * PS_TO_S
    helfand = cumulative_trapezoid(time_s, stress_pa)

    lags = choose_lags(time_ps.size, max_lag_fraction=max_lag_fraction, max_lags=max_msd_lags)
    lag_times_ps = time_ps[lags] - time_ps[0]
    lag_times_s = lag_times_ps * PS_TO_S
    msd, counts = compute_helfand_msd(helfand, lags)

    mask = _fit_mask(lag_times_ps, fit_min_ps, fit_max_ps)
    fit = linear_fit(lag_times_s[mask], msd[mask])
    eta = viscosity_from_helfand_slope_pa2s(fit["slope"], volume_nm3, temperature_K)
    loglog_slope = linear_fit(np.log(lag_times_s[mask]), np.log(np.maximum(msd[mask], 1e-300)))["slope"]

    quality = "low"
    if fit["r2"] >= 0.99:
        quality = "high"
    elif fit["r2"] >= 0.95:
        quality = "acceptable"

    return {
        "eta_Pa_s": float(eta),
        "slope_pa2_s": float(fit["slope"]),
        "intercept_pa2_s2": float(fit["intercept"]),
        "r2": float(fit["r2"]),
        "loglog_slope": float(loglog_slope),
        "fit_window_ps": [float(lag_times_ps[mask][0]), float(lag_times_ps[mask][-1])],
        "fit_points": int(fit["n_points"]),
        "stress_mean_bar": stress_mean_bar,
        "n_lags": int(len(lags)),
        "lag_count_min": int(np.min(counts)),
        "quality": quality,
        "high_fit_quality": bool(fit["r2"] >= 0.99),
        "low_fit_quality": bool(fit["r2"] < 0.95),
    }


def compute_viscosity(
    time_ps,
    stress_bar_components,
    volume_nm3,
    temperature_K,
    component_names=DEFAULT_COMPONENTS,
    discard_fraction=0.1,
    fit_min_ps=None,
    fit_max_ps=None,
    max_lag_fraction=0.25,
    max_msd_lags=400,
):
    time_ps = _as_1d_float("time_ps", time_ps)
    stress_bar = _as_2d_stress(stress_bar_components)
    if time_ps.size != stress_bar.shape[0]:
        raise ValueError("time_ps and stress_bar_components must have matching row counts")
    if len(component_names) != stress_bar.shape[1]:
        raise ValueError("component_names length must match stress component count")

    work_time_ps, work_stress_bar = discard_initial(time_ps, stress_bar, discard_fraction)
    components = {}
    etas = []
    r2_values = []
    for idx, name in enumerate(component_names):
        result = component_viscosity(
            work_time_ps,
            work_stress_bar[:, idx],
            volume_nm3,
            temperature_K,
            fit_min_ps=fit_min_ps,
            fit_max_ps=fit_max_ps,
            max_lag_fraction=max_lag_fraction,
            max_msd_lags=max_msd_lags,
        )
        components[name] = result
        etas.append(result["eta_Pa_s"])
        r2_values.append(result["r2"])

    etas_arr = np.asarray(etas, dtype=float)
    eta_mean = float(np.mean(etas_arr))
    eta_std = float(np.std(etas_arr, ddof=1)) if etas_arr.size > 1 else 0.0
    rel_std = float(eta_std / abs(eta_mean)) if eta_mean != 0.0 else float("inf")
    r2_min = float(np.min(r2_values))

    return {
        "method": "Einstein-Helfand multi-origin off-diagonal stress",
        "eta_Pa_s": eta_mean,
        "sigma_eta_Pa_s": eta_std,
        "component_relative_std": rel_std,
        "component_consistency_ok": bool(eta_mean > 0.0 and rel_std < 0.30),
        "r2_min": r2_min,
        "r2_mean": float(np.mean(r2_values)),
        "high_fit_quality": bool(r2_min >= 0.99),
        "low_fit_quality": bool(r2_min < 0.95),
        "discard_fraction": float(discard_fraction),
        "volume_nm3": float(volume_nm3),
        "temperature_K": float(temperature_K),
        "components": components,
    }

