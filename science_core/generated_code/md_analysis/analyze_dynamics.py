"""Supplemental dynamics analysis for completed LCB100 trajectories.

This script adds dynamics observables that are not produced by the structural
trajectory pipeline:

* local MSD log-log slope alpha(t)
* end-to-end vector autocorrelation C_Ree(t)
* carbon self-intermediate scattering function Fs(q,t)

It is designed to run on the old CPU server with staged nvt_prod trajectories,
but it can also be run locally to aggregate previously generated per-case
outputs.
"""

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional in headless smoke tests
    plt = None

try:
    from .canonical_outputs import ANALYSIS_VERSION, aggregate_dir, batch_dir, case_dir, default_output_root
    from .case_labels import label_for_case, load_case_label_map
    from .trajectory_helpers import (
        parse_xvg_numeric,
        resolve_trajectory,
        run_command,
        time_metadata,
        write_csv,
        write_json,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from canonical_outputs import ANALYSIS_VERSION, aggregate_dir, batch_dir, case_dir, default_output_root  # type: ignore
    from case_labels import label_for_case, load_case_label_map  # type: ignore
    from trajectory_helpers import (  # type: ignore
        parse_xvg_numeric,
        resolve_trajectory,
        run_command,
        time_metadata,
        write_csv,
        write_json,
    )


DEFAULT_CASES = [
    "lcb02_star3_b1_l08_c101",
    "lcb17_htype_b2_l20_c100",
    "lcb18_dbmid_b2_l08_c100",
    "lcb23_comb_b3_l16_c099",
    "lcb26_pompom_b3_l16_c099",
    "lcb62_tri_b3_l08_c099",
    "lcb66_q4comb_b4_l05_c101",
    "lcb67_q4comb_b4_l08_c101",
    "lcb68_q4comb_b4_l12_c099",
    "lcb69_q4pompom_b4_l03_c099",
    "lcb70_q4pompom_b4_l05_c101",
    "lcb71_q4pompom_b4_l08_c101",
    "lcb72_q4pompom_b4_l12_c099",
    "lcb73_q4clust_b4_l03_c099",
    "lcb74_q4clust_b4_l05_c101",
]

DYNAMICS_SUMMARY_FIELDS = [
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run supplemental dynamics analysis for LCB100 cases.")
    parser.add_argument("--input-root", type=Path, help="Old-server staged input root containing case/nvt_prod dirs.")
    parser.add_argument("--output-root", type=Path, default=default_output_root(), help="md_analysis output root.")
    parser.add_argument("--batch-date", default="20260504", help="Output date tag.")
    parser.add_argument("--cases", nargs="*", default=None, help="Explicit case_id list. Default: current 15 cases.")
    parser.add_argument("--cases-file", type=Path, help="Optional file with one case_id per line.")
    parser.add_argument("--gmx-bin", default="gmx", help="GROMACS executable.")
    parser.add_argument("--msd-batch", default="lcb100_msd_20260501", help="Existing MSD batch directory name.")
    parser.add_argument("--rdf-batch", default="lcb100_rdf_20260501", help="Existing RDF batch directory name.")
    parser.add_argument("--rdf-summary", type=Path, help="Explicit RDF summary CSV with first_peak_r_nm for Fs(q,t).")
    parser.add_argument("--coord-dt-ps", type=float, default=100.0, help="Coordinate sampling interval for Ree/Fs.")
    parser.add_argument("--max-corr-lag-ps", type=float, default=20000.0, help="Maximum lag time for Ree/Fs correlations.")
    parser.add_argument("--origin-stride", type=int, default=5, help="Time-origin stride for autocorrelation/Fs.")
    parser.add_argument("--max-fs-atoms", type=int, default=3000, help="Deterministic carbon-atom subsample for Fs; <=0 uses all.")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep large temporary XTC/XVG coordinate files.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing coordinate-derived outputs.")
    parser.add_argument("--skip-coordinates", action="store_true", help="Only compute alpha/aggregate from existing outputs.")
    parser.add_argument("--skip-msd-alpha", action="store_true", help="Only compute coordinate-derived outputs.")
    parser.add_argument("--fs-only", action="store_true", help="Only recompute Fs(q,t), preserving existing MSD/Ree outputs.")
    parser.add_argument(
        "--fs-q-nm-inv",
        type=float,
        default=None,
        help="Fixed q for Fs(q,t); debug only. Production runs should NOT set this flag.",
    )
    parser.add_argument(
        "--fs-q-source",
        default="debug fixed q override",
        help="Description recorded when --fs-q-nm-inv is used.",
    )
    return parser.parse_args()


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value, default=math.nan):
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def case_id_from_dirname(name):
    match = re.search(r"((?:lcb|sc)\d+_[A-Za-z0-9_]+?_c\d{3})(?=$|_)", str(name))
    return match.group(1) if match else str(name)


def load_case_ids(args):
    if args.cases:
        return list(args.cases)
    if args.cases_file:
        return [
            line.strip()
            for line in Path(args.cases_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return list(DEFAULT_CASES)


def find_case_input_dir(input_root, case_id):
    if not input_root:
        return None
    input_root = Path(input_root)
    exact = input_root / case_id
    if (exact / "nvt_prod").exists():
        return exact
    candidates = sorted(input_root.glob(f"{case_id}*"))
    for candidate in candidates:
        if (candidate / "nvt_prod" / "nvt_prod.tpr").exists():
            return candidate
    return None


def load_q_map(rdf_summary_path):
    q_map = {}
    for row in read_csv_rows(rdf_summary_path):
        if row.get("rdf_kind") != "inter_chain_cc":
            continue
        peak = safe_float(row.get("first_peak_r_nm"))
        if math.isfinite(peak) and peak > 0:
            q_map[row.get("case_id")] = {
                "q_nm_inv": 2.0 * math.pi / peak,
                "q_source": f"rdf_kind=inter_chain_cc first_peak_r_nm={peak:g}",
            }
    return q_map


def missing_q_info(case_id, reason="MISSING_INTER_RDF_PEAK"):
    return {
        "case_id": case_id,
        "q_nm_inv": None,
        "q_source": reason,
        "q_missing": True,
        "q_missing_reason": reason,
    }


def q_is_missing(q_info):
    return bool(q_info.get("q_missing")) or q_info.get("q_nm_inv") in (None, "")


def write_missing_fsqt_summary(case_id, dyn_case_dir, q_info, source_case_dir=""):
    dyn_case_dir = Path(dyn_case_dir)
    dyn_case_dir.mkdir(parents=True, exist_ok=True)
    write_csv(dyn_case_dir / "fsqt.csv", [], ["time_ps", "Fs_q_t", "n_origins", "n_atoms_sampled"])
    reason = q_info.get("q_missing_reason") or "MISSING_INTER_RDF_PEAK"
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "case_id": case_id,
        "status": reason,
        "method": "carbon self-intermediate scattering function skipped because RDF-derived q is missing",
        "n_chains": "",
        "n_frames": 0,
        "time_first_ps": "",
        "time_last_ps": "",
        "dt_ps": "",
        "source_case_dir": str(source_case_dir) if source_case_dir else "",
        "q_nm_inv": None,
        "q_source": reason,
        "fs_qt_values": None,
        "tau_fs_ps": None,
        "tau_fs_censored": True,
        "tau_alpha_ps": None,
        "tau_alpha_censored": True,
    }
    write_json(dyn_case_dir / "fsqt_summary.json", summary)
    return summary, []


def load_source_map(output_root):
    source_map = {}
    rg_summary = Path(output_root) / "lcb100_rg_20260501" / "aggregate" / "rg_summary.csv"
    for row in read_csv_rows(rg_summary):
        case_id = row.get("case_id")
        source = row.get("source_case_dir") or ""
        if source.endswith("/nvt_prod") or source.endswith("\\nvt_prod"):
            source = re.sub(r"[\\/]+nvt_prod$", "", source)
        if case_id and source:
            source_map[case_id] = source
    return source_map


def compute_local_msd_alpha(times_ps, msd_nm2, half_window=3, min_time_ps=1000.0):
    times = np.asarray(times_ps, dtype=float)
    msd = np.asarray(msd_nm2, dtype=float)
    valid = np.isfinite(times) & np.isfinite(msd) & (times >= min_time_ps) & (times > 0) & (msd > 0)
    out = []
    indices = np.where(valid)[0]
    for idx in indices:
        left = max(0, idx - int(half_window))
        right = min(times.size, idx + int(half_window) + 1)
        mask = valid[left:right]
        local_indices = np.arange(left, right)[mask]
        if local_indices.size < 3:
            alpha = math.nan
            r2 = math.nan
        else:
            x = np.log10(times[local_indices])
            y = np.log10(msd[local_indices])
            design = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
            predicted = slope * x + intercept
            ss_res = float(np.sum((y - predicted) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            alpha = float(slope)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
        out.append({"time_ps": float(times[idx]), "alpha": alpha, "fit_r2": r2})
    return out


def summarize_alpha(alpha_rows, begin_ps=100000.0, end_ps=300000.0):
    values = [
        safe_float(row.get("alpha"))
        for row in alpha_rows
        if begin_ps <= safe_float(row.get("time_ps")) <= end_ps
    ]
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if arr.size == 0:
        return math.nan, "missing"
    mean = float(np.mean(arr))
    if mean < 0.95:
        return mean, "sub-diffusive"
    if mean > 1.05:
        return mean, "super-diffusive"
    return mean, "ok"


def compute_case_msd_alpha(case_id, msd_batch_root, dyn_case_dir, force=False):
    msd_xvg = Path(msd_batch_root) / "cases" / case_id / "msd" / "msd.xvg"
    if not msd_xvg.exists():
        return None, []
    out_csv = dyn_case_dir / "msd_local_slope.csv"
    if out_csv.exists() and not force:
        rows = read_csv_rows(out_csv)
    else:
        data = parse_xvg_numeric(msd_xvg)
        rows = compute_local_msd_alpha(data[:, 0], data[:, 1])
        write_csv(out_csv, rows, ["time_ps", "alpha", "fit_r2"])
    alpha_mean, alpha_status = summarize_alpha(rows)
    return {"alpha_100_300_mean": alpha_mean, "alpha_100_300_status": alpha_status}, rows


def dump_tpr_topology(gmx_bin, tpr, out_path, force=False):
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return out_path
    result = subprocess.run(
        [str(gmx_bin), "dump", "-s", str(tpr)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gmx dump failed for {tpr}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.stdout, encoding="utf-8")
    return out_path


def parse_tpr_dump_for_indices(dump_path):
    text = Path(dump_path).read_text(encoding="utf-8", errors="ignore")
    natoms_match = re.search(r"natoms\s*=\s*(\d+)", text)
    nmol_match = re.search(r"#molecules\s*=\s*(\d+)", text)
    atom_count_match = re.search(r"atom \((\d+)\):", text)
    if not (natoms_match and nmol_match and atom_count_match):
        raise ValueError(f"Could not parse natoms/#molecules/atom count from {dump_path}")
    natoms = int(natoms_match.group(1))
    n_chains = int(nmol_match.group(1))
    atoms_per_chain = int(atom_count_match.group(1))
    atom_numbers = {}
    atom_pattern = re.compile(r"atom\[\s*(\d+)\]=\{.*?atomnumber=\s*(\d+)")
    for match in atom_pattern.finditer(text):
        idx = int(match.group(1))
        if idx >= atoms_per_chain:
            continue
        if idx not in atom_numbers:
            atom_numbers[idx] = int(match.group(2))
        if len(atom_numbers) >= atoms_per_chain:
            break
    if len(atom_numbers) != atoms_per_chain:
        raise ValueError(f"Parsed only {len(atom_numbers)} atoms for molecule size {atoms_per_chain} from {dump_path}")
    carbon_local = [idx for idx in range(atoms_per_chain) if atom_numbers[idx] == 6]
    heavy_local = [idx for idx in range(atoms_per_chain) if atom_numbers[idx] != 1]
    if len(heavy_local) < 2:
        raise ValueError(f"Fewer than two heavy atoms in molecule topology from {dump_path}")
    all_atoms = list(range(1, natoms + 1))
    carbon_atoms = []
    endpoint_atoms = []
    first_heavy = heavy_local[0]
    last_heavy = heavy_local[-1]
    for chain_idx in range(n_chains):
        base = chain_idx * atoms_per_chain
        carbon_atoms.extend(base + idx + 1 for idx in carbon_local)
        endpoint_atoms.extend([base + first_heavy + 1, base + last_heavy + 1])
    return {
        "natoms": natoms,
        "n_chains": n_chains,
        "atoms_per_chain": atoms_per_chain,
        "carbon_per_chain": len(carbon_local),
        "all_atoms": all_atoms,
        "carbon_atoms": carbon_atoms,
        "endpoint_atoms": endpoint_atoms,
        "endpoint_local_indices_1based": [first_heavy + 1, last_heavy + 1],
    }


def select_evenly(values, max_count):
    values = list(values)
    if not max_count or max_count <= 0 or len(values) <= max_count:
        return values
    indices = np.linspace(0, len(values) - 1, int(max_count), dtype=int)
    return [values[int(idx)] for idx in indices]


def write_dynamic_index(path, index_info, max_fs_atoms=3000):
    fs_carbon_atoms = select_evenly(index_info["carbon_atoms"], max_fs_atoms)
    groups = [
        ("chain_all", index_info["all_atoms"]),
        ("carbon_atoms", index_info["carbon_atoms"]),
        ("fs_carbon_atoms", fs_carbon_atoms),
        ("ree_endpoints", index_info["endpoint_atoms"]),
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, atom_numbers in groups:
            handle.write(f"[ {name} ]\n")
            for idx in range(0, len(atom_numbers), 15):
                handle.write(" ".join(str(num) for num in atom_numbers[idx:idx + 15]) + "\n")
            handle.write("\n")
    return path


def prepare_nocom_trajectory(gmx_bin, traj, tpr, index_path, dyn_case_dir, force=False):
    nojump = dyn_case_dir / "nvt_prod_nojump.xtc"
    nocom = dyn_case_dir / "nvt_prod_nocom.xtc"
    if force or not nojump.exists() or nojump.stat().st_size == 0:
        run_command(
            [gmx_bin, "trjconv", "-f", traj, "-s", tpr, "-n", index_path, "-o", nojump, "-pbc", "nojump"],
            input_text="chain_all\n",
        )
    if force or not nocom.exists() or nocom.stat().st_size == 0:
        run_command(
            [gmx_bin, "trjconv", "-f", nojump, "-s", tpr, "-n", index_path, "-o", nocom, "-fit", "translation"],
            input_text="chain_all\nchain_all\n",
        )
    return nocom


def extract_coords_xvg(gmx_bin, traj, tpr, index_path, group_name, output_path, dt_ps, force=False):
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return output_path
    cmd = [
        gmx_bin,
        "traj",
        "-f",
        traj,
        "-s",
        tpr,
        "-n",
        index_path,
        "-ox",
        output_path,
        "-xvg",
        "none",
    ]
    if dt_ps:
        cmd.extend(["-dt", str(int(dt_ps) if float(dt_ps).is_integer() else dt_ps)])
    run_command(cmd, input_text=f"{group_name}\n")
    return output_path


def parse_coord_xvg(path, expected_atoms=None):
    data = parse_xvg_numeric(path)
    if data.shape[1] < 4 or (data.shape[1] - 1) % 3 != 0:
        raise ValueError(f"Unexpected coordinate XVG width in {path}: {data.shape}")
    n_atoms = (data.shape[1] - 1) // 3
    if expected_atoms is not None and int(expected_atoms) != n_atoms:
        raise ValueError(f"Coordinate atom count mismatch in {path}: expected {expected_atoms}, got {n_atoms}")
    coords = data[:, 1:].reshape((data.shape[0], n_atoms, 3))
    return data[:, 0], coords


def first_crossing_time(times, values, threshold=math.exp(-1.0)):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]
    if times.size == 0:
        return math.nan, True
    below = np.where(values <= threshold)[0]
    if below.size == 0:
        return math.nan, True
    idx = int(below[0])
    if idx == 0:
        return float(times[0]), False
    t0, t1 = times[idx - 1], times[idx]
    v0, v1 = values[idx - 1], values[idx]
    if v1 == v0:
        return float(t1), False
    frac = (threshold - v0) / (v1 - v0)
    return float(t0 + frac * (t1 - t0)), False


def normalized_vector_autocorrelation(vectors, times_ps, origin_stride=5, max_lag_ps=None):
    vectors = np.asarray(vectors, dtype=float)
    n_frames = vectors.shape[0]
    rows = []
    for lag in range(n_frames):
        lag_time = float(times_ps[lag] - times_ps[0])
        if max_lag_ps is not None and lag_time > float(max_lag_ps):
            break
        origins = np.arange(0, n_frames - lag, max(1, int(origin_stride)))
        if origins.size == 0:
            continue
        v0 = vectors[origins]
        vt = vectors[origins + lag]
        numerator = float(np.mean(np.sum(v0 * vt, axis=2)))
        denominator = float(np.mean(np.sum(v0 * v0, axis=2)))
        c_value = numerator / denominator if denominator else math.nan
        rows.append({
            "time_ps": lag_time,
            "C_Ree": c_value,
            "n_origins": int(origins.size),
        })
    return rows


def compute_fsqt(coords, times_ps, q_nm_inv, origin_stride=5, max_lag_ps=None):
    coords = np.asarray(coords, dtype=float)
    n_frames, n_atoms, _ = coords.shape
    sampled_atoms = int(n_atoms)
    rows = []
    for lag in range(n_frames):
        lag_time = float(times_ps[lag] - times_ps[0])
        if max_lag_ps is not None and lag_time > float(max_lag_ps):
            break
        origins = np.arange(0, n_frames - lag, max(1, int(origin_stride)))
        if origins.size == 0:
            continue
        diff = coords[origins + lag] - coords[origins]
        distance = np.linalg.norm(diff, axis=2)
        qr = float(q_nm_inv) * distance
        with np.errstate(divide="ignore", invalid="ignore"):
            fs_components = np.where(np.abs(qr) < 1.0e-12, 1.0, np.sin(qr) / qr)
        fs_value = float(np.mean(fs_components))
        rows.append({
            "time_ps": lag_time,
            "Fs_q_t": fs_value,
            "n_origins": int(origins.size),
            "n_atoms_sampled": sampled_atoms,
        })
    return rows


def run_fsqt_dynamics(
    case_id,
    case_input_dir,
    dyn_case_dir,
    gmx_bin,
    q_info,
    coord_dt_ps,
    origin_stride,
    max_fs_atoms,
    max_corr_lag_ps,
    force=False,
    keep_intermediates=False,
):
    if q_is_missing(q_info):
        return write_missing_fsqt_summary(case_id, dyn_case_dir, q_info, source_case_dir=case_input_dir)

    case_input_dir = Path(case_input_dir)
    nvt_dir = case_input_dir / "nvt_prod"
    tpr = nvt_dir / "nvt_prod.tpr"
    if not tpr.exists():
        raise FileNotFoundError(f"Missing nvt_prod.tpr for {case_id}: {tpr}")
    dump_path = dyn_case_dir / "tpr_topology.dump"
    dump_tpr_topology(gmx_bin, tpr, dump_path, force=force)
    index_info = parse_tpr_dump_for_indices(dump_path)
    index_path = write_dynamic_index(dyn_case_dir / "dynamics.ndx", index_info, max_fs_atoms=max_fs_atoms)
    traj = resolve_trajectory(gmx_bin, nvt_dir, dyn_case_dir, force=force)
    nocom = prepare_nocom_trajectory(gmx_bin, traj, tpr, index_path, dyn_case_dir, force=force)
    temporary_paths = [
        dyn_case_dir / "nvt_prod_nojump.xtc",
        dyn_case_dir / "nvt_prod_nocom.xtc",
        dyn_case_dir / "nvt_prod.full.xtc",
        dyn_case_dir / "carbon_coords.xvg",
    ]

    q_nm_inv = float(q_info["q_nm_inv"])
    carbon_xvg = extract_coords_xvg(
        gmx_bin,
        nocom,
        tpr,
        index_path,
        "fs_carbon_atoms",
        dyn_case_dir / "carbon_coords.xvg",
        coord_dt_ps,
        force=force,
    )
    expected_fs_atoms = len(select_evenly(index_info["carbon_atoms"], max_fs_atoms))
    carbon_times, carbon_coords = parse_coord_xvg(carbon_xvg, expected_atoms=expected_fs_atoms)
    fs_rows = compute_fsqt(
        carbon_coords,
        carbon_times,
        q_nm_inv,
        origin_stride=origin_stride,
        max_lag_ps=max_corr_lag_ps,
    )
    write_csv(dyn_case_dir / "fsqt.csv", fs_rows, ["time_ps", "Fs_q_t", "n_origins", "n_atoms_sampled"])
    tau_fs_ps, tau_fs_censored = first_crossing_time(
        [row["time_ps"] for row in fs_rows],
        [row["Fs_q_t"] for row in fs_rows],
    )
    tau_alpha_ps = tau_fs_ps
    tau_alpha_censored = tau_fs_censored
    fs_summary = {
        "analysis_version": ANALYSIS_VERSION,
        "case_id": case_id,
        "status": "ok",
        "method": "multi-time-origin isotropic carbon self-intermediate scattering function sin(qr)/(qr)",
        "n_chains": index_info["n_chains"],
        **time_metadata(carbon_times),
        "requested_coord_dt_ps": float(coord_dt_ps),
        "max_corr_lag_ps": float(max_corr_lag_ps) if max_corr_lag_ps is not None else None,
        "origin_stride": int(origin_stride),
        "carbon_atoms_total": len(index_info["carbon_atoms"]),
        "carbon_atoms_sampled": fs_rows[0]["n_atoms_sampled"] if fs_rows else 0,
        "q_nm_inv": q_nm_inv,
        "q_source": q_info.get("q_source", "rdf_kind=inter_chain_cc first_peak_r_nm"),
        "tau_fs_ps": tau_fs_ps,
        "tau_fs_censored": bool(tau_fs_censored),
        "tau_alpha_ps": tau_alpha_ps,
        "tau_alpha_censored": bool(tau_alpha_censored),
    }
    write_json(dyn_case_dir / "fsqt_summary.json", fs_summary)
    if not keep_intermediates:
        remove_intermediates(temporary_paths)

    return {
        "n_chains": index_info["n_chains"],
        "tau_fs_ps": tau_fs_ps,
        "tau_fs_censored": bool(tau_fs_censored),
        "tau_alpha_ps": tau_alpha_ps,
        "tau_alpha_censored": bool(tau_alpha_censored),
        "q_nm_inv": q_nm_inv,
        "q_source": q_info.get("q_source", "rdf_kind=inter_chain_cc first_peak_r_nm"),
        "source_case_dir": str(case_input_dir),
    }, fs_rows


def remove_intermediates(paths):
    for path in paths:
        path = Path(path)
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def run_coordinate_dynamics(
    case_id,
    case_input_dir,
    dyn_case_dir,
    gmx_bin,
    q_info,
    coord_dt_ps,
    origin_stride,
    max_fs_atoms,
    max_corr_lag_ps,
    force=False,
    keep_intermediates=False,
):
    case_input_dir = Path(case_input_dir)
    nvt_dir = case_input_dir / "nvt_prod"
    tpr = nvt_dir / "nvt_prod.tpr"
    if not tpr.exists():
        raise FileNotFoundError(f"Missing nvt_prod.tpr for {case_id}: {tpr}")
    dump_path = dyn_case_dir / "tpr_topology.dump"
    dump_tpr_topology(gmx_bin, tpr, dump_path, force=force)
    index_info = parse_tpr_dump_for_indices(dump_path)
    index_path = write_dynamic_index(dyn_case_dir / "dynamics.ndx", index_info, max_fs_atoms=max_fs_atoms)
    traj = resolve_trajectory(gmx_bin, nvt_dir, dyn_case_dir, force=force)
    nocom = prepare_nocom_trajectory(gmx_bin, traj, tpr, index_path, dyn_case_dir, force=force)
    temporary_paths = [
        dyn_case_dir / "nvt_prod_nojump.xtc",
        dyn_case_dir / "nvt_prod_nocom.xtc",
        dyn_case_dir / "nvt_prod.full.xtc",
        dyn_case_dir / "carbon_coords.xvg",
        dyn_case_dir / "ree_endpoint_coords.xvg",
    ]

    endpoint_xvg = extract_coords_xvg(
        gmx_bin,
        nocom,
        tpr,
        index_path,
        "ree_endpoints",
        dyn_case_dir / "ree_endpoint_coords.xvg",
        coord_dt_ps,
        force=force,
    )
    endpoint_times, endpoint_coords = parse_coord_xvg(endpoint_xvg, expected_atoms=index_info["n_chains"] * 2)
    vectors = endpoint_coords[:, 1::2, :] - endpoint_coords[:, 0::2, :]
    ree_rows = normalized_vector_autocorrelation(
        vectors,
        endpoint_times,
        origin_stride=origin_stride,
        max_lag_ps=max_corr_lag_ps,
    )
    write_csv(dyn_case_dir / "ree_vector_autocorr.csv", ree_rows, ["time_ps", "C_Ree", "n_origins"])
    tau_ree_ps, tau_ree_censored = first_crossing_time(
        [row["time_ps"] for row in ree_rows],
        [row["C_Ree"] for row in ree_rows],
    )
    ree_summary = {
        "analysis_version": ANALYSIS_VERSION,
        "case_id": case_id,
        "status": "ok",
        "method": "multi-time-origin terminal-heavy-atom Ree vector autocorrelation",
        "n_chains": index_info["n_chains"],
        **time_metadata(endpoint_times),
        "coord_dt_ps": float(coord_dt_ps),
        "max_corr_lag_ps": float(max_corr_lag_ps) if max_corr_lag_ps is not None else None,
        "origin_stride": int(origin_stride),
        "tau_ree_ps": tau_ree_ps,
        "tau_ree_censored": bool(tau_ree_censored),
        "endpoint_local_indices_1based": index_info["endpoint_local_indices_1based"],
    }
    write_json(dyn_case_dir / "ree_vector_autocorr_summary.json", ree_summary)

    if q_is_missing(q_info):
        fs_summary, fs_rows = write_missing_fsqt_summary(
            case_id,
            dyn_case_dir,
            q_info,
            source_case_dir=case_input_dir,
        )
        if not keep_intermediates:
            remove_intermediates(temporary_paths)
        return {
            "status": fs_summary.get("status", "MISSING_INTER_RDF_PEAK"),
            "n_chains": index_info["n_chains"],
            "n_frames": int(len(endpoint_times)),
            "time_first_ps": float(endpoint_times[0]) if len(endpoint_times) else "",
            "time_last_ps": float(endpoint_times[-1]) if len(endpoint_times) else "",
            "dt_ps": float(coord_dt_ps),
            "tau_ree_ps": tau_ree_ps,
            "tau_ree_censored": bool(tau_ree_censored),
            "tau_fs_ps": None,
            "tau_fs_censored": True,
            "tau_alpha_ps": None,
            "tau_alpha_censored": True,
            "q_nm_inv": None,
            "q_source": fs_summary.get("q_source", "MISSING_INTER_RDF_PEAK"),
            "source_case_dir": str(case_input_dir),
        }, ree_rows, fs_rows

    q_nm_inv = float(q_info["q_nm_inv"])
    carbon_xvg = extract_coords_xvg(
        gmx_bin,
        nocom,
        tpr,
        index_path,
        "fs_carbon_atoms",
        dyn_case_dir / "carbon_coords.xvg",
        coord_dt_ps,
        force=force,
    )
    expected_fs_atoms = len(select_evenly(index_info["carbon_atoms"], max_fs_atoms))
    carbon_times, carbon_coords = parse_coord_xvg(carbon_xvg, expected_atoms=expected_fs_atoms)
    fs_rows = compute_fsqt(
        carbon_coords,
        carbon_times,
        q_nm_inv,
        origin_stride=origin_stride,
        max_lag_ps=max_corr_lag_ps,
    )
    write_csv(dyn_case_dir / "fsqt.csv", fs_rows, ["time_ps", "Fs_q_t", "n_origins", "n_atoms_sampled"])
    tau_fs_ps, tau_fs_censored = first_crossing_time(
        [row["time_ps"] for row in fs_rows],
        [row["Fs_q_t"] for row in fs_rows],
    )
    tau_alpha_ps = tau_fs_ps
    tau_alpha_censored = tau_fs_censored
    fs_summary = {
        "analysis_version": ANALYSIS_VERSION,
        "case_id": case_id,
        "status": "ok",
        "method": "multi-time-origin isotropic carbon self-intermediate scattering function sin(qr)/(qr)",
        "n_chains": index_info["n_chains"],
        **time_metadata(carbon_times),
        "coord_dt_ps": float(coord_dt_ps),
        "max_corr_lag_ps": float(max_corr_lag_ps) if max_corr_lag_ps is not None else None,
        "origin_stride": int(origin_stride),
        "carbon_atoms_total": len(index_info["carbon_atoms"]),
        "carbon_atoms_sampled": fs_rows[0]["n_atoms_sampled"] if fs_rows else 0,
        "q_nm_inv": q_nm_inv,
        "q_source": q_info.get("q_source", "rdf_kind=inter_chain_cc first_peak_r_nm"),
        "tau_fs_ps": tau_fs_ps,
        "tau_fs_censored": bool(tau_fs_censored),
        "tau_alpha_ps": tau_alpha_ps,
        "tau_alpha_censored": bool(tau_alpha_censored),
    }
    write_json(dyn_case_dir / "fsqt_summary.json", fs_summary)
    if not keep_intermediates:
        remove_intermediates(temporary_paths)

    return {
        "n_chains": index_info["n_chains"],
        "n_frames": int(min(len(endpoint_times), len(carbon_times))),
        "time_first_ps": float(min(endpoint_times[0], carbon_times[0])),
        "time_last_ps": float(min(endpoint_times[-1], carbon_times[-1])),
        "dt_ps": float(coord_dt_ps),
        "tau_ree_ps": tau_ree_ps,
        "tau_ree_censored": bool(tau_ree_censored),
        "tau_fs_ps": tau_fs_ps,
        "tau_fs_censored": bool(tau_fs_censored),
        "tau_alpha_ps": tau_alpha_ps,
        "tau_alpha_censored": bool(tau_alpha_censored),
        "q_nm_inv": q_nm_inv,
        "q_source": q_info.get("q_source", "rdf_kind=inter_chain_cc first_peak_r_nm"),
        "source_case_dir": str(case_input_dir),
    }, ree_rows, fs_rows


def read_optional_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_coordinate_summary(dyn_case_dir):
    existing = read_optional_json(dyn_case_dir / "dynamics_summary.json")
    ree = read_optional_json(dyn_case_dir / "ree_vector_autocorr_summary.json")
    fs = read_optional_json(dyn_case_dir / "fsqt_summary.json")
    merged = {
        "source_case_dir": existing.get("source_case_dir", ""),
    }
    if ree:
        merged.update({
            "n_chains": ree.get("n_chains", ""),
            "n_frames": ree.get("n_frames", ""),
            "time_first_ps": ree.get("time_first_ps", ""),
            "time_last_ps": ree.get("time_last_ps", ""),
            "dt_ps": ree.get("dt_ps", ""),
            "tau_ree_ps": ree.get("tau_ree_ps", ""),
            "tau_ree_censored": ree.get("tau_ree_censored", ""),
        })
    if fs:
        merged.update({
            "n_chains": merged.get("n_chains") or fs.get("n_chains", ""),
            "n_frames": merged.get("n_frames") or fs.get("n_frames", ""),
            "time_first_ps": merged.get("time_first_ps") or fs.get("time_first_ps", ""),
            "time_last_ps": merged.get("time_last_ps") or fs.get("time_last_ps", ""),
            "dt_ps": merged.get("dt_ps") or fs.get("dt_ps", ""),
            "tau_fs_ps": fs.get("tau_fs_ps", ""),
            "tau_fs_censored": fs.get("tau_fs_censored", ""),
            "tau_alpha_ps": fs.get("tau_alpha_ps", fs.get("tau_fs_ps", "")),
            "tau_alpha_censored": fs.get("tau_alpha_censored", fs.get("tau_fs_censored", "")),
            "q_nm_inv": fs.get("q_nm_inv", ""),
            "q_source": fs.get("q_source", ""),
        })
    return merged


def load_long_csv(path, case_id, value_fields):
    rows = []
    for row in read_csv_rows(path):
        row = dict(row)
        row["case_id"] = case_id
        rows.append({key: row.get(key, "") for key in ["case_id", "time_ps"] + value_fields})
    return rows


def make_plots(batch_root, alpha_long, ree_long, fs_long, summary_rows):
    if plt is None:
        return
    agg = aggregate_dir(batch_root)
    agg.mkdir(parents=True, exist_ok=True)
    labels_map = load_case_label_map(Path(batch_root).parent)

    def plot_long(rows, value_key, ylabel, stem, ylim=None):
        by_case = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row)
        if not by_case:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for case_id, case_rows in sorted(by_case.items()):
            x = [safe_float(row["time_ps"]) / 1000.0 for row in case_rows]
            y = [safe_float(row[value_key]) for row in case_rows]
            ax.plot(x, y, lw=1.2, label=label_for_case(case_id, labels_map))
        ax.set_xlabel("Time / ns")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(agg / f"{stem}.png", dpi=220)
        fig.savefig(agg / f"{stem}.pdf")
        plt.close(fig)

    plot_long(alpha_long, "alpha", "MSD local slope alpha(t)", "msd_alpha_15cases", ylim=(0.0, 2.0))
    plot_long(ree_long, "C_Ree", "C_Ree(t)", "ree_autocorr_15cases", ylim=(-0.2, 1.05))
    plot_long(fs_long, "Fs_q_t", "F_s(q,t)", "fsqt_15cases", ylim=(-0.2, 1.05))

    if summary_rows:
        labels = [label_for_case(row["case_id"], labels_map) for row in summary_rows]
        x = np.arange(len(labels))
        tau_ree = [safe_float(row.get("tau_ree_ps")) / 1000.0 for row in summary_rows]
        tau_fs = [safe_float(row.get("tau_fs_ps")) / 1000.0 for row in summary_rows]
        fig, ax = plt.subplots(figsize=(11, 5))
        width = 0.38
        ax.bar(x - width / 2.0, tau_ree, width=width, label="tau_Ree")
        ax.bar(x + width / 2.0, tau_fs, width=width, label="tau_Fs")
        ax.set_ylabel("Relaxation time / ns")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(agg / "dynamics_tau_bar_15cases.png", dpi=220)
        fig.savefig(agg / "dynamics_tau_bar_15cases.pdf")
        plt.close(fig)


def make_fs_plots(batch_root, fs_long, summary_rows):
    if plt is None:
        return
    agg = aggregate_dir(batch_root)
    agg.mkdir(parents=True, exist_ok=True)
    labels_map = load_case_label_map(Path(batch_root).parent)

    by_case = {}
    for row in fs_long:
        by_case.setdefault(row["case_id"], []).append(row)
    if by_case:
        fig, ax = plt.subplots(figsize=(10, 6))
        for case_id, case_rows in sorted(by_case.items()):
            x = [safe_float(row["time_ps"]) / 1000.0 for row in case_rows]
            y = [safe_float(row["Fs_q_t"]) for row in case_rows]
            ax.plot(x, y, lw=1.2, label=label_for_case(case_id, labels_map))
        ax.set_xlabel("Time / ns")
        ax.set_ylabel("F_s(q,t)")
        ax.set_ylim(-0.2, 1.05)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(agg / "fsqt_15cases.png", dpi=220)
        fig.savefig(agg / "fsqt_15cases.pdf")
        plt.close(fig)

    if summary_rows:
        labels = [label_for_case(row["case_id"], labels_map) for row in summary_rows]
        x = np.arange(len(labels))
        tau_ree = [safe_float(row.get("tau_ree_ps")) / 1000.0 for row in summary_rows]
        tau_fs = [safe_float(row.get("tau_fs_ps")) / 1000.0 for row in summary_rows]
        fig, ax = plt.subplots(figsize=(11, 5))
        width = 0.38
        ax.bar(x - width / 2.0, tau_ree, width=width, label="tau_Ree")
        ax.bar(x + width / 2.0, tau_fs, width=width, label="tau_Fs")
        ax.set_ylabel("Relaxation time / ns")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(agg / "dynamics_tau_bar_15cases.png", dpi=220)
        fig.savefig(agg / "dynamics_tau_bar_15cases.pdf")
        plt.close(fig)


def q_info_for_case(args, q_map, case_id):
    if args.fs_q_nm_inv is not None:
        return {
            "q_nm_inv": float(args.fs_q_nm_inv),
            "q_source": args.fs_q_source,
            "q_missing": False,
        }
    q_info = q_map.get(case_id)
    if q_info:
        return {**q_info, "q_missing": False}
    return missing_q_info(case_id)


def load_existing_summary_rows(batch_root):
    rows = {}
    for row in read_csv_rows(aggregate_dir(batch_root) / "dynamics_summary.csv"):
        case_id = row.get("case_id")
        if case_id:
            rows[case_id] = dict(row)
    return rows


def run_fs_only_analysis(args):
    cases = load_case_ids(args)
    batch_root = batch_dir(args.output_root, "dynamics", args.batch_date)
    agg = aggregate_dir(batch_root)
    agg.mkdir(parents=True, exist_ok=True)
    rdf_summary = args.rdf_summary or (Path(args.output_root) / args.rdf_batch / "aggregate" / "rdf_summary.csv")
    q_map = load_q_map(rdf_summary)
    source_map = load_source_map(args.output_root)
    existing_rows = load_existing_summary_rows(batch_root)

    summary_rows = []
    fs_long = []

    for case_id in cases:
        dyn_dir = case_dir(batch_root, case_id, "dynamics")
        dyn_dir.mkdir(parents=True, exist_ok=True)
        q_info = q_info_for_case(args, q_map, case_id)
        row = dict(existing_rows.get(case_id) or read_optional_json(dyn_dir / "dynamics_summary.json") or {})
        row.update({
            "case_id": case_id,
            "method": row.get("method") or "MSD local slope + Ree vector autocorrelation + carbon Fs(q,t)",
            "analysis_version": ANALYSIS_VERSION,
        })
        source_case_dir = row.get("source_case_dir") or source_map.get(case_id, "")
        case_input_dir = find_case_input_dir(args.input_root, case_id) if args.input_root else None
        if not case_input_dir and source_case_dir:
            candidate = Path(source_case_dir)
            if (candidate / "nvt_prod" / "nvt_prod.tpr").exists():
                case_input_dir = candidate
        if not case_input_dir:
            row["status"] = "missing_input"
            row["source_case_dir"] = source_case_dir
            write_json(dyn_dir / "dynamics_summary.json", row)
            summary_rows.append(row)
            continue

        try:
            fs_summary, fs_rows = run_fsqt_dynamics(
                case_id,
                case_input_dir,
                dyn_dir,
                args.gmx_bin,
                q_info,
                args.coord_dt_ps,
                args.origin_stride,
                args.max_fs_atoms,
                args.max_corr_lag_ps,
                force=args.force,
                keep_intermediates=args.keep_intermediates,
            )
            row["status"] = fs_summary.get("status", "ok")
            row["source_case_dir"] = fs_summary["source_case_dir"]
            for key in (
                "n_chains",
                "tau_fs_ps",
                "tau_fs_censored",
                "tau_alpha_ps",
                "tau_alpha_censored",
                "q_nm_inv",
                "q_source",
            ):
                row[key] = fs_summary.get(key, row.get(key, ""))
            fs_long.extend({"case_id": case_id, **fs_row} for fs_row in fs_rows)
            error_path = dyn_dir / "dynamics_error.json"
            if error_path.exists():
                error_path.unlink()
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            row["source_case_dir"] = str(case_input_dir)
            write_json(dyn_dir / "dynamics_error.json", {"case_id": case_id, "error": str(exc)})
        write_json(dyn_dir / "dynamics_summary.json", row)
        summary_rows.append(row)

    write_csv(agg / "fsqt_long.csv", fs_long, ["case_id", "time_ps", "Fs_q_t", "n_origins", "n_atoms_sampled"])
    write_csv(agg / "dynamics_summary.csv", summary_rows, DYNAMICS_SUMMARY_FIELDS)
    write_json(
        agg / "dynamics_manifest.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "batch": batch_root.name,
            "cases": cases,
            "coord_dt_ps": args.coord_dt_ps,
            "origin_stride": args.origin_stride,
            "max_fs_atoms": args.max_fs_atoms,
            "max_corr_lag_ps": args.max_corr_lag_ps,
            "fs_only": True,
            "fs_q_nm_inv": args.fs_q_nm_inv,
            "fs_q_source": args.fs_q_source if args.fs_q_nm_inv is not None else "rdf-derived/no-fallback",
        },
    )
    make_fs_plots(batch_root, fs_long, summary_rows)
    return batch_root


def run_analysis(args):
    cases = load_case_ids(args)
    batch_root = batch_dir(args.output_root, "dynamics", args.batch_date)
    agg = aggregate_dir(batch_root)
    agg.mkdir(parents=True, exist_ok=True)
    msd_batch_root = Path(args.output_root) / args.msd_batch
    rdf_summary = args.rdf_summary or (Path(args.output_root) / args.rdf_batch / "aggregate" / "rdf_summary.csv")
    q_map = load_q_map(rdf_summary)
    source_map = load_source_map(args.output_root)

    summary_rows = []
    alpha_long = []
    ree_long = []
    fs_long = []

    for case_id in cases:
        dyn_dir = case_dir(batch_root, case_id, "dynamics")
        dyn_dir.mkdir(parents=True, exist_ok=True)
        q_info = q_info_for_case(args, q_map, case_id)

        row = {
            "case_id": case_id,
            "status": "ok",
            "method": "MSD local slope + Ree vector autocorrelation + carbon Fs(q,t)",
            "analysis_version": ANALYSIS_VERSION,
            "source_case_dir": source_map.get(case_id, ""),
            "q_nm_inv": q_info.get("q_nm_inv"),
            "q_source": q_info.get("q_source"),
        }

        if not args.skip_msd_alpha:
            alpha_summary, alpha_rows = compute_case_msd_alpha(case_id, msd_batch_root, dyn_dir, force=args.force)
            if alpha_summary:
                row.update(alpha_summary)
                alpha_long.extend({"case_id": case_id, **alpha_row} for alpha_row in alpha_rows)
            else:
                row.update({"alpha_100_300_mean": "", "alpha_100_300_status": "missing_msd"})

        if not args.skip_coordinates:
            case_input_dir = find_case_input_dir(args.input_root, case_id)
            if not case_input_dir:
                row["status"] = "missing_input"
            else:
                try:
                    coord_summary, coord_ree_rows, coord_fs_rows = run_coordinate_dynamics(
                        case_id,
                        case_input_dir,
                        dyn_dir,
                        args.gmx_bin,
                        q_info,
                        args.coord_dt_ps,
                        args.origin_stride,
                        args.max_fs_atoms,
                        args.max_corr_lag_ps,
                        force=args.force,
                        keep_intermediates=args.keep_intermediates,
                    )
                    row.update(coord_summary)
                    ree_long.extend({"case_id": case_id, **ree_row} for ree_row in coord_ree_rows)
                    fs_long.extend({"case_id": case_id, **fs_row} for fs_row in coord_fs_rows)
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = str(exc)
                    write_json(dyn_dir / "dynamics_error.json", {"case_id": case_id, "error": str(exc)})
        else:
            existing_coord = load_existing_coordinate_summary(dyn_dir)
            row.update(existing_coord)
            row["source_case_dir"] = source_map.get(
                case_id,
            "",
            )
            ree_long.extend(load_long_csv(dyn_dir / "ree_vector_autocorr.csv", case_id, ["C_Ree", "n_origins"]))
            fs_long.extend(load_long_csv(dyn_dir / "fsqt.csv", case_id, ["Fs_q_t", "n_origins", "n_atoms_sampled"]))

        write_json(dyn_dir / "dynamics_summary.json", row)
        summary_rows.append(row)

    write_csv(agg / "dynamics_summary.csv", summary_rows, DYNAMICS_SUMMARY_FIELDS)
    write_csv(agg / "msd_local_slope_long.csv", alpha_long, ["case_id", "time_ps", "alpha", "fit_r2"])
    write_csv(agg / "ree_autocorr_long.csv", ree_long, ["case_id", "time_ps", "C_Ree", "n_origins"])
    write_csv(agg / "fsqt_long.csv", fs_long, ["case_id", "time_ps", "Fs_q_t", "n_origins", "n_atoms_sampled"])
    write_json(
        agg / "dynamics_manifest.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "batch": batch_root.name,
            "cases": cases,
            "coord_dt_ps": args.coord_dt_ps,
            "origin_stride": args.origin_stride,
            "max_fs_atoms": args.max_fs_atoms,
            "max_corr_lag_ps": args.max_corr_lag_ps,
            "msd_batch": args.msd_batch,
            "rdf_batch": args.rdf_batch,
        },
    )
    make_plots(batch_root, alpha_long, ree_long, fs_long, summary_rows)
    return batch_root


def main():
    args = parse_args()
    if args.fs_only:
        batch_root = run_fs_only_analysis(args)
    else:
        batch_root = run_analysis(args)
    print(f"Wrote dynamics batch: {batch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
