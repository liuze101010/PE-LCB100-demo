#!/usr/bin/env python3
"""Compute intrachain P(s) contact probability and g1/g3 MSD for LCB cases."""

import argparse
import csv
import json
import math
import os
import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

try:
    from case_labels import label_for_case, load_case_label_map
except ImportError:  # pragma: no cover
    from .case_labels import label_for_case, load_case_label_map  # type: ignore

try:
    from trajectory_helpers import resolve_analysis_tpr, resolve_trajectory as resolve_core_trajectory
except ImportError:  # pragma: no cover
    from .trajectory_helpers import (  # type: ignore
        resolve_analysis_tpr,
        resolve_trajectory as resolve_core_trajectory,
    )


ANALYSIS_VERSION = "canonical_v2"
CASE_DIR_SUFFIX = "_md_newserver_lcb100_manual_20260426"
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

G1_G3_TIMEPOINTS = [
    (1000.0, "1ns"),
    (10000.0, "10ns"),
    (50000.0, "50ns"),
    (100000.0, "100ns"),
]

FIXED_CONTACT_CUTOFFS_NM = [0.6, 0.8, 1.0]


def run(cmd, stdin=None, stdout=None, stderr=None):
    stdout_handle = open(stdout, "w") if stdout else subprocess.PIPE
    stderr_handle = open(stderr, "w") if stderr else subprocess.PIPE
    env = os.environ.copy()
    env.setdefault("GMX_MAXBACKUP", "-1")
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            input=stdin,
            universal_newlines=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            env=env,
        )
    finally:
        if stdout:
            stdout_handle.close()
        if stderr:
            stderr_handle.close()
    if proc.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(str(part) for part in cmd))
    return proc


def ensure_current_or_restore_backup(path):
    """Recover a GROMACS output that was moved to a numbered backup on retry."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    backups = sorted(
        path.parent.glob(f"#{path.name}.*#"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for backup in backups:
        if backup.exists() and backup.stat().st_size > 0:
            backup.replace(path)
            return path
    return path


def capture(cmd):
    proc = subprocess.run([str(part) for part in cmd], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or ("command failed: " + " ".join(str(part) for part in cmd)))
    return proc.stdout


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value, default=math.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_xvg_numeric(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("@", "#")):
            continue
        parts = stripped.split()
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: non-numeric XVG row") from exc
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = len(rows[0])
    bad = [idx for idx, row in enumerate(rows, start=1) if len(row) != width]
    if bad:
        raise ValueError(f"{path}: inconsistent XVG column count at numeric rows {bad[:5]}")
    return np.asarray(rows, dtype=float)


def parse_coord_xvg(path, expected_atoms):
    data = parse_xvg_numeric(path)
    if data.shape[1] != 1 + 3 * int(expected_atoms):
        raise ValueError(f"{path}: expected {1 + 3 * int(expected_atoms)} columns, got {data.shape[1]}")
    return data[:, 0], data[:, 1:].reshape((data.shape[0], int(expected_atoms), 3))


def parse_msd_xvg(path):
    data = parse_xvg_numeric(path)
    if data.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns")
    return data[:, 0], data[:, 1]


def parse_tpr_topology(gmx_bin, tpr_path):
    dump = capture([gmx_bin, "dump", "-s", str(tpr_path)])
    natoms_match = re.search(r"natoms\s*=\s*(\d+)", dump)
    nmol_match = re.search(r"#molecules\s*=\s*(\d+)", dump)
    mol_atoms_match = re.search(r"atom \((\d+)\):", dump)
    if not (natoms_match and nmol_match and mol_atoms_match):
        raise ValueError(f"failed to parse natoms/molecules/molecule atom count from {tpr_path}")
    natoms = int(natoms_match.group(1))
    n_chains = int(nmol_match.group(1))
    atoms_per_chain = int(mol_atoms_match.group(1))

    carbon_atoms = set()
    atom_records = {}
    all_bonds = []
    cc_bonds = []
    atom_re = re.compile(r"atom\[\s*(\d+)\]=\{")
    bond_re = re.compile(r"type=\d+\s+\((?:BONDS|CONSTR|CONSTRNC)\)\s+(\d+)\s+(\d+)")
    in_moltype0 = False
    in_atom_block = False
    in_iatoms = False
    for line in dump.splitlines():
        if re.match(r"\s*moltype\s+\(0\):", line):
            in_moltype0 = True
            in_atom_block = False
            in_iatoms = False
            continue
        if in_moltype0 and re.match(r"\s*moltype\s+\(\d+\):", line):
            break
        if not in_moltype0:
            continue
        if re.match(r"\s*atoms:", line):
            in_atom_block = True
            in_iatoms = False
            continue
        if re.match(r"\s*iatoms:", line):
            in_iatoms = True
            in_atom_block = False
            continue
        if in_atom_block:
            match = atom_re.search(line)
            if match and int(match.group(1)) < atoms_per_chain:
                local_index = int(match.group(1))
                atomic_number_match = re.search(r"atomnumber=\s*(\d+)", line)
                mass_match = re.search(r"\bm=\s*([-+0-9.eE]+)", line)
                resind_match = re.search(r"resind=\s*(\d+)", line)
                atomic_number = int(atomic_number_match.group(1)) if atomic_number_match else 0
                atom_records[local_index] = {
                    "local_index": local_index,
                    "atomnumber": atomic_number,
                    "mass": float(mass_match.group(1)) if mass_match else 1.0,
                    "resind": int(resind_match.group(1)) if resind_match else -1,
                }
                if atomic_number == 6:
                    carbon_atoms.add(local_index)
        elif in_iatoms:
            match = bond_re.search(line)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                if a < atoms_per_chain and b < atoms_per_chain:
                    all_bonds.append((a, b))
                if a in carbon_atoms and b in carbon_atoms:
                    cc_bonds.append((a, b))
    if not carbon_atoms or not cc_bonds:
        raise ValueError(f"failed to parse carbon graph from {tpr_path}")

    carbon_local = sorted(carbon_atoms)
    carbon_to_order = {atom: idx for idx, atom in enumerate(carbon_local)}
    adjacency = [[] for _ in carbon_local]
    for a, b in cc_bonds:
        if a in carbon_to_order and b in carbon_to_order:
            ia, ib = carbon_to_order[a], carbon_to_order[b]
            adjacency[ia].append(ib)
            adjacency[ib].append(ia)
    if len(graph_component(adjacency, 0)) != len(carbon_local):
        raise ValueError(f"carbon graph disconnected in {tpr_path}")

    carbon_global_atoms = []
    for chain_idx in range(n_chains):
        base = chain_idx * atoms_per_chain
        carbon_global_atoms.extend(base + atom + 1 for atom in carbon_local)
    return {
        "natoms": natoms,
        "n_chains": n_chains,
        "atoms_per_chain": atoms_per_chain,
        "carbon_per_chain": len(carbon_local),
        "carbon_local_atoms": carbon_local,
        "carbon_global_atoms": carbon_global_atoms,
        "all_atoms": list(range(1, natoms + 1)),
        "adjacency": adjacency,
        "all_bonds_local": all_bonds,
        "atom_records": [atom_records[index] for index in sorted(atom_records)],
    }


def graph_component(adjacency, start):
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def pair_distances_by_s(adjacency, min_s=4):
    pairs_by_s = defaultdict(list)
    for start in range(len(adjacency)):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        for end, s in dist.items():
            if end > start and s >= min_s:
                pairs_by_s[int(s)].append((start, end))
    return {s: np.asarray(pairs, dtype=np.int64) for s, pairs in sorted(pairs_by_s.items())}


def write_index(path, topology):
    groups = [
        ("chain_all", topology["all_atoms"]),
        ("carbon_atoms", topology["carbon_global_atoms"]),
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, atoms in groups:
            handle.write(f"[ {name} ]\n")
            for idx in range(0, len(atoms), 15):
                handle.write(" ".join(str(atom) for atom in atoms[idx : idx + 15]) + "\n")
            handle.write("\n")
    return path


def trajectory_time_bounds(gmx_bin, traj_path):
    proc = subprocess.run([gmx_bin, "check", "-f", str(traj_path)], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = proc.stdout + "\n" + proc.stderr
    times = [float(match.group(1)) for match in re.finditer(r"\btime\s+([-+]?\d+(?:\.\d*)?)", text)]
    if not times:
        return None
    return {"first_ps": min(times), "last_ps": max(times)}


def resolve_trajectory(gmx_bin, nvt_dir, out_dir, force=False, max_time_ps=None):
    return resolve_core_trajectory(
        gmx_bin,
        nvt_dir,
        out_dir,
        force=force,
        max_time_ps=max_time_ps,
    )


def prepare_nocom(gmx_bin, traj, tpr, ndx, out_dir, force=False):
    out_dir = Path(out_dir)
    nojump = out_dir / "nvt_prod_nojump.xtc"
    nocom = out_dir / "nvt_prod_nocom.xtc"
    if force or not nojump.exists() or nojump.stat().st_size == 0:
        run(
            [gmx_bin, "trjconv", "-f", traj, "-s", tpr, "-n", ndx, "-o", nojump, "-pbc", "nojump"],
            stdin="chain_all\n",
            stdout=out_dir / "gmx_trjconv_nojump.out",
            stderr=out_dir / "gmx_trjconv_nojump.err",
        )
        ensure_current_or_restore_backup(nojump)
    if force or not nocom.exists() or nocom.stat().st_size == 0:
        ensure_current_or_restore_backup(nojump)
        run(
            [gmx_bin, "trjconv", "-f", nojump, "-s", tpr, "-n", ndx, "-o", nocom, "-fit", "translation"],
            stdin="chain_all\nchain_all\n",
            stdout=out_dir / "gmx_trjconv_nocom.out",
            stderr=out_dir / "gmx_trjconv_nocom.err",
        )
        ensure_current_or_restore_backup(nocom)
    ensure_current_or_restore_backup(nojump)
    ensure_current_or_restore_backup(nocom)
    return nojump, nocom


def extract_carbon_coords(gmx_bin, traj, tpr, ndx, out_dir, dt_ps, force=False):
    out = Path(out_dir) / f"carbon_coords_dt{int(dt_ps)}.xvg"
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    run(
        [
            gmx_bin,
            "traj",
            "-f",
            traj,
            "-s",
            tpr,
            "-n",
            ndx,
            "-ox",
            out,
            "-dt",
            str(int(dt_ps) if float(dt_ps).is_integer() else dt_ps),
            "-xvg",
            "none",
        ],
        stdin="carbon_atoms\n",
        stdout=Path(out_dir) / "gmx_traj_carbon.out",
        stderr=Path(out_dir) / "gmx_traj_carbon.err",
    )
    return out


def run_msd(gmx_bin, traj, tpr, ndx, group_name, out_xvg, out_dir, mol=False, force=False):
    out_xvg = Path(out_xvg)
    if out_xvg.exists() and out_xvg.stat().st_size > 0 and not force:
        return out_xvg
    traj = ensure_current_or_restore_backup(traj)
    cmd = [
        gmx_bin,
        "msd",
        "-f",
        traj,
        "-s",
        tpr,
        "-n",
        ndx,
        "-o",
        out_xvg,
        "-trestart",
        "100",
        "-xvg",
        "none",
    ]
    if mol:
        cmd.extend(["-mol", Path(out_dir) / "g3_diffusion_per_chain.xvg"])
    run(
        cmd,
        stdin=f"{group_name}\n",
        stdout=Path(out_dir) / ("gmx_msd_g3.out" if mol else "gmx_msd_g1.out"),
        stderr=Path(out_dir) / ("gmx_msd_g3.err" if mol else "gmx_msd_g1.err"),
    )
    return out_xvg


def cutoff_key(cutoff):
    if str(cutoff).lower() == "rdfmin":
        return "rc_rdfmin"
    return "rc" + str(cutoff).replace(".", "p")


def cutoff_value(cutoff, cutoff_values=None):
    if str(cutoff).lower() == "rdfmin":
        if not cutoff_values or "rdfmin" not in cutoff_values:
            raise ValueError("rdfmin cutoff requested without a numeric rdf minimum")
        return float(cutoff_values["rdfmin"])
    return float(cutoff)


def contact_cutoffs_for_case(base_cutoffs, rdf_min_nm):
    cutoffs = list(base_cutoffs)
    cutoff_values = {}
    if math.isfinite(safe_float(rdf_min_nm)) and safe_float(rdf_min_nm) > 0:
        cutoffs.append("rdfmin")
        cutoff_values["rdfmin"] = float(rdf_min_nm)
    return cutoffs, cutoff_values


def load_rdf_min_maps(rdf_summary_path):
    rdf_summary_path = Path(rdf_summary_path) if rdf_summary_path else None
    if not rdf_summary_path or not rdf_summary_path.exists():
        return {}, {}
    out = {}
    sources = {}
    fallback = {}
    fallback_sources = {}
    for row in read_csv(rdf_summary_path):
        case_id = row.get("case_id")
        value = safe_float(row.get("first_min_r_nm"))
        if not case_id or not math.isfinite(value) or value <= 0:
            continue
        kind = row.get("rdf_kind", "")
        if kind == "excl":
            out[case_id] = value
            sources[case_id] = f"rdf_kind=excl first_min_r_nm={value:g}"
        elif kind == "intra_chain_nonbonded_cc":
            fallback.setdefault(case_id, value)
            fallback_sources.setdefault(case_id, f"rdf_kind=intra_chain_nonbonded_cc first_min_r_nm={value:g}")
    for case_id, value in fallback.items():
        out.setdefault(case_id, value)
        sources.setdefault(case_id, fallback_sources.get(case_id, "rdf_kind=intra_chain_nonbonded_cc first_min_r_nm"))
    return out, sources


def load_rdf_min_map(rdf_summary_path):
    values, _sources = load_rdf_min_maps(rdf_summary_path)
    return values


def load_tau_alpha_map(dynamics_summary_path):
    dynamics_summary_path = Path(dynamics_summary_path) if dynamics_summary_path else None
    if not dynamics_summary_path or not dynamics_summary_path.exists():
        return {}
    out = {}
    for row in read_csv(dynamics_summary_path):
        case_id = row.get("case_id")
        value = safe_float(row.get("tau_alpha_ps"), safe_float(row.get("tau_fs_ps")))
        if case_id and math.isfinite(value) and value > 0:
            out[case_id] = value
    return out


def derive_msd_alpha_window(times_ps, tau_alpha_ps):
    times = np.asarray(times_ps, dtype=float)
    finite_times = times[np.isfinite(times)]
    tau_alpha = safe_float(tau_alpha_ps)
    if math.isfinite(tau_alpha) and tau_alpha > 0:
        traj_end_ps = float(np.max(finite_times)) if finite_times.size else math.nan
        begin_ps = max(tau_alpha, 1000.0)
        if math.isfinite(traj_end_ps):
            end_ps = min(traj_end_ps / 3.0, 10.0 * tau_alpha)
        else:
            end_ps = 10.0 * tau_alpha
        # tau_alpha-derived window is only usable when non-degenerate. At the
        # inter-chain cage wavevector tau_alpha is ~tens of ps, so 10*tau_alpha
        # drops below the 1 ns begin floor and begin >= end; those cases revert
        # to the fixed window instead of being mislabeled tau_alpha_derived.
        if math.isfinite(end_ps) and end_ps > begin_ps:
            return begin_ps, end_ps, "tau_alpha_derived", tau_alpha
    return 100000.0, 300000.0, "FIXED_FALLBACK", None


def compute_contact_probability(coord_xvg, topology, pairs_by_s, cutoffs, cutoff_values=None):
    times, coords = parse_coord_xvg(coord_xvg, expected_atoms=len(topology["carbon_global_atoms"]))
    n_chains = topology["n_chains"]
    c_per_chain = topology["carbon_per_chain"]
    counts = {s: {cutoff: 0 for cutoff in cutoffs} for s in pairs_by_s}
    numeric_cutoffs = {cutoff: cutoff_value(cutoff, cutoff_values) for cutoff in cutoffs}
    eligible = {s: int(len(pairs) * n_chains * len(times)) for s, pairs in pairs_by_s.items()}
    for frame in coords:
        for chain_idx in range(n_chains):
            sub = frame[chain_idx * c_per_chain : (chain_idx + 1) * c_per_chain]
            for s, pairs in pairs_by_s.items():
                diff = sub[pairs[:, 0]] - sub[pairs[:, 1]]
                dist = np.sqrt(np.sum(diff * diff, axis=1))
                for cutoff in cutoffs:
                    counts[s][cutoff] += int(np.count_nonzero(dist < numeric_cutoffs[cutoff]))
    rows = []
    for s in sorted(pairs_by_s):
        row = {
            "s": s,
            "eligible_pairs_per_chain": int(len(pairs_by_s[s])),
            "eligible_observations": eligible[s],
        }
        for cutoff in cutoffs:
            key = cutoff_key(cutoff)
            row[f"P_contact_{key}"] = counts[s][cutoff] / eligible[s] if eligible[s] else math.nan
        rows.append(row)
    return rows, times


def local_alpha(times, values, begin_ps=None, end_ps=None):
    if begin_ps is None:
        begin_ps = 100000.0
    if end_ps is None:
        end_ps = 300000.0
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (times >= begin_ps) & (times <= end_ps) & (times > 0) & (values > 0) & np.isfinite(values)
    if int(np.count_nonzero(mask)) < 3:
        return math.nan
    x = np.log10(times[mask])
    y = np.log10(values[mask])
    return float(np.polyfit(x, y, 1)[0])


def interp_value(times, values, target_ps):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]
    if times.size == 0 or target_ps < times.min() or target_ps > times.max():
        return math.nan
    return float(np.interp(target_ps, times, values))


def finite_ratio(numerator, denominator):
    if math.isfinite(numerator) and math.isfinite(denominator) and denominator != 0:
        return numerator / denominator
    return ""


def series_from_rows(rows, value_name):
    times = []
    values = []
    for row in rows:
        try:
            times.append(float(row["time_ps"]))
            values.append(float(row[value_name]))
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def add_g1_g3_timepoint_metrics(summary, g1_times, g1_values, g3_times, g3_values):
    for target_ps, label in G1_G3_TIMEPOINTS:
        g1_value = interp_value(g1_times, g1_values, target_ps)
        g3_value = interp_value(g3_times, g3_values, target_ps)
        summary[f"g1_{label}_nm2"] = g1_value
        summary[f"g3_{label}_nm2"] = g3_value
        summary[f"g1_over_g3_{label}"] = finite_ratio(g1_value, g3_value)
    return summary


def write_msd_csv(path, times, values, value_name):
    rows = [{"time_ps": float(t), value_name: float(v)} for t, v in zip(times, values)]
    write_csv(path, rows, ["time_ps", value_name])
    return rows


def case_dir_from_id(input_root, case_id):
    direct = Path(input_root) / (case_id + CASE_DIR_SUFFIX)
    if direct.exists():
        return direct
    matches = sorted(Path(input_root).glob(case_id + "*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"case directory not found for {case_id} under {input_root}")


def analyze_case(
    case_id,
    input_root,
    batch_dir,
    gmx_bin,
    dt_ps,
    cutoffs,
    force=False,
    keep_intermediates=False,
    rdf_min_nm=None,
    rdf_min_source=None,
    tau_alpha_ps=None,
):
    case_input = case_dir_from_id(input_root, case_id)
    case_out = Path(batch_dir) / "cases" / case_id / "intrachain"
    case_out.mkdir(parents=True, exist_ok=True)
    nvt = case_input / "nvt_prod"
    tpr = resolve_analysis_tpr(gmx_bin, case_input, nvt, force=force)
    topology = parse_tpr_topology(gmx_bin, tpr)
    ndx = write_index(case_out / "intrachain.ndx", topology)
    traj = resolve_trajectory(gmx_bin, nvt, case_out, force=force)
    nojump, nocom = prepare_nocom(gmx_bin, traj, tpr, ndx, case_out, force=force)
    coords_xvg = extract_carbon_coords(gmx_bin, nocom, tpr, ndx, case_out, dt_ps, force=force)

    pairs_by_s = pair_distances_by_s(topology["adjacency"], min_s=4)
    contact_cutoffs, cutoff_values = contact_cutoffs_for_case(cutoffs, rdf_min_nm)
    contact_rows, contact_times = compute_contact_probability(
        coords_xvg,
        topology,
        pairs_by_s,
        contact_cutoffs,
        cutoff_values=cutoff_values,
    )
    contact_fields = ["s", "eligible_pairs_per_chain", "eligible_observations"] + [
        f"P_contact_{cutoff_key(c)}" for c in contact_cutoffs
    ]
    write_csv(case_out / "contact_probability.csv", contact_rows, contact_fields)

    contact_summary = {
        "case_id": case_id,
        "status": "ok",
        "analysis_version": ANALYSIS_VERSION,
        "method": "same-chain carbon graph P(s), graph distance >=4",
        "n_chains": topology["n_chains"],
        "n_frames": int(len(contact_times)),
        "time_first_ps": float(contact_times[0]) if len(contact_times) else "",
        "time_last_ps": float(contact_times[-1]) if len(contact_times) else "",
        "dt_ps": float(dt_ps),
        "carbon_per_chain": topology["carbon_per_chain"],
        "contact_cutoffs_nm": [cutoff_values.get(c, c) for c in contact_cutoffs],
        "rdf_min_used_nm": cutoff_values.get("rdfmin", ""),
        "rdf_min_status": "PRESENT" if "rdfmin" in cutoff_values else "MISSING",
        "rdf_min_source": rdf_min_source if "rdfmin" in cutoff_values else "",
        "min_graph_distance": 4,
        "source_case_dir": str(case_input.resolve()),
    }
    for s in (10, 20, 50):
        match = next((row for row in contact_rows if int(row["s"]) == s), None)
        if match:
            contact_summary[f"P_contact_rc0p8_s{s}"] = match.get("P_contact_rc0p8", "")
            contact_summary[f"P_contact_rc_rdfmin_s{s}"] = match.get("P_contact_rc_rdfmin", "")
    write_json(case_out / "contact_probability_summary.json", contact_summary)

    # Keep contact evidence independent from the more expensive MSD sub-steps.
    # The clean release gate can then consume contact evidence even when a
    # segmental-MSD helper fails on one large case.
    g1_t = np.asarray([], dtype=float)
    g1 = np.asarray([], dtype=float)
    g3_t = np.asarray([], dtype=float)
    g3 = np.asarray([], dtype=float)
    g1_rows = []
    g3_rows = []
    g1_error = ""
    g3_error = ""

    try:
        g3_xvg = run_msd(gmx_bin, nocom, tpr, ndx, "carbon_atoms", case_out / "g3_com_msd.xvg", case_out, mol=True, force=force)
        g3_t, g3 = parse_msd_xvg(g3_xvg)
        g3_rows = write_msd_csv(case_out / "g3_com_msd.csv", g3_t, g3, "g3_com_msd_nm2")
    except Exception as exc:
        g3_error = str(exc)

    try:
        g1_xvg = run_msd(gmx_bin, nocom, tpr, ndx, "carbon_atoms", case_out / "g1_segmental_msd.xvg", case_out, mol=False, force=force)
        g1_t, g1 = parse_msd_xvg(g1_xvg)
        g1_rows = write_msd_csv(case_out / "g1_segmental_msd.csv", g1_t, g1, "g1_segmental_msd_nm2")
    except Exception as exc:
        g1_error = str(exc)

    window_times = g3_t if len(g3_t) else g1_t
    msd_begin_ps, msd_end_ps, msd_source, tau_alpha_used = derive_msd_alpha_window(window_times, tau_alpha_ps)
    g1_alpha = local_alpha(g1_t, g1, begin_ps=msd_begin_ps, end_ps=msd_end_ps)
    g3_alpha = local_alpha(g3_t, g3, begin_ps=msd_begin_ps, end_ps=msd_end_ps)
    first_times = [arr[0] for arr in (g1_t, g3_t) if len(arr)]
    last_times = [arr[-1] for arr in (g1_t, g3_t) if len(arr)]

    g_summary = {
        "case_id": case_id,
        "status": "ok" if not (g1_error or g3_error) else "partial",
        "analysis_version": ANALYSIS_VERSION,
        "method": "g1 carbon skeleton segmental MSD and g3 chain COM MSD from nojump/nocom trajectory",
        "n_chains": topology["n_chains"],
        "n_frames_g1": int(len(g1_t)),
        "n_frames_g3": int(len(g3_t)),
        "time_first_ps": float(min(first_times)) if first_times else "",
        "time_last_ps": float(min(last_times)) if last_times else "",
        "carbon_per_chain": topology["carbon_per_chain"],
        "g1_alpha_100_300": g1_alpha,
        "g3_alpha_100_300": g3_alpha,
        "g1_alpha_window": g1_alpha,
        "g3_alpha_window": g3_alpha,
        "msd_window_begin_ps": msd_begin_ps,
        "msd_window_end_ps": msd_end_ps,
        "msd_window_source": msd_source,
        "tau_alpha_ps_used": tau_alpha_used if tau_alpha_used is not None else "",
        "trestart_ps": 100.0,
        "source_case_dir": str(case_input.resolve()),
        "g1_error": g1_error,
        "g3_error": g3_error,
        "g3_source": str((case_out / "g3_com_msd.csv").resolve()) if len(g3_rows) else "",
    }
    add_g1_g3_timepoint_metrics(g_summary, g1_t, g1, g3_t, g3)
    write_json(case_out / "g1_g3_summary.json", g_summary)

    if not keep_intermediates:
        for path in [coords_xvg, nojump, nocom, case_out / "nvt_prod.full.xtc"]:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
    return contact_summary, g_summary, contact_rows, g1_rows, g3_rows


def plot_outputs(batch_dir, contact_long, g1_long, g3_long, g_summaries):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    batch = Path(batch_dir)
    agg = batch / "aggregate"
    labels = load_case_label_map(batch.parent)
    outputs = []

    def to_float(value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return math.nan
        return result if math.isfinite(result) else math.nan

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    by_case = defaultdict(list)
    for row in contact_long:
        by_case[row["case_id"]].append(row)
    for case_id, rows in sorted(by_case.items()):
        rows = sorted(rows, key=lambda item: float(item["s"]))
        xs = [float(row["s"]) for row in rows]
        ys = [float(row.get("P_contact_rc0p8", "nan")) for row in rows]
        ax.loglog(xs, ys, label=label_for_case(case_id, labels), alpha=0.85)
    ax.set_xlabel("Carbon graph distance s")
    ax.set_ylabel("P(s), r_c = 0.8 nm")
    ax.set_title("Intrachain contact probability")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = agg / f"contact_probability_loglog_15cases.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for label, rows, field, style in [
        ("g1", g1_long, "g1_segmental_msd_nm2", "-"),
        ("g3", g3_long, "g3_com_msd_nm2", "--"),
    ]:
        by_case = defaultdict(list)
        for row in rows:
            by_case[row["case_id"]].append(row)
        for case_id, cr in sorted(by_case.items()):
            cr = sorted(cr, key=lambda item: float(item["time_ps"]))
            xs = [float(row["time_ps"]) for row in cr if float(row["time_ps"]) > 0]
            ys = [float(row[field]) for row in cr if float(row["time_ps"]) > 0]
            ax.loglog(xs, ys, style, label=f"{label_for_case(case_id, labels)} {label}", alpha=0.55)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("MSD (nm^2)")
    ax.set_title("g1 segmental MSD and g3 COM MSD")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = agg / f"g1_g3_loglog_15cases.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)

    cases = [label_for_case(row["case_id"], labels) for row in g_summaries]
    values = [to_float(row.get("g1_over_g3_100ns")) for row in g_summaries]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(range(len(cases)), values)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("g1(100 ns) / g3(100 ns)")
    ax.set_title("Segmental vs COM MSD ratio")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = agg / f"g1_over_g3_bar_15cases.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)

    cases = [label_for_case(row["case_id"], labels) for row in g_summaries]
    x = np.arange(len(cases), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for idx, (_, label) in enumerate(G1_G3_TIMEPOINTS):
        offset = (idx - (len(G1_G3_TIMEPOINTS) - 1) / 2.0) * width
        values = [to_float(row.get(f"g1_over_g3_{label}")) for row in g_summaries]
        ax.bar(x + offset, values, width=width, label=label)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("g1(t) / g3(t)")
    ax.set_title("Segmental vs COM MSD ratio at representative times")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="time")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = agg / f"g1_over_g3_multitime_bar_15cases.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def aggregate(batch_dir, cases):
    batch = Path(batch_dir)
    contact_long = []
    contact_summary = []
    g1_long = []
    g3_long = []
    g_summary = []
    for case_id in cases:
        case_out = batch / "cases" / case_id / "intrachain"
        if not case_out.exists():
            continue
        for row in read_csv(case_out / "contact_probability.csv"):
            row["case_id"] = case_id
            contact_long.append(row)
        csum_path = case_out / "contact_probability_summary.json"
        if csum_path.exists():
            contact_summary.append(json.loads(csum_path.read_text(encoding="utf-8")))
        g1_case_rows = read_csv(case_out / "g1_segmental_msd.csv")
        g3_case_rows = read_csv(case_out / "g3_com_msd.csv")
        for row in g1_case_rows:
            row["case_id"] = case_id
            g1_long.append(row)
        for row in g3_case_rows:
            row["case_id"] = case_id
            g3_long.append(row)
        gsum_path = case_out / "g1_g3_summary.json"
        if gsum_path.exists():
            summary = json.loads(gsum_path.read_text(encoding="utf-8"))
            g1_times, g1_values = series_from_rows(g1_case_rows, "g1_segmental_msd_nm2")
            g3_times, g3_values = series_from_rows(g3_case_rows, "g3_com_msd_nm2")
            add_g1_g3_timepoint_metrics(summary, g1_times, g1_values, g3_times, g3_values)
            write_json(gsum_path, summary)
            g_summary.append(summary)
    contact_long.sort(key=lambda row: (row["case_id"], float(row["s"])))
    contact_summary.sort(key=lambda row: row["case_id"])
    g1_long.sort(key=lambda row: (row["case_id"], float(row["time_ps"])))
    g3_long.sort(key=lambda row: (row["case_id"], float(row["time_ps"])))
    g_summary.sort(key=lambda row: row["case_id"])

    agg = batch / "aggregate"
    contact_fields = [
        "case_id",
        "s",
        "eligible_pairs_per_chain",
        "eligible_observations",
        "P_contact_rc0p6",
        "P_contact_rc0p8",
        "P_contact_rc1p0",
        "P_contact_rc_rdfmin",
    ]
    contact_summary_fields = [
        "case_id",
        "status",
        "analysis_version",
        "method",
        "n_chains",
        "n_frames",
        "time_first_ps",
        "time_last_ps",
        "dt_ps",
        "carbon_per_chain",
        "min_graph_distance",
        "rdf_min_used_nm",
        "rdf_min_status",
        "rdf_min_source",
        "P_contact_rc0p8_s10",
        "P_contact_rc0p8_s20",
        "P_contact_rc0p8_s50",
        "P_contact_rc_rdfmin_s10",
        "P_contact_rc_rdfmin_s20",
        "P_contact_rc_rdfmin_s50",
        "source_case_dir",
    ]
    g_fields = ["case_id", "time_ps", "g1_segmental_msd_nm2"]
    g3_fields = ["case_id", "time_ps", "g3_com_msd_nm2"]
    g_summary_fields = [
        "case_id",
        "status",
        "analysis_version",
        "method",
        "n_chains",
        "n_frames_g1",
        "n_frames_g3",
        "time_first_ps",
        "time_last_ps",
        "carbon_per_chain",
        "g1_1ns_nm2",
        "g3_1ns_nm2",
        "g1_over_g3_1ns",
        "g1_10ns_nm2",
        "g3_10ns_nm2",
        "g1_over_g3_10ns",
        "g1_50ns_nm2",
        "g3_50ns_nm2",
        "g1_over_g3_50ns",
        "g1_100ns_nm2",
        "g3_100ns_nm2",
        "g1_over_g3_100ns",
        "g1_alpha_100_300",
        "g3_alpha_100_300",
        "g1_alpha_window",
        "g3_alpha_window",
        "msd_window_begin_ps",
        "msd_window_end_ps",
        "msd_window_source",
        "tau_alpha_ps_used",
        "trestart_ps",
        "source_case_dir",
    ]
    write_csv(agg / "contact_probability_long.csv", contact_long, contact_fields)
    write_csv(agg / "contact_probability_summary.csv", contact_summary, contact_summary_fields)
    write_csv(agg / "g1_long.csv", g1_long, g_fields)
    write_csv(agg / "g3_long.csv", g3_long, g3_fields)
    write_csv(agg / "g1_g3_summary.csv", g_summary, g_summary_fields)
    plots = plot_outputs(batch, contact_long, g1_long, g3_long, g_summary)
    write_json(
        agg / "intrachain_dynamics_manifest.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "case_count": len(g_summary),
            "contact_case_count": len(contact_summary),
            "contact_cutoffs_nm": FIXED_CONTACT_CUTOFFS_NM,
            "contact_cutoffs_note": "Fixed 0.6/0.8/1.0 nm controls are retained; per-case RDF first-minimum cutoff is added when available.",
            "contact_min_graph_distance": 4,
            "g1_selection": "all carbon skeleton atoms",
            "g3_selection": "chain COM from gmx msd -mol on carbon_atoms",
            "g1_g3_recommended_segmental_ratio_fields": ["g1_over_g3_1ns", "g1_over_g3_10ns"],
            "g1_g3_transition_ratio_fields": ["g1_over_g3_50ns"],
            "g1_g3_long_time_reference_fields": ["g1_over_g3_100ns"],
            "g1_g3_timepoint_note": "1 ns and 10 ns are recommended for local/segmental dynamics; 100 ns is near COM-diffusion dominated and retained as a long-time reference.",
            "plots": plots,
        },
    )
    return contact_summary, g_summary


def parse_cases(args):
    if args.cases:
        return [item.strip() for item in args.cases.split(",") if item.strip()]
    if args.cases_file:
        return [line.strip() for line in Path(args.cases_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(DEFAULT_CASES)


def main():
    parser = argparse.ArgumentParser(description="Compute intrachain P(s) and g1/g3 MSD.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, default=Path("output/lcb100_intrachain_dynamics_20260504"))
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--cases")
    parser.add_argument("--cases-file")
    parser.add_argument("--dt-ps", type=float, default=1000.0)
    parser.add_argument("--rdf-summary", type=Path, help="RDF summary CSV with first_min_r_nm for RDF-derived P(s) cutoff.")
    parser.add_argument("--dynamics-summary", type=Path, help="Dynamics summary CSV with tau_alpha_ps for MSD alpha-window derivation.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()

    cases = parse_cases(args)
    batch = Path(args.batch_dir)
    batch.mkdir(parents=True, exist_ok=True)
    rdf_min_map, rdf_min_source_map = load_rdf_min_maps(args.rdf_summary)
    tau_alpha_map = load_tau_alpha_map(args.dynamics_summary)
    if not args.aggregate_only:
        for case_id in cases:
            print(f"==== {case_id} ====", flush=True)
            analyze_case(
                case_id,
                args.input_root,
                batch,
                args.gmx,
                args.dt_ps,
                FIXED_CONTACT_CUTOFFS_NM,
                force=args.force,
                keep_intermediates=args.keep_intermediates,
                rdf_min_nm=rdf_min_map.get(case_id),
                rdf_min_source=rdf_min_source_map.get(case_id),
                tau_alpha_ps=tau_alpha_map.get(case_id),
            )
    if not args.no_aggregate:
        aggregate(batch, cases)
    print(f"Intrachain dynamics complete: {len(cases)} requested cases")


if __name__ == "__main__":
    main()
