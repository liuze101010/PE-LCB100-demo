#!/usr/bin/env python3
"""Compute same-chain nonbonded C-C pair-distribution for LCB trajectories.

The descriptor counts only carbon-carbon pairs in the same molecule whose
graph distance on the carbon skeleton is greater than three. This excludes
1-2, 1-3, and 1-4 local topology contacts, leaving nonlocal intrachain
contacts.

Unlike an ordinary bulk RDF, the intrachain partner set is finite and tied to a
single chain conformation, so the raw histogram does not have to approach one
at long distance. The production descriptor therefore reports the same pair
selection after tail normalization over 1.5-2.0 nm. The raw curve is retained in
the long CSV for audit.
"""

import argparse
import json
import math
import re
import subprocess
from collections import deque
from pathlib import Path

import numpy as np


def run(cmd, stdin=None, stdout=None, stderr=None):
    with open(stdout, "w") if stdout else subprocess.DEVNULL as out, open(stderr, "w") if stderr else subprocess.DEVNULL as err:
        proc = subprocess.run(cmd, input=stdin, universal_newlines=True, stdout=out, stderr=err)
    if proc.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(str(x) for x in cmd))


def capture(cmd):
    proc = subprocess.run(cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or ("command failed: " + " ".join(str(x) for x in cmd)))
    return proc.stdout


def parse_index_atoms(path):
    atoms = []
    in_group = False
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("["):
            in_group = "Carbon" in stripped or "C*" in stripped
            continue
        if in_group:
            atoms.extend(int(x) for x in stripped.split())
    if not atoms:
        raise ValueError(f"no Carbon atoms found in index: {path}")
    return atoms


def parse_tpr_carbon_bonds(gmx_bin, tpr_path):
    """Return carbon local atom indices and C-C bonds from a TPR dump."""
    dump = capture([gmx_bin, "dump", "-s", str(tpr_path)])
    carbon_atoms = set()
    cc_bonds = []
    atom_re = re.compile(r"atom\[\s*(\d+)\]=\{.*atomnumber=\s*(\d+)")
    bond_re = re.compile(r"type=\d+\s+\(BONDS\)\s+(\d+)\s+(\d+)")
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
        if in_iatoms and line.strip().startswith("nr:"):
            continue

        if in_atom_block:
            match = atom_re.search(line)
            if match and int(match.group(2)) == 6:
                carbon_atoms.add(int(match.group(1)))
        elif in_iatoms:
            match = bond_re.search(line)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                if a in carbon_atoms and b in carbon_atoms:
                    cc_bonds.append((a, b))

    if not carbon_atoms:
        raise ValueError("failed to parse carbon atoms from TPR dump")
    if not cc_bonds:
        raise ValueError("failed to parse C-C bonds from TPR dump")
    return sorted(carbon_atoms), cc_bonds


def infer_carbons_per_chain(case_id, n_carbon, n_chains):
    match = re.search(r"_c(\d+)", case_id)
    if match:
        c_per_chain = int(match.group(1))
        if c_per_chain * n_chains == n_carbon:
            return c_per_chain
    if n_carbon % n_chains != 0:
        raise ValueError(f"carbon count {n_carbon} is not divisible by n_chains={n_chains}")
    return n_carbon // n_chains


def local_eligible_pairs(carbon_local_atoms, cc_bonds, max_excluded_distance=3):
    carbon_to_order = {atom: idx for idx, atom in enumerate(carbon_local_atoms)}
    adjacency = [[] for _ in carbon_local_atoms]
    for a, b in cc_bonds:
        if a in carbon_to_order and b in carbon_to_order:
            ia, ib = carbon_to_order[a], carbon_to_order[b]
            adjacency[ia].append(ib)
            adjacency[ib].append(ia)

    excluded = set()
    for start in range(len(carbon_local_atoms)):
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            node, dist = queue.popleft()
            if 0 < dist <= max_excluded_distance:
                excluded.add(tuple(sorted((start, node))))
            if dist >= max_excluded_distance:
                continue
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))

    pairs = []
    for i in range(len(carbon_local_atoms)):
        for j in range(i + 1, len(carbon_local_atoms)):
            if (i, j) not in excluded:
                pairs.append((i, j))
    if not pairs:
        raise ValueError("no eligible intrachain nonbonded C-C pairs after graph exclusion")
    return np.asarray(pairs, dtype=np.int64), len(excluded)


def load_rdf_centers(rmax, bin_width):
    count = int(round(rmax / bin_width))
    return np.arange(count, dtype=float) * bin_width


def rdf_edges_from_centers(r):
    r = np.asarray(r, dtype=float)
    dr = float(np.median(np.diff(r)))
    mids = (r[:-1] + r[1:]) / 2.0
    edges = np.empty(r.size + 1, dtype=float)
    edges[0] = max(0.0, r[0] - dr / 2.0)
    edges[1:-1] = mids
    edges[-1] = r[-1] + dr / 2.0
    return edges


def find_first_minimum_after_peak(
    r_grid,
    g_r,
    peak_r,
    min_separation_nm=0.10,
    search_low_nm=0.65,
    search_high_nm=0.90,
):
    """Return the deepest first-shell RDF trough after the supplied peak."""

    r = np.asarray(r_grid, dtype=float)
    g = np.asarray(g_r, dtype=float)
    if r.size < 3 or g.size != r.size or peak_r is None:
        return None
    try:
        peak_value = float(peak_r)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(peak_value):
        return None
    finite = np.isfinite(r) & np.isfinite(g)
    if int(np.count_nonzero(finite)) < 3:
        return None
    r = r[finite]
    g = g[finite]
    low = max(peak_value + float(min_separation_nm), float(search_low_nm))
    high = float(search_high_nm)
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        return None
    window = (r >= low) & (r <= high)
    if int(np.count_nonzero(window)) < 3:
        return None
    r_w = r[window]
    g_w = g[window]
    return float(r_w[int(np.argmin(g_w))])


def parse_box(path):
    boxes = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        vals = np.fromstring(stripped, sep=" ")
        if vals.size >= 4:
            boxes.append(vals[1:4].astype(float))
    if not boxes:
        raise ValueError(f"no box data: {path}")
    return np.asarray(boxes, dtype=float)


def iter_coord_frames(path, n_carbon):
    expected = 1 + 3 * n_carbon
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "@")):
                continue
            vals = np.fromstring(stripped, sep=" ")
            if vals.size != expected:
                raise ValueError(f"{path}:{line_no}: expected {expected} columns, got {vals.size}")
            yield float(vals[0]), vals[1:].reshape((n_carbon, 3))


def compute_histogram(coord_xvg, box_xvg, n_carbon, n_chains, c_per_chain, pairs, edges):
    boxes = parse_box(box_xvg)
    hist = np.zeros(len(edges) - 1, dtype=np.float64)
    nframes = 0
    vol_sum = 0.0
    rmax = edges[-1]
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    for frame_idx, (_time_ps, xyz) in enumerate(iter_coord_frames(coord_xvg, n_carbon)):
        if frame_idx >= len(boxes):
            raise ValueError("box frame count shorter than coordinate frame count")
        box = boxes[frame_idx]
        vol_sum += float(box[0] * box[1] * box[2])
        nframes += 1
        for chain_idx in range(n_chains):
            sub = xyz[chain_idx * c_per_chain : (chain_idx + 1) * c_per_chain, :]
            diff = sub[i_idx, :] - sub[j_idx, :]
            diff -= box * np.rint(diff / box)
            distances = np.sqrt(np.sum(diff * diff, axis=1))
            distances = distances[distances < rmax]
            if distances.size:
                hist += np.histogram(distances, bins=edges)[0]
    if nframes == 0:
        raise ValueError("no coordinate frames parsed")
    return hist / nframes, vol_sum / nframes, nframes


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(x) for x in row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--carbon-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--dt-ps", type=float, default=1000.0)
    parser.add_argument("--n-chains", type=int, default=150)
    parser.add_argument("--rmax", type=float, default=2.0)
    parser.add_argument("--bin", type=float, default=0.01)
    parser.add_argument("--exclude-graph-distance", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nvt = Path(args.case_dir) / "nvt_prod"
    tpr = nvt / "nvt_prod.tpr"
    xtc = nvt / "nvt_prod.xtc"
    if not tpr.exists() or not xtc.exists():
        raise FileNotFoundError(f"missing tpr/xtc under {nvt}")

    carbon_global_atoms = parse_index_atoms(args.carbon_index)
    n_carbon = len(carbon_global_atoms)
    c_per_chain = infer_carbons_per_chain(args.case_id, n_carbon, args.n_chains)
    carbon_local_atoms, cc_bonds = parse_tpr_carbon_bonds(args.gmx, tpr)
    if len(carbon_local_atoms) != c_per_chain:
        raise ValueError(
            f"TPR carbon count per chain ({len(carbon_local_atoms)}) != inferred c_per_chain ({c_per_chain})"
        )
    if c_per_chain * args.n_chains != n_carbon:
        raise ValueError("carbon index count does not match c_per_chain*n_chains")

    pairs, excluded_pairs = local_eligible_pairs(
        carbon_local_atoms,
        cc_bonds,
        max_excluded_distance=args.exclude_graph_distance,
    )

    coord = out / "carbon_coords_dt1000.xvg"
    box = out / "box_dt1000.xvg"
    if args.force or not coord.exists() or not box.exists():
        cmd = [
            args.gmx,
            "traj",
            "-f",
            str(xtc),
            "-s",
            str(tpr),
            "-n",
            str(args.carbon_index),
            "-ox",
            str(coord),
            "-ob",
            str(box),
            "-b",
            "0",
            "-e",
            "999999999",
            "-dt",
            str(args.dt_ps),
            "-xvg",
            "none",
        ]
        run(cmd, stdin="0\n", stdout=str(out / "gmx_traj.out"), stderr=str(out / "gmx_traj.err"))

    r = load_rdf_centers(args.rmax, args.bin)
    edges = rdf_edges_from_centers(r)
    shell_vol = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    unordered_avg, volume_nm3, nframes = compute_histogram(
        coord,
        box,
        n_carbon,
        args.n_chains,
        c_per_chain,
        pairs,
        edges,
    )

    intra_ordered_avg = 2.0 * unordered_avg
    eligible_ordered_pairs_per_chain = 2 * int(pairs.shape[0])
    eligible_partner_density = (eligible_ordered_pairs_per_chain / c_per_chain) / volume_nm3
    denominator = n_carbon * eligible_partner_density * shell_vol
    with np.errstate(divide="ignore", invalid="ignore"):
        g_intra_raw = np.where(denominator > 0, intra_ordered_avg / denominator, 0.0)
    raw_tail = g_intra_raw[(r >= 1.5) & (r <= 2.0)]
    raw_tail_mean = float(np.mean(raw_tail)) if raw_tail.size else math.nan
    if math.isfinite(raw_tail_mean) and raw_tail_mean > 0:
        g_intra = g_intra_raw / raw_tail_mean
        tail_normalization_status = "TAIL_NORMALIZED_1P5_2P0"
    else:
        g_intra = g_intra_raw
        tail_normalization_status = "RAW_FALLBACK_NO_VALID_TAIL"

    long_rows = []
    for vals in zip(r, g_intra, g_intra_raw, intra_ordered_avg):
        long_rows.append(
            [
                args.case_id,
                f"{vals[0]:.8g}",
                f"{vals[1]:.10g}",
                f"{vals[2]:.10g}",
                f"{vals[3]:.10g}",
                eligible_ordered_pairs_per_chain,
                f"{volume_nm3:.10g}",
                nframes,
            ]
        )
    write_csv(
        out / "rdf_cc_intra_nonbonded_long.csv",
        [
            "case_id",
            "r_nm",
            "g_intra_nonbonded",
            "g_intra_nonbonded_raw",
            "intra_nonbonded_ordered_avg",
            "eligible_ordered_pairs_per_chain",
            "volume_nm3_mean",
            "n_frames",
        ],
        long_rows,
    )

    mask_peak = (r >= 0.2) & (r <= 1.2)
    if np.any(mask_peak):
        peak_pos = int(np.argmax(g_intra[mask_peak]))
        peak_r = float(r[mask_peak][peak_pos])
        peak_g = float(g_intra[mask_peak][peak_pos])
    else:
        peak_r = None
        peak_g = None
    first_min_r = find_first_minimum_after_peak(r, g_intra, peak_r)
    tail = g_intra[(r >= 1.5) & (r <= 2.0)]
    summary = {
        "case_id": args.case_id,
        "rdf_kind": "intra_chain_nonbonded_cc",
        "selection": f"same-chain carbon-carbon excluding graph-distance <= {args.exclude_graph_distance}",
        "method": "tail-normalized intrachain nonbonded C-C pair-distribution from gmx traj coordinates + TPR bond graph",
        "n_chains": args.n_chains,
        "n_carbon_total": n_carbon,
        "carbon_per_chain": c_per_chain,
        "n_frames": nframes,
        "dt_ps": args.dt_ps,
        "volume_nm3_mean": volume_nm3,
        "graph_exclusion_max_distance": args.exclude_graph_distance,
        "excluded_unordered_pairs_per_chain": excluded_pairs,
        "eligible_unordered_pairs_per_chain": int(pairs.shape[0]),
        "eligible_ordered_pairs_per_chain": eligible_ordered_pairs_per_chain,
        "normalization": "tail-normalized intrachain pair-distribution descriptor; raw eligible-partner-density curve retained",
        "tail_normalization_status": tail_normalization_status,
        "raw_tail_g_mean_1p5_2p0": raw_tail_mean if math.isfinite(raw_tail_mean) else None,
        "raw_tail_g_std_1p5_2p0": float(np.std(raw_tail)) if raw_tail.size else None,
        "first_peak_r_nm_0p2_1p2": peak_r,
        "first_min_r_nm": first_min_r,
        "first_peak_g_0p2_1p2": peak_g,
        "tail_g_mean_1p5_2p0": float(np.mean(tail)) if tail.size else None,
        "tail_g_std_1p5_2p0": float(np.std(tail)) if tail.size else None,
    }
    (out / "rdf_intra_nonbonded_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{args.case_id} OK frames={nframes} C={n_carbon} C/chain={c_per_chain} "
        f"eligible_pairs={pairs.shape[0]} peak={peak_r},{peak_g}"
    )


if __name__ == "__main__":
    main()
