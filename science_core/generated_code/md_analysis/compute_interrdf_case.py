#!/usr/bin/env python3
"""Compute inter-chain C-C RDF from total C-C RDF and per-chain coordinates."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np


def run(cmd, stdin=None, stdout=None, stderr=None):
    with open(stdout, "w") if stdout else subprocess.DEVNULL as out, open(stderr, "w") if stderr else subprocess.DEVNULL as err:
        proc = subprocess.run(cmd, input=stdin, universal_newlines=True, stdout=out, stderr=err)
    if proc.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(str(x) for x in cmd))


def parse_index_atoms(path):
    atoms = []
    in_group = False
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("["):
            group = stripped.strip("[] ").strip().lower()
            in_group = group in {"carbon", "carbon_atoms", "c*"} or "carbon" in group
            continue
        if in_group:
            atoms.extend(int(x) for x in stripped.split())
    if not atoms:
        raise ValueError(f"no carbon atoms found in index: {path}")
    return atoms


def infer_carbons_per_chain(case_id, n_carbon, n_chains):
    match = re.search(r"_c(\d+)", case_id)
    if match:
        c_per_chain = int(match.group(1))
        if c_per_chain * n_chains == n_carbon:
            return c_per_chain
    if n_carbon % n_chains != 0:
        raise ValueError(f"carbon count {n_carbon} not divisible by n_chains={n_chains}")
    return n_carbon // n_chains


def load_rdf(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 10:
        raise ValueError(f"bad RDF file: {path}")
    return arr[:, 0], arr[:, 1]


def rdf_edges_from_centers(r):
    r = np.asarray(r, dtype=float)
    dr = float(np.median(np.diff(r)))
    mids = (r[:-1] + r[1:]) / 2.0
    edges = np.empty(r.size + 1, dtype=float)
    edges[0] = max(0.0, r[0] - dr / 2.0)
    edges[1:-1] = mids
    edges[-1] = r[-1] + dr / 2.0
    return edges


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
    with Path(path).open("r") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "@")):
                continue
            vals = np.fromstring(stripped, sep=" ")
            if vals.size != expected:
                raise ValueError(f"{path}:{line_no}: expected {expected} columns, got {vals.size}")
            yield float(vals[0]), vals[1:].reshape((n_carbon, 3))


def compute_intra_hist(coord_xvg, box_xvg, n_carbon, n_chains, c_per_chain, edges):
    boxes = parse_box(box_xvg)
    hist = np.zeros(len(edges) - 1, dtype=np.float64)
    iu = np.triu_indices(c_per_chain, k=1)
    nframes = 0
    vol_sum = 0.0
    rmax = edges[-1]
    for frame_idx, (_time_ps, xyz) in enumerate(iter_coord_frames(coord_xvg, n_carbon)):
        if frame_idx >= len(boxes):
            raise ValueError("box frame count shorter than coordinate frame count")
        box = boxes[frame_idx]
        vol_sum += float(box[0] * box[1] * box[2])
        nframes += 1
        for chain_idx in range(n_chains):
            sub = xyz[chain_idx * c_per_chain : (chain_idx + 1) * c_per_chain, :]
            diff = sub[:, None, :] - sub[None, :, :]
            diff -= box * np.rint(diff / box)
            distances = np.sqrt(np.sum(diff[iu] * diff[iu], axis=1))
            distances = distances[distances < rmax]
            if distances.size:
                hist += np.histogram(distances, bins=edges)[0]
    if nframes == 0:
        raise ValueError("no coordinate frames parsed")
    return hist / nframes, vol_sum / nframes, nframes


def find_first_minimum_after_peak(
    r_grid,
    g_r,
    peak_r,
    min_separation_nm=0.10,
    search_low_nm=0.65,
    search_high_nm=0.90,
):
    r = np.asarray(r_grid, dtype=float)
    g = np.asarray(g_r, dtype=float)
    if r.size < 3 or g.size != r.size or peak_r is None:
        return None
    peak = float(peak_r)
    if not math.isfinite(peak):
        return None
    finite = np.isfinite(r) & np.isfinite(g)
    r = r[finite]
    g = g[finite]
    if r.size < 3:
        return None
    low = max(peak + float(min_separation_nm), float(search_low_nm))
    high = float(search_high_nm)
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        return None
    window = (r >= low) & (r <= high)
    if int(np.count_nonzero(window)) < 3:
        return None
    r_w = r[window]
    g_w = g[window]
    return float(r_w[int(np.argmin(g_w))])


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(x) for x in row) + "\n")


def compute_inter_rdf(
    case_id,
    case_dir,
    rdf_all,
    carbon_index,
    out_dir,
    gmx,
    dt_ps=1000.0,
    n_chains=150,
    force=False,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nvt = Path(case_dir) / "nvt_prod"
    tpr = nvt / "nvt_prod.tpr"
    xtc = nvt / "nvt_prod.xtc"
    if not tpr.exists() or not xtc.exists():
        raise FileNotFoundError(f"missing tpr/xtc under {nvt}")

    atoms = parse_index_atoms(carbon_index)
    n_carbon = len(atoms)
    c_per_chain = infer_carbons_per_chain(case_id, n_carbon, n_chains)

    coord = out / "carbon_coords_dt1000.xvg"
    box = out / "box_dt1000.xvg"
    if force or not coord.exists() or not box.exists():
        cmd = [
            gmx,
            "traj",
            "-f",
            str(xtc),
            "-s",
            str(tpr),
            "-n",
            str(carbon_index),
            "-ox",
            str(coord),
            "-ob",
            str(box),
            "-b",
            "0",
            "-e",
            "999999999",
            "-dt",
            str(dt_ps),
            "-xvg",
            "none",
        ]
        run(cmd, stdin="0\n", stdout=str(out / "gmx_traj.out"), stderr=str(out / "gmx_traj.err"))

    r, g_all = load_rdf(rdf_all)
    edges = rdf_edges_from_centers(r)
    shell_vol = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    intra_unordered_avg, volume_nm3, nframes = compute_intra_hist(
        coord,
        box,
        n_carbon,
        n_chains,
        c_per_chain,
        edges,
    )

    rho_total = n_carbon / volume_nm3
    rho_inter = (n_carbon - c_per_chain) / volume_nm3
    total_ordered_avg = g_all * n_carbon * rho_total * shell_vol
    intra_ordered_avg = 2.0 * intra_unordered_avg
    inter_ordered_avg = np.maximum(total_ordered_avg - intra_ordered_avg, 0.0)
    g_inter = inter_ordered_avg / (n_carbon * rho_inter * shell_vol)

    write_csv(
        out / "rdf_cc_inter_long.csv",
        ["case_id", "r_nm", "g_inter", "g_all", "intra_ordered_avg", "total_ordered_avg", "inter_ordered_avg"],
        [
            [
                case_id,
                f"{vals[0]:.8g}",
                f"{vals[1]:.10g}",
                f"{vals[2]:.10g}",
                f"{vals[3]:.10g}",
                f"{vals[4]:.10g}",
                f"{vals[5]:.10g}",
            ]
            for vals in zip(r, g_inter, g_all, intra_ordered_avg, total_ordered_avg, inter_ordered_avg)
        ],
    )

    mask_peak = (r >= 0.2) & (r <= 1.2)
    if np.any(mask_peak):
        peak_idx = int(np.argmax(g_inter[mask_peak]))
        peak_r = float(r[mask_peak][peak_idx])
        peak_g = float(g_inter[mask_peak][peak_idx])
    else:
        peak_r = None
        peak_g = None
    first_min_r = find_first_minimum_after_peak(r, g_inter, peak_r)
    tail = g_inter[(r >= 1.5) & (r <= 2.0)]
    summary = {
        "case_id": case_id,
        "rdf_kind": "inter_chain_cc",
        "selection": "inter-chain carbon-carbon",
        "method": "inter-chain C-C RDF by subtracting same-chain contribution from total C-C RDF",
        "n_chains": n_chains,
        "n_carbon_total": n_carbon,
        "carbon_per_chain": c_per_chain,
        "n_frames": nframes,
        "dt_ps": dt_ps,
        "volume_nm3_mean": volume_nm3,
        "rho_total_C_per_nm3": rho_total,
        "rho_inter_C_per_nm3": rho_inter,
        "first_peak_r_nm_0p2_1p2": peak_r,
        "first_min_r_nm": first_min_r,
        "first_peak_g_0p2_1p2": peak_g,
        "tail_g_mean_1p5_2p0": float(np.mean(tail)) if tail.size else None,
        "tail_g_std_1p5_2p0": float(np.std(tail)) if tail.size else None,
    }
    (out / "rdf_inter_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--rdf-all", required=True)
    parser.add_argument("--carbon-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--dt-ps", type=float, default=1000.0)
    parser.add_argument("--n-chains", type=int, default=150)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = compute_inter_rdf(
        args.case_id,
        args.case_dir,
        args.rdf_all,
        args.carbon_index,
        args.out_dir,
        args.gmx,
        dt_ps=args.dt_ps,
        n_chains=args.n_chains,
        force=args.force,
    )
    print(
        f"{args.case_id} OK frames={summary['n_frames']} "
        f"C={summary['n_carbon_total']} C/chain={summary['carbon_per_chain']} "
        f"peak={summary['first_peak_r_nm_0p2_1p2']},{summary['first_peak_g_0p2_1p2']}"
    )


if __name__ == "__main__":
    main()
