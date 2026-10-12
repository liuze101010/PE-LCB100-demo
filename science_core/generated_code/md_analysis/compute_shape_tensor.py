#!/usr/bin/env python3
"""Compute strict carbon-skeleton gyration-tensor descriptors for LCB cases.

This script uses carbon coordinates, not gmx polystat molecule averages. It
also derives the carbon-graph diameter endpoints from the TPR bond graph and
measures their spatial distance from the same coordinate frames.
"""

import argparse
import csv
import json
import math
import re
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

try:
    from trajectory_helpers import resolve_analysis_tpr
except ImportError:  # pragma: no cover
    from .trajectory_helpers import resolve_analysis_tpr  # type: ignore


ANALYSIS_VERSION = "canonical_v2"


def run(cmd, stdin=None, stdout=None, stderr=None):
    stdout_handle = open(stdout, "w") if stdout else subprocess.PIPE
    stderr_handle = open(stderr, "w") if stderr else subprocess.PIPE
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            input=stdin,
            universal_newlines=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    finally:
        if stdout:
            stdout_handle.close()
        if stderr:
            stderr_handle.close()
    if proc.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(str(part) for part in cmd))
    return proc


def capture(cmd):
    proc = subprocess.run([str(part) for part in cmd], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or ("command failed: " + " ".join(str(part) for part in cmd)))
    return proc.stdout


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


def parse_gro_atoms(gro_path):
    lines = Path(gro_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    atom_count = int(lines[1].strip())
    atoms = []
    for idx, line in enumerate(lines[2 : 2 + atom_count], start=1):
        atom_name = line[10:15].strip()
        atom_number_text = line[15:20].strip()
        if not atom_name or not atom_number_text:
            parts = line.split()
            atom_name = parts[1] if len(parts) > 1 else ""
            atom_number_text = parts[2] if len(parts) > 2 else str(idx)
        try:
            atom_number = int(atom_number_text)
        except ValueError:
            atom_number = idx
        atoms.append({"atom_number": atom_number, "atom_name": atom_name})
    return atoms


def write_index(path, atoms, chain_count):
    if len(atoms) % chain_count != 0:
        raise ValueError(f"atom count {len(atoms)} is not divisible by chain_count={chain_count}")
    atoms_per_chain = len(atoms) // chain_count
    all_atoms = [atom["atom_number"] for atom in atoms]
    carbon_atoms = [atom["atom_number"] for atom in atoms if str(atom["atom_name"]).upper().startswith("C")]
    if len(carbon_atoms) % chain_count != 0:
        raise ValueError(f"carbon count {len(carbon_atoms)} is not divisible by chain_count={chain_count}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        write_group(handle, "chain_all", all_atoms)
        write_group(handle, "carbon_atoms", carbon_atoms)
    return {
        "path": path,
        "atom_count": len(atoms),
        "atoms_per_chain": atoms_per_chain,
        "carbon_count": len(carbon_atoms),
        "carbon_per_chain": len(carbon_atoms) // chain_count,
    }


def write_group(handle, name, atom_numbers):
    handle.write(f"[ {name} ]\n")
    for idx in range(0, len(atom_numbers), 15):
        handle.write(" ".join(str(num) for num in atom_numbers[idx : idx + 15]) + "\n")
    handle.write("\n")


def parse_tpr_carbon_graph(gmx_bin, tpr_path):
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
    if not carbon_atoms or not cc_bonds:
        raise ValueError("failed to parse carbon graph from TPR")
    carbon_atoms = sorted(carbon_atoms)
    carbon_to_order = {atom: idx for idx, atom in enumerate(carbon_atoms)}
    adjacency = [[] for _ in carbon_atoms]
    for a, b in cc_bonds:
        if a in carbon_to_order and b in carbon_to_order:
            ia, ib = carbon_to_order[a], carbon_to_order[b]
            adjacency[ia].append(ib)
            adjacency[ib].append(ia)
    component = graph_component(adjacency, 0)
    if len(component) != len(carbon_atoms):
        raise ValueError(f"carbon graph is disconnected: component={len(component)} total={len(carbon_atoms)}")
    dia_a, dia_b, dia_edges = graph_diameter(adjacency)
    return {
        "carbon_local_atoms": carbon_atoms,
        "cc_bonds": cc_bonds,
        "diameter_order_endpoints": (dia_a, dia_b),
        "diameter_local_atom_indices": (carbon_atoms[dia_a], carbon_atoms[dia_b]),
        "diameter_edges": dia_edges,
        "diameter_topological_carbons": dia_edges + 1,
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


def farthest_from(adjacency, start):
    parent = {start: None}
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in dist:
                parent[nxt] = node
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    farthest = max(dist, key=lambda item: dist[item])
    return farthest, dist[farthest], parent


def graph_diameter(adjacency):
    a, _dist, _parent = farthest_from(adjacency, 0)
    b, dist, _parent = farthest_from(adjacency, a)
    return a, b, dist


def trajectory_time_bounds(gmx_bin, traj_path):
    proc = subprocess.run([gmx_bin, "check", "-f", str(traj_path)], universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = proc.stdout + "\n" + proc.stderr
    times = []
    for match in re.finditer(r"\btime\s+([-+]?\d+(?:\.\d*)?)", text):
        times.append(float(match.group(1)))
    if not times:
        return None
    return {"first_ps": min(times), "last_ps": max(times)}


def resolve_trajectory(gmx_bin, nvt_dir, out_dir):
    nvt_dir = Path(nvt_dir)
    full = nvt_dir / "nvt_prod.full.xtc"
    base = nvt_dir / "nvt_prod.xtc"
    parts = sorted(nvt_dir.glob("nvt_prod.part*.xtc"))
    if full.exists() and full.stat().st_size > 0:
        return full
    if parts and base.exists():
        base_bounds = trajectory_time_bounds(gmx_bin, base)
        part_bounds = [trajectory_time_bounds(gmx_bin, part) for part in parts]
        part_bounds = [item for item in part_bounds if item]
        if base_bounds and part_bounds:
            if base_bounds["last_ps"] >= max(item["last_ps"] for item in part_bounds) - 1.0:
                return base
        merged = Path(out_dir) / "nvt_prod.full.xtc"
        if not merged.exists() or merged.stat().st_size == 0:
            run([gmx_bin, "trjcat", "-f", base, *parts, "-o", merged], stdout=Path(out_dir) / "gmx_trjcat.out", stderr=Path(out_dir) / "gmx_trjcat.err")
        return merged
    if base.exists() and base.stat().st_size > 0:
        return base
    raise FileNotFoundError(f"no usable trajectory in {nvt_dir}")


def extract_reference_gro(gmx_bin, tpr_path, traj_path, out_dir, force=False):
    """Extract a reference GRO when staged input lacks nvt_prod.gro."""
    out_dir = Path(out_dir)
    out = out_dir / "reference_from_xtc.gro"
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    run(
        [gmx_bin, "trjconv", "-s", str(tpr_path), "-f", str(traj_path), "-o", str(out), "-dump", "0"],
        stdin="0\n",
        stdout=out_dir / "gmx_trjconv_reference_gro.out",
        stderr=out_dir / "gmx_trjconv_reference_gro.err",
    )
    return out


def export_carbon_coords(gmx_bin, traj, tpr, index_path, out_dir, dt_ps, force=False):
    out_dir = Path(out_dir)
    nojump = out_dir / "nvt_prod_nojump.xtc"
    coord = out_dir / f"carbon_coords_dt{int(dt_ps)}.xvg"
    if force or not nojump.exists() or nojump.stat().st_size == 0:
        run(
            [gmx_bin, "trjconv", "-f", traj, "-s", tpr, "-n", index_path, "-o", nojump, "-pbc", "nojump"],
            stdin="chain_all\n",
            stdout=out_dir / "gmx_trjconv_nojump.out",
            stderr=out_dir / "gmx_trjconv_nojump.err",
        )
    if force or not coord.exists() or coord.stat().st_size == 0:
        run(
            [
                gmx_bin,
                "traj",
                "-f",
                nojump,
                "-s",
                tpr,
                "-n",
                index_path,
                "-ox",
                coord,
                "-b",
                "0",
                "-e",
                "999999999",
                "-dt",
                str(dt_ps),
                "-xvg",
                "none",
            ],
            stdin="carbon_atoms\n",
            stdout=out_dir / "gmx_traj_carbon.out",
            stderr=out_dir / "gmx_traj_carbon.err",
        )
    return coord, nojump


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


def shape_stats_from_coords(coord_xvg, n_chains, c_per_chain, diameter_pair):
    frame_rows = []
    chain_sum = {}
    nframes = 0
    n_bad_order = 0
    n_bad_negative = 0
    dia_i, dia_j = diameter_pair
    for time_ps, xyz in iter_coord_frames(coord_xvg, n_chains * c_per_chain):
        nframes += 1
        frame_values = []
        dia_values = []
        dia2_values = []
        for chain_idx in range(n_chains):
            sub = xyz[chain_idx * c_per_chain : (chain_idx + 1) * c_per_chain]
            center = np.mean(sub, axis=0)
            centered = sub - center
            tensor = centered.T @ centered / float(c_per_chain)
            eig = np.linalg.eigvalsh(tensor)[::-1]
            if not (eig[0] + 1e-10 >= eig[1] >= eig[2] - 1e-10):
                n_bad_order += 1
            if eig[2] < -1e-8:
                n_bad_negative += 1
            eig = np.maximum(eig, 0.0)
            rg2 = float(np.sum(eig))
            rg = math.sqrt(rg2)
            mean_lambda = rg2 / 3.0
            asphericity = float(eig[0] - 0.5 * (eig[1] + eig[2]))
            acylindricity = float(eig[1] - eig[2])
            rsa = float(1.5 * np.sum((eig - mean_lambda) ** 2) / (rg2 ** 2)) if rg2 > 0 else math.nan
            dia = float(np.linalg.norm(sub[dia_i] - sub[dia_j]))
            dia2 = dia * dia
            span_ratio = dia2 / (6.0 * rg2) if rg2 > 0 else math.nan
            frame_values.append((rg, rg2, eig[0], eig[1], eig[2], asphericity, acylindricity, rsa))
            dia_values.append(dia)
            dia2_values.append(dia2)
            acc = chain_sum.setdefault(chain_idx + 1, [])
            acc.append((rg, rg2, eig[0], eig[1], eig[2], asphericity, acylindricity, rsa, dia, dia2, span_ratio))
        arr = np.asarray(frame_values, dtype=float)
        rg2_mean = float(np.mean(arr[:, 1]))
        dia2_mean = float(np.mean(dia2_values))
        frame_rows.append(
            {
                "time_ps": time_ps,
                "rg_carbon_mean_nm": float(np.mean(arr[:, 0])),
                "rg2_carbon_mean_nm2": rg2_mean,
                "lambda1_mean_nm2": float(np.mean(arr[:, 2])),
                "lambda2_mean_nm2": float(np.mean(arr[:, 3])),
                "lambda3_mean_nm2": float(np.mean(arr[:, 4])),
                "asphericity_mean_nm2": float(np.mean(arr[:, 5])),
                "acylindricity_mean_nm2": float(np.mean(arr[:, 6])),
                "relative_shape_anisotropy_mean": float(np.mean(arr[:, 7])),
                "graph_diameter_distance_mean_nm": float(np.mean(dia_values)),
                "graph_diameter_distance2_mean_nm2": dia2_mean,
                "graph_diameter2_over_6rg2": dia2_mean / (6.0 * rg2_mean) if rg2_mean > 0 else math.nan,
            }
        )
    chain_rows = []
    for chain_idx, values in sorted(chain_sum.items()):
        arr = np.asarray(values, dtype=float)
        chain_rows.append(
            {
                "chain": chain_idx,
                "rg_carbon_mean_nm": float(np.mean(arr[:, 0])),
                "rg2_carbon_mean_nm2": float(np.mean(arr[:, 1])),
                "lambda1_mean_nm2": float(np.mean(arr[:, 2])),
                "lambda2_mean_nm2": float(np.mean(arr[:, 3])),
                "lambda3_mean_nm2": float(np.mean(arr[:, 4])),
                "asphericity_mean_nm2": float(np.mean(arr[:, 5])),
                "acylindricity_mean_nm2": float(np.mean(arr[:, 6])),
                "relative_shape_anisotropy_mean": float(np.mean(arr[:, 7])),
                "graph_diameter_distance_mean_nm": float(np.mean(arr[:, 8])),
                "graph_diameter_distance2_mean_nm2": float(np.mean(arr[:, 9])),
                "graph_diameter2_over_6rg2": float(np.mean(arr[:, 9])) / (6.0 * float(np.mean(arr[:, 1])))
                if float(np.mean(arr[:, 1])) > 0
                else math.nan,
            }
        )
    return frame_rows, chain_rows, {"n_frames": nframes, "n_bad_order": n_bad_order, "n_bad_negative": n_bad_negative}


def summarize_column(rows, key):
    arr = np.asarray([float(row[key]) for row in rows if row.get(key) not in ("", None)], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--case-dir", required=True, help="Case root with nvt_prod child.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--n-chains", type=int, default=150)
    parser.add_argument("--dt-ps", type=float, default=100.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nvt = Path(args.case_dir) / "nvt_prod"
    tpr = resolve_analysis_tpr(args.gmx, Path(args.case_dir), nvt, force=args.force)
    gro = nvt / "nvt_prod.gro"
    if not gro.exists():
        candidates = sorted(nvt.glob("nvt_prod.part*.gro"), reverse=True)
        if candidates:
            gro = candidates[0]
    traj = None
    if not gro.exists():
        traj = resolve_trajectory(args.gmx, nvt, out)
        gro = extract_reference_gro(args.gmx, tpr, traj, out, force=args.force)
    atoms = parse_gro_atoms(gro)
    index_meta = write_index(out / "shape_index.ndx", atoms, args.n_chains)
    graph = parse_tpr_carbon_graph(args.gmx, tpr)
    if len(graph["carbon_local_atoms"]) != index_meta["carbon_per_chain"]:
        raise ValueError(
            f"TPR carbon per chain {len(graph['carbon_local_atoms'])} != GRO carbon per chain {index_meta['carbon_per_chain']}"
        )
    if traj is None:
        traj = resolve_trajectory(args.gmx, nvt, out)
    coord, nojump = export_carbon_coords(args.gmx, traj, tpr, index_meta["path"], out, args.dt_ps, force=args.force)
    frame_rows, chain_rows, validation = shape_stats_from_coords(
        coord,
        args.n_chains,
        index_meta["carbon_per_chain"],
        graph["diameter_order_endpoints"],
    )

    times = [row["time_ps"] for row in frame_rows]
    dt = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else ""
    summary = {
        "case_id": args.case_id,
        "status": "ok",
        "analysis_version": ANALYSIS_VERSION,
        "method": "strict carbon-skeleton gyration tensor from nojump trajectory coordinates",
        "n_chains": args.n_chains,
        "n_frames": len(frame_rows),
        "time_first_ps": times[0] if times else "",
        "time_last_ps": times[-1] if times else "",
        "dt_ps": dt,
        "atom_count": index_meta["atom_count"],
        "atoms_per_chain": index_meta["atoms_per_chain"],
        "carbon_count": index_meta["carbon_count"],
        "carbon_per_chain": index_meta["carbon_per_chain"],
        "carbon_graph_connected": True,
        "graph_diameter_local_atom_indices_0based": list(graph["diameter_local_atom_indices"]),
        "graph_diameter_order_endpoints": list(graph["diameter_order_endpoints"]),
        "graph_diameter_topological_edges": graph["diameter_edges"],
        "graph_diameter_topological_carbons": graph["diameter_topological_carbons"],
        "validation": validation,
        "source_case_dir": str(Path(args.case_dir).resolve()),
    }
    for key in [
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
    ]:
        mean, std = summarize_column(frame_rows, key)
        summary[key] = mean
        summary[key.replace("_mean", "_std_time")] = std
    rg2_mean = summary.get("rg2_carbon_mean_nm2", math.nan)
    dia2_mean = summary.get("graph_diameter_distance2_mean_nm2", math.nan)
    summary["graph_diameter2_over_6rg2"] = dia2_mean / (6.0 * rg2_mean) if rg2_mean and rg2_mean > 0 else math.nan

    write_csv(
        out / "shape_timeseries.csv",
        frame_rows,
        [
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
        ],
    )
    write_csv(
        out / "shape_chains.csv",
        chain_rows,
        [
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
        ],
    )
    write_json(out / "shape_summary.json", summary)
    if not args.keep_intermediates:
        for path in (coord, nojump):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
    print(f"{args.case_id} OK frames={len(frame_rows)} carbon_per_chain={index_meta['carbon_per_chain']}")


if __name__ == "__main__":
    main()
