#!/usr/bin/env python3
"""Compute repeat-unit COM g1 MSD from LCB carbon-skeleton trajectories.

The output is intentionally separate from the existing carbon-atom ``g1``.  It
maps carbon atoms into repeat-unit groups, computes a carbon-skeleton COM for
each repeat unit, and then calculates the multi-time-origin MSD of those
repeat-unit COM positions.
"""

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from trajectory_helpers import resolve_analysis_tpr
except ImportError:  # pragma: no cover
    from .trajectory_helpers import resolve_analysis_tpr  # type: ignore

try:
    from analyze_intrachain_dynamics import (
        G1_G3_TIMEPOINTS,
        ANALYSIS_VERSION,
        case_dir_from_id,
        finite_ratio,
        interp_value,
        local_alpha,
        parse_coord_xvg,
        parse_tpr_topology,
        prepare_nocom,
        resolve_trajectory,
        write_csv,
        write_index,
        write_json,
    )
except ImportError:  # pragma: no cover
    from .analyze_intrachain_dynamics import (  # type: ignore
        G1_G3_TIMEPOINTS,
        ANALYSIS_VERSION,
        case_dir_from_id,
        finite_ratio,
        interp_value,
        local_alpha,
        parse_coord_xvg,
        parse_tpr_topology,
        prepare_nocom,
        resolve_trajectory,
        write_csv,
        write_index,
        write_json,
    )


CASE_DIR_SUFFIX = "_md_newserver_lcb100_manual_20260426"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def shortest_path(adjacency: List[List[int]], start: int, end: int) -> List[int]:
    parent = {start: -1}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for nxt in adjacency[node]:
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    if end not in parent:
        return []
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return list(reversed(path))


def residue_repeat_groups(topology: Dict[str, Any], expected_repeat_units: int) -> List[List[int]]:
    """Return graph-order carbon groups from TPR residue indices if exact."""

    carbon_local = list(topology["carbon_local_atoms"])
    local_to_graph = {local: idx for idx, local in enumerate(carbon_local)}
    records = topology.get("atom_records", [])
    groups: Dict[int, List[int]] = {}
    for record in records:
        if int(record.get("atomnumber", 0)) != 6:
            continue
        resind = int(record.get("resind", -1))
        local = int(record.get("local_index", -1))
        if resind < 0 or local not in local_to_graph:
            continue
        groups.setdefault(resind, []).append(local_to_graph[local])
    ordered = [sorted(groups[key]) for key in sorted(groups)]
    if len(ordered) != expected_repeat_units:
        return []
    if sorted(idx for group in ordered for idx in group) != list(range(len(carbon_local))):
        return []
    return ordered


def graph_repeat_groups(topology: Dict[str, Any], expected_repeat_units: int) -> List[List[int]]:
    """Infer repeat-unit carbon groups from the main backbone carbon path.

    The main path is expected to contain two backbone carbons per repeat unit.
    Side branches are assigned to the nearest backbone carbon's repeat unit.
    """

    adjacency = topology["adjacency"]
    n_carbon = len(adjacency)
    expected_backbone_carbons = 2 * int(expected_repeat_units)
    if expected_backbone_carbons <= 0 or expected_backbone_carbons > n_carbon:
        raise ValueError("invalid expected backbone carbon count")

    candidates: List[List[int]] = []
    first_last = shortest_path(adjacency, 0, n_carbon - 1)
    if len(first_last) == expected_backbone_carbons:
        candidates.append(first_last)
    terminals = [idx for idx, nbrs in enumerate(adjacency) if len(nbrs) == 1]
    for i, start in enumerate(terminals):
        for end in terminals[i + 1 :]:
            path = shortest_path(adjacency, start, end)
            if len(path) == expected_backbone_carbons:
                candidates.append(path)
    if not candidates:
        for start in range(n_carbon):
            for end in range(start + 1, n_carbon):
                path = shortest_path(adjacency, start, end)
                if len(path) == expected_backbone_carbons:
                    candidates.append(path)
    unique = []
    seen = set()
    for path in candidates:
        key = tuple(path)
        rev_key = tuple(reversed(path))
        if key not in seen and rev_key not in seen:
            unique.append(path)
            seen.add(key)
    if not unique:
        raise ValueError(
            f"could not find backbone path with {expected_backbone_carbons} carbons"
        )
    path = unique[0]
    path_set = set(path)
    path_position = {node: idx for idx, node in enumerate(path)}
    groups: List[List[int]] = [path[2 * idx : 2 * idx + 2] for idx in range(expected_repeat_units)]

    for start in range(n_carbon):
        if start in path_set:
            continue
        seen_nodes = {start}
        queue = deque([(start, 0)])
        nearest_path = None
        while queue and nearest_path is None:
            node, _dist = queue.popleft()
            for nxt in adjacency[node]:
                if nxt in seen_nodes:
                    continue
                if nxt in path_set:
                    nearest_path = nxt
                    break
                seen_nodes.add(nxt)
                queue.append((nxt, _dist + 1))
        if nearest_path is None:
            raise ValueError(f"off-backbone carbon {start} is not connected to backbone")
        monomer_idx = min(path_position[nearest_path] // 2, expected_repeat_units - 1)
        groups[monomer_idx].append(start)

    assigned = sorted(idx for group in groups for idx in group)
    if assigned != list(range(n_carbon)):
        raise ValueError("repeat-unit carbon grouping does not cover carbon skeleton exactly once")
    return [sorted(group) for group in groups]


def infer_repeat_groups(
    topology: Dict[str, Any],
    case_metadata: Dict[str, Any],
    mapping_mode: str = "auto",
) -> Tuple[List[List[int]], str]:
    expected = int(case_metadata.get("backbone_repeat_units") or 0)
    if expected <= 0:
        raise ValueError("case metadata lacks positive backbone_repeat_units")
    if mapping_mode in {"auto", "residue"}:
        groups = residue_repeat_groups(topology, expected)
        if groups:
            return groups, "tpr_residue_indices"
        if mapping_mode == "residue":
            raise ValueError("TPR residue indices do not match backbone_repeat_units")
    if mapping_mode in {"auto", "graph"}:
        return graph_repeat_groups(topology, expected), "carbon_graph_backbone_path"
    raise ValueError(f"unknown mapping mode: {mapping_mode}")


def all_atom_repeat_groups(topology: Dict[str, Any], carbon_repeat_groups: List[List[int]]) -> List[List[int]]:
    carbon_local = list(topology["carbon_local_atoms"])
    carbon_graph_to_local = {idx: local for idx, local in enumerate(carbon_local)}
    carbon_local_to_repeat: Dict[int, int] = {}
    for repeat_idx, group in enumerate(carbon_repeat_groups):
        for graph_idx in group:
            carbon_local_to_repeat[carbon_graph_to_local[graph_idx]] = repeat_idx

    atoms_per_chain = int(topology["atoms_per_chain"])
    adjacency = [[] for _ in range(atoms_per_chain)]
    for a, b in topology.get("all_bonds_local", []):
        if 0 <= int(a) < atoms_per_chain and 0 <= int(b) < atoms_per_chain:
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))

    groups: List[List[int]] = [[] for _ in carbon_repeat_groups]
    for local_idx in range(atoms_per_chain):
        if local_idx in carbon_local_to_repeat:
            groups[carbon_local_to_repeat[local_idx]].append(local_idx)
            continue
        seen = {local_idx}
        queue = deque([local_idx])
        assigned = None
        while queue and assigned is None:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt in seen:
                    continue
                if nxt in carbon_local_to_repeat:
                    assigned = carbon_local_to_repeat[nxt]
                    break
                seen.add(nxt)
                queue.append(nxt)
        if assigned is None:
            raise ValueError(f"atom {local_idx} cannot be assigned to a repeat-unit carbon group")
        groups[assigned].append(local_idx)

    assigned_atoms = sorted(idx for group in groups for idx in group)
    if assigned_atoms != list(range(atoms_per_chain)):
        raise ValueError("repeat-unit atom grouping does not cover all atoms exactly once")
    return [sorted(group) for group in groups]


def local_atom_masses(topology: Dict[str, Any]) -> np.ndarray:
    records = topology.get("atom_records", [])
    masses = np.ones(int(topology["atoms_per_chain"]), dtype=float)
    for record in records:
        idx = int(record.get("local_index", -1))
        if 0 <= idx < masses.size:
            masses[idx] = float(record.get("mass", 1.0))
    return masses


def extract_group_coords(gmx_bin, traj, tpr, ndx, out_dir, dt_ps, group_name, expected_atoms, force=False):
    out = Path(out_dir) / f"{group_name}_coords_dt{int(dt_ps)}.xvg"
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    from analyze_intrachain_dynamics import run  # local import keeps script execution fallback simple

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
        stdin=f"{group_name}\n",
        stdout=Path(out_dir) / f"gmx_traj_{group_name}.out",
        stderr=Path(out_dir) / f"gmx_traj_{group_name}.err",
    )
    return out


def monomer_com_coordinates(
    atom_coords: np.ndarray,
    repeat_groups: List[List[int]],
    masses_per_chain: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return [frame, chain*monomer, xyz] repeat-unit COM coordinates."""

    n_frames, n_atom_total, _ = atom_coords.shape
    atoms_per_chain = sum(len(group) for group in repeat_groups)
    if n_atom_total % atoms_per_chain != 0:
        raise ValueError(
            f"coordinate count {n_atom_total} is not divisible by atoms_per_chain {atoms_per_chain}"
        )
    n_chains = n_atom_total // atoms_per_chain
    n_repeat = len(repeat_groups)
    coords = atom_coords.reshape(n_frames, n_chains, atoms_per_chain, 3)
    out = np.empty((n_frames, n_chains * n_repeat, 3), dtype=float)
    masses = masses_per_chain if masses_per_chain is not None else np.ones(atoms_per_chain, dtype=float)
    for repeat_idx, group in enumerate(repeat_groups):
        weights = masses[group].astype(float)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            raise ValueError(f"repeat unit {repeat_idx} has non-positive total mass")
        out[:, repeat_idx::n_repeat, :] = np.sum(coords[:, :, group, :] * weights[None, None, :, None], axis=2) / weight_sum
    return out


def multi_origin_msd(times_ps: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = coords.shape[0]
    lag_times = np.empty(n_frames, dtype=float)
    msd = np.empty(n_frames, dtype=float)
    for lag in range(n_frames):
        if lag == 0:
            lag_times[lag] = 0.0
            msd[lag] = 0.0
            continue
        delta = coords[lag:, :, :] - coords[:-lag, :, :]
        lag_times[lag] = float(np.mean(times_ps[lag:] - times_ps[:-lag]))
        msd[lag] = float(np.mean(np.sum(delta * delta, axis=2)))
    return lag_times, msd


def write_msd(path: Path, times: np.ndarray, values: np.ndarray) -> List[Dict[str, Any]]:
    rows = [
        {"time_ps": float(t), "g1_monomer_com_msd_nm2": float(v)}
        for t, v in zip(times, values)
    ]
    write_csv(path, rows, ["time_ps", "g1_monomer_com_msd_nm2"])
    return rows


def write_warn_summary(
    batch_dir: Path,
    case_id: str,
    error: BaseException,
    *,
    stage: str = "analyze_case",
) -> Dict[str, Any]:
    """Record a per-case WARN so one auxiliary analysis cannot abort a batch."""

    case_out = case_output_dir(batch_dir, case_id)
    case_out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "case_id": case_id,
        "status": "warn",
        "analysis_version": ANALYSIS_VERSION,
        "method": "g1 repeat-unit all-atom COM MSD from nojump/nocom trajectory",
        "warning": "monomer_msd_analysis_failed",
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error)[-2000:],
        "n_frames_g1_monomer": 0,
        "time_first_ps": "",
        "time_last_ps": "",
        "dt_ps": "",
        "g1_monomer_alpha_100_300": "",
        "source_case_dir": "",
    }
    write_json(case_out / "g1_monomer_summary.json", summary)
    write_csv(case_out / "g1_monomer_com.csv", [], ["time_ps", "g1_monomer_com_msd_nm2"])
    return summary


def add_timepoint_metrics(summary: Dict[str, Any], times: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
    for target_ps, label in G1_G3_TIMEPOINTS:
        value = interp_value(times, values, target_ps)
        summary[f"g1_monomer_{label}_nm2"] = value
    return summary


def add_ratio_metrics(summary: Dict[str, Any], g3_summary: Dict[str, Any]) -> Dict[str, Any]:
    for _target_ps, label in G1_G3_TIMEPOINTS:
        g1_value = summary.get(f"g1_monomer_{label}_nm2")
        g3_value = g3_summary.get(f"g3_{label}_nm2")
        summary[f"g1_monomer_over_g3_{label}"] = finite_ratio(
            float(g1_value) if g1_value not in {"", None} else math.nan,
            float(g3_value) if g3_value not in {"", None} else math.nan,
        )
    return summary


def case_output_dir(batch_dir: Path, case_id: str) -> Path:
    return Path(batch_dir) / "cases" / case_id / "intrachain"


def analyze_case(
    case_id: str,
    input_root: Path,
    batch_dir: Path,
    gmx_bin: str,
    dt_ps: float,
    force: bool = False,
    mapping_mode: str = "auto",
    keep_intermediates: bool = False,
    trajectory_end_ps: float | None = None,
) -> Dict[str, Any]:
    case_input = case_dir_from_id(input_root, case_id)
    nvt = case_input / "nvt_prod"
    case_out = case_output_dir(batch_dir, case_id)
    case_out.mkdir(parents=True, exist_ok=True)
    tpr = resolve_analysis_tpr(gmx_bin, case_input, nvt, force=force)
    topology = parse_tpr_topology(gmx_bin, tpr)
    metadata = read_json(case_input / "case.json")
    carbon_repeat_groups, mapping_method = infer_repeat_groups(topology, metadata, mapping_mode=mapping_mode)
    repeat_groups = all_atom_repeat_groups(topology, carbon_repeat_groups)
    ndx = write_index(case_out / "monomer_msd.ndx", topology)
    traj = resolve_trajectory(
        gmx_bin,
        nvt,
        case_out,
        force=force,
        max_time_ps=trajectory_end_ps,
    )
    nojump, nocom = prepare_nocom(gmx_bin, traj, tpr, ndx, case_out, force=force)
    coords_xvg = extract_group_coords(
        gmx_bin,
        nocom,
        tpr,
        ndx,
        case_out,
        dt_ps,
        "chain_all",
        expected_atoms=len(topology["all_atoms"]),
        force=force,
    )
    times, atom_coords = parse_coord_xvg(coords_xvg, expected_atoms=len(topology["all_atoms"]))
    monomer_coords = monomer_com_coordinates(atom_coords, repeat_groups, masses_per_chain=local_atom_masses(topology))
    lag_times, msd = multi_origin_msd(times, monomer_coords)
    write_msd(case_out / "g1_monomer_com.csv", lag_times, msd)

    summary: Dict[str, Any] = {
        "case_id": case_id,
        "status": "ok",
        "analysis_version": ANALYSIS_VERSION,
        "method": "g1 repeat-unit all-atom COM MSD from nojump/nocom trajectory",
        "monomer_mapping_method": mapping_method,
        "n_chains": int(topology["n_chains"]),
        "n_frames_g1_monomer": int(len(lag_times)),
        "time_first_ps": float(lag_times[0]) if len(lag_times) else "",
        "time_last_ps": float(lag_times[-1]) if len(lag_times) else "",
        "dt_ps": float(dt_ps),
        "carbon_per_chain": int(topology["carbon_per_chain"]),
        "backbone_repeat_units": int(metadata.get("backbone_repeat_units") or len(repeat_groups)),
        "monomers_per_chain": int(len(repeat_groups)),
        "g1_monomer_alpha_100_300": local_alpha(lag_times, msd),
        "source_case_dir": str(case_input.resolve()),
    }
    add_timepoint_metrics(summary, lag_times, msd)
    g3_summary_path = case_out / "g1_g3_summary.json"
    if g3_summary_path.exists():
        add_ratio_metrics(summary, read_json(g3_summary_path))
    write_json(case_out / "g1_monomer_summary.json", summary)
    write_json(
        case_out / "monomer_mapping.json",
        {
            "case_id": case_id,
            "mapping_method": mapping_method,
            "backbone_repeat_units": summary["backbone_repeat_units"],
            "monomers_per_chain": summary["monomers_per_chain"],
            "carbon_per_chain": summary["carbon_per_chain"],
            "carbon_repeat_groups_graph_order_0based": carbon_repeat_groups,
            "repeat_groups_local_atom_indices_0based": repeat_groups,
            "repeat_groups_local_carbon_atom_indices_0based": [
                [topology["carbon_local_atoms"][idx] for idx in group] for group in carbon_repeat_groups
            ],
        },
    )
    if not keep_intermediates:
        for path in [coords_xvg, nojump, nocom, case_out / "nvt_prod.full.xtc"]:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
    return summary


def aggregate(batch_dir: Path, cases: List[str]) -> None:
    batch = Path(batch_dir)
    summaries = []
    long_rows = []
    for case_id in cases:
        case_out = case_output_dir(batch, case_id)
        summary_path = case_out / "g1_monomer_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        g3_summary_path = case_out / "g1_g3_summary.json"
        if g3_summary_path.exists():
            add_ratio_metrics(summary, read_json(g3_summary_path))
            write_json(summary_path, summary)
        summaries.append(summary)
        for row in read_csv_rows(case_out / "g1_monomer_com.csv"):
            row["case_id"] = case_id
            long_rows.append(row)
    summaries.sort(key=lambda row: row["case_id"])
    long_rows.sort(key=lambda row: (row["case_id"], float(row["time_ps"])))
    agg = batch / "aggregate"
    summary_fields = [
        "case_id",
        "status",
        "analysis_version",
        "method",
        "monomer_mapping_method",
        "n_chains",
        "n_frames_g1_monomer",
        "time_first_ps",
        "time_last_ps",
        "dt_ps",
        "carbon_per_chain",
        "backbone_repeat_units",
        "monomers_per_chain",
        "g1_monomer_1ns_nm2",
        "g1_monomer_10ns_nm2",
        "g1_monomer_50ns_nm2",
        "g1_monomer_100ns_nm2",
        "g1_monomer_alpha_100_300",
        "g1_monomer_over_g3_1ns",
        "g1_monomer_over_g3_10ns",
        "g1_monomer_over_g3_50ns",
        "g1_monomer_over_g3_100ns",
        "source_case_dir",
    ]
    write_csv(agg / "g1_monomer_summary.csv", summaries, summary_fields)
    write_csv(
        agg / "g1_monomer_long.csv",
        long_rows,
        ["case_id", "time_ps", "g1_monomer_com_msd_nm2"],
    )
    write_json(
        agg / "g1_monomer_manifest.json",
        {
            "analysis_version": ANALYSIS_VERSION,
            "method": "repeat-unit all-atom COM MSD; graph/residue mapping fails instead of falling back to atom g1",
            "case_count": len(summaries),
            "recommended_fields": [
                "g1_monomer_1ns_nm2",
                "g1_monomer_10ns_nm2",
                "g1_monomer_over_g3_10ns",
                "g1_monomer_alpha_100_300",
            ],
        },
    )


def read_cases_from_file(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("${PELCB100_REMOTE_ROOT}/lcb_rg_20260501/input"))
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--gmx", default="${PELCB100_REMOTE_ROOT}/gmx2026_node8/bin/gmx")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--dt-ps", type=float, default=1000.0)
    parser.add_argument(
        "--trajectory-end-ps",
        type=float,
        default=None,
        help="Optional inclusive trajectory cutoff in ps, used to enforce a common sampling horizon.",
    )
    parser.add_argument("--mapping-mode", choices=["auto", "residue", "graph"], default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Write a per-case WARN summary and continue when an auxiliary "
            "monomer-MSD extraction fails. Intended for large postprocess batches."
        ),
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = list(args.cases)
    if args.cases_file:
        cases.extend(read_cases_from_file(args.cases_file))
    cases = sorted(dict.fromkeys(cases))
    if not cases:
        raise SystemExit("No cases supplied; pass --cases or --cases-file")
    if not args.aggregate_only:
        for case_id in cases:
            try:
                analyze_case(
                    case_id,
                    args.input_root,
                    args.batch_dir,
                    args.gmx,
                    args.dt_ps,
                    force=args.force,
                    mapping_mode=args.mapping_mode,
                    keep_intermediates=args.keep_intermediates,
                    trajectory_end_ps=args.trajectory_end_ps,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                summary = write_warn_summary(args.batch_dir, case_id, exc)
                print(
                    f"WARN {case_id} {summary['warning']}: "
                    f"{summary['error_type']}: {summary['error_message']}",
                    flush=True,
                )
    if not args.no_aggregate:
        aggregate(args.batch_dir, cases)


if __name__ == "__main__":
    main()

