#!/usr/bin/env python3
"""Compute the complete PE-LCB100 RDF suite for one completed case.

The formal RDF observable contains three C-C RDF definitions plus one
intrachain pair-distribution descriptor: all pairs, GROMACS exclusions,
inter-chain pairs, and tail-normalized same-chain nonbonded pairs. Downstream
dynamics uses the inter-chain first peak for Fs(Q,t), while contact probability
uses the excluded all-carbon first minimum as its primary RDF-derived cutoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows local smoke support
    fcntl = None

from canonical_outputs import ANALYSIS_VERSION, RDF_FIELDS, aggregate_dir, batch_dir, case_dir, default_date_tag
from compute_interrdf_case import compute_inter_rdf
from trajectory_helpers import (
    find_first_minimum_after_peak,
    find_structure_gro,
    infer_case_id,
    infer_chain_layout,
    load_case_metadata,
    parse_gro_atoms,
    parse_xvg_numeric,
    resolve_analysis_tpr,
    resolve_case_root,
    resolve_nvt_dir,
    resolve_trajectory,
    run_command,
    validate_gromacs_binary,
    write_index_file,
    write_json,
)


REQUIRED_RDF_KINDS = ("all", "excl", "inter_chain_cc", "intra_chain_nonbonded_cc")
RDF_LONG_FILES = (
    "rdf_cc_all_long.csv",
    "rdf_cc_excl_long.csv",
    "rdf_cc_inter_long.csv",
    "rdf_cc_intra_nonbonded_long.csv",
)
SELECTIONS = {
    "all": "carbon-carbon all pairs",
    "excl": "carbon-carbon excluding GROMACS bonded exclusions",
    "inter_chain_cc": "inter-chain carbon-carbon",
    "intra_chain_nonbonded_cc": "tail-normalized same-chain carbon-carbon pair distribution excluding graph-distance <= 3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--gmx-bin", default="gmx")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--batch-date", default=None)
    parser.add_argument("--chain-count", type=int, default=150)
    parser.add_argument("--dt-ps", type=float, default=1000.0)
    parser.add_argument("--rmax", type=float, default=2.0)
    parser.add_argument("--bin", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_long_from_xvg(path: Path, case_id: str, xvg_path: Path, value_col: str = "g_r") -> None:
    data = parse_xvg_numeric(xvg_path)
    rows = [{"case_id": case_id, "r_nm": float(r), value_col: float(g)} for r, g in data[:, :2]]
    write_csv(path, rows, ["case_id", "r_nm", value_col])


def summarize_xvg(
    *,
    case_id: str,
    xvg_path: Path,
    rdf_kind: str,
    method: str,
    source_case_dir: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = parse_xvg_numeric(xvg_path)
    g_r = data[:, 1] if data.shape[1] > 1 else np.asarray([])
    peak_idx = int(np.argmax(g_r)) if g_r.size else 0
    peak_r = float(data[peak_idx, 0]) if g_r.size else math.nan
    tail_mask = (data[:, 0] >= 1.5) & (data[:, 0] <= 2.0) if data.shape[1] > 1 else np.asarray([])
    tail = g_r[tail_mask] if g_r.size and tail_mask.size else np.asarray([])
    summary = {
        "case_id": case_id,
        "status": "ok",
        "n_chains": "",
        "n_frames": "",
        "time_first_ps": "",
        "time_last_ps": "",
        "dt_ps": "",
        "method": method,
        "analysis_version": ANALYSIS_VERSION,
        "source_case_dir": source_case_dir,
        "rdf_kind": rdf_kind,
        "selection": SELECTIONS[rdf_kind],
        "first_peak_r_nm": peak_r,
        "first_min_r_nm": find_first_minimum_after_peak(data[:, 0], g_r, peak_r) if g_r.size else "",
        "first_peak_g": float(g_r[peak_idx]) if g_r.size else "",
        "tail_g_mean": float(np.mean(tail)) if tail.size else "",
        "tail_g_std": float(np.std(tail, ddof=1)) if tail.size > 1 else "",
    }
    if extra:
        summary.update(extra)
    return summary


def normalize_custom_summary(raw: dict[str, Any], source_case_dir: str) -> dict[str, Any]:
    kind = str(raw.get("rdf_kind", ""))
    return {
        "case_id": raw.get("case_id", ""),
        "status": "ok",
        "n_chains": raw.get("n_chains", ""),
        "n_frames": raw.get("n_frames", ""),
        "time_first_ps": "",
        "time_last_ps": "",
        "dt_ps": raw.get("dt_ps", ""),
        "method": raw.get("method", ""),
        "analysis_version": ANALYSIS_VERSION,
        "source_case_dir": source_case_dir,
        "rdf_kind": kind,
        "selection": raw.get("selection", SELECTIONS.get(kind, "")),
        "first_peak_r_nm": raw.get("first_peak_r_nm", raw.get("first_peak_r_nm_0p2_1p2", "")),
        "first_min_r_nm": raw.get("first_min_r_nm", ""),
        "first_peak_g": raw.get("first_peak_g", raw.get("first_peak_g_0p2_1p2", "")),
        "tail_g_mean": raw.get("tail_g_mean", raw.get("tail_g_mean_1p5_2p0", "")),
        "tail_g_std": raw.get("tail_g_std", raw.get("tail_g_std_1p5_2p0", "")),
        "normalization": raw.get("normalization", ""),
        "tail_normalization_status": raw.get("tail_normalization_status", ""),
        "raw_tail_g_mean": raw.get("raw_tail_g_mean", raw.get("raw_tail_g_mean_1p5_2p0", "")),
        "raw_tail_g_std": raw.get("raw_tail_g_std", raw.get("raw_tail_g_std_1p5_2p0", "")),
    }


def write_carbon_index(path: Path, atoms: list[dict[str, Any]]) -> Path:
    carbon_atoms = [
        int(atom["atom_number"])
        for atom in atoms
        if str(atom.get("atom_name", "")).strip().upper().startswith("C")
    ]
    if not carbon_atoms:
        raise ValueError("no carbon atoms found in structure")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[ Carbon ]\n")
        for idx in range(0, len(carbon_atoms), 15):
            handle.write(" ".join(str(num) for num in carbon_atoms[idx : idx + 15]) + "\n")
        handle.write("\n")
    return path


def force_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_case_view(case_out: Path, tpr: Path, traj: Path) -> Path:
    view_nvt = case_out / "case_view" / "nvt_prod"
    view_nvt.mkdir(parents=True, exist_ok=True)
    force_link(tpr, view_nvt / "nvt_prod.tpr")
    force_link(traj, view_nvt / "nvt_prod.xtc")
    return case_out / "case_view"


def run_gmx_rdf(
    *,
    gmx_bin: str,
    traj: Path,
    tpr: Path,
    index_path: Path,
    out_xvg: Path,
    rdf_kind: str,
    dt_ps: float,
    rmax: float,
    bin_width: float,
    force: bool,
) -> None:
    cmd = [
        gmx_bin,
        "rdf",
        "-f",
        str(traj),
        "-s",
        str(tpr),
        "-n",
        str(index_path),
        "-o",
        str(out_xvg),
        "-xvg",
        "none",
        "-dt",
        str(dt_ps),
        "-rmax",
        str(rmax),
        "-bin",
        str(bin_width),
        "-ref",
        'group "carbon_atoms"',
        "-sel",
        'group "carbon_atoms"',
    ]
    if rdf_kind == "excl":
        cmd.append("-excl")
    if force or not out_xvg.exists() or out_xvg.stat().st_size == 0:
        run_command(cmd)


def run_intra_nonbonded(
    *,
    case_id: str,
    case_view: Path,
    carbon_index: Path,
    case_out: Path,
    gmx_bin: str,
    dt_ps: float,
    rmax: float,
    bin_width: float,
    force: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "compute_intra_nonbonded_rdf.py"),
        "--case-id",
        case_id,
        "--case-dir",
        str(case_view),
        "--carbon-index",
        str(carbon_index),
        "--out-dir",
        str(case_out),
        "--gmx",
        str(gmx_bin),
        "--dt-ps",
        str(dt_ps),
        "--rmax",
        str(rmax),
        "--bin",
        str(bin_width),
        "--exclude-graph-distance",
        "3",
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)
    return json.loads((case_out / "rdf_intra_nonbonded_summary.json").read_text(encoding="utf-8"))


def copy_case_outputs_to_final(case_batch_dir: Path, final_case_dir: Path, entries: list[dict[str, Any]]) -> None:
    final_case_dir.mkdir(parents=True, exist_ok=True)
    for path in case_batch_dir.iterdir():
        if path.is_file() and path.name not in {"rdf_summary.json"}:
            shutil.copy2(path, final_case_dir / path.name)
    write_json(final_case_dir / "rdf_summary.json", {"case_id": entries[0]["case_id"], "entries": entries})


def rebuild_final_summary(final_rdf_root: Path) -> None:
    summary_dir = final_rdf_root / "summary"
    raw_root = final_rdf_root / "raw" / "cases"
    rebuild_summary_from_case_jsons(raw_root, summary_dir)
    write_json(
        summary_dir / "rdf_normalization_manifest.json",
        {
            "observable": "rdf",
            "analysis_version": ANALYSIS_VERSION,
            "required_rdf_kinds": list(REQUIRED_RDF_KINDS),
            "case_count": len({row.get("case_id", "") for row in read_csv(summary_dir / "rdf_summary.csv")}) if (summary_dir / "rdf_summary.csv").exists() else 0,
            "row_count": len(read_csv(summary_dir / "rdf_summary.csv")) if (summary_dir / "rdf_summary.csv").exists() else 0,
        },
    )
    rebuild_long_csvs(raw_root, summary_dir)


def rebuild_batch_summary(batch_root: Path) -> None:
    rebuild_summary_from_case_jsons(Path(batch_root) / "cases", aggregate_dir(batch_root))
    rebuild_long_csvs(Path(batch_root) / "cases", aggregate_dir(batch_root))


def rebuild_summary_from_case_jsons(raw_root: Path, summary_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if raw_root.exists():
        for path in sorted(raw_root.glob("*/rdf/rdf_summary.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload.get("entries", [])
            if not isinstance(entries, list):
                entries = [payload]
            for entry in entries:
                row = dict(entry)
                row.setdefault("case_id", path.parents[1].name)
                row.setdefault("analysis_version", ANALYSIS_VERSION)
                rows.append(row)
    rows.sort(key=lambda row: (row.get("case_id", ""), REQUIRED_RDF_KINDS.index(row.get("rdf_kind", "all")) if row.get("rdf_kind") in REQUIRED_RDF_KINDS else 99))
    write_csv(summary_dir / "rdf_summary.csv", rows, RDF_FIELDS)
    return rows


def rebuild_long_csvs(raw_root: Path, summary_dir: Path) -> None:
    for filename in RDF_LONG_FILES:
        rows: list[dict[str, Any]] = []
        fields: list[str] | None = None
        for path in sorted(raw_root.glob(f"*/rdf/{filename}")):
            file_rows = read_csv(path)
            if not file_rows:
                continue
            if fields is None:
                fields = list(file_rows[0].keys())
            rows.extend(file_rows)
        if fields:
            rows.sort(key=lambda row: (row.get("case_id", ""), safe_float(row.get("r_nm"))))
            write_csv(summary_dir / filename, rows, fields)


def rebuild_final_summary_locked(final_rdf_root: Path) -> None:
    lock_path = final_rdf_root / "summary" / ".rdf_summary.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        rebuild_final_summary(final_rdf_root)
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)


def compute_full_rdf_case(
    *,
    case_dir_arg: Path,
    gmx_bin: str,
    output_root: Path,
    batch_date: str | None,
    chain_count: int,
    dt_ps: float,
    rmax: float,
    bin_width: float,
    force: bool,
) -> dict[str, Any]:
    validate_gromacs_binary(gmx_bin)
    case_root = resolve_case_root(case_dir_arg)
    nvt_dir = resolve_nvt_dir(case_dir_arg)
    case_id = infer_case_id(case_root, nvt_dir)
    metadata = load_case_metadata(case_root)
    date_tag = batch_date or default_date_tag()
    batch_root = batch_dir(output_root, "rdf", date_tag)
    case_out = batch_root / "cases" / case_id / "rdf"
    case_out.mkdir(parents=True, exist_ok=True)

    tpr = resolve_analysis_tpr(gmx_bin, case_root, nvt_dir, force=force)
    gro = find_structure_gro(case_root, nvt_dir, case_id=case_id)
    atoms = parse_gro_atoms(gro)
    layout = infer_chain_layout(atoms, metadata, chain_count=chain_count)
    index_path = write_index_file(case_out / "lcb_chains.ndx", atoms, layout["chain_count"], layout["atoms_per_chain"])
    carbon_index = write_carbon_index(case_out / "carbon.ndx", atoms)
    traj = resolve_trajectory(gmx_bin, nvt_dir, case_out, force=force)

    entries: list[dict[str, Any]] = []
    for kind in ("all", "excl"):
        out_xvg = case_out / f"rdf_cc_{kind}.xvg"
        run_gmx_rdf(
            gmx_bin=gmx_bin,
            traj=traj,
            tpr=tpr,
            index_path=index_path,
            out_xvg=out_xvg,
            rdf_kind=kind,
            dt_ps=dt_ps,
            rmax=rmax,
            bin_width=bin_width,
            force=force,
        )
        write_long_from_xvg(case_out / f"rdf_cc_{kind}_long.csv", case_id, out_xvg)
        entries.append(
            summarize_xvg(
                case_id=case_id,
                xvg_path=out_xvg,
                rdf_kind=kind,
                method="gmx rdf",
                source_case_dir=str(case_root.resolve()),
                extra={"rdf_group": "carbon_atoms", "use_exclusions": kind == "excl"},
            )
        )

    case_view = prepare_case_view(case_out, tpr, traj)
    inter_raw = compute_inter_rdf(
        case_id,
        case_view,
        case_out / "rdf_cc_all.xvg",
        carbon_index,
        case_out,
        gmx_bin,
        dt_ps=dt_ps,
        n_chains=layout["chain_count"],
        force=force,
    )
    entries.append(normalize_custom_summary(inter_raw, str(case_root.resolve())))
    intra_raw = run_intra_nonbonded(
        case_id=case_id,
        case_view=case_view,
        carbon_index=carbon_index,
        case_out=case_out,
        gmx_bin=gmx_bin,
        dt_ps=dt_ps,
        rmax=rmax,
        bin_width=bin_width,
        force=force,
    )
    entries.append(normalize_custom_summary(intra_raw, str(case_root.resolve())))

    kinds = {entry.get("rdf_kind") for entry in entries}
    missing = [kind for kind in REQUIRED_RDF_KINDS if kind not in kinds]
    if missing:
        raise RuntimeError(f"{case_id}: missing RDF kinds after full suite: {missing}")

    write_json(case_out / "rdf_summary.json", {"case_id": case_id, "entries": entries})
    rebuild_batch_summary(batch_root)

    final_case_dir = output_root / f"{os.environ.get('LCB_PROJECT_ID', 'lcb300')}_integrated" / "RDF" / "raw" / "cases" / case_id / "rdf"
    copy_case_outputs_to_final(case_out, final_case_dir, entries)
    rebuild_final_summary_locked(output_root / f"{os.environ.get('LCB_PROJECT_ID', 'lcb300')}_integrated" / "RDF")
    return {"case_id": case_id, "entries": entries, "batch_root": str(batch_root), "final_case_dir": str(final_case_dir)}


def main() -> int:
    args = parse_args()
    result = compute_full_rdf_case(
        case_dir_arg=args.case_dir,
        gmx_bin=args.gmx_bin,
        output_root=args.output_root,
        batch_date=args.batch_date,
        chain_count=args.chain_count,
        dt_ps=args.dt_ps,
        rmax=args.rmax,
        bin_width=args.bin,
        force=args.force,
    )
    print(f"{result['case_id']} full RDF OK kinds={','.join(REQUIRED_RDF_KINDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
