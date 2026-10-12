"""Trajectory-based structural analysis for completed LCB100 nvt_prod runs."""

import argparse
from pathlib import Path

try:
    from .canonical_outputs import batch_dir, default_date_tag
    from .trajectory_helpers import (
        find_structure_gro,
        infer_case_id,
        infer_chain_layout,
        infer_terminal_heavy_endpoints,
        load_case_metadata,
        parse_gro_atoms,
        resolve_analysis_tpr,
        resolve_case_root,
        resolve_nvt_dir,
        resolve_trajectory,
        run_msd,
        run_rdf,
        run_ree,
        run_rg_batch,
        validate_gromacs_binary,
        write_index_file,
        write_json,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from canonical_outputs import batch_dir, default_date_tag  # type: ignore
    from trajectory_helpers import (  # type: ignore
        find_structure_gro,
        infer_case_id,
        infer_chain_layout,
        infer_terminal_heavy_endpoints,
        load_case_metadata,
        parse_gro_atoms,
        resolve_analysis_tpr,
        resolve_case_root,
        resolve_nvt_dir,
        resolve_trajectory,
        run_msd,
        run_rdf,
        run_ree,
        run_rg_batch,
        validate_gromacs_binary,
        write_index_file,
        write_json,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Rg/MSD/REE/RDF analysis for one completed LCB100 nvt_prod trajectory."
    )
    parser.add_argument("--case-dir", type=Path, required=True, help="Case root or nvt_prod directory.")
    parser.add_argument("--gmx-bin", default="gmx", help="GROMACS executable, e.g. gmx or gmx_mpi.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Legacy override. Default writes canonical v2 batches under md_analysis/output/lcb100_<observable>_<date>.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Canonical output root. Default: md_analysis/output next to this script.",
    )
    parser.add_argument("--batch-date", default=None, help="Canonical batch date tag. Default: today, YYYYMMDD.")
    parser.add_argument(
        "--analyses",
        default="rg,msd,ree,rdf",
        help="Comma-separated subset of rg,msd,ree,rdf.",
    )
    parser.add_argument("--chain-count", type=int, help="Override chain_count if case.json is unavailable.")
    parser.add_argument("--max-parallel", type=int, default=2, help="Parallel gmx gyrate processes for per-chain Rg.")
    parser.add_argument(
        "--msd-beginfit-ps",
        type=float,
        default=100000.0,
        help="Begin time for gmx msd diffusion fit, in ps. Default: 100000 ps.",
    )
    parser.add_argument(
        "--msd-endfit-ps",
        type=float,
        default=300000.0,
        help="End time for gmx msd diffusion fit, in ps. Default: 300000 ps.",
    )
    parser.add_argument(
        "--msd-trestart-ps",
        type=float,
        default=100.0,
        help="Time-origin spacing for gmx msd -trestart, in ps. Default: 100 ps.",
    )
    parser.add_argument(
        "--trajectory-end-ps",
        type=float,
        default=None,
        help="Optional inclusive trajectory cutoff in ps, used to enforce a common sampling horizon.",
    )
    parser.add_argument("--rdf-group", default="heavy_atoms", choices=["heavy_atoms", "carbon_atoms", "chain_all"])
    parser.add_argument("--no-rdf-excl", action="store_true", help="Do not pass -excl to gmx rdf.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing intermediate/output files.")
    return parser.parse_args()


def analyze_trajectory(
    case_dir,
    gmx_bin="gmx",
    output_dir=None,
    analyses=("rg", "msd", "ree", "rdf"),
    chain_count=None,
    max_parallel=2,
    msd_beginfit_ps=100000.0,
    msd_endfit_ps=300000.0,
    msd_trestart_ps=100.0,
    trajectory_end_ps=None,
    rdf_group="heavy_atoms",
    rdf_excl=True,
    force=False,
    batch_date=None,
    output_root=None,
):
    validate_gromacs_binary(gmx_bin)
    case_root = resolve_case_root(case_dir)
    nvt_dir = resolve_nvt_dir(case_dir)
    case_id = infer_case_id(case_root, nvt_dir)
    metadata = load_case_metadata(case_root)
    output_root = Path(output_root) if output_root else Path(__file__).resolve().parent / "output"
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        common_dir = output_dir

        def observable_parent(_name):
            return output_dir

        summary_path = output_dir / "trajectory_analysis.json"
    else:
        date_tag = batch_date or default_date_tag()
        common_dir = batch_dir(output_root, "trajectory", date_tag) / "cases" / case_id / "trajectory"
        common_dir.mkdir(parents=True, exist_ok=True)

        def observable_parent(name):
            return batch_dir(output_root, name, date_tag) / "cases" / case_id

        summary_path = common_dir / "trajectory_analysis.json"

    tpr = resolve_analysis_tpr(gmx_bin, case_root, nvt_dir, force=force)
    gro = find_structure_gro(case_root, nvt_dir, case_id=case_id)
    atoms = parse_gro_atoms(gro)
    layout = infer_chain_layout(atoms, metadata, chain_count=chain_count)
    index_path = write_index_file(common_dir / "lcb_chains.ndx", atoms, layout["chain_count"], layout["atoms_per_chain"])
    traj = resolve_trajectory(
        gmx_bin,
        nvt_dir,
        common_dir,
        force=force,
        max_time_ps=trajectory_end_ps,
    )
    endpoints = infer_terminal_heavy_endpoints(atoms, layout["chain_count"], layout["atoms_per_chain"])

    selected = {item.strip().lower() for item in analyses if item.strip()}
    unknown = selected - {"rg", "msd", "ree", "rdf"}
    if unknown:
        raise ValueError(f"Unknown analysis name(s): {', '.join(sorted(unknown))}")

    summary = {
        "case_id": case_id,
        "inputs": {
            "case_root": str(case_root),
            "nvt_dir": str(nvt_dir),
            "tpr": str(tpr),
            "trajectory": str(traj),
            "structure_gro": str(gro),
            "index": str(index_path),
            "gmx_bin": str(gmx_bin),
        },
        "layout": layout,
        "endpoint_strategy": "terminal_heavy_atoms_in_molecule_order",
        "analyses_requested": sorted(selected),
        "results": {},
        "warnings": [
            "REE endpoints are inferred as first/last heavy atoms in each molecule's atom-order range; validate before final reporting.",
        ],
    }

    if "rg" in selected:
        summary["results"]["rg"] = run_rg_batch(
            gmx_bin,
            traj,
            tpr,
            index_path,
            observable_parent("rg"),
            layout["chain_count"],
            max_parallel=max_parallel,
            force=force,
        )
    if "msd" in selected:
        summary["results"]["msd"] = run_msd(
            gmx_bin,
            traj,
            tpr,
            index_path,
            observable_parent("msd"),
            force=force,
            beginfit_ps=msd_beginfit_ps,
            endfit_ps=msd_endfit_ps,
            trestart_ps=msd_trestart_ps,
        )
    if "ree" in selected:
        summary["results"]["ree"] = run_ree(gmx_bin, traj, tpr, observable_parent("ree"), endpoints, force=force)
    if "rdf" in selected:
        summary["results"]["rdf"] = run_rdf(
            gmx_bin,
            traj,
            tpr,
            index_path,
            observable_parent("rdf"),
            rdf_group=rdf_group,
            use_exclusions=rdf_excl,
            force=force,
        )

    write_json(summary_path, summary)
    return summary


def main():
    args = parse_args()
    summary = analyze_trajectory(
        args.case_dir,
        gmx_bin=args.gmx_bin,
        output_dir=args.output_dir,
        analyses=args.analyses.split(","),
        chain_count=args.chain_count,
        max_parallel=args.max_parallel,
        msd_beginfit_ps=args.msd_beginfit_ps,
        msd_endfit_ps=args.msd_endfit_ps,
        msd_trestart_ps=args.msd_trestart_ps,
        trajectory_end_ps=args.trajectory_end_ps,
        rdf_group=args.rdf_group,
        rdf_excl=not args.no_rdf_excl,
        force=args.force,
        batch_date=args.batch_date,
        output_root=args.output_root,
    )
    print(f"Wrote {Path(summary['inputs']['index']).parent / 'trajectory_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
