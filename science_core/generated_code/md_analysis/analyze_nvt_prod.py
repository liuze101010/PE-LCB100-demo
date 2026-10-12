"""Analyze one completed nvt_prod stage for Einstein-Helfand viscosity."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

try:
    from .common_xvg import read_xvg, write_json
    from .equilibration_check import run_equilibration_checks
    from .viscosity_einstein import DEFAULT_COMPONENTS, compute_viscosity
except ImportError:  # pragma: no cover - script execution fallback
    from common_xvg import read_xvg, write_json  # type: ignore
    from equilibration_check import run_equilibration_checks  # type: ignore
    from viscosity_einstein import DEFAULT_COMPONENTS, compute_viscosity  # type: ignore


TERM_ALIASES = {
    "Temperature": ["Temperature"],
    "Total-Energy": ["Total-Energy", "Total Energy"],
    "Conserved-Energy": ["Conserved-En.", "Conserved En.", "Conserved-En", "Conserved Energy"],
    "Pressure": ["Pressure"],
    "Volume": ["Volume"],
    "Density": ["Density"],
    "Pres-XY": ["Pres-XY", "Pres-xy", "Pxy"],
    "Pres-XZ": ["Pres-XZ", "Pres-xz", "Pxz"],
    "Pres-YZ": ["Pres-YZ", "Pres-yz", "Pyz"],
    "Box-X": ["Box-X"],
    "Box-Y": ["Box-Y"],
    "Box-Z": ["Box-Z"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze one nvt_prod GROMACS stage.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Path to a case root or its nvt_prod directory.",
    )
    parser.add_argument("--gmx-bin", default="gmx", help="GROMACS executable, e.g. gmx or gmx_mpi.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for analysis.json.")
    parser.add_argument("--target-temperature", type=float, default=400.0)
    parser.add_argument("--discard-fraction", type=float, default=0.1)
    parser.add_argument("--fit-min-ps", type=float)
    parser.add_argument("--fit-max-ps", type=float)
    parser.add_argument("--force", action="store_true", help="Re-extract energy XVG even if it exists.")
    parser.add_argument("--scalars-only", action="store_true", help="Only extract scalar energy terms; skip viscosity.")
    return parser.parse_args()


def resolve_nvt_dir(path):
    path = Path(path)
    if any((path / name).exists() for name in ("nvt_prod.edr", "nvt_prod.full.edr")) or list(
        path.glob("nvt_prod.part*.edr")
    ):
        return path.resolve()
    nvt_child = path / "nvt_prod"
    if any((nvt_child / name).exists() for name in ("nvt_prod.edr", "nvt_prod.full.edr")) or list(
        nvt_child.glob("nvt_prod.part*.edr")
    ):
        return (path / "nvt_prod").resolve()
    raise FileNotFoundError(f"Could not find nvt_prod.edr below {path}")


def infer_case_id(nvt_dir):
    candidates = [nvt_dir / "case.json", nvt_dir.parent / "case.json"]
    for candidate in candidates:
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for key in ("case_id", "id"):
                if key in payload:
                    return str(payload[key])
    name = nvt_dir.parent.name if nvt_dir.name == "nvt_prod" else nvt_dir.name
    match = re.search(r"((?:lcb|sc)\d+_[A-Za-z0-9_]+?_c\d{3})(?=$|_)", name)
    return match.group(1) if match else name


def resolve_energy_edr(gmx_bin, nvt_dir, output_dir, force=False):
    """Return an EDR path, merging noappend part files when needed."""
    full_edr = nvt_dir / "nvt_prod.full.edr"
    base_edr = nvt_dir / "nvt_prod.edr"
    part_edrs = sorted(nvt_dir.glob("nvt_prod.part*.edr"))
    if full_edr.exists() and full_edr.stat().st_size > 0:
        return full_edr, {
            "mode": "existing_full_edr",
            "source_edrs": [str(full_edr)],
        }
    if part_edrs:
        if not base_edr.exists() or base_edr.stat().st_size == 0:
            raise FileNotFoundError(f"Found part EDR files but missing base nvt_prod.edr in {nvt_dir}")
        merged_edr = output_dir / "nvt_prod.full.edr"
        source_edrs = [base_edr] + part_edrs
        if force or not merged_edr.exists() or merged_edr.stat().st_size == 0:
            merge_energy_edrs(gmx_bin, source_edrs, merged_edr)
        return merged_edr, {
            "mode": "merged_parts",
            "source_edrs": [str(path) for path in source_edrs],
        }
    if base_edr.exists() and base_edr.stat().st_size > 0:
        return base_edr, {
            "mode": "single_edr",
            "source_edrs": [str(base_edr)],
        }
    raise FileNotFoundError(f"Could not find usable nvt_prod energy file in {nvt_dir}")


def merge_energy_edrs(gmx_bin, source_edrs, merged_edr):
    merged_edr.parent.mkdir(parents=True, exist_ok=True)
    cmd = [gmx_bin, "eneconv", "-f"] + [str(path) for path in source_edrs] + ["-o", str(merged_edr)]
    result = subprocess.run(
        cmd,
        input="",
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not merged_edr.exists() or merged_edr.stat().st_size == 0:
        raise RuntimeError(
            "gmx eneconv failed while merging EDR parts\n"
            f"COMMAND: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def run_gmx_energy_list(gmx_bin, edr_path):
    result = subprocess.run(
        [gmx_bin, "energy", "-f", str(edr_path)],
        input="0\n",
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = result.stdout + "\n" + result.stderr
    if result.returncode != 0 and "End your selection" not in text:
        raise RuntimeError(f"gmx energy term listing failed for {edr_path}\n{text}")
    return parse_energy_terms(text)


def parse_energy_terms(text):
    terms = {}
    for line in text.splitlines():
        matches = list(re.finditer(r"(?<!\S)(\d+)\s+", line))
        for idx, match in enumerate(matches):
            number = int(match.group(1))
            name_start = match.end()
            name_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            name = re.sub(r"\s+", " ", line[name_start:name_end].strip())
            if name and re.search(r"[A-Za-z]", name):
                terms[name] = number
    if not terms:
        raise ValueError("Could not parse any gmx energy terms")
    return terms


def match_energy_terms(available_terms):
    selected = {}
    for canonical, aliases in TERM_ALIASES.items():
        for alias in aliases:
            if alias in available_terms:
                selected[canonical] = {"name": alias, "number": int(available_terms[alias])}
                break
    required = [
        "Temperature",
        "Total-Energy",
        "Conserved-Energy",
        "Pressure",
        "Pres-XY",
        "Pres-XZ",
        "Pres-YZ",
    ]
    missing = [key for key in required if key not in selected]
    if missing:
        raise KeyError(f"Missing required GROMACS energy terms: {', '.join(missing)}")
    return selected


def latest_by_part_number(paths):
    def key(path):
        match = re.search(r"part(\d+)", Path(path).name)
        return int(match.group(1)) if match else -1

    return sorted(paths, key=key)


def resolve_final_gro(nvt_dir):
    nvt_dir = Path(nvt_dir)
    base = nvt_dir / "nvt_prod.gro"
    parts = latest_by_part_number(nvt_dir.glob("nvt_prod.part*.gro"))
    if parts:
        return parts[-1]
    if base.exists() and base.stat().st_size > 0:
        return base
    alt = nvt_dir / "all.gro"
    if alt.exists() and alt.stat().st_size > 0:
        return alt
    raise FileNotFoundError(f"Could not find final GRO with box vectors in {nvt_dir}")


def volume_from_gro_box(gro_path):
    lines = Path(gro_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty GRO file: {gro_path}")
    values = [float(item) for item in lines[-1].split()]
    if len(values) < 3:
        raise ValueError(f"Cannot parse GRO box line in {gro_path}: {lines[-1]!r}")
    if len(values) == 3:
        return values[0] * values[1] * values[2]
    # Triclinic GRO order: v1x v2y v3z v1y v1z v2x v2z v3x v3y.
    v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y = values[:9]
    matrix = np.asarray([[v1x, v1y, v1z], [v2x, v2y, v2z], [v3x, v3y, v3z]], dtype=float)
    return abs(float(np.linalg.det(matrix)))


def extract_energy_xvg(gmx_bin, edr_path, selected_terms, out_xvg, force=False):
    if out_xvg.exists() and not force:
        return
    out_xvg.parent.mkdir(parents=True, exist_ok=True)
    numbers = [str(selected_terms[key]["number"]) for key in selected_terms]
    selection = "\n".join(numbers) + "\n0\n"
    result = subprocess.run(
        [gmx_bin, "energy", "-f", str(edr_path), "-o", str(out_xvg), "-xvg", "none"],
        input=selection,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gmx energy extraction failed for {edr_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def summarize_series(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    args = parse_args()
    nvt_dir = resolve_nvt_dir(args.case_dir)
    case_id = infer_case_id(nvt_dir)
    output_dir = (args.output_dir or (Path(__file__).resolve().parent / "output" / case_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    edr_path, edr_info = resolve_energy_edr(args.gmx_bin, nvt_dir, output_dir, force=args.force)

    available_terms = run_gmx_energy_list(args.gmx_bin, edr_path)
    selected_terms = match_energy_terms(available_terms)
    write_json(
        output_dir / "energy_terms.json",
        {
            "case_id": case_id,
            "edr_path": str(edr_path),
            "edr_resolution": edr_info,
            "selected_terms": selected_terms,
            "available_term_count": len(available_terms),
        },
    )

    # gmx energy writes columns in ascending energy-term number order, not
    # necessarily in the order terms were requested on stdin.
    ordered_keys = sorted(selected_terms, key=lambda key: selected_terms[key]["number"])
    xvg_path = output_dir / "nvt_prod_energy_terms.xvg"
    extract_energy_xvg(args.gmx_bin, edr_path, selected_terms, xvg_path, force=args.force)
    frame = read_xvg(xvg_path, column_names=["time_ps"] + ordered_keys)

    time_ps = frame["time_ps"].to_numpy(dtype=float)
    volume_source = "edr:Volume"
    if "Volume" in frame:
        volume_nm3 = float(np.mean(frame["Volume"]))
    elif all(key in frame for key in ("Box-X", "Box-Y", "Box-Z")):
        volume_source = "edr:Box-X/Y/Z"
        volume_nm3 = float(np.mean(frame["Box-X"] * frame["Box-Y"] * frame["Box-Z"]))
    else:
        gro_for_volume = resolve_final_gro(nvt_dir)
        volume_source = f"gro:{gro_for_volume.name}"
        volume_nm3 = volume_from_gro_box(gro_for_volume)
    temperature_K = float(np.mean(frame["Temperature"]))
    stress = frame[list(DEFAULT_COMPONENTS)].to_numpy(dtype=float)
    scalars = {
        key: frame[key].to_numpy(dtype=float)
        for key in ["Temperature", "Total-Energy", "Conserved-Energy", "Pressure", "Volume", "Density"]
        if key in frame
    }
    if args.scalars_only:
        analysis = {
            "case_id": case_id,
            "inputs": {
                "nvt_dir": str(nvt_dir),
                "edr_path": str(edr_path),
                "edr_resolution": edr_info,
                "energy_xvg": str(xvg_path),
                "gmx_bin": args.gmx_bin,
                "scalars_only": True,
            },
            "scalars": {key: summarize_series(values) for key, values in scalars.items()},
            "volume_nm3": volume_nm3,
            "volume_source": volume_source,
            "temperature_K": temperature_K,
            "warnings": [],
        }
        write_json(output_dir / "analysis.json", analysis)
        print(f"Wrote {output_dir / 'analysis.json'}")
        return 0

    viscosity = compute_viscosity(
        time_ps,
        stress,
        volume_nm3=volume_nm3,
        temperature_K=temperature_K,
        discard_fraction=args.discard_fraction,
        fit_min_ps=args.fit_min_ps,
        fit_max_ps=args.fit_max_ps,
    )
    eq_scalars = {
        "Temperature": scalars["Temperature"],
        "Conserved-Energy": scalars["Conserved-Energy"],
    }
    equilibration = run_equilibration_checks(
        time_ps,
        eq_scalars,
        stress,
        list(DEFAULT_COMPONENTS),
        viscosity,
        target_K=args.target_temperature,
    )

    analysis = {
        "case_id": case_id,
        "inputs": {
            "nvt_dir": str(nvt_dir),
            "edr_path": str(edr_path),
            "edr_resolution": edr_info,
            "energy_xvg": str(xvg_path),
            "gmx_bin": args.gmx_bin,
            "discard_fraction": args.discard_fraction,
        },
        "scalars": {key: summarize_series(values) for key, values in scalars.items()},
        "volume_nm3_for_viscosity": volume_nm3,
        "volume_source_for_viscosity": volume_source,
        "temperature_K_for_viscosity": temperature_K,
        "viscosity": viscosity,
        "equilibration": equilibration,
        "warnings": [],
    }
    if not equilibration["equilibrated"]:
        analysis["warnings"].append("not_equilibrated")
    if viscosity["low_fit_quality"]:
        analysis["warnings"].append("low_fit_quality")
    if not viscosity["component_consistency_ok"]:
        analysis["warnings"].append("component_eta_spread_large")

    write_json(output_dir / "analysis.json", analysis)
    print(f"Wrote {output_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
