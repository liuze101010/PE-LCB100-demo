"""Helpers for LCB100 trajectory-based structural analysis."""

import csv
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    from .canonical_outputs import ANALYSIS_VERSION
except ImportError:  # pragma: no cover - script execution fallback
    from canonical_outputs import ANALYSIS_VERSION  # type: ignore


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


def parse_xvg_numeric(path):
    rows = []
    row_widths = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue
        try:
            row = [float(part) for part in line.split()]
            rows.append(row)
            row_widths[len(row)] = row_widths.get(len(row), 0) + 1
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric XVG rows found in {path}")
    if len(row_widths) != 1:
        raise ValueError(f"Inconsistent numeric XVG row widths in {path}: {row_widths}")
    return np.asarray(rows, dtype=float)


def find_first_minimum_after_peak(
    r_grid,
    g_r,
    peak_r,
    *,
    min_separation_nm=0.10,
    search_low_nm=0.65,
    search_high_nm=0.90,
):
    """Return the first-shell RDF boundary after the main peak.

    The immediate post-peak shoulder in PE C-C RDFs can produce shallow local
    minima around 0.53-0.55 nm.  Those are too close to the first peak to be a
    stable shell boundary. Weak derivative sign changes can also appear around
    0.65-0.70 nm, so the production default uses the deepest point in the
    physically relevant first-shell trough window, 0.65-0.90 nm, and at least
    0.10 nm beyond the detected peak.
    """

    r = np.asarray(r_grid, dtype=float)
    g = np.asarray(g_r, dtype=float)
    if r.size < 3 or g.size != r.size or not math.isfinite(float(peak_r)):
        return math.nan
    finite = np.isfinite(r) & np.isfinite(g)
    if int(np.count_nonzero(finite)) < 3:
        return math.nan
    r = r[finite]
    g = g[finite]
    peak = float(peak_r)
    low = max(peak + float(min_separation_nm), float(search_low_nm))
    high = float(search_high_nm)
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        return math.nan
    window = (r >= low) & (r <= high)
    if int(np.count_nonzero(window)) < 3:
        return math.nan
    r_w = r[window]
    g_w = g[window]
    return float(r_w[int(np.argmin(g_w))])


def resolve_case_root(path):
    path = Path(path).resolve()
    return path.parent if path.name == "nvt_prod" else path


def resolve_nvt_dir(path):
    path = Path(path).resolve()
    child = path / "nvt_prod"
    path_has_tpr = (path / "nvt_prod.tpr").exists()
    child_has_tpr = (child / "nvt_prod.tpr").exists()

    def has_base_trajectory(candidate):
        return any(
            trajectory.exists() and trajectory.stat().st_size > 0
            for trajectory in (
                candidate / "nvt_prod.full.xtc",
                candidate / "nvt_prod.xtc",
            )
        )

    # Extension staging can place continuation part files at the case root
    # while retaining the complete base-plus-parts trajectory in nvt_prod/.
    # Prefer the complete candidate so a continuation is never analysed alone.
    if child_has_tpr and has_base_trajectory(child):
        return child
    if path_has_tpr and has_base_trajectory(path):
        return path
    if child_has_tpr:
        return child
    if path_has_tpr:
        return path
    raise FileNotFoundError(f"Could not find nvt_prod/nvt_prod.tpr below {path}")


def infer_case_id(case_root, nvt_dir=None):
    for candidate in [Path(case_root) / "case.json", Path(nvt_dir or "") / "case.json"]:
        if candidate.exists():
            payload = read_json(candidate)
            if payload.get("case_id"):
                return str(payload["case_id"])
    name = Path(case_root).name
    match = re.search(r"((?:lcb|sc)\d+_[A-Za-z0-9_]+?_c\d{3})(?=$|_)", name)
    return match.group(1) if match else name


def load_case_metadata(case_root):
    case_json = Path(case_root) / "case.json"
    return read_json(case_json) if case_json.exists() else {}


def find_structure_gro(case_root, nvt_dir, case_id=None):
    case_root = Path(case_root)
    nvt_dir = Path(nvt_dir)
    candidates = []
    candidates.extend(sorted(nvt_dir.glob("nvt_prod.part*.gro"), reverse=True))
    candidates.extend([
        nvt_dir / "nvt_prod.gro",
        nvt_dir / "all.gro",
        nvt_dir / "confout.gro",
        case_root / "pr" / "pr.gro",
        case_root / "rlx" / "rlx.gro",
    ])
    if case_id:
        candidates.append(case_root / f"{case_id}.gro")
    candidates.extend(sorted(case_root.glob("*.gro")))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find a .gro structure file in {case_root} or {nvt_dir}")


def _tpr_readable_by_gmx(gmx_bin, tpr_path):
    tpr_path = Path(tpr_path)
    if not tpr_path.exists() or tpr_path.stat().st_size == 0:
        return False, "missing_or_empty_tpr"
    result = subprocess.run(
        [str(gmx_bin), "dump", "-s", str(tpr_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if result.returncode == 0:
        return True, "readable"
    text = (result.stdout + "\n" + result.stderr).strip()
    marker = "version_incompatible_tpr" if "version" in text.lower() and "program" in text.lower() else "unreadable_tpr"
    return False, marker


def _write_default_analysis_mdp(path):
    path = Path(path)
    path.write_text(
        "\n".join(
            [
                "; Auto-generated analysis-only mdp.",
                "; Used only to build a local-version-compatible TPR for postprocessing.",
                "integrator = md",
                "dt = 0.001",
                "nsteps = 0",
                "continuation = yes",
                "cutoff-scheme = Verlet",
                "nstlist = 10",
                "rlist = 1.0",
                "coulombtype = PME",
                "rcoulomb = 1.0",
                "vdwtype = Cut-off",
                "rvdw = 1.0",
                "pbc = xyz",
                "tcoupl = no",
                "pcoupl = no",
                "constraints = h-bonds",
                "constraint-algorithm = lincs",
                "gen-vel = no",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def resolve_analysis_tpr(gmx_bin, case_root, nvt_dir=None, force=False, maxwarn=4):
    """Return a TPR readable by the local analysis GROMACS binary.

    Production TPR files can be newer than the old-server analysis binary.
    When that happens, build an analysis-only TPR from the staged topology and
    final NVT structure. The generated TPR is for postprocessing only and does
    not alter the production trajectory or MD parameters.
    """

    case_root = resolve_case_root(case_root)
    nvt_dir = Path(nvt_dir) if nvt_dir else resolve_nvt_dir(case_root)
    nvt_dir = nvt_dir.resolve()
    production_tpr = nvt_dir / "nvt_prod.tpr"
    analysis_tpr = nvt_dir / "nvt_prod_analysis.tpr"
    provenance = nvt_dir / "nvt_prod_analysis_tpr_provenance.json"

    readable, reason = _tpr_readable_by_gmx(gmx_bin, production_tpr)
    if readable:
        return production_tpr
    if analysis_tpr.exists() and analysis_tpr.stat().st_size > 0 and not force:
        readable_analysis, _analysis_reason = _tpr_readable_by_gmx(gmx_bin, analysis_tpr)
        if readable_analysis:
            return analysis_tpr

    topol_candidates = [nvt_dir / "topol.top", case_root / "topol.top"]
    topol = next((path for path in topol_candidates if path.exists() and path.stat().st_size > 0), None)
    if topol is None:
        raise FileNotFoundError(f"Could not find topol.top for analysis TPR generation in {case_root} or {nvt_dir}")

    gro = find_structure_gro(case_root, nvt_dir)
    mdp_candidates = [
        nvt_dir / "nvt_prod.mdp",
        case_root / "nvt_prod.mdp",
        nvt_dir / "mdout.mdp",
        case_root / "mdout.mdp",
    ]
    mdp = next((path for path in mdp_candidates if path.exists() and path.stat().st_size > 0), None)
    mdp_source = "staged"
    if mdp is None:
        mdp = _write_default_analysis_mdp(nvt_dir / "nvt_prod_analysis.mdp")
        mdp_source = "auto_generated_analysis_only"

    command = [
        str(gmx_bin),
        "grompp",
        "-f",
        str(mdp.resolve()),
        "-c",
        str(gro.resolve()),
        "-p",
        str(topol.resolve()),
        "-o",
        str(analysis_tpr.resolve()),
        "-maxwarn",
        str(int(maxwarn)),
    ]
    result = subprocess.run(
        command,
        cwd=str(topol.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    payload = {
        "status": "ok" if result.returncode == 0 else "failed",
        "reason": reason,
        "production_tpr": str(production_tpr),
        "analysis_tpr": str(analysis_tpr),
        "gmx_bin": str(gmx_bin),
        "mdp": str(mdp),
        "mdp_source": mdp_source,
        "structure_gro": str(gro),
        "topology": str(topol),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
    write_json(provenance, payload)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to generate analysis-only TPR\n"
            f"COMMAND: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    readable_analysis, analysis_reason = _tpr_readable_by_gmx(gmx_bin, analysis_tpr)
    if not readable_analysis:
        raise RuntimeError(f"Generated analysis TPR is still unreadable by {gmx_bin}: {analysis_reason}")
    return analysis_tpr


def parse_gro_atoms(gro_path):
    path = Path(gro_path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {path}")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"Could not parse atom count in {path}") from exc
    atom_lines = lines[2:2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"GRO atom count mismatch in {path}: expected {atom_count}, got {len(atom_lines)}")
    atoms = []
    for idx, line in enumerate(atom_lines, start=1):
        atom_name = line[10:15].strip() if len(line) >= 15 else ""
        atom_number_text = line[15:20].strip() if len(line) >= 20 else ""
        if not atom_name or not atom_number_text:
            parts = line.split()
            atom_name = parts[1] if len(parts) > 1 else f"atom{idx}"
            atom_number_text = parts[2] if len(parts) > 2 else str(idx)
        try:
            atom_number = int(atom_number_text)
        except ValueError:
            atom_number = idx
        atoms.append({"atom_number": atom_number, "atom_name": atom_name})
    return atoms


def infer_chain_layout(atoms, case_metadata, chain_count=None):
    chain_count = chain_count or case_metadata.get("chain_count")
    if not chain_count:
        raise ValueError("chain_count is required; provide --chain-count or case.json with chain_count")
    chain_count = int(chain_count)
    atom_count = len(atoms)
    if chain_count <= 0:
        raise ValueError("chain_count must be positive")
    if atom_count % chain_count != 0:
        raise ValueError(f"Atom count {atom_count} is not divisible by chain_count {chain_count}")
    atoms_per_chain = atom_count // chain_count
    return {
        "atom_count": atom_count,
        "chain_count": chain_count,
        "atoms_per_chain": atoms_per_chain,
    }


def is_heavy_atom(atom_name):
    stripped = str(atom_name).strip()
    return bool(stripped) and not stripped.upper().startswith("H")


def is_carbon_atom(atom_name):
    return str(atom_name).strip().upper().startswith("C")


def chain_atom_numbers(chain_index, atoms_per_chain):
    start = chain_index * atoms_per_chain + 1
    return list(range(start, start + atoms_per_chain))


def infer_terminal_heavy_endpoints(atoms, chain_count, atoms_per_chain):
    endpoints = []
    for chain_idx in range(chain_count):
        start = chain_idx * atoms_per_chain
        chain_atoms = atoms[start:start + atoms_per_chain]
        heavy = [atom for atom in chain_atoms if is_heavy_atom(atom["atom_name"])]
        if len(heavy) < 2:
            raise ValueError(f"Chain {chain_idx + 1} has fewer than two heavy atoms")
        endpoints.append((int(heavy[0]["atom_number"]), int(heavy[-1]["atom_number"])))
    return endpoints


def write_index_file(path, atoms, chain_count, atoms_per_chain):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_atoms = [atom["atom_number"] for atom in atoms]
    heavy_atoms = [atom["atom_number"] for atom in atoms if is_heavy_atom(atom["atom_name"])]
    carbon_atoms = [atom["atom_number"] for atom in atoms if is_carbon_atom(atom["atom_name"])]
    with path.open("w", encoding="utf-8") as handle:
        _write_ndx_group(handle, "chain_all", all_atoms)
        _write_ndx_group(handle, "heavy_atoms", heavy_atoms)
        _write_ndx_group(handle, "carbon_atoms", carbon_atoms or heavy_atoms)
        for chain_idx in range(chain_count):
            _write_ndx_group(handle, f"chain_{chain_idx + 1:04d}", chain_atom_numbers(chain_idx, atoms_per_chain))
    return path


def _write_ndx_group(handle, name, atom_numbers):
    handle.write(f"[ {name} ]\n")
    for idx in range(0, len(atom_numbers), 15):
        handle.write(" ".join(str(num) for num in atom_numbers[idx:idx + 15]) + "\n")
    handle.write("\n")


def run_command(cmd, input_text=None, cwd=None):
    result = subprocess.run(
        [str(part) for part in cmd],
        input=input_text,
        universal_newlines=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"COMMAND: {' '.join(str(part) for part in cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def validate_gromacs_binary(gmx_bin):
    try:
        run_command([gmx_bin, "--version"])
    except Exception as exc:
        raise RuntimeError(
            "Could not run GROMACS. If this is the new GPU server, source the required "
            "compiler/MPI/GROMACS environment files before analysis, e.g. "
            "compiler_gnu-7.2.0.sh, mpi_openmpi-gnu-2.1.2.sh, and "
            "apps_gromacs-2021.2-single-gpu.sh."
        ) from exc


def summarize_series(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def time_metadata(times):
    arr = np.asarray(times, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n_frames": 0, "time_first_ps": "", "time_last_ps": "", "dt_ps": ""}
    return {
        "n_frames": int(arr.size),
        "time_first_ps": float(arr[0]),
        "time_last_ps": float(arr[-1]),
        "dt_ps": float((arr[-1] - arr[0]) / (arr.size - 1)) if arr.size > 1 else "",
    }


def sem(std_value, n):
    try:
        n = int(n)
        if n <= 0:
            return ""
        return float(std_value) / math.sqrt(n)
    except (TypeError, ValueError):
        return ""


def format_gmx_number(value):
    if value is None:
        return None
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def parse_gmx_msd_metadata(path):
    metadata = {}
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    fit_match = re.search(
        r"Diffusion constants fitted from time\s+([0-9.eE+-]+)\s+to\s+([0-9.eE+-]+)\s+ps",
        text,
    )
    if fit_match:
        metadata["gmx_fit_begin_ps"] = float(fit_match.group(1))
        metadata["gmx_fit_end_ps"] = float(fit_match.group(2))
    diff_match = re.search(
        r"D\[\s*System\s*\]\s*=\s*([0-9.eE+-]+)\s*\(\+/-\s*([0-9.eE+-]+)\)\s*\(1e-5 cm\^2/s\)",
        text,
    )
    if diff_match:
        value = float(diff_match.group(1))
        stderr = float(diff_match.group(2))
        metadata["gmx_diffusion_system_1e-5_cm2_s"] = value
        metadata["gmx_diffusion_system_stderr_1e-5_cm2_s"] = stderr
        metadata["gmx_diffusion_system_m2_s"] = value * 1.0e-9
        metadata["gmx_diffusion_system_stderr_m2_s"] = stderr * 1.0e-9
    return metadata


def linear_fit(x_values, y_values):
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.allclose(x_arr, x_arr[0]):
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan}
    design = np.vstack([x_arr, np.ones_like(x_arr)]).T
    slope, intercept = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    predicted = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - predicted) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def fit_msd_diffusion(times_ps, msd_nm2, begin_ps, end_ps):
    times = np.asarray(times_ps, dtype=float)
    msd = np.asarray(msd_nm2, dtype=float)
    mask = (times >= float(begin_ps)) & (times <= float(end_ps))
    fit = linear_fit(times[mask], msd[mask])
    slope_nm2_per_ps = fit["slope"]
    if not math.isfinite(slope_nm2_per_ps):
        d_1e5_cm2_s = math.nan
    else:
        # D = slope / 6. 1 nm^2/ps = 1000 * 1e-5 cm^2/s.
        d_1e5_cm2_s = slope_nm2_per_ps / 6.0 * 1000.0
    return {
        "D_1e-5_cm2_s": float(d_1e5_cm2_s),
        "D_m2_s": float(d_1e5_cm2_s * 1.0e-9) if math.isfinite(d_1e5_cm2_s) else math.nan,
        "R2": fit["r2"],
        "slope_nm2_per_ps": slope_nm2_per_ps,
    }


def fit_loglog_slope(times_ps, msd_nm2, begin_ps, end_ps):
    times = np.asarray(times_ps, dtype=float)
    msd = np.asarray(msd_nm2, dtype=float)
    mask = (times >= float(begin_ps)) & (times <= float(end_ps)) & (times > 0) & (msd > 0)
    fit = linear_fit(np.log10(times[mask]), np.log10(msd[mask]))
    return {"slope": fit["slope"], "r2": fit["r2"]}


def classify_diffusive_regime(loglog_slope, lower=0.95, upper=1.05):
    if not math.isfinite(float(loglog_slope)):
        return {"diffusive_regime_ok": False, "regime_warning": "missing_loglog_slope"}
    slope = float(loglog_slope)
    if slope < lower:
        return {"diffusive_regime_ok": False, "regime_warning": "sub-diffusive"}
    if slope > upper:
        return {"diffusive_regime_ok": False, "regime_warning": "super-diffusive"}
    return {"diffusive_regime_ok": True, "regime_warning": ""}


def fit_window_point_count(times_ps, begin_ps, end_ps):
    if begin_ps is None or end_ps is None:
        return 0
    times = np.asarray(times_ps, dtype=float)
    mask = np.isfinite(times) & (times >= float(begin_ps)) & (times <= float(end_ps))
    return int(np.count_nonzero(mask))


def nan_msd_fit():
    return {"D_1e-5_cm2_s": math.nan, "D_m2_s": math.nan, "R2": math.nan, "slope_nm2_per_ps": math.nan}


def nan_loglog_fit():
    return {"slope": math.nan, "r2": math.nan}


def robust_diffusion_stats(diffusion_1e5_cm2_s):
    arr = np.asarray(diffusion_1e5_cm2_s, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "D_median_1e-5_cm2_s": math.nan,
            "D_mad_1e-5_cm2_s": math.nan,
            "D_median_m2_s": math.nan,
            "D_mad_m2_s": math.nan,
            "D_n_outliers": 0,
            "D_outlier_fraction": math.nan,
        }
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad > 0:
        outlier_mask = np.abs(arr - median) > 5.0 * mad
    else:
        outlier_mask = np.zeros(arr.shape, dtype=bool)
    if median > 0:
        outlier_mask = outlier_mask | (arr < 0) | (arr > 5.0 * median)
    else:
        outlier_mask = outlier_mask | (arr < 0)
    n_outliers = int(np.sum(outlier_mask))
    return {
        "D_median_1e-5_cm2_s": median,
        "D_mad_1e-5_cm2_s": mad,
        "D_median_m2_s": median * 1.0e-9,
        "D_mad_m2_s": mad * 1.0e-9,
        "D_n_outliers": n_outliers,
        "D_outlier_fraction": float(n_outliers / arr.size),
    }


def latest_by_version(paths):
    def key(path):
        match = re.search(r"part(\d+)", Path(path).name)
        return int(match.group(1)) if match else -1
    return sorted(paths, key=key)


def trajectory_time_bounds(gmx_bin, traj_path):
    """Return approximate first/last times from gmx check output, in ps."""
    try:
        result = run_command([gmx_bin, "check", "-f", traj_path])
    except RuntimeError:
        return None
    times = []
    for match in re.finditer(r"\btime\s+([-+]?\d+(?:\.\d*)?)", result.stdout + "\n" + result.stderr):
        try:
            times.append(float(match.group(1)))
        except ValueError:
            continue
    if not times:
        return None
    return {"first_ps": min(times), "last_ps": max(times)}


def resolve_trajectory(gmx_bin, nvt_dir, output_dir, force=False, max_time_ps=None):
    nvt_dir = Path(nvt_dir)
    output_dir = Path(output_dir)
    full_xtc = nvt_dir / "nvt_prod.full.xtc"
    base_xtc = nvt_dir / "nvt_prod.xtc"
    part_xtcs = latest_by_version(nvt_dir.glob("nvt_prod.part*.xtc"))

    def apply_time_cap(traj_path):
        if max_time_ps is None:
            return traj_path
        bounds = trajectory_time_bounds(gmx_bin, traj_path)
        if bounds and bounds["last_ps"] <= float(max_time_ps) + 1.0:
            return traj_path
        capped = output_dir / f"nvt_prod.through_{int(float(max_time_ps))}ps.xtc"
        if force or not capped.exists() or capped.stat().st_size == 0:
            run_command(
                [
                    gmx_bin,
                    "trjconv",
                    "-f",
                    traj_path,
                    "-s",
                    nvt_dir / "nvt_prod.tpr",
                    "-o",
                    capped,
                    "-e",
                    format_gmx_number(max_time_ps),
                ],
                input_text="0\n",
            )
        return capped

    if part_xtcs:
        if not base_xtc.exists() or base_xtc.stat().st_size == 0:
            raise FileNotFoundError(f"Found part XTC files but missing base nvt_prod.xtc in {nvt_dir}")
        base_bounds = trajectory_time_bounds(gmx_bin, base_xtc)
        part_bounds = [trajectory_time_bounds(gmx_bin, part) for part in part_xtcs]
        valid_part_bounds = [item for item in part_bounds if item]
        if full_xtc.exists() and full_xtc.stat().st_size > 0 and not force:
            full_bounds = trajectory_time_bounds(gmx_bin, full_xtc)
            max_part_time = max((item["last_ps"] for item in valid_part_bounds), default=None)
            if full_bounds and (max_part_time is None or full_bounds["last_ps"] >= max_part_time - 1.0):
                return apply_time_cap(full_xtc)
        if base_bounds and valid_part_bounds:
            max_part_time = max(item["last_ps"] for item in valid_part_bounds)
            min_part_time = min(item["first_ps"] for item in valid_part_bounds)
            if base_bounds["last_ps"] >= max_part_time - 1.0 and min_part_time <= base_bounds["last_ps"] + 1.0:
                return apply_time_cap(base_xtc)
        merged = output_dir / "nvt_prod.full.xtc"
        if force or not merged.exists() or merged.stat().st_size == 0:
            run_command([gmx_bin, "trjcat", "-f", base_xtc, *part_xtcs, "-o", merged])
        return apply_time_cap(merged)
    if full_xtc.exists() and full_xtc.stat().st_size > 0:
        return apply_time_cap(full_xtc)
    if base_xtc.exists() and base_xtc.stat().st_size > 0:
        return apply_time_cap(base_xtc)
    raise FileNotFoundError(f"Could not find usable nvt_prod trajectory in {nvt_dir}")


def run_rg_for_chain(gmx_bin, traj, tpr, index_path, output_path, group_name):
    run_command(
        [gmx_bin, "gyrate", "-f", traj, "-s", tpr, "-n", index_path, "-o", output_path],
        input_text=f"{group_name}\n",
    )
    data = parse_xvg_numeric(output_path)
    if data.shape[1] < 2:
        raise ValueError(f"Unexpected gyrate output width in {output_path}")
    return data


def run_rg_batch(gmx_bin, traj, tpr, index_path, output_dir, chain_count, max_parallel=2, force=False):
    rg_dir = Path(output_dir) / "rg"
    rg_dir.mkdir(parents=True, exist_ok=True)

    def one(chain_idx):
        group = f"chain_{chain_idx + 1:04d}"
        out_xvg = rg_dir / f"{group}_rg.xvg"
        if force or not out_xvg.exists() or out_xvg.stat().st_size == 0:
            data = run_rg_for_chain(gmx_bin, traj, tpr, index_path, out_xvg, group)
        else:
            data = parse_xvg_numeric(out_xvg)
        return chain_idx, data

    chain_data = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_parallel))) as executor:
        for chain_idx, data in executor.map(one, range(chain_count)):
            chain_data[chain_idx] = data

    min_len = min(data.shape[0] for data in chain_data.values())
    times = chain_data[0][:min_len, 0]
    rg_matrix = np.column_stack([chain_data[idx][:min_len, 1] for idx in range(chain_count)])
    mean_rg = np.mean(rg_matrix, axis=1)

    chain_rows = []
    for idx in range(chain_count):
        vals = chain_data[idx][:, 1]
        stats = summarize_series(vals)
        chain_rows.append({"chain": idx + 1, **stats})
    write_csv(rg_dir / "rg_chains.csv", chain_rows, ["chain", "mean", "std", "min", "max"])
    write_csv(
        rg_dir / "rg_timeseries.csv",
        [{"time_ps": float(t), "mean_Rg_nm": float(v)} for t, v in zip(times, mean_rg)],
        ["time_ps", "mean_Rg_nm"],
    )
    mean_stats = summarize_series(mean_rg)
    time_meta = time_metadata(times)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "ok",
        "method": "per-chain gmx gyrate averaged over chains",
        "n_chains": chain_count,
        **time_meta,
        "rg_mean_nm": mean_stats["mean"],
        "rg_std_time_nm": mean_stats["std"],
        "rg_sem_time_nm": sem(mean_stats["std"], time_meta["n_frames"]),
        "rg2_mean_nm2": float(np.mean(rg_matrix * rg_matrix)),
        "rg2_over_linear_rg2": "",
        "linear_baseline_case_id": "",
        "baseline_status": "",
        "source_has_no_timeseries": False,
        "chain_count": chain_count,
        "mean_Rg_nm": mean_stats,
        "chain_mean_Rg_nm": summarize_series([row["mean"] for row in chain_rows]),
    }
    write_json(rg_dir / "rg_summary.json", summary)
    return summary


def write_ree_selection_file(path, endpoints):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for start, end in endpoints:
            # GROMACS 2026 accepts unnamed pair selections in -sf files, while
            # named assignments can fail for position-pair expressions.
            # Semicolons keep each pair as a separate selection in -sf files.
            handle.write(f"atomnr {start} plus atomnr {end};\n")
    return path


def run_ree(gmx_bin, traj, tpr, output_dir, endpoints, force=False):
    ree_dir = Path(output_dir) / "ree"
    ree_dir.mkdir(parents=True, exist_ok=True)
    selection_file = write_ree_selection_file(ree_dir / "ree_selections.dat", endpoints)
    out_xvg = ree_dir / "ree_distances.xvg"
    if force or not out_xvg.exists() or out_xvg.stat().st_size == 0:
        run_command([gmx_bin, "distance", "-s", tpr, "-f", traj, "-sf", selection_file, "-oall", out_xvg])
    data = parse_xvg_numeric(out_xvg)
    if data.shape[1] < 2:
        raise ValueError(f"Unexpected distance output width in {out_xvg}")
    times = data[:, 0]
    distances = data[:, 1:]
    ree2 = distances * distances
    mean_ree = np.mean(distances, axis=1)
    mean_ree2 = np.mean(ree2, axis=1)
    chain_rows = []
    for idx in range(distances.shape[1]):
        stats = summarize_series(distances[:, idx])
        stats2 = summarize_series(ree2[:, idx])
        chain_rows.append({
            "chain": idx + 1,
            "mean_Ree_nm": stats["mean"],
            "std_Ree_nm": stats["std"],
            "mean_Ree2_nm2": stats2["mean"],
        })
    write_csv(ree_dir / "ree_chains.csv", chain_rows, ["chain", "mean_Ree_nm", "std_Ree_nm", "mean_Ree2_nm2"])
    write_csv(
        ree_dir / "ree_timeseries.csv",
        [
            {"time_ps": float(t), "mean_Ree_nm": float(r), "mean_Ree2_nm2": float(r2)}
            for t, r, r2 in zip(times, mean_ree, mean_ree2)
        ],
        ["time_ps", "mean_Ree_nm", "mean_Ree2_nm2"],
    )
    ree_stats = summarize_series(mean_ree)
    ree2_stats = summarize_series(mean_ree2)
    chain_mean_stats = summarize_series([row["mean_Ree_nm"] for row in chain_rows])
    time_meta = time_metadata(times)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "ok",
        "method": "gmx distance terminal heavy atom pairs",
        "n_chains": len(endpoints),
        **time_meta,
        "ree_endpoint_strategy": "terminal_heavy_atoms_in_molecule_order",
        "ree_mean_nm": ree_stats["mean"],
        "ree_std_time_nm": ree_stats["std"],
        "ree_chain_mean_nm": chain_mean_stats["mean"],
        "ree2_mean_nm2": ree2_stats["mean"],
        "endpoint_strategy": "terminal_heavy_atoms_in_molecule_order",
        "chain_count": len(endpoints),
        "mean_Ree_nm": ree_stats,
        "mean_Ree2_nm2": ree2_stats,
    }
    write_json(ree_dir / "ree_summary.json", summary)
    return summary


def run_msd(
    gmx_bin,
    traj,
    tpr,
    index_path,
    output_dir,
    force=False,
    beginfit_ps=100000,
    endfit_ps=300000,
    trestart_ps=100,
):
    msd_dir = Path(output_dir) / "msd"
    msd_dir.mkdir(parents=True, exist_ok=True)
    nojump = msd_dir / "nvt_prod_nojump.xtc"
    nocom = msd_dir / "nvt_prod_nocom.xtc"
    if force or not nojump.exists() or nojump.stat().st_size == 0:
        run_command(
            [gmx_bin, "trjconv", "-f", traj, "-s", tpr, "-n", index_path, "-o", nojump, "-pbc", "nojump"],
            input_text="chain_all\n",
        )
    if force or not nocom.exists() or nocom.stat().st_size == 0:
        # Remove only whole-system translational drift before molecule MSD.
        # This is not molecule centering; it prevents thermostat/integration COM drift from contaminating D.
        run_command(
            [gmx_bin, "trjconv", "-f", nojump, "-s", tpr, "-n", index_path, "-o", nocom, "-fit", "translation"],
            input_text="chain_all\nchain_all\n",
        )
    warning_issues = []
    bounds = trajectory_time_bounds(gmx_bin, nocom)
    gmx_beginfit_ps = beginfit_ps
    gmx_endfit_ps = endfit_ps
    if beginfit_ps is not None and endfit_ps is not None:
        if float(beginfit_ps) >= float(endfit_ps):
            warning_issues.append("invalid_requested_fit_window")
            gmx_beginfit_ps = None
            gmx_endfit_ps = None
        elif bounds and float(endfit_ps) > float(bounds["last_ps"]):
            # Keep the timeseries and per-chain MSD evidence, but do not let a
            # too-long publication fit window abort the whole postprocess bundle.
            warning_issues.append("fit_window_exceeds_trajectory")
            gmx_beginfit_ps = None
            gmx_endfit_ps = None
    msd_xvg = msd_dir / "msd.xvg"
    diffusion_xvg = msd_dir / "diffusion.xvg"
    if force or not msd_xvg.exists() or msd_xvg.stat().st_size == 0:
        cmd = [
            gmx_bin,
            "msd",
            "-f",
            nocom,
            "-s",
            tpr,
            "-n",
            index_path,
            "-o",
            msd_xvg,
            "-mol",
            diffusion_xvg,
            "-trestart",
            format_gmx_number(trestart_ps),
        ]
        if gmx_beginfit_ps is not None:
            cmd.extend(["-beginfit", format_gmx_number(gmx_beginfit_ps)])
        if gmx_endfit_ps is not None:
            cmd.extend(["-endfit", format_gmx_number(gmx_endfit_ps)])
        try:
            run_command(
                cmd,
                input_text="chain_all\n",
            )
        except RuntimeError:
            if beginfit_ps is None or endfit_ps is None:
                raise
            warning_issues.append("fit_window_rejected_by_gmx")
            # A failed fit-window run can leave partial XVG files behind. The
            # no-fit fallback must start from clean outputs; otherwise GROMACS
            # or the parser may trip over stale files and abort the whole
            # postprocessing bundle instead of emitting a bounded WARN.
            for partial in (msd_xvg, diffusion_xvg):
                try:
                    partial.unlink()
                except FileNotFoundError:
                    pass
            fallback_cmd = [
                gmx_bin,
                "msd",
                "-f",
                nocom,
                "-s",
                tpr,
                "-n",
                index_path,
                "-o",
                msd_xvg,
                "-mol",
                diffusion_xvg,
                "-trestart",
                format_gmx_number(trestart_ps),
            ]
            try:
                run_command(
                    fallback_cmd,
                    input_text="chain_all\n",
                )
            except RuntimeError as fallback_exc:
                warning_issues.append("gmx_msd_failed")
                write_csv(
                    msd_dir / "msd_timeseries.csv",
                    [],
                    ["time_ps", "msd_nm2"],
                )
                failed_fit = nan_msd_fit()
                failed_loglog = nan_loglog_fit()
                summary = {
                    "analysis_version": ANALYSIS_VERSION,
                    "status": "warn",
                    "n_chains": "",
                    **time_metadata([]),
                    "msd_final_nm2": math.nan,
                    "D_mean_m2_s": failed_fit["D_m2_s"],
                    "D_std_m2_s": math.nan,
                    "D_mean_1e-5_cm2_s": failed_fit["D_1e-5_cm2_s"],
                    "fit_begin_ps": math.nan,
                    "fit_end_ps": math.nan,
                    "loglog_slope": failed_loglog["slope"],
                    "fit_quality": "gmx_msd_failed",
                    "diffusive_regime_ok": False,
                    "regime_warning": "gmx_msd_failed",
                    "warnings": warning_issues,
                    "requested_fit_point_count": 0,
                    "trajectory_time_bounds_ps": bounds or {},
                    "window_sensitive": True,
                    "window_D_cv_pct": math.nan,
                    "method": "gmx msd -mol chain COM failed; retained as WARN evidence",
                    "fit_window_requested_ps": {
                        "begin": float(beginfit_ps) if beginfit_ps is not None else None,
                        "end": float(endfit_ps) if endfit_ps is not None else None,
                    },
                    "gmx_fit_window_actual_ps": {"begin": None, "end": None},
                    "msd_timeseries_csv": str(msd_dir / "msd_timeseries.csv"),
                    "final_MSD_nm2": math.nan,
                    "D_median_m2_s": math.nan,
                    "D_mad_m2_s": math.nan,
                    "D_outlier_fraction": math.nan,
                    "gmx_msd_error_tail": str(fallback_exc)[-4000:],
                }
                write_json(msd_dir / "msd_summary.json", summary)
                return summary
    msd_data = parse_xvg_numeric(msd_xvg)
    if msd_data.shape[1] < 2:
        raise ValueError(f"Unexpected MSD output width in {msd_xvg}: expected at least 2 columns")
    write_csv(
        msd_dir / "msd_timeseries.csv",
        [{"time_ps": float(t), "msd_nm2": float(msd)} for t, msd in msd_data[:, :2]],
        ["time_ps", "msd_nm2"],
    )
    metadata = parse_gmx_msd_metadata(msd_xvg)
    time_meta = time_metadata(msd_data[:, 0])
    fit_begin_actual = float(beginfit_ps) if beginfit_ps is not None else metadata.get("gmx_fit_begin_ps", "")
    fit_end_actual = float(endfit_ps) if endfit_ps is not None else metadata.get("gmx_fit_end_ps", "")
    requested_fit_point_count = fit_window_point_count(msd_data[:, 0], beginfit_ps, endfit_ps)
    requested_fit_valid = requested_fit_point_count >= 3
    if not requested_fit_valid:
        if "fit_window_exceeds_trajectory" not in warning_issues:
            warning_issues.append("fit_window_insufficient_points")
        primary_fit = nan_msd_fit()
        primary_loglog = nan_loglog_fit()
    else:
        primary_fit = fit_msd_diffusion(msd_data[:, 0], msd_data[:, 1], fit_begin_actual, fit_end_actual)
        primary_loglog = fit_loglog_slope(msd_data[:, 0], msd_data[:, 1], fit_begin_actual, fit_end_actual)
    regime = classify_diffusive_regime(primary_loglog["slope"])
    window_fits = {}
    for label, begin_ps, end_ps in [
        ("40_360ns", 40000.0, 360000.0),
        ("80_320ns", 80000.0, 320000.0),
        ("100_300ns", 100000.0, 300000.0),
        ("150_350ns", 150000.0, 350000.0),
        ("200_400ns", 200000.0, 400000.0),
    ]:
        window_fits[label] = fit_msd_diffusion(msd_data[:, 0], msd_data[:, 1], begin_ps, end_ps)
    window_d_values = np.asarray(
        [item["D_1e-5_cm2_s"] for item in window_fits.values() if math.isfinite(item["D_1e-5_cm2_s"])],
        dtype=float,
    )
    window_d_mean = float(np.mean(window_d_values)) if window_d_values.size else math.nan
    window_d_cv_pct = (
        float(np.std(window_d_values, ddof=1) / window_d_mean * 100.0)
        if window_d_values.size > 1 and window_d_mean
        else math.nan
    )
    window_sensitive = bool(math.isfinite(window_d_cv_pct) and window_d_cv_pct > 5.0)
    fit_quality = "ok"
    status = "ok"
    if not regime["diffusive_regime_ok"] or window_sensitive or not math.isfinite(primary_fit["R2"]) or primary_fit["R2"] < 0.95:
        fit_quality = "check"
    if (not math.isfinite(primary_fit["D_1e-5_cm2_s"])) or primary_fit["D_1e-5_cm2_s"] <= 0:
        fit_quality = "low_fit_quality"
    if warning_issues:
        status = "warn"
        fit_quality = "insufficient_fit_window"
        regime = {"diffusive_regime_ok": False, "regime_warning": warning_issues[0]}
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "n_chains": "",
        **time_meta,
        "msd_final_nm2": float(msd_data[-1, 1]) if msd_data.shape[1] > 1 else math.nan,
        "D_mean_m2_s": primary_fit["D_m2_s"],
        "D_std_m2_s": metadata.get("gmx_diffusion_system_stderr_m2_s"),
        "D_mean_1e-5_cm2_s": primary_fit["D_1e-5_cm2_s"],
        "fit_begin_ps": fit_begin_actual,
        "fit_end_ps": fit_end_actual,
        "loglog_slope": primary_loglog["slope"],
        "fit_quality": fit_quality,
        **regime,
        "warnings": warning_issues,
        "requested_fit_point_count": requested_fit_point_count,
        "trajectory_time_bounds_ps": bounds or {},
        "window_sensitive": window_sensitive,
        "window_D_cv_pct": window_d_cv_pct,
        "method": "gmx msd -mol chain COM; D from mean MSD fit 100-300 ns",
        "fit_window_requested_ps": {
            "begin": float(beginfit_ps) if beginfit_ps is not None else None,
            "end": float(endfit_ps) if endfit_ps is not None else None,
        },
        "gmx_fit_window_actual_ps": {
            "begin": metadata.get("gmx_fit_begin_ps"),
            "end": metadata.get("gmx_fit_end_ps"),
        },
        "msd_timeseries_csv": str(msd_dir / "msd_timeseries.csv"),
        "final_MSD_nm2": float(msd_data[-1, 1]) if msd_data.shape[1] > 1 else math.nan,
        "time_final_ps": float(msd_data[-1, 0]),
        "diffusion_unit_gmx": "1e-5 cm^2/s",
        "diffusion_unit_si": "m^2/s",
        "diffusion_conversion": "D_m2_s = D_1e-5_cm2_s * 1.0e-9",
        "gmx_system_diffusion": {
            "D_1e-5_cm2_s": metadata.get("gmx_diffusion_system_1e-5_cm2_s"),
            "stderr_1e-5_cm2_s": metadata.get("gmx_diffusion_system_stderr_1e-5_cm2_s"),
            "D_m2_s": metadata.get("gmx_diffusion_system_m2_s"),
            "stderr_m2_s": metadata.get("gmx_diffusion_system_stderr_m2_s"),
        },
        "python_fit": {
            "D_1e-5_cm2_s": primary_fit["D_1e-5_cm2_s"],
            "D_m2_s": primary_fit["D_m2_s"],
            "R2": primary_fit["R2"],
            "loglog_slope": primary_loglog["slope"],
            "loglog_R2": primary_loglog["r2"],
        },
        "window_fits": window_fits,
        "notes": [
            "nvt_prod_nojump.xtc removes PBC jumps.",
            "nvt_prod_nocom.xtc removes whole-system translation only; molecules are not individually centered.",
            "The per-molecule diffusion values in diffusion.xvg use the GROMACS default unit 1e-5 cm^2/s.",
        ],
    }
    # Backward-compatible key retained for existing aggregate.py consumers.
    summary["mean_final_MSD_nm2"] = summary["final_MSD_nm2"]
    if diffusion_xvg.exists() and diffusion_xvg.stat().st_size > 0:
        diff_data = parse_xvg_numeric(diffusion_xvg)
        if diff_data.shape[1] > 1:
            diffusion_rows = [
                {
                    "molecule_index": int(row[0]),
                    "D_1e-5_cm2_s": float(row[1]),
                    "D_m2_s": float(row[1]) * 1.0e-9,
                }
                for row in diff_data[:, :2]
            ]
            write_csv(
                msd_dir / "diffusion_per_molecule.csv",
                diffusion_rows,
                ["molecule_index", "D_1e-5_cm2_s", "D_m2_s"],
            )
            summary["n_chains"] = len(diffusion_rows)
            summary["diffusion_per_molecule"] = {
                "unit": "1e-5 cm^2/s (GROMACS gmx msd default)",
                "unit_si": "m^2/s",
                "conversion_to_si": "multiply by 1.0e-9",
                "D_1e-5_cm2_s": summarize_series(diff_data[:, 1]),
                "D_m2_s": summarize_series(diff_data[:, 1] * 1.0e-9),
                "csv": str(msd_dir / "diffusion_per_molecule.csv"),
            }
            robust_stats = robust_diffusion_stats(diff_data[:, 1])
            summary.update(robust_stats)
            summary["D_std_m2_s"] = summary["diffusion_per_molecule"]["D_m2_s"]["std"]
            # Backward-compatible key retained for existing consumers.
            summary["diffusion_column_1"] = summarize_series(diff_data[:, 1])
    write_json(msd_dir / "msd_summary.json", summary)
    return summary


def run_rdf(gmx_bin, traj, tpr, index_path, output_dir, rdf_group="heavy_atoms", use_exclusions=True, force=False):
    rdf_dir = Path(output_dir) / "rdf"
    rdf_dir.mkdir(parents=True, exist_ok=True)
    rdf_xvg = rdf_dir / f"rdf_{rdf_group}.xvg"
    cmd = [
        gmx_bin,
        "rdf",
        "-f",
        traj,
        "-s",
        tpr,
        "-n",
        index_path,
        "-o",
        rdf_xvg,
        "-ref",
        f'group "{rdf_group}"',
        "-sel",
        f'group "{rdf_group}"',
    ]
    if use_exclusions:
        cmd.append("-excl")
    if force or not rdf_xvg.exists() or rdf_xvg.stat().st_size == 0:
        run_command(cmd)
    data = parse_xvg_numeric(rdf_xvg)
    g_r = data[:, 1] if data.shape[1] > 1 else np.asarray([])
    peak_idx = int(np.argmax(g_r)) if g_r.size else 0
    peak_r = float(data[peak_idx, 0]) if g_r.size else math.nan
    tail_mask = (data[:, 0] >= 1.5) & (data[:, 0] <= 2.0) if data.shape[1] > 1 else np.asarray([])
    tail = g_r[tail_mask] if g_r.size and tail_mask.size else np.asarray([])
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "ok",
        "method": "gmx rdf",
        "rdf_kind": "excl" if use_exclusions else "all",
        "selection": rdf_group,
        "first_peak_r_nm": peak_r,
        "first_min_r_nm": find_first_minimum_after_peak(data[:, 0], g_r, peak_r) if g_r.size else math.nan,
        "first_peak_g": float(g_r[peak_idx]) if g_r.size else math.nan,
        "tail_g_mean": float(np.mean(tail)) if tail.size else "",
        "tail_g_std": float(np.std(tail, ddof=1)) if tail.size > 1 else "",
        "rdf_group": rdf_group,
        "use_exclusions": bool(use_exclusions),
        "peak_r_nm": float(data[peak_idx, 0]) if g_r.size else math.nan,
        "peak_g_r": float(g_r[peak_idx]) if g_r.size else math.nan,
    }
    write_json(rdf_dir / "rdf_summary.json", summary)
    return summary


