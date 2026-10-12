import argparse
import gzip
import json
import math
import shutil
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple


KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1


AtomRecord = namedtuple('AtomRecord', ['atom_id', 'resid', 'resname', 'atom_name', 'atom_type', 'charge', 'mass'])
CoordRecord = namedtuple('CoordRecord', ['atom_id', 'x_ang', 'y_ang', 'z_ang'])


def write_text_unix(path: Path, content: str, encoding: str = "utf-8") -> None:
    with path.open("w", encoding=encoding, newline="\n") as handle:
        handle.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge EMC PSF/PDB outputs into a minimal GROMACS test package."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Case directory containing case.json and remote_build artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for the generated GROMACS package. Defaults to <case-dir>/gromacs_bridge_min_test.",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=50,
        help="Number of steepest-descent minimization steps for the minimal bridge test.",
    )
    parser.add_argument(
        "--partition",
        default="intel",
        help="Default Slurm partition in the generated sbatch script.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=8,
        help="Default CPU count in the generated sbatch script.",
    )
    parser.add_argument(
        "--time-limit",
        default="00:20:00",
        help="Default Slurm time limit in the generated sbatch script.",
    )
    return parser.parse_args()


def read_case_metadata(case_dir: Path) -> dict:
    return json.loads((case_dir / "case.json").read_text(encoding="utf-8"))


def find_case_file(remote_build_dir: Path, case_id: str, suffix: str) -> Path:
    direct = remote_build_dir / f"{case_id}{suffix}"
    if direct.exists():
        return direct

    # Search in all screen_*/ subdirectories for the file. LCB production
    # builds use screen_lcbXX, while linear alpha-olefin baselines use
    # labels such as screen_c1_c20.
    try:
        for item in remote_build_dir.iterdir():
            if item.is_dir() and item.name.startswith("screen_"):
                try:
                    for f in item.iterdir():
                        if f.is_file() and f.name.startswith(case_id) and f.name.endswith(suffix):
                            return f
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError):
        pass

    # Fallback: search directly in remote_build_dir for flat structure
    try:
        for f in remote_build_dir.iterdir():
            if f.is_file() and f.name.startswith(case_id) and f.name.endswith(suffix):
                return f
    except (OSError, PermissionError):
        pass

    # Production runs are commonly stored under a run-label directory rather
    # than directly under remote_build or screen_*. Search those descendants
    # while keeping the filename tied to the requested case.
    try:
        matches = sorted(
            path
            for path in remote_build_dir.rglob(f"{case_id}*{suffix}")
            if path.is_file()
        )
        if matches:
            return matches[0]
    except (OSError, PermissionError):
        pass

    raise FileNotFoundError(f"Could not find {case_id}*{suffix} under {remote_build_dir}")


def read_text_maybe_gz(path: Path) -> List[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.readlines()
    return path.read_text(encoding="utf-8", errors="replace").splitlines(True)


def section_indexes(lines: List[str], names: List[str]) -> Dict[str, int]:
    found = {}
    for idx, line in enumerate(lines):
        for name in names:
            if name in line and name not in found:
                found[name] = idx
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(f"Missing sections in PSF: {missing}")
    return found


def parse_section_count(line: str) -> int:
    """Extract count from PSF section header line (e.g., '!NATOM' line)."""
    try:
        parts = line.split()
        if parts:
            return int(parts[0])
    except (ValueError, IndexError):
        pass
    return 0


def parse_params(params_path: Path) -> Dict[str, Dict]:
    atom_masses = {}
    pair_params = {}
    bond_params = {}
    angle_params = {}
    dihedral_params = {}

    for raw in params_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        body, _, comment = raw.partition("#")
        comment = comment.strip()
        parts = body.split()
        if not parts:
            continue
        head = parts[0]
        if head == "mass":
            atom_masses[comment] = float(parts[2])
        elif head == "pair_coeff":
            labels = tuple(item.strip() for item in comment.split(","))
            epsilon_kcal = float(parts[3])
            sigma_ang = float(parts[4])
            pair_params[labels] = (sigma_ang * ANGSTROM_TO_NM, epsilon_kcal * KCAL_TO_KJ)
        elif head == "bond_coeff":
            labels = tuple(item.strip() for item in comment.split(","))
            k_lammps = float(parts[2])
            r0_ang = float(parts[3])
            k_gmx = 2.0 * k_lammps * KCAL_TO_KJ / (ANGSTROM_TO_NM ** 2)
            bond_params[labels] = (r0_ang * ANGSTROM_TO_NM, k_gmx)
        elif head == "angle_coeff":
            labels = tuple(item.strip() for item in comment.split(","))
            k_lammps = float(parts[2])
            theta_deg = float(parts[3])
            k_gmx = 2.0 * k_lammps * KCAL_TO_KJ
            angle_params[labels] = (theta_deg, k_gmx)
        elif head == "dihedral_coeff":
            labels = tuple(item.strip() for item in comment.split(","))
            coeffs = [float(value) * KCAL_TO_KJ for value in parts[2:]]
            rb = []
            for idx in range(6):
                if idx < len(coeffs):
                    rb.append(coeffs[idx] * ((-1.0) ** idx))
                else:
                    rb.append(0.0)
            dihedral_params[labels] = tuple(rb)

    return {
        "atom_masses": atom_masses,
        "pair_params": pair_params,
        "bond_params": bond_params,
        "angle_params": angle_params,
        "dihedral_params": dihedral_params,
    }


def parse_group_section(lines: List[str], start: int, end: int, group_size: int, atom_limit: int) -> List[Tuple[int, ...]]:
    items = []
    buffer = []
    for line in lines[start:end]:
        tokens = line.split()
        if not tokens:
            continue
        # Skip PSF section markers (like !NIMPHI:) and non-numeric lines
        if tokens[0].startswith("!"):
            break
        try:
            buffer.extend(int(token) for token in tokens)
        except ValueError:
            # Stop on any non-numeric token
            break
        while len(buffer) >= group_size:
            item = tuple(buffer[:group_size])
            del buffer[:group_size]
            if max(item) > atom_limit:
                return items
            items.append(item)
    return items


def parse_psf_template(psf_path: Path, chain_count: int) -> Tuple[List[AtomRecord], List[Tuple[int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int, int]], int]:
    lines = read_text_maybe_gz(psf_path)
    sections = section_indexes(lines, ["!NATOM", "!NBOND", "!NTHETA", "!NPHI"])
    total_atoms = int(lines[sections["!NATOM"]].split()[0])
    if total_atoms % chain_count != 0:
        raise ValueError(f"Total atoms {total_atoms} is not divisible by chain_count {chain_count}")
    atoms_per_molecule = total_atoms // chain_count

    atoms = []
    atom_start = sections["!NATOM"] + 1
    for line in lines[atom_start : atom_start + atoms_per_molecule]:
        parts = line.split()
        atoms.append(
            AtomRecord(
                atom_id=int(parts[0]),
                resid=int(parts[2]),
                resname=parts[3],
                atom_name=parts[4],
                atom_type=parts[5],
                charge=float(parts[6]),
                mass=float(parts[7]),
            )
        )

    # Parse topology and verify completeness
    expected_nbond_total = parse_section_count(lines[sections["!NBOND"]])
    expected_nangle_total = parse_section_count(lines[sections["!NTHETA"]])
    expected_ndihedral_total = parse_section_count(lines[sections["!NPHI"]])

    if expected_nbond_total % chain_count != 0:
        raise ValueError(
            f"Total bond count {expected_nbond_total} is not divisible by chain_count {chain_count}"
        )
    if expected_nangle_total % chain_count != 0:
        raise ValueError(
            f"Total angle count {expected_nangle_total} is not divisible by chain_count {chain_count}"
        )
    if expected_ndihedral_total % chain_count != 0:
        raise ValueError(
            f"Total dihedral count {expected_ndihedral_total} is not divisible by chain_count {chain_count}"
        )

    expected_nbond = expected_nbond_total // chain_count
    expected_nangle = expected_nangle_total // chain_count
    expected_ndihedral = expected_ndihedral_total // chain_count

    bonds = parse_group_section(lines, sections["!NBOND"] + 1, sections["!NTHETA"], 2, atoms_per_molecule)
    angles = parse_group_section(lines, sections["!NTHETA"] + 1, sections["!NPHI"], 3, atoms_per_molecule)
    dihedrals = parse_group_section(lines, sections["!NPHI"] + 1, len(lines), 4, atoms_per_molecule)

    # Warn if topology appears truncated
    if expected_nbond > 0 and len(bonds) < expected_nbond * 0.95:
        import sys
        print("WARNING: Bond count may be incomplete ({} of {} expected)".format(len(bonds), expected_nbond), file=sys.stderr)
    if expected_nangle > 0 and len(angles) < expected_nangle * 0.95:
        import sys
        print("WARNING: Angle count may be incomplete ({} of {} expected)".format(len(angles), expected_nangle), file=sys.stderr)
    if expected_ndihedral > 0 and len(dihedrals) < expected_ndihedral * 0.95:
        import sys
        print("WARNING: Dihedral count may be incomplete ({} of {} expected)".format(len(dihedrals), expected_ndihedral), file=sys.stderr)

    return atoms, bonds, angles, dihedrals, total_atoms


def parse_pdb_coords(pdb_path: Path) -> Tuple[List[CoordRecord], Tuple[float, float, float]]:
    lines = read_text_maybe_gz(pdb_path)
    coords = []
    box_ang = None
    for line in lines:
        if line.startswith("CRYST1"):
            parts = line.split()
            box_ang = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif line.startswith(("ATOM", "HETATM")):
            coords.append(
                CoordRecord(
                    atom_id=int(line[6:11]),
                    x_ang=float(line[30:38]),
                    y_ang=float(line[38:46]),
                    z_ang=float(line[46:54]),
                )
            )
    if box_ang is None:
        raise ValueError("PDB is missing CRYST1 box information.")
    return coords, box_ang


def lookup_param(mapping: Dict[Tuple[str, ...], Tuple[float, ...]], labels: Tuple[str, ...]) -> Tuple[float, ...]:
    if labels in mapping:
        return mapping[labels]
    reversed_labels = tuple(reversed(labels))
    if reversed_labels in mapping:
        return mapping[reversed_labels]
    raise KeyError(f"Could not find parameters for labels {labels}")


def atomic_number(atom_type: str) -> int:
    first = atom_type[0].lower()
    if first == "c":
        return 6
    if first == "h":
        return 1
    return 0


def format_gro_atom(resnr: int, resname: str, atom_name: str, atomnr: int, x_nm: float, y_nm: float, z_nm: float) -> str:
    return f"{resnr % 100000:5d}{resname[:5]:<5}{atom_name[:5]:>5}{atomnr % 100000:5d}{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}"


def write_gro(
    output_path: Path,
    coords: List[CoordRecord],
    template_atoms: List[AtomRecord],
    chain_count: int,
    box_ang: Tuple[float, float, float],
) -> None:
    atoms_per_molecule = len(template_atoms)
    residues_per_molecule = max(atom.resid for atom in template_atoms)
    lines = [
        "Generated from EMC PDB/PSF for minimal GROMACS bridge test",
        f"{len(coords):5d}",
    ]
    for global_index, coord in enumerate(coords, start=1):
        local_index = (global_index - 1) % atoms_per_molecule
        molecule_index = (global_index - 1) // atoms_per_molecule
        atom = template_atoms[local_index]
        global_resnr = molecule_index * residues_per_molecule + atom.resid
        lines.append(
            format_gro_atom(
                resnr=global_resnr,
                resname=atom.resname.upper(),
                atom_name=atom.atom_name,
                atomnr=global_index,
                x_nm=coord.x_ang * ANGSTROM_TO_NM,
                y_nm=coord.y_ang * ANGSTROM_TO_NM,
                z_nm=coord.z_ang * ANGSTROM_TO_NM,
            )
        )
    lines.append(
        f"{box_ang[0] * ANGSTROM_TO_NM:10.5f}{box_ang[1] * ANGSTROM_TO_NM:10.5f}{box_ang[2] * ANGSTROM_TO_NM:10.5f}"
    )
    write_text_unix(output_path, "\n".join(lines) + "\n", encoding="ascii")


def build_pair_list(dihedrals: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    # Current bridge assumes an acyclic bonded graph, so every dihedral endpoint
    # pair is also the intended 1-4 pair. If we later add cyclic topologies,
    # this should be replaced with a shortest-path-based 1-4 detector that
    # excludes any pair also connected through 1-2 or 1-3 paths.
    seen = set()
    pairs = []
    for atom_i, _, _, atom_l in dihedrals:
        pair = (atom_i, atom_l) if atom_i < atom_l else (atom_l, atom_i)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def write_itp(
    output_path: Path,
    case_id: str,
    template_atoms: List[AtomRecord],
    bonds: List[Tuple[int, int]],
    pairs: List[Tuple[int, int]],
    angles: List[Tuple[int, int, int]],
    dihedrals: List[Tuple[int, int, int, int]],
    params: Dict[str, dict],
    molecule_name: str,
) -> None:
    lines = [
        f"; Molecule topology derived from EMC outputs for {case_id}",
        "; Assumption: LAMMPS multi/harmonic dihedrals are mapped to GROMACS RB dihedrals",
        "; using psi = phi - 180 deg, so coefficients alternate in sign.",
        "",
        "[ moleculetype ]",
        "; name  nrexcl",
        f"{molecule_name}  3",
        "",
        "[ atoms ]",
        "; nr  type  resnr  residue  atom  cgnr  charge  mass",
    ]

    local_atoms = {atom.atom_id: atom for atom in template_atoms}
    for new_index, atom in enumerate(template_atoms, start=1):
        lines.append(
            f"{new_index:5d}  {atom.atom_type:<4}  {atom.resid:4d}  {atom.resname.upper():<5}  {atom.atom_name:<5}  {new_index:5d}  {atom.charge: .5f}  {atom.mass:.5f}"
        )

    lines.extend(["", "[ bonds ]", "; ai  aj  funct  b0(nm)  kb(kJ/mol/nm^2)"])
    for atom_i, atom_j in bonds:
        labels = (local_atoms[atom_i].atom_type, local_atoms[atom_j].atom_type)
        r0_nm, k_gmx = lookup_param(params["bond_params"], labels)
        lines.append(f"{atom_i:5d}  {atom_j:5d}  1  {r0_nm:.6f}  {k_gmx:.3f}")

    lines.extend(["", "[ pairs ]", "; ai  aj  funct"])
    for atom_i, atom_j in pairs:
        lines.append(f"{atom_i:5d}  {atom_j:5d}  1")

    lines.extend(["", "[ angles ]", "; ai  aj  ak  funct  theta(deg)  ktheta(kJ/mol/rad^2)"])
    for atom_i, atom_j, atom_k in angles:
        labels = (
            local_atoms[atom_i].atom_type,
            local_atoms[atom_j].atom_type,
            local_atoms[atom_k].atom_type,
        )
        theta_deg, k_gmx = lookup_param(params["angle_params"], labels)
        lines.append(f"{atom_i:5d}  {atom_j:5d}  {atom_k:5d}  1  {theta_deg:.3f}  {k_gmx:.3f}")

    lines.extend(["", "[ dihedrals ]", "; ai  aj  ak  al  funct  c0  c1  c2  c3  c4  c5"])
    for atom_i, atom_j, atom_k, atom_l in dihedrals:
        labels = (
            local_atoms[atom_i].atom_type,
            local_atoms[atom_j].atom_type,
            local_atoms[atom_k].atom_type,
            local_atoms[atom_l].atom_type,
        )
        coeffs = lookup_param(params["dihedral_params"], labels)
        coeff_text = "  ".join(f"{value:.6f}" for value in coeffs)
        lines.append(f"{atom_i:5d}  {atom_j:5d}  {atom_k:5d}  {atom_l:5d}  3  {coeff_text}")

    write_text_unix(output_path, "\n".join(lines) + "\n", encoding="ascii")


def write_top(
    output_path: Path,
    itp_name: str,
    molecule_name: str,
    chain_count: int,
    params: Dict[str, dict],
) -> None:
    lines = [
        "; Minimal GROMACS topology generated from EMC outputs",
        "; Scope: topology/coordinate bridge test only, not yet a production-validated force-field translation.",
        "",
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
        "1         3          yes        0.5      0.5",
        "",
        "[ atomtypes ]",
        "; name  at.num  mass     charge  ptype  sigma(nm)  epsilon(kJ/mol)",
    ]

    for atom_type in sorted(params["atom_masses"]):
        sigma_nm, epsilon_kj = lookup_param(params["pair_params"], (atom_type, atom_type))
        lines.append(
            f"{atom_type:<6}{atomic_number(atom_type):6d}{params['atom_masses'][atom_type]:10.5f}{0.0:9.4f}  A  {sigma_nm:10.6f}{epsilon_kj:14.6f}"
        )

    lines.extend(
        [
            "",
            f'#include "{itp_name}"',
            "",
            "[ system ]",
            f"EMC bridge test for {molecule_name}",
            "",
            "[ molecules ]",
            "; name  count",
            f"{molecule_name}  {chain_count}",
        ]
    )
    write_text_unix(output_path, "\n".join(lines) + "\n", encoding="ascii")


def write_em_mdp(output_path: Path, nsteps: int) -> None:
    lines = [
        "integrator              = steep",
        "nsteps                  = %d" % nsteps,
        "emtol                   = 1000.0",
        "emstep                  = 0.001",
        "cutoff-scheme           = Verlet",
        "nstlist                 = 20",
        "verlet-buffer-tolerance = 0.005",
        "coulombtype             = PME",
        "rcoulomb                = 0.95",
        "vdwtype                 = Cut-off",
        "rvdw                    = 0.95",
        "DispCorr                = EnerPres",
        "pbc                     = xyz",
        "constraints             = none",
        "nstenergy               = 10",
        "nstlog                  = 10",
        "",
    ]
    write_text_unix(output_path, "\n".join(lines), encoding="ascii")


def write_submit_script(output_path: Path, case_id: str, cpus_per_task: int, partition: str, time_limit: str) -> None:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={case_id}-gmx",
        f"#SBATCH --partition={partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --output=slurm-%j.out",
        "",
        "set -euo pipefail",
        "",
        "module purge >/dev/null 2>&1",
        "module load intel${PELCB100_REMOTE_PATH}",
        "",
        'WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"',
        'cd "$WORK_DIR"',
        'GMX_BIN="$(command -v gmx_mpi)"',
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        "",
        'echo "GMX_BRIDGE_TEST_START=$(date -Is)"',
        'echo "HOSTNAME=$(hostname)"',
        'echo "PWD=$(pwd)"',
        'echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"',
        'echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unknown}"',
        'echo "GMX_BIN=$GMX_BIN"',
        'echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"',
        'echo "GMX_VERSION_BEGIN"',
        '"$GMX_BIN" --version | sed -n \'1,25p\'',
        'echo "GMX_VERSION_END"',
        "",
        'echo "GROMPP_START=$(date -Is)"',
        f'"$GMX_BIN" grompp -f em_bridge_min.mdp -c {case_id}.gro -p {case_id}.top -o {case_id}_em.tpr -po mdout.mdp -maxwarn 0 > grompp-screen.out 2>&1',
        "",
        'echo "MDRUN_START=$(date -Is)"',
        f'"$GMX_BIN" mdrun -deffnm {case_id}_em -ntomp "$OMP_NUM_THREADS" -pin off > mdrun-screen.out 2>&1',
        "",
        'echo "GMX_BRIDGE_TEST_END=$(date -Is)"',
        'echo "GROMPP_LOG=$PWD/grompp-screen.out"',
        'echo "MDRUN_LOG=$PWD/mdrun-screen.out"',
    ]
    write_text_unix(output_path, "\n".join(lines) + "\n", encoding="ascii")


def write_readme(
    output_path: Path,
    case_meta: dict,
    atoms_per_molecule: int,
    params: Dict[str, dict],
    output_dir: Path,
) -> None:
    monomer_name = case_meta.get('monomer_name', 'alpha-olefin')
    lines = [
        f"# EMC to GROMACS Bridge Test: {case_meta['case_id']}",
        "",
        "This package is the first minimal EMC -> GROMACS continuation test.",
        "",
        "Scope:",
        "- read EMC-generated `PSF/PDB/params`",
        "- generate a self-contained `.gro/.top/.itp` package",
        "- run a short steepest-descent minimization in `GROMACS`",
        "",
        "Current assumptions:",
        "- molecule template is taken from the first chain in the EMC PSF and reused `chain_count` times",
        "- `special_bonds lj/coul 0 0 0.5` is mapped to `nrexcl = 3` and `fudgeLJ = fudgeQQ = 0.5`",
        "- `multi/harmonic` dihedrals are mapped to GROMACS RB dihedrals with sign alternation due to `psi = phi - 180 deg`",
        "- this is a bridge smoke test, not yet a full force-field equivalence validation against the LAMMPS setup",
        "",
        "Case facts:",
        f"- case id: `{case_meta['case_id']}`",
        f"- chemistry: `poly({monomer_name})`",
        f"- chain count: `{case_meta['chain_count']}`",
        f"- template atoms per chain: `{atoms_per_molecule}`",
        f"- output directory: `{output_dir}`",
        "",
        "Generated files:",
        f"- `{case_meta['case_id']}.gro`",
        f"- `{case_meta['case_id']}.top`",
        "- `topol.top`",
        f"- `{case_meta['case_id']}_polymer.itp`",
        "- `case.json`",
        "- `em_bridge_min.mdp`",
        "- `submit_gromacs_bridge_test.sbatch`",
        "",
        "Suggested first command on the cluster:",
        "```bash",
        "sbatch submit_gromacs_bridge_test.sbatch",
        "```",
        "",
    ]
    write_text_unix(output_path, "\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = (args.output_dir or (case_dir / "gromacs_bridge_min_test")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_meta = read_case_metadata(case_dir)
    case_id = case_meta["case_id"]

    # Try remote_build first, fall back to emc_production_build
    remote_build_dir = case_dir / "remote_build"
    if not remote_build_dir.exists():
        emc_prod = case_dir / "emc_production_build"
        if emc_prod.exists():
            remote_build_dir = emc_prod
    psf_path = find_case_file(remote_build_dir, case_id, ".psf.gz")
    pdb_path = find_case_file(remote_build_dir, case_id, ".pdb.gz")
    params_path = find_case_file(remote_build_dir, case_id, ".params")

    params = parse_params(params_path)
    template_atoms, bonds, angles, dihedrals, total_atoms = parse_psf_template(psf_path, case_meta["chain_count"])
    coords, box_ang = parse_pdb_coords(pdb_path)
    if len(coords) != total_atoms:
        raise ValueError(f"PDB atom count {len(coords)} does not match PSF atom count {total_atoms}")

    pairs = build_pair_list(dihedrals)
    molecule_name = case_id.upper()
    itp_name = f"{case_id}_polymer.itp"

    write_gro(output_dir / f"{case_id}.gro", coords, template_atoms, case_meta["chain_count"], box_ang)
    write_itp(output_dir / itp_name, case_id, template_atoms, bonds, pairs, angles, dihedrals, params, molecule_name)
    case_top = output_dir / f"{case_id}.top"
    write_top(case_top, itp_name, molecule_name, case_meta["chain_count"], params)
    shutil.copy2(case_top, output_dir / "topol.top")
    write_text_unix(output_dir / "case.json", json.dumps(case_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_em_mdp(output_dir / "em_bridge_min.mdp", args.nsteps)
    write_submit_script(output_dir / "submit_gromacs_bridge_test.sbatch", case_id, args.cpus_per_task, args.partition, args.time_limit)
    write_readme(output_dir / "README.md", case_meta, len(template_atoms), params, output_dir)

    print(f"Generated GROMACS bridge package in {output_dir}")
    print(f"Template atoms per chain: {len(template_atoms)}")
    print(f"Bonds/pairs/angles/dihedrals per chain: {len(bonds)} / {len(pairs)} / {len(angles)} / {len(dihedrals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
