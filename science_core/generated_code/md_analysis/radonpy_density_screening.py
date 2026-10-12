#!/usr/bin/env python3
"""RadonPy-only density screening for PE-LCB100.

The density interval is derived from public RadonPy PI1070 records only.
PE-LCB100 results and local control simulations are never read by this module.
Structure similarity is reported separately to show how close each PE-LCB case
is to its nearest public reference. It does not change the density interval.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    _RDKIT_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - optional dependency
    Chem = None
    DataStructs = None
    rdBase = None
    rdFingerprintGenerator = None
    _RDKIT_AVAILABLE = False


RADONPY_PI1070_URL = (
    "https://raw.githubusercontent.com/RadonPy/RadonPy/develop${PELCB100_REMOTE_PATH}"
)
REFERENCE_PACK_SCHEMA_VERSION = "radonpy_density_reference_pack.v1"
CASE_MATCH_SCHEMA_VERSION = "pelcb_radonpy_structure_matches.v1"
ALGORITHM_VERSION = "full_chain_atom_pair_count_tanimoto.v1"
AUTOMATIC_MATCH_SCHEMA_VERSION = "automatic_radonpy_density_match.v1"
TARGET_TEMPERATURE_K = 400.0
MAX_TEMPERATURE_DELTA_K = 150.0
EXPECTED_REFERENCE_COUNT = 40
PASS_LOW_G_CM3 = 0.73
PASS_HIGH_G_CM3 = 0.87
RAW_LOW_G_CM3 = 0.7315756468839724
RAW_HIGH_G_CM3 = 0.866582790659682

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PACK = SCRIPT_DIR / "data" / "radonpy_density_reference_pack_v1.json"
DEFAULT_REFERENCE_CSV = SCRIPT_DIR / "data" / "radonpy_density_reference_records_v1.csv"
DEFAULT_CASE_MATCH_PACK = SCRIPT_DIR / "data" / "pelcb100_radonpy_structure_matches_v1.json"
DEFAULT_CASE_MATCH_CSV = SCRIPT_DIR / "data" / "pelcb100_radonpy_structure_matches_v1.csv"


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        raise RuntimeError(
            "RadonPy structure matching requires RDKit, but RDKit is not installed."
        )


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _monomer_sort_key(value: str) -> tuple[int, str]:
    digits = "".join(character for character in value if character.isdigit())
    return (int(digits) if digits else 10**9, value)


def _is_saturated_acyclic_hydrocarbon_repeat(smiles: str) -> bool:
    _require_rdkit()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False
    real_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 0]
    terminal_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if not real_atoms or not any(atom.GetAtomicNum() == 6 for atom in real_atoms):
        return False
    if any(atom.GetAtomicNum() not in {1, 6} for atom in real_atoms):
        return False
    if len(terminal_atoms) != 2 or any(atom.GetDegree() != 1 for atom in terminal_atoms):
        return False
    if molecule.GetRingInfo().NumRings() != 0:
        return False
    return all(
        not bond.GetIsAromatic() and bond.GetBondType() == Chem.BondType.SINGLE
        for bond in molecule.GetBonds()
    )


def _temperature_adjusted_bounds(row: dict[str, str]) -> tuple[float, float] | None:
    source_temperature = _as_float(row.get("temp"))
    density_min = _as_float(row.get("density_min"))
    density_max = _as_float(row.get("density_max"))
    alpha = _as_float(row.get("volume_expansion"))
    alpha_std = _as_float(row.get("volume_expansion_std"))
    if None in (source_temperature, density_min, density_max, alpha, alpha_std):
        return None
    assert source_temperature is not None
    assert density_min is not None
    assert density_max is not None
    assert alpha is not None
    assert alpha_std is not None
    delta_temperature = TARGET_TEMPERATURE_K - source_temperature
    if delta_temperature < 0 or delta_temperature > MAX_TEMPERATURE_DELTA_K:
        return None
    if alpha <= 0 or alpha_std < 0 or density_min <= 0 or density_max <= 0:
        return None
    lower_denominator = 1.0 + (alpha + 2.0 * alpha_std) * delta_temperature
    upper_alpha = max(0.0, alpha - 2.0 * alpha_std)
    upper_denominator = 1.0 + upper_alpha * delta_temperature
    if lower_denominator <= 0 or upper_denominator <= 0:
        return None
    return density_min / lower_denominator, density_max / upper_denominator


def select_density_references(csv_bytes: bytes) -> list[dict[str, Any]]:
    """Return saturated, acyclic hydrocarbon records usable at 400 K."""

    text = csv_bytes.decode("utf-8-sig")
    references: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(text)):
        smiles = str(row.get("smiles") or "").strip()
        density = _as_float(row.get("density"))
        adjusted_bounds = _temperature_adjusted_bounds(row)
        if density is None or adjusted_bounds is None:
            continue
        if not _is_saturated_acyclic_hydrocarbon_repeat(smiles):
            continue
        source_temperature = float(row["temp"])
        delta_temperature = TARGET_TEMPERATURE_K - source_temperature
        alpha = float(row["volume_expansion"])
        adjusted_density = density / (1.0 + alpha * delta_temperature)
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        monomer_id = str(row.get("monomer_ID") or "unknown").strip()
        references.append(
            {
                "reference_id": monomer_id,
                "repeat_unit_smiles": smiles,
                "repeat_unit_carbon_count": sum(
                    atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()
                ),
                "source_temperature_K": source_temperature,
                "source_pressure_bar": _as_float(row.get("press")),
                "source_density_g_cm3": density,
                "source_density_min_g_cm3": float(row["density_min"]),
                "source_density_max_g_cm3": float(row["density_max"]),
                "source_density_std_g_cm3": _as_float(row.get("density_std")),
                "source_density_count": int(float(row.get("density_count") or 0)),
                "volume_expansion_coefficient_K_inv": alpha,
                "volume_expansion_coefficient_std_K_inv": float(
                    row["volume_expansion_std"]
                ),
                "density_400K_g_cm3": adjusted_density,
                "density_400K_low_g_cm3": adjusted_bounds[0],
                "density_400K_high_g_cm3": adjusted_bounds[1],
                "force_field": "GAFF2",
                "calculation_software": "LAMMPS",
            }
        )
    return sorted(references, key=lambda item: _monomer_sort_key(item["reference_id"]))


def derive_density_range(references: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(references)
    if not records:
        raise ValueError("No eligible RadonPy density records were supplied.")
    raw_low = min(float(record["density_400K_low_g_cm3"]) for record in records)
    raw_high = max(float(record["density_400K_high_g_cm3"]) for record in records)
    pass_low = math.floor(raw_low * 100.0) / 100.0
    pass_high = math.ceil(raw_high * 100.0) / 100.0
    return {
        "raw_range_g_cm3": [raw_low, raw_high],
        "pass_range_g_cm3": [pass_low, pass_high],
        "rounding": "lower bound rounded down and upper bound rounded up to 0.01 g/cm3",
        "reference_count": len(records),
    }


def build_reference_pack(
    csv_bytes: bytes,
    *,
    downloaded_at_utc: str | None = None,
    source_url: str = RADONPY_PI1070_URL,
) -> dict[str, Any]:
    references = select_density_references(csv_bytes)
    density_range = derive_density_range(references)
    return {
        "schema_version": REFERENCE_PACK_SCHEMA_VERSION,
        "source": {
            "name": "RadonPy PI1070.csv",
            "url": source_url,
            "downloaded_at_utc": downloaded_at_utc
            or datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        },
        "selection": {
            "elements": "carbon and hydrogen only",
            "bonding": "saturated and acyclic",
            "required_fields": [
                "density",
                "density_min",
                "density_max",
                "volume_expansion",
                "volume_expansion_std",
            ],
            "target_temperature_K": TARGET_TEMPERATURE_K,
            "maximum_temperature_delta_K": MAX_TEMPERATURE_DELTA_K,
        },
        "density_conversion": {
            "lower_formula": (
                "density_min / [1 + (alpha + 2*alpha_std)*(400 - source_temperature)]"
            ),
            "upper_formula": (
                "density_max / [1 + max(0, alpha - 2*alpha_std)*(400 - source_temperature)]"
            ),
        },
        "density_range": density_range,
        "records": references,
    }


def load_reference_pack(path: str | Path = DEFAULT_REFERENCE_PACK) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_case_match_pack(path: str | Path = DEFAULT_CASE_MATCH_PACK) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _repeat_count(target_carbons: int, repeat_carbons: int) -> int:
    if target_carbons <= 0 or repeat_carbons <= 0:
        raise ValueError("Carbon counts must be positive.")
    estimate = target_carbons / repeat_carbons
    candidates = {max(1, math.floor(estimate)), max(1, math.ceil(estimate))}
    return min(candidates, key=lambda count: (abs(count * repeat_carbons - target_carbons), count))


def expand_repeat_unit(repeat_unit_smiles: str, target_carbons: int) -> tuple[Chem.Mol, dict[str, int]]:
    _require_rdkit()
    """Expand a two-ended repeat unit to the closest target carbon count."""

    repeat = Chem.MolFromSmiles(repeat_unit_smiles)
    if repeat is None:
        raise ValueError(f"Invalid repeat-unit SMILES: {repeat_unit_smiles}")
    dummy_indices = [atom.GetIdx() for atom in repeat.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_indices) != 2:
        raise ValueError("A RadonPy repeat unit must contain exactly two terminal atoms.")
    if any(repeat.GetAtomWithIdx(index).GetDegree() != 1 for index in dummy_indices):
        raise ValueError("Each terminal atom must have exactly one neighbor.")
    repeat_carbons = sum(atom.GetAtomicNum() == 6 for atom in repeat.GetAtoms())
    copies = _repeat_count(target_carbons, repeat_carbons)
    atom_count = repeat.GetNumAtoms()
    combined = repeat
    for _ in range(1, copies):
        combined = Chem.CombineMols(combined, repeat)
    editable = Chem.RWMol(combined)
    left_dummy, right_dummy = sorted(dummy_indices)
    left_neighbor = repeat.GetAtomWithIdx(left_dummy).GetNeighbors()[0].GetIdx()
    right_neighbor = repeat.GetAtomWithIdx(right_dummy).GetNeighbors()[0].GetIdx()
    for copy_index in range(copies - 1):
        editable.AddBond(
            copy_index * atom_count + right_neighbor,
            (copy_index + 1) * atom_count + left_neighbor,
            Chem.BondType.SINGLE,
        )
    terminal_dummies = {left_dummy, (copies - 1) * atom_count + right_dummy}
    removable_dummies = [
        copy_index * atom_count + dummy_index
        for copy_index in range(copies)
        for dummy_index in dummy_indices
        if copy_index * atom_count + dummy_index not in terminal_dummies
    ]
    for atom_index in sorted(removable_dummies, reverse=True):
        editable.RemoveAtom(atom_index)
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    expanded_carbons = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    return molecule, {
        "repeat_count": copies,
        "repeat_unit_carbon_count": repeat_carbons,
        "expanded_carbon_count": expanded_carbons,
    }


def atom_pair_count_tanimoto(target: Chem.Mol, reference: Chem.Mol) -> float:
    _require_rdkit()
    generator = rdFingerprintGenerator.GetAtomPairGenerator()
    target_fingerprint = generator.GetSparseCountFingerprint(target)
    reference_fingerprint = generator.GetSparseCountFingerprint(reference)
    return float(DataStructs.TanimotoSimilarity(target_fingerprint, reference_fingerprint))


def structure_similarity_tier(score: float) -> str:
    if score >= 0.90:
        return "high"
    if score >= 0.80:
        return "medium"
    return "distant"


def _first_structure_value(
    payloads: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
) -> tuple[str | None, str | None]:
    for payload in payloads:
        for key in keys:
            value = str(payload.get(key) or "").strip()
            if value:
                return value, key
    return None, None


def _context_structure(context: dict[str, Any]) -> tuple[str | None, str | None]:
    raw_context = context.get("raw_context")
    payloads = [context]
    if isinstance(raw_context, dict):
        payloads.append(raw_context)
    return _first_structure_value(
        payloads,
        (
            "canonical_chain_smiles",
            "chain_smiles",
            "polymer_chain_smiles",
            "oligomer_smiles",
            "canonical_segment_smiles",
            "normalized_segment_smiles",
            "repeat_unit_smiles",
            "canonical_repeat_unit_smiles",
            "poly_smiles",
            "smiles",
        ),
    )


def _target_carbon_count(context: dict[str, Any], target: Chem.Mol) -> int | None:
    raw_context = context.get("raw_context")
    payloads = [context]
    if isinstance(raw_context, dict):
        payloads.append(raw_context)
    for payload in payloads:
        for key in (
            "actual_total_carbons_per_chain",
            "target_total_carbons_per_chain",
            "chain_carbon_count",
        ):
            value = _as_float(payload.get(key))
            if value is not None and value > 0:
                return int(round(value))

    repeat_carbons = sum(atom.GetAtomicNum() == 6 for atom in target.GetAtoms())
    if repeat_carbons <= 0:
        return None
    for payload in payloads:
        for key in ("degree_of_polymerization", "polymerization_degree", "DP", "dp"):
            value = _as_float(payload.get(key))
            if value is not None and value > 0:
                return int(round(repeat_carbons * value))
    return None


def _comparison_molecule(smiles: str, target_carbons: int | None) -> Chem.Mol:
    _require_rdkit()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid polymer structure: {smiles}")
    terminal_count = sum(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms())
    if target_carbons and terminal_count == 2:
        expanded, _ = expand_repeat_unit(smiles, target_carbons)
        return expanded
    return molecule


def _record_structure(record: dict[str, Any]) -> str | None:
    raw_fields = record.get("raw_fields")
    payloads = [record]
    if isinstance(raw_fields, dict):
        payloads.append(raw_fields)
    value, _ = _first_structure_value(
        payloads,
        (
            "repeat_unit_smiles",
            "canonical_repeat_unit_smiles",
            "poly_smiles",
            "smiles",
        ),
    )
    return value


def match_context_to_density_references(
    context: dict[str, Any],
    records: Iterable[dict[str, Any]],
    *,
    min_similarity: float = 0.80,
    high_similarity: float = 0.90,
    pressure_tolerance_bar: float = 1.0,
) -> dict[str, Any]:
    """Find structure-matched RadonPy records for a new melt polymer.

    Structure similarity selects candidate records. Temperature, pressure,
    state and record completeness are checked separately. Only high-similarity
    records can support automatic activation. Medium-similarity records are
    retained as a WARN input for the audit layer.
    """

    if not 0.0 <= min_similarity <= high_similarity <= 1.0:
        raise ValueError(
            "similarity thresholds must satisfy 0 <= min_similarity <= high_similarity <= 1"
        )

    target_smiles, target_field = _context_structure(context)
    if not target_smiles:
        return {
            "schema_version": AUTOMATIC_MATCH_SCHEMA_VERSION,
            "status": "warn",
            "rule_verdict": "WARN",
            "warn_source": "density_rule_profile_missing",
            "audit_required": True,
            "reason": "target_polymer_structure_missing",
            "target_structure_field": None,
            "reference_records": [],
            "candidates": [],
            "profile_activatable": False,
        }
    try:
        parsed_target = Chem.MolFromSmiles(target_smiles)
        if parsed_target is None:
            raise ValueError("invalid target structure")
        target_carbons = _target_carbon_count(context, parsed_target)
        target = _comparison_molecule(target_smiles, target_carbons)
    except ValueError:
        return {
            "schema_version": AUTOMATIC_MATCH_SCHEMA_VERSION,
            "status": "warn",
            "rule_verdict": "WARN",
            "warn_source": "density_rule_profile_missing",
            "audit_required": True,
            "reason": "target_polymer_structure_invalid",
            "target_structure_field": target_field,
            "reference_records": [],
            "candidates": [],
            "profile_activatable": False,
        }

    raw_context = context.get("raw_context")
    target_context = raw_context if isinstance(raw_context, dict) else context
    target_state = str(context.get("state") or target_context.get("state") or "melt").lower()
    target_temperature = _as_float(
        context.get("temperature_K")
        if context.get("temperature_K") is not None
        else target_context.get("temperature_K")
    )
    target_pressure = _as_float(
        context.get("pressure_bar")
        if context.get("pressure_bar") is not None
        else target_context.get("pressure_bar")
    )
    if target_state != "melt":
        return {
            "schema_version": AUTOMATIC_MATCH_SCHEMA_VERSION,
            "status": "warn",
            "rule_verdict": "WARN",
            "warn_source": "density_rule_profile_missing",
            "audit_required": True,
            "reason": "target_state_not_melt",
            "target_structure_field": target_field,
            "reference_records": [],
            "candidates": [],
            "profile_activatable": False,
        }

    candidates: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("source_type") or "").lower() != "radonpy":
            continue
        reference_smiles = _record_structure(record)
        if not reference_smiles:
            continue
        record_temperature = _as_float(record.get("temperature_K"))
        record_pressure = _as_float(record.get("pressure_bar"))
        state_ok = str(record.get("state") or "").lower() == "melt"
        temperature_ok = (
            target_temperature is not None
            and record_temperature is not None
            and abs(target_temperature - record_temperature) <= 1e-6
        )
        pressure_ok = (
            target_pressure is None
            or record_pressure is None
            or abs(target_pressure - record_pressure) <= pressure_tolerance_bar
        )
        bounds_ok = _as_float(record.get("value_min")) is not None and _as_float(
            record.get("value_max")
        ) is not None
        value_ok = _as_float(record.get("value")) is not None and bool(record.get("units"))
        try:
            reference = _comparison_molecule(reference_smiles, target_carbons)
            score = atom_pair_count_tanimoto(target, reference)
        except ValueError:
            continue
        tier = structure_similarity_tier(score)
        condition_reasons: list[str] = []
        if not state_ok:
            condition_reasons.append("reference_state_not_melt")
        if not temperature_ok:
            condition_reasons.append("reference_temperature_not_adjusted_to_target")
        if not pressure_ok:
            condition_reasons.append("reference_pressure_mismatch")
        if not value_ok:
            condition_reasons.append("reference_density_or_units_missing")
        if not bounds_ok:
            condition_reasons.append("reference_density_bounds_missing")
        candidates.append(
            {
                "reference_id": str(
                    record.get("source_id")
                    or record.get("reference_id")
                    or record.get("independent_key")
                    or ""
                ),
                "structure_similarity_score": round(score, 6),
                "structure_similarity_tier": tier,
                "conditions_match": not condition_reasons,
                "condition_reasons": condition_reasons,
                "record": record,
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item["structure_similarity_score"]),
            str(item["reference_id"]),
        )
    )
    high_candidates = [
        item
        for item in candidates
        if item["conditions_match"]
        and float(item["structure_similarity_score"]) >= high_similarity
    ]
    medium_candidates = [
        item
        for item in candidates
        if item["conditions_match"]
        and min_similarity <= float(item["structure_similarity_score"]) < high_similarity
    ]
    selected = high_candidates[:10]
    annotated_records: list[dict[str, Any]] = []
    for item in selected:
        annotated = dict(item["record"])
        annotated["structure_similarity_score"] = item["structure_similarity_score"]
        annotated["structure_similarity_tier"] = item["structure_similarity_tier"]
        annotated["nearest_radonpy_reference"] = item["reference_id"]
        annotated["structure_similarity_method"] = (
            "unhashed sparse atom-pair count fingerprint and count Tanimoto"
        )
        annotated_records.append(annotated)

    independent_keys = {
        str(record.get("independent_key") or record.get("source_id") or "")
        for record in annotated_records
        if str(record.get("independent_key") or record.get("source_id") or "")
    }
    single_record_screening = bool(
        len(annotated_records) == 1
        and annotated_records[0].get("single_source_screening_eligible")
    )
    activatable = len(independent_keys) >= 2 or single_record_screening
    if activatable:
        status = "ready_for_profile_proposal"
        reason = "high_similarity_records_with_complete_conditions"
    elif high_candidates:
        status = "warn"
        reason = "high_similarity_reference_support_is_insufficient"
    elif medium_candidates:
        status = "warn"
        reason = "only_medium_similarity_references_found"
    else:
        status = "warn"
        reason = "no_compatible_similar_reference"

    public_candidates = [
        {key: value for key, value in item.items() if key != "record"}
        for item in candidates[:10]
    ]
    return {
        "schema_version": AUTOMATIC_MATCH_SCHEMA_VERSION,
        "status": status,
        "rule_verdict": None if activatable else "WARN",
        "warn_source": None if activatable else "density_rule_profile_missing",
        "audit_required": not activatable,
        "reason": reason,
        "target_structure_field": target_field,
        "target_carbon_count": target_carbons,
        "min_similarity": min_similarity,
        "high_similarity": high_similarity,
        "nearest_reference": public_candidates[0] if public_candidates else None,
        "high_similarity_reference_count": len(high_candidates),
        "medium_similarity_reference_count": len(medium_candidates),
        "reference_records": annotated_records,
        "candidates": public_candidates,
        "profile_activatable": activatable,
    }


def match_case_to_references(
    case: dict[str, Any],
    references: Iterable[dict[str, Any]],
    *,
    density_range: list[float] | tuple[float, float] = (PASS_LOW_G_CM3, PASS_HIGH_G_CM3),
) -> dict[str, Any]:
    target_smiles = str(case.get("monomer_smiles") or case.get("chain_smiles") or "").strip()
    target = Chem.MolFromSmiles(target_smiles)
    if target is None:
        raise ValueError(f"Invalid target structure for {case.get('case_id')}")
    target_carbons = int(
        float(
            case.get("actual_total_carbons_per_chain")
            or case.get("target_total_carbons_per_chain")
            or sum(atom.GetAtomicNum() == 6 for atom in target.GetAtoms())
        )
    )
    candidates: list[dict[str, Any]] = []
    for record in references:
        expanded, expansion = expand_repeat_unit(record["repeat_unit_smiles"], target_carbons)
        score = atom_pair_count_tanimoto(target, expanded)
        candidates.append(
            {
                "reference_id": record["reference_id"],
                "structure_similarity_score": score,
                **expansion,
            }
        )
    if not candidates:
        raise ValueError("No RadonPy references were available for structure matching.")
    nearest = min(
        candidates,
        key=lambda item: (
            -float(item["structure_similarity_score"]),
            _monomer_sort_key(str(item["reference_id"])),
        ),
    )
    score = float(nearest["structure_similarity_score"])
    return {
        "case_id": str(case.get("case_id") or ""),
        "structure_similarity_score": round(score, 6),
        "structure_similarity_tier": structure_similarity_tier(score),
        "nearest_radonpy_reference": nearest["reference_id"],
        "target_carbon_count": target_carbons,
        "reference_repeat_count": nearest["repeat_count"],
        "reference_expanded_carbon_count": nearest["expanded_carbon_count"],
        "density_screening_range": [float(density_range[0]), float(density_range[1])],
        "density_range_source": "radonpy_pi1070_saturated_acyclic_40_record_envelope_400K",
    }


def match_manifest_to_references(
    manifest_rows: Iterable[dict[str, Any]],
    reference_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    density_range = reference_pack["density_range"]["pass_range_g_cm3"]
    return [
        match_case_to_references(row, reference_pack["records"], density_range=density_range)
        for row in manifest_rows
    ]


def build_case_match_pack(
    manifest_rows: Iterable[dict[str, Any]],
    reference_pack: dict[str, Any],
) -> dict[str, Any]:
    matches = match_manifest_to_references(manifest_rows, reference_pack)
    tier_counts = {
        tier: sum(match["structure_similarity_tier"] == tier for match in matches)
        for tier in ("high", "medium", "distant")
    }
    return {
        "schema_version": CASE_MATCH_SCHEMA_VERSION,
        "algorithm": {
            "version": ALGORITHM_VERSION,
            "rdkit_version": rdBase.rdkitVersion,
            "fingerprint": "unhashed sparse atom-pair count fingerprint",
            "similarity": "count Tanimoto",
            "reference_expansion": "repeat unit expanded to the closest target carbon count",
            "tier_thresholds": {"high": ">=0.90", "medium": "0.80-0.90", "distant": "<0.80"},
        },
        "source_sha256": reference_pack["source"]["sha256"],
        "density_screening_range": reference_pack["density_range"]["pass_range_g_cm3"],
        "case_count": len(matches),
        "tier_counts": tier_counts,
        "matches": matches,
    }
