#!/usr/bin/env python3
"""Compact polymer-chain descriptions and deterministic atomistic expansion.

The planning model supplies repeat-unit templates, a compact sequence rule and
optional branch rules.  The full molecular graph is created only when the
registered builder materializes simulation input, so the planning payload does
not grow with polymerization degree.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from typing import Any

try:
    from rdkit import Chem  # type: ignore
except Exception:  # pragma: no cover - depends on the local runtime.
    Chem = None  # type: ignore


ALLOWED_SEQUENCE_MODES = frozenset(
    {"homopolymer", "periodic", "alternating", "block", "random"}
)


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    try:
        result = int(value)
    except Exception as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return result


def _identifier(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if not all(character.isalnum() or character in "_.-" for character in text):
        raise ValueError(f"{field} must be a simple identifier")
    return text


def _parse_template(
    item: dict[str, Any],
    *,
    field: str,
    required_dummy_maps: tuple[int, ...],
) -> tuple[dict[str, Any], Any]:
    if Chem is None:
        raise ValueError("RDKit is required for template-based polymer construction")
    template_id = _identifier(item.get("template_id") or item.get("id"), f"{field}.template_id")
    smiles = str(item.get("smiles") or "").strip()
    if not smiles:
        raise ValueError(f"{field}[{template_id}].smiles is required")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"{field}[{template_id}].smiles is not RDKit-parseable")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError(f"{field}[{template_id}] must describe one connected template")

    dummy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    dummy_maps = [int(atom.GetAtomMapNum()) for atom in dummy_atoms]
    if sorted(dummy_maps) != sorted(required_dummy_maps):
        expected = ", ".join(str(value) for value in required_dummy_maps)
        raise ValueError(
            f"{field}[{template_id}] must contain dummy connection maps {expected} exactly once"
        )
    for atom in dummy_atoms:
        if atom.GetDegree() != 1:
            raise ValueError(f"{field}[{template_id}] connection markers must have one neighbor")
        bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), atom.GetNeighbors()[0].GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError(f"{field}[{template_id}] connection bonds must be single bonds")

    real_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 0]
    if not real_atoms:
        raise ValueError(f"{field}[{template_id}] must contain at least one real atom")
    normalized = {
        "template_id": template_id,
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
        "real_atom_count": len(real_atoms),
        "carbon_count": sum(atom.GetSymbol() == "C" for atom in real_atoms),
        "elements": sorted({atom.GetSymbol() for atom in real_atoms}),
    }
    return normalized, molecule


def _normalize_sequence(raw: Any, template_ids: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("sequence must be an object")
    mode = str(raw.get("mode") or "").strip().lower()
    aliases = {"explicit_pattern": "periodic", "blocks": "block"}
    mode = aliases.get(mode, mode)
    if mode not in ALLOWED_SEQUENCE_MODES:
        raise ValueError(f"sequence.mode must be one of {sorted(ALLOWED_SEQUENCE_MODES)}")

    normalized: dict[str, Any] = {"mode": mode}
    if mode == "homopolymer":
        template_id = _identifier(raw.get("template_id"), "sequence.template_id")
        repeat_units = _integer(raw.get("repeat_units"), "sequence.repeat_units", minimum=1)
        normalized.update(template_id=template_id, repeat_units=repeat_units)
    elif mode in {"periodic", "alternating"}:
        pattern = raw.get("pattern")
        if not isinstance(pattern, list) or not pattern:
            raise ValueError("sequence.pattern must be a nonempty list")
        pattern_ids = [_identifier(value, "sequence.pattern") for value in pattern]
        if mode == "alternating" and len(pattern_ids) < 2:
            raise ValueError("an alternating sequence requires at least two templates")
        repeat_units = _integer(raw.get("repeat_units"), "sequence.repeat_units", minimum=1)
        normalized.update(pattern=pattern_ids, repeat_units=repeat_units)
    elif mode == "block":
        blocks = raw.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("sequence.blocks must be a nonempty list")
        normalized_blocks: list[dict[str, Any]] = []
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                raise ValueError(f"sequence.blocks[{index}] must be an object")
            normalized_blocks.append(
                {
                    "template_id": _identifier(
                        block.get("template_id"), f"sequence.blocks[{index}].template_id"
                    ),
                    "count": _integer(
                        block.get("count"), f"sequence.blocks[{index}].count", minimum=1
                    ),
                }
            )
        normalized.update(
            blocks=normalized_blocks,
            repeat_units=sum(block["count"] for block in normalized_blocks),
        )
    else:
        composition = raw.get("composition")
        if not isinstance(composition, dict) or not composition:
            raise ValueError("sequence.composition must be a nonempty object")
        fractions: dict[str, float] = {}
        for key, value in composition.items():
            template_id = _identifier(key, "sequence.composition template")
            try:
                fraction = float(value)
            except Exception as exc:
                raise ValueError(f"sequence.composition[{template_id}] must be numeric") from exc
            if not math.isfinite(fraction) or fraction <= 0:
                raise ValueError(f"sequence.composition[{template_id}] must be > 0")
            fractions[template_id] = fraction
        total = sum(fractions.values())
        fractions = {key: fractions[key] / total for key in sorted(fractions)}
        repeat_units = _integer(raw.get("repeat_units"), "sequence.repeat_units", minimum=1)
        seed = _integer(raw.get("seed", 0), "sequence.seed", minimum=0)
        normalized.update(composition=fractions, repeat_units=repeat_units, seed=seed)

    referenced: set[str]
    if mode == "homopolymer":
        referenced = {normalized["template_id"]}
    elif mode in {"periodic", "alternating"}:
        referenced = set(normalized["pattern"])
    elif mode == "block":
        referenced = {block["template_id"] for block in normalized["blocks"]}
    else:
        referenced = set(normalized["composition"])
    unknown = sorted(referenced - template_ids)
    if unknown:
        raise ValueError(f"sequence references unknown templates: {unknown}")
    return normalized


def _sequence_counts(sequence: dict[str, Any]) -> dict[str, int]:
    mode = sequence["mode"]
    total = int(sequence["repeat_units"])
    if mode == "homopolymer":
        return {str(sequence["template_id"]): total}
    if mode in {"periodic", "alternating"}:
        pattern = list(sequence["pattern"])
        quotient, remainder = divmod(total, len(pattern))
        counts = Counter(pattern * quotient + pattern[:remainder])
        return dict(counts)
    if mode == "block":
        counts: Counter[str] = Counter()
        for block in sequence["blocks"]:
            counts[str(block["template_id"])] += int(block["count"])
        return dict(counts)

    fractions = dict(sequence["composition"])
    raw_counts = {key: total * value for key, value in fractions.items()}
    counts = {key: int(math.floor(value)) for key, value in raw_counts.items()}
    remainder = total - sum(counts.values())
    order = sorted(fractions, key=lambda key: (-(raw_counts[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _sequence_ids(sequence: dict[str, Any]) -> list[str]:
    mode = sequence["mode"]
    total = int(sequence["repeat_units"])
    if mode == "homopolymer":
        return [str(sequence["template_id"])] * total
    if mode in {"periodic", "alternating"}:
        pattern = [str(value) for value in sequence["pattern"]]
        return [pattern[index % len(pattern)] for index in range(total)]
    if mode == "block":
        result: list[str] = []
        for block in sequence["blocks"]:
            result.extend([str(block["template_id"])] * int(block["count"]))
        return result
    result = [key for key, count in _sequence_counts(sequence).items() for _ in range(count)]
    random.Random(int(sequence["seed"])).shuffle(result)
    return result


def _normalize_branch_positions(raw: dict[str, Any], repeat_units: int, field: str) -> list[int]:
    if "positions" in raw:
        values = raw.get("positions")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field}.positions must be a nonempty list")
        positions = [_integer(value, f"{field}.positions", minimum=1) for value in values]
    else:
        placement = raw.get("placement")
        if not isinstance(placement, dict):
            raise ValueError(f"{field} requires positions or placement")
        mode = str(placement.get("mode") or "").strip().lower()
        count = _integer(placement.get("count"), f"{field}.placement.count", minimum=1)
        start = _integer(placement.get("start", 1), f"{field}.placement.start", minimum=1)
        end = _integer(
            placement.get("end", repeat_units), f"{field}.placement.end", minimum=start
        )
        if end > repeat_units:
            raise ValueError(f"{field}.placement.end exceeds the backbone length")
        available = list(range(start, end + 1))
        if count > len(available):
            raise ValueError(f"{field}.placement.count exceeds available backbone positions")
        if mode == "uniform":
            if count == 1:
                positions = [available[len(available) // 2]]
            else:
                positions = [
                    start + round(index * (end - start) / (count - 1))
                    for index in range(count)
                ]
        elif mode == "random":
            seed = _integer(placement.get("seed", 0), f"{field}.placement.seed", minimum=0)
            positions = sorted(random.Random(seed).sample(available, count))
        else:
            raise ValueError(f"{field}.placement.mode must be uniform or random")
    if len(set(positions)) != len(positions):
        raise ValueError(f"{field} contains duplicate backbone positions")
    if min(positions) < 1 or max(positions) > repeat_units:
        raise ValueError(f"{field} positions must be within 1..{repeat_units}")
    return sorted(positions)


def normalize_template_plan(candidate: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a compact polymer description."""

    templates = candidate.get("repeat_unit_templates")
    if not isinstance(templates, list) or not templates:
        raise ValueError("repeat_unit_templates must be a nonempty list")
    normalized_templates: list[dict[str, Any]] = []
    template_molecules: dict[str, Any] = {}
    for index, item in enumerate(templates):
        if not isinstance(item, dict):
            raise ValueError(f"repeat_unit_templates[{index}] must be an object")
        normalized, molecule = _parse_template(
            item,
            field="repeat_unit_templates",
            required_dummy_maps=(1, 2),
        )
        if normalized["template_id"] in template_molecules:
            raise ValueError(f"duplicate repeat-unit template: {normalized['template_id']}")
        normalized_templates.append(normalized)
        template_molecules[normalized["template_id"]] = molecule

    sequence = _normalize_sequence(candidate.get("sequence"), set(template_molecules))
    repeat_units = int(sequence["repeat_units"])

    branch_templates_raw = candidate.get("branch_templates") or []
    if not isinstance(branch_templates_raw, list):
        raise ValueError("branch_templates must be a list")
    normalized_branch_templates: list[dict[str, Any]] = []
    branch_molecules: dict[str, Any] = {}
    for index, item in enumerate(branch_templates_raw):
        if not isinstance(item, dict):
            raise ValueError(f"branch_templates[{index}] must be an object")
        normalized, molecule = _parse_template(
            item,
            field="branch_templates",
            required_dummy_maps=(1,),
        )
        if normalized["template_id"] in branch_molecules:
            raise ValueError(f"duplicate branch template: {normalized['template_id']}")
        normalized_branch_templates.append(normalized)
        branch_molecules[normalized["template_id"]] = molecule

    branch_rules_raw = candidate.get("branch_rules") or []
    if not isinstance(branch_rules_raw, list):
        raise ValueError("branch_rules must be a list")
    normalized_branch_rules: list[dict[str, Any]] = []
    occupied_sites: set[tuple[int, int]] = set()
    for index, rule in enumerate(branch_rules_raw):
        field = f"branch_rules[{index}]"
        if not isinstance(rule, dict):
            raise ValueError(f"{field} must be an object")
        template_id = _identifier(rule.get("template_id"), f"{field}.template_id")
        if template_id not in branch_molecules:
            raise ValueError(f"{field} references an unknown branch template")
        atom_index = _integer(rule.get("backbone_atom_index"), f"{field}.backbone_atom_index")
        positions = _normalize_branch_positions(rule, repeat_units, field)
        sequence_ids = _sequence_ids(sequence)
        for position in positions:
            unit_template = next(
                item for item in normalized_templates if item["template_id"] == sequence_ids[position - 1]
            )
            if atom_index >= int(unit_template["real_atom_count"]):
                raise ValueError(
                    f"{field}.backbone_atom_index is outside repeat unit at position {position}"
                )
            site = (position, atom_index)
            if site in occupied_sites:
                raise ValueError(f"multiple branches target the same backbone atom: {site}")
            occupied_sites.add(site)
        normalized_branch_rules.append(
            {
                "template_id": template_id,
                "backbone_atom_index": atom_index,
                "positions": positions,
            }
        )

    template_by_id = {item["template_id"]: item for item in normalized_templates}
    branch_by_id = {item["template_id"]: item for item in normalized_branch_templates}
    counts = _sequence_counts(sequence)
    carbon_count = sum(count * int(template_by_id[key]["carbon_count"]) for key, count in counts.items())
    atom_count = sum(count * int(template_by_id[key]["real_atom_count"]) for key, count in counts.items())
    elements = {element for item in normalized_templates for element in item["elements"]}
    for rule in normalized_branch_rules:
        branch = branch_by_id[rule["template_id"]]
        count = len(rule["positions"])
        carbon_count += count * int(branch["carbon_count"])
        atom_count += count * int(branch["real_atom_count"])
        elements.update(branch["elements"])

    compact_seed = {
        "repeat_unit_templates": normalized_templates,
        "sequence": sequence,
        "branch_templates": normalized_branch_templates,
        "branch_rules": normalized_branch_rules,
    }
    hash_seed = {
        **compact_seed,
        "repeat_unit_templates": [
            {**item, "smiles": item["canonical_smiles"]} for item in normalized_templates
        ],
        "branch_templates": [
            {**item, "smiles": item["canonical_smiles"]}
            for item in normalized_branch_templates
        ],
    }
    design_hash = hashlib.sha256(
        json.dumps(hash_seed, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    return {
        "schema_version": "compact_polymer_template.v1",
        **compact_seed,
        "repeat_units_per_chain": repeat_units,
        "actual_total_carbons_per_chain": carbon_count,
        "estimated_real_atoms_per_chain": atom_count,
        "structure_elements": sorted(elements),
        "template_design_hash": design_hash,
    }


def _dummy_neighbor(molecule: Any, atom_map: int) -> int:
    matches = [
        atom for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 0 and int(atom.GetAtomMapNum()) == atom_map
    ]
    if len(matches) != 1:
        raise ValueError(f"template is missing connection map {atom_map}")
    return int(matches[0].GetNeighbors()[0].GetIdx())


def _append_real_atoms(output: Any, molecule: Any) -> tuple[dict[int, int], list[int]]:
    atom_map: dict[int, int] = {}
    ordered_real_atoms: list[int] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        copied = Chem.Atom(atom)
        copied.SetAtomMapNum(0)
        new_index = int(output.AddAtom(copied))
        atom_map[int(atom.GetIdx())] = new_index
        ordered_real_atoms.append(new_index)
    for bond in molecule.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin in atom_map and end in atom_map:
            output.AddBond(atom_map[begin], atom_map[end], bond.GetBondType())
    return atom_map, ordered_real_atoms


def _group_smiles_and_connections(molecule: Any) -> tuple[str, int, int]:
    """Return EMC group SMILES and 1-based head/tail * occurrence indices."""

    mapped_smiles = Chem.MolToSmiles(molecule, canonical=False)
    parsed = Chem.MolFromSmiles(mapped_smiles)
    if parsed is None:
        raise ValueError("could not serialize a repeat-unit group")
    dummies = [atom for atom in parsed.GetAtoms() if atom.GetAtomicNum() == 0]
    by_map = {int(atom.GetAtomMapNum()): index + 1 for index, atom in enumerate(dummies)}
    if set(by_map) != {1, 2}:
        raise ValueError("repeat-unit group lost its head or tail connection marker")
    for atom in parsed.GetAtoms():
        atom.SetAtomMapNum(0)
    plain_smiles = Chem.MolToSmiles(parsed, canonical=False)
    check = Chem.MolFromSmiles(plain_smiles)
    if check is None or sum(atom.GetAtomicNum() == 0 for atom in check.GetAtoms()) != 2:
        raise ValueError("repeat-unit group could not preserve two connection markers")
    return plain_smiles, by_map[1], by_map[2]


def _build_group_variant(
    backbone: Any,
    attachments: tuple[tuple[Any, int], ...],
) -> tuple[str, int, int]:
    output = Chem.RWMol(Chem.Mol(backbone))
    backbone_real_atoms = [
        int(atom.GetIdx()) for atom in backbone.GetAtoms() if atom.GetAtomicNum() > 0
    ]
    for branch, backbone_atom_index in attachments:
        local_to_global, _ = _append_real_atoms(output, branch)
        branch_head = local_to_global[_dummy_neighbor(branch, 1)]
        output.AddBond(
            backbone_real_atoms[int(backbone_atom_index)],
            branch_head,
            Chem.BondType.SINGLE,
        )
    molecule = output.GetMol()
    Chem.SanitizeMol(molecule)
    return _group_smiles_and_connections(molecule)


def _run_length_groups(groups: list[str]) -> list[tuple[str, int]]:
    if not groups:
        return []
    result: list[tuple[str, int]] = []
    current = groups[0]
    count = 1
    for group in groups[1:]:
        if group == current:
            count += 1
            continue
        result.append((current, count))
        current = group
        count = 1
    result.append((current, count))
    return result


def render_template_esh(plan: dict[str, Any], builder_spec: dict[str, Any]) -> dict[str, Any]:
    """Render a scalable EMC setup file without creating one giant chain SMILES."""

    if Chem is None:
        raise ValueError("RDKit is required for template-based polymer construction")
    sequence = _sequence_ids(dict(plan["sequence"]))
    templates = {
        item["template_id"]: Chem.MolFromSmiles(str(item["smiles"]))
        for item in plan["repeat_unit_templates"]
    }
    branches = {
        item["template_id"]: Chem.MolFromSmiles(str(item["smiles"]))
        for item in plan.get("branch_templates", [])
    }
    attachments_by_position: dict[int, list[tuple[str, int]]] = {}
    for rule in plan.get("branch_rules", []):
        for position in rule["positions"]:
            attachments_by_position.setdefault(int(position), []).append(
                (str(rule["template_id"]), int(rule["backbone_atom_index"]))
            )

    group_definitions: dict[str, tuple[str, int, int]] = {}
    position_groups: list[str] = []
    variant_names: dict[tuple[Any, ...], str] = {}
    for position, template_id in enumerate(sequence, start=1):
        attachments = tuple(sorted(attachments_by_position.get(position, [])))
        key: tuple[Any, ...] = (template_id, attachments)
        group_name = variant_names.get(key)
        if group_name is None:
            group_name = f"u{len(variant_names) + 1}"
            variant_names[key] = group_name
            group_definitions[group_name] = _build_group_variant(
                templates[template_id],
                tuple((branches[branch_id], atom_index) for branch_id, atom_index in attachments),
            )
        position_groups.append(group_name)

    sequence_mode = str(plan["sequence"]["mode"])
    has_branches = bool(plan.get("branch_rules"))
    polymer_mode = "block"
    polymer_parts: list[str] = []
    if sequence_mode == "homopolymer" and not has_branches:
        polymer_mode = "alternate"
        polymer_parts = [position_groups[0], str(len(position_groups))]
    elif sequence_mode in {"periodic", "alternating"} and not has_branches:
        polymer_mode = "alternate"
        group_counts = Counter(position_groups)
        ordered_groups = list(dict.fromkeys(position_groups))
        for group_name in ordered_groups:
            polymer_parts.extend([group_name, str(group_counts[group_name])])
    elif sequence_mode == "random" and not has_branches:
        polymer_mode = "random"
        counts = Counter(position_groups)
        ordered_groups = sorted(counts)
        selector = ":".join(ordered_groups) + "=" + ":".join(
            str(counts[group_name]) for group_name in ordered_groups
        )
        polymer_parts = [selector, str(len(position_groups))]
    else:
        polymer_mode = "sequence"
        for group_name, count in _run_length_groups(position_groups):
            polymer_parts.extend([group_name, str(count)])
    polymer_parts.extend(["term", "2"])

    all_group_names = sorted(group_definitions, key=lambda value: int(value[1:]))
    group_lines: list[str] = []
    for group_name in all_group_names:
        smiles, head, tail = group_definitions[group_name]
        connections = [f"{head},{target}:{group_definitions[target][2]}" for target in all_group_names]
        connections.extend([f"{head},term:1", f"{tail},term:1"])
        group_lines.append(f"{group_name}\t{smiles}," + ",".join(connections))

    field = str(builder_spec.get("force_field") or "opls/2024/opls-aa")
    if field.lower() in {"opls-aa", "oplsaa", "opls"}:
        field = "opls/2024/opls-aa"
    lines = [
        "#!/usr/bin/env emc.pl",
        "#",
        "# Generated from compact repeat-unit, sequence, and branch definitions.",
        f"# Repeat units/chain: {plan['repeat_units_per_chain']}",
        f"# Actual total carbons/chain: {plan['actual_total_carbons_per_chain']}",
        f"# Template design hash: {plan['template_design_hash']}",
        "",
        "ITEM\tOPTIONS",
        "replace\ttrue",
        f"field\t{field}",
        "number\ttrue",
        f"density\t{float(builder_spec.get('initial_density_g_per_cc') or 0.85)}",
        f"temperature\t{float(builder_spec.get('temperature_k') or 400.0)}",
        f"pressure\t{float(builder_spec.get('pressure_atm') or 1.0)}",
        "build_dir\t.",
        "focus\ttrue",
        "ITEM\tEND",
        "",
        "ITEM\tGROUPS",
        *group_lines,
        "term\t*C",
        "ITEM\tEND",
        "",
        "ITEM\tCLUSTERS",
        f"polymer\t{polymer_mode},{int(builder_spec.get('chain_count') or 1)}",
        "ITEM\tEND",
        "",
        "ITEM\tPOLYMERS",
        "polymer\torder=list",
        "1\t" + ",".join(polymer_parts),
        "ITEM\tEND",
        "",
    ]
    return {
        "esh": "\n".join(lines),
        "materialized_structure": {
            "representation": "emc_polymer_sequence",
            "polymer_mode": polymer_mode,
            "repeat_units_per_chain": int(plan["repeat_units_per_chain"]),
            "actual_total_carbons_per_chain": int(plan["actual_total_carbons_per_chain"]),
            "estimated_real_atoms_per_chain": int(plan["estimated_real_atoms_per_chain"]),
            "group_count": len(group_definitions),
            "polymer_segment_count": len(polymer_parts) // 2,
            "complete_chain_smiles_generated": False,
        },
    }


def expand_template_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Create the atomistic macro-SMILES required by the registered EMC path."""

    if Chem is None:
        raise ValueError("RDKit is required for template-based polymer construction")
    sequence = _sequence_ids(dict(plan["sequence"]))
    templates = {
        item["template_id"]: Chem.MolFromSmiles(str(item["smiles"]))
        for item in plan["repeat_unit_templates"]
    }
    branches = {
        item["template_id"]: Chem.MolFromSmiles(str(item["smiles"]))
        for item in plan.get("branch_templates", [])
    }
    output = Chem.RWMol()
    unit_real_atoms: list[list[int]] = []
    first_head: int | None = None
    previous_tail: int | None = None
    for template_id in sequence:
        molecule = templates[template_id]
        local_to_global, ordered_real = _append_real_atoms(output, molecule)
        head = local_to_global[_dummy_neighbor(molecule, 1)]
        tail = local_to_global[_dummy_neighbor(molecule, 2)]
        if first_head is None:
            first_head = head
        if previous_tail is not None:
            output.AddBond(previous_tail, head, Chem.BondType.SINGLE)
        previous_tail = tail
        unit_real_atoms.append(ordered_real)

    for rule in plan.get("branch_rules", []):
        branch = branches[rule["template_id"]]
        for position in rule["positions"]:
            local_to_global, _ = _append_real_atoms(output, branch)
            branch_head = local_to_global[_dummy_neighbor(branch, 1)]
            backbone_atom = unit_real_atoms[int(position) - 1][int(rule["backbone_atom_index"])]
            output.AddBond(backbone_atom, branch_head, Chem.BondType.SINGLE)

    if first_head is None or previous_tail is None:
        raise ValueError("template plan generated an empty polymer chain")
    left_dummy = Chem.Atom(0)
    right_dummy = Chem.Atom(0)
    left_index = int(output.AddAtom(left_dummy))
    right_index = int(output.AddAtom(right_dummy))
    output.AddBond(left_index, first_head, Chem.BondType.SINGLE)
    output.AddBond(previous_tail, right_index, Chem.BondType.SINGLE)

    molecule = output.GetMol()
    Chem.SanitizeMol(molecule)
    smiles = Chem.MolToSmiles(molecule, canonical=False)
    real_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 0]
    return {
        "monomer_smiles": smiles,
        "actual_total_carbons_per_chain": sum(atom.GetSymbol() == "C" for atom in real_atoms),
        "actual_real_atoms_per_chain": len(real_atoms),
        "repeat_units_per_chain": 1,
    }


__all__ = ["expand_template_plan", "normalize_template_plan", "render_template_esh"]
