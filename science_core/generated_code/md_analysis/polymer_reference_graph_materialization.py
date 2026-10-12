#!/usr/bin/env python3
"""Materialize a complete heavy-atom graph from a validated polymer builder spec."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter, deque
from pathlib import Path
from typing import Any

from rdkit import Chem

from polymer_template_builder import normalize_template_plan


def _mapped_repeat_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles or "").strip())
    if molecule is None:
        raise ValueError("polymer repeat-unit SMILES is not RDKit-parseable")
    dummies = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise ValueError("polymer repeat unit must contain exactly two connection markers")
    for atom_map, atom in enumerate(sorted(dummies, key=lambda item: item.GetIdx()), start=1):
        atom.SetAtomMapNum(atom_map)
    return Chem.MolToSmiles(molecule, canonical=False)


def _sequence_ids(sequence: dict[str, Any]) -> list[str]:
    mode = str(sequence["mode"])
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

    fractions = dict(sequence["composition"])
    raw_counts = {key: total * float(value) for key, value in fractions.items()}
    counts = {key: int(value) for key, value in raw_counts.items()}
    remainder = total - sum(counts.values())
    order = sorted(fractions, key=lambda key: (-(raw_counts[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    result = [key for key, count in counts.items() for _ in range(count)]
    random.Random(int(sequence["seed"])).shuffle(result)
    return result


def _repeat_plan(builder_spec: dict[str, Any]) -> dict[str, Any]:
    repeat_smiles = str(
        builder_spec.get("repeat_unit_smiles")
        or builder_spec.get("monomer_smiles")
        or ""
    ).strip()
    return normalize_template_plan(
        {
            "repeat_unit_templates": [
                {
                    "template_id": "A",
                    "smiles": _mapped_repeat_smiles(repeat_smiles),
                }
            ],
            "sequence": {
                "mode": "homopolymer",
                "template_id": "A",
                "repeat_units": int(builder_spec.get("repeat_units_per_chain") or 1),
            },
        }
    )


def _molecule_components(molecule: Chem.Mol) -> tuple[list[dict[str, Any]], list[tuple[int, int, str]]]:
    atoms = [
        {
            "element": atom.GetSymbol(),
            "formal_charge": int(atom.GetFormalCharge()),
            "aromatic": bool(atom.GetIsAromatic()),
        }
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 0
    ]
    real_index = {
        atom.GetIdx(): index
        for index, atom in enumerate(
            atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 0
        )
    }
    bonds = []
    for bond in molecule.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin in real_index and end in real_index:
            bonds.append(
                (
                    real_index[begin],
                    real_index[end],
                    str(bond.GetBondType()),
                )
            )
    return atoms, bonds


def _connection_neighbor(molecule: Chem.Mol, atom_map: int) -> int:
    matches = [
        atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 0 and int(atom.GetAtomMapNum()) == atom_map
    ]
    if len(matches) != 1 or matches[0].GetDegree() != 1:
        raise ValueError(f"template connection marker {atom_map} is invalid")
    return int(matches[0].GetNeighbors()[0].GetIdx())


def _template_components(item: dict[str, Any], *, branch: bool) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(str(item["smiles"]))
    if molecule is None:
        raise ValueError(f"template {item['template_id']} is not RDKit-parseable")
    atoms, bonds = _molecule_components(molecule)
    real_index = {
        atom.GetIdx(): index
        for index, atom in enumerate(
            atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 0
        )
    }
    head = real_index[_connection_neighbor(molecule, 1)]
    tail = None if branch else real_index[_connection_neighbor(molecule, 2)]
    return {
        "atoms": atoms,
        "bonds": bonds,
        "head": head,
        "tail": tail,
    }


def _append_component(
    graph_atoms: list[dict[str, Any]],
    graph_bonds: list[tuple[int, int, str]],
    component: dict[str, Any],
) -> tuple[int, list[int]]:
    offset = len(graph_atoms)
    graph_atoms.extend(component["atoms"])
    graph_bonds.extend(
        (offset + begin, offset + end, bond_type)
        for begin, end, bond_type in component["bonds"]
    )
    return offset, list(range(offset, offset + len(component["atoms"])))


def _build_from_plan(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[int, int, str]]]:
    templates = {
        str(item["template_id"]): _template_components(item, branch=False)
        for item in plan["repeat_unit_templates"]
    }
    branches = {
        str(item["template_id"]): _template_components(item, branch=True)
        for item in plan.get("branch_templates", [])
    }
    atoms: list[dict[str, Any]] = []
    bonds: list[tuple[int, int, str]] = []
    unit_atoms: list[list[int]] = []
    previous_tail: int | None = None
    for template_id in _sequence_ids(dict(plan["sequence"])):
        component = templates[template_id]
        offset, indices = _append_component(atoms, bonds, component)
        head = offset + int(component["head"])
        tail = offset + int(component["tail"])
        if previous_tail is not None:
            bonds.append((previous_tail, head, "SINGLE"))
        previous_tail = tail
        unit_atoms.append(indices)

    for rule in plan.get("branch_rules", []):
        component = branches[str(rule["template_id"])]
        for position in rule["positions"]:
            offset, _indices = _append_component(atoms, bonds, component)
            branch_head = offset + int(component["head"])
            backbone_atom = unit_atoms[int(position) - 1][
                int(rule["backbone_atom_index"])
            ]
            bonds.append((backbone_atom, branch_head, "SINGLE"))
    return atoms, bonds


def _build_graph(builder_spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[int, int, str]]]:
    architecture = str(builder_spec.get("architecture") or "")
    if architecture == "explicit_polymer_chain":
        molecule = Chem.MolFromSmiles(str(builder_spec.get("polymer_smiles") or ""))
        if molecule is None:
            raise ValueError("explicit polymer chain is not RDKit-parseable")
        return _molecule_components(molecule)
    if architecture == "templated_polymer_chain":
        plan = builder_spec.get("template_plan")
        if not isinstance(plan, dict):
            raise ValueError("templated polymer builder spec is missing template_plan")
        return _build_from_plan(plan)
    if architecture in {"linear_homopolymer", "linear_alpha_olefin", "pe_lcb"}:
        return _build_from_plan(_repeat_plan(builder_spec))
    raise ValueError(f"complete graph materialization is not registered for {architecture}")


def _connected_component_count(atom_count: int, bonds: list[tuple[int, int, str]]) -> int:
    if atom_count == 0:
        return 0
    adjacency = [[] for _ in range(atom_count)]
    for begin, end, _bond_type in bonds:
        adjacency[begin].append(end)
        adjacency[end].append(begin)
    unseen = set(range(atom_count))
    components = 0
    while unseen:
        components += 1
        start = unseen.pop()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return components


def _write_graph(
    path: Path,
    atoms: list[dict[str, Any]],
    bonds: list[tuple[int, int, str]],
) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        header = json.dumps(
            {
                "kind": "header",
                "schema_version": "polymer_complete_graph.v1",
                "atom_count": len(atoms),
                "bond_count": len(bonds),
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        handle.write(header + "\n")
        digest.update((header + "\n").encode("ascii"))
        for atom_id, atom in enumerate(atoms):
            line = json.dumps(
                {"kind": "atom", "atom_id": atom_id, **atom},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            handle.write(line + "\n")
            digest.update((line + "\n").encode("ascii"))
        for begin, end, bond_type in bonds:
            line = json.dumps(
                {
                    "kind": "bond",
                    "begin": begin,
                    "end": end,
                    "bond_type": bond_type,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
            handle.write(line + "\n")
            digest.update((line + "\n").encode("ascii"))
    return digest.hexdigest()


def materialize_complete_polymer_graph(
    case_dir: str | Path,
    builder_spec: dict[str, Any],
) -> dict[str, Any]:
    """Build, verify and save one complete heavy-atom molecular graph."""

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    atoms, bonds = _build_graph(builder_spec)
    component_count = _connected_component_count(len(atoms), bonds)
    if component_count != 1:
        raise ValueError("expanded polymer chain must contain one connected component")

    carbon_count = sum(atom["element"] == "C" for atom in atoms)
    expected_carbon_count = int(
        builder_spec.get("actual_total_carbons_per_chain") or carbon_count
    )
    if carbon_count != expected_carbon_count:
        raise ValueError(
            "expanded polymer carbon count does not match the validated design: "
            f"{carbon_count} != {expected_carbon_count}"
        )
    expected_atom_count = builder_spec.get("estimated_real_atoms_per_chain")
    if expected_atom_count is not None and len(atoms) != int(expected_atom_count):
        raise ValueError(
            "expanded polymer atom count does not match the validated template: "
            f"{len(atoms)} != {int(expected_atom_count)}"
        )

    graph_path = case_dir / "complete_heavy_atom_graph.jsonl.gz"
    summary_path = case_dir / "complete_heavy_atom_graph.json"
    graph_sha256 = _write_graph(graph_path, atoms, bonds)
    summary = {
        "schema_version": "polymer_complete_graph.v1",
        "case_id": str(builder_spec.get("case_id") or case_dir.name),
        "architecture": str(builder_spec.get("architecture") or ""),
        "atom_count": len(atoms),
        "carbon_count": carbon_count,
        "bond_count": len(bonds),
        "connected_component_count": component_count,
        "element_counts": dict(sorted(Counter(atom["element"] for atom in atoms).items())),
        "graph_sha256": graph_sha256,
        "graph_path": str(graph_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        **summary,
        "summary_path": str(summary_path),
    }


__all__ = ["materialize_complete_polymer_graph"]
