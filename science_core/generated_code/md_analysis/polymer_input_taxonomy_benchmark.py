#!/usr/bin/env python3
"""Benchmark polymer-input validation across major polymer classes.

The benchmark is pre-registered in code: every case carries an expected
decision before the production validator is called. It measures whether the
preparation layer correctly:

1. accepts inputs that can be materialized by a registered builder;
2. rejects internally invalid molecular descriptions; and
3. routes scientifically valid but unsupported architectures to REVIEW.

The benchmark does not claim to enumerate every polymer molecule. It covers
the major architecture classes and all input representations currently
accepted by the framework.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from polymer_builder_dispatch import materialize_registered_polymer_input
from polymer_design_validator import validate_polymer_design_candidate
from polymer_reference_graph_materialization import (
    materialize_complete_polymer_graph,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "polymer_input_taxonomy_benchmark"
SCHEMA_VERSION = "polymer_input_taxonomy_benchmark.v1"
VARIANT_COUNT = 5

TAXONOMY_SOURCES = (
    {
        "name": "IUPAC Brief Guide to Polymerization Terminology",
        "url": (
            "https://iupac.org/wp-content/uploads/2022/09/"
            "Brief-Guide-to-Polymerization-Terminology-web20220815.pdf"
        ),
        "role": "major polymer architecture and copolymer classes",
    },
    {
        "name": (
            "IUPAC structure-based nomenclature for irregular linear, star, "
            "comb and brush polymers"
        ),
        "url": (
            "https://iupac.org/recommendation/"
            "structure-based-nomenclature-for-irregular-linear-star-comb-and-brush-polymers/"
        ),
        "role": "linear, star, comb and brush architecture coverage",
    },
)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _template_candidate(
    case_id: str,
    *,
    family: str,
    declared_architecture: str,
    templates: list[dict[str, str]],
    sequence: dict[str, Any],
    branches: list[dict[str, str]] | None = None,
    branch_rules: list[dict[str, Any]] | None = None,
    chain_count: int = 2,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "candidate_id": case_id,
        "polymer_family": family,
        "architecture": declared_architecture,
        "repeat_unit_templates": deepcopy(templates),
        "sequence": deepcopy(sequence),
        "chain_count": chain_count,
    }
    if branches is not None:
        candidate["branch_templates"] = deepcopy(branches)
    if branch_rules is not None:
        candidate["branch_rules"] = deepcopy(branch_rules)
    return candidate


def _repeat_candidate(
    case_id: str,
    *,
    family: str,
    repeat_smiles: str,
    target_carbons: int,
) -> dict[str, Any]:
    return {
        "candidate_id": case_id,
        "polymer_family": family,
        "repeat_unit_smiles": repeat_smiles,
        "target_total_carbons_per_chain": target_carbons,
        "chain_count": 2,
    }


def _explicit_candidate(
    case_id: str,
    *,
    family: str,
    polymer_smiles: str,
    declared_architecture: str,
) -> dict[str, Any]:
    return {
        "candidate_id": case_id,
        "polymer_family": family,
        "architecture": declared_architecture,
        "polymer_smiles": polymer_smiles,
        "chain_count": 2,
    }


def _special_candidate(
    case_id: str,
    *,
    family: str,
    architecture: str,
    repeat_smiles: str,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": case_id,
        "polymer_family": family,
        "architecture": architecture,
        "repeat_unit_smiles": repeat_smiles,
        **deepcopy(dict(fields)),
    }


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target: Any = payload
    for part in parts[:-1]:
        if part.isdigit():
            target = target[int(part)]
        else:
            target = target[part]
    leaf = parts[-1]
    if leaf.isdigit():
        target[int(leaf)] = value
    else:
        target[leaf] = value


def _case_pair(
    category_id: str,
    category_label: str,
    coverage_group: str,
    variant: int,
    reference: dict[str, Any],
    *,
    reference_status: str,
    challenge_path: str,
    challenge_value: Any,
    challenge_reason: str,
    reference_reason: str = "",
) -> list[dict[str, Any]]:
    pair_id = f"{category_id.upper()}_{variant:02d}"
    reference = deepcopy(reference)
    reference["candidate_id"] = f"{category_id}_reference_{variant:02d}"
    challenge = deepcopy(reference)
    challenge["candidate_id"] = f"{category_id}_challenge_{variant:02d}"
    _set_path(challenge, challenge_path, challenge_value)
    return [
        {
            "case_id": reference["candidate_id"],
            "pair_id": pair_id,
            "case_role": "reference",
            "category_id": category_id,
            "category_label": category_label,
            "coverage_group": coverage_group,
            "expected_status": reference_status,
            "expected_reason_substring": reference_reason,
            "mutation": "none",
            "candidate": reference,
        },
        {
            "case_id": challenge["candidate_id"],
            "pair_id": pair_id,
            "case_role": "challenge",
            "category_id": category_id,
            "category_label": category_label,
            "coverage_group": coverage_group,
            "expected_status": "reject",
            "expected_reason_substring": challenge_reason,
            "mutation": f"{challenge_path}={challenge_value!r}",
            "candidate": challenge,
        },
    ]


def benchmark_cases() -> list[dict[str, Any]]:
    """Create the pre-registered 24-category, 240-input benchmark."""

    cases: list[dict[str, Any]] = []
    for variant in range(1, VARIANT_COUNT + 1):
        dp = 20 + 10 * variant
        target = 80 + 4 * variant
        chain_count = 1 + variant

        cases += _case_pair(
            "linear_hydrocarbon",
            "Linear hydrocarbon homopolymer",
            "composition_and_chain",
            variant,
            _repeat_candidate(
                "placeholder",
                family="polyethylene",
                repeat_smiles="*CC*",
                target_carbons=target,
            ),
            reference_status="pass",
            challenge_path="repeat_unit_smiles",
            challenge_value="not a smiles",
            challenge_reason="not RDKit-parseable",
        )

        cases += _case_pair(
            "linear_heteroatom",
            "Linear heteroatom homopolymer",
            "composition_and_chain",
            variant,
            _repeat_candidate(
                "placeholder",
                family="polyether",
                repeat_smiles="*CCO*",
                target_carbons=target,
            ),
            reference_status="pass",
            challenge_path="repeat_unit_smiles",
            challenge_value="*CCO(",
            challenge_reason="not RDKit-parseable",
        )

        cases += _case_pair(
            "explicit_linear_chain",
            "Explicit linear polymer chain",
            "composition_and_chain",
            variant,
            _explicit_candidate(
                "placeholder",
                family="polyether",
                polymer_smiles="CCOCCOCCOCC",
                declared_architecture="linear_polymer",
            ),
            reference_status="pass",
            challenge_path="polymer_smiles",
            challenge_value="CCO.CCO",
            challenge_reason="one connected polymer chain",
        )

        cases += _case_pair(
            "explicit_branched_chain",
            "Explicit branched polymer chain",
            "branch_architecture",
            variant,
            _explicit_candidate(
                "placeholder",
                family="branched_polyether",
                polymer_smiles="CC(C)CCOCC(C)C",
                declared_architecture="branched_polymer",
            ),
            reference_status="pass",
            challenge_path="polymer_smiles",
            challenge_value="*CC(C)CCO*",
            challenge_reason="open connection markers",
        )

        templates_ab = [
            {"template_id": "A", "smiles": "[*:1]CC[*:2]"},
            {"template_id": "B", "smiles": "[*:1]CO[*:2]"},
        ]

        cases += _case_pair(
            "alternating_copolymer",
            "Alternating copolymer",
            "copolymer_sequence",
            variant,
            _template_candidate(
                "placeholder",
                family="copolymer",
                declared_architecture="alternating_copolymer",
                templates=templates_ab,
                sequence={
                    "mode": "alternating",
                    "pattern": ["A", "B"],
                    "repeat_units": dp,
                },
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="sequence.pattern",
            challenge_value=["A"],
            challenge_reason="requires at least two templates",
        )

        cases += _case_pair(
            "periodic_copolymer",
            "Sequence-defined periodic copolymer",
            "copolymer_sequence",
            variant,
            _template_candidate(
                "placeholder",
                family="copolymer",
                declared_architecture="periodic_copolymer",
                templates=templates_ab,
                sequence={
                    "mode": "periodic",
                    "pattern": ["A", "A", "B"],
                    "repeat_units": dp,
                },
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="sequence.pattern",
            challenge_value=[],
            challenge_reason="must be a nonempty list",
        )

        cases += _case_pair(
            "block_copolymer",
            "Block copolymer",
            "copolymer_sequence",
            variant,
            _template_candidate(
                "placeholder",
                family="copolymer",
                declared_architecture="block_copolymer",
                templates=templates_ab,
                sequence={
                    "mode": "block",
                    "blocks": [
                        {"template_id": "A", "count": dp},
                        {"template_id": "B", "count": dp // 2},
                    ],
                },
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="sequence.blocks.1.count",
            challenge_value=0,
            challenge_reason="count must be >= 1",
        )

        cases += _case_pair(
            "random_copolymer",
            "Random or statistical copolymer",
            "copolymer_sequence",
            variant,
            _template_candidate(
                "placeholder",
                family="copolymer",
                declared_architecture="random_copolymer",
                templates=templates_ab,
                sequence={
                    "mode": "random",
                    "composition": {"A": 0.7, "B": 0.3},
                    "repeat_units": dp,
                    "seed": 100 + variant,
                },
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="sequence.composition.B",
            challenge_value=-0.3,
            challenge_reason="must be > 0",
        )

        branch_templates = [{"template_id": "S", "smiles": "[*:1]CCCC"}]

        cases += _case_pair(
            "graft_copolymer",
            "Graft copolymer",
            "branch_architecture",
            variant,
            _template_candidate(
                "placeholder",
                family="graft_copolymer",
                declared_architecture="graft_copolymer",
                templates=templates_ab,
                sequence={
                    "mode": "periodic",
                    "pattern": ["A", "B"],
                    "repeat_units": dp,
                },
                branches=branch_templates,
                branch_rules=[
                    {
                        "template_id": "S",
                        "backbone_atom_index": 0,
                        "positions": [5, 10 + variant],
                    }
                ],
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="branch_rules.0.positions",
            challenge_value=[dp + 1],
            challenge_reason="positions must be within",
        )

        cases += _case_pair(
            "comb_brush_polymer",
            "Comb or brush polymer",
            "branch_architecture",
            variant,
            _template_candidate(
                "placeholder",
                family="comb_brush_polymer",
                declared_architecture="comb_polymer",
                templates=[templates_ab[0]],
                sequence={
                    "mode": "homopolymer",
                    "template_id": "A",
                    "repeat_units": dp,
                },
                branches=branch_templates,
                branch_rules=[
                    {
                        "template_id": "S",
                        "backbone_atom_index": 0,
                        "placement": {
                            "mode": "uniform",
                            "count": min(5 + variant, dp),
                        },
                    }
                ],
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="branch_rules.0.placement.count",
            challenge_value=dp + 1,
            challenge_reason="exceeds available backbone positions",
        )

        cases += _case_pair(
            "randomly_branched_polymer",
            "Randomly branched polymer",
            "branch_architecture",
            variant,
            _template_candidate(
                "placeholder",
                family="branched_polymer",
                declared_architecture="randomly_branched_polymer",
                templates=[templates_ab[0]],
                sequence={
                    "mode": "homopolymer",
                    "template_id": "A",
                    "repeat_units": dp,
                },
                branches=branch_templates,
                branch_rules=[
                    {
                        "template_id": "S",
                        "backbone_atom_index": 0,
                        "placement": {
                            "mode": "random",
                            "count": 3 + variant,
                            "seed": 200 + variant,
                        },
                    }
                ],
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="branch_rules.0.placement.mode",
            challenge_value="adaptive",
            challenge_reason="must be uniform or random",
        )

        pe_lcb = {
            "candidate_id": "placeholder",
            "polymer_family": "PE",
            "architecture": "pe_lcb",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "chain_count": 12 + variant,
            "target_total_carbons_per_chain": 100,
            "branch_points": min(5, variant),
            "branch_carbons": [3, 4, 5, 8, 12][variant - 1],
            "branch_position_pattern": {
                1: "mid",
                2: "two-ends",
                3: "three-point",
                4: "comb-uniform-4",
                5: "comb-uniform-5",
            }[variant],
        }
        cases += _case_pair(
            "pe_lcb",
            "Long-chain branched polyethylene",
            "branch_architecture",
            variant,
            pe_lcb,
            reference_status="pass",
            challenge_path="branch_points",
            challenge_value=0,
            challenge_reason="branch_points must be >= 1",
        )

        alpha_olefin = {
            "candidate_id": "placeholder",
            "polymer_family": "poly_alpha_olefin",
            "architecture": "linear_alpha_olefin",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "chain_count": 12 + variant,
            "target_total_carbons_per_chain": target,
            "sidechain_carbons": 2 + variant,
        }
        cases += _case_pair(
            "poly_alpha_olefin",
            "Linear poly-alpha-olefin",
            "composition_and_chain",
            variant,
            alpha_olefin,
            reference_status="pass",
            challenge_path="sidechain_carbons",
            challenge_value=0,
            challenge_reason="sidechain_carbons must be >= 1",
        )

        cases += _case_pair(
            "high_degree_polymer",
            "High-degree linear polymer",
            "scale_and_representation",
            variant,
            _template_candidate(
                "placeholder",
                family="polyethylene",
                declared_architecture="linear_homopolymer",
                templates=[templates_ab[0]],
                sequence={
                    "mode": "homopolymer",
                    "template_id": "A",
                    "repeat_units": 1000 * variant,
                },
                chain_count=1,
            ),
            reference_status="pass",
            challenge_path="sequence.repeat_units",
            challenge_value=0,
            challenge_reason="repeat_units must be >= 1",
        )

        cases += _case_pair(
            "halogenated_polymer",
            "Halogenated polymer",
            "chemical_scope",
            variant,
            _template_candidate(
                "placeholder",
                family="halogenated_polymer",
                declared_architecture="linear_homopolymer",
                templates=[
                    {"template_id": "V", "smiles": "[*:1]C(F)C(F)[*:2]"}
                ],
                sequence={
                    "mode": "homopolymer",
                    "template_id": "V",
                    "repeat_units": dp,
                },
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="repeat_unit_templates.0.smiles",
            challenge_value="[*:1]C(F)C(F)",
            challenge_reason="dummy connection maps",
        )

        cases += _case_pair(
            "conjugated_aromatic_polymer",
            "Conjugated or aromatic polymer",
            "chemical_scope",
            variant,
            _template_candidate(
                "placeholder",
                family="conjugated_polymer",
                declared_architecture="linear_homopolymer",
                templates=[
                    {
                        "template_id": "P",
                        "smiles": "[*:1]c1ccc(cc1)[*:2]",
                    }
                ],
                sequence={
                    "mode": "homopolymer",
                    "template_id": "P",
                    "repeat_units": dp,
                },
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="repeat_unit_templates.0.smiles",
            challenge_value="[*:1]c1ccccc1",
            challenge_reason="dummy connection maps",
        )

        cases += _case_pair(
            "polyamide_like_polymer",
            "Polyamide-like polymer",
            "chemical_scope",
            variant,
            _template_candidate(
                "placeholder",
                family="polyamide",
                declared_architecture="linear_homopolymer",
                templates=[
                    {
                        "template_id": "N",
                        "smiles": "[*:1]C(=O)NCC[*:2]",
                    }
                ],
                sequence={
                    "mode": "homopolymer",
                    "template_id": "N",
                    "repeat_units": dp,
                },
                chain_count=chain_count,
            ),
            reference_status="pass",
            challenge_path="repeat_unit_templates.0.smiles",
            challenge_value="[*:1]C(=O)NCC(",
            challenge_reason="not RDKit-parseable",
        )

        cases += _case_pair(
            "charged_polymer",
            "Charged polymer or polyelectrolyte",
            "chemical_scope",
            variant,
            _explicit_candidate(
                "placeholder",
                family="polyelectrolyte",
                polymer_smiles="CC[N+](C)(C)CC",
                declared_architecture="linear_polyelectrolyte",
            ),
            reference_status="pass",
            challenge_path="polymer_smiles",
            challenge_value="CC[N+](C)(C)CC.CC",
            challenge_reason="one connected polymer chain",
        )

        special_specs = (
            (
                "cyclic_polymer",
                "Cyclic polymer",
                "cyclic_polymer",
                {"repeat_units_per_cycle": 20 + variant},
                "repeat_units_per_cycle",
                2,
                "must be >= 3",
            ),
            (
                "star_polymer",
                "Star polymer",
                "star_polymer",
                {
                    "core_smiles": "C",
                    "arm_count": 3 + variant,
                    "arm_repeat_units": 10 + variant,
                },
                "arm_count",
                2,
                "must be >= 3",
            ),
            (
                "network_polymer",
                "Network polymer",
                "network_polymer",
                {
                    "crosslinker_smiles": "NCCN",
                    "crosslink_fraction": 0.05 * variant,
                },
                "crosslink_fraction",
                1.5,
                "must be within 0.0..1.0",
            ),
            (
                "hyperbranched_polymer",
                "Hyperbranched polymer",
                "hyperbranched_polymer",
                {
                    "repeat_units_per_chain": 20 + variant,
                    "degree_of_branching": 0.1 * variant,
                },
                "degree_of_branching",
                1.2,
                "must be within 0.0..1.0",
            ),
            (
                "dendrimer",
                "Dendrimer",
                "dendrimer",
                {
                    "core_smiles": "N",
                    "generation": variant,
                    "core_functionality": 3 + variant,
                },
                "generation",
                0,
                "must be >= 1",
            ),
        )
        for (
            category_id,
            category_label,
            architecture,
            fields,
            challenge_path,
            challenge_value,
            challenge_reason,
        ) in special_specs:
            cases += _case_pair(
                category_id,
                category_label,
                "special_architecture",
                variant,
                _special_candidate(
                    "placeholder",
                    family=category_id,
                    architecture=architecture,
                    repeat_smiles="*CC*",
                    fields=fields,
                ),
                reference_status="review",
                reference_reason="requires a registered builder",
                challenge_path=challenge_path,
                challenge_value=challenge_value,
                challenge_reason=challenge_reason,
            )

        cases += _case_pair(
            "metal_inorganic_polymer",
            "Metal-containing or inorganic polymer",
            "chemical_scope",
            variant,
            _repeat_candidate(
                "placeholder",
                family="metal_containing_polymer",
                repeat_smiles="*C[Fe]C*",
                target_carbons=target,
            ),
            reference_status="review",
            reference_reason="registered_force_fields_do_not_cover_detected_elements",
            challenge_path="repeat_unit_smiles",
            challenge_value="*C[Fe]C(",
            challenge_reason="not RDKit-parseable",
        )

    return cases


def _validation_text(validation: Mapping[str, Any]) -> str:
    parts = [
        *(str(item) for item in validation.get("errors", [])),
        *(str(item) for item in validation.get("warnings", [])),
    ]
    setup = validation.get("automatic_setup")
    if isinstance(setup, Mapping):
        for section_name in (
            "architecture_selection",
            "builder_selection",
            "force_field_selection",
        ):
            section = setup.get(section_name)
            if isinstance(section, Mapping):
                parts.extend(str(value) for value in section.values())
    return " | ".join(parts)


def _pre_registration_payload(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "variant_count_per_category": VARIANT_COUNT,
        "category_count": len({case["category_id"] for case in cases}),
        "case_count": len(cases),
        "taxonomy_sources": TAXONOMY_SOURCES,
        "decision_definitions": {
            "pass": "valid input supported by a registered builder",
            "review": (
                "valid polymer class or chemistry requiring a specialized "
                "builder, force field or parameterization check"
            ),
            "reject": "internally invalid or inconsistent polymer description",
        },
        "cases": cases,
    }


def run_benchmark(output_root: Path) -> dict[str, Any]:
    cases = benchmark_cases()
    preregistration_path = output_root / "polymer_input_taxonomy_preregistration.json"
    _write_json(preregistration_path, _pre_registration_payload(cases))
    preregistration_hash = _sha256(preregistration_path)

    rows: list[dict[str, Any]] = []
    materialized_root = output_root / "materialized_inputs"
    for case in cases:
        validation = validate_polymer_design_candidate(case["candidate"])
        actual_status = str(validation.get("validator_status") or "")
        expected_status = str(case["expected_status"])
        reason_text = _validation_text(validation)
        expected_reason = str(case.get("expected_reason_substring") or "")
        reason_match = (
            not expected_reason
            or expected_reason.lower() in reason_text.lower()
        )

        materialized = False
        materialization_error = ""
        materialized_esh = ""
        materialized_case_json = ""
        graph_materialized = False
        graph_materialization_error = ""
        graph_file = ""
        graph_summary = ""
        graph_atom_count = ""
        graph_carbon_count = ""
        graph_bond_count = ""
        if expected_status == "pass" and actual_status == "pass":
            try:
                case_dir = materialized_root / case["case_id"]
                artifacts = materialize_registered_polymer_input(
                    case_dir,
                    validation["builder_spec"],
                    validation,
                    {
                        "source": "polymer_input_taxonomy_benchmark",
                        "preregistration_sha256": preregistration_hash,
                    },
                )
                esh_path = Path(artifacts["esh"])
                case_json_path = Path(artifacts["case_json"])
                materialized = (
                    esh_path.is_file()
                    and esh_path.stat().st_size > 0
                    and case_json_path.is_file()
                    and case_json_path.stat().st_size > 0
                )
                materialized_esh = str(esh_path)
                materialized_case_json = str(case_json_path)
                graph = materialize_complete_polymer_graph(
                    case_dir,
                    validation["builder_spec"],
                )
                graph_file_path = Path(str(graph["graph_path"]))
                graph_summary_path = Path(str(graph["summary_path"]))
                graph_materialized = (
                    graph_file_path.is_file()
                    and graph_file_path.stat().st_size > 0
                    and graph_summary_path.is_file()
                    and graph_summary_path.stat().st_size > 0
                    and int(graph["connected_component_count"]) == 1
                )
                graph_file = str(graph_file_path)
                graph_summary = str(graph_summary_path)
                graph_atom_count = int(graph["atom_count"])
                graph_carbon_count = int(graph["carbon_count"])
                graph_bond_count = int(graph["bond_count"])
            except Exception as exc:  # pragma: no cover - recorded as experiment output
                materialization_error = f"{type(exc).__name__}: {exc}"
                graph_materialization_error = materialization_error

        status_match = actual_status == expected_status
        operational_match = (
            status_match
            and reason_match
            and (
                materialized and graph_materialized
                if expected_status == "pass"
                else not bool(validation.get("builder_ready"))
            )
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "case_role": case["case_role"],
                "category_id": case["category_id"],
                "category_label": case["category_label"],
                "coverage_group": case["coverage_group"],
                "expected_status": expected_status,
                "actual_status": actual_status,
                "status_match": status_match,
                "expected_reason_substring": expected_reason,
                "reason_match": reason_match,
                "operational_match": operational_match,
                "builder_ready": bool(validation.get("builder_ready")),
                "production_ready": bool(validation.get("production_ready")),
                "next_action": validation.get("next_action"),
                "decision_without_input_validation": "forwarded",
                "materialized_input": materialized,
                "materialized_esh": materialized_esh,
                "materialized_case_json": materialized_case_json,
                "materialization_error": materialization_error,
                "complete_graph_materialized": graph_materialized,
                "complete_graph_file": graph_file,
                "complete_graph_summary": graph_summary,
                "complete_graph_atom_count": graph_atom_count,
                "complete_graph_carbon_count": graph_carbon_count,
                "complete_graph_bond_count": graph_bond_count,
                "complete_graph_error": graph_materialization_error,
                "mutation": case["mutation"],
                "errors": " | ".join(
                    str(item) for item in validation.get("errors", [])
                ),
                "warnings": " | ".join(
                    str(item) for item in validation.get("warnings", [])
                ),
            }
        )

    result_path = output_root / "polymer_input_taxonomy_results.csv"
    _write_csv(result_path, rows)

    category_rows: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category_id"])].append(row)
    for category_id, category_cases in sorted(by_category.items()):
        successes = sum(bool(row["operational_match"]) for row in category_cases)
        low, high = _wilson_interval(successes, len(category_cases))
        category_rows.append(
            {
                "category_id": category_id,
                "category_label": category_cases[0]["category_label"],
                "coverage_group": category_cases[0]["coverage_group"],
                "case_count": len(category_cases),
                "correct_count": successes,
                "accuracy": successes / len(category_cases),
                "wilson_95ci_low": low,
                "wilson_95ci_high": high,
                "reference_expected_status": next(
                    row["expected_status"]
                    for row in category_cases
                    if row["case_role"] == "reference"
                ),
            }
        )
    category_path = output_root / "polymer_input_taxonomy_category_summary.csv"
    _write_csv(category_path, category_rows)

    total_successes = sum(bool(row["operational_match"]) for row in rows)
    total = len(rows)
    low, high = _wilson_interval(total_successes, total)
    references = [row for row in rows if row["case_role"] == "reference"]
    challenges = [row for row in rows if row["case_role"] == "challenge"]
    pass_references = [
        row for row in references if row["expected_status"] == "pass"
    ]
    review_references = [
        row for row in references if row["expected_status"] == "review"
    ]
    category_accuracies = [float(row["accuracy"]) for row in category_rows]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration_hash,
        "taxonomy_sources": TAXONOMY_SOURCES,
        "category_count": len(category_rows),
        "coverage_group_count": len({row["coverage_group"] for row in rows}),
        "variant_count_per_category": VARIANT_COUNT,
        "case_count": total,
        "matched_pair_count": total // 2,
        "expected_status_counts": dict(
            sorted(Counter(str(row["expected_status"]) for row in rows).items())
        ),
        "actual_status_counts": dict(
            sorted(Counter(str(row["actual_status"]) for row in rows).items())
        ),
        "correct_count": total_successes,
        "micro_accuracy": total_successes / total,
        "micro_accuracy_wilson_95ci": [low, high],
        "macro_category_accuracy": sum(category_accuracies) / len(category_accuracies),
        "supported_reference_retention": (
            sum(bool(row["operational_match"]) for row in pass_references)
            / len(pass_references)
        ),
        "unsupported_reference_review_routing": (
            sum(bool(row["operational_match"]) for row in review_references)
            / len(review_references)
        ),
        "invalid_input_interception": (
            sum(bool(row["operational_match"]) for row in challenges)
            / len(challenges)
        ),
        "invalid_input_forwarded_without_validation": sum(
            row["decision_without_input_validation"] == "forwarded"
            for row in challenges
        ),
        "supported_reference_materialization": (
            sum(bool(row["materialized_input"]) for row in pass_references)
            / len(pass_references)
        ),
        "supported_reference_complete_graph_materialization": (
            sum(bool(row["complete_graph_materialized"]) for row in pass_references)
            / len(pass_references)
        ),
        "all_categories_fully_correct": all(
            float(row["accuracy"]) == 1.0 for row in category_rows
        ),
        "failed_case_ids": [
            str(row["case_id"]) for row in rows if not bool(row["operational_match"])
        ],
        "artifacts": {
            "results_csv": str(result_path),
            "category_summary_csv": str(category_path),
            "materialized_inputs": str(materialized_root),
        },
        "claim_boundary": (
            "Accuracy applies to the pre-registered major-class benchmark and "
            "all implemented input representations. It is not a claim that "
            "every possible polymer molecule has been simulated."
        ),
    }
    summary_path = output_root / "polymer_input_taxonomy_summary.json"
    _write_json(summary_path, summary)

    manifest_rows = [
        {
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "case_role": case["case_role"],
            "category_id": case["category_id"],
            "coverage_group": case["coverage_group"],
            "expected_status": case["expected_status"],
            "expected_reason_substring": case["expected_reason_substring"],
            "candidate": case["candidate"],
        }
        for case in cases
    ]
    _write_jsonl(
        output_root / "polymer_input_taxonomy_cases.jsonl",
        manifest_rows,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    summary = run_benchmark(args.output_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["failed_case_ids"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
