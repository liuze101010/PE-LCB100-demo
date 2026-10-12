from __future__ import annotations

import re
import json
from functools import lru_cache
from hashlib import sha256
from typing import Any, Iterable


BRANCH_CARBONS = (3, 4, 5, 8, 12, 16, 20)
PATTERN_TO_TOPOLOGY: dict[int, dict[str, str]] = {
    1: {
        "mid": "three-arm-star",
        "near-end": "center-branched",
        "far-end": "three-arm-star",
        "near-start": "center-branched",
    },
    2: {
        "uniform": "random-comb",
        "two-ends": "H-type",
        "mid-pair": "dumbbell-mid",
        "H-asymmetric": "H-type",
        "clustered": "random-comb",
        "dumbbell-far": "dumbbell-far",
    },
    3: {
        "uniform": "random-comb",
        "two-ends+mid": "pom-pom",
        "clustered": "random-comb",
        "three-point": "three-point",
        "end-cluster": "asymmetric-comb",
        "pompom-dense": "pom-pom",
    },
    4: {
        "comb-uniform-4": "quad-comb",
        "pompom-4": "quad-pompom",
        "cluster-4": "quad-cluster",
        "gradient": "quad-gradient",
    },
    5: {
        "comb-uniform-5": "penta-comb",
        "pompom-5": "penta-pompom",
        "cluster-5": "penta-cluster",
        "gradient-5": "penta-gradient",
    },
}

COARSE_CODES = {
    "mid": "mid",
    "near-end": "end",
    "far-end": "end",
    "near-start": "end",
    "uniform": "uni",
    "two-ends": "term",
    "mid-pair": "db",
    "H-asymmetric": "asym",
    "clustered": "clus",
    "dumbbell-far": "db",
    "two-ends+mid": "term",
    "three-point": "tri",
    "end-cluster": "clus",
    "pompom-dense": "term",
    "comb-uniform-4": "uni",
    "pompom-4": "term",
    "cluster-4": "clus",
    "gradient": "grad",
    "comb-uniform-5": "uni",
    "pompom-5": "term",
    "cluster-5": "clus",
    "gradient-5": "grad",
}


def clamp_position(backbone_repeat_units: int, position: int) -> int:
    if backbone_repeat_units <= 2:
        return max(1, min(backbone_repeat_units, position))
    return max(2, min(backbone_repeat_units - 1, position))


def normalize_positions(backbone_repeat_units: int, positions: Iterable[int]) -> list[int]:
    normalized = sorted({clamp_position(backbone_repeat_units, int(position)) for position in positions})
    if not normalized:
        raise ValueError("branch positions collapsed to an empty set")
    return normalized


def evenly_spaced(backbone_repeat_units: int, count: int) -> list[int]:
    left = 2 if backbone_repeat_units > 3 else 1
    right = backbone_repeat_units - 1 if backbone_repeat_units > 3 else backbone_repeat_units
    if count == 1:
        return [clamp_position(backbone_repeat_units, int((left + right) / 2))]
    return normalize_positions(
        backbone_repeat_units,
        [int(round(left + ((right - left) * index) / (count - 1))) for index in range(count)],
    )


def branch_positions(backbone_repeat_units: int, branch_points: int, pattern: str) -> list[int]:
    if pattern not in PATTERN_TO_TOPOLOGY.get(branch_points, {}):
        raise ValueError("placement pattern is incompatible with branch_points")
    near_end = clamp_position(backbone_repeat_units, max(2, int(round(backbone_repeat_units * 0.2))))
    far_end = clamp_position(backbone_repeat_units, backbone_repeat_units - near_end + 1)
    center = clamp_position(backbone_repeat_units, int(round(backbone_repeat_units / 2)))
    if pattern == "mid":
        positions = [center]
    elif pattern == "near-end":
        positions = [near_end]
    elif pattern == "near-start":
        positions = [2]
    elif pattern == "far-end":
        positions = [backbone_repeat_units - 1]
    elif pattern in {"uniform", "comb-uniform-4", "comb-uniform-5"}:
        positions = evenly_spaced(backbone_repeat_units, branch_points)
    elif pattern == "two-ends":
        positions = [near_end, far_end]
    elif pattern == "mid-pair":
        positions = [center - 2, center + 2]
    elif pattern == "H-asymmetric":
        positions = [near_end, center]
    elif pattern == "clustered":
        zone = max(branch_points + 2, backbone_repeat_units // 3)
        positions = evenly_spaced(zone, branch_points)
    elif pattern == "dumbbell-far":
        positions = [backbone_repeat_units // 4, 3 * backbone_repeat_units // 4]
    elif pattern == "two-ends+mid":
        positions = [near_end, center, far_end]
    elif pattern == "three-point":
        positions = [backbone_repeat_units // 4, center, 3 * backbone_repeat_units // 4]
    elif pattern == "end-cluster":
        positions = [near_end, near_end + 2, near_end + 4]
    elif pattern == "pompom-dense":
        positions = [near_end, max(2, center - 2), far_end]
    elif pattern == "pompom-4":
        positions = [2, 4, backbone_repeat_units - 3, backbone_repeat_units - 1]
    elif pattern == "cluster-4":
        positions = [center - 3, center - 1, center + 1, center + 3]
    elif pattern == "gradient":
        positions = [
            2,
            max(4, int(round(backbone_repeat_units * 0.25))),
            max(5, int(round(backbone_repeat_units * 0.55))),
            backbone_repeat_units - 1,
        ]
    elif pattern == "pompom-5":
        positions = [2, 4, center, backbone_repeat_units - 3, backbone_repeat_units - 1]
    elif pattern == "cluster-5":
        positions = [center - 4, center - 2, center, center + 2, center + 4]
    elif pattern == "gradient-5":
        positions = [2, 4, max(5, center - 2), min(backbone_repeat_units - 2, center + 4), backbone_repeat_units - 1]
    else:
        raise ValueError("unsupported placement pattern")
    result = normalize_positions(backbone_repeat_units, positions)
    if len(result) != branch_points:
        raise ValueError("placement pattern did not produce the requested number of branch points")
    return result


def macro_smiles(backbone_repeat_units: int, positions: list[int], branch_carbons: int) -> str:
    branch_lookup = set(positions)
    branch = "C" * branch_carbons
    fragments = [f"C({branch})C" if index in branch_lookup else "CC" for index in range(1, backbone_repeat_units + 1)]
    return "*" + "".join(fragments) + "*"


def target_carbons(branch_points: int, branch_carbons: int) -> int:
    return 100 if (100 - branch_points * branch_carbons) % 2 == 0 else 101


def placement_slug(pattern: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", pattern.lower())


def design_key(branch_points: int, branch_carbons: int, pattern: str) -> str:
    return f"b{branch_points}-L{branch_carbons}-{pattern}"


def build_candidate(branch_points: int, branch_carbons: int, pattern: str, *, sequence: int, rationale: str = "") -> dict[str, Any]:
    total = target_carbons(branch_points, branch_carbons)
    backbone_carbons = total - branch_points * branch_carbons
    if backbone_carbons < 20 or backbone_carbons % 2:
        raise ValueError("candidate does not satisfy the C100/C101 backbone budget")
    repeats = backbone_carbons // 2
    positions = branch_positions(repeats, branch_points, pattern)
    smiles = macro_smiles(repeats, positions, branch_carbons)
    key = design_key(branch_points, branch_carbons, pattern)
    case_id = f"gpt_pe_{sequence:03d}_bp{branch_points:02d}_bc{branch_carbons:03d}_pd{placement_slug(pattern)}_c{total}"
    structure_hash = sha256(smiles.encode()).hexdigest()
    rationale = re.sub(r"\s+", " ", rationale).replace(",", ";").strip()[:500]
    return {
        "case_id": case_id,
        "design_key": key,
        "branch_points": branch_points,
        "branch_carbons": branch_carbons,
        "branch_distribution_code": COARSE_CODES[pattern],
        "placement_code": pattern,
        "topology": PATTERN_TO_TOPOLOGY[branch_points][pattern],
        "target_total_carbons": total,
        "backbone_repeat_units": repeats,
        "branch_positions": positions,
        "macro_smiles": smiles,
        "structure_sha256": structure_hash,
        "design_rationale": rationale,
        "builder_spec": {
            "adapter": "pe_lcb",
            "polymer_family": "PE",
            "backbone_repeat_units": repeats,
            "branch_points": branch_points,
            "branch_carbons": branch_carbons,
            "placement_code": pattern,
            "branch_positions": positions,
            "monomer_smiles": smiles,
            "target_total_carbons": total,
            "force_field": "opls/2024/opls-aa",
        },
        "emc_ready": False,
    }


@lru_cache(maxsize=1)
def design_catalog() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    seen_structures: set[str] = set()
    sequence = 1
    for branch_points in sorted(PATTERN_TO_TOPOLOGY):
        for branch_carbons in BRANCH_CARBONS:
            for pattern in PATTERN_TO_TOPOLOGY[branch_points]:
                try:
                    candidate = build_candidate(
                        branch_points,
                        branch_carbons,
                        pattern,
                        sequence=sequence,
                    )
                except ValueError:
                    continue
                if candidate["structure_sha256"] in seen_structures:
                    continue
                seen_structures.add(candidate["structure_sha256"])
                result[candidate["design_key"]] = candidate
                sequence += 1
    return result


def catalog_sha256() -> str:
    payload = [
        {
            "design_key": key,
            "structure_sha256": value["structure_sha256"],
            "builder_spec": value["builder_spec"],
        }
        for key, value in sorted(design_catalog().items())
    ]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def candidates_from_choices(choices: list[dict[str, str]]) -> list[dict[str, Any]]:
    catalog = design_catalog()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sequence, choice in enumerate(choices, start=1):
        key = str(choice.get("design_key") or "")
        if key in seen:
            raise ValueError(f"duplicate design key: {key}")
        if key not in catalog:
            raise ValueError(f"unregistered design key: {key}")
        seen.add(key)
        base = catalog[key]
        result.append(
            build_candidate(
                int(base["branch_points"]),
                int(base["branch_carbons"]),
                str(base["placement_code"]),
                sequence=sequence,
                rationale=str(choice.get("rationale") or ""),
            )
        )
    return result


def candidates_from_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    choices = [
        {
            "design_key": design_key(
                int(item.get("branch_points")),
                int(item.get("branch_carbons")),
                str(item.get("placement_code")),
            ),
            "rationale": str(item.get("rationale") or ""),
        }
        for item in proposals
    ]
    return candidates_from_choices(choices)
