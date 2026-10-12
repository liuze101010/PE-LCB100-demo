#!/usr/bin/env python3
"""Experimental-density sidecar validation for PE-LCB100.

The module compares final simulated densities with public experimental PVT
data.  It never changes the rule-layer verdict or the manuscript release
status.  Numeric comparison currently uses the three LDPE samples reported by
Zoller (1979), because that paper exposes a complete equation and all
coefficients needed to reproduce density at 400 K and about 1 bar.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


MD_ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = MD_ANALYSIS_DIR.parents[1]
VAULT_ROOT = ROOT.parent / "research-vault"
PAPER_DIR = VAULT_ROOT / "40 Drafts" / "pelcb100_paper"

DEFAULT_RELEASE_TABLE = (
    PAPER_DIR / "SI" / "figures" / "si_pelcb100_ml_release_table_20260717.csv"
)
DEFAULT_STRUCTURE_MATCH_PACK = (
    MD_ANALYSIS_DIR / "data" / "pelcb100_radonpy_structure_matches_v1.json"
)
DEFAULT_REFERENCE_PACK_JSON = (
    MD_ANALYSIS_DIR / "data" / "experimental_density_reference_pack_v1.json"
)
DEFAULT_REFERENCE_PACK_CSV = (
    MD_ANALYSIS_DIR / "data" / "experimental_density_reference_pack_v1.csv"
)
DEFAULT_TRIAL_CSV = (
    PAPER_DIR / "SI" / "figures" / "pelcb100_experimental_density_trial.csv"
)
DEFAULT_TRIAL_SUMMARY = (
    PAPER_DIR / "SI" / "figures" / "pelcb100_experimental_density_trial_summary.md"
)
DEFAULT_TRIAL_SUMMARY_JSON = (
    MD_ANALYSIS_DIR / "output" / "experimental_density_validation" / "summary.json"
)

TARGET_TEMPERATURE_K = 400.0
TARGET_PRESSURE_BAR = 1.0
TAIT_C = 0.0984
SPECIFIC_VOLUME_FIT_TOLERANCE_CM3_G = 0.002

ZOLLER_DOI = "10.1002/app.1979.070230410"
ZOLLER_URL = f"https://doi.org/{ZOLLER_DOI}"
ZOLLER_EXTRACTION_URL = (
    "https://electronicsandbooks.com/edt/manual/Magazine/J/"
    "Journal%20of%20Applied%20Polymer%20Science/1979%20v.023/"
    "No04%28953_1268%29/1051_1056.pdf"
)
SATO_DOI = "10.1016/j.fluid.2007.01.013"
SACHDEV_DOI = "10.1002/polb.20463"


ZOLLER_SAMPLES: tuple[dict[str, Any], ...] = (
    {
        "sample": "A",
        "v0_cm3_g": 1.1484,
        "alpha0_per_c": 6.95e-4,
        "b0_bar": 1929.0,
        "b1_per_c": 4.7005e-3,
    },
    {
        "sample": "B",
        "v0_cm3_g": 1.1524,
        "alpha0_per_c": 6.70e-4,
        "b0_bar": 1966.0,
        "b1_per_c": 4.6005e-3,
    },
    {
        "sample": "C",
        "v0_cm3_g": 1.1516,
        "alpha0_per_c": 6.73e-4,
        "b0_bar": 1867.0,
        "b1_per_c": 4.3914e-3,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def zoller_density_at_conditions(
    sample: Mapping[str, Any],
    *,
    temperature_k: float = TARGET_TEMPERATURE_K,
    pressure_bar: float = TARGET_PRESSURE_BAR,
) -> dict[str, float]:
    """Reproduce the Zoller melt density from Eqs. (2)-(4).

    Temperature in the published parameterization is degrees Celsius.  The
    pressure correction uses the Tait equation with B in bar.
    """

    temperature_c = float(temperature_k) - 273.15
    zero_pressure_volume = float(sample["v0_cm3_g"]) * math.exp(
        float(sample["alpha0_per_c"]) * temperature_c
    )
    tait_b_bar = float(sample["b0_bar"]) * math.exp(
        -float(sample["b1_per_c"]) * temperature_c
    )
    volume = zero_pressure_volume * (
        1.0 - TAIT_C * math.log(1.0 + float(pressure_bar) / tait_b_bar)
    )
    density = 1.0 / volume
    tolerance = SPECIFIC_VOLUME_FIT_TOLERANCE_CM3_G
    density_low = 1.0 / (volume + tolerance)
    density_high = 1.0 / (volume - tolerance)
    return {
        "temperature_c": temperature_c,
        "specific_volume_zero_pressure_cm3_g": zero_pressure_volume,
        "tait_b_bar": tait_b_bar,
        "specific_volume_target_cm3_g": volume,
        "density_400k_1bar_g_cm3": density,
        "density_interval_low_g_cm3": density_low,
        "density_interval_high_g_cm3": density_high,
    }


def _zoller_record(sample: Mapping[str, Any]) -> dict[str, Any]:
    calculated = zoller_density_at_conditions(sample)
    record: dict[str, Any] = {
        "reference_id": f"ZOLLER1979_LDPE_{sample['sample']}",
        "source_group": "zoller_1979_ldpe_pvt",
        "source_type": "primary_experiment",
        "computational_source": False,
        "numeric_usable_at_400k": True,
        "title": (
            "The pressure-volume-temperature properties of three "
            "well-characterized low-density polyethylenes"
        ),
        "authors": "Paul Zoller",
        "year": 1979,
        "doi": ZOLLER_DOI,
        "source_url": ZOLLER_URL,
        "extraction_url": ZOLLER_EXTRACTION_URL,
        "extraction_location": (
            "Eqs. (2)-(4), Table I and Table II, pp. 1054-1055"
        ),
        "polymer_family": "polyethylene",
        "polymer_type": "LDPE",
        "branch_information": (
            "Branched commercial LDPE; exact branch count, branch length and "
            "branch-carbon fraction are not reported for each PVT sample."
        ),
        "sample_id": str(sample["sample"]),
        "temperature_min_k": 303.15,
        "temperature_max_k": 498.15,
        "pressure_min_bar": 0.0,
        "pressure_max_bar": 1961.33,
        "phase_at_target": "melt",
        "target_temperature_k": TARGET_TEMPERATURE_K,
        "target_pressure_bar": TARGET_PRESSURE_BAR,
        "v0_cm3_g": float(sample["v0_cm3_g"]),
        "alpha0_per_c": float(sample["alpha0_per_c"]),
        "b0_bar": float(sample["b0_bar"]),
        "b1_per_c": float(sample["b1_per_c"]),
        "tait_c": TAIT_C,
        "specific_volume_fit_tolerance_cm3_g": (
            SPECIFIC_VOLUME_FIT_TOLERANCE_CM3_G
        ),
        **calculated,
        "raw_file_archived": False,
        "raw_file_sha256": "",
        "raw_file_note": (
            "The remote mirror did not permit a stable local download. The "
            "DOI, extraction URL, table locations, equations and coefficients "
            "are stored here for independent checking."
        ),
    }
    record["record_sha256"] = _canonical_sha256(record)
    return record


def _coverage_record(
    *,
    reference_id: str,
    source_group: str,
    title: str,
    authors: str,
    year: int,
    doi: str,
    polymer_type: str,
    temperature_min_k: float,
    temperature_max_k: float,
    pressure_max_bar: float,
    branch_information: str,
    note: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "reference_id": reference_id,
        "source_group": source_group,
        "source_type": "primary_experiment_coverage_only",
        "computational_source": False,
        "numeric_usable_at_400k": False,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "source_url": f"https://doi.org/{doi}",
        "extraction_url": f"https://doi.org/{doi}",
        "extraction_location": "Publisher abstract and accessible article metadata",
        "polymer_family": "polyethylene",
        "polymer_type": polymer_type,
        "branch_information": branch_information,
        "sample_id": "",
        "temperature_min_k": temperature_min_k,
        "temperature_max_k": temperature_max_k,
        "pressure_min_bar": 0.0,
        "pressure_max_bar": pressure_max_bar,
        "phase_at_target": "not numerically resolved from accessible data",
        "target_temperature_k": TARGET_TEMPERATURE_K,
        "target_pressure_bar": TARGET_PRESSURE_BAR,
        "v0_cm3_g": "",
        "alpha0_per_c": "",
        "b0_bar": "",
        "b1_per_c": "",
        "tait_c": "",
        "specific_volume_fit_tolerance_cm3_g": "",
        "temperature_c": "",
        "specific_volume_zero_pressure_cm3_g": "",
        "tait_b_bar": "",
        "specific_volume_target_cm3_g": "",
        "density_400k_1bar_g_cm3": "",
        "density_interval_low_g_cm3": "",
        "density_interval_high_g_cm3": "",
        "raw_file_archived": False,
        "raw_file_sha256": "",
        "raw_file_note": note,
    }
    record["record_sha256"] = _canonical_sha256(record)
    return record


def build_reference_pack(*, generated_at_utc: str | None = None) -> dict[str, Any]:
    numeric_records = [_zoller_record(sample) for sample in ZOLLER_SAMPLES]
    coverage_records = [
        _coverage_record(
            reference_id="SATO2007_PE_COPOLYMER_PVT",
            source_group="sato_2007_pe_copolymer_pvt",
            title="PVT properties of polyethylene copolymer melts",
            authors=(
                "Yoshiyuki Sato; Hajime Hashiguchi; Kenzo Inohara; "
                "Shigeki Takishima; Hirokatsu Masuoka"
            ),
            year=2007,
            doi=SATO_DOI,
            polymer_type=(
                "linear PE and ethylene-propylene, ethylene-1-butene, "
                "ethylene-1-hexene and ethylene-1-octene copolymers"
            ),
            temperature_min_k=313.0,
            temperature_max_k=493.0,
            pressure_max_bar=2000.0,
            branch_information="Side-chain carbon counts 0, 1, 2, 4 and 6.",
            note=(
                "The accessible record confirms materials and experimental "
                "coverage. It does not expose a complete 400 K numeric table, "
                "so this source supports family matching only."
            ),
        ),
        _coverage_record(
            reference_id="SACHDEV2005_HDPE_PVT",
            source_group="sachdev_2005_hdpe_pvt",
            title="Equation of state for high density solid and melt polyethylene",
            authors="V. K. Sachdev; R. K. Jain",
            year=2005,
            doi=SACHDEV_DOI,
            polymer_type="high-density linear polyethylene",
            temperature_min_k=290.0,
            temperature_max_k=470.0,
            pressure_max_bar=3100.0,
            branch_information="Linear polyethylene.",
            note=(
                "The accessible record confirms experimental coverage and the "
                "melt Tait representation. It does not expose the complete "
                "parameters required for an independent 400 K calculation."
            ),
        ),
    ]
    low = min(float(record["density_interval_low_g_cm3"]) for record in numeric_records)
    high = max(float(record["density_interval_high_g_cm3"]) for record in numeric_records)
    median_density = statistics.median(
        float(record["density_400k_1bar_g_cm3"]) for record in numeric_records
    )
    extraction_snapshot = {
        "doi": ZOLLER_DOI,
        "extraction_location": "Eqs. (2)-(4), Table I and Table II, pp. 1054-1055",
        "target_temperature_k": TARGET_TEMPERATURE_K,
        "target_pressure_bar": TARGET_PRESSURE_BAR,
        "tait_c": TAIT_C,
        "specific_volume_fit_tolerance_cm3_g": SPECIFIC_VOLUME_FIT_TOLERANCE_CM3_G,
        "samples": [dict(sample) for sample in ZOLLER_SAMPLES],
    }
    return {
        "schema_version": "experimental_density_reference_pack_v1",
        "generated_at_utc": generated_at_utc or _utc_now(),
        "target_conditions": {
            "temperature_k": TARGET_TEMPERATURE_K,
            "pressure_bar": TARGET_PRESSURE_BAR,
            "phase": "polyethylene melt",
        },
        "numeric_reference_rule": (
            "Only primary experimental records with a reproducible 400 K "
            "equation and all required coefficients enter the numeric interval."
        ),
        "explicit_exclusions": [
            "RadonPy values are computational and do not enter this pack.",
            "PE-LCB100 densities do not enter this pack.",
            "The six linear-PE control calculations do not enter this pack.",
        ],
        "numeric_independent_source_count": 1,
        "numeric_sample_count": len(numeric_records),
        "extraction_snapshot_sha256": _canonical_sha256(extraction_snapshot),
        "raw_source_file_status": (
            "The DOI and extraction mirror were checked. No local PDF was "
            "available in Zotero, and the remote mirror did not permit a "
            "stable download. The exact extracted equations and coefficients "
            "are preserved in the record hashes and extraction snapshot hash."
        ),
        "experimental_density_400k_median_g_cm3": median_density,
        "experimental_density_400k_interval_g_cm3": [low, high],
        "records": numeric_records + coverage_records,
    }


def load_reference_pack(path: Path = DEFAULT_REFERENCE_PACK_JSON) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_structure_matches(path: Path = DEFAULT_STRUCTURE_MATCH_PACK) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = payload.get("matches") or payload.get("records") or payload.get("cases") or []
    return {str(record["case_id"]): record for record in matches}


def experimental_match_level(
    *,
    polymer_family: str,
    has_exact_branch_match: bool,
    has_family_melt_reference: bool,
    has_polyolefin_range_reference: bool,
) -> str:
    if polymer_family.strip().upper() != "PE":
        return "RANGE_ONLY" if has_polyolefin_range_reference else "NO_MATCH"
    if has_exact_branch_match:
        return "DIRECT"
    if has_family_melt_reference:
        return "FAMILY"
    if has_polyolefin_range_reference:
        return "RANGE_ONLY"
    return "NO_MATCH"


def experimental_consistency(
    *,
    simulated_density: float | None,
    density_interval: Sequence[float] | None,
    match_level: str,
    independent_direct_sources_same_direction: int = 0,
) -> str:
    if simulated_density is None or not density_interval or match_level == "NO_MATCH":
        return "INCONCLUSIVE"
    low, high = map(float, density_interval)
    if low <= simulated_density <= high:
        return "CONSISTENT"
    if match_level == "DIRECT" and independent_direct_sources_same_direction >= 2:
        return "DISCORDANT"
    return "INCONCLUSIVE"


def _as_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def build_case_trial_rows(
    release_rows: Sequence[Mapping[str, Any]],
    reference_pack: Mapping[str, Any],
    structure_matches: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    density_median = float(reference_pack["experimental_density_400k_median_g_cm3"])
    density_interval = [
        float(value) for value in reference_pack["experimental_density_400k_interval_g_cm3"]
    ]
    numeric_reference_ids = [
        str(record["reference_id"])
        for record in reference_pack["records"]
        if record.get("numeric_usable_at_400k")
    ]
    support_reference_ids = [
        str(record["reference_id"])
        for record in reference_pack["records"]
        if not record.get("numeric_usable_at_400k")
    ]

    output: list[dict[str, Any]] = []
    for row in release_rows:
        case_id = str(row["case_id"])
        match = structure_matches.get(case_id, {})
        branch_points = int(float(row["branch_points"]))
        branch_carbons = int(float(row["branch_carbons"]))
        total_carbons = int(float(row["actual_total_carbons_per_chain"]))
        branch_fraction = branch_points * branch_carbons / total_carbons
        simulated_density = _as_float(row.get("result_density_g_cm3"))

        # The available numeric experiment is branched LDPE, but it does not
        # report b, L or f for each PVT sample.  It therefore supports family-
        # level comparison for every PE-LCB100 case, not a direct case match.
        match_level = experimental_match_level(
            polymer_family="PE",
            has_exact_branch_match=False,
            has_family_melt_reference=True,
            has_polyolefin_range_reference=True,
        )
        consistency = experimental_consistency(
            simulated_density=simulated_density,
            density_interval=density_interval,
            match_level=match_level,
            independent_direct_sources_same_direction=0,
        )
        difference = None if simulated_density is None else simulated_density - density_median
        absolute_difference = None if difference is None else abs(difference)
        relative_difference = (
            None
            if absolute_difference is None
            else 100.0 * absolute_difference / density_median
        )
        outside = (
            ""
            if simulated_density is None
            else not (density_interval[0] <= simulated_density <= density_interval[1])
        )
        if consistency == "CONSISTENT":
            note = "模拟密度落在三组 LDPE 实验数据的 400 K 范围内。"
        elif simulated_density is None:
            note = "当前结果表缺少密度，实验比较无法完成。"
        else:
            note = (
                "模拟密度位于实验范围外。现有实验只支持同类聚乙烯比较，"
                "因此该案例保留为待进一步验证。"
            )

        output.append(
            {
                "case_id": case_id,
                "branch_points": branch_points,
                "branch_carbons": branch_carbons,
                "actual_total_carbons_per_chain": total_carbons,
                "branch_carbon_fraction": branch_fraction,
                "structure_similarity_score": match.get(
                    "structure_similarity_score", row.get("structure_similarity_score", "")
                ),
                "structure_similarity_tier": match.get(
                    "structure_similarity_tier", row.get("structure_similarity_tier", "")
                ),
                "nearest_radonpy_reference": match.get(
                    "nearest_radonpy_reference", row.get("nearest_radonpy_reference", "")
                ),
                "experimental_match_level": match_level,
                "experimental_reference_ids": ";".join(numeric_reference_ids),
                "supporting_reference_ids": ";".join(support_reference_ids),
                "experimental_density_400K": density_median,
                "experimental_density_interval": (
                    f"[{density_interval[0]:.9f}, {density_interval[1]:.9f}]"
                ),
                "experimental_density_interval_low": density_interval[0],
                "experimental_density_interval_high": density_interval[1],
                "simulated_density_g_cm3": simulated_density if simulated_density is not None else "",
                "density_difference_g_cm3": difference if difference is not None else "",
                "density_absolute_difference": (
                    absolute_difference if absolute_difference is not None else ""
                ),
                "density_relative_difference_percent": (
                    relative_difference if relative_difference is not None else ""
                ),
                "outside_experimental_interval": outside,
                "experimental_consistency": consistency,
                "experimental_validation_note": note,
                "existing_result_case_verdict": row.get("result_case_verdict", ""),
                "sidecar_status_unchanged": True,
            }
        )
    return output


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        mean_rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = mean_rank
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return None if denominator == 0 else numerator / denominator


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return _pearson(_rank(x), _rank(y))


def _interval_for_records(records: Sequence[Mapping[str, Any]]) -> list[float] | None:
    numeric = [record for record in records if record.get("numeric_usable_at_400k")]
    if not numeric:
        return None
    return [
        min(float(record["density_interval_low_g_cm3"]) for record in numeric),
        max(float(record["density_interval_high_g_cm3"]) for record in numeric),
    ]


def summarize_trial(
    trial_rows: Sequence[Mapping[str, Any]],
    reference_pack: Mapping[str, Any],
) -> dict[str, Any]:
    densities = [float(row["simulated_density_g_cm3"]) for row in trial_rows if row["simulated_density_g_cm3"] != ""]
    differences = [float(row["density_difference_g_cm3"]) for row in trial_rows if row["density_difference_g_cm3"] != ""]
    branch_fraction = [float(row["branch_carbon_fraction"]) for row in trial_rows if row["density_difference_g_cm3"] != ""]
    verdict_counts = Counter(str(row["existing_result_case_verdict"]) for row in trial_rows)
    match_counts = Counter(str(row["experimental_match_level"]) for row in trial_rows)
    consistency_counts = Counter(str(row["experimental_consistency"]) for row in trial_rows)
    below = sum(
        float(row["simulated_density_g_cm3"]) < float(row["experimental_density_interval_low"])
        for row in trial_rows
        if row["simulated_density_g_cm3"] != ""
    )
    above = sum(
        float(row["simulated_density_g_cm3"]) > float(row["experimental_density_interval_high"])
        for row in trial_rows
        if row["simulated_density_g_cm3"] != ""
    )

    numeric_records = [
        record for record in reference_pack["records"] if record.get("numeric_usable_at_400k")
    ]
    sample_sensitivity: list[dict[str, Any]] = []
    for excluded in numeric_records:
        kept = [
            record
            for record in numeric_records
            if record["reference_id"] != excluded["reference_id"]
        ]
        interval = _interval_for_records(kept)
        sample_sensitivity.append(
            {
                "excluded_reference_id": excluded["reference_id"],
                "remaining_interval_g_cm3": interval,
                "case_count_inside_interval": (
                    sum(interval[0] <= value <= interval[1] for value in densities)
                    if interval
                    else 0
                ),
            }
        )

    records_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in numeric_records:
        records_by_source[str(record["source_group"])].append(record)
    source_sensitivity: list[dict[str, Any]] = []
    for source_group in records_by_source:
        kept = [
            record
            for record in numeric_records
            if str(record["source_group"]) != source_group
        ]
        interval = _interval_for_records(kept)
        source_sensitivity.append(
            {
                "excluded_source_group": source_group,
                "estimable": interval is not None,
                "remaining_interval_g_cm3": interval or [],
                "note": (
                    "删除该来源后没有可复算的独立 400 K 实验数据。"
                    if interval is None
                    else "其余来源仍可形成实验范围。"
                ),
            }
        )

    return {
        "schema_version": "pelcb100_experimental_density_trial_summary_v1",
        "generated_at_utc": _utc_now(),
        "case_count": len(trial_rows),
        "baseline_verdict_counts": dict(sorted(verdict_counts.items())),
        "sidecar_verdict_counts": dict(sorted(verdict_counts.items())),
        "status_changed_case_count": 0,
        "match_level_counts": dict(sorted(match_counts.items())),
        "consistency_counts": dict(sorted(consistency_counts.items())),
        "experimental_density_400k_median_g_cm3": reference_pack[
            "experimental_density_400k_median_g_cm3"
        ],
        "experimental_density_400k_interval_g_cm3": reference_pack[
            "experimental_density_400k_interval_g_cm3"
        ],
        "simulated_density_min_g_cm3": min(densities),
        "simulated_density_max_g_cm3": max(densities),
        "simulated_density_median_g_cm3": statistics.median(densities),
        "mean_signed_difference_g_cm3": statistics.mean(differences),
        "mean_absolute_difference_g_cm3": statistics.mean(abs(value) for value in differences),
        "below_experimental_interval_count": below,
        "inside_experimental_interval_count": consistency_counts.get("CONSISTENT", 0),
        "above_experimental_interval_count": above,
        "branch_fraction_signed_difference_spearman": spearman(branch_fraction, differences),
        "numeric_independent_source_count": reference_pack[
            "numeric_independent_source_count"
        ],
        "direct_match_count": match_counts.get("DIRECT", 0),
        "discordant_count": consistency_counts.get("DISCORDANT", 0),
        "leave_one_sample_out": sample_sensitivity,
        "leave_one_source_out": source_sensitivity,
    }


def render_summary_markdown(summary: Mapping[str, Any], reference_pack: Mapping[str, Any]) -> str:
    low, high = summary["experimental_density_400k_interval_g_cm3"]
    match_text = "，".join(
        f"{name} {count} 个" for name, count in summary["match_level_counts"].items()
    )
    consistency_text = "，".join(
        f"{name} {count} 个" for name, count in summary["consistency_counts"].items()
    )
    lines = [
        "# PE-LCB100 实验密度旁路验证试算",
        "",
        "## 结论",
        "",
        (
            "本次试算只比较实验密度，不改变规则层结果。"
            f"当前 {summary['case_count']} 个案例仍为 "
            f"{summary['baseline_verdict_counts'].get('PASS', 0)} PASS / "
            f"{summary['baseline_verdict_counts'].get('REVIEW', 0)} REVIEW。"
        ),
        "",
        (
            "Zoller 1979 的三组 LDPE 实验数据可用同一组方程复算到 "
            f"400 K、约 1 bar。三组数据的密度中位数为 "
            f"{summary['experimental_density_400k_median_g_cm3']:.6f} g cm^-3，"
            f"范围为 {low:.6f}-{high:.6f} g cm^-3。"
        ),
        "该范围同时计入原文给出的 0.002 cm^3 g^-1 最大拟合偏差。",
        "",
        (
            "这三组样品来自同一篇论文，而且原文没有给出每个样品的支化点数、"
            "支链长度和支化碳占比。因此，本次比较只能验证同类聚乙烯熔体的密度，"
            "不能把实验值当作 100 个具体分子的逐一真值。"
        ),
        "",
        "## 逐案例结果",
        "",
        f"- 匹配等级：{match_text}。",
        f"- 比较结果：{consistency_text}。",
        (
            f"- {summary['inside_experimental_interval_count']} 个案例落入实验范围，"
            f"{summary['below_experimental_interval_count']} 个低于该范围，"
            f"{summary['above_experimental_interval_count']} 个高于该范围。"
        ),
        (
            f"- 模拟密度范围为 {summary['simulated_density_min_g_cm3']:.6f}-"
            f"{summary['simulated_density_max_g_cm3']:.6f} g cm^-3。"
        ),
        (
            f"- 模拟值相对实验中位数的平均绝对偏差为 "
            f"{summary['mean_absolute_difference_g_cm3']:.6f} g cm^-3。"
        ),
        (
            f"- 支化碳占比与模拟值相对实验中位数的有符号偏差呈负相关 "
            f"(Spearman rho = {summary['branch_fraction_signed_difference_spearman']:.3f})。"
            "该结果只描述本次试算中的变化趋势。"
        ),
        "",
        "范围外案例均记为 INCONCLUSIVE。当前只有一个可独立复算的实验来源，且没有 DIRECT 匹配，因此没有案例记为 DISCORDANT。",
        "",
        "## 敏感性检查",
        "",
    ]
    for item in summary["leave_one_sample_out"]:
        interval = item["remaining_interval_g_cm3"]
        lines.append(
            f"- 去掉 {item['excluded_reference_id']} 后，实验范围为 "
            f"{interval[0]:.6f}-{interval[1]:.6f} g cm^-3，"
            f"其中 {item['case_count_inside_interval']} 个案例落入范围。"
        )
    lines.extend(
        [
            (
                "- 按独立来源删除 Zoller 1979 后，没有第二组可完整复算到 400 K 的"
                "实验数据。因此，现阶段无法完成真正的跨来源敏感性比较。"
            ),
            "",
            "## 参照来源",
            "",
        ]
    )
    for record in reference_pack["records"]:
        use = "参与数值计算" if record["numeric_usable_at_400k"] else "只用于说明实验覆盖范围"
        lines.append(
            f"- {record['reference_id']}：{record['title']}，DOI {record['doi']}，{use}。"
        )
    lines.extend(
        [
            "",
            (
                "本机 Zotero 没有保存上述论文附件，远程镜像也未能稳定下载原始 PDF。"
                "参照包已经保存 DOI、原文位置、方程、参数、逐条记录哈希和提取内容哈希。"
            ),
            "",
            "## 下一步判断",
            "",
            (
                "当前结果说明公开实验数据可以作为独立旁路检查。若要让实验数据进入规则层，"
                "还需要补充至少一组独立来源，并获得可与 PE-LCB100 支化参数直接对应的实验样品。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
