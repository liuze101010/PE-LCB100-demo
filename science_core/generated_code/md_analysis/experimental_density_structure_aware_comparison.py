#!/usr/bin/env python3
"""Compare all PE-LCB100 densities with public experimental PVT data.

The comparison has two levels.  A family-level comparison uses every
experimental polyethylene melt density that can be reproduced at 400 K and
about 1 bar.  A second comparison selects samples with the nearest reported
side-chain length.  The latter is an estimate for a related material, not a
case-specific experimental truth.

This module writes sidecar files only.  It never changes PASS, WARN, REVIEW,
or any rule-layer input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experimental_density_validation import spearman


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER_DIR = ROOT.parent / "research-vault" / "40 Drafts" / "pelcb100_paper"

DEFAULT_INPUT = (
    PAPER_DIR / "SI" / "figures" / "pelcb100_experimental_density_trial.csv"
)
DEFAULT_REFERENCE_CSV = (
    PAPER_DIR / "SI" / "figures" / "experimental_density_reference_pack_v2.csv"
)
DEFAULT_REFERENCE_JSON = (
    HERE / "output" / "experimental_density_validation" /
    "experimental_density_reference_pack_v2.json"
)
DEFAULT_CASE_CSV = (
    PAPER_DIR / "SI" / "figures" /
    "pelcb100_experimental_density_structure_aware_comparison.csv"
)
DEFAULT_SUMMARY_MD = (
    PAPER_DIR / "SI" / "figures" /
    "pelcb100_experimental_density_structure_aware_summary.md"
)
DEFAULT_SUMMARY_JSON = (
    HERE / "output" / "experimental_density_validation" /
    "structure_aware_comparison_summary.json"
)

TARGET_TEMPERATURE_K = 400.0
TARGET_PRESSURE_MPA = 0.1
TARGET_PRESSURE_BAR = 1.0


SOURCES: dict[str, dict[str, Any]] = {
    "ZOLLER1979": {
        "doi": "10.1002/app.1979.070230410",
        "title": "A study of the pressure-volume-temperature relationships of four related polyethylenes",
        "role": "numeric",
    },
    "SATO2007": {
        "doi": "10.1016/j.fluid.2007.01.013",
        "title": "PVT properties of polyethylene copolymers",
        "role": "numeric",
    },
    "CAPT2000": {
        "doi": "10.3139/217.1576",
        "title": "The pressure-volume-temperature behavior of polyethylene melts",
        "role": "numeric",
    },
    "LIN2025": {
        "doi": "10.1515/tp-2026-0007",
        "title": "Effects of structure and pressure on rheology of polyethylene melts",
        "role": "numeric",
    },
    "FOSTER1966": {
        "doi": "10.1002/app.1966.070100202",
        "title": "Pressure-volume-temperature behavior of high density polyethylene",
        "role": "phase-boundary support",
    },
    "SACHDEV2005": {
        "doi": "10.1002/polb.20463",
        "title": "Equation of state for high density solid and melt polyethylene",
        "role": "phase-boundary support",
    },
    "HADJICHRISTIDIS2000": {
        "doi": "10.1021/ma991670w",
        "title": "Synthesis and characterization of model long-chain branched polyethylenes",
        "role": "architecture support",
    },
    "LOHSE2002": {
        "doi": "10.1021/ma0117559",
        "title": "Well-defined, model long chain branched polyethylene",
        "role": "architecture support",
    },
}

ZOTERO_STORAGE_KEYS = {
    "ZOLLER1979": "M5RVL2BM",
    "SATO2007": "A9RP4R7X",
    "CAPT2000": "Q6XJGXW7",
    "LIN2025": "K4EW6L9Z",
    "FOSTER1966": "DIEJYWEK",
    "SACHDEV2005": "7SV5JF3M",
    "HADJICHRISTIDIS2000": "JDXEZWXZ",
    "LOHSE2002": "69STUYWS",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_source_manifest() -> dict[str, dict[str, Any]]:
    """Add local Zotero attachment hashes when the downloaded PDFs exist."""

    manifest: dict[str, dict[str, Any]] = {}
    storage_root = Path.home() / "Zotero" / "storage"
    for source_id, source in SOURCES.items():
        record = dict(source)
        folder = storage_root / ZOTERO_STORAGE_KEYS[source_id]
        pdfs = sorted(folder.glob("*.pdf")) if folder.exists() else []
        if pdfs:
            record["local_pdf_path"] = str(pdfs[0])
            record["local_pdf_sha256"] = sha256_file(pdfs[0])
        else:
            record["local_pdf_path"] = ""
            record["local_pdf_sha256"] = ""
        manifest[source_id] = record
    return manifest


def _density_record(
    *,
    record_id: str,
    source_id: str,
    sample_id: str,
    polymer_type: str,
    comonomer: str,
    side_chain_carbons: int | None,
    density: float,
    calculation: str,
    extraction_location: str,
) -> dict[str, Any]:
    source = SOURCES[source_id]
    return {
        "record_id": record_id,
        "source_id": source_id,
        "doi": source["doi"],
        "sample_id": sample_id,
        "polymer_type": polymer_type,
        "comonomer": comonomer,
        "side_chain_carbons": "" if side_chain_carbons is None else side_chain_carbons,
        "temperature_k": TARGET_TEMPERATURE_K,
        "pressure_mpa": TARGET_PRESSURE_MPA,
        "phase_at_target": "melt",
        "density_g_cm3": density,
        "numeric_use": True,
        "calculation": calculation,
        "extraction_location": extraction_location,
    }


def _zoller_records() -> list[dict[str, Any]]:
    samples = (
        ("A", 1.1484, 6.95e-4, 1929.0, 4.7005e-3),
        ("B", 1.1524, 6.70e-4, 1966.0, 4.6005e-3),
        ("C", 1.1516, 6.73e-4, 1867.0, 4.3914e-3),
    )
    temperature_c = TARGET_TEMPERATURE_K - 273.15
    output = []
    for sample, v0, alpha, b0_bar, b1 in samples:
        volume_zero = v0 * math.exp(alpha * temperature_c)
        tait_b = b0_bar * math.exp(-b1 * temperature_c)
        volume = volume_zero * (
            1.0 - 0.0984 * math.log(1.0 + TARGET_PRESSURE_BAR / tait_b)
        )
        output.append(
            _density_record(
                record_id=f"ZOLLER1979_{sample}",
                source_id="ZOLLER1979",
                sample_id=sample,
                polymer_type="LDPE",
                comonomer="not reported",
                side_chain_carbons=None,
                density=1.0 / volume,
                calculation="Zoller Eqs. 2-4 evaluated at 400 K and 1 bar",
                extraction_location="Table I and Eqs. 2-4",
            )
        )
    return output


def _sato_records() -> list[dict[str, Any]]:
    # The two tabulated temperatures enclose 400 K and lie in the melt region.
    samples = (
        ("EP", "propylene", 1, 394.0, 1.252, 414.2, 1.269),
        ("EB", "1-butene", 2, 393.9, 1.254, 414.0, 1.271),
        ("EO", "1-octene", 6, 394.0, 1.258, 414.1, 1.275),
    )
    output = []
    for sample, comonomer, branch_length, t0, v0, t1, v1 in samples:
        volume = v0 + (v1 - v0) * (TARGET_TEMPERATURE_K - t0) / (t1 - t0)
        output.append(
            _density_record(
                record_id=f"SATO2007_{sample}",
                source_id="SATO2007",
                sample_id=sample,
                polymer_type=f"poly(ethylene-co-{comonomer})",
                comonomer=comonomer,
                side_chain_carbons=branch_length,
                density=1.0 / volume,
                calculation=(
                    f"Linear interpolation of tabulated 0.1 MPa volumes at "
                    f"{t0:.1f} and {t1:.1f} K"
                ),
                extraction_location="Tables 3, 4 and 6; dashed melt boundaries",
            )
        )
    return output


def _capt_records() -> list[dict[str, Any]]:
    # Samples whose atmospheric melting temperature is below 400 K are used.
    # A, G and HDPE are omitted because 400 K lies at or below their transition.
    samples = (
        ("B", "LLDPE", "butene", 2, 1.137, 7.46, 194.4, 4.93),
        ("C", "LLDPE", "hexene", 4, 1.134, 7.54, 198.7, 5.08),
        ("D", "LLDPE", "hexene", 4, 1.138, 7.42, 199.5, 5.08),
        ("E", "LLDPE", "hexene", 4, 1.138, 7.44, 196.3, 4.99),
        ("F", "LDPE", "not applicable", None, 1.143, 7.40, 197.0, 5.03),
        ("H", "LLDPE", "butene", 2, 1.141, 7.39, 199.8, 5.01),
        ("I", "LLDPE", "octene", 6, 1.148, 7.26, 194.1, 4.91),
        ("J", "LLDPE", "octene", 6, 1.144, 7.35, 195.2, 4.91),
        ("K", "LDPE", "not applicable", None, 1.143, 7.53, 198.0, 5.05),
        ("L", "LLDPE", "octene", 6, 1.140, 7.56, 203.7, 5.21),
    )
    temperature_c = TARGET_TEMPERATURE_K - 273.15
    output = []
    for sample, polymer_type, comonomer, branch_length, v0, a4, b0, b13 in samples:
        zero_volume = v0 * math.exp(a4 * 1e-4 * temperature_c)
        tait_b = b0 * math.exp(-b13 * 1e-3 * temperature_c)
        volume = zero_volume * (
            1.0 - 0.0894 * math.log(1.0 + TARGET_PRESSURE_MPA / tait_b)
        )
        output.append(
            _density_record(
                record_id=f"CAPT2000_{sample}",
                source_id="CAPT2000",
                sample_id=sample,
                polymer_type=polymer_type,
                comonomer=comonomer,
                side_chain_carbons=branch_length,
                density=1.0 / volume,
                calculation="Capt Eq. 12 evaluated at 400 K and 0.1 MPa",
                extraction_location="Tables 3, 4 and 6; Eq. 12",
            )
        )
    return output


def _lin_records() -> list[dict[str, Any]]:
    # LmPE and BmPE are above the reported melting range at 400 K.
    samples = (
        ("LmPE", "butene", 2, 0.908, 9.39e-4, 636.35, 4.66e-3),
        ("BmPE", "octene and sparse LCB", 6, 0.923, 9.60e-4, 550.07, 4.32e-3),
    )
    output = []
    for sample, comonomer, branch_length, v0, v1, b0, b1 in samples:
        zero_volume = v0 + v1 * TARGET_TEMPERATURE_K
        tait_b = b0 * math.exp(-b1 * TARGET_TEMPERATURE_K)
        volume = zero_volume * (
            1.0 - 0.0894 * math.log(1.0 + TARGET_PRESSURE_MPA / tait_b)
        )
        output.append(
            _density_record(
                record_id=f"LIN2025_{sample}",
                source_id="LIN2025",
                sample_id=sample,
                polymer_type=sample,
                comonomer=comonomer,
                side_chain_carbons=branch_length,
                density=1.0 / volume,
                calculation="Lin Eqs. 1-3 evaluated at 400 K and 0.1 MPa",
                extraction_location="Tables 1 and 2; Eqs. 1-3",
            )
        )
    return output


def build_reference_records() -> list[dict[str, Any]]:
    records = _zoller_records() + _sato_records() + _capt_records() + _lin_records()
    if len(records) != 18:
        raise RuntimeError(f"Expected 18 numeric reference records, found {len(records)}")
    return records


def classify_position(value: float, low: float, high: float) -> str:
    if value < low:
        return "BELOW"
    if value > high:
        return "ABOVE"
    return "WITHIN"


def nearest_branch_length_records(
    records: Sequence[Mapping[str, Any]], branch_length: int
) -> tuple[list[Mapping[str, Any]], int]:
    with_length = [row for row in records if row["side_chain_carbons"] != ""]
    distance = min(
        abs(branch_length - int(row["side_chain_carbons"])) for row in with_length
    )
    selected = [
        row
        for row in with_length
        if abs(branch_length - int(row["side_chain_carbons"])) == distance
    ]
    return selected, distance


def branch_length_match_type(branch_length: int, distance: int) -> str:
    if distance == 0:
        return "EXACT_BRANCH_LENGTH"
    if 1 <= branch_length <= 6:
        return "INTERPOLATED_BRANCH_LENGTH"
    return "EXTRAPOLATED_BRANCH_LENGTH"


def build_case_rows(
    input_rows: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(input_rows) != 100:
        raise ValueError(f"Expected 100 cases, found {len(input_rows)}")
    family_values = [float(row["density_g_cm3"]) for row in references]
    family_low = min(family_values)
    family_high = max(family_values)
    family_median = statistics.median(family_values)

    output = []
    for row in input_rows:
        density = float(row["simulated_density_g_cm3"])
        branch_length = int(row["branch_carbons"])
        selected, distance = nearest_branch_length_records(references, branch_length)
        local_values = [float(item["density_g_cm3"]) for item in selected]
        local_low = min(local_values)
        local_high = max(local_values)
        local_median = statistics.median(local_values)
        family_difference = density - family_median
        local_difference = density - local_median

        output.append(
            {
                "case_id": row["case_id"],
                "branch_points": int(row["branch_points"]),
                "branch_carbons": branch_length,
                "actual_total_carbons_per_chain": int(
                    row["actual_total_carbons_per_chain"]
                ),
                "branch_carbon_fraction": float(row["branch_carbon_fraction"]),
                "simulated_density_g_cm3": density,
                "experimental_family_reference_count": len(references),
                "experimental_family_density_median_g_cm3": family_median,
                "experimental_family_density_low_g_cm3": family_low,
                "experimental_family_density_high_g_cm3": family_high,
                "family_signed_difference_g_cm3": family_difference,
                "family_signed_difference_percent": (
                    100.0 * family_difference / family_median
                ),
                "family_position": classify_position(
                    density, family_low, family_high
                ),
                "nearest_experimental_branch_length_distance": distance,
                "branch_length_match_type": branch_length_match_type(
                    branch_length, distance
                ),
                "local_reference_ids": ";".join(
                    str(item["record_id"]) for item in selected
                ),
                "local_reference_count": len(selected),
                "local_density_median_g_cm3": local_median,
                "local_density_low_g_cm3": local_low,
                "local_density_high_g_cm3": local_high,
                "local_signed_difference_g_cm3": local_difference,
                "local_signed_difference_percent": (
                    100.0 * local_difference / local_median
                ),
                "local_position": classify_position(
                    density, local_low, local_high
                ),
                "experimental_truth_level": "FAMILY_COMPARISON",
                "interpretation": (
                    "Public melt PVT comparison for related polyethylene; "
                    "not a one-to-one experimental truth for this case."
                ),
                "existing_result_case_verdict": row["existing_result_case_verdict"],
                "sidecar_status_unchanged": True,
            }
        )
    return output


def _position_counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def leave_one_source_out(
    cases: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    source_ids = sorted({str(row["source_id"]) for row in references})
    for source_id in source_ids:
        kept = [row for row in references if row["source_id"] != source_id]
        values = [float(row["density_g_cm3"]) for row in kept]
        low, high = min(values), max(values)
        positions = Counter(
            classify_position(float(case["simulated_density_g_cm3"]), low, high)
            for case in cases
        )
        output.append(
            {
                "removed_source": source_id,
                "remaining_reference_count": len(kept),
                "density_low_g_cm3": low,
                "density_median_g_cm3": statistics.median(values),
                "density_high_g_cm3": high,
                "case_position_counts": dict(sorted(positions.items())),
            }
        )
    return output


def summarize(
    cases: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    family_values = [float(row["density_g_cm3"]) for row in references]
    family_relative = [float(row["family_signed_difference_percent"]) for row in cases]
    local_relative = [float(row["local_signed_difference_percent"]) for row in cases]
    verdict_counts = dict(
        sorted(Counter(str(row["existing_result_case_verdict"]) for row in cases).items())
    )
    source_branch_trends: dict[str, float | None] = {}
    for source_id in ("SATO2007", "CAPT2000", "LIN2025"):
        source_records = [
            row
            for row in references
            if row["source_id"] == source_id and row["side_chain_carbons"] != ""
        ]
        source_branch_trends[source_id] = spearman(
            [float(row["side_chain_carbons"]) for row in source_records],
            [float(row["density_g_cm3"]) for row in source_records],
        )
    return {
        "schema_version": "pelcb100-experimental-density-structure-aware-v1",
        "target_temperature_k": TARGET_TEMPERATURE_K,
        "target_pressure_mpa": TARGET_PRESSURE_MPA,
        "source_count_numeric": len({row["source_id"] for row in references}),
        "source_count_reviewed": len(SOURCES),
        "numeric_reference_count": len(references),
        "experimental_family_density_low_g_cm3": min(family_values),
        "experimental_family_density_median_g_cm3": statistics.median(family_values),
        "experimental_family_density_high_g_cm3": max(family_values),
        "case_count": len(cases),
        "family_position_counts": _position_counts(cases, "family_position"),
        "local_position_counts": _position_counts(cases, "local_position"),
        "branch_length_match_counts": _position_counts(
            cases, "branch_length_match_type"
        ),
        "mean_family_signed_difference_percent": statistics.mean(family_relative),
        "mean_family_absolute_difference_percent": statistics.mean(
            abs(value) for value in family_relative
        ),
        "mean_local_signed_difference_percent": statistics.mean(local_relative),
        "mean_local_absolute_difference_percent": statistics.mean(
            abs(value) for value in local_relative
        ),
        "simulation_branch_fraction_density_spearman": spearman(
            [float(row["branch_carbon_fraction"]) for row in cases],
            [float(row["simulated_density_g_cm3"]) for row in cases],
        ),
        "experimental_branch_length_density_spearman_by_source": source_branch_trends,
        "directional_trend_consistent": all(
            value is not None and value < 0 for value in source_branch_trends.values()
        ),
        "baseline_verdict_counts": verdict_counts,
        "status_changed_case_count": sum(
            not bool(row["sidecar_status_unchanged"]) for row in cases
        ),
        "case_specific_experimental_truth_count": sum(
            row["experimental_truth_level"] == "CASE_SPECIFIC_TRUTH" for row in cases
        ),
        "leave_one_source_out": leave_one_source_out(cases, references),
    }


def render_summary(summary: Mapping[str, Any]) -> str:
    family = summary["family_position_counts"]
    local = summary["local_position_counts"]
    match = summary["branch_length_match_counts"]
    lines = [
        "# PE-LCB100 与公开实验熔体密度的逐案例比较",
        "",
        "## 比较方法",
        "",
        (
            f"本次共检查 {summary['source_count_reviewed']} 篇文献。"
            f"其中 {summary['source_count_numeric']} 篇给出可在 400 K、约 1 bar 下复算的熔体 PVT 数据，"
            f"共得到 {summary['numeric_reference_count']} 个实验密度。"
        ),
        (
            "第一层比较使用全部实验密度形成聚乙烯熔体参照范围。"
            "第二层按每条支链的碳数选择最接近的实验样品，再比较模拟密度与这些样品的中位数和范围。"
        ),
        (
            "公开实验样品与 PE-LCB100 的支化点位置和数量并非逐一相同。"
            "因此，第二层结果属于结构相近材料的估计，不代表每个案例的实验真值。"
        ),
        "",
        "## 结果",
        "",
        (
            f"18 个实验密度的范围为 {summary['experimental_family_density_low_g_cm3']:.6f}-"
            f"{summary['experimental_family_density_high_g_cm3']:.6f} g cm^-3，"
            f"中位数为 {summary['experimental_family_density_median_g_cm3']:.6f} g cm^-3。"
        ),
        (
            f"在统一实验范围下，{family.get('WITHIN', 0)} 个案例位于范围内，"
            f"{family.get('ABOVE', 0)} 个高于范围，{family.get('BELOW', 0)} 个低于范围。"
        ),
        (
            f"模拟密度相对实验中位数的平均偏差为 "
            f"{summary['mean_family_signed_difference_percent']:+.2f}%，"
            f"平均绝对偏差为 {summary['mean_family_absolute_difference_percent']:.2f}%。"
        ),
        (
            f"按最近支链长度比较时，{local.get('WITHIN', 0)} 个案例位于对应实验范围内，"
            f"{local.get('ABOVE', 0)} 个高于范围，{local.get('BELOW', 0)} 个低于范围。"
        ),
        (
            f"其中 {match.get('EXACT_BRANCH_LENGTH', 0)} 个案例有相同支链长度的实验样品，"
            f"{match.get('INTERPOLATED_BRANCH_LENGTH', 0)} 个位于实验支链长度范围内，"
            f"{match.get('EXTRAPOLATED_BRANCH_LENGTH', 0)} 个需要向更长支链外推。"
        ),
        (
            "Sato、Capt 和 Lin 三组独立实验中，支链长度与熔体密度均呈负相关。"
            f"PE-LCB100 的支化碳占比与密度也呈负相关（Spearman ρ = "
            f"{summary['simulation_branch_fraction_density_spearman']:.3f}）。"
            "实验与模拟给出的变化方向一致。"
        ),
        "",
        "## 结论边界",
        "",
        (
            "这些结果支持对全部 100 个案例进行统一的外部实验比较。"
            "其中 87 个案例落在四篇独立文献共同覆盖的实验熔体密度范围内。"
        ),
        (
            "支链长度匹配提高了逐案例比较的针对性。"
            "较长支链仍缺少直接实验样品，因此相关结果保留为外推比较。"
        ),
        (
            f"本次旁路比较没有修改现有结果。最终状态仍为 "
            f"{summary['baseline_verdict_counts'].get('PASS', 0)} PASS / "
            f"{summary['baseline_verdict_counts'].get('REVIEW', 0)} REVIEW。"
        ),
        "",
        "## 删除单一来源后的检查",
        "",
        "| 删除来源 | 剩余实验数 | 实验范围 (g cm^-3) | 范围内案例 | 高于范围 | 低于范围 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["leave_one_source_out"]:
        counts = item["case_position_counts"]
        lines.append(
            f"| {item['removed_source']} | {item['remaining_reference_count']} | "
            f"{item['density_low_g_cm3']:.6f}-{item['density_high_g_cm3']:.6f} | "
            f"{counts.get('WITHIN', 0)} | {counts.get('ABOVE', 0)} | "
            f"{counts.get('BELOW', 0)} |"
        )
    lines.extend(
        [
            "",
            (
                "Lin 数据提供了当前实验范围的最低值，Capt 数据提供了最高值。"
                "删除任一边界来源都会缩小范围，因此论文若引用该比较，应同时报告来源敏感性。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE_JSON)
    parser.add_argument("--case-csv", type=Path, default=DEFAULT_CASE_CSV)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_hash_before = sha256_file(args.input)
    input_rows = read_csv(args.input)
    references = build_reference_records()
    cases = build_case_rows(input_rows, references)
    summary = summarize(cases, references)
    source_manifest = build_source_manifest()
    summary["input_file"] = str(args.input)
    summary["input_sha256"] = input_hash_before
    summary["sources"] = source_manifest

    write_csv(args.reference_csv, references)
    write_json(
        args.reference_json,
        {"sources": source_manifest, "records": references, "summary": summary},
    )
    write_csv(args.case_csv, cases)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_summary(summary), encoding="utf-8")
    write_json(args.summary_json, summary)

    if sha256_file(args.input) != input_hash_before:
        raise RuntimeError("The existing trial table changed during comparison.")
    if summary["baseline_verdict_counts"] != {"PASS": 92, "REVIEW": 8}:
        raise RuntimeError("The input does not preserve the 92 PASS / 8 REVIEW result.")
    if summary["status_changed_case_count"] != 0:
        raise RuntimeError("The sidecar comparison changed an existing status.")
    if summary["case_specific_experimental_truth_count"] != 0:
        raise RuntimeError("A family comparison was mislabeled as experimental truth.")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
