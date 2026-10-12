from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .schemas import ProjectSpec
from .design_grammar import build_candidate, catalog_sha256
from ..live_design import benchmark_manifest


VALID_BRANCH_CARBONS = {3, 4, 5, 8, 12, 16, 20}
VALID_DISTRIBUTIONS = {"asym", "clus", "db", "end", "grad", "mid", "term", "tri", "uni"}
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PELCB100_MANIFEST = PACKAGE_ROOT / "data" / "pelcb100_design_manifest.json"
PELCB100_RELEASE_ORACLE = PACKAGE_ROOT / "data" / "pelcb100_release_oracle.json"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    domain_status: str
    errors: list[dict[str, str]]
    warnings: list[dict[str, str]]
    manifest: list[dict[str, Any]]
    manifest_sha256: str


class PELCBAdapter:
    adapter_id = "pe_lcb"
    validated_profile = "pelcb100_validated_v1"

    def validate(self, spec: ProjectSpec) -> ValidationResult:
        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []
        seen: set[tuple[int, int, str]] = set()
        seen_ids: set[str] = set()
        seen_structures: set[str] = set()
        manifest: list[dict[str, Any]] = []
        if spec.planning:
            if spec.planning.requested_count != len(spec.candidates):
                errors.append({"field": "planning.requested_count", "code": "planned_count_mismatch"})
            expected_catalog_sha = catalog_sha256()
            if spec.planning.mode == "reproduce_pelcb100":
                _, expected_benchmark_sha = benchmark_manifest()
                if spec.planning.benchmark_manifest_sha256 != expected_benchmark_sha:
                    errors.append({"field": "planning.benchmark_manifest_sha256", "code": "benchmark_manifest_version_mismatch"})
            if spec.planning.catalog_sha256 != expected_catalog_sha:
                errors.append({"field": "planning.catalog_sha256", "code": "design_catalog_version_mismatch"})
        in_domain = (
            spec.polymer_family.upper() == "PE"
            and spec.architecture == "pe_lcb"
            and spec.force_field.lower() == "opls/2024/opls-aa"
            and abs(spec.temperature_k - 400.0) < 1e-9
            and abs(spec.pressure_atm - 1.0) < 1e-9
            and spec.chain_count == 150
            and spec.production_ns == 400
        )
        expected_protocol = {
            "polymer_family": spec.polymer_family.upper() == "PE",
            "architecture": spec.architecture == "pe_lcb",
            "force_field": spec.force_field.lower() == "opls/2024/opls-aa",
            "temperature_k": abs(spec.temperature_k - 400.0) < 1e-9,
            "pressure_atm": abs(spec.pressure_atm - 1.0) < 1e-9,
            "chain_count": spec.chain_count == 150,
            "production_ns": spec.production_ns == 400,
        }
        for field, ok in expected_protocol.items():
            if not ok:
                warnings.append({"field": field, "code": "outside_validated_pelcb100_protocol"})
        for index, candidate in enumerate(spec.candidates):
            loc = f"candidates[{index}]"
            key = (
                candidate.branch_points,
                candidate.branch_carbons,
                candidate.placement_code or candidate.branch_distribution_code,
            )
            if candidate.case_id in seen_ids:
                errors.append({"field": f"{loc}.case_id", "code": "duplicate_case_id"})
            if key in seen:
                warnings.append({"field": loc, "code": "repeated_coarse_design_coordinate"})
            seen_ids.add(candidate.case_id)
            seen.add(key)
            if candidate.branch_carbons not in VALID_BRANCH_CARBONS:
                in_domain = False
                warnings.append({"field": f"{loc}.branch_carbons", "code": "outside_validated_branch_lengths"})
            if candidate.branch_distribution_code not in VALID_DISTRIBUTIONS:
                in_domain = False
                warnings.append({"field": f"{loc}.branch_distribution_code", "code": "outside_validated_placement_grammar"})
            if candidate.target_total_carbons not in {100, 101}:
                in_domain = False
                warnings.append({"field": f"{loc}.target_total_carbons", "code": "outside_validated_carbon_budget"})
            backbone_carbons = candidate.target_total_carbons - candidate.branch_points * candidate.branch_carbons
            if backbone_carbons < 20:
                errors.append({"field": loc, "code": "backbone_too_short"})
            candidate_payload = candidate.model_dump()
            if candidate.placement_code:
                try:
                    expected = build_candidate(
                        candidate.branch_points,
                        candidate.branch_carbons,
                        candidate.placement_code,
                        sequence=index + 1,
                        rationale=candidate.design_rationale,
                    )
                except ValueError:
                    errors.append({"field": f"{loc}.placement_code", "code": "placement_incompatible_or_unbuildable"})
                else:
                    checked_fields = (
                        "design_key",
                        "target_total_carbons",
                        "topology",
                        "backbone_repeat_units",
                        "branch_positions",
                        "macro_smiles",
                        "structure_sha256",
                        "builder_spec",
                    )
                    if any(
                        candidate_payload.get(field) is not None and candidate_payload.get(field) != expected.get(field)
                        for field in checked_fields
                    ):
                        errors.append({"field": loc, "code": "deterministic_builder_spec_mismatch"})
                    if expected["structure_sha256"] in seen_structures:
                        errors.append({"field": loc, "code": "duplicate_molecular_structure"})
                    seen_structures.add(expected["structure_sha256"])
                    candidate_payload = expected
                    if spec.planning and spec.planning.mode == "reproduce_pelcb100":
                        candidate_payload["case_id"] = candidate.case_id
                    candidate_payload["emc_ready"] = True
            manifest.append(
                {
                    **candidate_payload,
                    "polymer_family": spec.polymer_family,
                    "architecture": spec.architecture,
                    "force_field": spec.force_field,
                    "temperature_k": spec.temperature_k,
                    "pressure_atm": spec.pressure_atm,
                    "chain_count": spec.chain_count,
                    "production_ns": spec.production_ns,
                }
            )
        if spec.rule_profile != self.validated_profile:
            in_domain = False
            warnings.append({"field": "rule_profile", "code": "rule_profile_missing_or_unvalidated"})
        for row in manifest:
            row["rule_profile"] = spec.rule_profile if in_domain and not errors else None
        raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return ValidationResult(
            valid=not errors,
            domain_status="validated_domain" if in_domain and not errors else "warn",
            errors=errors,
            warnings=warnings,
            manifest=manifest,
            manifest_sha256=sha256(raw).hexdigest(),
        )

    @staticmethod
    def manifest_csv(manifest: list[dict[str, Any]]) -> str:
        if not manifest:
            return ""
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
        return stream.getvalue()


def compare_release_with_frozen_oracle(cases: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_payload = json.loads(PELCB100_MANIFEST.read_text(encoding="utf-8"))
    expected_ids = [str(item["case_id"]) for item in manifest_payload["candidates"]]
    oracle = json.loads(PELCB100_RELEASE_ORACLE.read_text(encoding="utf-8"))
    actual_by_id = {str(item.get("case_id")): str(item.get("quality_status")) for item in cases}
    if set(actual_by_id) != set(expected_ids):
        return {
            "applicable": False,
            "reason": "release_case_ids_do_not_match_the_frozen_pelcb100_benchmark",
            "expected_case_count": 100,
            "actual_case_count": len(actual_by_id),
        }
    review_ids = set(oracle["review_case_ids"])
    expected = {case_id: "review" if case_id in review_ids else "pass" for case_id in expected_ids}
    mismatches = [
        {"case_id": case_id, "expected": expected[case_id], "actual": actual_by_id[case_id]}
        for case_id in expected_ids
        if actual_by_id[case_id] != expected[case_id]
    ]
    counts = {
        "pass": sum(value == "pass" for value in actual_by_id.values()),
        "review": sum(value == "review" for value in actual_by_id.values()),
    }
    return {
        "applicable": True,
        "exact_match": not mismatches and counts == oracle["expected_counts"],
        "expected_counts": oracle["expected_counts"],
        "actual_counts": counts,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "oracle_source_plan_sha256": oracle["source_plan_sha256"],
        "interpretation": "comparison_only_not_used_for_rule_decisions",
    }


def build_release_payload(
    *, project: dict[str, Any], run: dict[str, Any] | None, events: list[dict[str, Any]], reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest = project.get("manifest", [])
    return {
        "schema_version": "polymer_agent.release.v1",
        "project": project,
        "run": run,
        "cases": [
            {
                **case,
                "quality_status": "pending_run",
                "property_masks": {"density": False, "static": False, "diffusion_strict": False},
            }
            for case in manifest
        ],
        "event_ledger": events,
        "reviews": reviews,
    }


def _yes_no(value: Any) -> str:
    return "可用" if bool(value) else "不可用/需复核"


def _status_label(value: Any) -> str:
    mapping = {
        "pass": "可直接使用",
        "review": "需人工复核",
        "failed": "失败",
        "pending_run": "尚未完成计算",
        "publication_pending": "尚未达到发布级",
    }
    return mapping.get(str(value), str(value or "未知"))


def _fmt_value(value: Any, unit: str = "") -> str:
    if value is None or value == "":
        return "未报告"
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def build_readable_release_markdown(payload: dict[str, Any]) -> str:
    """Return a Chinese, experimenter-facing summary for a release payload."""
    project = payload.get("project") or {}
    run = payload.get("run") or {}
    cases = payload.get("cases") or []
    reviews = payload.get("reviews") or []
    counts: dict[str, int] = {}
    for case in cases:
        status = str(case.get("quality_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1

    title = project.get("name") or "PE-LCB 项目"
    lines = [
        f"# {title} 结果",
        "",
        "## 总览",
        "",
        f"- 发布状态：{_status_label(payload.get('publication_state') or run.get('state') or project.get('status'))}",
        f"- 设计数量：{len(cases)} 个 case",
        f"- 可直接使用：{counts.get('pass', 0)} 个",
        f"- 需人工复核：{counts.get('review', 0)} 个",
        f"- 失败：{counts.get('failed', 0)} 个",
        f"- manifest 哈希：`{project.get('manifest_sha256') or '未生成'}`",
        f"- 配置哈希：`{payload.get('config_sha256') or '未提供'}`",
        "",
        "## 如何使用这份结果",
        "",
        "- `可直接使用` 表示该 case 当前没有出版级警示，相关物性可按下方掩码使用。",
        "- `需人工复核` 表示至少一个关键物性仍有质量边界，例如扩散区未确认、规则配置缺失或审计分歧。",
        "- 密度、静态结构和严格扩散是按物性分别标记的；一个 case 的密度可用，并不代表扩散系数也可用。",
        "",
        "## 物性可用性汇总",
        "",
        "| case | 状态 | 密度 | 静态结构 | 严格扩散 | 说明 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        masks = case.get("property_masks") or {}
        notes = []
        if case.get("review_reason"):
            notes.append(str(case.get("review_reason")))
        if case.get("quality_status") == "review" and case.get("msd_warn"):
            notes.append("MSD 仍需复核")
        if case.get("repair_history"):
            notes.append(f"修复记录 {len(case.get('repair_history') or [])} 条")
        note = "; ".join(notes) or "无额外说明"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("case_id") or "-"),
                    _status_label(case.get("quality_status")),
                    _yes_no(masks.get("density")),
                    _yes_no(masks.get("static")),
                    _yes_no(masks.get("diffusion_strict")),
                    note.replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 每个 case 的主要数值", "", "| case | 密度 | Rg | 扩散系数 | 当前建议 |", "|---|---:|---:|---:|---|"])
    for case in cases:
        recommendation = "可进入建模/统计" if case.get("quality_status") == "pass" else "先查看复核包"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("case_id") or "-"),
                    _fmt_value(case.get("density_g_cm3") or case.get("target_density_g_cm3"), "g/cm3"),
                    _fmt_value(case.get("rg_nm") or case.get("rg_mean_nm"), "nm"),
                    _fmt_value(case.get("diffusion_cm2_s") or case.get("D_cm2_s"), "cm2/s"),
                    recommendation,
                ]
            )
            + " |"
        )

    if reviews:
        lines.extend(["", "## 复核项", ""])
        for item in reviews:
            lines.append(f"- {item.get('reason_code', 'review')}: {_status_label(item.get('status'))}")

    lines.extend(
        [
            "",
            "## 机器可读文件",
            "",
            "同一版本还提供 JSON 文件，用于脚本复现、机器学习训练和审计追踪。阅读版只改变呈现方式，不改变任何数值或质量状态。",
            "",
        ]
    )
    return "\n".join(lines)


def standard_pelcb100_candidates() -> list[dict[str, Any]]:
    payload = json.loads(PELCB100_MANIFEST.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if len(candidates) != 100:
        raise RuntimeError("PE-LCB100 design manifest must contain exactly 100 candidates")
    return candidates


def draft_spec_from_goal(goal: str) -> ProjectSpec:
    """Create the historical benchmark draft for replay and regression tests.

    The production ``/design-drafts`` endpoint uses live GPT-constrained design
    and never calls this fixed-manifest helper.
    """
    single_case = re.search(r"\b(single|one case|smoke)\b|单个|单例|小试", goal, re.IGNORECASE)
    if not single_case:
        return ProjectSpec(natural_language_goal=goal, candidates=standard_pelcb100_candidates())

    bp_match = re.search(r"(?:bp|branch[_ -]?points?|支化点数?|支化点)\s*[=:：]?\s*([1-5])", goal, re.IGNORECASE)
    bc_match = re.search(
        r"(?:bc|branch[_ -]?carbons?|支链碳数?|支链长度|C)\s*[=:：]?\s*(3|4|5|8|12|16|20)\b",
        goal,
        re.IGNORECASE,
    )
    distribution = next((item for item in VALID_DISTRIBUTIONS if re.search(rf"\b{item}\b", goal, re.IGNORECASE)), "mid")
    bp = int(bp_match.group(1)) if bp_match else 1
    bc = int(bc_match.group(1)) if bc_match else 5
    case_id = f"draft_bp{bp:02d}_bc{bc:03d}_{distribution}"
    return ProjectSpec(
        natural_language_goal=goal,
        candidates=[
            {
                "case_id": case_id,
                "branch_points": bp,
                "branch_carbons": bc,
                "branch_distribution_code": distribution,
                "target_total_carbons": 100,
            }
        ],
    )


def validate_remote_release(payload: dict[str, Any], *, project_domain_status: str) -> None:
    public_text = json.dumps(payload, ensure_ascii=False)
    forbidden_patterns = [
        r"[A-Za-z]:\\Users\\",
        r"/(?:home|public/home)/[A-Za-z0-9_.-]+/",
        r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}\b",
        r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
    ]
    if any(re.search(pattern, public_text, re.IGNORECASE) for pattern in forbidden_patterns):
        raise ValueError("remote release contains a private path, address or credential")
    if payload.get("schema_version") != "polymer_agent.remote_release.v1":
        raise ValueError("unexpected remote release schema")
    if payload.get("publication_state") not in {"pass", "review"}:
        raise ValueError("remote release lacks a terminal publication state")
    config_sha = str(payload.get("config_sha256") or "")
    if not re.fullmatch(r"[a-f0-9]{64}", config_sha):
        raise ValueError("remote release lacks a valid config hash")
    if not isinstance(payload.get("provenance"), list) or not payload["provenance"]:
        raise ValueError("remote release lacks provenance records")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("remote release lacks case records")
    for case in cases:
        if not isinstance(case, dict) or not case.get("case_id"):
            raise ValueError("remote release contains an invalid case record")
        if case.get("quality_status") not in {"pass", "review", "failed"}:
            raise ValueError("case lacks a valid quality status")
        masks = case.get("property_masks")
        if not isinstance(masks, dict) or not {"density", "static", "diffusion_strict"} <= set(masks):
            raise ValueError("case lacks required property-level masks")
        if not isinstance(case.get("audit_votes", []), list) or not isinstance(case.get("repair_history", []), list):
            raise ValueError("case audit or repair history has an invalid shape")
        if bool(case.get("msd_warn")) and float(case.get("msd_total_ns") or 0) >= 800 and case.get("quality_status") == "pass":
            raise ValueError("persistent MSD warning at 800 ns cannot be published as pass")
    if project_domain_status != "validated_domain" and payload.get("publication_state") == "pass":
        raise ValueError("out-of-domain project cannot be automatically published as pass")
