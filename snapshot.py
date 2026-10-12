from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_IDS = (
    "lcb_pe_006_bp01_bc008_pdend_c100",
    "lcb_pe_002_bp01_bc008_pdmid_c100",
    "lcb_pe_068_bp04_bc012_pduni_c100",
)

PACKAGE_ROOT = Path(__file__).resolve().parent
TERMINAL_OUTCOMES_PATH = PACKAGE_ROOT / "data" / "terminal_release_outcomes.json"

SOURCE_PATHS = {
    "tri_model_audit": Path(
        "generated_code/md_analysis/output/pelcb100_warn_audit_diagnosis/"
        "final_tri_majority_glm_completed_20260602/warn_audit_diagnosis_merged.jsonl"
    ),
    "tri_model_summary": Path(
        "generated_code/md_analysis/output/pelcb100_warn_audit_diagnosis/"
        "final_tri_majority_glm_completed_20260602/warn_audit_merged_summary.json"
    ),
    "msd_overlay": Path(
        "generated_code/md_analysis/output/pelcb100_integrated/aggregate/overlays/"
        "repair_loop_execute_msd_window_20260602_msd_window_overlay/"
        "msd_window_overlay_changes.csv"
    ),
    "msd_overlay_summary": Path(
        "generated_code/md_analysis/output/pelcb100_integrated/aggregate/overlays/"
        "repair_loop_execute_msd_window_20260602_msd_window_overlay/"
        "msd_window_overlay_summary.json"
    ),
    "clean_density": Path(
        "generated_code/md_analysis/output/pelcb100_supervisor/"
        "o7_clean_cuda2024_closure/o7_clean_density_summary.csv"
    ),
    "clean_status": Path(
        "generated_code/md_analysis/output/pelcb100_supervisor/"
        "o7_clean_cuda2024_closure/o7_clean_run_status.csv"
    ),
    "clean_summary": Path(
        "generated_code/md_analysis/output/pelcb100_supervisor/"
        "o7_clean_cuda2024_closure/o7_clean_closure_summary.json"
    ),
    "mcp_server": Path("generated_code/md_analysis/mcp/pelcb_mcp_server.py"),
}

CASE_DISPLAY = {
    "lcb_pe_006_bp01_bc008_pdend_c100": {
        "short_id": "PE006",
        "title": "密度 WARN：回跳 MD",
        "accent": "orange",
        "cause_label": "压力松弛与有效运行参数敏感",
        "route_label": "使用 allowlisted clean MD profile，回跳 PR/NVT",
        "outcome_label": "密度进入冻结规则区间，MD 全阶段完成",
    },
    "lcb_pe_002_bp01_bc008_pdmid_c100": {
        "short_id": "PE002",
        "title": "MSD WARN：仅修后处理",
        "accent": "teal",
        "cause_label": "MSD 拟合窗口选择敏感",
        "route_label": "扫描允许的 MSD 窗口，不重跑 MD",
        "outcome_label": "严格窗口候选通过，sidecar overlay 已重新判定",
    },
    "lcb_pe_068_bp04_bc012_pduni_c100": {
        "short_id": "PE068",
        "title": "证据缺失：恢复 MD",
        "accent": "red",
        "cause_label": "原始 MD 输出缺失",
        "route_label": "从归档输入恢复 clean MD，再进入后处理",
        "outcome_label": "缺失输出已恢复，完整后处理与规则复查通过",
    },
}

DIAGNOSIS_LABELS = {
    "engine_or_protocol_sensitive_result": "压力松弛与有效运行参数敏感",
    "fit_window_or_method_issue": "拟合窗口或后处理方法敏感",
    "missing_md_outputs": "MD 输出缺失",
    "cross_model_disagreement": "模型意见未收敛",
    "final_publication_pass": "最终无当前出版级 WARN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _load_mcp_module(project_root: Path):
    path = project_root / SOURCE_PATHS["mcp_server"]
    spec = importlib.util.spec_from_file_location("polymer_agent_demo_mcp_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load MCP source: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _index_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in rows if row.get("case_id")}


CASE_ID_RE = re.compile(r"lcb_pe_(\d+)_bp(\d+)_bc(\d+)_pd([a-z0-9-]+)_c(\d+)")


def _short_id(case_id: str) -> str:
    match = CASE_ID_RE.match(case_id)
    return f"PE{int(match.group(1)):03d}" if match else case_id


def _design_from_case_id(case_id: str) -> dict[str, Any]:
    match = CASE_ID_RE.match(case_id)
    if not match:
        return {
            "branch_points": None,
            "branch_carbons": None,
            "branch_position": None,
            "carbon_count": None,
        }
    return {
        "branch_points": int(match.group(2)),
        "branch_carbons": int(match.group(3)),
        "branch_position": match.group(4),
        "carbon_count": int(match.group(5)),
    }


def _build_corpus(
    audit_rows: list[dict[str, Any]],
    representative_ids: set[str],
    terminal_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Lightweight per-case records for the full 100-case cohort (filterable)."""
    corpus: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in audit_rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        seen.add(case_id)
        diagnosis = row.get("primary_audit_diagnosis") or row.get("final_diagnosis")
        terminal = terminal_index.get(case_id, {})
        corpus.append(
            {
                "case_id": case_id,
                "short_id": _short_id(case_id),
                **_design_from_case_id(case_id),
                "warn_sources": row.get("warn_sources") or [],
                "final_diagnosis": diagnosis,
                "final_diagnosis_label": DIAGNOSIS_LABELS.get(diagnosis, "需要审查"),
                "confidence": row.get("confidence"),
                "module": row.get("recommended_reentry_module"),
                "repair_tool": row.get("repair_tool"),
                "ml_usable": _as_bool(row.get("ml_usable")),
                "cross_model_disagreement": _as_bool(row.get("cross_model_disagreement")),
                "is_representative": case_id in representative_ids,
                "terminal_verdict": terminal.get("terminal_verdict"),
                "terminal_gate_pass": terminal.get("terminal_gate_pass"),
                "terminal_release_state": terminal.get("terminal_release_state"),
            }
        )
    manifest_path = PACKAGE_ROOT / "data" / "pelcb100_design_manifest.json"
    if manifest_path.exists():
        for candidate in json.loads(manifest_path.read_text(encoding="utf-8")).get("candidates", []):
            case_id = str(candidate.get("case_id") or "")
            if not case_id or case_id in seen:
                continue
            terminal = terminal_index.get(case_id, {})
            corpus.append(
                {
                    "case_id": case_id,
                    "short_id": _short_id(case_id),
                    **_design_from_case_id(case_id),
                    "warn_sources": [],
                    "final_diagnosis": "final_publication_pass",
                    "final_diagnosis_label": DIAGNOSIS_LABELS["final_publication_pass"],
                    "confidence": None,
                    "module": None,
                    "repair_tool": None,
                    "ml_usable": True,
                    "cross_model_disagreement": False,
                    "is_representative": case_id in representative_ids,
                    "terminal_verdict": terminal.get("terminal_verdict"),
                    "terminal_gate_pass": terminal.get("terminal_gate_pass"),
                    "terminal_release_state": terminal.get("terminal_release_state"),
                }
            )
    corpus.sort(key=lambda row: row["case_id"])
    return corpus


def _density_by_case_and_stage(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["case_id"], row["stage"]): row
        for row in rows
        if row.get("case_id") and row.get("stage")
    }


def _public_vote(vote: dict[str, Any]) -> dict[str, Any]:
    return {
        "client": vote.get("client", "unknown"),
        "diagnosis": vote.get("primary_audit_diagnosis")
        or vote.get("cause_diagnosis"),
        "diagnosis_label": DIAGNOSIS_LABELS.get(
            vote.get("primary_audit_diagnosis") or vote.get("cause_diagnosis"),
            "需要审查",
        ),
        "confidence": vote.get("confidence"),
        "evidence_basis": vote.get("diagnosis_basis", []),
        "rationale": vote.get("rationale", ""),
        "repair_tool": vote.get("repair_tool"),
        "tool_parameters": vote.get("tool_parameters", {}),
    }


def _public_audit_record(row: dict[str, Any]) -> dict[str, Any]:
    diagnosis = row.get("primary_audit_diagnosis") or row.get("final_diagnosis")
    return {
        "case_id": row["case_id"],
        "warn_sources": list(row.get("warn_sources") or []),
        "strategy": row.get("strategy", "tri-model-majority"),
        "model_votes": [_public_vote(vote) for vote in row.get("model_votes") or []],
        "primary_audit_diagnosis": diagnosis,
        "primary_audit_diagnosis_label": DIAGNOSIS_LABELS.get(
            diagnosis, "需要审查"
        ),
        "confidence": row.get("confidence"),
        "selected_by": row.get("selected_by"),
        "evidence_sufficiency": row.get("evidence_sufficiency"),
        "recommended_reentry_module": row.get("recommended_reentry_module"),
        "repair_tool": row.get("repair_tool"),
        "tool_parameters": dict(row.get("tool_parameters") or {}),
        "cache_key": row.get("cache_key"),
    }


_LEAKY_EVIDENCE_KEYS = {
    "available_repair_tools",
    "repair_affordances",
    "primary_audit_diagnosis",
    "audit_diagnosis",
    "diagnosis_candidates",
    "repair_instruction",
    "recommended_reentry_module",
    "recommended_action",
    "repair_route",
    "repair_tool",
    "tool_parameters",
    "module",
    "diagnosis",
}


def _public_evidence_packet(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_evidence_packet(child)
            for key, child in value.items()
            if key not in _LEAKY_EVIDENCE_KEYS
        }
    if isinstance(value, list):
        return [_public_evidence_packet(item) for item in value]
    return value


def _public_design(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "human_scope": record.get("human_scope"),
        "polymer_family": record.get("polymer_family"),
        "architecture": record.get("architecture"),
        "design_variable": record.get("design_variable"),
        "branch_points": record.get("branch_points"),
        "branch_carbons": record.get("branch_carbons"),
        "branch_position_pattern": record.get("branch_position_pattern"),
        "branch_topology": record.get("branch_topology"),
        "carbon_count": record.get("actual_total_carbons_per_chain"),
        "chain_count": record.get("chain_count"),
        "temperature_k": record.get("temperature_k"),
        "pressure_atm": record.get("pressure_atm"),
        "source_anchor": record.get("source_anchor"),
        "source_ids": record.get("mapped_source_ids", []),
        "design_rationale": record.get("design_rationale"),
        "validator_status": record.get("validator_status"),
        "manifest_row_hash": record.get("manifest_row_hash"),
        "case_json_hash": record.get("case_json_hash"),
        "builder_spec_hash": record.get("builder_spec_hash"),
        "canonical_manifest_modified": False,
    }


def public_tool_parameters(
    case_id: str,
    route: dict[str, Any],
) -> dict[str, Any]:
    machine_parameters = route.get("tool_parameters", {})
    repair_tool = str(route.get("repair_tool") or "")
    public_keys = {
        "repair_kind",
        "window_strategy",
        "engine_profile",
        "reentry_stage",
        "rerun_scope",
        "input_source",
        "review_kind",
        "expected_success_condition",
    }
    return {
        key: value for key, value in machine_parameters.items() if key in public_keys
    }


def _public_allowed_route(option: dict[str, Any]) -> dict[str, Any]:
    route = {
        "case_id": "",
        "repair_tool": option.get("repair_tool"),
        "tool_parameters": dict(option.get("tool_parameters") or {}),
    }
    return {
        "diagnosis": option.get("diagnosis"),
        "module": option.get("recommended_reentry_module"),
        "tool": option.get("repair_tool"),
        "parameters": public_tool_parameters("", route),
        "machine_route": option.get("repair_route"),
    }


def _select_two_stage_route(mcp: Any, case_id: str, audit: dict[str, Any]) -> dict[str, Any]:
    options = mcp.plan_repair_route_impl(case_id)
    if options.get("status") != "ok" or options.get("planning_selection_status") != "awaiting_planning_model":
        raise RuntimeError(f"{case_id}: MCP did not return route options")
    desired_tool = str(audit.get("repair_tool") or "")
    allowed = {
        str(item.get("repair_tool")): item for item in options.get("allowed_routes") or []
    }
    if desired_tool not in allowed:
        raise RuntimeError(
            f"{case_id}: locked audit tool {desired_tool!r} is not in MCP allowed routes"
        )
    proposal = {
        "repair_tool": desired_tool,
        "parameters": dict(allowed[desired_tool].get("tool_parameters") or {}),
        "rationale": "Select the registered route recorded by the locked tri-model audit.",
    }
    selected = mcp.plan_repair_route_impl(
        case_id, planning_proposal_json=json.dumps(proposal, ensure_ascii=False)
    )
    if selected.get("status") != "ok" or selected.get("planning_selection_status") != "selected":
        raise RuntimeError(
            f"{case_id}: MCP rejected locked audit route: "
            f"{selected.get('planning_rejection_reason', 'unknown reason')}"
        )
    return selected


def _public_route(route: dict[str, Any]) -> dict[str, Any]:
    module = route.get("recommended_reentry_module")
    case_id = route["case_id"]
    return {
        "dry_run": True,
        "module": module,
        "tool": route.get("repair_tool"),
        "parameters": public_tool_parameters(case_id, route),
        "machine_parameters": route.get("tool_parameters", {}),
        "action": route.get("recommended_action"),
        "machine_route": route.get("repair_route"),
        "allowed_routes": [
            _public_allowed_route(option) for option in route.get("allowed_routes") or []
        ],
        "planning_selection_status": route.get("planning_selection_status"),
        "route_selected_by": route.get("route_selected_by"),
        "planning_selection_rationale": route.get("planning_selection_rationale"),
        "planning_guard": route.get("planning_guard", {}),
        "guard": {
            "verdict": "pass",
            "read_only_demo": True,
            "requires_for_execution": route.get("execution_guard", {}).get(
                "mutating_tools_require",
                ["execute=true", "project_id", "reason", "allowlist_tag"],
            ),
            "canonical_release_modified": False,
        },
    }


def _case_record(
    case_id: str,
    mcp: Any,
    audit: dict[str, Any],
    msd: dict[str, str] | None,
    density_rows: dict[tuple[str, str], dict[str, str]],
    clean_status: dict[str, str],
    terminal: dict[str, Any],
) -> dict[str, Any]:
    summary = mcp.get_case_summary_impl(case_id)
    design = mcp.get_design_origin_impl(case_id)["record"]
    route = _select_two_stage_route(mcp, case_id, audit)
    quality = summary["quality_mask"]
    labels = CASE_DISPLAY[case_id]

    record: dict[str, Any] = {
        "case_id": case_id,
        **labels,
        "goal": "计算受控支化聚乙烯体系，并识别可复用的结构—性质趋势。",
        "design": _public_design(design),
        "rule": {
            "verdict": "WARN",
            "warn_sources": audit.get("warn_sources", []),
            "machine_status": quality.get("case_status"),
            "density_tier": quality.get("density_tier_si_o"),
            "warning_reason": quality.get("warning_reason"),
        },
        "audit": {
            "strategy": audit.get("strategy", "tri-model-majority"),
            "model_votes": [_public_vote(vote) for vote in audit["model_votes"]],
            "final_diagnosis": audit.get("primary_audit_diagnosis"),
            "final_diagnosis_label": labels["cause_label"],
            "confidence": audit.get("confidence"),
            "selected_by": audit.get("selected_by"),
            "cross_model_disagreement": audit.get(
                "cross_model_disagreement", False
            ),
            "evidence_sufficiency": audit.get("evidence_sufficiency"),
        },
        "route": {
            **_public_route(route),
            "display_label": labels["route_label"],
        },
        "evidence": [
            {
                "type": "design",
                "label": "候选设计与 validator freeze",
                "source_id": "mcp_design_origin",
                "hash": design.get("manifest_row_hash"),
            },
            {
                "type": "audit",
                "label": "三模型冻结审计记录",
                "source_id": "tri_model_audit",
                "hash": audit.get("cache_key"),
            },
        ],
        "evidence_packet": _public_evidence_packet(audit.get("warn_evidence_packets")),
        "metrics": {},
        "outcome": {},
        "terminal": terminal,
        "provenance": {
            "model_panel": [
                vote.get("client", "unknown") for vote in audit["model_votes"]
            ],
            "source_records": [
                "tri_model_audit",
                "clean_density",
                "clean_status",
                "msd_overlay" if msd else None,
                "terminal_outcomes",
            ],
            "canonical_release_modified": False,
        },
    }
    record["provenance"]["source_records"] = [
        item for item in record["provenance"]["source_records"] if item
    ]

    all_md_done = all(
        _as_bool(clean_status.get(f"{stage}_done"))
        for stage in ("steep", "rlx", "pr", "nvt_prod")
    )
    record["provenance"]["clean_md_all_stages_done"] = all_md_done

    if case_id == "lcb_pe_006_bp01_bc008_pdend_c100":
        clean_pr = density_rows[(case_id, "pr")]
        record["metrics"] = {
            "density_g_cm3": {
                "before": quality.get("rho_final_gcc_si_o"),
                "after": terminal.get("density_g_cm3")
                or _as_float(clean_pr["density_last10ns_gcc"]),
                "unit": "g cm^-3",
                "source": "same-input clean-control PR tail",
            }
        }
        record["outcome"] = {
            "status": "resolved",
            "re_gate_status": "resolved_by_clean_density_overlay",
            "strict_rule_pass": terminal.get("terminal_gate_pass") is True,
            "label": labels["outcome_label"],
        }
    elif case_id == "lcb_pe_002_bp01_bc008_pdmid_c100":
        if msd is None:
            raise RuntimeError("PE002 MSD overlay record is missing")
        clean_pr = density_rows[(case_id, "pr")]
        record["metrics"] = {
            "density_g_cm3": {
                "before": quality.get("rho_final_gcc_si_o"),
                "after": _as_float(clean_pr["density_last10ns_gcc"]),
                "unit": "g cm^-3",
                "source": "same-input clean-control PR tail",
            },
            "msd_loglog_slope": {
                "before": _as_float(msd["v1_loglog_slope"]),
                "overlay_candidate": _as_float(msd["overlay_loglog_slope"]),
                "after": terminal.get("msd_loglog_slope"),
                "fit_window_ns": msd["fit_window_ns"],
                "local_d_cv_pct": _as_float(msd["local_D_cv_pct"]),
                "source": "strict sidecar MSD-window overlay",
            },
        }
        record["outcome"] = {
            "status": "resolved",
            "re_gate_status": "resolved_by_postprocess_overlay",
            "strict_rule_pass": terminal.get("terminal_gate_pass") is True,
            "label": labels["outcome_label"],
        }
        record["evidence"].append(
            {
                "type": "repair_overlay",
                "label": "MSD strict-window sidecar overlay",
                "source_id": "msd_overlay",
                "hash": None,
            }
        )
    else:
        clean_pr = density_rows[(case_id, "pr")]
        record["metrics"] = {
            "pr_density_g_cm3": {
                "before": None,
                "after": terminal.get("density_g_cm3")
                or _as_float(clean_pr["density_last10ns_gcc"]),
                "unit": "g cm^-3",
                "source": "recovered clean MD PR tail",
            },
            "msd_loglog_slope": {
                "before": None,
                "after": terminal.get("msd_loglog_slope"),
                "source": "terminal clean postprocess and rule re-gate",
            },
        }
        record["outcome"] = {
            "status": "resolved",
            "re_gate_status": "resolved_after_missing_output_recovery",
            "strict_rule_pass": terminal.get("terminal_gate_pass") is True,
            "label": labels["outcome_label"],
        }

    return record


def build_demo_snapshot(
    project_root: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_path = Path(output_path)
    paths = {name: project_root / relative for name, relative in SOURCE_PATHS.items()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        bundled_dir = PACKAGE_ROOT / "data"
        bundled_snapshot = bundled_dir / "demo_snapshot.json"
        bundled_manifest = bundled_dir / "source_manifest.json"
        if not bundled_snapshot.exists() or not bundled_manifest.exists():
            raise FileNotFoundError(f"Missing demo evidence sources: {', '.join(missing)}")

        # Standalone delivery packages intentionally carry the frozen, hashed
        # replay artifact rather than the private production evidence roots.
        snapshot = json.loads(bundled_snapshot.read_text(encoding="utf-8"))
        manifest = json.loads(bundled_manifest.read_text(encoding="utf-8"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_path.with_name("source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return snapshot

    audit_rows = _load_jsonl(paths["tri_model_audit"])
    audit_index = _index_by_case(audit_rows)
    audit_summary = _load_json(paths["tri_model_summary"])
    msd_index = _index_by_case(_load_csv(paths["msd_overlay"]))
    msd_summary = _load_json(paths["msd_overlay_summary"])
    density_index = _density_by_case_and_stage(_load_csv(paths["clean_density"]))
    clean_status_index = _index_by_case(_load_csv(paths["clean_status"]))
    clean_summary = _load_json(paths["clean_summary"])
    mcp = _load_mcp_module(project_root)
    terminal_payload = _load_json(TERMINAL_OUTCOMES_PATH)
    terminal_index = _index_by_case(terminal_payload["cases"])

    model_call_count = sum(len(row.get("model_votes") or []) for row in audit_rows)
    complete_panel_count = sum(
        len(row.get("model_votes") or []) == 3 for row in audit_rows
    )
    if len(audit_rows) != 97 or len(audit_index) != 97:
        raise RuntimeError("locked audit ledger must contain 97 unique cases")
    if model_call_count != 291 or complete_panel_count != 97:
        raise RuntimeError("locked audit ledger must contain 291 votes and 97 complete panels")
    if terminal_payload.get("summary") != {"case_count": 100, "pass": 92, "review": 8}:
        raise RuntimeError("terminal outcome ledger must contain 92 PASS and 8 REVIEW cases")

    cases = {
        case_id: _case_record(
            case_id=case_id,
            mcp=mcp,
            audit=audit_index[case_id],
            msd=msd_index.get(case_id),
            density_rows=density_index,
            clean_status=clean_status_index[case_id],
            terminal=terminal_index[case_id],
        )
        for case_id in CASE_IDS
    }

    diagnosis_counts = Counter(
        row.get("primary_audit_diagnosis") for row in audit_rows
    )
    route_counts = Counter(
        row.get("recommended_reentry_module") for row in audit_rows
    )
    source_roles = {
        "tri_model_audit": "真实三模型逐案例投票与最终归因",
        "tri_model_summary": "全量 WARN 审计汇总",
        "msd_overlay": "MSD 窗口修复 sidecar 明细",
        "msd_overlay_summary": "MSD overlay 与 ML-usable 汇总",
        "clean_density": "clean-control 密度证据",
        "clean_status": "clean MD 阶段完成状态",
        "clean_summary": "clean closure 运行汇总",
        "mcp_server": "只读查询与 dry-run 路由实现",
    }
    source_manifest = [
        {
            "source_id": source_id,
            "public_name": path.name,
            "role": source_roles[source_id],
            "sha256": _sha256(path),
        }
        for source_id, path in paths.items()
    ]
    source_manifest.append(
        {
            "source_id": "terminal_outcomes",
            "public_name": TERMINAL_OUTCOMES_PATH.name,
            "role": "100 个案例的当前终态、结果门禁与最终指标",
            "sha256": _sha256(TERMINAL_OUTCOMES_PATH),
        }
    )

    audit_ledger = [_public_audit_record(row) for row in audit_rows]

    snapshot = {
        "schema_version": "polymer_agent_demo.v2",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "data_version": "pelcb100_closed_loop_final_20260718",
        "overview": {
            "title": "Polymer-Agent 闭环证据 Demo",
            "goal": "把自然语言聚合物目标转化为可追溯、可诊断、可局部修复的分子动力学工作流。",
            "value": "WARN 不再只是失败标签，而是进入证据归因、受限工具路由和重新判定的触发器。",
            "workflow": [
                "自然语言目标",
                "文献先验",
                "候选验证",
                "规则判定",
                "三模型归因",
                "MCP 路由",
                "修复证据",
                "重新判定",
            ],
            "cohort": {
                "total_cases": 100,
                "audited_cases": len(audit_rows),
                "model_calls": model_call_count,
                "complete_three_model_panels": complete_panel_count,
                "diagnosis_distribution": dict(sorted(diagnosis_counts.items())),
                "selected_route_distribution": dict(sorted(route_counts.items())),
                "repair_domain_distribution": {
                    "density_clean_regate": 77,
                    "msd_strict_gate_or_review": 41,
                    "missing_output_backfill": 6,
                },
                "density_msd_overlap": 31,
                "publication_pass": 92,
                "publication_review": 8,
                "ml_usable_before": 53,
                "ml_usable_after": msd_summary["ml_usable_count"],
                "ml_usable_is_strict_pass": False,
                "strict_msd_overlay_resolved": msd_summary[
                    "strict_overlay_applied_count"
                ],
                "clean_runs_scanned": clean_summary["runs"],
                "clean_full_postprocess_ready": clean_summary[
                    "full_postprocess_ready"
                ],
            },
            "boundaries": [
                "本 Demo 不调用在线模型。",
                "本 Demo 不提交远程 MD。",
                "所有 MCP 路由均为 dry-run。",
                "ML-usable 不等于严格规则 PASS。",
                "canonical 数据未被修改。",
            ],
        },
        "cases": cases,
        "corpus": _build_corpus(audit_rows, set(CASE_IDS), terminal_index),
        "audit_ledger": audit_ledger,
        "source_manifest": source_manifest,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_path.with_name("source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "polymer_agent_demo.source_manifest.v1",
                "data_version": snapshot["data_version"],
                "generated_at_utc": snapshot["generated_at_utc"],
                "sources": source_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return snapshot
