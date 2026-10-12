"""Offline auto-orchestration for the Polymer-Agent demo.

Turns a natural-language goal into:
- a structured intent (with a heuristic confidence), parsed fully offline;
- an ordered, replayable timeline over the 8-step closed loop;
- a "当前效果" before/after effect table;
- a "证据来源/验证" evidence block;
- an exportable Demo-说明 report (Markdown / JSON).

Everything is derived from the already-frozen demo snapshot. No online model
calls, no MD execution, no disk writes — the orchestrator is a pure function
layer so it stays deterministic and unit-testable.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

# --- offline intent vocabulary (no bare 2-letter tokens to avoid false hits) ---

POLYMER_FAMILY_TERMS = {
    "PEO": ["聚氧化乙烯", "聚环氧乙烷", "polyethylene oxide", "poly(ethylene oxide)", "peo"],
    "PVC": ["聚氯乙烯", "polyvinyl chloride", "pvc"],
    "PTFE": ["聚四氟乙烯", "polytetrafluoroethylene", "ptfe"],
    "PS": ["聚苯乙烯", "polystyrene"],
    "PA6": ["尼龙6", "尼龙 6", "nylon 6", "polyamide 6", "聚酰胺"],
    "PPS": ["聚苯硫醚", "polyphenylene sulfide", "pps"],
    "PDMS": ["聚二甲基硅氧烷", "polydimethylsiloxane", "pdms"],
    "PET": ["聚对苯二甲酸乙二醇酯", "polyethylene terephthalate", "pet"],
    "PMMA": ["聚甲基丙烯酸甲酯", "polymethyl methacrylate", "pmma"],
    "PP": ["聚丙烯", "丙烯", "polypropylene", "propylene", "pp-lcb", "pplcb"],
    "PE": ["聚乙烯", "polyethylene", "pe-lcb", "pelcb", "pe100"],
}
ARCHITECTURE_TERMS = {
    "LCB": ["支化", "支链", "长支链", "lcb", "branched", "branch", "long chain branch", "long-chain"],
    "linear": ["线性", "直链", "linear", "unbranched"],
    "cyclic": ["环状聚合物", "环形聚合物", "cyclic polymer", "ring polymer"],
    "star": ["星形聚合物", "星型聚合物", "star polymer"],
    "network": ["交联网络", "网络聚合物", "crosslinked network", "network polymer"],
}
OBSERVABLE_TERMS = {
    "density": ["密度", "density", "rho"],
    "dynamics": ["动力学", "扩散", "msd", "diffusion", "松弛", "relaxation", "mobility", "dynamics"],
    "structure": ["回转半径", "rg", "ree", "末端距", "形状", "shape", "尺寸"],
    "rheology": ["流变", "rheology", "粘度", "viscosity", "模量", "modulus"],
}
TASK_TYPE_TERMS = {
    "structure_property_trend": [
        "结构-性质", "结构—性质", "结构性质", "趋势", "trend", "规律",
        "可复用", "structure-property", "structure property",
    ],
    "compute": ["计算", "compute", "模拟", "simulate", "仿真", "跑", "run", "predict", "预测"],
}
CONDITION_TERMS = {
    "temperature": ["温度", "temperature", "400k", "400 k", "开尔文"],
    "chain_count": ["链数", "chains", "体系规模", "条链", "chain count"],
    "branch_length": ["支链长度", "branch length", "侧链长度", "c8", "c12", "c16", "c20"],
    "branch_point": ["支化点", "branch point", "支化密度", "branch density"],
}

# headline metric per final diagnosis, used by the effect table
HEADLINE_METRIC = {
    "engine_or_protocol_sensitive_result": ("density_g_cm3", "密度 (g·cm⁻³)"),
    "fit_window_or_method_issue": ("msd_loglog_slope", "MSD log–log slope"),
    "missing_md_outputs": ("pr_density_g_cm3", "PR 密度 (g·cm⁻³)"),
    "cross_model_disagreement": (None, "需复核"),
}

WORKFLOW_DESCRIPTIONS = {
    "自然语言目标": "把一句自然语言计算需求解析为结构化意图(离线,带置信度)。",
    "文献先验": "Zotero/RAG 与设计来源账本提供候选的文献先验与拓扑锚点。",
    "候选验证": "注册 adapter 将先验约束为可验证候选,validator 冻结设计哈希。",
    "规则判定": "generic QC 与 frozen rule profile 只输出 PASS/WARN,WARN 触发审计。",
    "三模型归因": "WARN 封装为冻结证据包,DeepSeek/Qwen/GLM 独立归因后 majority 收敛。",
    "MCP 路由": "MCP 仅暴露注册工具;deterministic guard 校验工具名、参数与执行边界。",
    "修复证据": "受限工具产出修复证据并写入 sidecar overlay,不改 canonical。",
    "重新判定": "overlay 重新进入 rule gate(re-gate),给出可追溯的结案状态。",
}


def _match(text: str, term_map: dict[str, list[str]]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for key, terms in term_map.items():
        matched = [t for t in terms if t in text]
        if matched:
            hits[key] = matched
    return hits


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "MISSING"
    return f"{float(value):.{digits}f}"


def _case_order(snapshot: dict[str, Any]) -> list[str]:
    return list(snapshot["cases"].keys())


# --------------------------------------------------------------------------- #
# 1. intent
# --------------------------------------------------------------------------- #

def parse_intent(goal: str) -> dict[str, Any]:
    """Offline parse of a natural-language goal into a structured intent."""
    text = (goal or "").lower()
    families = _match(text, POLYMER_FAMILY_TERMS)
    architectures = _match(text, ARCHITECTURE_TERMS)
    observables = _match(text, OBSERVABLE_TERMS)
    task_types = _match(text, TASK_TYPE_TERMS)
    conditions = _match(text, CONDITION_TERMS)

    matched_axes = sum(bool(x) for x in (families, architectures, observables, task_types))
    confidence = round(min(0.50 + 0.12 * matched_axes, 0.98), 2)

    family = next(iter(families), "unspecified")
    architecture = next(iter(architectures), "unspecified")
    evidence_scope_match = family == "PE" and architecture == "LCB"
    detected_terms = sorted(
        {
            term
            for group in (families, architectures, observables, task_types, conditions)
            for terms in group.values()
            for term in terms
        }
    )

    note: str | None = None
    if not text.strip():
        confidence = 0.40
        note = "未检测到具体意图,已按默认 PE-LCB100 冻结 cohort 回放。"
    elif family == "unspecified":
        confidence = min(confidence, 0.55)
        note = "未识别到具体聚合物族；PE-LCB100 仅作为界面与闭环流程对照回放。"
    elif not evidence_scope_match:
        note = (
            f"识别到请求材料 {family}-{architecture}，但冻结证据仅覆盖 PE-LCB；"
            "当前结果是对照回放，不构成该请求材料的计算证据。"
        )
    if matched_axes == 0:
        confidence = 0.45
        note = "输入与聚合物计算意图匹配度低,已映射到默认 PE-LCB100 冻结 cohort。"

    task_type = (
        "structure_property_trend"
        if "structure_property_trend" in task_types
        else ("compute" if task_types else "compute")
    )

    return {
        "goal": goal or "",
        "mapped_to_cohort": "pelcb100",
        "evidence_scope_match": evidence_scope_match,
        "evidence_scope": {
            "polymer_family": "PE",
            "architecture": "LCB",
            "cohort": "pelcb100",
        },
        "confidence": confidence,
        "structured": {
            "polymer_family": family,
            "polymer_family_detected": family,
            "architecture": architecture,
            "architecture_detected": bool(architectures),
            "observables": list(observables.keys()) or ["density", "dynamics"],
            "task_type": task_type,
            "condition_filters": list(conditions.keys()),
        },
        "detected_terms": detected_terms,
        "note": note,
        "engine": "offline",
    }


def resolve_intent(goal: str, *, allow_live: bool = True) -> dict[str, Any]:
    """Live LLM intent if explicitly opted in (upgrade 1), else offline parse.

    Falls back to the deterministic offline parser on any LLM error so the demo
    never breaks. Default boundary (no online model) is preserved unless the user
    sets POLYMER_AGENT_LIVE_INTENT=1 with a valid API key.
    """
    if not allow_live:
        return parse_intent(goal)
    from . import live_intent

    if live_intent.llm_enabled():
        try:
            return live_intent.parse_intent_llm(goal)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to offline
            intent = parse_intent(goal)
            intent["llm_error"] = f"{type(exc).__name__}: {exc}"
            return intent
    return parse_intent(goal)


# --------------------------------------------------------------------------- #
# 1b. corpus filtering (upgrade 3): intent actually selects from the 100-case cohort
# --------------------------------------------------------------------------- #

def build_corpus_view(snapshot: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    """Filter the full 100-case cohort by the parsed intent.

    This is the upgrade that makes the input genuinely change *which* real cases
    are shown, instead of always replaying the same three representatives.
    """
    corpus = snapshot.get("corpus", [])
    total = len(corpus)
    # Filter ONLY on the raw goal text. parse_intent defaults observables to
    # [density, dynamics] when nothing matched, so using observables here would
    # make every low-match input silently filter by density+msd.
    goal = (intent.get("goal") or "").lower()

    def has_warn(case: dict[str, Any], warn: str) -> bool:
        return warn in (case.get("warn_sources") or [])

    filters: list[tuple[str, Any]] = []
    if "密度" in goal or "density" in goal:
        filters.append(("密度 WARN", lambda c: has_warn(c, "density_warn")))
    if "msd" in goal or "动力学" in goal or "扩散" in goal or "diffusion" in goal or "dynamics" in goal:
        filters.append(("MSD WARN", lambda c: has_warn(c, "msd_warn")))
    if "缺失" in goal or "missing" in goal or "incomplete" in goal:
        filters.append(("证据缺失", lambda c: has_warn(c, "evidence_incomplete_warn")))
    if "分歧" in goal or "复核" in goal or "disagree" in goal or "review" in goal:
        filters.append(("模型分歧", lambda c: c.get("cross_model_disagreement")))
    if "拟合" in goal or "窗口" in goal or "fit" in goal or "window" in goal:
        filters.append(("拟合窗口归因", lambda c: c.get("final_diagnosis") == "fit_window_or_method_issue"))
    if "引擎" in goal or "压力松弛" in goal or "protocol" in goal or "engine" in goal:
        filters.append(
            ("引擎/协议归因", lambda c: c.get("final_diagnosis") == "engine_or_protocol_sensitive_result")
        )
    if "支化点多" in goal or "高支化" in goal or "多支" in goal or "支化多" in goal:
        filters.append(("支化点 ≥3", lambda c: (c.get("branch_points") or 0) >= 3))
    if "单支" in goal or "少支" in goal:
        filters.append(("单支化点", lambda c: c.get("branch_points") == 1))

    if filters:
        filter_label = " + ".join(label for label, _ in filters)
        matched = [c for c in corpus if all(pred(c) for _, pred in filters)]
    else:
        filter_label = "全部 WARN 案例（无显式筛选）"
        matched = list(corpus)
    if intent.get("evidence_scope_match") is False:
        filter_label = f"PE-LCB100 对照回放；{filter_label}"

    rep_ids = list(snapshot.get("cases", {}).keys())
    matched_ids = {c["case_id"] for c in matched}
    matched_representative_ids = [rid for rid in rep_ids if rid in matched_ids]

    return {
        "filter_label": filter_label,
        "filtered": bool(filters),
        "evidence_scope_match": bool(intent.get("evidence_scope_match")),
        "matched_count": len(matched),
        "total": total,
        "by_diagnosis": dict(Counter(c.get("final_diagnosis") for c in matched)),
        "matched_representative_ids": matched_representative_ids,
        "cases": matched[:60],
    }


def _deep_case_ids(snapshot: dict[str, Any], corpus_view: dict[str, Any] | None) -> list[str]:
    rep_ids = list(snapshot["cases"].keys())
    matched = (corpus_view or {}).get("matched_representative_ids")
    return matched if matched else rep_ids


# --------------------------------------------------------------------------- #
# 2. timeline
# --------------------------------------------------------------------------- #

def _case_brief(case: dict[str, Any]) -> dict[str, Any]:
    return {"case_id": case["case_id"], "short_id": case["short_id"], "title": case["title"]}


def _attrib_payload(case: dict[str, Any]) -> dict[str, Any]:
    audit = case["audit"]
    return {
        **_case_brief(case),
        "final_diagnosis": audit["final_diagnosis"],
        "final_diagnosis_label": audit["final_diagnosis_label"],
        "confidence": audit["confidence"],
        "selected_by": audit["selected_by"],
        "model_votes": [
            {
                "client": v["client"],
                "diagnosis_label": v["diagnosis_label"],
                "confidence": v["confidence"],
            }
            for v in audit["model_votes"]
        ],
    }


def _route_payload(
    case: dict[str, Any],
    route_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Upgrade 2: in connected mode auto_run injects live read-only MCP routes;
    # otherwise we fall back to the frozen snapshot route.
    override = (route_overrides or {}).get(case["case_id"])
    route = override or case["route"]
    return {
        **_case_brief(case),
        "module": route["module"],
        "tool": route["tool"],
        "parameters": route["parameters"],
        "guard": route["guard"]["verdict"],
        "requires_for_execution": route["guard"]["requires_for_execution"],
        "source": "live read-only MCP plan" if override else "hashed frozen replay",
    }


def _repair_payload(case: dict[str, Any]) -> dict[str, Any]:
    diag = case["audit"]["final_diagnosis"]
    key, label = HEADLINE_METRIC.get(diag, (None, "需复核"))
    metric = case["metrics"].get(key) if key else None
    return {
        **_case_brief(case),
        "metric_label": label,
        "before": _fmt_float(metric.get("before")) if metric else "—",
        "after": _fmt_float(metric.get("after")) if metric else "—",
        "source": metric.get("source") if metric else None,
    }


def _regate_payload(case: dict[str, Any]) -> dict[str, Any]:
    outcome = case["outcome"]
    return {
        **_case_brief(case),
        "re_gate_status": outcome["re_gate_status"],
        "strict_rule_pass": outcome["strict_rule_pass"],
        "label": outcome["label"],
    }


def build_timeline(
    snapshot: dict[str, Any],
    intent: dict[str, Any],
    route_overrides: dict[str, Any] | None = None,
    corpus_view: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Ordered events for the 8-step closed loop, streamed by the frontend.

    `route_overrides` (case_id -> public route) lets connected mode replace the
    frozen snapshot routes with live read-only MCP plans (upgrade 2).
    `corpus_view` selects which representative cases are deep-dived based on the
    intent filter (upgrade 3).
    """
    overview = snapshot["overview"]
    cohort = overview["cohort"]
    workflow = overview["workflow"]
    cases = [snapshot["cases"][cid] for cid in _deep_case_ids(snapshot, corpus_view)]

    events: list[dict[str, Any]] = []

    def stage(step_index: int, kind: str, title: str, narration: str, payload: dict[str, Any]) -> None:
        events.append(
            {
                "step_index": step_index,
                "step": workflow[step_index - 1],
                "kind": kind,
                "title": title,
                "narration": narration,
                "payload": payload,
            }
        )

    def case_event(step_index: int, title: str, payload: dict[str, Any]) -> None:
        events.append(
            {
                "step_index": step_index,
                "step": workflow[step_index - 1],
                "kind": "case",
                "title": title,
                "narration": "",
                "payload": payload,
            }
        )

    s = intent["structured"]
    stage(
        1,
        "stage",
        "自然语言目标 → 结构化意图",
        f"已解析意图(置信度 {intent['confidence']}):{s['polymer_family']}-{s['architecture']} · "
        f"观测量 {', '.join(s['observables'])} · 任务 {s['task_type']}。",
        {"goal": intent["goal"], "intent": intent},
    )

    source_ids = sorted(
        {sid for case in cases for sid in case["design"].get("source_ids", [])}
    )
    stage(
        2,
        "stage",
        "文献先验",
        "从 Zotero/RAG 与设计来源账本载入候选先验与拓扑锚点。",
        {"source_ids": source_ids, "anchor": cases[0]["design"].get("source_anchor")},
    )

    stage(
        3,
        "stage",
        "候选验证",
        "注册 adapter 约束先验为可验证候选,validator 冻结设计哈希。",
        {
            "validator_status": "pass",
            "frozen_designs": len(cases),
            "human_scope": cases[0]["design"].get("human_scope"),
        },
    )

    cv = corpus_view or {}
    if cv.get("filtered"):
        step4_narration = (
            f"frozen rule profile 在 {cohort['total_cases']} 体系上判定;"
            f"按意图筛选「{cv['filter_label']}」→ 命中 "
            f"{cv['matched_count']}/{cohort['audited_cases']} 个审计案例。"
        )
        if not cv.get("matched_representative_ids"):
            step4_narration += " 无 deep 代表案例命中，下方展示全部代表案例。"
    else:
        step4_narration = (
            f"frozen rule profile 在 {cohort['total_cases']} 体系上判定,"
            f"{cohort['audited_cases']} 个 case 进入三模型审计（未显式筛选）。"
        )
    stage(
        4,
        "stage",
        "规则判定(PASS/WARN)",
        step4_narration,
        {
            "total_cases": cohort["total_cases"],
            "audited_cases": cohort["audited_cases"],
            "repair_domain_distribution": cohort.get("repair_domain_distribution", {}),
            "matched_count": cv.get("matched_count", cohort["audited_cases"]),
            "filter_label": cv.get("filter_label"),
        },
    )

    stage(
        5,
        "stage",
        "三模型归因",
        "WARN 封装冻结证据,DeepSeek/Qwen/GLM 独立归因后 majority 收敛。",
        {"strategy": "tri-model-majority", "model_calls": cohort["model_calls"]},
    )
    for case in cases:
        case_event(5, f"{case['short_id']} 归因", _attrib_payload(case))

    route_source = "live read-only MCP plan" if route_overrides else "hashed frozen replay"
    stage(
        6,
        "stage",
        "MCP 路由(dry-run)",
        f"MCP 只暴露注册工具;deterministic guard 校验工具名、参数与执行边界。路由来源:{route_source}。",
        {"guard": "pass", "dry_run": True, "route_source": route_source},
    )
    for case in cases:
        case_event(6, f"{case['short_id']} 路由", _route_payload(case, route_overrides))

    stage(
        7,
        "stage",
        "修复证据(sidecar overlay)",
        "受限工具产出修复证据并写入 sidecar,不修改 canonical。",
        {"writes_canonical": False},
    )
    for case in cases:
        case_event(7, f"{case['short_id']} 修复前后", _repair_payload(case))

    stage(
        8,
        "stage",
        "重新判定(re-gate)",
        f"overlay 重新进入 rule gate;最终发布状态为 "
        f"{cohort['publication_pass']} PASS / {cohort['publication_review']} review。",
        {
            "publication_pass": cohort["publication_pass"],
            "publication_review": cohort["publication_review"],
            "strict_msd_overlay_resolved": cohort["strict_msd_overlay_resolved"],
            "ml_usable_is_strict_pass": cohort["ml_usable_is_strict_pass"],
        },
    )
    for case in cases:
        case_event(8, f"{case['short_id']} 结案", _regate_payload(case))

    events.append(
        {
            "step_index": 8,
            "step": "结案报告",
            "kind": "report_ready",
            "title": "结案报告已生成",
            "narration": "可导出按一~七结构组织的 Demo 说明报告(.md / .json)。",
            "payload": {"formats": ["md", "json"]},
        }
    )
    return events


# --------------------------------------------------------------------------- #
# 3. effect table + evidence block
# --------------------------------------------------------------------------- #

def build_effect_table(
    snapshot: dict[str, Any],
    representative_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    cohort = snapshot["overview"]["cohort"]

    rows: list[dict[str, Any]] = [
        {
            "metric": "最终发布状态",
            "before": "修复前 WARN 队列",
            "after": f"{cohort['publication_pass']} PASS / {cohort['publication_review']} review",
            "delta": "有界闭环终局",
            "highlight": True,
        },
        {
            "metric": "WARN 案例归因覆盖",
            "before": "—",
            "after": f"{cohort['audited_cases']} / {cohort['audited_cases']}",
            "delta": f"{cohort['model_calls']} 次真实模型调用",
            "highlight": False,
        },
        {
            "metric": "密度 / MSD / 缺失输出路线",
            "before": "—",
            "after": "77 / 41 / 6",
            "delta": "重新判定或 review",
            "highlight": False,
        },
    ]

    for cid in (representative_ids or _case_order(snapshot)):
        case = snapshot["cases"].get(cid)
        if not case:
            continue
        diag = case["audit"]["final_diagnosis"]
        key, label = HEADLINE_METRIC.get(diag, (None, "需复核"))
        metric = case["metrics"].get(key) if key else None
        if not metric:
            continue
        resolved = case["outcome"]["strict_rule_pass"]
        rows.append(
            {
                "metric": f"{case['short_id']} · {label}",
                "before": _fmt_float(metric.get("before")) + (
                    " (WARN)" if metric.get("before") is not None else " (MISSING)"
                ),
                "after": _fmt_float(metric.get("after")),
                "delta": "resolved 🎯" if resolved else "recovered · closure pending",
                "highlight": bool(resolved),
            }
        )
    return rows


def build_evidence_block(snapshot: dict[str, Any]) -> dict[str, Any]:
    cohort = snapshot["overview"]["cohort"]
    cases = snapshot["cases"]
    model_panel = next(
        (c["provenance"]["model_panel"] for c in cases.values() if c.get("provenance")),
        ["deepseek-v4-pro", "qwen3-max", "glm-4.6"],
    )
    validation = (
        f"全量 {cohort['total_cases']} 体系;三模型"
        f"({'/'.join(model_panel)})完成 {cohort['model_calls']} 次投票;"
        f"{cohort['audited_cases']} 个审计案例均保留完整的三模型面板;"
        f"最终发布为 {cohort['publication_pass']} PASS / {cohort['publication_review']} review。"
    )
    return {
        "cohort_scale": {
            "total_cases": cohort["total_cases"],
            "warn_audited": cohort["audited_cases"],
            "model_calls": cohort["model_calls"],
            "publication_pass": cohort["publication_pass"],
            "publication_review": cohort["publication_review"],
            "complete_three_model_panels": cohort["complete_three_model_panels"],
            "diagnosis_distribution": cohort["diagnosis_distribution"],
            "selected_route_distribution": cohort["selected_route_distribution"],
            "strict_msd_overlay_resolved": cohort["strict_msd_overlay_resolved"],
            "clean_runs_scanned": cohort["clean_runs_scanned"],
            "clean_full_postprocess_ready": cohort["clean_full_postprocess_ready"],
        },
        "model_panel": model_panel,
        "validation": validation,
        "sources": [
            {
                "source_id": src["source_id"],
                "public_name": src["public_name"],
                "role": src["role"],
                "sha256": src["sha256"],
            }
            for src in snapshot["source_manifest"]
        ],
    }


# --------------------------------------------------------------------------- #
# 4. report (Demo-说明 一~七)
# --------------------------------------------------------------------------- #

def _live_aware_boundaries(boundaries: list[str], intent: dict[str, Any]) -> list[str]:
    """Keep the exported boundaries honest when live intent was used."""
    engine = intent.get("engine", "offline")
    if isinstance(engine, str) and engine.startswith("llm"):
        kept = [b for b in boundaries if "不调用在线模型" not in b]
        return [f"意图解析调用了在线模型（{engine}）；闭环下游 MD 仍不实时执行。", *kept]
    return list(boundaries)


def _report_data(
    snapshot: dict[str, Any],
    intent: dict[str, Any],
    mode: str,
    corpus_view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overview = snapshot["overview"]
    deep_ids = _deep_case_ids(snapshot, corpus_view)
    cases_out = []
    for cid in deep_ids:
        case = snapshot["cases"].get(cid)
        if not case:
            continue
        design = case["design"]
        audit = case["audit"]
        route = case["route"]
        cases_out.append(
            {
                "short_id": case["short_id"],
                "title": case["title"],
                "design": {
                    "human_scope": design.get("human_scope"),
                    "topology": design.get("branch_topology"),
                    "branch_points": design.get("branch_points"),
                    "branch_carbons": design.get("branch_carbons"),
                    "branch_position_pattern": design.get("branch_position_pattern"),
                    "chain_count": design.get("chain_count"),
                    "temperature_k": design.get("temperature_k"),
                    "pressure_atm": design.get("pressure_atm"),
                },
                "rule": {"verdict": case["rule"]["verdict"], "warn_sources": case["rule"]["warn_sources"]},
                "attribution": {
                    "final_diagnosis_label": audit["final_diagnosis_label"],
                    "confidence": audit["confidence"],
                    "selected_by": audit["selected_by"],
                    "model_votes": [
                        {"client": v["client"], "diagnosis_label": v["diagnosis_label"], "confidence": v["confidence"]}
                        for v in audit["model_votes"]
                    ],
                },
                "route": {
                    "module": route["module"],
                    "tool": route["tool"],
                    "parameters": route["parameters"],
                    "guard": route["guard"]["verdict"],
                    "requires_for_execution": route["guard"]["requires_for_execution"],
                },
                "metrics": case["metrics"],
                "outcome": case["outcome"],
            }
        )

    return {
        "schema_version": "polymer_agent_demo.report.v1",
        "data_version": snapshot["data_version"],
        "generated_at_utc": snapshot["generated_at_utc"],
        "mode": mode,
        "title": "Polymer-Agent 闭环 Demo — 自动编排结案报告",
        "input_goal": intent["goal"],
        "intent": intent,
        "capabilities": [{"step": step, "desc": WORKFLOW_DESCRIPTIONS[step]} for step in overview["workflow"]],
        "guarded_route_tools": sorted(
            {
                case["route"]["tool"]
                for case in snapshot["cases"].values()
                if case.get("route", {}).get("tool")
            }
        ),
        "cases": cases_out,
        "principles": [
            "固定主链 EMC→GROMACS→后处理:确定性正向链路,不被 agent 改写。",
            "generic QC + frozen rule profile:只输出 PASS/WARN,WARN 触发审计而非结论。",
            "WARN 封装冻结证据包,三模型独立归因后 majority 收敛。",
            "MCP 只暴露注册工具;deterministic guard 校验工具名、参数与执行边界。",
            "修复证据写入 sidecar overlay,重新进入 rule gate(re-gate),canonical 不变。",
        ],
        "boundaries": _live_aware_boundaries(overview["boundaries"], intent),
        "corpus": {
            "filter_label": (corpus_view or {}).get(
                "filter_label", "全部 WARN 案例（无显式筛选）"
            ),
            "filtered": bool((corpus_view or {}).get("filtered")),
            "matched_count": (corpus_view or {}).get(
                "matched_count", overview["cohort"]["audited_cases"]
            ),
            "total": (corpus_view or {}).get(
                "total", overview["cohort"]["audited_cases"]
            ),
            "by_diagnosis": (corpus_view or {}).get("by_diagnosis", {}),
        },
        "effect_table": build_effect_table(snapshot, deep_ids),
        "evidence": build_evidence_block(snapshot),
        "next_steps": [
            "扩展到新的 polymer adapter 与 online-built rule profile。",
            "增加视觉模型可读的诊断图。",
            "执行权限仍由 MCP schema 与 deterministic guard 控制,不放开物理参数自由控制。",
        ],
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _render_markdown(data: dict[str, Any]) -> str:
    s = data["intent"]["structured"]
    lines: list[str] = []
    lines.append(f"# {data['title']}")
    lines.append("")
    lines.append(f"- 数据版本: `{data['data_version']}`")
    lines.append(f"- 生成时间(UTC): {data['generated_at_utc']}")
    lines.append(f"- 运行模式: {data['mode']}")
    lines.append("")

    lines.append("## 一、任务目标与输入意图")
    lines.append("")
    lines.append(f"- 自然语言输入: {data['input_goal'] or '(空)'}")
    engine = data["intent"].get("engine", "offline")
    lines.append(
        f"- 解析意图(引擎 {engine},置信度 {data['intent']['confidence']}): "
        f"{s['polymer_family']}-{s['architecture']} · 观测量 {', '.join(s['observables'])} · "
        f"任务 {s['task_type']} · 条件 {', '.join(s['condition_filters']) or '默认'}"
    )
    scope_state = "范围内证据" if data["intent"].get("evidence_scope_match") else "跨材料对照回放"
    lines.append(
        f"- 证据 cohort: PE-LCB100({data['evidence']['cohort_scale']['total_cases']} 体系) · {scope_state}"
    )
    if data["intent"].get("note"):
        lines.append(f"- 说明: {data['intent']['note']}")
    lines.append("")

    lines.append("## 二、已完成的核心能力(闭环八步)")
    lines.append("")
    for i, cap in enumerate(data["capabilities"], start=1):
        lines.append(f"{i}. **{cap['step']}** — {cap['desc']}")
    lines.append(
        "- 当前受控修复工具: "
        + ", ".join(f"`{tool}`" for tool in data["guarded_route_tools"])
    )
    lines.append("")

    lines.append("## 三、Demo 输出示例(逐案例)")
    lines.append("")
    corpus = data.get("corpus", {})
    lines.append(
        f"- 本次按意图筛选「{corpus.get('filter_label')}」命中 "
        f"{corpus.get('matched_count')}/{corpus.get('total')} 例 WARN 案例；以下深入展示其中的代表案例。"
    )
    if corpus.get("by_diagnosis"):
        dist = "，".join(f"{k}={v}" for k, v in corpus["by_diagnosis"].items())
        lines.append(f"- 命中归因分布：{dist}")
    lines.append("")
    for case in data["cases"]:
        d = case["design"]
        a = case["attribution"]
        r = case["route"]
        lines.append(f"### {case['short_id']} · {case['title']}")
        lines.append(
            f"- 设计: {d['human_scope']} · {d['topology']} · 支化点 {d['branch_points']} · "
            f"支链 C{d['branch_carbons']} · 位置 {d['branch_position_pattern']} · "
            f"{d['chain_count']} chains · {d['temperature_k']}K/{d['pressure_atm']}atm"
        )
        lines.append(f"- 规则判定: {case['rule']['verdict']}（{', '.join(case['rule']['warn_sources'])}）")
        votes = "；".join(f"{v['client']} {v['diagnosis_label']}({v['confidence']})" for v in a["model_votes"])
        lines.append(
            f"- 三模型归因: 最终 **{a['final_diagnosis_label']}**"
            f"(confidence {a['confidence']}, {a['selected_by']}) — {votes}"
        )
        lines.append(
            f"- MCP 路由: `{r['module']} → {r['tool']}` · 参数 `{json.dumps(r['parameters'], ensure_ascii=False)}` · "
            f"guard {r['guard']} · 执行需 {', '.join(r['requires_for_execution'])}"
        )
        for key, metric in case["metrics"].items():
            lines.append(
                f"- 修复证据 [{key}]: {_fmt_float(metric.get('before'))} → "
                f"{_fmt_float(metric.get('after'))} ({metric.get('source')})"
            )
        lines.append(
            f"- 重新判定: {case['outcome']['re_gate_status']} · "
            f"strict_rule_pass={case['outcome']['strict_rule_pass']} · {case['outcome']['label']}"
        )
        lines.append("")

    lines.append("## 四、核心原理")
    lines.append("")
    for item in data["principles"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 五、当前边界")
    lines.append("")
    for item in data["boundaries"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 六、价值与验证结论")
    lines.append("")
    lines.append("**当前效果**")
    lines.append("")
    lines.append(
        _md_table(
            ["指标", "修复前", "修复后", "变化"],
            [[r["metric"], r["before"], r["after"], r["delta"]] for r in data["effect_table"]],
        )
    )
    lines.append("")
    ev = data["evidence"]
    lines.append(f"**验证**: {ev['validation']}")
    lines.append("")
    lines.append("**证据来源(可校验哈希)**")
    lines.append("")
    lines.append(
        _md_table(
            ["source_id", "public_name", "role", "sha256"],
            [[x["source_id"], x["public_name"], x["role"], x["sha256"]] for x in ev["sources"]],
        )
    )
    lines.append("")

    lines.append("## 七、下一步可演进方向")
    lines.append("")
    for item in data["next_steps"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_report(
    snapshot: dict[str, Any],
    intent: dict[str, Any],
    fmt: str,
    mode: str = "replay",
    corpus_view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return {content: bytes, media_type, filename} for an exportable report."""
    if fmt not in ("md", "json"):
        raise ValueError(f"unsupported report format: {fmt!r} (use 'md' or 'json')")

    data = _report_data(snapshot, intent, mode, corpus_view)
    stem = f"polymer_agent_demo_report_{data['data_version']}"
    if fmt == "json":
        content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        return {"content": content, "media_type": "application/json; charset=utf-8", "filename": f"{stem}.json"}

    content = _render_markdown(data).encode("utf-8")
    return {"content": content, "media_type": "text/markdown; charset=utf-8", "filename": f"{stem}.md"}
