"""Optional live tri-model attribution (upgrade 4).

OFF BY DEFAULT. Re-runs DeepSeek / Qwen / GLM live on a case's real frozen
evidence packet and majority-aggregates a fresh diagnosis — the same panel and
the same evidence the frozen audit used, recomputed on demand.

Only activates when the user opts in with `POLYMER_AGENT_LIVE_ATTRIBUTION=1` and
keys for all three audit-panel providers are present. Otherwise the demo keeps
replaying the frozen votes. Read-only: this is adjudication, it mutates nothing.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from typing import Any

# OpenAI-compatible panel (same providers/keys as tri_llm_judge wrappers).
PANEL = [
    {
        "provider": "deepseek",
        "key_envs": ["DEEPSEEK_API_KEY"],
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "display": "deepseek (live)",
    },
    {
        "provider": "qwen",
        "key_envs": ["DASHSCOPE_API_KEY"],
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
        "display": "qwen (live)",
    },
    {
        "provider": "glm",
        "key_envs": ["ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"],
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.6",
        "display": "glm (live)",
    },
]

DIAGNOSIS_LABELS = {
    "engine_or_protocol_sensitive_result": "压力松弛与有效运行参数敏感",
    "fit_window_or_method_issue": "拟合窗口或后处理方法敏感",
    "missing_md_outputs": "MD 输出缺失",
    "cross_model_disagreement": "模型意见未收敛",
}

_SYSTEM_PROMPT = (
    "你是分子动力学 WARN 案例的归因专家。基于给定证据，从 diagnosis_options 里"
    "选择最可能的单一诊断。只输出一个 JSON 对象（不要解释、不要代码块）："
    '{"diagnosis": "<options 中的键>", "confidence": 0.0-1.0, "rationale": "中文一句话依据"}'
)


def _key_for(member: dict[str, Any]) -> str | None:
    for env in member["key_envs"]:
        value = os.environ.get(env)
        if value:
            return value
    return None


def _model_for(member: dict[str, Any]) -> str:
    for env in member.get("model_envs", []):
        value = os.environ.get(env)
        if value:
            return value
    return str(member["model"])


def available_panel() -> list[dict[str, Any]]:
    return [m for m in PANEL if _key_for(m)]


def attribution_enabled() -> bool:
    """True only when opted in AND the full DeepSeek/Qwen/GLM panel has keys."""
    flag = (os.environ.get("POLYMER_AGENT_LIVE_ATTRIBUTION") or "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    return len(available_panel()) == len(PANEL)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("model response contained no JSON object")
        return json.loads(match.group(0))


def _diagnosis_options(evidence_packet: Any) -> dict[str, str]:
    """Return the registered taxonomy without reading deterministic candidates."""
    return dict(DIAGNOSIS_LABELS)


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


def _sanitize_evidence_packet(value: Any) -> Any:
    """Remove deterministic diagnosis and route hints before prompting models."""
    if isinstance(value, dict):
        return {
            key: _sanitize_evidence_packet(child)
            for key, child in value.items()
            if key not in _LEAKY_EVIDENCE_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_evidence_packet(item) for item in value]
    return value


def _build_messages(case: dict[str, Any], options: dict[str, str]) -> list[dict[str, str]]:
    design = case.get("design", {}) or {}
    payload = {
        "case_id": case.get("case_id"),
        "warn_sources": case.get("rule", {}).get("warn_sources", []),
        "design": {
            k: design.get(k)
            for k in ("polymer_family", "branch_topology", "branch_points", "branch_carbons", "temperature_k")
        },
        "evidence": _sanitize_evidence_packet(case.get("evidence_packet")),
        "diagnosis_options": options,
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _call(member: dict[str, Any], messages: list[dict[str, str]]) -> str:
    from openai import OpenAI  # lazy import; only when live mode is on

    client = OpenAI(api_key=_key_for(member), base_url=member["base_url"], timeout=40, max_retries=0)
    response = client.chat.completions.create(model=_model_for(member), temperature=0, messages=messages)
    return response.choices[0].message.content or ""


def aggregate_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    """Majority-aggregate per-model votes (pure, unit-testable)."""
    tally = Counter(v["diagnosis"] for v in votes)
    top, top_n = tally.most_common(1)[0]
    if top_n >= 2:
        final = top
        agreement = f"{top_n}/{len(votes)} 一致"
        disagreement = top_n < len(votes)
        confidence = round(
            sum(v["confidence"] for v in votes if v["diagnosis"] == top) / top_n, 2
        )
    else:
        final = "cross_model_disagreement"
        agreement = "未收敛"
        disagreement = True
        confidence = round(sum(v["confidence"] for v in votes) / len(votes), 2)
    return {
        "final_diagnosis": final,
        "final_diagnosis_label": DIAGNOSIS_LABELS.get(final, final),
        "confidence": confidence,
        "agreement": agreement,
        "cross_model_disagreement": disagreement,
    }


def run_live_attribution(case: dict[str, Any]) -> dict[str, Any]:
    """Re-run the panel live on a case's evidence packet. Raises on hard failure."""
    panel = available_panel()
    if len(panel) != len(PANEL):
        raise RuntimeError("need all three audit-panel providers with API keys")

    options = _diagnosis_options(case.get("evidence_packet"))
    messages = _build_messages(case, options)

    votes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.time()
    for member in panel:
        try:
            data = _extract_json(_call(member, messages))
            diagnosis = data.get("diagnosis")
            try:
                confidence = round(min(max(float(data.get("confidence", 0.5)), 0.0), 1.0), 2)
            except (TypeError, ValueError):
                confidence = 0.5
            votes.append(
                {
                    "client": member["display"],
                    "diagnosis": diagnosis,
                    "diagnosis_label": options.get(diagnosis, DIAGNOSIS_LABELS.get(diagnosis, str(diagnosis))),
                    "confidence": confidence,
                    "rationale": str(data.get("rationale", ""))[:300],
                }
            )
        except Exception as exc:  # noqa: BLE001 - tolerate single-model failure
            errors.append({"client": member["display"], "error": f"{type(exc).__name__}: {exc}"})

    if len(votes) != len(PANEL):
        raise RuntimeError("one or more audit-panel models failed: " + json.dumps(errors, ensure_ascii=False))

    result = aggregate_votes(votes)
    result.update(
        {
            "engine": "live tri-model (" + "+".join(m["provider"] for m in panel) + ")",
            "model_panel": [m["display"] for m in panel],
            "votes": votes,
            "errors": errors,
            "latency_s": round(time.time() - started, 1),
        }
    )
    return result
