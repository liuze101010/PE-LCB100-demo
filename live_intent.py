"""Optional live LLM intent parsing (upgrade 1).

OFF BY DEFAULT. The demo's stated boundary is "no online model"; this module
only activates when the user explicitly opts in with `POLYMER_AGENT_LIVE_INTENT=1`
AND an API key for the selected provider is present. Otherwise the orchestrator
falls back to the deterministic offline keyword parser in `auto_orchestrator`.

It reuses the same OpenAI-compatible providers / env keys as the project's
`tri_llm_judge` clients, but stays self-contained (no heavy imports) so the demo
can still run standalone in replay mode.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# OpenAI-compatible providers (same endpoints/keys as tri_llm_judge wrappers).
PROVIDERS = {
    "openai": {
        "key_envs": ["OPENAI_API_KEY"],
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.5",
    },
    "deepseek": {
        "key_envs": ["DEEPSEEK_API_KEY"],
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "qwen": {
        "key_envs": ["DASHSCOPE_API_KEY"],
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
    },
    "glm": {
        "key_envs": ["ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"],
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.6",
    },
}

KNOWN_OBSERVABLES = ("density", "dynamics", "structure", "rheology")
KNOWN_TASK_TYPES = ("compute", "structure_property_trend")
KNOWN_CONDITIONS = ("temperature", "chain_count", "branch_length", "branch_point")
KNOWN_ARCHITECTURES = ("linear", "LCB", "cyclic", "star", "network", "unspecified")

_SYSTEM_PROMPT = (
    "你是聚合物计算需求的意图解析器。读取用户的一句自然语言目标，"
    "只输出一个 JSON 对象（不要解释、不要代码块），字段如下：\n"
    '{"polymer_family_detected": "PE|PP|PEO|PVC|PTFE|PS|PA6|PPS|PDMS|PET|PMMA|other", '
    '"architecture_detected": "linear|LCB|cyclic|star|network|unspecified", '
    '"observables": ["density"|"dynamics"|"structure"|"rheology"], '
    '"task_type": "compute|structure_property_trend", '
    '"condition_filters": ["temperature"|"chain_count"|"branch_length"|"branch_point"], '
    '"confidence": 0.0-1.0, "note": "可选的中文说明"}'
)


def _provider_name() -> str:
    return (os.environ.get("POLYMER_AGENT_INTENT_PROVIDER") or "deepseek").strip().lower()


def _provider_config() -> dict[str, Any]:
    return PROVIDERS.get(_provider_name(), PROVIDERS["deepseek"])


def _api_key() -> str | None:
    for env in _provider_config()["key_envs"]:
        value = os.environ.get(env)
        if value:
            return value
    return None


def llm_enabled() -> bool:
    """True only when explicitly opted in AND a key is available."""
    flag = (os.environ.get("POLYMER_AGENT_LIVE_INTENT") or "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    return _api_key() is not None


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("LLM response contained no JSON object")
        return json.loads(match.group(0))


def _clean_list(value: Any, allowed: tuple[str, ...]) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if item in allowed]


def _normalize(goal: str, raw: dict[str, Any], engine: str) -> dict[str, Any]:
    family_detected = str(raw.get("polymer_family_detected") or "unspecified").upper()
    architecture_raw = raw.get("architecture_detected", "unspecified")
    if isinstance(architecture_raw, bool):
        architecture = "LCB" if architecture_raw else "unspecified"
    else:
        architecture = str(architecture_raw or "unspecified")
        if architecture.lower() == "lcb":
            architecture = "LCB"
        elif architecture not in KNOWN_ARCHITECTURES:
            architecture = "unspecified"
    evidence_scope_match = family_detected == "PE" and architecture == "LCB"
    observables = _clean_list(raw.get("observables"), KNOWN_OBSERVABLES) or ["density", "dynamics"]
    task_type = raw.get("task_type")
    if task_type not in KNOWN_TASK_TYPES:
        task_type = "compute"
    conditions = _clean_list(raw.get("condition_filters"), KNOWN_CONDITIONS)

    try:
        confidence = round(min(max(float(raw.get("confidence", 0.5)), 0.0), 0.99), 2)
    except (TypeError, ValueError):
        confidence = 0.5

    note = raw.get("note") or None
    if not evidence_scope_match:
        mapping = (
            f"识别到请求材料 {family_detected}-{architecture}，但冻结证据仅覆盖 PE-LCB；"
            "当前结果是对照回放，不构成该请求材料的计算证据。"
        )
        note = f"{note}；{mapping}" if note else mapping

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
            "polymer_family": family_detected,
            "polymer_family_detected": family_detected,
            "architecture": architecture,
            "architecture_detected": architecture != "unspecified",
            "observables": observables,
            "task_type": task_type,
            "condition_filters": conditions,
        },
        "detected_terms": [],
        "note": note,
        "engine": engine,
    }


def parse_intent_llm(goal: str) -> dict[str, Any]:
    """Call the configured OpenAI-compatible model to parse intent. Raises on failure."""
    config = _provider_config()
    key = _api_key()
    if not key:
        raise RuntimeError("no API key for live intent provider")
    model = os.environ.get("POLYMER_AGENT_INTENT_MODEL") or config["model"]

    from openai import OpenAI  # lazy import; only needed when live mode is on

    client = OpenAI(api_key=key, base_url=config["base_url"], timeout=30, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": goal or ""},
        ],
    )
    text = response.choices[0].message.content or ""
    raw = _extract_json(text)
    return _normalize(goal, raw, engine=f"llm:{_provider_name()}:{model}")
