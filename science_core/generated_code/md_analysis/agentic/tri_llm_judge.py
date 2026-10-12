"""Tri-LLM cross-model adjudication for MD analysis audit decisions.

This module is intentionally data-independent: callers provide a case directory
containing JSON evidence, and LLM clients are injected. The default code path
does not require LangChain or any specific vendor SDK.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:
    from .schemas.judgment import Judgment, judgment_schema_json, validate_judgment
except ImportError:  # pragma: no cover - direct script execution fallback
    from schemas.judgment import Judgment, judgment_schema_json, validate_judgment  # type: ignore


MODULE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = MODULE_DIR / "prompts"
DEFAULT_CACHE_DIR = MODULE_DIR / "cache"
SCHEMA_VERSION = "tri_llm_judge.v1"
DECISION_TYPES = {"equilibration", "fit_quality", "anomaly"}
MODES = {"blind", "audit"}
VERDICT_SCORE = {"pass": 0, "warn": 1, "fail": 2}
SCORE_VERDICT = {value: key for key, value in VERDICT_SCORE.items()}
LOCKED_PAPER_MODEL_NAMES = ("deepseek-v3-plus", "claude-sonnet-4-6", "glm-4.6")
DEEPSEEK_MODEL = "deepseek-chat"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
GLM_MODEL = "glm-4.6"
QWEN_MODEL = "qwen3.7-max"
THERMAL_DELTA_WARN_K = 5.0
THERMAL_DELTA_FAIL_K = 10.0
THERMAL_SLOPE_WARN_K_PER_SAMPLE = 0.02
THERMAL_SLOPE_FAIL_K_PER_SAMPLE = 0.05
ENERGY_NORM_SLOPE_WARN = 2.0e-4
ENERGY_NORM_SLOPE_FAIL = 1.0e-3
VOLUME_NORM_SLOPE_WARN = 2.0e-4
VOLUME_NORM_SLOPE_FAIL = 1.0e-3
# Public-data density screening defaults used only when no activated profile is supplied.
# These thresholds are deliberately not included in blind evidence bundles.
COMPACTION_RHO_PASS_LOW = 0.73
COMPACTION_RHO_PASS_HIGH = 0.87
MSD_SLOPE_LOW = 0.85
MSD_SLOPE_HIGH = 1.05
MSD_MIN_WINDOW_FRAMES = 10
MSD_WINDOW_SENSITIVITY_WARN = 0.15
VOID_LOW_FRACTION_WARN = 0.03
VOID_LOW_FRACTION_FAIL = 0.08
VOID_COMPONENT_FRACTION_WARN = 0.01
VOID_COMPONENT_FRACTION_FAIL = 0.03
AMBIGUITY_FLAG_SPLIT_RE = re.compile(r"[;,/|]+|\s+\+\s+|\s+and\s+", flags=re.IGNORECASE)


class LLMClient(Protocol):
    """Minimal interface required by judge()."""

    name: str

    def generate(self, prompt: str) -> str | dict[str, Any]:
        """Return a JSON string or dict matching Judgment."""


def _openai_compatible_content(response: Any, provider: str) -> str:
    """Return final answer text without treating hidden reasoning as the answer.

    Some reasoning-capable endpoints can consume the full completion budget
    before emitting final JSON.  The metadata below makes that failure
    auditable while deliberately excluding reasoning text and response text.
    """

    choices = list(getattr(response, "choices", None) or [])
    if not choices:
        raise RuntimeError(f"{provider} returned no choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content:
        return str(content)

    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
    finish_reason = getattr(choice, "finish_reason", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
    raise RuntimeError(
        f"{provider} returned empty content "
        f"(finish_reason={finish_reason!r}, completion_tokens={completion_tokens!r}, "
        f"reasoning_tokens={reasoning_tokens!r})"
    )


@dataclass
class StaticJSONClient:
    """Small deterministic client used by smoke tests and local dry-runs."""

    name: str
    payload: dict[str, Any]
    calls: int = 0
    fail_first: bool = False

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError(f"{self.name} transient failure")
        return json.dumps(self.payload)

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


@dataclass
class DeepSeekClient:
    """DeepSeek V3+ wrapper using the OpenAI-compatible SDK."""

    api_key: str | None = None
    model: str = DEEPSEEK_MODEL
    name: str = "deepseek-v3-plus"
    timeout_s: float = 60.0
    max_tokens: int = 800
    calls: int = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("Install openai>=1.0 to use DeepSeekClient") from exc
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=self.timeout_s, max_retries=0)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict scientific audit model. Output exactly one JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=self.max_tokens,
        )
        return _openai_compatible_content(response, "DeepSeek")

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


@dataclass
class AnthropicClient:
    """Anthropic Claude wrapper.

    Claude does not expose a native JSON-mode switch in the Messages API, so
    the prompt is hardened and the shared post-hoc JSON parser validates output.
    """

    api_key: str | None = None
    model: str = ANTHROPIC_MODEL
    name: str = "claude-sonnet-4-6"
    timeout_s: float = 60.0
    max_tokens: int = 800
    calls: int = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("Install anthropic>=0.40 to use AnthropicClient") from exc
        client = Anthropic(api_key=key, timeout=self.timeout_s, max_retries=0)
        hardened_prompt = (
            "Output exactly one JSON object, no markdown fence, no prose outside JSON.\n\n"
            f"{prompt}"
        )
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": hardened_prompt}],
        )
        parts: list[str] = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        text = "\n".join(parts).strip()
        if not text:
            raise RuntimeError("Anthropic returned empty content")
        return text

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


@dataclass
class GLMClient:
    """Zhipu GLM wrapper using the OpenAI-compatible endpoint."""

    api_key: str | None = None
    model: str = GLM_MODEL
    name: str = "glm-4.6"
    timeout_s: float = 60.0
    max_tokens: int = 800
    calls: int = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        key = self.api_key or os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
        if not key:
            raise RuntimeError("ZHIPUAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("Install openai>=1.0 to use GLMClient") from exc
        client = OpenAI(
            api_key=key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            timeout=self.timeout_s,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict scientific audit model. Output exactly one JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return _openai_compatible_content(response, "GLM")

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


@dataclass
class QwenClient:
    """Qwen wrapper through DashScope's OpenAI-compatible endpoint."""

    api_key: str | None = None
    model: str = QWEN_MODEL
    name: str | None = None
    timeout_s: float = 60.0
    max_tokens: int = 800
    calls: int = 0

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.model

    def generate(self, prompt: str) -> str:
        self.calls += 1
        key = self.api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("Install openai>=1.0 to use QwenClient") from exc
        client = OpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=self.timeout_s,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict scientific audit model. Output exactly one JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return _openai_compatible_content(response, "Qwen")

    def complete(self, prompt: str) -> str:
        return self.generate(prompt)


def judge(
    case_dir: str | Path,
    decision_type: str,
    mode: str = "blind",
    *,
    clients: list[LLMClient] | tuple[LLMClient, ...] | None = None,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
    retries: int = 2,
) -> dict[str, Any]:
    """Adjudicate one decision for one case and return a strict JSON verdict.

    Parameters
    ----------
    case_dir:
        Directory containing per-case JSON files such as ``analysis.json`` and
        ``trajectory/trajectory_analysis.json``.
    decision_type:
        One of ``equilibration``, ``fit_quality``, or ``anomaly``.
    mode:
        ``blind`` uses evidence only; ``audit`` also includes the rule verdict.
    clients:
        Exactly three LLM clients. Tests should pass mock clients here. Real API
        clients can be wrapped to expose ``name`` and ``generate(prompt)``.
    cache_dir:
        Cache location. Defaults to ``agentic/cache``.
    force_refresh:
        If true, bypass existing cache and call clients again.
    retries:
        Number of retries per model after a failed call or invalid JSON.
    """

    decision_type = _validate_choice("decision_type", decision_type, DECISION_TYPES)
    mode = _validate_choice("mode", mode, MODES)
    case_path = Path(case_dir)
    evidence = collect_case_evidence(case_path, decision_type)
    rule_verdict = compute_rule_verdict(evidence, decision_type)
    prompt = build_prompt(evidence, rule_verdict, decision_type, mode)

    if clients is None:
        clients = _default_clients()
    if len(clients) != 3:
        raise ValueError(f"tri-LLM adjudication requires exactly 3 clients, got {len(clients)}")

    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = make_cache_key(
        evidence=evidence,
        rule_verdict=rule_verdict,
        decision_type=decision_type,
        mode=mode,
        prompt=prompt,
        client_names=[client.name for client in clients],
    )
    cache_path = cache_root / f"{cache_key}.json"
    if cache_path.exists() and not force_refresh:
        payload = _read_json(cache_path)
        cached = validate_judgment(payload["judgment"])
        return _dump_model(cached)

    individual = []
    for client in clients:
        judgment = _call_client_with_retry(client, prompt, mode=mode, rule_verdict=rule_verdict, retries=retries)
        individual.append({"client": client.name, "judgment": _dump_model(judgment)})

    final = aggregate_judgments(
        [validate_judgment(item["judgment"]) for item in individual],
        mode=mode,
        rule_verdict=rule_verdict,
    )
    cache_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "case_id": evidence.get("case_id"),
        "decision_type": decision_type,
        "mode": mode,
        "cache_key": cache_key,
        "client_names": [client.name for client in clients],
        "rule_verdict": rule_verdict,
        "judgment": _dump_model(final),
        "model_votes": individual,
    }
    cache_path.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return _dump_model(final)


def collect_case_evidence(case_dir: Path, decision_type: str) -> dict[str, Any]:
    """Read and compact per-case evidence for the six rule classes."""

    analysis_path = case_dir / "analysis.json"
    trajectory_paths = [
        case_dir / "trajectory" / "trajectory_analysis.json",
        case_dir / "trajectory_analysis.json",
    ]
    analysis = _read_json(analysis_path) if analysis_path.exists() else {}
    trajectory_path = next((path for path in trajectory_paths if path.exists()), None)
    trajectory = _read_json(trajectory_path) if trajectory_path else {}

    case_id = str(analysis.get("case_id") or trajectory.get("case_id") or case_dir.name)
    scalars = analysis.get("scalars", {}) if isinstance(analysis, dict) else {}
    warnings = analysis.get("warnings", []) if isinstance(analysis, dict) else []
    trajectory_results = trajectory.get("results", {}) if isinstance(trajectory, dict) else {}
    rule_config = _load_rule_config(case_dir, analysis)

    evidence = {
        "case_id": case_id,
        "decision_type": decision_type,
        "files": {
            "analysis_json": str(analysis_path) if analysis_path.exists() else None,
            "trajectory_analysis_json": str(trajectory_path) if trajectory_path else None,
        },
        "thermal": _extract_thermal_metrics(analysis, scalars, trajectory_results),
        "energetic": _extract_energetic_metrics(analysis, scalars, trajectory_results),
        "compaction": _extract_compaction_metrics(analysis, scalars, trajectory_results, rule_config),
        "void": _extract_void_metrics(case_dir, analysis, trajectory_results),
        "fit_quality": _extract_fit_quality_metrics(trajectory, trajectory_results),
        "runtime": _extract_runtime_metrics(case_dir, analysis),
        "warnings": warnings,
        "available_result_keys": sorted(str(key) for key in trajectory_results.keys())
        if isinstance(trajectory_results, dict)
        else [],
        "missing_core_files": [
            label
            for label, found in {
                "analysis.json": analysis_path.exists(),
                "trajectory_analysis.json": bool(trajectory_path),
            }.items()
            if not found
        ],
    }
    return _json_safe(evidence)


def compute_rule_verdict(evidence: dict[str, Any], decision_type: str) -> dict[str, Any]:
    """Compute deterministic rule verdicts mapped to Methods section 5."""

    sub_rules = {
        "thermal": _thermal_rule(evidence.get("thermal", {}) or {}),
        "energetic": _energetic_rule(evidence.get("energetic", {}) or {}),
        "compaction": _compaction_rule(evidence.get("compaction", {}) or {}),
        "void": _void_rule(evidence.get("void", {}) or {}),
        "fit_quality": _fit_quality_rule(evidence.get("fit_quality", {}) or {}),
        "runtime": _runtime_rule(evidence.get("runtime", {}) or {}),
    }
    if decision_type == "equilibration":
        consulted = ["thermal", "energetic", "compaction"]
        verdict = _all_pass_any_fail(sub_rules, consulted)
        reason = "thermal + energetic + compaction aggregation"
    elif decision_type == "fit_quality":
        consulted = ["fit_quality"]
        verdict = sub_rules["fit_quality"]["verdict"]
        reason = "direct MSD log-log slope rule"
    elif decision_type == "anomaly":
        consulted = ["runtime", "void"]
        verdict = _any_fail_any_warn(sub_rules, consulted)
        reason = "runtime anomaly + void detection aggregation"
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported decision type: {decision_type}")

    triggered = [
        {"rule_class": name, **sub_rules[name]}
        for name in consulted
        if sub_rules[name]["verdict"] != "pass"
    ]
    return {
        "verdict": verdict,
        "reason": reason,
        "consulted_rule_classes": consulted,
        "triggered_rules": triggered,
        "sub_rules": {name: sub_rules[name] for name in consulted},
    }


def build_prompt(
    evidence: dict[str, Any],
    rule_verdict: dict[str, Any],
    decision_type: str,
    mode: str,
) -> str:
    """Render one prompt from markdown templates."""

    template_path = PROMPT_DIR / f"{decision_type}_{mode}.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing prompt template: {template_path}")
    schema_obj = judgment_schema_json()
    schema_text = json.dumps(schema_obj, indent=2, ensure_ascii=False)
    return (
        template_path.read_text(encoding="utf-8")
        .replace("{{DECISION_TYPE}}", decision_type)
        .replace("{{SCHEMA_JSON}}", schema_text)
        .replace("{{RULE_VERDICT_JSON}}", json.dumps(rule_verdict, indent=2, ensure_ascii=False))
        .replace("{{CASE_EVIDENCE_JSON}}", json.dumps(evidence, indent=2, ensure_ascii=False))
    )


def make_cache_key(
    *,
    evidence: dict[str, Any],
    rule_verdict: dict[str, Any],
    decision_type: str,
    mode: str,
    prompt: str,
    client_names: list[str],
) -> str:
    """Build a deterministic audit cache key.

    For the locked PE-LCB100 paper protocol, ``client_names`` should be the
    exact model-family payload listed in ``LOCKED_PAPER_MODEL_NAMES``:
    ``deepseek-v3-plus``, ``claude-sonnet-4-6``, and ``glm-4.6``.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "decision_type": decision_type,
        "mode": mode,
        "case_id": evidence.get("case_id"),
        "evidence": evidence,
        "rule_verdict": rule_verdict,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "client_names": client_names,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def aggregate_judgments(
    judgments: list[Judgment],
    *,
    mode: str,
    rule_verdict: dict[str, Any],
) -> Judgment:
    verdicts = [item.verdict for item in judgments]
    counts = {verdict: verdicts.count(verdict) for verdict in VERDICT_SCORE}
    max_count = max(counts.values())
    winners = [verdict for verdict, count in counts.items() if count == max_count]
    if len(winners) == 1:
        final_verdict = winners[0]
    else:
        sorted_scores = sorted(VERDICT_SCORE[item.verdict] for item in judgments)
        final_verdict = SCORE_VERDICT[sorted_scores[len(sorted_scores) // 2]]

    agreement_fraction = counts[final_verdict] / len(judgments)
    confidence = max(0.0, min(1.0, sum(item.confidence for item in judgments) / len(judgments) * agreement_fraction))
    if mode == "blind":
        agree_with_rule = None
    else:
        agree_with_rule = final_verdict == rule_verdict.get("verdict")

    ambiguity_parts = []
    if agreement_fraction < 1.0:
        ambiguity_parts.append("split_vote")
    if min(item.confidence for item in judgments) < 0.5:
        ambiguity_parts.append("low_model_confidence")
    ambiguity_flag = ";".join(ambiguity_parts) if ambiguity_parts else "none"
    rationale = (
        f"Tri-LLM adjudication selected {final_verdict} from votes "
        f"{', '.join(verdicts)} with agreement_fraction={agreement_fraction:.2f}."
    )
    return validate_judgment(
        {
            "verdict": final_verdict,
            "confidence": round(confidence, 4),
            "agree_with_rule": agree_with_rule,
            "rationale": rationale,
            "ambiguity_flag": ambiguity_flag,
        }
    )


def _call_client_with_retry(
    client: LLMClient,
    prompt: str,
    *,
    mode: str,
    rule_verdict: dict[str, Any],
    retries: int,
) -> Judgment:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            payload = parse_judgment_payload(_call_client_generate(client, prompt))
            if mode == "blind":
                payload["agree_with_rule"] = None
            elif payload.get("agree_with_rule") is None:
                payload["agree_with_rule"] = payload.get("verdict") == rule_verdict.get("verdict")
            return validate_judgment(payload)
        except Exception as exc:  # noqa: BLE001 - retry boundary
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"{client.name} did not return a valid judgment after {retries + 1} attempt(s): {last_error}")


def _call_client_generate(client: LLMClient, prompt: str) -> str | dict[str, Any]:
    if hasattr(client, "generate"):
        return client.generate(prompt)
    if hasattr(client, "complete"):
        return client.complete(prompt)  # type: ignore[attr-defined]
    raise TypeError(f"{getattr(client, 'name', type(client).__name__)} does not implement generate(prompt)")


def parse_judgment_payload(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse JSON returned by a model, accepting fenced JSON blocks."""

    if isinstance(raw, dict):
        return normalize_judgment_payload(dict(raw))
    text = str(raw).strip()
    try:
        return normalize_judgment_payload(json.loads(text))
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return normalize_judgment_payload(json.loads(fenced.group(1)))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return normalize_judgment_payload(json.loads(text[start : end + 1]))
    raise ValueError("LLM response did not contain a JSON object")


def normalize_judgment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize model JSON before schema validation.

    The public schema uses a single dominant ambiguity flag. Some providers may
    return comma- or semicolon-separated flags despite prompt constraints; keep
    the first tag as the dominant flag so audit records remain schema-stable.
    """

    if "ambiguity_flag" not in payload:
        return payload
    flag = str(payload.get("ambiguity_flag") or "none").strip()
    parts = [part.strip() for part in AMBIGUITY_FLAG_SPLIT_RE.split(flag) if part.strip()]
    if len(parts) > 1:
        payload["ambiguity_flag"] = parts[0]
    elif not flag:
        payload["ambiguity_flag"] = "none"
    return payload


def _default_clients() -> list[LLMClient]:
    """Construct clients from environment variables if explicitly configured."""

    if os.environ.get("TRI_LLM_USE_STATIC_CLIENTS") == "1":
        return [
            StaticJSONClient(
                f"static-{idx}",
                {
                    "verdict": "warn",
                    "confidence": 0.5,
                    "agree_with_rule": None,
                    "rationale": "Static dry-run client.",
                    "ambiguity_flag": "static_client",
                },
            )
            for idx in range(3)
        ]
    required = {
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "ZHIPUAI_API_KEY": (
            os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
        ),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing API keys: {missing}. Set them or use TRI_LLM_USE_STATIC_CLIENTS=1 for dry-run."
        )
    return [
        DeepSeekClient(model=os.environ.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL)),
        AnthropicClient(model=os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_MODEL)),
        GLMClient(model=os.environ.get("GLM_MODEL") or os.environ.get("ZHIPU_MODEL", GLM_MODEL)),
    ]


def build_evidence(case_dir: str | Path, decision_type: str) -> dict[str, Any]:
    """Compatibility wrapper used by Level 3 smoke scripts."""

    return collect_case_evidence(Path(case_dir), decision_type)


def _extract_thermal_metrics(
    analysis: dict[str, Any],
    scalars: dict[str, Any],
    trajectory_results: dict[str, Any],
) -> dict[str, Any]:
    metric = _first_nonempty(
        analysis.get("thermal"),
        _extract_named_scalars(scalars, ("Temperature", "temperature")),
        _first_present(trajectory_results, ("temperature", "Temperature")),
    )
    series = _extract_series(metric)
    stats = _window_stats(metric, series)
    return {
        "temperature_mean_K": _first_number(metric, ("temperature_mean_K", "mean", "mean_K")) or stats["mean"],
        "early_late_mean_delta_K": _first_number(
            metric,
            ("early_late_mean_delta_K", "front_back_delta_K", "mean_delta_K", "mean_delta"),
        )
        or stats["early_late_delta"],
        "late_linear_slope_K_per_sample": _first_number(
            metric,
            ("late_linear_slope_K_per_sample", "late_slope", "slope", "slope_K_per_sample"),
        )
        or stats["late_slope"],
        "series_points": stats["n"],
    }


def _extract_energetic_metrics(
    analysis: dict[str, Any],
    scalars: dict[str, Any],
    trajectory_results: dict[str, Any],
) -> dict[str, Any]:
    metric = _first_nonempty(
        analysis.get("energetic"),
        analysis.get("energy"),
        _extract_named_scalars(scalars, ("Total-Energy", "Total Energy", "E_total", "total_energy")),
        _first_present(trajectory_results, ("energy", "Energy")),
    )
    series = _extract_series(metric)
    stats = _window_stats(metric, series)
    mean_abs = abs(stats["mean"]) if stats["mean"] not in {None, 0.0} else None
    if mean_abs is None:
        scalar_mean = _first_number(metric, ("mean", "mean_kJ_mol", "total_energy_mean"))
        if scalar_mean not in {None, 0.0}:
            mean_abs = abs(float(scalar_mean))
    normalized = _first_number(
        metric,
        (
            "total_energy_late_slope_normalized",
            "normalized_late_slope",
            "energy_slope_over_abs_mean",
        ),
    )
    if normalized is None and stats["late_slope"] is not None and mean_abs:
        normalized = abs(stats["late_slope"]) / mean_abs
    return {
        "total_energy_late_slope": _first_number(metric, ("late_slope", "slope")) or stats["late_slope"],
        "total_energy_mean_abs": mean_abs,
        "total_energy_late_slope_over_abs_mean": normalized,
        "series_points": stats["n"],
    }


def _extract_compaction_metrics(
    analysis: dict[str, Any],
    scalars: dict[str, Any],
    trajectory_results: dict[str, Any],
    rule_config: dict[str, Any],
) -> dict[str, Any]:
    density_metric = _first_nonempty(
        analysis.get("compaction"),
        analysis.get("density"),
        _extract_named_scalars(scalars, ("Density", "density")),
        _first_present(trajectory_results, ("density", "Density")),
    )
    volume_metric = _first_nonempty(
        analysis.get("volume"),
        analysis.get("compaction"),
        _extract_named_scalars(scalars, ("Volume", "Box-Volume", "box_volume")),
        _first_present(trajectory_results, ("volume", "Volume")),
    )
    volume_stats = _window_stats(volume_metric, _extract_series(volume_metric))
    mean_abs = abs(volume_stats["mean"]) if volume_stats["mean"] not in {None, 0.0} else None
    normalized_volume_slope = _first_number(
        volume_metric,
        (
            "box_volume_late_slope_over_abs_mean",
            "volume_late_slope_over_abs_mean",
            "normalized_late_slope",
            "plateau_slope_over_abs_mean",
        ),
    )
    if normalized_volume_slope is None and volume_stats["late_slope"] is not None and mean_abs:
        normalized_volume_slope = abs(volume_stats["late_slope"]) / mean_abs
    return {
        "final_density_g_per_cc": _first_number(
            density_metric,
            ("final_density_g_per_cc", "rho_final_g_per_cc", "final", "mean"),
        ),
        "box_volume_late_slope_over_abs_mean": normalized_volume_slope,
        "box_volume_series_points": volume_stats["n"],
    }


def _extract_void_metrics(
    case_dir: Path,
    analysis: dict[str, Any],
    trajectory_results: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        analysis.get("void"),
        analysis.get("grid_void"),
        _first_present(trajectory_results, ("void", "grid_void")),
    ]
    for rel in (
        "grid_void.json",
        "void_metrics.json",
        "grid_void_metrics.json",
        "void/grid_void.json",
        "void/void_metrics.json",
    ):
        path = case_dir / rel
        if path.exists():
            payload = _read_json(path)
            payload["_source_path"] = str(path)
            candidates.append(payload)
    metric = _first_nonempty(*candidates)
    return {
        "void_flag": _first_bool(metric, ("void_flag", "flag")),
        "low_density_voxel_fraction": _first_number(
            metric,
            ("low_density_voxel_fraction", "low_density_fraction", "f_low"),
        ),
        "largest_connected_low_density_fraction": _first_number(
            metric,
            (
                "largest_connected_low_density_fraction",
                "largest_component_fraction",
                "largest_low_density_component_volume_fraction",
            ),
        ),
        "source_path": metric.get("_source_path") if isinstance(metric, dict) else None,
    }


def _extract_fit_quality_metrics(
    trajectory: dict[str, Any],
    trajectory_results: dict[str, Any],
) -> dict[str, Any]:
    metric = _first_nonempty(
        trajectory.get("fit_quality"),
        trajectory.get("msd"),
        _first_present(trajectory_results, ("msd", "MSD", "displacement")),
    )
    return {
        "msd_loglog_slope": _first_number(
            metric,
            ("msd_loglog_slope", "loglog_slope", "slope"),
        ),
        "fit_window_frames": _first_number(
            metric,
            ("fit_window_frames", "window_frames", "n_fit_frames", "fit_points"),
        ),
        "window_sensitivity": _first_number(
            metric,
            ("window_sensitivity", "slope_window_sensitivity", "window_delta"),
        ),
    }


def _extract_runtime_metrics(case_dir: Path, analysis: dict[str, Any]) -> dict[str, Any]:
    log_text = ""
    log_files = []
    for rel in ("manual.out", "manual.err"):
        path = case_dir / rel
        if path.exists():
            log_files.append(str(path))
            log_text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    fatal_count = _count_regex(
        log_text,
        r"Fatal error|LINCS|CUDA.*(?:error|illegal)|segmentation fault|abnormal termination",
    )
    nonfinite_count = _count_regex(log_text, r"\bNaN\b|\bInf\b|non-?finite")
    runtime_payload = analysis.get("runtime", {}) if isinstance(analysis, dict) else {}
    fatal_count += int(_first_number(runtime_payload, ("fatal_log_marker_count",)) or 0)
    nonfinite_count += int(_first_number(runtime_payload, ("non_finite_energy_count", "nonfinite_energy_count")) or 0)
    checkpoint_integrity = _first_bool(
        runtime_payload,
        ("checkpoint_integrity", "checkpoint_or_done_integrity"),
    )
    if checkpoint_integrity is None:
        has_checkpoint = any(case_dir.glob("**/*.cpt"))
        has_done_marker = "Stage nvt_prod completed at" in log_text or "All stages completed" in log_text
        if log_text or has_checkpoint:
            checkpoint_integrity = bool(has_checkpoint or has_done_marker)
    return {
        "fatal_log_marker_count": fatal_count,
        "non_finite_energy_count": nonfinite_count,
        "checkpoint_integrity": checkpoint_integrity,
        "log_files_scanned": log_files,
    }


def _thermal_rule(metric: dict[str, Any]) -> dict[str, Any]:
    delta = _as_float(metric.get("early_late_mean_delta_K"))
    slope = _as_float(metric.get("late_linear_slope_K_per_sample"))
    missing = [name for name, value in {"early_late_mean_delta_K": delta, "late_linear_slope_K_per_sample": slope}.items() if value is None]
    if missing:
        return _rule("warn", "thermal stability evidence incomplete", missing)
    if abs(delta) > THERMAL_DELTA_FAIL_K or abs(slope) > THERMAL_SLOPE_FAIL_K_PER_SAMPLE:
        return _rule("fail", "temperature drift exceeds failure threshold", ["temperature_drift"])
    if abs(delta) > THERMAL_DELTA_WARN_K or abs(slope) > THERMAL_SLOPE_WARN_K_PER_SAMPLE:
        return _rule("warn", "temperature drift is borderline", ["temperature_drift_borderline"])
    return _rule("pass", "temperature windows and late slope are stable", [])


def _energetic_rule(metric: dict[str, Any]) -> dict[str, Any]:
    normalized = _as_float(metric.get("total_energy_late_slope_over_abs_mean"))
    if normalized is None:
        return _rule("warn", "total-energy drift evidence incomplete", ["missing_energy_drift"])
    if abs(normalized) > ENERGY_NORM_SLOPE_FAIL:
        return _rule("fail", "normalized total-energy drift exceeds failure threshold", ["energy_drift"])
    if abs(normalized) > ENERGY_NORM_SLOPE_WARN:
        return _rule("warn", "normalized total-energy drift is borderline", ["energy_drift_borderline"])
    return _rule("pass", "total-energy late-window drift is stable", [])


def _compaction_rule(metric: dict[str, Any]) -> dict[str, Any]:
    final_density = _as_float(metric.get("final_density_g_per_cc"))
    rho_pass_low = _as_float(metric.get("rho_pass_low_g_per_cc"))
    rho_pass_high = _as_float(metric.get("rho_pass_high_g_per_cc"))
    if rho_pass_low is None:
        rho_pass_low = COMPACTION_RHO_PASS_LOW
    if rho_pass_high is None:
        rho_pass_high = COMPACTION_RHO_PASS_HIGH
    volume_slope = _as_float(metric.get("box_volume_late_slope_over_abs_mean"))
    missing = []
    if final_density is None:
        missing.append("final_density_g_per_cc")
    if volume_slope is None:
        missing.append("box_volume_late_slope_over_abs_mean")
    if missing:
        return _rule("warn", "compaction evidence incomplete", missing)
    if abs(volume_slope) > VOLUME_NORM_SLOPE_FAIL:
        return _rule("fail", "box volume is not plateaued", ["compaction_failure"])
    triggered = []
    if final_density < rho_pass_low or final_density > rho_pass_high:
        triggered.append("density_warn_tier")
    if abs(volume_slope) > VOLUME_NORM_SLOPE_WARN:
        triggered.append("box_volume_plateau_borderline")
    if triggered:
        return _rule("warn", "short-chain compaction is retained as a warning-tier case", triggered)
    return _rule("pass", "density tier and box-volume plateau pass", [])


def _void_rule(metric: dict[str, Any]) -> dict[str, Any]:
    flag = metric.get("void_flag")
    low_fraction = _as_float(metric.get("low_density_voxel_fraction"))
    component_fraction = _as_float(metric.get("largest_connected_low_density_fraction"))
    if flag is True:
        return _rule("fail", "grid void flag is set", ["void_flag"])
    if low_fraction is None and component_fraction is None and flag is None:
        return _rule("warn", "grid void evidence missing", ["missing_void_metrics"])
    if (low_fraction is not None and low_fraction >= VOID_LOW_FRACTION_FAIL) or (
        component_fraction is not None and component_fraction >= VOID_COMPONENT_FRACTION_FAIL
    ):
        return _rule("fail", "low-density voxel region exceeds failure threshold", ["large_low_density_region"])
    if (low_fraction is not None and low_fraction >= VOID_LOW_FRACTION_WARN) or (
        component_fraction is not None and component_fraction >= VOID_COMPONENT_FRACTION_WARN
    ):
        return _rule("warn", "low-density voxel region is borderline", ["borderline_low_density_region"])
    return _rule("pass", "grid void metrics pass screening thresholds", [])


def _fit_quality_rule(metric: dict[str, Any]) -> dict[str, Any]:
    slope = _as_float(metric.get("msd_loglog_slope"))
    frames = _as_float(metric.get("fit_window_frames"))
    sensitivity = _as_float(metric.get("window_sensitivity"))
    if slope is None:
        return _rule("warn", "MSD log-log slope missing", ["missing_msd_slope"])
    if slope < MSD_SLOPE_LOW or slope > MSD_SLOPE_HIGH:
        return _rule("fail", "MSD log-log slope outside target band", ["msd_slope_outside_band"])
    if frames is not None and frames < MSD_MIN_WINDOW_FRAMES:
        return _rule("fail", "MSD fitting window is too short", ["fit_window_too_short"])
    if frames is None or sensitivity is None:
        return _rule("warn", "MSD fit diagnostics incomplete", ["missing_fit_diagnostics"])
    if sensitivity > MSD_WINDOW_SENSITIVITY_WARN:
        return _rule("warn", "MSD fit is window-sensitive", ["window_sensitive_fit"])
    return _rule("pass", "MSD log-log slope and window diagnostics pass", [])


def _runtime_rule(metric: dict[str, Any]) -> dict[str, Any]:
    fatal_count = int(_as_float(metric.get("fatal_log_marker_count")) or 0)
    nonfinite_count = int(_as_float(metric.get("non_finite_energy_count")) or 0)
    checkpoint_ok = metric.get("checkpoint_integrity")
    if fatal_count > 0 or nonfinite_count > 0 or checkpoint_ok is False:
        return _rule("fail", "runtime marker, non-finite scan, or checkpoint integrity failed", ["runtime_failure"])
    if checkpoint_ok is None:
        return _rule("warn", "checkpoint integrity evidence missing", ["missing_checkpoint_integrity"])
    return _rule("pass", "runtime logs and checkpoint integrity pass", [])


def _all_pass_any_fail(sub_rules: dict[str, dict[str, Any]], names: list[str]) -> str:
    verdicts = [sub_rules[name]["verdict"] for name in names]
    if any(verdict == "fail" for verdict in verdicts):
        return "fail"
    if all(verdict == "pass" for verdict in verdicts):
        return "pass"
    return "warn"


def _any_fail_any_warn(sub_rules: dict[str, dict[str, Any]], names: list[str]) -> str:
    verdicts = [sub_rules[name]["verdict"] for name in names]
    if any(verdict == "fail" for verdict in verdicts):
        return "fail"
    if any(verdict == "warn" for verdict in verdicts):
        return "warn"
    return "pass"


def _rule(verdict: str, reason: str, triggered: list[str]) -> dict[str, Any]:
    return {"verdict": verdict, "reason": reason, "triggered": triggered}


def _load_rule_config(case_dir: Path, analysis: dict[str, Any]) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for key in ("rule_config", "quality_config", "protocol_config"):
        value = analysis.get(key) if isinstance(analysis, dict) else None
        if isinstance(value, dict):
            configs.append(value)
    for rel in ("rule_config.json", "quality_config.json", "protocol_config.json"):
        path = case_dir / rel
        if path.exists():
            payload = _read_json(path)
            payload["_source_path"] = str(path)
            configs.append(payload)
    env_path = os.environ.get("LCB_PROJECT_CONFIG")
    if env_path and Path(env_path).exists():
        payload = _read_json(Path(env_path))
        payload["_source_path"] = env_path
        configs.append(payload)
    default_config = MODULE_DIR.parent / "configs" / "pelcb100.json"
    if default_config.exists():
        payload = _read_json(default_config)
        payload["_source_path"] = str(default_config)
        configs.append(payload)
    return {"sources": configs}


def _extract_target_density_range(rule_config: dict[str, Any]) -> dict[str, Any]:
    for source in rule_config.get("sources", []):
        found = _find_first_range(
            source,
            (
                "protocol_target_density_range_g_per_cc",
                "density_target_range_g_per_cc",
                "target_density_range_g_per_cc",
                "target_density_range",
                "rho_target_range",
            ),
        )
        if found:
            return {"range": found, "source": source.get("_source_path") or "analysis_or_config"}
        low = _find_first_number(source, ("rho_target_low", "density_target_low_g_per_cc", "target_density_low"))
        high = _find_first_number(source, ("rho_target_high", "density_target_high_g_per_cc", "target_density_high"))
        if low is not None and high is not None:
            return {"range": [low, high], "source": source.get("_source_path") or "analysis_or_config"}
    return {"range": None, "source": None}


def _extract_compaction_density_tiers(rule_config: dict[str, Any]) -> dict[str, Any]:
    for source in rule_config.get("sources", []):
        pass_low = _find_first_number(
            source,
            (
                "rho_pass_low",
                "compaction_rho_pass_low_g_per_cc",
                "density_pass_low_g_per_cc",
                "rho_density_pass_low_g_per_cc",
            ),
        )
        pass_high = _find_first_number(
            source,
            (
                "rho_pass_high",
                "compaction_rho_pass_high_g_per_cc",
                "density_pass_high_g_per_cc",
                "rho_density_pass_high_g_per_cc",
            ),
        )
        tiers = source.get("compaction_density_tiers") if isinstance(source, dict) else None
        if isinstance(tiers, dict):
            pass_low = pass_low if pass_low is not None else _find_first_number(tiers, ("pass_low", "rho_pass_low"))
            pass_high = pass_high if pass_high is not None else _find_first_number(tiers, ("pass_high", "rho_pass_high"))
        if None not in {pass_low, pass_high}:
            return {
                "pass_low": pass_low,
                "pass_high": pass_high,
                "source": source.get("_source_path") or "analysis_or_config",
            }
    return {
        "pass_low": COMPACTION_RHO_PASS_LOW,
        "pass_high": COMPACTION_RHO_PASS_HIGH,
        "source": "default_c100_short_chain_tier",
    }


def _first_nonempty(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _extract_series(metric: Any) -> list[float]:
    if not isinstance(metric, dict):
        return []
    for key in ("series", "values", "time_series", "timeseries", "data"):
        value = metric.get(key)
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, (list, tuple)) and item:
                    number = _as_float(item[-1])
                elif isinstance(item, dict):
                    number = _first_number(item, ("value", "y", "density", "temperature", "energy", "volume"))
                else:
                    number = _as_float(item)
                if number is not None:
                    out.append(number)
            return out
    return []


def _window_stats(metric: dict[str, Any], series: list[float]) -> dict[str, Any]:
    mean = _first_number(metric, ("mean", "average"))
    early_late_delta = _first_number(metric, ("early_late_delta", "early_late_mean_delta", "front_back_delta"))
    late_slope = _first_number(metric, ("late_slope", "slope", "linear_slope"))
    if series:
        if mean is None:
            mean = sum(series) / len(series)
        half = max(1, len(series) // 2)
        early = series[:half]
        late = series[-half:]
        if early_late_delta is None:
            early_late_delta = (sum(late) / len(late)) - (sum(early) / len(early))
        if late_slope is None and len(late) >= 2:
            late_slope = _linear_slope(late)
    return {"mean": mean, "early_late_delta": early_late_delta, "late_slope": late_slope, "n": len(series)}


def _linear_slope(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    denom = sum((i - x_mean) ** 2 for i in range(n))
    if denom == 0:
        return None
    return sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values)) / denom


def _first_number(metric: Any, names: tuple[str, ...]) -> float | None:
    if not isinstance(metric, dict):
        return None
    for name in names:
        if name in metric:
            number = _as_float(metric.get(name))
            if number is not None:
                return number
    lowered = {str(key).lower(): value for key, value in metric.items()}
    for name in names:
        value = lowered.get(name.lower())
        number = _as_float(value)
        if number is not None:
            return number
    return None


def _first_bool(metric: Any, names: tuple[str, ...]) -> bool | None:
    if not isinstance(metric, dict):
        return None
    lowered = {str(key).lower(): value for key, value in metric.items()}
    for name in names:
        value = metric.get(name, lowered.get(name.lower()))
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
    return None


def _find_first_number(value: Any, names: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        number = _first_number(value, names)
        if number is not None:
            return number
        for item in value.values():
            number = _find_first_number(item, names)
            if number is not None:
                return number
    elif isinstance(value, list):
        for item in value:
            number = _find_first_number(item, names)
            if number is not None:
                return number
    return None


def _find_first_range(value: Any, names: tuple[str, ...]) -> list[float] | None:
    if isinstance(value, dict):
        lowered = {str(key).lower(): item for key, item in value.items()}
        for name in names:
            item = value.get(name, lowered.get(name.lower()))
            parsed = _parse_range(item)
            if parsed:
                return parsed
        for item in value.values():
            parsed = _find_first_range(item, names)
            if parsed:
                return parsed
    elif isinstance(value, list):
        for item in value:
            parsed = _find_first_range(item, names)
            if parsed:
                return parsed
    return None


def _parse_range(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        low = _first_number(value, ("low", "min", "lower"))
        high = _first_number(value, ("high", "max", "upper"))
        if low is not None and high is not None:
            return [low, high]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        low = _as_float(value[0])
        high = _as_float(value[1])
        if low is not None and high is not None:
            return [low, high]
    return None


def _valid_range(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and _as_float(value[0]) is not None
        and _as_float(value[1]) is not None
    )


def _count_regex(text: str, pattern: str) -> int:
    if not text:
        return 0
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _extract_named_scalars(scalars: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(scalars, dict):
        return {}
    lowered = tuple(name.lower() for name in names)
    return {key: value for key, value in scalars.items() if any(name in str(key).lower() for name in lowered)}


def _first_present(payload: dict[str, Any], names: tuple[str, ...]) -> Any:
    if not isinstance(payload, dict):
        return {}
    lowered = tuple(name.lower() for name in names)
    for key, value in payload.items():
        if any(name in str(key).lower() for name in lowered):
            return value
    return {}


def _has_severe_warning(warnings: Any) -> bool:
    if not warnings:
        return False
    text = json.dumps(warnings, ensure_ascii=False).lower()
    return any(marker in text for marker in ("fatal", "nan", "lincs", "cuda", "exploded", "segmentation"))


def _as_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_choice(name: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _dump_model(model: Judgment) -> dict[str, Any]:
    try:
        return model.model_dump()
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return model.dict()


__all__ = [
    "DECISION_TYPES",
    "MODES",
    "LLMClient",
    "StaticJSONClient",
    "aggregate_judgments",
    "AnthropicClient",
    "build_evidence",
    "build_prompt",
    "collect_case_evidence",
    "compute_rule_verdict",
    "DeepSeekClient",
    "GLMClient",
    "judge",
    "make_cache_key",
    "parse_judgment_payload",
    "QwenClient",
    "_default_clients",
]
