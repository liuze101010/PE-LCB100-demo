"""Live GPT planning for bounded-autonomous PE-LCB candidate design.

The model proposes molecular architecture coordinates from the research goal.
It does not receive an enumerated answer catalogue. Concrete branch positions,
macro-SMILES and builder specifications are generated and checked independently
by the deterministic adapter after the model call.
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .control_plane.design_grammar import (
    BRANCH_CARBONS,
    PATTERN_TO_TOPOLOGY,
    build_candidate,
    candidates_from_proposals,
    catalog_sha256,
    design_catalog,
)


PlacementCode = Literal[
    "mid", "near-end", "far-end", "near-start",
    "uniform", "two-ends", "mid-pair", "H-asymmetric", "clustered", "dumbbell-far",
    "two-ends+mid", "three-point", "end-cluster", "pompom-dense",
    "comb-uniform-4", "pompom-4", "cluster-4", "gradient",
    "comb-uniform-5", "pompom-5", "cluster-5", "gradient-5",
]


class ModelDesignChoice(BaseModel):
    branch_points: int = Field(ge=1, le=5)
    branch_carbons: Literal[3, 4, 5, 8, 12, 16, 20]
    placement_code: PlacementCode
    rationale: str = Field(default="", max_length=180)


class ModelDesignPlan(BaseModel):
    mode: Literal["new_design", "reproduce_pelcb100"] = "new_design"
    requested_count: int = Field(ge=1, le=100)
    strategy_summary: str = Field(min_length=1, max_length=500)
    choices: list[ModelDesignChoice] = Field(default_factory=list, max_length=100)


class DesignPlanError(RuntimeError):
    pass


_DESIGN_VERBS = "\u8bbe\u8ba1|\u751f\u6210|\u63d0\u51fa|\u7b5b\u9009|\u6784\u5efa"
_COUNT_NOUNS = "\u4e2a|\u79cd|\u6761|\u4f8b"
_ARABIC_COUNT = re.compile(
    rf"(?:{_DESIGN_VERBS}|design|generate|propose)\s*(\d{{1,3}})\s*(?:{_COUNT_NOUNS}|cases?|structures?)?",
    re.IGNORECASE,
)
_COUNT_BEFORE_NOUN = re.compile(rf"(\d{{1,3}})\s*(?:{_COUNT_NOUNS}|cases?|structures?)", re.IGNORECASE)
_CHINESE_COUNT = re.compile(
    rf"(?:{_DESIGN_VERBS})([\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24\u3007\u96f6]{{1,4}})(?:{_COUNT_NOUNS})"
)


def _chinese_integer(text: str) -> int | None:
    digits = {
        "\u96f6": 0, "\u3007": 0, "\u4e00": 1, "\u4e8c": 2, "\u4e24": 2,
        "\u4e09": 3, "\u56db": 4, "\u4e94": 5, "\u516d": 6,
        "\u4e03": 7, "\u516b": 8, "\u4e5d": 9,
    }
    hundred = "\u767e"
    ten = "\u5341"
    if text == hundred:
        return 100
    if hundred in text:
        left, _, right = text.partition(hundred)
        hundreds = digits.get(left, 1) if left else 1
        tail = _chinese_integer(right) if right else 0
        return hundreds * 100 + (tail or 0)
    if ten in text:
        left, _, right = text.partition(ten)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    if all(char in digits for char in text):
        return int("".join(str(digits[char]) for char in text))
    return None


def requested_count_from_goal(goal: str, default: int = 100) -> int:
    if re.search(r"\b(single|one case|smoke)\b|\u5355\u4e2a|\u5355\u4f8b|\u5c0f\u8bd5", goal, re.IGNORECASE):
        return 1
    match = _ARABIC_COUNT.search(goal) or _COUNT_BEFORE_NOUN.search(goal)
    if match:
        count = int(match.group(1))
    else:
        chinese = _CHINESE_COUNT.search(goal)
        count = _chinese_integer(chinese.group(1)) if chinese else default
    if count is None or not 1 <= count <= 100:
        raise ValueError("requested design count must be between 1 and 100")
    return count


def design_enabled(api_key: str | None = None) -> bool:
    flag = (os.getenv("POLYMER_AGENT_LIVE_DESIGN") or "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and bool(api_key or os.getenv("OPENAI_API_KEY"))


def benchmark_goal(goal: str) -> bool:
    return bool(re.search(r"(?:复现|重现|reproduce|replicate).{0,40}PE[- ]?LCB100|PE[- ]?LCB100.{0,40}(?:复现|重现|reproduce|replicate)", goal, re.IGNORECASE))


def benchmark_manifest() -> tuple[list[dict[str, Any]], str]:
    bundle = Path(__file__).resolve().parent / "deploy" / "science_core_bundle.zip"
    member = "science_core/generated_code/emc_alpha_olefin_builder/output/screen_pelcb100/manifest.json"
    if not bundle.is_file():
        raise DesignPlanError("the pinned PE-LCB100 benchmark manifest is not installed")
    with zipfile.ZipFile(bundle) as archive:
        raw = archive.read(member)
    payload = json.loads(raw)
    rows = payload.get("cases") or []
    if len(rows) != 100:
        raise DesignPlanError("the pinned PE-LCB100 benchmark manifest is incomplete")
    return rows, hashlib.sha256(raw).hexdigest()


def benchmark_candidates() -> tuple[list[dict[str, Any]], str]:
    rows, digest = benchmark_manifest()
    candidates = []
    for sequence, row in enumerate(rows, start=1):
        candidate = build_candidate(
            int(row["branch_points"]),
            int(row["branch_carbons"]),
            str(row["branch_position_pattern"]),
            sequence=sequence,
            rationale="locked PE-LCB100 benchmark design",
        )
        candidate["case_id"] = str(row["case_id"])
        candidates.append(candidate)
    if len({item["structure_sha256"] for item in candidates}) != 100:
        raise DesignPlanError("the pinned PE-LCB100 benchmark contains duplicate structures")
    return candidates, digest


def _prompt(goal: str, count: int, correction: str = "") -> str:
    placement_rules = "\n".join(
        f"- branch_points={branch_points}: {', '.join(patterns)}"
        for branch_points, patterns in PATTERN_TO_TOPOLOGY.items()
    )
    correction_text = (
        f"\nThe previous proposal failed application validation: {correction}. "
        "Regenerate the complete plan under the constraints."
        if correction else ""
    )
    return (
        f"Research goal supplied by the user: {goal}\n"
        "Set mode=reproduce_pelcb100 only when the user explicitly requests reproduction of the PE-LCB100 paper/benchmark; "
        "in that mode requested_count must be 100 and choices must be empty because the application will load the pinned manifest. "
        "Otherwise set mode=new_design and propose choices.\n"
        f"Autonomously propose exactly {count} unique, verifiable PE-LCB/C100 molecular architectures.\n"
        "For every candidate, provide branch_points, branch_carbons and placement_code directly. "
        "Do not output a design_key, SMILES, branch positions, or unrequested fields.\n"
        f"Allowed branch_carbons values: {', '.join(map(str, BRANCH_CARBONS))}.\n"
        "Each placement_code must be compatible with branch_points:\n"
        f"{placement_rules}\n"
        "The chain manifest carbon budget is C_manifest = 2N + bL with target C100. Use C101 when "
        "100-bL cannot yield an integer N. Backbone carbons must be at least 20.\n"
        "Use the research goal to organize the design space. Cover relevant variation in branch count, "
        "branch length and placement, and give one concise, auditable scientific rationale per candidate. "
        "All candidates must be distinct. Do not reveal hidden reasoning."
        f"{correction_text}"
    )


def _validate_plan(plan: ModelDesignPlan, count: int, goal: str) -> tuple[list[dict[str, Any]], str | None]:
    if plan.requested_count != count:
        raise DesignPlanError(f"model returned requested_count={plan.requested_count}, expected {count}")
    expected_benchmark = benchmark_goal(goal)
    if expected_benchmark and plan.mode != "reproduce_pelcb100":
        raise DesignPlanError("explicit PE-LCB100 reproduction request was not classified as reproduce_pelcb100")
    if not expected_benchmark and plan.mode == "reproduce_pelcb100":
        raise DesignPlanError("reproduce_pelcb100 requires an explicit benchmark reproduction request")
    if plan.mode == "reproduce_pelcb100":
        if count != 100 or plan.choices:
            raise DesignPlanError("benchmark reproduction requires requested_count=100 and no generated choices")
        candidates, digest = benchmark_candidates()
        return candidates, digest
    choices = [choice.model_dump() for choice in plan.choices]
    if len(choices) != count:
        raise DesignPlanError(f"model returned {len(choices)} choices, expected {count}")
    coordinates = {
        (choice["branch_points"], choice["branch_carbons"], choice["placement_code"])
        for choice in choices
    }
    if len(coordinates) != count:
        raise DesignPlanError("model returned duplicate molecular architecture coordinates")
    try:
        return candidates_from_proposals(choices), None
    except (TypeError, ValueError) as exc:
        raise DesignPlanError(str(exc)) from exc


def propose_designs(goal: str, count: int, api_key: str | None = None) -> dict[str, Any]:
    if not design_enabled(api_key):
        raise DesignPlanError("live GPT design is not enabled or OPENAI_API_KEY is missing")
    from openai import OpenAI

    model = os.getenv("POLYMER_AGENT_DESIGN_MODEL") or "gpt-5.5"
    effort = (os.getenv("POLYMER_AGENT_DESIGN_REASONING") or "medium").strip().lower()
    if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        effort = "medium"
    key = api_key or os.environ["OPENAI_API_KEY"]
    kwargs: dict[str, Any] = {"api_key": key, "timeout": 120.0, "max_retries": 1}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(**kwargs)
    error = ""
    for attempt in range(1, 3):
        response = client.responses.parse(
            model=model,
            instructions=(
                "You are the PE-LCB molecular-design planning model. Autonomously propose candidates from "
                "the research goal, but remain inside the registered chemistry and architecture boundary. "
                "Do not alter the force field, thermodynamic protocol, chain count, rule thresholds, or "
                "requested candidate count. Return only the required structured output."
            ),
            input=_prompt(goal, count, error),
            text_format=ModelDesignPlan,
            reasoning={"effort": effort},
            max_output_tokens=min(16000, max(3000, count * 140)),
            store=False,
        )
        plan = response.output_parsed
        if plan is None:
            error = "the model did not return a parseable structured design"
            continue
        try:
            candidates, benchmark_sha256 = _validate_plan(plan, count, goal)
        except DesignPlanError as exc:
            error = str(exc)
            continue
        return {
            "engine": f"openai_responses:{model}",
            "model": model,
            "response_id": response.id,
            "requested_count": count,
            "mode": plan.mode,
            "benchmark_manifest_sha256": benchmark_sha256,
            "strategy_summary": plan.strategy_summary,
            "attempt": attempt,
            "catalog_size": len(design_catalog()),
            "catalog_sha256": catalog_sha256(),
            "candidates": candidates,
        }
    raise DesignPlanError(error or "model design failed application validation")
