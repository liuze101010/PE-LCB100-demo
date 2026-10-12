"""Upper-LLM bounded execution planning for PE-LCB runs.

This module gives the top-level OpenAI model one auditable planning turn before
the run is submitted.  The returned plan is metadata only: execution remains
controlled by registered repair contracts, tool guards and rule re-gating.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from .control_plane.config import model_configuration


ContractName = Literal[
    "run_registered_md_pipeline",
    "density_clean_profile_regate",
    "missing_output_backfill",
    "clean_msd_bounded_extension",
    "postprocess_recompute",
    "review_hold",
]


class ExecutionPlanItem(BaseModel):
    case_selector: str = Field(min_length=1, max_length=120)
    trigger_state: str = Field(min_length=1, max_length=120)
    proposed_contract: ContractName
    priority: int = Field(ge=1, le=100)
    rationale: str = Field(min_length=1, max_length=240)


class UpperExecutionPlan(BaseModel):
    plan_scope: Literal["pre_run_bounded_closed_loop"]
    summary: str = Field(min_length=1, max_length=500)
    items: list[ExecutionPlanItem] = Field(min_length=1, max_length=12)


RepairContractName = Literal[
    "density_clean_profile_regate",
    "missing_output_backfill",
    "clean_msd_bounded_extension",
    "postprocess_recompute",
    "review_hold",
]


class RepairPlanItem(BaseModel):
    case_ids: list[str] = Field(min_length=1, max_length=100)
    proposed_contract: RepairContractName
    priority: int = Field(ge=1, le=100)
    rationale: str = Field(min_length=1, max_length=500)


class UpperRepairPlan(BaseModel):
    plan_scope: Literal["post_audit_bounded_reentry"]
    summary: str = Field(min_length=1, max_length=800)
    items: list[RepairPlanItem] = Field(min_length=1, max_length=100)


class ExecutionPlanError(RuntimeError):
    pass


def planner_enabled(api_key: str | None = None) -> bool:
    if str(os.getenv("POLYMER_AGENT_UPPER_PLANNER") or "1").strip().lower() in {"0", "false", "off", "no"}:
        return False
    return bool(api_key or os.getenv("OPENAI_API_KEY"))


def planner_required() -> bool:
    return str(os.getenv("POLYMER_AGENT_REQUIRE_UPPER_PLANNER") or "").strip().lower() in {"1", "true", "yes", "on"}


def _project_digest(project: dict[str, Any]) -> dict[str, Any]:
    manifest = project.get("manifest") or []
    spec = project.get("spec") or {}
    return {
        "project_id": project.get("id"),
        "name": project.get("name"),
        "status": project.get("status"),
        "domain_status": project.get("domain_status"),
        "manifest_sha256": project.get("manifest_sha256"),
        "case_count": len(manifest) or len(spec.get("candidates") or []),
        "polymer_family": spec.get("polymer_family"),
        "architecture": spec.get("architecture"),
        "force_field": spec.get("force_field"),
        "temperature_k": spec.get("temperature_k"),
        "pressure_atm": spec.get("pressure_atm"),
        "production_ns": spec.get("production_ns"),
        "rule_profile": spec.get("rule_profile"),
        "validation": project.get("validation"),
        "case_ids": [row.get("case_id") for row in manifest[:100] if isinstance(row, dict)],
    }


def _prompt(project: dict[str, Any], run: dict[str, Any], approval: dict[str, Any]) -> str:
    payload = {
        "project": _project_digest(project),
        "run": {
            "id": run.get("id"),
            "state": run.get("state"),
            "cluster_id_hash_only": bool(run.get("cluster_id")),
        },
        "approved_budget": {
            "max_repair_attempts": approval.get("max_repair_attempts", 1),
            "max_additional_md_ns": approval.get("max_additional_md_ns", 400),
            "max_cases": approval.get("max_cases", 100),
        },
        "registered_contracts": [
            "run_registered_md_pipeline",
            "density_clean_profile_regate",
            "missing_output_backfill",
            "clean_msd_bounded_extension",
            "postprocess_recompute",
            "review_hold",
        ],
        "hard_guards": [
            "Do not change force field, temperature, pressure, rule thresholds or manifest.",
            "Any density/MSD/missing-output repair must match a registered contract.",
            "Persistent MSD warning after about 800 ns must become review_hold, not unbounded extension.",
            "Final PASS is decided only by rule re-gating, not by this plan.",
        ],
    }
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_upper_execution_plan(project: dict[str, Any], run: dict[str, Any], approval: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    if not planner_enabled(api_key):
        raise ExecutionPlanError("upper GPT execution planner is disabled or OPENAI_API_KEY is missing")
    from openai import OpenAI

    model = model_configuration()["upper_planner"]
    effort = (os.getenv("POLYMER_AGENT_UPPER_PLANNER_REASONING") or "low").strip().lower()
    if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        effort = "low"
    key = api_key or os.environ["OPENAI_API_KEY"]
    kwargs: dict[str, Any] = {"api_key": key, "timeout": 120.0, "max_retries": 1}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(**kwargs)
    response = client.responses.parse(
        model=model,
        instructions=(
            "You are the upper PE-LCB closed-loop planning model. Generate a bounded execution plan from "
            "the project validation state, run budget and registered repair contracts. The plan is advisory "
            "metadata only: tools, physical parameters, rule thresholds and final publication status remain "
            "controlled by registered guards and rule re-gating. Return only the structured plan."
        ),
        input=_prompt(project, run, approval),
        text_format=UpperExecutionPlan,
        reasoning={"effort": effort},
        max_output_tokens=2500,
        store=False,
    )
    plan = response.output_parsed
    if plan is None:
        raise ExecutionPlanError("upper GPT planner did not return a parseable structured plan")
    return {
        "schema_version": "polymer_agent.upper_execution_plan.v1",
        "engine": f"openai_responses:{model}",
        "model": model,
        "response_id": response.id,
        "guard_status": "metadata_only_requires_registered_contract_match",
        "plan": plan.model_dump(),
    }


def build_upper_repair_plan(
    audit_cases: list[dict[str, Any]],
    approval: dict[str, Any],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    if not planner_enabled(api_key):
        raise ExecutionPlanError("upper GPT repair planner is disabled or OPENAI_API_KEY is missing")
    from openai import OpenAI
    import json

    model = model_configuration()["upper_planner"]
    effort = (os.getenv("POLYMER_AGENT_UPPER_PLANNER_REASONING") or "low").strip().lower()
    if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        effort = "low"
    key = api_key or os.environ["OPENAI_API_KEY"]
    kwargs: dict[str, Any] = {"api_key": key, "timeout": 180.0, "max_retries": 1}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    payload = {
        "cases": audit_cases,
        "approved_budget": {
            "max_repair_attempts": approval.get("max_repair_attempts", 1),
            "max_additional_md_ns": approval.get("max_additional_md_ns", 400),
        },
        "hard_guards": [
            "Choose only a contract listed in allowed_contracts for that case.",
            "Do not change force field, physical parameters, thresholds or the locked manifest.",
            "An MSD warning after a bounded extension to 800 ns must use review_hold.",
            "PASS is assigned only after deterministic rule re-evaluation.",
        ],
    }
    response = OpenAI(**kwargs).responses.parse(
        model=model,
        instructions=(
            "You are the upper PE-LCB closed-loop planning model. Read each case's deterministic rule state "
            "and independent three-model audit consensus, then select one registered re-entry contract per case. "
            "Do not invent tools or alter physics. Return only the structured plan."
        ),
        input=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        text_format=UpperRepairPlan,
        reasoning={"effort": effort},
        max_output_tokens=8000,
        store=False,
    )
    if response.output_parsed is None:
        raise ExecutionPlanError("upper GPT repair planner did not return a structured plan")
    return {
        "schema_version": "polymer_agent.upper_repair_plan.v1",
        "model": model,
        "response_id": response.id,
        "plan": response.output_parsed.model_dump(),
    }


def guard_upper_repair_plan(plan: dict[str, Any], allowed: dict[str, set[str]]) -> dict[str, Any]:
    selected: dict[str, str] = {}
    rejected: list[dict[str, str]] = []
    for item in (plan.get("plan") or {}).get("items", []):
        contract = str(item.get("proposed_contract") or "")
        for case_id in item.get("case_ids", []):
            case_id = str(case_id)
            if case_id not in allowed:
                rejected.append({"case_id": case_id, "contract": contract, "reason": "unknown_case"})
            elif contract not in allowed[case_id]:
                rejected.append({"case_id": case_id, "contract": contract, "reason": "contract_not_registered_for_case"})
            elif case_id in selected:
                rejected.append({"case_id": case_id, "contract": contract, "reason": "duplicate_case_plan"})
            else:
                selected[case_id] = contract
    for case_id in allowed:
        if case_id not in selected:
            rejected.append({"case_id": case_id, "contract": "review_hold", "reason": "case_omitted_by_planner"})
            selected[case_id] = "review_hold"
    return {"accepted": selected, "rejected": rejected, "guard_pass": not rejected}
