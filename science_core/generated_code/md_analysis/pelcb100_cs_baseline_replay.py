#!/usr/bin/env python3
"""Blinded component-replacement replay for PE-LCB100 CS baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from pelcb100_policy_free_replay import PolicyFreeReplayState
from pelcb100_react_replay import ReplayState


SCHEMA_VERSION = "pelcb100_cs_component_replay.v2"
MODES = {
    "direct_judge",
    "llm_result_judge",
    "llm_repair_planner",
    "plan_and_execute",
    "self_critique_react",
}
RESULT_CONDITIONS = {
    "missing_output",
    "density",
    "msd",
    "strict_msd",
    "provenance_review",
}
DECISIONS = {"PASS", "REVIEW", "FAIL"}
JUDGEMENTS = {"PASS", "WARN", "FAIL"}
REPAIR_ACTIONS = {
    "rerun_md",
    "reanalyze_msd",
    "extend_trajectory",
    "open_review",
}
CRITERIA = {
    "required_outputs": [
        "density",
        "Rg",
        "RDF",
        "shape_anisotropy",
        "MSD",
        "Fsqt",
        "contact_probability",
    ],
    "density_g_cm3": {"minimum": 0.73, "maximum": 0.87},
    "msd_free_diffusion_window": {
        "loglog_slope_minimum": 0.95,
        "loglog_slope_maximum": 1.05,
        "linear_R2_minimum": 0.95,
        "local_D_cv_percent_maximum": 5.0,
        "window_start_fraction_minimum": 0.25,
        "window_end_fraction_maximum": 0.85,
        "window_width_fraction_minimum": 0.30,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_payload(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CSBaselineReplayState:
    mode: str
    policy_fixture: dict[str, Any]
    react_fixture: dict[str, Any]
    state_path: Path | None = None
    tool_log_path: Path | None = None
    started: bool = False
    finished: bool = False
    phase: str = "not_started"
    agent_decision: str = ""
    agent_reason: str = ""
    tool_call_count: int = 0
    invalid_calls: int = 0
    last_result_judgement: dict[str, Any] | None = None
    last_rule_check: dict[str, Any] | None = None
    authorized_routes: list[dict[str, Any]] = field(default_factory=list)
    submitted_plan: dict[str, Any] | None = None
    submitted_self_critique: dict[str, Any] | None = None
    policy_state: PolicyFreeReplayState = field(init=False)
    react_state: ReplayState = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.policy_fixture["case_id"] != self.react_fixture["case_id"]:
            raise ValueError("policy-free and rule-guided fixtures refer to different cases")
        self.policy_state = PolicyFreeReplayState(fixture=self.policy_fixture)
        self.react_state = ReplayState(fixture=self.react_fixture)

    @classmethod
    def from_files(
        cls,
        mode: str,
        policy_fixture_path: Path,
        react_fixture_path: Path,
        state_path: Path | None = None,
        tool_log_path: Path | None = None,
    ) -> "CSBaselineReplayState":
        return cls(
            mode=mode,
            policy_fixture=_load_json(policy_fixture_path),
            react_fixture=_load_json(react_fixture_path),
            state_path=state_path,
            tool_log_path=tool_log_path,
        )

    @property
    def case_id(self) -> str:
        return self.policy_fixture["case_id"]

    def _hidden_open_conditions(self) -> list[str]:
        if self.mode == "llm_repair_planner":
            return list(self.react_state.pending_conditions)
        return list(self.policy_state.evaluator_open_issues)

    def _result_actions_used(self) -> int:
        if self.mode == "llm_repair_planner":
            return int(self.react_state.result_changing_actions)
        return int(self.policy_state.result_changing_actions)

    def _public_state(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "case_id": self.case_id,
            "started": self.started,
            "finished": self.finished,
            "phase": self.phase,
            "agent_decision": self.agent_decision,
            "agent_reason": self.agent_reason,
            "tool_call_count": self.tool_call_count,
            "invalid_calls": self.invalid_calls,
            "result_changing_actions_used": self._result_actions_used(),
            "last_result_judgement": self.last_result_judgement,
            "last_rule_check": self.last_rule_check,
            "submitted_plan": self.submitted_plan,
            "submitted_self_critique": self.submitted_self_critique,
            "hidden_open_conditions_at_finish": (
                self._hidden_open_conditions() if self.finished else []
            ),
        }

    def _visible_evidence(self) -> dict[str, Any]:
        if self.mode == "llm_repair_planner":
            return {
                "initial_evidence": self.policy_fixture["public_initial_evidence"],
                "unlocked_results": list(self.react_state.unlocked_results),
                "last_rule_check": self.last_rule_check,
            }
        return {
            "design": self.policy_fixture["public_initial_evidence"]["design"],
            "current_output_package": dict(self.policy_state.current_output_package),
            "current_run_configuration": dict(self.policy_state.current_configuration),
            "latest_msd_record": dict(self.policy_state.latest_msd_record),
            "unlocked_results": list(self.policy_state.unlocked_evidence),
            "public_check_conditions": CRITERIA,
        }

    def _persist(self) -> None:
        if self.state_path:
            _write_json(self.state_path, self._public_state())

    def _record(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.tool_call_count += 1
        record = {
            "recorded_at": datetime.now().isoformat(timespec="milliseconds"),
            "case_id": self.case_id,
            "mode": self.mode,
            "tool": tool,
            "arguments": dict(arguments),
            "result": dict(result),
            "visible_evidence_sha256": _sha256_payload(self._visible_evidence()),
            "result_changing_actions_used": self._result_actions_used(),
        }
        if self.tool_log_path:
            _append_jsonl(self.tool_log_path, record)
        self._persist()
        return dict(result)

    def _error(
        self,
        tool: str,
        message: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.invalid_calls += 1
        return self._record(
            tool,
            arguments or {},
            {"status": "error", "message": message},
        )

    def _require_active(self, tool: str) -> dict[str, Any] | None:
        if not self.started:
            return self._error(tool, "Call start_case first.")
        if self.finished:
            return self._error(tool, "The case has already been finished.")
        return None

    def start_case(self, case_id: str) -> dict[str, Any]:
        if self.started:
            return self._error(
                "start_case",
                "This isolated session already has an active case.",
                {"case_id": case_id},
            )
        if case_id != self.case_id:
            return self._error(
                "start_case",
                "The requested case is unavailable in this isolated session.",
                {"case_id": case_id},
            )
        self.started = True
        self.phase = "initial_evidence"
        if self.mode == "llm_repair_planner":
            self.react_state.start_case(case_id)
        else:
            self.policy_state.start_case(case_id)
        return self._record(
            "start_case",
            {"case_id": case_id},
            {
                "status": "ok",
                "case_id": case_id,
                "mode": self.mode,
                "maximum_result_changing_actions": 2,
            },
        )

    def get_current_evidence(self) -> dict[str, Any]:
        blocked = self._require_active("get_current_evidence")
        if blocked:
            return blocked
        return self._record(
            "get_current_evidence",
            {},
            {
                "status": "ok",
                "case_id": self.case_id,
                "evidence": self._visible_evidence(),
            },
        )

    def submit_execution_plan(
        self,
        steps: list[str],
        reason: str,
    ) -> dict[str, Any]:
        arguments = {"steps": steps, "reason": reason}
        blocked = self._require_active("submit_execution_plan")
        if blocked:
            return blocked
        if self.mode != "plan_and_execute":
            return self._error(
                "submit_execution_plan",
                "This tool is unavailable in the current baseline.",
                arguments,
            )
        normalized_steps = [str(item).strip() for item in steps if str(item).strip()]
        if not normalized_steps or not str(reason).strip():
            return self._error(
                "submit_execution_plan",
                "At least one plan step and a concise reason are required.",
                arguments,
            )
        self.submitted_plan = {
            "steps": normalized_steps,
            "reason": str(reason).strip(),
        }
        self.phase = "execution_plan_submitted"
        return self._record(
            "submit_execution_plan",
            arguments,
            {
                "status": "ok",
                "plan_recorded": True,
                "message": "Execute the plan with the available evidence and primitive tools.",
            },
        )

    def submit_self_critique(
        self,
        draft_decision: str,
        unresolved_risks: list[str],
        reason: str,
    ) -> dict[str, Any]:
        arguments = {
            "draft_decision": draft_decision,
            "unresolved_risks": unresolved_risks,
            "reason": reason,
        }
        blocked = self._require_active("submit_self_critique")
        if blocked:
            return blocked
        if self.mode != "self_critique_react":
            return self._error(
                "submit_self_critique",
                "This tool is unavailable in the current baseline.",
                arguments,
            )
        normalized = str(draft_decision).strip().upper()
        if normalized not in DECISIONS:
            return self._error(
                "submit_self_critique",
                "draft_decision must be PASS, REVIEW, or FAIL.",
                arguments,
            )
        if not str(reason).strip():
            return self._error(
                "submit_self_critique",
                "A concise critique is required.",
                arguments,
            )
        self.submitted_self_critique = {
            "draft_decision": normalized,
            "unresolved_risks": [
                str(item).strip() for item in unresolved_risks if str(item).strip()
            ],
            "reason": str(reason).strip(),
        }
        self.phase = "self_critique_submitted"
        return self._record(
            "submit_self_critique",
            arguments,
            {
                "status": "ok",
                "critique_recorded": True,
                "message": "Use this self-check to make the final decision.",
            },
        )

    def _require_generic_agent_mode(
        self,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        result_changing: bool = False,
    ) -> dict[str, Any] | None:
        blocked = self._require_active(tool)
        if blocked:
            return blocked
        if self.mode not in {"plan_and_execute", "self_critique_react"}:
            return self._error(
                tool,
                "This primitive tool is unavailable in the current baseline.",
                arguments,
            )
        if result_changing and self.mode == "plan_and_execute" and not self.submitted_plan:
            return self._error(
                tool,
                "Submit an execution plan before a result-changing operation.",
                arguments,
            )
        return None

    def _record_policy_result(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        if result.get("status") != "ok":
            self.invalid_calls += 1
        self.phase = str(self.policy_state.phase)
        return self._record(tool, arguments, result)

    def inspect_outputs(self) -> dict[str, Any]:
        blocked = self._require_generic_agent_mode("inspect_outputs")
        if blocked:
            return blocked
        return self._record_policy_result(
            "inspect_outputs",
            {},
            self.policy_state.inspect_outputs(),
        )

    def inspect_run_configuration(self) -> dict[str, Any]:
        blocked = self._require_generic_agent_mode("inspect_run_configuration")
        if blocked:
            return blocked
        return self._record_policy_result(
            "inspect_run_configuration",
            {},
            self.policy_state.inspect_run_configuration(),
        )

    def query_reference_data(self, topic: str) -> dict[str, Any]:
        arguments = {"topic": topic}
        blocked = self._require_generic_agent_mode("query_reference_data", arguments)
        if blocked:
            return blocked
        return self._record_policy_result(
            "query_reference_data",
            arguments,
            self.policy_state.query_reference_data(topic),
        )

    def inspect_msd_diagnostics(self) -> dict[str, Any]:
        blocked = self._require_generic_agent_mode("inspect_msd_diagnostics")
        if blocked:
            return blocked
        return self._record_policy_result(
            "inspect_msd_diagnostics",
            {},
            self.policy_state.inspect_msd_diagnostics(),
        )

    def rerun_md(
        self,
        engine_profile: str,
        pressure_verlet_buffer_tolerance: float | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "engine_profile": engine_profile,
            "pressure_verlet_buffer_tolerance": pressure_verlet_buffer_tolerance,
        }
        blocked = self._require_generic_agent_mode(
            "rerun_md",
            arguments,
            result_changing=True,
        )
        if blocked:
            return blocked
        return self._record_policy_result(
            "rerun_md",
            arguments,
            self.policy_state.rerun_md(
                engine_profile,
                pressure_verlet_buffer_tolerance,
            ),
        )

    def run_msd_analysis(self, window_strategy: str) -> dict[str, Any]:
        arguments = {"window_strategy": window_strategy}
        blocked = self._require_generic_agent_mode(
            "run_msd_analysis",
            arguments,
            result_changing=True,
        )
        if blocked:
            return blocked
        return self._record_policy_result(
            "run_msd_analysis",
            arguments,
            self.policy_state.run_msd_analysis(window_strategy),
        )

    def extend_trajectory(self, target_time_ns: float) -> dict[str, Any]:
        arguments = {"target_time_ns": target_time_ns}
        blocked = self._require_generic_agent_mode(
            "extend_trajectory",
            arguments,
            result_changing=True,
        )
        if blocked:
            return blocked
        return self._record_policy_result(
            "extend_trajectory",
            arguments,
            self.policy_state.extend_trajectory(target_time_ns),
        )

    def _audited_routes(self, conditions: list[str]) -> list[dict[str, Any]]:
        routes: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            if condition in {"missing_output", "density"}:
                expected = self.policy_fixture["raw_history"]["clean_configuration"]
                route = {
                    "operation": "rerun_md",
                    "parameters": {
                        "engine_profile": expected["engine_profile"],
                        "pressure_verlet_buffer_tolerance": expected[
                            "pressure_verlet_buffer_tolerance"
                        ],
                    },
                    "addresses": [condition],
                }
            elif condition == "msd":
                route = {
                    "operation": "run_msd_analysis",
                    "parameters": {"window_strategy": "bounded_window_scan"},
                    "addresses": [condition],
                }
            elif condition == "strict_msd":
                route = {
                    "operation": "extend_trajectory",
                    "parameters": {"target_time_ns": 800.0},
                    "addresses": [condition],
                }
            else:
                route = {
                    "operation": "open_review",
                    "parameters": {"review_kind": "provenance_review"},
                    "addresses": [condition],
                }
            key = route["operation"] + json.dumps(route["parameters"], sort_keys=True)
            if key in routes:
                routes[key]["addresses"] = sorted(
                    set(routes[key]["addresses"] + route["addresses"])
                )
            else:
                routes[key] = route
        return list(routes.values())

    def submit_result_judgement(
        self,
        verdict: str,
        open_conditions: list[str],
        reason: str,
    ) -> dict[str, Any]:
        arguments = {
            "verdict": verdict,
            "open_conditions": open_conditions,
            "reason": reason,
        }
        blocked = self._require_active("submit_result_judgement")
        if blocked:
            return blocked
        if self.mode != "llm_result_judge":
            return self._error(
                "submit_result_judgement",
                "This tool is unavailable in the current baseline.",
                arguments,
            )
        normalized = str(verdict).strip().upper()
        conditions = sorted(set(str(item).strip() for item in open_conditions if str(item).strip()))
        if normalized not in JUDGEMENTS:
            return self._error(
                "submit_result_judgement",
                "verdict must be PASS, WARN, or FAIL.",
                arguments,
            )
        if any(item not in RESULT_CONDITIONS for item in conditions):
            return self._error(
                "submit_result_judgement",
                "open_conditions contains an unsupported value.",
                arguments,
            )
        if normalized == "PASS" and conditions:
            return self._error(
                "submit_result_judgement",
                "PASS requires an empty open_conditions list.",
                arguments,
            )
        if normalized == "WARN" and not conditions:
            return self._error(
                "submit_result_judgement",
                "WARN requires at least one open condition.",
                arguments,
            )
        if not str(reason).strip():
            return self._error(
                "submit_result_judgement",
                "A concise scientific reason is required.",
                arguments,
            )
        self.last_result_judgement = {
            "verdict": normalized,
            "open_conditions": conditions,
            "reason": str(reason).strip(),
        }
        self.authorized_routes = self._audited_routes(conditions) if normalized == "WARN" else []
        self.phase = "result_judged"
        return self._record(
            "submit_result_judgement",
            arguments,
            {
                "status": "ok",
                "recorded_verdict": normalized,
                "audit_summary": (
                    {
                        "status": "WARN accepted for cause review",
                        "allowed_routes": self.authorized_routes,
                    }
                    if normalized == "WARN"
                    else {"status": "no audit route opened"}
                ),
            },
        )

    def execute_audited_route(
        self,
        operation: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = dict(parameters or {})
        arguments = {"operation": operation, "parameters": parameters}
        blocked = self._require_active("execute_audited_route")
        if blocked:
            return blocked
        if self.mode != "llm_result_judge":
            return self._error(
                "execute_audited_route",
                "This tool is unavailable in the current baseline.",
                arguments,
            )
        match = next(
            (
                item
                for item in self.authorized_routes
                if item["operation"] == operation and item["parameters"] == parameters
            ),
            None,
        )
        if not match:
            return self._error(
                "execute_audited_route",
                "Use one operation and the exact parameters returned by the retained audit layer.",
                arguments,
            )
        if operation == "rerun_md":
            result = self.policy_state.rerun_md(**parameters)
        elif operation == "run_msd_analysis":
            result = self.policy_state.run_msd_analysis(**parameters)
        elif operation == "extend_trajectory":
            result = self.policy_state.extend_trajectory(**parameters)
        else:
            result = {
                "status": "ok",
                "operation": "open_review",
                "review_packet_opened": True,
            }
        if result.get("status") != "ok":
            self.invalid_calls += 1
        self.authorized_routes = []
        self.last_result_judgement = None
        self.phase = "route_executed"
        return self._record("execute_audited_route", arguments, result)

    def run_result_checks(self) -> dict[str, Any]:
        blocked = self._require_active("run_result_checks")
        if blocked:
            return blocked
        if self.mode != "llm_repair_planner":
            return self._error(
                "run_result_checks",
                "This tool is unavailable in the current baseline.",
            )
        result = self.react_state.run_result_checks()
        self.last_rule_check = result
        self.phase = "rule_check_complete"
        return self._record("run_result_checks", {}, result)

    def rerun_result_checks(self) -> dict[str, Any]:
        blocked = self._require_active("rerun_result_checks")
        if blocked:
            return blocked
        if self.mode != "llm_repair_planner":
            return self._error(
                "rerun_result_checks",
                "This tool is unavailable in the current baseline.",
            )
        result = self.react_state.rerun_result_checks()
        self.last_rule_check = result
        self.phase = "rule_recheck_complete"
        return self._record("rerun_result_checks", {}, result)

    @staticmethod
    def _action_matches_route(action: str, route: Mapping[str, Any]) -> bool:
        name = route.get("route")
        parameters = route.get("parameters", {})
        if action == "rerun_md":
            return name == "submit_md_clean_engine"
        if action == "reanalyze_msd":
            return (
                name == "run_postprocess_repair"
                and parameters.get("repair_kind") == "msd_window"
            )
        if action == "extend_trajectory":
            return (
                name == "run_postprocess_repair"
                and parameters.get("repair_kind") == "msd_bounded_extension"
            )
        return action == "open_review" and name == "open_review_packet"

    def submit_cause_and_route(
        self,
        diagnosis: str,
        target_condition: str,
        action: str,
        reason: str,
    ) -> dict[str, Any]:
        arguments = {
            "diagnosis": diagnosis,
            "target_condition": target_condition,
            "action": action,
            "reason": reason,
        }
        blocked = self._require_active("submit_cause_and_route")
        if blocked:
            return blocked
        if self.mode != "llm_repair_planner":
            return self._error(
                "submit_cause_and_route",
                "This tool is unavailable in the current baseline.",
                arguments,
            )
        if not self.last_rule_check or self.last_rule_check.get("verdict") != "WARN":
            return self._error(
                "submit_cause_and_route",
                "Run the retained result checks and obtain WARN first.",
                arguments,
            )
        if action not in REPAIR_ACTIONS:
            return self._error(
                "submit_cause_and_route",
                "action must be rerun_md, reanalyze_msd, extend_trajectory, or open_review.",
                arguments,
            )
        if target_condition not in self.react_state.pending_conditions:
            return self._error(
                "submit_cause_and_route",
                "target_condition is not currently open.",
                arguments,
            )
        if not str(diagnosis).strip() or not str(reason).strip():
            return self._error(
                "submit_cause_and_route",
                "diagnosis and reason are required.",
                arguments,
            )
        expected_route = self.react_state._route_for(target_condition)
        if not self._action_matches_route(action, expected_route):
            return self._error(
                "submit_cause_and_route",
                "The selected action does not address the stated condition.",
                arguments,
            )
        self.react_state.authorized_routes.append(dict(expected_route))
        result = self.react_state.execute_registered_route(
            expected_route["route"],
            expected_route["parameters"],
        )
        if result.get("status") != "ok":
            self.invalid_calls += 1
        self.last_rule_check = None
        self.phase = "llm_route_executed"
        return self._record(
            "submit_cause_and_route",
            arguments,
            {
                "status": result.get("status"),
                "recorded_diagnosis": str(diagnosis).strip(),
                "selected_action": action,
                "route_result": result,
            },
        )

    def finish_case(self, decision: str, reason: str) -> dict[str, Any]:
        arguments = {"decision": decision, "reason": reason}
        blocked = self._require_active("finish_case")
        if blocked:
            return blocked
        normalized = str(decision).strip().upper()
        if normalized not in DECISIONS:
            return self._error(
                "finish_case",
                "decision must be PASS, REVIEW, or FAIL.",
                arguments,
            )
        if not str(reason).strip():
            return self._error(
                "finish_case",
                "A concise reason is required.",
                arguments,
            )
        if self.mode == "plan_and_execute" and not self.submitted_plan:
            return self._error(
                "finish_case",
                "Submit an execution plan before finishing the case.",
                arguments,
            )
        if self.mode == "self_critique_react" and not self.submitted_self_critique:
            return self._error(
                "finish_case",
                "Submit a self-critique before finishing the case.",
                arguments,
            )
        self.finished = True
        self.agent_decision = normalized
        self.agent_reason = str(reason).strip()
        self.phase = "finished"
        result = {
            "status": "ok",
            "case_id": self.case_id,
            "recorded_decision": normalized,
            "message": "Decision recorded. The hidden execution reference is not returned.",
        }
        return self._record("finish_case", arguments, result)
