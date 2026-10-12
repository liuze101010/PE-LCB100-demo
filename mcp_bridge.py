from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from . import auto_orchestrator
from .snapshot import CASE_IDS, SOURCE_PATHS, public_tool_parameters


class DemoService:
    def __init__(
        self,
        snapshot_path: Path | str,
        project_root: Path | str | None = None,
        force_mode: str | None = None,
        *,
        allow_live: bool = True,
    ) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.snapshot = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        self.project_root = Path(project_root).resolve() if project_root else None
        self._mcp = None
        self.allow_live = allow_live

        connected_available = bool(
            self.project_root
            and (self.project_root / SOURCE_PATHS["mcp_server"]).exists()
        )
        if force_mode == "connected" and not connected_available:
            raise RuntimeError("Connected mode requested but MCP source is unavailable")
        if force_mode not in (None, "connected", "replay"):
            raise ValueError(f"Unsupported demo mode: {force_mode}")

        self.mode = (
            force_mode
            if force_mode
            else ("connected" if connected_available else "replay")
        )
        if self.mode == "connected":
            self._mcp = self._load_mcp()

    def _load_mcp(self):
        path = self.project_root / SOURCE_PATHS["mcp_server"]
        spec = importlib.util.spec_from_file_location(
            "polymer_agent_demo_connected_mcp", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load connected MCP source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def health(self) -> dict[str, Any]:
        capabilities = {"live_intent": False, "live_attribution": False, "real_smoke": False}
        if self.allow_live:
            from . import live_attribution, live_intent, live_smoke

            capabilities = {
                "live_intent": live_intent.llm_enabled(),
                "live_attribution": live_attribution.attribution_enabled(),
                "real_smoke": live_smoke.smoke_enabled(),
            }

        return {
            "status": "ok",
            "mode": self.mode,
            "read_only": True,
            "data_version": self.snapshot["data_version"],
            "schema_version": self.snapshot["schema_version"],
            **capabilities,
        }

    def overview(self) -> dict[str, Any]:
        return self.snapshot["overview"]

    def list_cases(self) -> list[dict[str, Any]]:
        return [
            {
                "case_id": case["case_id"],
                "short_id": case["short_id"],
                "title": case["title"],
                "accent": case["accent"],
                "warn_sources": case["rule"]["warn_sources"],
                "diagnosis": case["audit"]["final_diagnosis"],
                "route_module": case["route"]["module"],
                "outcome_status": case["outcome"]["status"],
            }
            for case in self.snapshot["cases"].values()
        ]

    def get_case(self, case_id: str) -> dict[str, Any]:
        if case_id not in CASE_IDS:
            raise KeyError(case_id)
        return self.snapshot["cases"][case_id]

    def audit_ledger(self) -> dict[str, Any]:
        records = list(self.snapshot.get("audit_ledger") or [])
        return {
            "schema_version": "polymer_agent_demo.audit_ledger.v1",
            "case_count": len(records),
            "model_vote_count": sum(
                len(record.get("model_votes") or []) for record in records
            ),
            "records": records,
        }

    @staticmethod
    def _connected_public_route(route: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": route["case_id"],
            "dry_run": True,
            "module": route.get("recommended_reentry_module"),
            "tool": route.get("repair_tool"),
            "parameters": public_tool_parameters(route["case_id"], route),
            "action": route.get("recommended_action"),
            "machine_route": route.get("repair_route"),
            "allowed_routes": [
                {
                    "diagnosis": option.get("diagnosis"),
                    "module": option.get("recommended_reentry_module"),
                    "tool": option.get("repair_tool"),
                    "parameters": public_tool_parameters(
                        route["case_id"],
                        {
                            "repair_tool": option.get("repair_tool"),
                            "tool_parameters": option.get("tool_parameters") or {},
                        },
                    ),
                    "machine_route": option.get("repair_route"),
                }
                for option in route.get("allowed_routes") or []
            ],
            "planning_selection_status": route.get("planning_selection_status"),
            "route_selected_by": route.get("route_selected_by"),
            "planning_selection_rationale": route.get(
                "planning_selection_rationale"
            ),
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

    def route_options(self, case_id: str) -> dict[str, Any]:
        case = self.get_case(case_id)
        if self.mode == "connected":
            raw = self._mcp.plan_repair_route_impl(case_id)
            if raw.get("status") != "ok":
                raise ValueError(raw.get("reason") or "route option lookup failed")
            options = self._connected_public_route(raw)["allowed_routes"]
        else:
            options = list(case["route"].get("allowed_routes") or [])
        return {
            "case_id": case_id,
            "dry_run": True,
            "stage": "route_options",
            "planning_selection_status": "awaiting_planning_model",
            "allowed_routes": options,
            "mode": self.mode,
            "source": (
                "live read-only MCP options"
                if self.mode == "connected"
                else "hashed frozen route options"
            ),
        }

    def select_route(
        self,
        case_id: str,
        *,
        repair_tool: str,
        parameters: dict[str, Any] | None = None,
        rationale: str,
        execute: bool = False,
    ) -> dict[str, Any]:
        if execute:
            raise PermissionError("Mutating requests are disabled in this demo")
        if not rationale.strip():
            raise ValueError("route selection requires a rationale")
        case = self.get_case(case_id)
        options_payload = self.route_options(case_id)
        allowed = {
            str(option.get("tool")): option
            for option in options_payload["allowed_routes"]
        }
        if repair_tool not in allowed:
            raise ValueError(f"route tool {repair_tool!r} is not allowed for this case")
        chosen = allowed[repair_tool]
        selected_parameters = dict(parameters or chosen.get("parameters") or {})
        if selected_parameters != dict(chosen.get("parameters") or {}):
            raise ValueError("route parameters do not match the registered allowlist")

        if self.mode == "connected":
            proposal = {
                "repair_tool": repair_tool,
                "parameters": selected_parameters,
                "rationale": rationale,
            }
            raw = self._mcp.plan_repair_route_impl(
                case_id, planning_proposal_json=json.dumps(proposal, ensure_ascii=False)
            )
            if raw.get("status") != "ok":
                raise ValueError(
                    raw.get("planning_rejection_reason")
                    or raw.get("reason")
                    or "route selection failed"
                )
            route = self._connected_public_route(raw)
        else:
            route = {
                "case_id": case_id,
                "dry_run": True,
                "module": chosen.get("module"),
                "tool": repair_tool,
                "parameters": selected_parameters,
                "action": case["route"].get("action")
                if repair_tool == case["route"].get("tool")
                else None,
                "machine_route": chosen.get("machine_route"),
                "allowed_routes": options_payload["allowed_routes"],
                "planning_selection_status": "selected",
                "route_selected_by": "planning_model",
                "planning_selection_rationale": rationale,
                "planning_guard": {"guard_verdict": "pass"},
                "guard": case["route"]["guard"],
            }
        route["stage"] = "route_selection"
        route["mode"] = self.mode
        route["source"] = (
            "live guarded MCP selection"
            if self.mode == "connected"
            else "guarded frozen replay selection"
        )
        return route

    def dry_run(self, case_id: str, execute: bool = False) -> dict[str, Any]:
        if execute:
            raise PermissionError("Mutating requests are disabled in this demo")
        case = self.get_case(case_id)
        return self.select_route(
            case_id,
            repair_tool=case["route"]["tool"],
            parameters=case["route"]["parameters"],
            rationale="Replay the route selected in the locked tri-model audit.",
            execute=False,
        )

    def auto_run(self, goal: str) -> dict[str, Any]:
        """Offline NL -> structured intent + replayable closed-loop timeline.

        Upgrade 2: in connected mode the MCP route step is recomputed live
        (read-only) per case rather than read from the frozen snapshot.
        """
        intent = auto_orchestrator.resolve_intent(goal, allow_live=self.allow_live)
        route_overrides = self._live_routes()
        corpus_view = auto_orchestrator.build_corpus_view(self.snapshot, intent)
        deep_ids = auto_orchestrator._deep_case_ids(self.snapshot, corpus_view)
        return {
            "goal": goal,
            "mode": self.mode,
            "route_source": (
                "live read-only MCP plan" if route_overrides else "hashed frozen replay"
            ),
            "intent": intent,
            "corpus_view": corpus_view,
            "timeline": auto_orchestrator.build_timeline(
                self.snapshot, intent, route_overrides, corpus_view
            ),
            "effect_table": auto_orchestrator.build_effect_table(self.snapshot, deep_ids),
            "evidence": auto_orchestrator.build_evidence_block(self.snapshot),
        }

    def _live_routes(self) -> dict[str, Any] | None:
        """Per-case live read-only MCP routes (connected mode only)."""
        if self.mode != "connected" or self._mcp is None:
            return None
        routes: dict[str, Any] = {}
        for case_id in self.snapshot["cases"].keys():
            routes[case_id] = self.dry_run(case_id)
        return routes

    def auto_report(self, goal: str, fmt: str) -> dict[str, Any]:
        """Build an exportable Demo-说明 report. Returns content/media_type/filename."""
        intent = auto_orchestrator.resolve_intent(goal, allow_live=self.allow_live)
        corpus_view = auto_orchestrator.build_corpus_view(self.snapshot, intent)
        return auto_orchestrator.build_report(
            self.snapshot, intent, fmt, mode=self.mode, corpus_view=corpus_view
        )

    def run_attribution(self, case_id: str) -> dict[str, Any]:
        """Live tri-model re-adjudication for one representative case (upgrade 4)."""
        if not self.allow_live:
            return {"enabled": False, "reason": "Offline replay uses the recorded audit votes."}
        from . import live_attribution

        if not live_attribution.attribution_enabled():
            return {
                "enabled": False,
                "reason": "live 三模型归因未开启（需 POLYMER_AGENT_LIVE_ATTRIBUTION=1 且 deepseek/qwen/glm 三家 key 均可用）",
            }
        case = self.snapshot["cases"].get(case_id)
        if case is None:
            raise KeyError(case_id)
        result = live_attribution.run_live_attribution(case)
        frozen = case["audit"]
        result["frozen"] = {
            "final_diagnosis_label": frozen["final_diagnosis_label"],
            "confidence": frozen["confidence"],
            "votes": [
                {
                    "client": vote["client"],
                    "diagnosis_label": vote["diagnosis_label"],
                    "confidence": vote["confidence"],
                }
                for vote in frozen["model_votes"]
            ],
        }
        result["enabled"] = True
        result["case_id"] = case_id
        return result

    def start_smoke(
        self,
        case_id: str,
        mode: str = "reproduce",
        confirm: bool = False,
        traj: str | None = None,
        bridge_dir: str | None = None,
    ) -> dict[str, Any]:
        """Trigger a real-compute smoke as a tracked background job (upgrade 5)."""
        if mode not in ("reproduce", "smoke"):
            raise ValueError(f"unsupported smoke mode: {mode!r}")
        # Remote submit is a write to shared compute -> require an explicit confirm.
        if mode == "smoke" and not confirm:
            raise PermissionError("remote smoke submit requires confirm=true")
        if not self.allow_live:
            return {"enabled": False, "reason": "Real execution is available in explicit platform mode."}
        from . import live_smoke

        if not live_smoke.smoke_enabled():
            return {
                "enabled": False,
                "reason": "真实 smoke 未开启（需 POLYMER_AGENT_REAL_SMOKE=1，且仓库内有 run_real_smoke.py）",
            }
        if case_id not in self.snapshot["cases"] and mode != "reproduce":
            raise KeyError(case_id)
        return live_smoke.start_job(mode=mode, case_id=case_id, traj=traj, bridge_dir=bridge_dir)

    def smoke_status(self, job_id: str) -> dict[str, Any]:
        if not self.allow_live:
            raise KeyError(job_id)
        from . import live_smoke

        status = live_smoke.get_status(job_id)
        if status is None:
            raise KeyError(job_id)
        return status
