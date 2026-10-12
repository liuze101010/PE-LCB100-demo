#!/usr/bin/env python3
"""Mode-specific MCP server for one blinded PE-LCB100 CS baseline case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


SCRIPT_DIR = Path(__file__).resolve().parent
MD_ANALYSIS_DIR = SCRIPT_DIR.parent
if str(MD_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(MD_ANALYSIS_DIR))

from pelcb100_cs_baseline_replay import CSBaselineReplayState  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--policy-fixture", type=Path, required=True)
    parser.add_argument("--react-fixture", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--tool-log", type=Path, required=True)
    return parser.parse_args()


ARGS = _parse_args()
REPLAY = CSBaselineReplayState.from_files(
    mode=ARGS.mode,
    policy_fixture_path=ARGS.policy_fixture,
    react_fixture_path=ARGS.react_fixture,
    state_path=ARGS.state,
    tool_log_path=ARGS.tool_log,
)

mcp = FastMCP(
    "PE-LCB100 component-replacement baseline",
    instructions=(
        "This server exposes one blinded historical case. It only registers tools "
        "needed by the selected baseline and never returns the hidden final state."
    ),
)


@mcp.tool()
def start_case(case_id: str) -> dict[str, Any]:
    """Start the single case assigned to this isolated session."""
    return REPLAY.start_case(case_id)


if ARGS.mode not in {"plan_and_execute", "self_critique_react"}:

    @mcp.tool()
    def get_current_evidence() -> dict[str, Any]:
        """Read raw evidence and any results unlocked by earlier operations."""
        return REPLAY.get_current_evidence()


if ARGS.mode in {"plan_and_execute", "self_critique_react"}:

    @mcp.tool()
    def inspect_outputs() -> dict[str, Any]:
        """Read the currently available raw simulation outputs."""
        return REPLAY.inspect_outputs()

    @mcp.tool()
    def inspect_run_configuration() -> dict[str, Any]:
        """Read current settings and registered runtime and analysis options."""
        return REPLAY.inspect_run_configuration()

    @mcp.tool()
    def query_reference_data(topic: str) -> dict[str, Any]:
        """Read raw reference information for density, MSD, or required outputs."""
        return REPLAY.query_reference_data(topic)

    @mcp.tool()
    def inspect_msd_diagnostics() -> dict[str, Any]:
        """Read the currently available MSD fit statistics."""
        return REPLAY.inspect_msd_diagnostics()

    @mcp.tool()
    def rerun_md(
        engine_profile: str,
        pressure_verlet_buffer_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Request an MD rerun using one registered runtime profile."""
        return REPLAY.rerun_md(
            engine_profile,
            pressure_verlet_buffer_tolerance,
        )

    @mcp.tool()
    def run_msd_analysis(window_strategy: str) -> dict[str, Any]:
        """Request a new MSD analysis using a registered strategy."""
        return REPLAY.run_msd_analysis(window_strategy)

    @mcp.tool()
    def extend_trajectory(target_time_ns: float) -> dict[str, Any]:
        """Request a longer trajectory when an archived result is available."""
        return REPLAY.extend_trajectory(target_time_ns)


if ARGS.mode == "plan_and_execute":

    @mcp.tool()
    def submit_execution_plan(
        steps: list[str],
        reason: str,
    ) -> dict[str, Any]:
        """Record the evidence checks, operations, and stopping rule before execution."""
        return REPLAY.submit_execution_plan(steps, reason)


if ARGS.mode == "self_critique_react":

    @mcp.tool()
    def submit_self_critique(
        draft_decision: str,
        unresolved_risks: list[str],
        reason: str,
    ) -> dict[str, Any]:
        """Review the draft decision and unresolved scientific risks before finishing."""
        return REPLAY.submit_self_critique(
            draft_decision,
            unresolved_risks,
            reason,
        )


if ARGS.mode == "llm_result_judge":

    @mcp.tool()
    def submit_result_judgement(
        verdict: str,
        open_conditions: list[str],
        reason: str,
    ) -> dict[str, Any]:
        """Replace the rule layer by reporting PASS, WARN, or FAIL and open conditions."""
        return REPLAY.submit_result_judgement(verdict, open_conditions, reason)

    @mcp.tool()
    def execute_audited_route(
        operation: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one exact route returned by the retained audit layer."""
        return REPLAY.execute_audited_route(operation, parameters)


if ARGS.mode == "llm_repair_planner":

    @mcp.tool()
    def run_result_checks() -> dict[str, Any]:
        """Run the retained deterministic result checks."""
        return REPLAY.run_result_checks()

    @mcp.tool()
    def submit_cause_and_route(
        diagnosis: str,
        target_condition: str,
        action: str,
        reason: str,
    ) -> dict[str, Any]:
        """Replace the audit layer by diagnosing one open condition and selecting an action."""
        return REPLAY.submit_cause_and_route(
            diagnosis,
            target_condition,
            action,
            reason,
        )

    @mcp.tool()
    def rerun_result_checks() -> dict[str, Any]:
        """Repeat the retained checks after a selected action returns new evidence."""
        return REPLAY.rerun_result_checks()


@mcp.tool()
def finish_case(decision: str, reason: str) -> dict[str, Any]:
    """Record PASS, REVIEW, or FAIL without revealing the hidden execution reference."""
    return REPLAY.finish_case(decision, reason)


def main() -> int:
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
