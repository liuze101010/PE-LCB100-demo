#!/usr/bin/env python3
"""MCP server for one policy-free PE-LCB100 ReAct replay case."""

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

from pelcb100_policy_free_replay import PolicyFreeReplayState  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--tool-log", type=Path, required=True)
    return parser.parse_args()


ARGS = _parse_args()
REPLAY = PolicyFreeReplayState.from_file(ARGS.fixture, state_path=ARGS.state, tool_log_path=ARGS.tool_log)

mcp = FastMCP(
    "PE-LCB100 ordinary ReAct replay",
    instructions=(
        "This server exposes one frozen case through raw evidence and primitive operations. "
        "It provides no rule verdict, audit diagnosis, recommended route, or hidden reference decision."
    ),
)


@mcp.tool()
def start_case(case_id: str) -> dict[str, Any]:
    """Start the single case assigned to this isolated session."""
    return REPLAY.start_case(case_id)


@mcp.tool()
def inspect_outputs() -> dict[str, Any]:
    """Read the currently available raw simulation outputs."""
    return REPLAY.inspect_outputs()


@mcp.tool()
def inspect_run_configuration() -> dict[str, Any]:
    """Read current simulation settings and the globally registered runtime profiles."""
    return REPLAY.inspect_run_configuration()


@mcp.tool()
def query_reference_data(topic: str) -> dict[str, Any]:
    """Read raw reference information for density, MSD, or required outputs."""
    return REPLAY.query_reference_data(topic)


@mcp.tool()
def inspect_msd_diagnostics() -> dict[str, Any]:
    """Read the MSD fit statistics currently available for this case."""
    return REPLAY.inspect_msd_diagnostics()


@mcp.tool()
def rerun_md(engine_profile: str, pressure_verlet_buffer_tolerance: float | None = None) -> dict[str, Any]:
    """Request an MD rerun with chosen settings; archived evidence is returned when available."""
    return REPLAY.rerun_md(engine_profile, pressure_verlet_buffer_tolerance)


@mcp.tool()
def run_msd_analysis(window_strategy: str) -> dict[str, Any]:
    """Request a new MSD analysis strategy; archived evidence is returned when available."""
    return REPLAY.run_msd_analysis(window_strategy)


@mcp.tool()
def extend_trajectory(target_time_ns: float) -> dict[str, Any]:
    """Request a longer trajectory; archived evidence is returned when available."""
    return REPLAY.extend_trajectory(target_time_ns)


@mcp.tool()
def finish_case(decision: str, reason: str) -> dict[str, Any]:
    """Record PASS, REVIEW, or FAIL without revealing the hidden reference decision."""
    return REPLAY.finish_case(decision, reason)


def main() -> int:
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
