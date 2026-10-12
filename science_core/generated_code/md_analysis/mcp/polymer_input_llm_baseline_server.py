#!/usr/bin/env python3
"""Expose one blinded polymer input to a GPT validation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--tool-log", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


ARGS = _parse_args()
FIXTURE = _read_json(ARGS.fixture)
STATE: dict[str, Any] = {
    "assigned_case_id": str(FIXTURE["case_id"]),
    "started": False,
    "finished": False,
    "agent_decision": "",
    "agent_reason": "",
    "tool_call_count": 0,
    "invalid_calls": 0,
}
_write_json(ARGS.state, STATE)
ARGS.tool_log.parent.mkdir(parents=True, exist_ok=True)
ARGS.tool_log.write_text("", encoding="utf-8")


def _record(tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
    STATE["tool_call_count"] = int(STATE.get("tool_call_count", 0)) + 1
    _write_json(ARGS.state, STATE)
    with ARGS.tool_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"tool": tool, "arguments": arguments, "result": result},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _error(tool: str, arguments: dict[str, Any], message: str) -> dict[str, Any]:
    STATE["invalid_calls"] = int(STATE.get("invalid_calls", 0)) + 1
    result = {"status": "error", "message": message}
    _record(tool, arguments, result)
    return result


mcp = FastMCP(
    "Blinded polymer-input validation baseline",
    instructions=(
        "This server exposes one polymer input. Expected states, paired inputs, "
        "mutations and benchmark labels are never returned."
    ),
)


@mcp.tool()
def start_case(case_id: str) -> dict[str, Any]:
    """Start the single polymer-input case assigned to this isolated session."""
    arguments = {"case_id": case_id}
    if case_id != STATE["assigned_case_id"]:
        return _error("start_case", arguments, "case_id does not match this session")
    if STATE["finished"]:
        return _error("start_case", arguments, "case already finished")
    STATE["started"] = True
    result = {"status": "ok", "case_id": case_id}
    _record("start_case", arguments, result)
    return result


@mcp.tool()
def get_polymer_input() -> dict[str, Any]:
    """Read the assigned molecular design and the public validation definitions."""
    arguments: dict[str, Any] = {}
    if not STATE["started"]:
        return _error("get_polymer_input", arguments, "start_case must be called first")
    if STATE["finished"]:
        return _error("get_polymer_input", arguments, "case already finished")
    result = {
        "status": "ok",
        "case_id": FIXTURE["case_id"],
        "candidate": FIXTURE["candidate"],
        "decision_definitions": FIXTURE["decision_definitions"],
        "registered_capabilities": FIXTURE["registered_capabilities"],
        "validation_points": FIXTURE["validation_points"],
    }
    _record("get_polymer_input", arguments, result)
    return result


@mcp.tool()
def finish_case(decision: str, reason: str) -> dict[str, Any]:
    """Return PASS, REVIEW or REJECT for the assigned polymer input."""
    arguments = {"decision": decision, "reason": reason}
    if not STATE["started"]:
        return _error("finish_case", arguments, "start_case must be called first")
    if STATE["finished"]:
        return _error("finish_case", arguments, "finish_case may be called only once")
    normalized = str(decision).strip().upper()
    if normalized not in {"PASS", "REVIEW", "REJECT"}:
        return _error(
            "finish_case",
            arguments,
            "decision must be PASS, REVIEW or REJECT",
        )
    STATE["finished"] = True
    STATE["agent_decision"] = normalized
    STATE["agent_reason"] = str(reason).strip()
    result = {"status": "ok", "decision": normalized}
    _record("finish_case", arguments, result)
    return result


def main() -> int:
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
