#!/usr/bin/env python3
"""Run GPT-5.5 component-replacement baselines on blinded PE-LCB100 replay cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_ROOT = SCRIPT_DIR / "output" / "pelcb100_policy_free_react_baseline"
REACT_ROOT = SCRIPT_DIR / "output" / "pelcb100_react_baseline"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "pelcb100_cs_baselines"
SERVER = SCRIPT_DIR / "mcp" / "pelcb100_cs_baseline_server.py"
PILOT_CASES = POLICY_ROOT / "pilot_12_cases.json"
POLICY_INDEX = POLICY_ROOT / "fixture_index.csv"
REACT_INDEX = REACT_ROOT / "fixture_index.csv"
SCHEMA_VERSION = "pelcb100_cs_baseline_runner.v1"

ARM_TOOLS = {
    "direct_judge": {
        "start_case",
        "get_current_evidence",
        "finish_case",
    },
    "llm_result_judge": {
        "start_case",
        "get_current_evidence",
        "submit_result_judgement",
        "execute_audited_route",
        "finish_case",
    },
    "llm_repair_planner": {
        "start_case",
        "get_current_evidence",
        "run_result_checks",
        "submit_cause_and_route",
        "rerun_result_checks",
        "finish_case",
    },
    "plan_and_execute": {
        "start_case",
        "inspect_outputs",
        "inspect_run_configuration",
        "query_reference_data",
        "inspect_msd_diagnostics",
        "submit_execution_plan",
        "rerun_md",
        "run_msd_analysis",
        "extend_trajectory",
        "finish_case",
    },
    "self_critique_react": {
        "start_case",
        "inspect_outputs",
        "inspect_run_configuration",
        "query_reference_data",
        "inspect_msd_diagnostics",
        "rerun_md",
        "run_msd_analysis",
        "extend_trajectory",
        "submit_self_critique",
        "finish_case",
    },
}
FUTURE_LEAK_TOKENS = (
    "reference_decision",
    "reference_publication_state",
    "publication_state",
    "effective_publication_grade",
    "hidden_open_conditions",
    "terminal state",
    "92 pass",
    "8 review",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"type": "unparseable_event", "raw": line})
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    return "[" + ",".join(_toml_string(value) for value in values) + "]"


def build_prompt(arm: str, case_id: str, case_short: str) -> str:
    if arm in {"plan_and_execute", "self_critique_react"}:
        start_instruction = (
            "Start with `start_case`, inspect the raw evidence with the primitive tools, "
            "and call `finish_case` exactly once."
        )
    else:
        start_instruction = (
            "Start with `start_case`, use `get_current_evidence`, "
            "and call `finish_case` exactly once."
        )
    common = f"""You are GPT-5.5 in a blinded polymer molecular-dynamics Agent benchmark.

Process only case {case_id} ({case_short}) with tools from the `component_replay` MCP server. {start_instruction} PASS means every required scientific condition is satisfied. REVIEW means the visible evidence or the two-action limit cannot support release. FAIL is reserved for an explicitly unsupported calculation.

Do not use shell commands, file search, web access, code execution, or tools outside `component_replay`. Any such call invalidates the run. Never guess hidden final labels or invent unavailable results. After `finish_case`, reply with one short sentence and stop.
"""
    if arm == "direct_judge":
        specific = """
This is a direct LLM-judge baseline. You receive the raw output package and public scientific conditions, but no result-checking, diagnosis, or repair tool. Read the evidence and make one terminal decision. Missing statistics or unresolved scientific conditions require REVIEW rather than PASS.
"""
    elif arm == "llm_result_judge":
        specific = """
The deterministic result-checking layer has been removed. You must replace it by calling `submit_result_judgement` with PASS, WARN, or FAIL and the exact open conditions supported by the visible values. Allowed conditions are missing_output, density, msd, strict_msd, and provenance_review. Use the public density, required-output, and MSD conditions literally.

When you report WARN, the retained audit layer returns allowed routes. You may execute an exact returned route, inspect the new evidence, and judge the result again. At most two result-changing operations are available. Do not finish PASS unless your latest result judgement is PASS.
"""
    elif arm == "llm_repair_planner":
        specific = """
The deterministic result checks remain active, but the audit layer has been removed. Start by calling `run_result_checks`. When it returns WARN, you must diagnose one open condition and call `submit_cause_and_route` yourself.

The registered actions are: rerun_md for simulation-setting or missing-output problems; reanalyze_msd for a fitting-window or analysis problem; extend_trajectory when the available trajectory does not establish free diffusion; and open_review when evidence remains insufficient or requires human review. After each successful action, call `rerun_result_checks`. At most two result-changing operations are available. Finish PASS only after the retained checks return PASS.
"""
    elif arm == "plan_and_execute":
        specific = """
This is a standard Plan-and-Execute baseline. The rule and audit layers are absent. Inspect the raw outputs, current settings, and reference data needed for this case. Then call `submit_execution_plan` before any result-changing operation. The plan must state the evidence checks, any primitive operation, and the stopping condition. Execute the plan with the primitive tools and revise it when new evidence changes the situation.

At most two result-changing operations are available. The tools never return a scientific verdict, cause diagnosis, recommended route, or hidden final state. Use the visible values to decide whether every required output, density condition, and MSD condition is satisfied.
"""
    elif arm == "self_critique_react":
        specific = """
This is a ReAct Agent with a mandatory final self-critique. The rule and audit layers are absent. Inspect evidence, choose primitive operations, and decide when to stop. At most two result-changing operations are available. The tools never return a scientific verdict, cause diagnosis, recommended route, or hidden final state.

Before `finish_case`, call `submit_self_critique` with your draft decision, every unresolved scientific risk, and a concise assessment of whether the visible evidence supports release. Use that critique to make the final decision.
"""
    else:
        raise ValueError(f"unknown arm: {arm}")
    return common + specific


def _event_item(event: Mapping[str, Any]) -> Mapping[str, Any]:
    item = event.get("item")
    return item if isinstance(item, dict) else {}


def inspect_events(events: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    violations: list[str] = []
    runtime_errors: list[str] = []
    calls: list[dict[str, Any]] = []
    reasoning = ""
    reasoning_by_call: list[str] = []
    usage: dict[str, int] = {}
    allowed = ARM_TOOLS[arm]
    for event in events:
        event_type = str(event.get("type", ""))
        item = _event_item(event)
        item_type = str(item.get("type", ""))
        if item_type == "reasoning" and item.get("text"):
            reasoning = str(item["text"])
        if item_type == "mcp_tool_call" and event_type.endswith("completed"):
            server = str(item.get("server", ""))
            tool = str(item.get("tool", item.get("name", "")))
            calls.append({"server": server, "tool": tool})
            reasoning_by_call.append(reasoning)
            if server != "component_replay" or tool not in allowed:
                violations.append(f"unregistered_mcp:{server}:{tool}")
        elif item_type in {
            "command_execution",
            "file_change",
            "web_search",
            "image_generation",
            "browser_action",
        }:
            violations.append(f"forbidden_builtin_tool:{item_type}")
        if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
            for key, value in event["usage"].items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        if event_type == "error":
            runtime_errors.append(str(event.get("message", "runtime error")))
        if event_type == "turn.failed":
            violations.append("runtime_event:turn.failed")
        if event_type == "unparseable_event":
            violations.append("unparseable_event")
    return {
        "violations": sorted(set(violations)),
        "calls": calls,
        "reasoning_by_call": reasoning_by_call,
        "usage": usage,
        "runtime_errors": runtime_errors,
    }


def _future_leaks(tool_log: list[dict[str, Any]]) -> list[str]:
    leaks: list[str] = []
    for index, row in enumerate(tool_log, start=1):
        if row.get("tool") == "finish_case":
            continue
        text = json.dumps(row.get("result", {}), ensure_ascii=False).lower()
        for token in FUTURE_LEAK_TOKENS:
            if token in text:
                leaks.append(f"step{index}:{token}")
    return leaks


def _fixture_maps() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    policy = {row["case_id"]: row for row in _read_csv(POLICY_INDEX)}
    react = {row["case_id"]: row for row in _read_csv(REACT_INDEX)}
    if set(policy) != set(react) or len(policy) != 100:
        raise ValueError("component baseline fixture indexes do not contain the same 100 cases")
    return policy, react


def selected_cases(
    scope: str,
    requested_cases: Iterable[str] = (),
) -> list[dict[str, str]]:
    policy, _ = _fixture_maps()
    if scope == "all":
        cases = [
            {"case_id": case_id, "case_short": row["case_short"], "stratum": "all"}
            for case_id, row in policy.items()
        ]
    else:
        pilot = _read_json(PILOT_CASES)["cases"]
        cases = [dict(row) for row in pilot]

    requested = {str(item).strip() for item in requested_cases if str(item).strip()}
    if not requested:
        return cases
    selected = [
        case
        for case in cases
        if case["case_id"] in requested or case["case_short"] in requested
    ]
    matched = {
        value
        for case in selected
        for value in (case["case_id"], case["case_short"])
        if value in requested
    }
    unknown = sorted(requested - matched)
    if unknown:
        raise ValueError(f"unknown case identifiers: {', '.join(unknown)}")
    return selected


def existing_case_results(run_dir: Path) -> list[dict[str, Any]]:
    cases_dir = run_dir / "cases"
    if not cases_dir.exists():
        return []
    return [
        _read_json(path)
        for path in sorted(cases_dir.glob("*/case_result.json"))
    ]


def run_one_case(
    arm: str,
    case: Mapping[str, str],
    run_dir: Path,
    model: str,
    timeout_s: int,
) -> dict[str, Any]:
    case_id = case["case_id"]
    case_short = case["case_short"]
    policy_index, react_index = _fixture_maps()
    policy_fixture_path = Path(policy_index[case_id]["fixture_path"])
    react_fixture_path = Path(react_index[case_id]["fixture_path"])

    case_dir = run_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    state_path = case_dir / "replay_state.json"
    tool_log_path = case_dir / "replay_tool_log.jsonl"
    event_log_path = case_dir / "codex_events.jsonl"
    stderr_path = case_dir / "codex_stderr.txt"
    last_message_path = case_dir / "codex_last_message.txt"
    for stale_path in (
        state_path,
        tool_log_path,
        event_log_path,
        stderr_path,
        last_message_path,
        case_dir / "case_result.json",
    ):
        stale_path.unlink(missing_ok=True)
    prompt = build_prompt(arm, case_id, case_short)
    (case_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    server_args = [
        str(SERVER),
        "--mode",
        arm,
        "--policy-fixture",
        str(policy_fixture_path),
        "--react-fixture",
        str(react_fixture_path),
        "--state",
        str(state_path),
        "--tool-log",
        str(tool_log_path),
    ]
    command = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        "--color",
        "never",
        "--model",
        model,
        "--dangerously-bypass-approvals-and-sandbox",
        "--cd",
        str(case_dir),
        "--output-last-message",
        str(last_message_path),
        "-c",
        f"mcp_servers.component_replay.command={_toml_string(sys.executable)}",
        "-c",
        f"mcp_servers.component_replay.args={_toml_array(server_args)}",
        "-c",
        "mcp_servers.component_replay.startup_timeout_sec=120",
        "-c",
        'approval_policy="never"',
        prompt,
    ]
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_s,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        returncode = 124
        timed_out = True
    runtime_s = time.perf_counter() - started
    event_log_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    events = _read_jsonl(event_log_path)
    event_info = inspect_events(events, arm)
    tool_log = _read_jsonl(tool_log_path)
    state = _read_json(state_path) if state_path.exists() else {}
    policy_fixture = _read_json(policy_fixture_path)

    violations = list(event_info["violations"])
    violations.extend(f"future_evidence_leak:{item}" for item in _future_leaks(tool_log))
    if timed_out:
        violations.append("timeout")
    if returncode != 0:
        violations.append(f"codex_returncode:{returncode}")
    if not state.get("finished"):
        violations.append("finish_case_not_called")
    if len(event_info["calls"]) != len(tool_log):
        violations.append("mcp_event_tool_log_count_mismatch")
    violations = sorted(set(violations))

    reference = policy_fixture["private_reference"]["reference_decision"]
    initial_conditions = policy_fixture["private_reference"]["initial_conditions"]
    decision = str(state.get("agent_decision", ""))
    hidden_open = state.get("hidden_open_conditions_at_finish", [])
    if not isinstance(hidden_open, list):
        hidden_open = []
    all_closed = not hidden_open
    recoverable = reference == "PASS" and bool(initial_conditions)
    usage = event_info["usage"]
    result = {
        "arm": arm,
        "case_id": case_id,
        "case_short": case_short,
        "stratum": case.get("stratum", ""),
        "model": model,
        "valid_run": not violations,
        "violations": "|".join(violations),
        "reference_decision": reference,
        "agent_decision": decision,
        "decision_match": decision == reference,
        "recoverable_reference_pass": recoverable,
        "recoverable_case_recovered": recoverable and decision == "PASS" and all_closed,
        "terminal_review_correctly_retained": reference == "REVIEW" and decision == "REVIEW",
        "released_with_closed_conditions": decision != "PASS" or all_closed,
        "all_reference_conditions_closed_at_finish": all_closed,
        "hidden_open_conditions_at_finish": "|".join(hidden_open),
        "result_changing_actions": int(state.get("result_changing_actions_used", 0)),
        "invalid_or_unsupported_calls": int(state.get("invalid_calls", 0)),
        "tool_calls": int(state.get("tool_call_count", len(tool_log))),
        "input_tokens": int(usage.get("input_tokens", usage.get("input_token_count", 0))),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", usage.get("output_token_count", 0))),
        "runtime_seconds": round(runtime_s, 3),
        "returncode": returncode,
        "prompt_sha256": _sha256_text(prompt),
        "policy_fixture_path": str(policy_fixture_path),
        "react_fixture_path": str(react_fixture_path),
        "state_path": str(state_path),
        "tool_log_path": str(tool_log_path),
        "event_log_path": str(event_log_path),
    }
    _write_json(case_dir / "case_result.json", result)
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if bool(row.get("valid_run"))]
    recoverable = [row for row in valid if bool(row.get("recoverable_reference_pass"))]
    reviews = [row for row in valid if row.get("reference_decision") == "REVIEW"]
    passes = [row for row in valid if row.get("agent_decision") == "PASS"]
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": rows[0]["arm"] if rows else "",
        "case_count": len(rows),
        "valid_case_count": len(valid),
        "invalid_run_count": len(rows) - len(valid),
        "agent_decisions": dict(Counter(str(row.get("agent_decision")) for row in valid)),
        "terminal_agreement": _mean([float(bool(row["decision_match"])) for row in valid]),
        "recoverable_case_recovery": _mean(
            [float(bool(row["recoverable_case_recovered"])) for row in recoverable]
        ),
        "review_retention": _mean(
            [float(bool(row["terminal_review_correctly_retained"])) for row in reviews]
        ),
        "released_with_closed_conditions": _mean(
            [float(bool(row["released_with_closed_conditions"])) for row in passes]
        ),
        "all_conditions_closed_at_finish": _mean(
            [float(bool(row["all_reference_conditions_closed_at_finish"])) for row in valid]
        ),
        "invalid_or_unsupported_calls": sum(
            int(row["invalid_or_unsupported_calls"]) for row in valid
        ),
        "mean_tool_calls": _mean([float(row["tool_calls"]) for row in valid]),
        "mean_runtime_seconds": _mean([float(row["runtime_seconds"]) for row in valid]),
        "mean_output_tokens": _mean([float(row["output_tokens"]) for row in valid]),
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARM_TOOLS:
        raise ValueError(f"unknown arm: {args.arm}")
    cases = selected_cases(args.scope, args.case_ids)
    run_dir = OUTPUT_ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_results = existing_case_results(run_dir) if args.merge_existing else []
    config_path = run_dir / "run_config.json"
    existing_config = _read_json(config_path) if config_path.exists() else {}
    if args.merge_existing and not existing_results:
        raise ValueError("--merge-existing requires an existing run with case results")
    if args.merge_existing and existing_config:
        if existing_config.get("arm") != args.arm:
            raise ValueError("existing run uses a different arm")
        if existing_config.get("model") != args.model:
            raise ValueError("existing run uses a different model")

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                run_one_case,
                args.arm,
                case,
                run_dir,
                args.model,
                args.timeout,
            ): case
            for case in cases
        }
        for future in concurrent.futures.as_completed(future_map):
            case = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "arm": args.arm,
                    "case_id": case["case_id"],
                    "case_short": case["case_short"],
                    "stratum": case.get("stratum", ""),
                    "valid_run": False,
                    "violations": f"runner_exception:{type(exc).__name__}:{exc}",
                }
            results.append(result)
            print(
                json.dumps(
                    {
                        "case_short": result.get("case_short"),
                        "valid_run": result.get("valid_run"),
                        "decision": result.get("agent_decision"),
                        "match": result.get("decision_match"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    results.sort(key=lambda row: str(row.get("case_id", "")))
    if args.merge_existing:
        merged = {
            str(row["case_id"]): row
            for row in existing_results
        }
        merged.update({str(row["case_id"]): row for row in results})
        results = sorted(merged.values(), key=lambda row: str(row.get("case_id", "")))

    _write_csv(run_dir / "case_results.csv", results)
    summary = summarize_rows(results)
    _write_json(run_dir / "summary.json", summary)
    created_at = existing_config.get(
        "created_at",
        datetime.now().isoformat(timespec="seconds"),
    )
    _write_json(
        config_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "arm": args.arm,
            "scope": args.scope,
            "model": args.model,
            "case_count": len(results),
            "last_selected_case_count": len(cases),
            "last_selected_cases": [case["case_id"] for case in cases],
            "merge_existing": bool(args.merge_existing),
            "workers": args.workers,
            "timeout_seconds": args.timeout,
        },
    )
    return {"run_dir": str(run_dir), "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=sorted(ARM_TOOLS), required=True)
    run.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    run.add_argument("--run-name", required=True)
    run.add_argument("--model", default="gpt-5.5")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--timeout", type=int, default=300)
    run.add_argument(
        "--case-ids",
        nargs="*",
        default=[],
        help="Run only these case IDs or short IDs within the selected scope.",
    )
    run.add_argument(
        "--merge-existing",
        action="store_true",
        help="Replace selected cases in an existing run and recompute its summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        result = run_command(args)
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
