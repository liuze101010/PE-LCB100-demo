#!/usr/bin/env python3
"""Run and score GPT-5.5 without the PE-LCB100 rule and audit layers."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPLAY_ROOT = SCRIPT_DIR / "output" / "pelcb100_policy_free_react_baseline"
FIXTURE_DIR = REPLAY_ROOT / "fixtures"
FIXTURE_INDEX = REPLAY_ROOT / "fixture_index.csv"
PILOT_CASES = REPLAY_ROOT / "pilot_12_cases.json"
MCP_SERVER = SCRIPT_DIR / "mcp" / "pelcb100_policy_free_replay_server.py"
RUN_ROOT = REPLAY_ROOT / "runs"
SCHEMA_VERSION = "pelcb100_policy_free_baseline.v1"

ALLOWED_MCP_TOOLS = {
    "start_case",
    "inspect_outputs",
    "inspect_run_configuration",
    "query_reference_data",
    "inspect_msd_diagnostics",
    "rerun_md",
    "run_msd_analysis",
    "extend_trajectory",
    "finish_case",
}
OPERATION_TOOLS = {"rerun_md", "run_msd_analysis", "extend_trajectory"}
FUTURE_LEAK_TOKENS = (
    "reference_decision",
    "reference_full_pass",
    "publication_state",
    "final diagnosis",
    "terminal state",
    "retained for review",
    "allowed route",
    "primary diagnosis",
    "open_conditions",
    "pending_conditions",
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    return "[" + ",".join(_toml_string(value) for value in values) + "]"


def build_prompt(case_id: str, case_short: str) -> str:
    return f"""You are an ordinary ReAct/tool-calling Agent in a blinded polymer molecular-dynamics replay benchmark.

Process only case {case_id} ({case_short}). The rule layer and audit layer have been removed. You alone must inspect the raw outputs, consult reference evidence when useful, choose primitive operations, and decide when the result is scientifically usable. The tools never provide a scientific PASS/WARN verdict, a cause diagnosis, a recommended route, or the hidden reference decision.

Use only tools from the `policy_free_replay` MCP server. Start with `start_case`. At most two result-changing operations are permitted. PASS means you judge all required outputs and scientific checks adequate. REVIEW means the available evidence or operations cannot support release. FAIL is reserved for an explicitly unsupported calculation. Call `finish_case` exactly once.

Do not use shell commands, file search, web access, code execution, or any tool outside `policy_free_replay`. Any such call invalidates the run. Do not invent unavailable results. After `finish_case`, reply with one short sentence and stop."""


def _event_item(event: Mapping[str, Any]) -> Mapping[str, Any]:
    item = event.get("item")
    return item if isinstance(item, dict) else {}


def inspect_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    runtime_errors: list[str] = []
    mcp_calls: list[dict[str, Any]] = []
    reasoning = ""
    reasoning_by_call: list[str] = []
    usage: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type", ""))
        item = _event_item(event)
        item_type = str(item.get("type", ""))
        if item_type == "reasoning" and item.get("text"):
            reasoning = str(item["text"])
        if item_type == "mcp_tool_call" and event_type.endswith("completed"):
            server = str(item.get("server", ""))
            tool = str(item.get("tool", item.get("name", "")))
            mcp_calls.append({"server": server, "tool": tool, "item": dict(item)})
            reasoning_by_call.append(reasoning)
            if server != "policy_free_replay" or tool not in ALLOWED_MCP_TOOLS:
                violations.append(f"unregistered_mcp:{server}:{tool}")
        elif item_type in {"command_execution", "file_change", "web_search", "image_generation", "browser_action"}:
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
        "mcp_calls": mcp_calls,
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


def _merge_action_trace(
    tool_log: list[dict[str, Any]], event_info: Mapping[str, Any]
) -> list[dict[str, Any]]:
    calls = list(event_info.get("mcp_calls", []))
    reasoning = list(event_info.get("reasoning_by_call", []))
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(tool_log):
        tool = str(record.get("tool", ""))
        args = record.get("arguments", {})
        summary = reasoning[index] if index < len(reasoning) else ""
        if not summary:
            if tool.startswith("inspect") or tool == "query_reference_data":
                summary = f"The model chose to obtain raw evidence with {tool}."
            elif tool in OPERATION_TOOLS:
                summary = f"The model chose primitive operation {tool} with {args}."
            elif tool == "finish_case":
                summary = str(args.get("reason", ""))
            else:
                summary = "The model started the isolated case."
        event_call = calls[index] if index < len(calls) else {}
        rows.append(
            {
                "step": index + 1,
                "case_id": record.get("case_id", ""),
                "model_reasoning_summary": summary,
                "tool": tool,
                "parameters": args,
                "tool_return": record.get("result", {}),
                "new_visible_evidence_sha256": record.get("visible_evidence_sha256", ""),
                "evaluator_open_issues_after_step": record.get("evaluator_open_issues_after_step", []),
                "event_server": event_call.get("server", ""),
                "event_tool": event_call.get("tool", ""),
            }
        )
    return rows


def run_one_case(case_id: str, case_short: str, run_dir: Path, model: str, timeout_s: int) -> dict[str, Any]:
    case_dir = run_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"{case_id}.json"
    state_path = case_dir / "replay_state.json"
    tool_log_path = case_dir / "replay_tool_log.jsonl"
    event_log_path = case_dir / "codex_events.jsonl"
    stderr_path = case_dir / "codex_stderr.txt"
    last_message_path = case_dir / "codex_last_message.txt"
    action_trace_path = case_dir / "action_trace.jsonl"
    prompt = build_prompt(case_id, case_short)
    (case_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    server_args = [
        str(MCP_SERVER),
        "--fixture",
        str(fixture_path),
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
        f"mcp_servers.policy_free_replay.command={_toml_string(sys.executable)}",
        "-c",
        f"mcp_servers.policy_free_replay.args={_toml_array(server_args)}",
        "-c",
        "mcp_servers.policy_free_replay.startup_timeout_sec=30",
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
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        returncode = 124
        timed_out = True
    runtime_s = time.perf_counter() - started
    event_log_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    events = _read_jsonl(event_log_path)
    event_info = inspect_events(events)
    tool_log = _read_jsonl(tool_log_path)
    state = _load_json(state_path) if state_path.exists() else {}
    fixture = _load_json(fixture_path)
    _write_jsonl(action_trace_path, _merge_action_trace(tool_log, event_info))

    violations = list(event_info["violations"])
    violations.extend(f"future_evidence_leak:{item}" for item in _future_leaks(tool_log))
    if timed_out:
        violations.append("timeout")
    if returncode != 0:
        violations.append(f"codex_returncode:{returncode}")
    if not state.get("finished"):
        violations.append("finish_case_not_called")
    event_call_count = len(event_info["mcp_calls"])
    if event_call_count != len(tool_log):
        violations.append("mcp_event_tool_log_count_mismatch")
    violations = sorted(set(violations))

    reference = fixture["private_reference"]["reference_decision"]
    decision = str(state.get("agent_decision", ""))
    open_issues = state.get("evaluator_open_issues", []) if isinstance(state.get("evaluator_open_issues"), list) else []
    all_closed = not open_issues
    operation_calls = [row for row in tool_log if row.get("tool") in OPERATION_TOOLS]
    successful_operations = [row for row in operation_calls if row.get("result", {}).get("status") == "ok"]
    evidence_tools = {row.get("tool") for row in tool_log}
    usage = event_info["usage"]
    total_tokens = sum(value for key, value in usage.items() if "token" in key and "cached" not in key)
    runtime_error_text = " | ".join(str(item) for item in event_info.get("runtime_errors", []))
    quota_exhausted = "usage limit" in runtime_error_text.lower()
    initial_conditions = fixture["private_reference"]["initial_conditions"]
    result = {
        "case_id": case_id,
        "case_short": case_short,
        "model": model,
        "valid_run": not violations,
        "violations": "|".join(violations),
        "reference_decision": reference,
        "agent_decision": decision,
        "decision_match": decision == reference,
        "reference_path_msd_statistics_complete": fixture["private_reference"].get(
            "reference_path_msd_statistics_complete", False
        ),
        "initial_requires_action": bool([item for item in initial_conditions if item != "retained_audit_signal"]),
        "recoverable_reference_pass": reference == "PASS" and bool(initial_conditions),
        "recoverable_case_recovered": reference == "PASS" and bool(initial_conditions) and decision == "PASS" and all_closed,
        "terminal_review_correctly_retained": reference == "REVIEW" and decision == "REVIEW",
        "all_reference_conditions_closed_at_finish": all_closed,
        "released_with_closed_conditions": decision != "PASS" or all_closed,
        "evaluator_open_issues_at_finish": "|".join(open_issues),
        "result_changing_actions": state.get("result_changing_actions_used", 0),
        "primitive_operation_calls": len(operation_calls),
        "successful_primitive_operations": len(successful_operations),
        "operation_success_rate": len(successful_operations) / len(operation_calls) if operation_calls else 1.0,
        "invalid_or_unsupported_operation_calls": state.get("invalid_operation_calls", 0),
        "inspected_outputs": "inspect_outputs" in evidence_tools,
        "inspected_configuration": "inspect_run_configuration" in evidence_tools,
        "inspected_msd": "inspect_msd_diagnostics" in evidence_tools,
        "queried_reference_data": "query_reference_data" in evidence_tools,
        "tool_calls": state.get("tool_call_count", len(tool_log)),
        "event_mcp_calls": event_call_count,
        "tool_log_coverage": len(tool_log) / event_call_count if event_call_count else (1.0 if not tool_log else 0.0),
        "input_tokens": usage.get("input_tokens", usage.get("input_token_count", 0)),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", usage.get("output_token_count", 0)),
        "total_tokens_reported": total_tokens,
        "runtime_seconds": round(runtime_s, 3),
        "transient_runtime_error_count": len(event_info.get("runtime_errors", [])),
        "external_failure_type": "model_usage_limit" if quota_exhausted else "",
        "runtime_error_message": runtime_error_text,
        "returncode": returncode,
        "prompt_sha256": _sha256_text(prompt),
        "fixture_path": str(fixture_path),
        "state_path": str(state_path),
        "tool_log_path": str(tool_log_path),
        "event_log_path": str(event_log_path),
        "action_trace_path": str(action_trace_path),
    }
    _write_json(case_dir / "case_result.json", result)
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if bool(row.get("valid_run"))]
    released = [row for row in valid if row.get("agent_decision") == "PASS"]
    recoverable = [row for row in valid if bool(row.get("recoverable_reference_pass"))]
    reviews = [row for row in valid if row.get("reference_decision") == "REVIEW"]
    operation_calls = sum(int(row.get("primitive_operation_calls", 0)) for row in valid)
    operation_success = sum(int(row.get("successful_primitive_operations", 0)) for row in valid)
    complete_evidence = [row for row in valid if bool(row.get("reference_path_msd_statistics_complete"))]
    return {
        "n_cases": len(rows),
        "valid_runs": len(valid),
        "invalid_runs": len(rows) - len(valid),
        "model_usage_limit_runs": sum(row.get("external_failure_type") == "model_usage_limit" for row in rows),
        "agent_pass": sum(row.get("agent_decision") == "PASS" for row in valid),
        "agent_review": sum(row.get("agent_decision") == "REVIEW" for row in valid),
        "agent_fail": sum(row.get("agent_decision") == "FAIL" for row in valid),
        "exact_terminal_accuracy": _mean([float(bool(row.get("decision_match"))) for row in valid]),
        "complete_evidence_case_count": len(complete_evidence),
        "exact_terminal_accuracy_complete_evidence": _mean(
            [float(bool(row.get("decision_match"))) for row in complete_evidence]
        ),
        "publication_condition_satisfied_rate": _mean(
            [float(bool(row.get("released_with_closed_conditions"))) for row in released]
        ),
        "recoverable_case_recovery_rate": _mean(
            [float(bool(row.get("recoverable_case_recovered"))) for row in recoverable]
        ),
        "terminal_review_retention_rate": _mean(
            [float(bool(row.get("terminal_review_correctly_retained"))) for row in reviews]
        ),
        "all_reference_conditions_closed_at_finish_rate": _mean(
            [float(bool(row.get("all_reference_conditions_closed_at_finish"))) for row in valid]
        ),
        "primitive_operation_success_rate": operation_success / operation_calls if operation_calls else 1.0,
        "invalid_or_unsupported_operation_calls": sum(
            int(row.get("invalid_or_unsupported_operation_calls", 0)) for row in valid
        ),
        "raw_output_inspection_rate": _mean([float(bool(row.get("inspected_outputs"))) for row in valid]),
        "reference_query_rate": _mean([float(bool(row.get("queried_reference_data"))) for row in valid]),
        "mean_tool_calls": _mean([float(row.get("tool_calls", 0)) for row in valid]),
        "mean_tokens": _mean([float(row.get("total_tokens_reported", 0)) for row in valid]),
        "mean_runtime_seconds": _mean([float(row.get("runtime_seconds", 0)) for row in valid]),
        "complete_call_log_coverage_rate": _mean(
            [float(float(row.get("tool_log_coverage", 0)) == 1.0) for row in valid]
        ),
    }


def _selected_cases(selection: str) -> list[dict[str, str]]:
    rows = _read_csv(FIXTURE_INDEX)
    if selection == "all":
        return rows
    by_id = {row["case_id"]: row for row in rows}
    pilot = _load_json(PILOT_CASES)
    return [by_id[item["case_id"]] for item in pilot["cases"]]


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    cases = _selected_cases(args.cases)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.run_name or f"{args.cases}_repeat{args.repeat_index}_{stamp}"
    run_dir = RUN_ROOT / name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        _write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "selection": args.cases,
                "repeat_index": args.repeat_index,
                "model": args.model,
                "case_count": len(cases),
                "max_workers": args.max_workers,
                "timeout_seconds_per_case": args.timeout_seconds,
                "fixture_manifest": str(REPLAY_ROOT / "fixture_manifest.json"),
                "mcp_server": str(MCP_SERVER),
            },
        )

    order = {row["case_id"]: index for index, row in enumerate(cases)}
    previous: dict[str, dict[str, Any]] = {}
    if args.resume:
        for row in cases:
            result_path = run_dir / "cases" / row["case_id"] / "case_result.json"
            if result_path.exists():
                previous[row["case_id"]] = _load_json(result_path)
    results_by_id = {
        case_id: result for case_id, result in previous.items() if bool(result.get("valid_run"))
    }
    pending = [row for row in cases if row["case_id"] not in results_by_id]
    for row in pending:
        old_case_dir = run_dir / "cases" / row["case_id"]
        if old_case_dir.exists():
            archive_dir = run_dir / "failed_attempts" / row["case_id"] / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_case_dir), str(archive_dir))

    def write_progress() -> None:
        current = sorted(results_by_id.values(), key=lambda row: order.get(str(row.get("case_id")), 10**9))
        _write_csv(run_dir / "case_results.csv", current)
        summary = summarize_rows(current)
        summary.update(
            {
                "run_dir": str(run_dir),
                "model": args.model,
                "repeat_index": args.repeat_index,
                "target_case_count": len(cases),
                "pending_case_count": len(cases) - len(current),
            }
        )
        _write_json(run_dir / "run_summary.json", summary)

    print(f"Resuming {len(results_by_id)} completed cases; {len(pending)} cases remain.", flush=True)
    quota_blocked = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        pending_iter = iter(pending)
        futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}

        def submit_next() -> bool:
            try:
                row = next(pending_iter)
            except StopIteration:
                return False
            future = pool.submit(
                run_one_case,
                row["case_id"],
                row["case_short"],
                run_dir,
                args.model,
                args.timeout_seconds,
            )
            futures[future] = row["case_id"]
            return True

        for _ in range(min(args.max_workers, len(pending))):
            submit_next()
        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                case_id = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover
                    result = {
                        "case_id": case_id,
                        "valid_run": False,
                        "violations": f"runner_exception:{exc}",
                        "agent_decision": "",
                        "external_failure_type": "runner_exception",
                    }
                results_by_id[case_id] = result
                write_progress()
                print(
                    f"[{len(results_by_id)}/{len(cases)}] {case_id}: {result.get('agent_decision','')} "
                    f"valid={result.get('valid_run')}",
                    flush=True,
                )
                if result.get("external_failure_type") == "model_usage_limit":
                    quota_blocked = True
            if not quota_blocked:
                while len(futures) < args.max_workers and submit_next():
                    pass
    results = sorted(results_by_id.values(), key=lambda row: order.get(str(row.get("case_id")), 10**9))
    _write_csv(run_dir / "case_results.csv", results)
    summary = summarize_rows(results)
    summary.update(
        {
            "run_dir": str(run_dir),
            "model": args.model,
            "repeat_index": args.repeat_index,
            "target_case_count": len(cases),
            "pending_case_count": len(cases) - len(results),
            "stopped_by_model_usage_limit": quota_blocked,
        }
    )
    _write_json(run_dir / "run_summary.json", summary)
    (RUN_ROOT / f"{args.cases}_latest.json").write_text(
        json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", choices=["pilot", "all"], default="pilot")
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    run_batch(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
