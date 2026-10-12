#!/usr/bin/env python3
"""Run and aggregate a blinded GPT-5.5 polymer-input validation baseline."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR / "output" / "polymer_input_taxonomy_benchmark"
CASES_PATH = BENCHMARK_ROOT / "polymer_input_taxonomy_cases.jsonl"
BENCHMARK_RESULTS = BENCHMARK_ROOT / "polymer_input_taxonomy_results.csv"
OUTPUT_ROOT = SCRIPT_DIR / "output" / "polymer_input_llm_baseline"
SERVER = SCRIPT_DIR / "mcp" / "polymer_input_llm_baseline_server.py"
SCHEMA_VERSION = "polymer_input_llm_baseline_runner.v1"
OPAQUE_SALT = "polymer-input-validation-blind-v1"
ALLOWED_TOOLS = {"start_case", "get_polymer_input", "finish_case"}
LEAK_TOKENS = (
    "expected_status",
    "expected_reason",
    "case_role",
    "pair_id",
    "mutation",
    "_reference_",
    "_challenge_",
)

DECISION_DEFINITIONS = {
    "PASS": (
        "The molecular description is internally valid and is supported by a "
        "registered molecular builder and force-field selection route."
    ),
    "REVIEW": (
        "The molecular description is internally valid, but the architecture "
        "or chemistry requires a specialized builder, force field or "
        "parameterization check that is not registered."
    ),
    "REJECT": (
        "The molecular description is malformed, incomplete or internally "
        "inconsistent."
    ),
}

REGISTERED_CAPABILITIES = {
    "accepted_input_forms": [
        "a repeat-unit SMILES plus chain length or target atom count",
        "an explicit finite-chain molecular SMILES",
        "named repeat-unit templates plus a linear or copolymer sequence",
        "repeat-unit templates plus branch templates and branch-placement rules",
    ],
    "registered_architectures": [
        "linear homopolymer",
        "alternating, periodic, block and random copolymer",
        "graft, comb and brush polymer",
        "randomly branched polymer",
        "long-chain branched polyethylene",
        "explicit finite linear or branched chain",
    ],
    "registered_chemistry_scope": [
        "hydrocarbon, heteroatom, halogenated, aromatic or conjugated repeat units",
        "charged repeat units and polyamide-like repeat units",
    ],
    "requires_review": [
        "cyclic polymer",
        "network polymer",
        "star polymer",
        "hyperbranched polymer",
        "dendrimer",
        "metal-containing or inorganic polymer",
    ],
}

VALIDATION_POINTS = [
    "All required fields must be present and use the declared input form.",
    "Every SMILES string must be chemically parseable.",
    "A repeat unit must expose valid polymer connection points; an explicit finite chain must contain one connected molecule and no open connection markers.",
    "Counts, chain length, degree of polymerization, arm count, branch count and side-chain length must satisfy their positive lower bounds.",
    "Every sequence or branch rule must reference a declared repeat-unit or branch template.",
    "Composition fractions and probabilities must lie between zero and one and must be mutually consistent.",
    "Branch positions must lie within the available backbone and cannot request more sites than the backbone provides.",
    "The declared architecture, sequence information and branch information must agree with one another.",
]


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    return "[" + ",".join(_toml_string(value) for value in values) + "]"


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def load_private_cases() -> list[dict[str, Any]]:
    cases = _read_jsonl(CASES_PATH)
    results = {row["case_id"]: row for row in _read_csv(BENCHMARK_RESULTS)}
    if len(cases) != 240 or set(results) != {row["case_id"] for row in cases}:
        raise ValueError("polymer-input benchmark does not contain 240 matched cases")
    ordered = sorted(
        cases,
        key=lambda row: _sha256_text(f"{OPAQUE_SALT}:{row['case_id']}"),
    )
    private: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        opaque_id = f"PI{index:03d}"
        candidate = json.loads(json.dumps(row["candidate"]))
        candidate["candidate_id"] = opaque_id
        benchmark = results[row["case_id"]]
        private.append(
            {
                "opaque_id": opaque_id,
                "original_case_id": row["case_id"],
                "pair_id": row["pair_id"],
                "case_role": row["case_role"],
                "category_id": row["category_id"],
                "coverage_group": row["coverage_group"],
                "expected_status": str(row["expected_status"]).upper(),
                "candidate": candidate,
                "production_operational_match": _as_bool(
                    benchmark["operational_match"]
                ),
                "complete_graph_materialized": _as_bool(
                    benchmark["complete_graph_materialized"]
                ),
            }
        )
    return private


def public_fixture(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case["opaque_id"],
        "candidate": case["candidate"],
        "decision_definitions": DECISION_DEFINITIONS,
        "registered_capabilities": REGISTERED_CAPABILITIES,
        "validation_points": VALIDATION_POINTS,
    }


def build_prompt(case_id: str) -> str:
    return f"""You are GPT-5.5 in a blinded polymer-input validation benchmark.

Process only opaque case {case_id} with tools from the `polymer_input_validation` MCP server. Call `start_case`, then `get_polymer_input`, and finally `finish_case` exactly once.

Judge the molecular description using the public decision definitions, registered preparation capabilities and validation points returned by the tool. Return PASS only when the input is internally valid and covered by a registered builder route. Return REVIEW for a valid input that needs a specialized builder, force field or parameterization check. Return REJECT for a malformed, incomplete or internally inconsistent input.

Do not use shell commands, file search, web access, code execution or tools outside `polymer_input_validation`. Any such call invalidates the run. The case identifier is opaque. Never infer hidden labels, paired cases or mutations. After `finish_case`, reply with one short sentence and stop.
"""


def _event_item(event: Mapping[str, Any]) -> Mapping[str, Any]:
    item = event.get("item")
    return item if isinstance(item, dict) else {}


def inspect_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    calls: list[dict[str, str]] = []
    usage: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type", ""))
        item = _event_item(event)
        item_type = str(item.get("type", ""))
        if item_type == "mcp_tool_call" and event_type.endswith("completed"):
            server = str(item.get("server", ""))
            tool = str(item.get("tool", item.get("name", "")))
            calls.append({"server": server, "tool": tool})
            if server != "polymer_input_validation" or tool not in ALLOWED_TOOLS:
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
        if event_type == "turn.failed":
            violations.append("runtime_event:turn.failed")
        if event_type == "unparseable_event":
            violations.append("unparseable_event")
    return {"violations": sorted(set(violations)), "calls": calls, "usage": usage}


def _tool_log_leaks(tool_log: list[dict[str, Any]]) -> list[str]:
    leaks: list[str] = []
    for index, row in enumerate(tool_log, start=1):
        text = json.dumps(row.get("result", {}), ensure_ascii=False).lower()
        for token in LEAK_TOKENS:
            if token in text:
                leaks.append(f"step{index}:{token}")
    return leaks


def selected_cases(
    scope: str,
    requested_cases: Iterable[str] = (),
) -> list[dict[str, Any]]:
    cases = load_private_cases()
    if scope == "pilot":
        pass_refs = [
            row
            for row in cases
            if row["case_role"] == "reference" and row["expected_status"] == "PASS"
        ][:4]
        review_refs = [
            row
            for row in cases
            if row["case_role"] == "reference" and row["expected_status"] == "REVIEW"
        ][:2]
        selected_pairs = {row["pair_id"] for row in pass_refs + review_refs}
        cases = [row for row in cases if row["pair_id"] in selected_pairs]
    requested = {str(value).strip() for value in requested_cases if str(value).strip()}
    if not requested:
        return cases
    selected = [
        row
        for row in cases
        if row["opaque_id"] in requested or row["original_case_id"] in requested
    ]
    matched = {
        value
        for row in selected
        for value in (row["opaque_id"], row["original_case_id"])
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
    case: Mapping[str, Any],
    run_dir: Path,
    model: str,
    timeout_s: int,
) -> dict[str, Any]:
    case_id = str(case["opaque_id"])
    case_dir = run_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = case_dir / "public_fixture.json"
    state_path = case_dir / "state.json"
    tool_log_path = case_dir / "tool_log.jsonl"
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
    fixture = public_fixture(case)
    _write_json(fixture_path, fixture)
    prompt = build_prompt(case_id)
    (case_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    server_args = [
        str(SERVER),
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
        f"mcp_servers.polymer_input_validation.command={_toml_string(sys.executable)}",
        "-c",
        f"mcp_servers.polymer_input_validation.args={_toml_array(server_args)}",
        "-c",
        "mcp_servers.polymer_input_validation.startup_timeout_sec=120",
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
        stdout, stderr, returncode = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
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
    event_info = inspect_events(events)
    tool_log = _read_jsonl(tool_log_path)
    state = _read_json(state_path) if state_path.exists() else {}
    violations = list(event_info["violations"])
    violations.extend(
        f"hidden_label_leak:{value}" for value in _tool_log_leaks(tool_log)
    )
    if timed_out:
        violations.append("timeout")
    if returncode != 0:
        violations.append(f"codex_returncode:{returncode}")
    if not state.get("finished"):
        violations.append("finish_case_not_called")
    if len(event_info["calls"]) != len(tool_log):
        violations.append("mcp_event_tool_log_count_mismatch")
    violations = sorted(set(violations))

    expected = str(case["expected_status"]).upper()
    decision = str(state.get("agent_decision", "")).upper()
    safe_pass = decision != "PASS" or (
        expected == "PASS"
        and bool(case["production_operational_match"])
        and bool(case["complete_graph_materialized"])
    )
    usage = event_info["usage"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "opaque_id": case_id,
        "original_case_id": case["original_case_id"],
        "pair_id": case["pair_id"],
        "case_role": case["case_role"],
        "category_id": case["category_id"],
        "coverage_group": case["coverage_group"],
        "model": model,
        "valid_run": not violations,
        "violations": "|".join(violations),
        "expected_status": expected,
        "agent_decision": decision,
        "decision_match": decision == expected,
        "invalid_input_intercepted": expected == "REJECT" and decision == "REJECT",
        "invalid_input_forwarded": expected == "REJECT" and decision == "PASS",
        "supported_reference_accepted": expected == "PASS" and decision == "PASS",
        "unsupported_reference_reviewed": (
            expected == "REVIEW" and decision == "REVIEW"
        ),
        "safe_pass": safe_pass,
        "agent_reason": state.get("agent_reason", ""),
        "invalid_calls": int(state.get("invalid_calls", 0)),
        "tool_calls": int(state.get("tool_call_count", len(tool_log))),
        "input_tokens": int(
            usage.get("input_tokens", usage.get("input_token_count", 0))
        ),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
        "output_tokens": int(
            usage.get("output_tokens", usage.get("output_token_count", 0))
        ),
        "runtime_seconds": round(runtime_s, 3),
        "returncode": returncode,
        "prompt_sha256": _sha256_text(prompt),
        "fixture_sha256": _sha256_text(
            json.dumps(fixture, ensure_ascii=False, sort_keys=True)
        ),
        "state_path": str(state_path),
        "tool_log_path": str(tool_log_path),
        "event_log_path": str(event_log_path),
    }
    _write_json(case_dir / "case_result.json", result)
    return result


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else math.nan


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if bool(row.get("valid_run"))]
    invalid_inputs = [row for row in valid if row["expected_status"] == "REJECT"]
    pass_refs = [row for row in valid if row["expected_status"] == "PASS"]
    review_refs = [row for row in valid if row["expected_status"] == "REVIEW"]
    pass_decisions = [row for row in valid if row["agent_decision"] == "PASS"]
    correct = sum(bool(row["decision_match"]) for row in valid)
    low, high = _wilson_interval(correct, len(valid))
    return {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(rows),
        "valid_case_count": len(valid),
        "invalid_run_count": len(rows) - len(valid),
        "agent_decisions": dict(
            sorted(Counter(str(row["agent_decision"]) for row in valid).items())
        ),
        "accuracy": correct / len(valid) if valid else math.nan,
        "accuracy_wilson_95ci": [low, high],
        "invalid_input_interception": _mean(
            float(bool(row["invalid_input_intercepted"])) for row in invalid_inputs
        ),
        "invalid_input_false_pass_rate": _mean(
            float(bool(row["invalid_input_forwarded"])) for row in invalid_inputs
        ),
        "supported_reference_acceptance": _mean(
            float(bool(row["supported_reference_accepted"])) for row in pass_refs
        ),
        "unsupported_reference_review_routing": _mean(
            float(bool(row["unsupported_reference_reviewed"]))
            for row in review_refs
        ),
        "safe_pass_rate": _mean(
            float(bool(row["safe_pass"])) for row in pass_decisions
        ),
        "mean_tool_calls": _mean(float(row["tool_calls"]) for row in valid),
        "mean_runtime_seconds": _mean(
            float(row["runtime_seconds"]) for row in valid
        ),
        "mean_output_tokens": _mean(float(row["output_tokens"]) for row in valid),
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    cases = selected_cases(args.scope, args.case_ids)
    run_dir = OUTPUT_ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = {row["opaque_id"] for row in cases}
    if args.merge_existing:
        retained = [
            row
            for row in existing_case_results(run_dir)
            if row["opaque_id"] not in selected_ids
        ]
    else:
        retained = []
    completed_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one_case, case, run_dir, args.model, args.timeout): case
            for case in cases
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed_rows.append(row)
            print(
                json.dumps(
                    {
                        "opaque_id": row["opaque_id"],
                        "valid_run": row["valid_run"],
                        "decision": row["agent_decision"],
                        "match": row["decision_match"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    rows = sorted(retained + completed_rows, key=lambda row: row["opaque_id"])
    _write_csv(run_dir / "case_results.csv", rows)
    summary = summarize_rows(rows)
    summary.update(
        {
            "run_name": args.run_name,
            "scope": args.scope,
            "model": args.model,
        }
    )
    _write_json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary}


def _metric(
    rows: list[Mapping[str, Any]],
    selector: Callable[[Mapping[str, Any]], bool],
    value: Callable[[Mapping[str, Any]], bool],
) -> float:
    selected = [row for row in rows if selector(row)]
    return _mean(float(value(row)) for row in selected)


def _bootstrap_cases(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["opaque_id"]].append(row)
    case_ids = sorted(by_case)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        replicate = [
            row
            for case_id in sampled
            for row in by_case[str(case_id)]
        ]
        value = metric(replicate)
        if not math.isnan(value):
            values.append(value)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def aggregate_command(args: argparse.Namespace) -> dict[str, Any]:
    run_names = args.run_names
    all_rows: list[dict[str, Any]] = []
    for repeat_index, run_name in enumerate(run_names, start=1):
        path = OUTPUT_ROOT / "runs" / run_name / "case_results.csv"
        rows = _read_csv(path)
        if len(rows) != 240:
            raise ValueError(f"{run_name} contains {len(rows)} rather than 240 rows")
        for row in rows:
            if not _as_bool(row["valid_run"]):
                raise ValueError(f"{run_name} contains invalid run {row['opaque_id']}")
            normalized = dict(row)
            normalized["repeat_index"] = repeat_index
            for key in (
                "decision_match",
                "invalid_input_intercepted",
                "invalid_input_forwarded",
                "supported_reference_accepted",
                "unsupported_reference_reviewed",
                "safe_pass",
            ):
                normalized[key] = _as_bool(normalized[key])
            all_rows.append(normalized)

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_case[row["opaque_id"]].append(row)
    repeat_consistency = _mean(
        float(len({row["agent_decision"] for row in rows}) == 1)
        for rows in by_case.values()
    )
    metrics: dict[str, Callable[[list[dict[str, Any]]], float]] = {
        "accuracy": lambda rows: _metric(
            rows, lambda row: True, lambda row: bool(row["decision_match"])
        ),
        "invalid_input_interception": lambda rows: _metric(
            rows,
            lambda row: row["expected_status"] == "REJECT",
            lambda row: bool(row["invalid_input_intercepted"]),
        ),
        "invalid_input_false_pass_rate": lambda rows: _metric(
            rows,
            lambda row: row["expected_status"] == "REJECT",
            lambda row: bool(row["invalid_input_forwarded"]),
        ),
        "supported_reference_acceptance": lambda rows: _metric(
            rows,
            lambda row: row["expected_status"] == "PASS",
            lambda row: bool(row["supported_reference_accepted"]),
        ),
        "unsupported_reference_review_routing": lambda rows: _metric(
            rows,
            lambda row: row["expected_status"] == "REVIEW",
            lambda row: bool(row["unsupported_reference_reviewed"]),
        ),
        "safe_pass_rate": lambda rows: _metric(
            rows,
            lambda row: row["agent_decision"] == "PASS",
            lambda row: bool(row["safe_pass"]),
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_names": run_names,
        "repeat_count": len(run_names),
        "case_count": len(by_case),
        "case_run_count": len(all_rows),
        "repeat_consistency": repeat_consistency,
        "agent_decisions": dict(
            sorted(Counter(row["agent_decision"] for row in all_rows).items())
        ),
    }
    for index, (name, function) in enumerate(metrics.items()):
        summary[name] = function(all_rows)
        low, high = _bootstrap_cases(
            all_rows,
            function,
            iterations=args.bootstrap_iterations,
            seed=args.seed + index,
        )
        summary[f"{name}_ci95"] = [low, high]

    category_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["category_id"]].append(row)
    for category_id, rows in sorted(grouped.items()):
        category_rows.append(
            {
                "category_id": category_id,
                "case_count": len({row["opaque_id"] for row in rows}),
                "case_run_count": len(rows),
                "accuracy": _metric(
                    rows,
                    lambda row: True,
                    lambda row: bool(row["decision_match"]),
                ),
                "repeat_consistency": _mean(
                    float(
                        len(
                            {
                                item["agent_decision"]
                                for item in by_case[row["opaque_id"]]
                            }
                        )
                        == 1
                    )
                    for row in {item["opaque_id"]: item for item in rows}.values()
                ),
            }
        )
    output_dir = OUTPUT_ROOT / "aggregate"
    _write_csv(output_dir / "polymer_input_llm_case_results_long.csv", all_rows)
    _write_csv(output_dir / "polymer_input_llm_category_summary.csv", category_rows)
    _write_json(output_dir / "polymer_input_llm_summary.json", summary)
    _write_csv(output_dir / "polymer_input_llm_summary.csv", [summary])
    return {"output_dir": str(output_dir), "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    run.add_argument("--run-name", required=True)
    run.add_argument("--model", default="gpt-5.5")
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--timeout", type=int, default=300)
    run.add_argument("--case-ids", nargs="*", default=[])
    run.add_argument("--merge-existing", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-names", nargs="+", required=True)
    aggregate.add_argument("--bootstrap-iterations", type=int, default=10000)
    aggregate.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        result = run_command(args)
    else:
        result = aggregate_command(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
