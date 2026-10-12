#!/usr/bin/env python3
"""Aggregate blinded PE-LCB100 CS baselines with case-clustered bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
CS_RUN_ROOT = HERE / "output" / "pelcb100_cs_baselines" / "runs"
REACT_RUN_ROOT = HERE / "output" / "pelcb100_policy_free_react_baseline" / "runs"
OUTPUT_ROOT = HERE / "output" / "pelcb100_cs_baselines" / "aggregate"
SCHEMA_VERSION = "pelcb100_cs_baseline_aggregate.v1"

COMPLETE_RUNS = {
    "Direct GPT judge": [
        CS_RUN_ROOT / "all_direct_v1" / "case_results.csv",
        CS_RUN_ROOT / "all_direct_v2" / "case_results.csv",
        CS_RUN_ROOT / "all_direct_v3" / "case_results.csv",
    ],
    "GPT result judge + workflow": [
        CS_RUN_ROOT / "all_result_judge_v1" / "case_results.csv",
        CS_RUN_ROOT / "all_result_judge_v2" / "case_results.csv",
        CS_RUN_ROOT / "all_result_judge_v3" / "case_results.csv",
    ],
    "Plan-and-Execute": [
        CS_RUN_ROOT / "all_plan_execute_v1" / "case_results.csv",
        CS_RUN_ROOT / "all_plan_execute_v2" / "case_results.csv",
        CS_RUN_ROOT / "all_plan_execute_v3" / "case_results.csv",
    ],
    "Policy-free ReAct": [
        REACT_RUN_ROOT / "all_repeat1" / "case_results.csv",
        REACT_RUN_ROOT / "all_repeat2" / "case_results.csv",
        REACT_RUN_ROOT / "all_repeat3" / "case_results.csv",
    ],
    "Scientific checks + GPT planner": [
        CS_RUN_ROOT / "all_repair_planner_v1" / "case_results.csv",
        CS_RUN_ROOT / "all_repair_planner_v2" / "case_results.csv",
        CS_RUN_ROOT / "all_repair_planner_v3" / "case_results.csv",
    ],
}

PILOT_RUNS = {
    "Plan-and-Execute": CS_RUN_ROOT / "pilot_plan_execute_v1" / "case_results.csv",
    "ReAct + self-critique": CS_RUN_ROOT / "pilot_self_critique_v1" / "case_results.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def count_tool_errors(row: Mapping[str, str]) -> int:
    path_value = str(row.get("tool_log_path", "")).strip()
    if path_value:
        path = Path(path_value)
        if path.exists():
            count = 0
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    count += 1
                    continue
                if record.get("result", {}).get("status") == "error":
                    count += 1
            return count
    return int(
        float(
            row.get(
                "invalid_or_unsupported_calls",
                row.get("invalid_or_unsupported_operation_calls", 0),
            )
            or 0
        )
    )


def normalized_row(
    row: Mapping[str, str],
    *,
    system: str,
    repeat_index: int,
    source_path: Path,
) -> dict[str, Any]:
    if "invalid_or_unsupported_operation_calls" in row:
        decision_match = as_bool(row.get("decision_match"))
        recoverable = as_bool(row.get("recoverable_reference_pass"))
        recovered = as_bool(row.get("recoverable_case_recovered"))
        review_retained = as_bool(row.get("terminal_review_correctly_retained"))
        safe_release = as_bool(row.get("released_with_closed_conditions"))
    else:
        decision_match = as_bool(row.get("decision_match"))
        recoverable = as_bool(row.get("recoverable_reference_pass"))
        recovered = as_bool(row.get("recoverable_case_recovered"))
        review_retained = as_bool(row.get("terminal_review_correctly_retained"))
        safe_release = as_bool(row.get("released_with_closed_conditions"))
    invalid_calls = count_tool_errors(row)
    return {
        "system": system,
        "repeat_index": repeat_index,
        "case_id": row.get("case_id", ""),
        "case_short": row.get("case_short", ""),
        "valid_run": as_bool(row.get("valid_run")),
        "reference_decision": row.get("reference_decision", ""),
        "agent_decision": row.get("agent_decision", ""),
        "decision_match": decision_match,
        "recoverable_reference_pass": recoverable,
        "recoverable_case_recovered": recovered,
        "terminal_review_correctly_retained": review_retained,
        "released_with_closed_conditions": safe_release,
        "invalid_calls": invalid_calls,
        "tool_calls": float(row.get("tool_calls", 0) or 0),
        "output_tokens": float(
            row.get("output_tokens", row.get("total_tokens_reported", 0)) or 0
        ),
        "runtime_seconds": float(row.get("runtime_seconds", 0) or 0),
        "source_path": str(source_path),
    }


def load_system(system: str, paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat_index, path in enumerate(paths, start=1):
        for row in read_csv(path):
            rows.append(
                normalized_row(
                    row,
                    system=system,
                    repeat_index=repeat_index,
                    source_path=path,
                )
            )
    return [row for row in rows if row["valid_run"]]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def metric_agreement(rows: list[Mapping[str, Any]]) -> float:
    return mean(float(row["decision_match"]) for row in rows)


def metric_recovery(rows: list[Mapping[str, Any]]) -> float:
    selected = [row for row in rows if row["recoverable_reference_pass"]]
    return mean(float(row["recoverable_case_recovered"]) for row in selected)


def metric_review_retention(rows: list[Mapping[str, Any]]) -> float:
    selected = [row for row in rows if row["reference_decision"] == "REVIEW"]
    return mean(float(row["terminal_review_correctly_retained"]) for row in selected)


def metric_safe_release(rows: list[Mapping[str, Any]]) -> float:
    selected = [row for row in rows if row["agent_decision"] == "PASS"]
    return mean(float(row["released_with_closed_conditions"]) for row in selected)


METRICS: dict[str, Callable[[list[Mapping[str, Any]]], float]] = {
    "terminal_agreement": metric_agreement,
    "recoverable_case_recovery": metric_recovery,
    "review_retention": metric_review_retention,
    "safe_pass_rate": metric_safe_release,
}


def clustered_bootstrap(
    rows: list[dict[str, Any]],
    metric: Callable[[list[Mapping[str, Any]]], float],
    *,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float, float]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    case_ids = sorted(by_case)
    estimate = metric(rows)
    draws: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        sample_rows: list[dict[str, Any]] = []
        for case_id in sampled:
            sample_rows.extend(by_case[str(case_id)])
        value = metric(sample_rows)
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return estimate, float("nan"), float("nan")
    low, high = np.percentile(np.asarray(draws), [2.5, 97.5])
    return estimate, float(low), float(high)


def repeat_consistency(rows: list[dict[str, Any]]) -> float:
    by_case: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_case[str(row["case_id"])].add(str(row["agent_decision"]))
    if not by_case:
        return float("nan")
    return mean(float(len(decisions) == 1) for decisions in by_case.values())


def summarize_system(
    system: str,
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, Any]:
    case_ids = {str(row["case_id"]) for row in rows}
    repeats = {int(row["repeat_index"]) for row in rows}
    summary: dict[str, Any] = {
        "system": system,
        "case_count": len(case_ids),
        "target_case_count": 100,
        "completion_fraction": len(case_ids) / 100,
        "repeat_count": len(repeats),
        "decision_counts": json.dumps(
            Counter(str(row["agent_decision"]) for row in rows),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "repeat_consistency": repeat_consistency(rows) if len(repeats) > 1 else "",
        "invalid_calls_total": sum(int(row["invalid_calls"]) for row in rows),
        "invalid_calls_per_100_case_runs": (
            100 * sum(int(row["invalid_calls"]) for row in rows) / len(rows)
        ),
        "mean_tool_calls": mean(float(row["tool_calls"]) for row in rows),
        "mean_output_tokens": mean(float(row["output_tokens"]) for row in rows),
        "mean_runtime_seconds": mean(float(row["runtime_seconds"]) for row in rows),
    }
    for metric_name, metric in METRICS.items():
        estimate, low, high = clustered_bootstrap(
            rows,
            metric,
            rng=rng,
            iterations=iterations,
        )
        summary[metric_name] = estimate
        summary[f"{metric_name}_ci95_low"] = low
        summary[f"{metric_name}_ci95_high"] = high
    return summary


def pilot_summary() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for system, path in PILOT_RUNS.items():
        rows = load_system(system, [path])
        summaries.append(
            {
                "system": system,
                "case_count": len({row["case_id"] for row in rows}),
                "terminal_agreement": metric_agreement(rows),
                "recoverable_case_recovery": metric_recovery(rows),
                "review_retention": metric_review_retention(rows),
                "safe_pass_rate": metric_safe_release(rows),
                "invalid_calls_total": sum(int(row["invalid_calls"]) for row in rows),
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for system, paths in COMPLETE_RUNS.items():
        rows = load_system(system, paths)
        all_rows.extend(rows)
        summaries.append(
            summarize_system(
                system,
                rows,
                rng=rng,
                iterations=args.bootstrap_iterations,
            )
        )

    complete = {
        "system": "Complete Agent + workflow",
        "case_count": 100,
        "target_case_count": 100,
        "completion_fraction": 1.0,
        "repeat_count": 1,
        "decision_counts": json.dumps({"PASS": 92, "REVIEW": 8}),
        "repeat_consistency": "",
        "invalid_calls_total": 0,
        "invalid_calls_per_100_case_runs": 0.0,
        "mean_tool_calls": "",
        "mean_output_tokens": "",
        "mean_runtime_seconds": "",
    }
    for metric_name in METRICS:
        complete[metric_name] = 1.0
        complete[f"{metric_name}_ci95_low"] = 1.0
        complete[f"{metric_name}_ci95_high"] = 1.0
    summaries.append(complete)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "cs_baseline_case_results_long.csv", all_rows)
    write_csv(args.output_root / "cs_baseline_summary.csv", summaries)
    pilots = pilot_summary()
    write_csv(args.output_root / "cs_baseline_pilot_summary.csv", pilots)
    notes = [
        "ReAct + self-critique currently has only the fixed 12-case pilot and is excluded from the 100-case main comparison.",
        "Complete Agent + workflow is the execution reference supported by the manuscript checks and is not a stochastic model estimate.",
    ]
    plan_summary = next(
        (row for row in summaries if row["system"] == "Plan-and-Execute"),
        None,
    )
    if plan_summary and int(plan_summary["case_count"]) < int(
        plan_summary["target_case_count"]
    ):
        notes.insert(
            0,
            "Plan-and-Execute is incomplete and is excluded from the main comparison.",
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "systems": summaries,
        "pilot_systems": pilots,
        "notes": notes,
    }
    write_json(args.output_root / "cs_baseline_aggregate.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
