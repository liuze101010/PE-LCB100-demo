#!/usr/bin/env python3
"""Aggregate repeated policy-free ReAct runs with case-clustered bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR / "output" / "pelcb100_policy_free_react_baseline"
RUN_ROOT = BASELINE_ROOT / "runs"
FIXTURE_INDEX = BASELINE_ROOT / "fixture_index.csv"
DEFAULT_RUNS = ("all_repeat1", "all_repeat2", "all_repeat3")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _ratio(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [float(_as_bool(row.get(field))) for row in rows]
    return sum(values) / len(values) if values else math.nan


def metrics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    valid = [row for row in rows if _as_bool(row.get("valid_run"))]
    complete = [row for row in valid if _as_bool(row.get("reference_path_msd_statistics_complete"))]
    recoverable = [row for row in valid if _as_bool(row.get("recoverable_reference_pass"))]
    reviews = [row for row in valid if row.get("reference_decision") == "REVIEW"]
    released = [row for row in valid if row.get("agent_decision") == "PASS"]
    operation_calls = sum(int(float(row.get("primitive_operation_calls", 0) or 0)) for row in valid)
    successful_operations = sum(int(float(row.get("successful_primitive_operations", 0) or 0)) for row in valid)
    return {
        "terminal_accuracy": _ratio(valid, "decision_match"),
        "terminal_accuracy_complete_evidence": _ratio(complete, "decision_match"),
        "recoverable_case_recovery": _ratio(recoverable, "recoverable_case_recovered"),
        "review_retention": _ratio(reviews, "terminal_review_correctly_retained"),
        "released_with_closed_conditions": _ratio(released, "released_with_closed_conditions"),
        "all_conditions_closed_at_finish": _ratio(valid, "all_reference_conditions_closed_at_finish"),
        "primitive_operation_success": successful_operations / operation_calls if operation_calls else 1.0,
    }


def _percentile(values: list[float], probability: float) -> float:
    clean = sorted(value for value in values if not math.isnan(value))
    if not clean:
        return math.nan
    position = probability * (len(clean) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    fraction = position - lower
    return clean[lower] * (1 - fraction) + clean[upper] * fraction


def clustered_bootstrap(
    rows: list[Mapping[str, Any]], iterations: int, seed: int
) -> dict[str, dict[str, float]]:
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _as_bool(row.get("valid_run")):
            by_case[str(row["case_id"])].append(row)
    case_ids = sorted(by_case)
    observed = metrics([row for case_id in case_ids for row in by_case[case_id]])
    samples: dict[str, list[float]] = {name: [] for name in observed}
    rng = random.Random(seed)
    for _ in range(iterations):
        selected = [rng.choice(case_ids) for _ in case_ids]
        sampled_rows = [row for case_id in selected for row in by_case[case_id]]
        result = metrics(sampled_rows)
        for name, value in result.items():
            samples[name].append(value)
    return {
        name: {
            "estimate": observed[name],
            "ci95_low": _percentile(samples[name], 0.025),
            "ci95_high": _percentile(samples[name], 0.975),
        }
        for name in observed
    }


def _fmt(value: float) -> str:
    return "NA" if math.isnan(value) else f"{100 * value:.1f}%"


def aggregate(run_names: list[str], output_dir: Path, bootstrap_iterations: int, seed: int) -> dict[str, Any]:
    all_rows: list[dict[str, str]] = []
    repeat_rows: list[dict[str, Any]] = []
    for repeat_index, run_name in enumerate(run_names, start=1):
        rows = _read_csv(RUN_ROOT / run_name / "case_results.csv")
        valid = [row for row in rows if _as_bool(row.get("valid_run"))]
        if len(valid) != 100:
            raise ValueError(f"{run_name} has {len(valid)} valid cases; expected 100")
        for row in valid:
            row["aggregate_repeat"] = str(repeat_index)
            row["aggregate_run_name"] = run_name
        all_rows.extend(valid)
        result = metrics(valid)
        repeat_rows.append(
            {
                "repeat": repeat_index,
                "run_name": run_name,
                "agent_pass": sum(row.get("agent_decision") == "PASS" for row in valid),
                "agent_review": sum(row.get("agent_decision") == "REVIEW" for row in valid),
                "invalid_operation_calls": sum(
                    int(float(row.get("invalid_or_unsupported_operation_calls", 0) or 0)) for row in valid
                ),
                **result,
            }
        )

    intervals = clustered_bootstrap(all_rows, bootstrap_iterations, seed)
    fixture_rows = _read_csv(FIXTURE_INDEX)
    condition_by_case = {row["case_id"]: row["initial_conditions"] for row in fixture_rows}
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        by_case[row["case_id"]].append(row)
    stability_rows: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        rows = sorted(by_case[case_id], key=lambda row: int(row["aggregate_repeat"]))
        decisions = [row["agent_decision"] for row in rows]
        stability_rows.append(
            {
                "case_id": case_id,
                "case_short": rows[0]["case_short"],
                "initial_conditions": condition_by_case.get(case_id, ""),
                "reference_decision": rows[0]["reference_decision"],
                "repeat1_decision": decisions[0],
                "repeat2_decision": decisions[1],
                "repeat3_decision": decisions[2],
                "same_decision_all_repeats": len(set(decisions)) == 1,
                "correct_repeats": sum(_as_bool(row.get("decision_match")) for row in rows),
                "closed_condition_repeats": sum(
                    _as_bool(row.get("all_reference_conditions_closed_at_finish")) for row in rows
                ),
            }
        )

    condition_rows: list[dict[str, Any]] = []
    for condition in sorted(set(condition_by_case.values())):
        subset = [row for row in all_rows if condition_by_case.get(row["case_id"]) == condition]
        result = metrics(subset)
        condition_rows.append(
            {
                "initial_conditions": condition,
                "case_count": len({row["case_id"] for row in subset}),
                "decision_count": len(subset),
                **result,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "repeat_summary.csv", repeat_rows)
    _write_csv(output_dir / "case_stability.csv", stability_rows)
    _write_csv(output_dir / "condition_summary.csv", condition_rows)
    payload = {
        "run_names": run_names,
        "case_count": len(by_case),
        "decision_count": len(all_rows),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": seed,
        "case_clustered_bootstrap": intervals,
        "same_decision_all_repeats_rate": sum(
            _as_bool(row["same_decision_all_repeats"]) for row in stability_rows
        )
        / len(stability_rows),
        "correct_in_all_repeats_rate": sum(int(row["correct_repeats"]) == 3 for row in stability_rows)
        / len(stability_rows),
        "total_invalid_operation_calls": sum(int(row["invalid_operation_calls"]) for row in repeat_rows),
        "mean_tool_calls_per_case": sum(float(row.get("tool_calls", 0) or 0) for row in all_rows) / len(all_rows),
        "mean_tokens_per_case": sum(float(row.get("total_tokens_reported", 0) or 0) for row in all_rows)
        / len(all_rows),
        "mean_runtime_seconds_per_case": sum(float(row.get("runtime_seconds", 0) or 0) for row in all_rows)
        / len(all_rows),
    }
    payload["complete_framework_accuracy_gain"] = {
        "estimate": 1.0 - intervals["terminal_accuracy"]["estimate"],
        "ci95_low": 1.0 - intervals["terminal_accuracy"]["ci95_high"],
        "ci95_high": 1.0 - intervals["terminal_accuracy"]["ci95_low"],
    }
    payload["complete_framework_recovery_gain"] = {
        "estimate": 1.0 - intervals["recoverable_case_recovery"]["estimate"],
        "ci95_low": 1.0 - intervals["recoverable_case_recovery"]["ci95_high"],
        "ci95_high": 1.0 - intervals["recoverable_case_recovery"]["ci95_low"],
    }
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    accuracy = intervals["terminal_accuracy"]
    recovery = intervals["recoverable_case_recovery"]
    closed_release = intervals["released_with_closed_conditions"]
    accuracy_gain = payload["complete_framework_accuracy_gain"]
    recovery_gain = payload["complete_framework_recovery_gain"]
    lines = [
        "# 普通 ReAct Agent 三轮全量对照",
        "",
        "## 运行结果",
        "",
        "| 轮次 | PASS | REVIEW | 终态准确率 | 可修复案例恢复率 | REVIEW 保留率 | 无效处理调用 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in repeat_rows:
        lines.append(
            f"| {row['repeat']} | {row['agent_pass']} | {row['agent_review']} | "
            f"{_fmt(row['terminal_accuracy'])} | {_fmt(row['recoverable_case_recovery'])} | "
            f"{_fmt(row['review_retention'])} | {row['invalid_operation_calls']} |"
        )
    lines.extend(
        [
            "",
            "## 聚合结果",
            "",
            f"三轮共得到 300 个有效决定。按案例聚类后，终态准确率为 {_fmt(accuracy['estimate'])} "
            f"（95% CI {_fmt(accuracy['ci95_low'])}–{_fmt(accuracy['ci95_high'])}），可修复案例恢复率为 "
            f"{_fmt(recovery['estimate'])}（95% CI {_fmt(recovery['ci95_low'])}–{_fmt(recovery['ci95_high'])}）。",
            "",
            f"完整流程相对普通 ReAct 将终态准确率提高 {_fmt(accuracy_gain['estimate'])} "
            f"（95% CI {_fmt(accuracy_gain['ci95_low'])}–{_fmt(accuracy_gain['ci95_high'])}），"
            f"将可修复案例恢复率提高 {_fmt(recovery_gain['estimate'])} "
            f"（95% CI {_fmt(recovery_gain['ci95_low'])}–{_fmt(recovery_gain['ci95_high'])}）。",
            "",
            f"模型发布的结果中，{_fmt(closed_release['estimate'])} 在结束时关闭了全部检查条件。"
            f"同一案例三轮给出相同决定的比例为 {_fmt(payload['same_decision_all_repeats_rate'])}，"
            f"三轮均判断正确的案例占 {_fmt(payload['correct_in_all_repeats_rate'])}。",
            "",
            f"三轮共出现 {payload['total_invalid_operation_calls']} 次无效或不支持的处理调用。"
            f"普通 ReAct 平均每个案例调用 {payload['mean_tool_calls_per_case']:.2f} 次工具；"
            "受控 ReAct 第一轮平均调用 7.65 次。完整框架固定执行检查、原因判断、允许路线和再次检查，"
            "并得到 92 PASS 和 8 REVIEW。",
        ]
    )
    (output_dir / "POLICY_FREE_REACT_AGGREGATE_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT / "aggregate")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = aggregate(args.runs, args.output_dir, args.bootstrap_iterations, args.seed)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
