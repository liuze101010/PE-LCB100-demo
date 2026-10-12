from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path


DEFAULT_CASE_ROOT = Path(
    r"${PELCB100_PROJECT_ROOT}\_runs\prep_smoke"
    r"\mcp_generic_emc_smoke_prep_smoke_20260723\cases"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "pelcb100_three_stage_ablation"
RUN_LABEL = "mcp_generic_emc_smoke_prep_smoke_20260723"


@dataclass(frozen=True)
class ExpectedCase:
    case_id: str
    polymer_group: str
    expected_status: str
    expected_reason: str = ""


EXPECTED_CASES = (
    ExpectedCase("mcp_smoke_linear_c4_c100", "poly(alpha-olefin), C4 repeat", "pass"),
    ExpectedCase("mcp_smoke_linear_c10_c100", "poly(alpha-olefin), C10 repeat", "pass"),
    ExpectedCase("mcp_pelcb_b4l4", "long-chain-branched polyethylene", "pass"),
    ExpectedCase("mcp_smoke_linear_pe_c100", "linear polyethylene", "pass"),
    ExpectedCase(
        "mcp_smoke_polyether_c100",
        "oxygen-containing polyether",
        "review",
        "Missing force field parameters",
    ),
)


def read_code(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def has_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def evaluate_case(case_root: Path, expected: ExpectedCase) -> dict[str, object]:
    case_dir = case_root / expected.case_id
    remote_dir = case_dir / "remote_build" / RUN_LABEL
    bridge_dir = case_dir / "gromacs_bridge_min_test"

    setup_code = read_code(remote_dir / "setup.exitcode")
    build_code = read_code(remote_dir / "build.exitcode")
    build_log = read_code(remote_dir / "build.log")
    emc_outputs = all(
        has_nonempty(remote_dir / f"{expected.case_id}{suffix}")
        for suffix in (".data", ".params", ".vmd", ".pdb.gz", ".psf.gz")
    )
    bridge_outputs = all(
        has_nonempty(bridge_dir / name)
        for name in (
            f"{expected.case_id}.gro",
            f"{expected.case_id}_polymer.itp",
            "topol.top",
        )
    )
    grompp_ok = has_nonempty(bridge_dir / "em_bridge_min.tpr")
    minimization_ok = all(
        has_nonempty(bridge_dir / name)
        for name in ("em_bridge_min_run.log", "em_bridge_min_run.gro")
    )

    if (
        setup_code == "0"
        and build_code == "0"
        and emc_outputs
        and bridge_outputs
        and grompp_ok
        and minimization_ok
    ):
        observed_status = "pass"
        observed_reason = "structure, topology, preprocessing and minimization completed"
    elif (
        setup_code == "0"
        and build_code not in ("", "0")
        and expected.expected_reason.lower() in build_log.lower()
    ):
        observed_status = "review"
        observed_reason = expected.expected_reason
    else:
        observed_status = "fail"
        observed_reason = "expected build or preprocessing evidence was incomplete"

    return {
        **asdict(expected),
        "setup_exitcode": setup_code,
        "build_exitcode": build_code,
        "emc_outputs_complete": emc_outputs,
        "gromacs_bridge_complete": bridge_outputs,
        "grompp_passed": grompp_ok,
        "minimization_passed": minimization_ok,
        "observed_status": observed_status,
        "observed_reason": observed_reason,
        "status_match": observed_status == expected.expected_status,
        "reason_match": (
            True
            if not expected.expected_reason
            else expected.expected_reason.lower() in observed_reason.lower()
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize real EMC/GROMACS preparation probes for representative polymers."
    )
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    rows = [evaluate_case(args.case_root, expected) for expected in EXPECTED_CASES]
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "preparation_real_build_validation.csv"
    json_path = args.output_root / "preparation_real_build_validation_summary.json"
    write_csv(csv_path, rows)

    correct = sum(bool(row["status_match"]) and bool(row["reason_match"]) for row in rows)
    summary = {
        "schema_version": "pelcb100_preparation_real_build_validation.v1",
        "case_count": len(rows),
        "correct_case_count": correct,
        "handling_accuracy": correct / len(rows),
        "supported_build_count": sum(row["observed_status"] == "pass" for row in rows),
        "review_count": sum(row["observed_status"] == "review" for row in rows),
        "all_supported_cases_preprocessed": all(
            bool(row["grompp_passed"]) and bool(row["minimization_passed"])
            for row in rows
            if row["expected_status"] == "pass"
        ),
        "case_results": rows,
    }
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if correct == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
