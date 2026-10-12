#!/usr/bin/env python3
"""Read-only onboarding checks for a configured LCB automation project."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lcb_project_config import LCBProjectConfig, load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REQUIRED_MANIFEST_COLUMNS = {"case_id"}
RECOMMENDED_MANIFEST_COLUMNS = {
    "branch_points",
    "branch_carbons",
    "branch_topology",
    "branch_position_pattern",
    "actual_total_carbons_per_chain",
    "chain_count",
    "monomer_smiles",
}


@dataclass
class Check:
    name: str
    status: str
    message: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
        }


def run(command: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout)


def ssh(host: str, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, command], timeout=timeout + 5)


def status_counts(checks: list[Check]) -> dict[str, int]:
    out = {"OK": 0, "WARN": 0, "FAIL": 0}
    for check in checks:
        out[check.status] = out.get(check.status, 0) + 1
    return out


def check_python_packages() -> list[Check]:
    checks: list[Check] = []
    for package in ["pandas", "numpy", "matplotlib"]:
        if importlib.util.find_spec(package):
            checks.append(Check(f"python:{package}", "OK", f"{package} is importable"))
        else:
            checks.append(Check(f"python:{package}", "FAIL", f"{package} is missing", "Install the Python environment used by md_analysis."))
    return checks


def check_manifest(config: LCBProjectConfig) -> tuple[list[Check], pd.DataFrame]:
    checks: list[Check] = []
    if not config.manifest_csv.exists():
        return [Check("manifest", "FAIL", f"manifest not found: {config.manifest_csv}")], pd.DataFrame()
    try:
        df = pd.read_csv(config.manifest_csv, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        return [Check("manifest", "FAIL", "manifest could not be read", str(exc))], pd.DataFrame()

    missing_required = sorted(REQUIRED_MANIFEST_COLUMNS - set(df.columns))
    if missing_required:
        checks.append(Check("manifest:columns", "FAIL", "required manifest columns are missing", ",".join(missing_required)))
    else:
        case_count = int(
            df["case_id"]
            .astype(str)
            .str.match(r"(?:lcb\d+_|lcb_[a-z]+_\d+_)", case=False)
            .sum()
        )
        status = "OK" if case_count == len(df) else "WARN"
        checks.append(Check("manifest:case_id", status, f"{case_count}/{len(df)} LCB case rows detected"))
    missing_recommended = sorted(RECOMMENDED_MANIFEST_COLUMNS - set(df.columns))
    if missing_recommended:
        checks.append(
            Check(
                "manifest:recommended_columns",
                "WARN",
                "recommended topology/chemistry columns are missing",
                ",".join(missing_recommended),
            )
        )
    if "case_id" in df.columns:
        duplicate = df[df["case_id"].duplicated(keep=False)]["case_id"].astype(str).unique().tolist()
        if duplicate:
            checks.append(Check("manifest:duplicates", "FAIL", f"{len(duplicate)} duplicate case_id values", ";".join(duplicate[:20])))
        else:
            checks.append(Check("manifest:duplicates", "OK", "case_id values are unique"))
    return checks, df


def check_local_paths(config: LCBProjectConfig, manifest: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []
    for root in config.case_roots:
        status = "OK" if root.path.exists() else "FAIL"
        checks.append(Check(f"case_root:{root.label}", status, str(root.path)))
    for idx, root in enumerate(config.bridge_roots, start=1):
        status = "OK" if root.exists() else "WARN"
        checks.append(Check(f"bridge_root:{idx}", status, str(root)))

    if not manifest.empty and "case_id" in manifest.columns:
        case_ids = manifest["case_id"].astype(str).tolist()
        found = 0
        for case_id in case_ids:
            if any((root.path / case_id).exists() for root in config.case_roots):
                found += 1
        if found == len(case_ids):
            checks.append(Check("case_roots:coverage", "OK", f"{found}/{len(case_ids)} manifest cases found in case roots"))
        elif found:
            checks.append(Check("case_roots:coverage", "WARN", f"{found}/{len(case_ids)} manifest cases found in case roots"))
        else:
            checks.append(Check("case_roots:coverage", "FAIL", "no manifest cases found in case roots"))

    for path in [config.integrated_root, config.runtime_output_dir, config.supervisor_output_dir, config.emc_output_dir]:
        parent = path.parent
        if parent.exists():
            checks.append(Check(f"output_parent:{path.name}", "OK", str(parent)))
        else:
            checks.append(Check(f"output_parent:{path.name}", "WARN", f"parent does not exist yet: {parent}"))
    return checks


def check_remote(config: LCBProjectConfig, timeout: int) -> list[Check]:
    checks: list[Check] = []
    if not shutil.which("ssh"):
        return [Check("ssh", "FAIL", "ssh executable not found on PATH")]

    remote_specs = [
        ("old_server", config.old_server, "command -v sbatch >/dev/null && command -v squeue >/dev/null && echo slurm-ok"),
        # The private-4e2f23ff.invalid entry point is a login node; GPU visibility is checked
        # by the runtime dashboard on compute-node-*.  Doctor only verifies SSH and
        # the configured GROMACS binary from the login context.
        ("new_server", config.new_server, f"test -x {config.new_gmx} && echo login-gmx-ok"),
        ("newnew_server", config.newnew_server, f"test -x {config.newnew_gmx} && command -v nvidia-smi >/dev/null && echo gpu-ok"),
    ]
    for name, host, command in remote_specs:
        try:
            proc = ssh(host, command, timeout)
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(f"remote:{name}", "FAIL", f"{host} unreachable", str(exc)))
            continue
        if proc.returncode == 0:
            checks.append(Check(f"remote:{name}", "OK", f"{host}: {proc.stdout.strip() or 'reachable'}"))
        else:
            checks.append(Check(f"remote:{name}", "FAIL", f"{host} check failed", (proc.stderr or proc.stdout).strip()[:500]))
    return checks


def write_reports(output_dir: Path, config: LCBProjectConfig, checks: list[Check]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_id": config.project_id,
        "config": {
            "manifest_csv": str(config.manifest_csv),
            "integrated_root": str(config.integrated_root),
            "runtime_output_dir": str(config.runtime_output_dir),
            "supervisor_output_dir": str(config.supervisor_output_dir),
        },
        "status_counts": status_counts(checks),
        "checks": [check.as_dict() for check in checks],
    }
    json_path = output_dir / "doctor_report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# {config.project_id.upper()} Doctor Report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Manifest: `{config.manifest_csv}`",
        f"- Integrated root: `{config.integrated_root}`",
        "",
        "## Summary",
        "",
    ]
    for status, count in payload["status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Checks", ""])
    for check in checks:
        detail = f" - {check.detail}" if check.detail else ""
        lines.append(f"- `{check.status}` **{check.name}**: {check.message}{detail}")
    md_path = output_dir / "doctor_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "lcb300.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-remote", action="store_true", help="Only run local checks.")
    parser.add_argument("--remote-timeout", type=int, default=10)
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_lcb_config(args.config)
    output_dir = args.output_dir or (SCRIPT_DIR / "output" / f"{config.project_id}_onboarding")
    checks: list[Check] = []
    checks.extend(check_python_packages())
    manifest_checks, manifest = check_manifest(config)
    checks.extend(manifest_checks)
    checks.extend(check_local_paths(config, manifest))
    if args.skip_remote:
        checks.append(Check("remote", "WARN", "remote checks skipped by --skip-remote"))
    else:
        checks.extend(check_remote(config, args.remote_timeout))

    json_path, md_path = write_reports(output_dir, config, checks)
    if not args.json_only:
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        counts = status_counts(checks)
        print(f"OK={counts.get('OK', 0)} WARN={counts.get('WARN', 0)} FAIL={counts.get('FAIL', 0)}")
    return 1 if any(check.status == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
