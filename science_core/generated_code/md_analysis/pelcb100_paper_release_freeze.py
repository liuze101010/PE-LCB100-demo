#!/usr/bin/env python3
"""Freeze PE-LCB100 paper artifacts into a local, hash-addressed release folder.

The script is intentionally local-only.  It never connects to remote compute
servers and never modifies running MD/postprocessing jobs.  Use ``--dry-run`` to
preview the artifact set and preflight checks without copying payload files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lcb_project_config import ROOT, SCRIPT_DIR, load_lcb_config


DEFAULT_PAPER_DIR = Path(r"${PELCB100_PAPER_ROOT}")
RELEASE_LEVELS = {"internal_snapshot", "advisor_report", "paper_freeze_candidate", "submission_candidate", "publication_release"}
PAPER_MD_FILES = [
    "01_abstract_intro.md",
    "02_results.md",
    "03_methods.md",
    "04_discussion_si.md",
    "si_protocol_details.md",
    "si_j_anomaly.md",
]
CORE_AGGREGATE_FILES = [
    "pelcb100_summary_wide.csv",
    "pelcb100_summary_wide.json",
    "pelcb100_analysis_status_matrix.csv",
    "provenance_cases.csv",
    "provenance_cases.json",
    "provenance_nodes.csv",
    "provenance_edges.csv",
    "automation_metrics.csv",
    "automation_metrics.md",
    "project_report.md",
    "ml_features.csv",
    "ml_targets.csv",
    "ml_quality_mask.csv",
    "ml_dataset_manifest.json",
]
SUPERVISOR_FILES = [
    "supervisor_heartbeat.json",
    "supervisor_state.json",
    "supervisor_last_decision.json",
    "supervisor_decisions.md",
    "supervisor_events.csv",
    "supervisor_disk_status.md",
    "runtime_eta_summary.csv",
    "runtime_eta_summary.md",
    "runtime_eta_calibration.json",
    "runtime_eta_calibration_cases.csv",
    "recovery_classification.csv",
    "recovery_classification.md",
]
RUNTIME_FILES = [
    "runtime_cases.csv",
    "runtime_active_processes.csv",
    "runtime_backlog_submit.csv",
    "runtime_duplicates.csv",
    "runtime_gpu_snapshot.csv",
    "runtime_manifest.json",
]
SCRIPT_HASH_GLOBS = [
    "lcb300_*.py",
    "pelcb100_*.py",
    "build_lcb300_overview.py",
    "lcb_publication_assets.py",
    "lcb_report.py",
    "lcb_project_config.py",
    "analyze_*.py",
    "compute_*.py",
    "merge_*.py",
    "aggregate_*.py",
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, cwd=str(ROOT), timeout=timeout)


def relative_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def iter_files(root: Path, suffixes: set[str] | None = None) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if "__pycache__" in parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        files.append(path)
    return sorted(files)


def add_file(artifacts: list[dict[str, Any]], src: Path, root: Path, rel_dst: str, artifact_type: str, required: bool = False) -> None:
    exists = src.exists() and src.is_file()
    artifacts.append(
        {
            "artifact_type": artifact_type,
            "required": required,
            "exists": exists,
            "source_path": str(src),
            "source_rel": relative_label(src) if exists else str(src),
            "frozen_path": str(root / rel_dst),
            "relative_frozen_path": rel_dst.replace("\\", "/"),
            "size_bytes": src.stat().st_size if exists else "",
            "sha256": sha256_file(src) if exists else "",
            "copied": False,
        }
    )


def copy_artifacts(artifacts: list[dict[str, Any]], *, dry_run: bool) -> None:
    if dry_run:
        return
    for row in artifacts:
        if not row.get("exists"):
            continue
        src = Path(str(row["source_path"]))
        dst = Path(str(row["frozen_path"]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        row["copied"] = True


def artifact_plan(config, paper_dir: Path, root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    add_file(artifacts, Path(__file__).resolve(), root, "code/pelcb100_paper_release_freeze.py", "freeze_script", required=True)
    add_file(artifacts, SCRIPT_DIR / "configs" / "pelcb100.json", root, "config/pelcb100.json", "config", required=True)
    add_file(artifacts, config.manifest_csv, root, "manifest/manifest.csv", "manifest", required=True)
    for mdp in sorted((ROOT / "generated_code" / "lcb_md_pipeline" / "mdp").glob("*.mdp")):
        add_file(artifacts, mdp, root, f"protocol/mdp/{mdp.name}", "mdp", required=True)
    for name in PAPER_MD_FILES:
        add_file(artifacts, paper_dir / name, root, f"manuscript/{name}", "manuscript", required=True)
    paper_asset_root = config.output_root / "pelcb100_paper"
    for src in iter_files(paper_asset_root):
        rel = src.relative_to(paper_asset_root).as_posix()
        add_file(artifacts, src, root, f"paper_assets/{rel}", "paper_asset")
    aggregate = config.integrated_output_dir
    for name in CORE_AGGREGATE_FILES:
        add_file(artifacts, aggregate / name, root, f"integrated/aggregate/{name}", "integrated", required=name.startswith("ml_") or name == "pelcb100_summary_wide.csv")
    for src in iter_files(aggregate / "plots", {".png", ".pdf", ".svg", ".csv", ".json", ".md"}):
        rel = src.relative_to(aggregate / "plots").as_posix()
        add_file(artifacts, src, root, f"integrated/plots/{rel}", "integrated_plot")
    for name in SUPERVISOR_FILES:
        add_file(artifacts, config.supervisor_output_dir / name, root, f"runtime/supervisor/{name}", "supervisor")
    for name in RUNTIME_FILES:
        add_file(artifacts, config.runtime_output_dir / name, root, f"runtime/dashboard/{name}", "runtime")
    for src in sorted(path for path in config.emc_output_dir.iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}):
        add_file(artifacts, src, root, f"runtime/emc_orchestrator/{src.name}", "emc_orchestrator")
    return artifacts


def script_hashes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in SCRIPT_HASH_GLOBS:
        for path in sorted(SCRIPT_DIR.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            rows.append({"script": relative_label(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def git_snapshot(root: Path) -> dict[str, Any]:
    commit = run(["git", "rev-parse", "HEAD"], timeout=20)
    branch = run(["git", "branch", "--show-current"], timeout=20)
    status = run(["git", "status", "--short"], timeout=20)
    diff_name = run(["git", "diff", "--name-only"], timeout=20)
    payload = {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "",
        "branch": branch.stdout.strip() if branch.returncode == 0 else "",
        "status_returncode": status.returncode,
        "status_short": status.stdout,
        "dirty": bool(status.stdout.strip()),
        "diff_files": [line.strip() for line in diff_name.stdout.splitlines() if line.strip()] if diff_name.returncode == 0 else [],
    }
    (root / "git_status.txt").write_text(status.stdout + status.stderr, encoding="utf-8")
    return payload


def quality_counts(config) -> dict[str, Any]:
    wide = read_csv(config.integrated_summary_csv)
    runtime = read_csv(config.runtime_output_dir / "runtime_cases.csv")
    duplicates = read_csv(config.runtime_output_dir / "runtime_duplicates.csv")
    out: dict[str, Any] = {
        "manifest_cases": int(len(read_csv(config.manifest_csv))),
        "integrated_rows": int(len(wide)),
        "runtime_rows": int(len(runtime)),
        "duplicate_rows": int(len(duplicates)),
    }
    if not wide.empty:
        for col in ["publication_grade", "ml_usable", "rdf_publication_usable"]:
            if col in wide.columns:
                out[f"{col}_true"] = int(wide[col].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
        for col in [item for item in wide.columns if item.startswith("status_")]:
            out[f"{col}_counts"] = wide[col].fillna("MISSING").astype(str).value_counts().to_dict()
    if not runtime.empty and "runtime_status" in runtime.columns:
        out["runtime_status_counts"] = runtime["runtime_status"].fillna("").astype(str).value_counts().to_dict()
    return out


def preflight_checks(config, paper_dir: Path, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    missing_required = [row for row in artifacts if row.get("required") and not row.get("exists")]
    checks.append(
        {
            "check": "required_artifacts_exist",
            "status": "PASS" if not missing_required else "FAIL",
            "detail": "; ".join(str(row.get("source_path")) for row in missing_required),
        }
    )
    duplicates = read_csv(config.runtime_output_dir / "runtime_duplicates.csv")
    checks.append(
        {
            "check": "runtime_duplicates_empty",
            "status": "PASS" if duplicates.empty else "FAIL",
            "detail": f"rows={len(duplicates)}",
        }
    )
    for name in ["ml_features.csv", "ml_targets.csv", "ml_quality_mask.csv", "ml_dataset_manifest.json"]:
        path = config.integrated_output_dir / name
        checks.append({"check": f"ml_output_exists:{name}", "status": "PASS" if path.exists() else "FAIL", "detail": str(path)})
    refs = []
    for src in [paper_dir / "01_abstract_intro.md", paper_dir / "03_methods.md", paper_dir / "si_protocol_details.md"]:
        if src.exists():
            refs.extend([line for line in src.read_text(encoding="utf-8").splitlines() if "[PLACEHOLDER]" in line])
    checks.append(
        {
            "check": "placeholder_lines_recorded",
            "status": "WARN" if refs else "PASS",
            "detail": f"placeholder_lines={len(refs)}",
        }
    )
    return checks


def refresh_assets(config_path: Path) -> list[dict[str, Any]]:
    commands = [
        [sys.executable, str(SCRIPT_DIR / "build_lcb300_overview.py"), "--config", str(config_path)],
        [sys.executable, str(SCRIPT_DIR / "lcb_publication_assets.py"), "--config", str(config_path)],
        [sys.executable, str(SCRIPT_DIR / "lcb_report.py"), "--config", str(config_path)],
    ]
    rows: list[dict[str, Any]] = []
    for command in commands:
        proc = run(command, timeout=600)
        rows.append(
            {
                "command": " ".join(command),
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        )
        if proc.returncode != 0:
            break
    return rows


def write_release_notes(path: Path, *, config, level: str, dry_run: bool, counts: dict[str, Any], git: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    failing = [row for row in checks if row["status"] == "FAIL"]
    warnings = [row for row in checks if row["status"] == "WARN"]
    lines = [
        f"# {config.project_id.upper()} Paper Release Freeze",
        "",
        f"- Generated: `{now_iso()}`",
        f"- Release level: `{level}`",
        f"- Dry run: `{dry_run}`",
        f"- Git commit: `{git.get('commit', '')}`",
        f"- Git dirty: `{git.get('dirty', False)}`",
        f"- Preflight FAIL/WARN: `{len(failing)}/{len(warnings)}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Freeze Policy",
            "",
            "- Paper numbers should be read from this freeze directory, not live output directories.",
            "- This freeze is local-only and does not submit, stop, restart, or collect remote jobs.",
            "- Failed/recovered cases are preserved as provenance; the freeze does not erase runtime history.",
            "- Viscosity, entanglement, and active-learning candidate submission remain outside the default PE-LCB100 release gate.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "configs" / "pelcb100.json")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--release-level", choices=sorted(RELEASE_LEVELS), default="paper_freeze_candidate")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write manifests/checks but do not copy payload files.")
    parser.add_argument("--refresh-assets", action="store_true", help="Rebuild overview/publication assets/report before freezing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if args.refresh_assets:
        refresh_rows = refresh_assets(config_path)
    else:
        refresh_rows = []
    config = load_lcb_config(config_path)
    stamp = now_stamp()
    root = args.output_dir or (config.output_root / f"{config.project_id}_paper_releases" / f"{stamp}_{args.release_level}")
    if args.dry_run and args.output_dir is None:
        root = config.output_root / f"{config.project_id}_paper_releases" / f"{stamp}_{args.release_level}_preview"
    root.mkdir(parents=True, exist_ok=True)

    artifacts = artifact_plan(config, args.paper_dir, root)
    checks = preflight_checks(config, args.paper_dir, artifacts)
    copy_artifacts(artifacts, dry_run=args.dry_run)
    counts = quality_counts(config)
    git = git_snapshot(root)
    scripts = script_hashes()

    write_csv(
        root / "source_hashes.csv",
        artifacts,
        ["artifact_type", "required", "exists", "source_path", "source_rel", "frozen_path", "relative_frozen_path", "size_bytes", "sha256", "copied"],
    )
    write_csv(root / "script_hashes.csv", scripts, ["script", "size_bytes", "sha256"])
    write_csv(root / "preflight_checks.csv", checks, ["check", "status", "detail"])
    write_csv(root / "refresh_assets.csv", refresh_rows, ["command", "returncode", "stdout", "stderr"])
    write_release_notes(root / "release_notes.md", config=config, level=args.release_level, dry_run=args.dry_run, counts=counts, git=git, checks=checks)
    manifest = {
        "schema_version": "pelcb100_paper_release_freeze.v1",
        "generated_at": now_iso(),
        "project_id": config.project_id,
        "release_level": args.release_level,
        "dry_run": args.dry_run,
        "refresh_assets": bool(args.refresh_assets),
        "freeze_root": str(root),
        "paper_dir": str(args.paper_dir),
        "config": str(config_path),
        "quality_counts": counts,
        "git": git,
        "preflight": {
            "fail_count": sum(1 for row in checks if row["status"] == "FAIL"),
            "warn_count": sum(1 for row in checks if row["status"] == "WARN"),
        },
        "artifact_count": len(artifacts),
        "copied_artifact_count": sum(1 for row in artifacts if row.get("copied")),
        "script_hash_count": len(scripts),
        "outputs": {
            "source_hashes_csv": str(root / "source_hashes.csv"),
            "script_hashes_csv": str(root / "script_hashes.csv"),
            "preflight_checks_csv": str(root / "preflight_checks.csv"),
            "release_notes_md": str(root / "release_notes.md"),
            "git_status_txt": str(root / "git_status.txt"),
        },
    }
    write_json(root / "freeze_manifest.json", manifest)
    print(f"Wrote PE-LCB100 paper freeze manifest: {root / 'freeze_manifest.json'}")
    if manifest["preflight"]["fail_count"]:
        print(f"WARNING: preflight failures={manifest['preflight']['fail_count']} (see preflight_checks.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
