"""Build a portable, frozen paper-case replay for GitHub Pages.

Uses only Python's standard library and the shipped public JSON records.
It never imports the application, initializes a database, or contacts a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "demo_snapshot.json",
    "terminal_release_outcomes.json",
    "pelcb100_design_manifest.json",
    "source_manifest.json",
)
CSV_FIELDS = (
    "case_id", "short_id", "branch_points", "branch_carbons",
    "branch_position", "carbon_count", "terminal_verdict",
    "density_g_cm3", "msd_loglog_slope", "final_diagnosis",
    "repair_tool", "publication_state_reason",
)
STEP_SOURCES = (
    ("自然语言目标", "auto_orchestrator.py"),
    ("设计", "science_core/generated_code/md_analysis/polymer_design_validator.py"),
    ("构建与校验1", "science_core/generated_code/md_analysis/emc_builder_registry.py"),
    ("模拟分析", "science_core/generated_code/lcb_md_pipeline/"),
    ("校验2", "science_core/generated_code/md_analysis/polymer_qc_rules.py"),
    ("审计处理", "science_core/generated_code/md_analysis/pelcb_warn_audit_llm.py"),
    ("再次校验", "science_core/generated_code/md_analysis/pelcb_repair_loop.py"),
    ("归档", "snapshot.py"),
)


def load_records(root: Path = ROOT) -> dict:
    records = {
        name: json.loads((root / "data" / name).read_text(encoding="utf-8"))
        for name in SOURCE_FILES
    }
    snapshot = records["demo_snapshot.json"]
    terminal = records["terminal_release_outcomes.json"]
    manifest = records["pelcb100_design_manifest.json"]
    ids = lambda rows: {row["case_id"] for row in rows}
    terminal_ids = ids(terminal["cases"])
    if len(terminal["cases"]) != 100 or len(terminal_ids) != 100:
        raise ValueError("Expected 100 unique terminal paper cases")
    if ids(snapshot["corpus"]) != terminal_ids or ids(manifest["candidates"]) != terminal_ids:
        raise ValueError("Snapshot, terminal records and design manifest must have the same case IDs")
    counts = {state: sum(row["terminal_verdict"] == state for row in terminal["cases"])
              for state in ("PASS", "REVIEW")}
    if counts != {"PASS": 92, "REVIEW": 8}:
        raise ValueError("Frozen release records must contain 92 PASS and 8 REVIEW")
    ledger = snapshot["audit_ledger"]
    votes = sum(len(row.get("model_votes", [])) for row in ledger)
    if len(ledger) != 97 or len(ids(ledger)) != 97 or votes != 291:
        raise ValueError("Frozen audit ledger must contain 97 cases and 291 model votes")
    return records


def public_payload(records: dict, repository: str, revision: str = "HEAD") -> dict:
    snapshot = records["demo_snapshot.json"]
    terminal = {row["case_id"]: row for row in records["terminal_release_outcomes.json"]["cases"]}
    candidates = {row["case_id"]: row for row in records["pelcb100_design_manifest.json"]["candidates"]}
    audit = {row["case_id"]: row for row in snapshot["audit_ledger"]}
    cases = []
    for row in snapshot["corpus"]:
        case_id = row["case_id"]
        cases.append({**row, **terminal[case_id], "design": candidates[case_id],
                      "audit": audit.get(case_id),
                      "representative": snapshot["cases"].get(case_id),
                      "source_records": [
                          {"file": "demo_snapshot.json", "record": f"corpus / {case_id}"},
                          {"file": "terminal_release_outcomes.json", "record": f"cases / {case_id}"},
                          {"file": "pelcb100_design_manifest.json", "record": f"candidates / {case_id}"},
                      ]})
    cases.sort(key=lambda row: row["case_seq"])
    source_files = [
        {"file": name, "sha256": "sha256:" + hashlib.sha256(
            (ROOT / "data" / name).read_bytes()).hexdigest()}
        for name in SOURCE_FILES
    ]
    repository = repository.rstrip("/")
    return {
        "schema_version": "polymer_agent.public_pages.v1",
        "meta": {"title": "Validating LLM Agents for Reliable Polymer Simulations",
                 "data_version": snapshot["data_version"],
                 "recorded_at": snapshot["generated_at_utc"],
                 "mode": "frozen_paper_replay", "repository": repository,
                 "revision": revision, "case_count": 100, "pass": 92, "review": 8,
                 "audited_cases": 97, "model_votes": 291,
                 "steps": [{"label": label, "source": repository +
                            ("/tree/" if path.endswith("/") else "/blob/") + revision + "/" + path}
                           for label, path in STEP_SOURCES]},
        "cases": cases,
        "representative_ids": list(snapshot["cases"]),
        "source_files": source_files,
        "source_manifest": records["source_manifest.json"]["sources"],
    }


def write_markdown(payload: dict, destination: Path) -> None:
    meta = payload["meta"]
    lines = [f"# {meta['title']}", "", "论文案例回放；所有数值来自随包保存的公开记录。", "",
             "100例最终状态：92 PASS / 8 REVIEW。审计台账范围：97例 / 291张模型票。", "",
             "密度单位为 g/cm³；MSD log–log斜率为无量纲量。", "",
             "| 案例 | 支化点 | 支链碳数 | 位置 | 最终状态 | 密度 (g/cm³) | MSD斜率 |",
             "|---|---:|---:|---|---|---:|---:|"]
    for row in payload["cases"]:
        density = row.get("density_g_cm3")
        slope = row.get("msd_loglog_slope")
        lines.append(f"| {row['short_id']} | {row['branch_points']} | {row['branch_carbons']} | "
                     f"{row['branch_position']} | {row['terminal_verdict']} | "
                     f"{density:.6f} | {slope:.6f} |" if density is not None and slope is not None
                     else f"| {row['short_id']} | {row['branch_points']} | {row['branch_carbons']} | "
                          f"{row['branch_position']} | {row['terminal_verdict']} | {density} | {slope} |")
    lines.extend(["", "## 数据来源", ""])
    for record in payload["source_files"]:
        lines.append(f"- `{record['file']}`：`{record['sha256']}`")
    lines.extend(["", f"[源码]({meta['repository']})", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def build(output: Path, repository: str, revision: str = "HEAD") -> dict:
    records = load_records()
    payload = public_payload(records, repository, revision)
    output = output.resolve()
    if output == ROOT or ROOT in output.parents and output.name in {"data", "static", "scripts"}:
        raise ValueError("Use a separate output directory, such as site/")
    assets = output / "assets"
    data = output / "data"
    assets.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    for filename in ("styles.css", "auto.css", "pages.css", "pages.js"):
        shutil.copy2(ROOT / "static" / filename, assets / filename)
    shutil.copy2(ROOT / "static" / "pages.html", output / "index.html")
    (output / ".nojekyll").write_text("", encoding="utf-8")
    (data / "pages_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / "data" / name, data / name)
    with (data / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["cases"])
    write_markdown(payload, data / "paper_replay.md")
    return {"output": str(output), "cases": 100, "pass": 92, "review": 8,
            "audited_cases": 97, "model_votes": 291}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    parser.add_argument("--repository", default="https://github.com/" + os.environ.get(
        "GITHUB_REPOSITORY", "liuze101010/PE-LCB100-demo"))
    parser.add_argument("--revision", default="HEAD", help="GitHub source revision, default: HEAD")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.repository, args.revision), ensure_ascii=False))


if __name__ == "__main__":
    main()
