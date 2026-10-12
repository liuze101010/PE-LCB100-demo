"""Targeted checks for data joins, portable paths and paper-case exports."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_pages", ROOT / "scripts" / "build_pages.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_static_export_preserves_release_and_audit_scopes(tmp_path):
    result = builder.build(tmp_path / "site", "https://github.com/example/PE-LCB100-demo")
    site = Path(result["output"])
    payload = json.loads((site / "data" / "pages_data.json").read_text(encoding="utf-8"))
    assert len(payload["cases"]) == 100
    assert sum(row["terminal_verdict"] == "PASS" for row in payload["cases"]) == 92
    assert sum(row["terminal_verdict"] == "REVIEW" for row in payload["cases"]) == 8
    audited = [row for row in payload["cases"] if row["audit"]]
    assert len(audited) == 97
    assert sum(len(row["audit"]["model_votes"]) for row in audited) == 291
    assert len(payload["representative_ids"]) == 3
    for record in payload["source_files"]:
        output = site / "data" / record["file"]
        assert output.read_bytes() == (ROOT / "data" / record["file"]).read_bytes()
        assert record["sha256"] == "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()


def test_pages_work_beneath_a_github_project_path(tmp_path):
    site = tmp_path / "PE-LCB100-demo"
    builder.build(site, "https://github.com/example/PE-LCB100-demo")
    html = (site / "index.html").read_text(encoding="utf-8")
    js = (site / "assets" / "pages.js").read_text(encoding="utf-8")
    assert 'href="/assets/' not in html
    assert 'src="/assets/' not in html
    assert 'fetch("data/pages_data.json")' in js
    assert "/api/" not in js
    assert 'method: "POST"' not in js
    assert "OPENAI_API_KEY" not in js
    assert all((site / "assets" / name).is_file() for name in ("styles.css", "auto.css", "pages.css", "pages.js"))
    data = json.loads((site / "data" / "pages_data.json").read_text(encoding="utf-8"))
    assert [step["label"] for step in data["meta"]["steps"]] == [
        "自然语言目标", "设计", "构建与校验1", "模拟分析", "校验2", "审计处理", "再次校验", "归档"]
    assert all("/HEAD/" in step["source"] for step in data["meta"]["steps"])


def test_csv_and_report_export_all_cases(tmp_path):
    site = tmp_path / "site"
    builder.build(site, "https://github.com/example/demo")
    with (site / "data" / "cases.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100
    assert rows[0]["short_id"] == "PE001"
    assert rows[-1]["short_id"] == "PE100"
    assert rows[0]["terminal_verdict"] == "REVIEW"
    report = (site / "data" / "paper_replay.md").read_text(encoding="utf-8")
    assert "92 PASS / 8 REVIEW" in report
    assert "97例 / 291张模型票" in report
    assert "| PE100 |" in report
