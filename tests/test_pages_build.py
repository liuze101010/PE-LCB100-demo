"""Checks for guide publishing, portable URLs and unchanged optional archives."""
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


def test_homepage_is_a_deployment_guide_without_case_records(tmp_path):
    site = tmp_path / "site"
    result = builder.build(site, "https://github.com/example/PE-LCB100-demo")
    assert result["mode"] == "usage_guide"
    assert not (site / "archive").exists()
    assert not (site / "data" / "pages_data.json").exists()
    guide = json.loads((site / "data" / "guide.json").read_text(encoding="utf-8"))
    assert guide["mode"] == "usage_guide"
    assert "cases" not in guide
    assert len(guide["workflow"]) == 8
    assert ".[science]" in guide["install_commands"]["windows"]
    assert "python3.12 -m venv" in guide["install_commands"]["linux"]
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "面向聚合物分子动力学模拟的智能体框架" in html
    assert "论文" not in html
    assert "PE006" not in html
    assert "92 PASS" not in html
    assert "caseRows" not in html
    assert "promptForm" in html


def test_guide_paths_and_documentation_links_are_portable(tmp_path):
    site = tmp_path / "PE-LCB100-demo"
    builder.build(site, "https://github.com/example/PE-LCB100-demo")
    html = (site / "index.html").read_text(encoding="utf-8")
    js = (site / "assets" / "pages.js").read_text(encoding="utf-8")
    assert 'href="/assets/' not in html
    assert 'src="/assets/' not in html
    assert 'fetch("data/guide.json")' in js
    assert "/api/" not in js
    assert 'method: "POST"' not in js
    assert "OPENAI_API_KEY" not in js
    guide = json.loads((site / "data" / "guide.json").read_text(encoding="utf-8"))
    assert guide["links"]["codex"].endswith("/HEAD/docs/CODEX_SETUP.md")
    assert guide["links"]["linux_template"].endswith("/HEAD/examples/codex/linux.toml")
    assert guide["links"]["remote"].endswith("/HEAD/docs/REMOTE_SETUP.md")
    assert all((site / "assets" / name).is_file() for name in ("styles.css", "pages.css", "pages.js"))


def test_optional_archive_retains_terminal_records_votes_and_exports(tmp_path):
    site = tmp_path / "site"
    builder.build(site, "https://github.com/example/demo", include_archive=True)
    archive = site / "archive"
    payload = json.loads((archive / "data" / "pages_data.json").read_text(encoding="utf-8"))
    assert len(payload["cases"]) == 100
    assert sum(row["terminal_verdict"] == "PASS" for row in payload["cases"]) == 92
    assert sum(row["terminal_verdict"] == "REVIEW" for row in payload["cases"]) == 8
    audited = [row for row in payload["cases"] if row["audit"]]
    assert len(audited) == 97
    assert sum(len(row["audit"]["model_votes"]) for row in audited) == 291
    for record in payload["source_files"]:
        exported = archive / "data" / record["file"]
        assert exported.read_bytes() == (ROOT / "data" / record["file"]).read_bytes()
        assert record["sha256"] == "sha256:" + hashlib.sha256(exported.read_bytes()).hexdigest()
    with (archive / "data" / "cases.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 100
    assert "| PE100 |" in (archive / "data" / "paper_replay.md").read_text(encoding="utf-8")
    assert (archive / "assets" / "case_replay.js").is_file()
