import json
import re
from pathlib import Path

import pytest

from polymer_agent_demo.snapshot import CASE_IDS, build_demo_snapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (REPOSITORY_ROOT, *REPOSITORY_ROOT.parents)
        if (candidate / "generated_code" / "md_analysis").is_dir()
    ),
    REPOSITORY_ROOT,
)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    output = tmp_path_factory.mktemp("snapshot") / "demo_snapshot.json"
    return build_demo_snapshot(PROJECT_ROOT, output)


def test_snapshot_contains_complete_three_case_closed_loops(snapshot):
    assert set(snapshot["cases"]) == set(CASE_IDS)
    for case_id, case in snapshot["cases"].items():
        assert case["case_id"] == case_id
        assert case["design"]["validator_status"] == "pass"
        assert case["rule"]["verdict"] == "WARN"
        assert len(case["audit"]["model_votes"]) == 3
        assert case["route"]["dry_run"] is True
        assert case["route"]["guard"]["verdict"] == "pass"
        assert case["outcome"]["re_gate_status"]


def test_representative_values_and_routes_are_real(snapshot):
    pe006 = snapshot["cases"]["lcb_pe_006_bp01_bc008_pdend_c100"]
    assert pe006["route"]["module"] == "md"
    assert pe006["route"]["tool"] == "submit_md_clean_engine"
    assert pe006["route"]["parameters"]["engine_profile"] == "gromacs_2024_4_clean"
    assert pe006["route"]["parameters"]["reentry_stage"] == "md"
    assert pe006["route"]["planning_selection_status"] == "selected"
    assert pe006["metrics"]["density_g_cm3"]["before"] == pytest.approx(0.6868324926250567)
    assert pe006["metrics"]["density_g_cm3"]["after"] == pytest.approx(0.7999665922965703)
    assert pe006["terminal"]["msd_loglog_slope"] == pytest.approx(0.986748975325196)

    pe002 = snapshot["cases"]["lcb_pe_002_bp01_bc008_pdmid_c100"]
    assert pe002["route"]["module"] == "postprocess"
    assert pe002["route"]["tool"] == "run_postprocess_repair"
    assert pe002["route"]["parameters"]["repair_kind"] == "msd_window"
    assert pe002["route"]["parameters"]["window_strategy"] == "c100_aware"
    assert pe002["metrics"]["msd_loglog_slope"]["before"] == pytest.approx(1.0656169553864454)
    assert pe002["metrics"]["msd_loglog_slope"]["overlay_candidate"] == pytest.approx(1.0001259459238176)
    assert pe002["metrics"]["msd_loglog_slope"]["after"] == pytest.approx(1.0007809255823097)
    assert pe002["outcome"]["re_gate_status"] == "resolved_by_postprocess_overlay"

    pe068 = snapshot["cases"]["lcb_pe_068_bp04_bc012_pduni_c100"]
    assert pe068["route"]["module"] == "md"
    assert pe068["route"]["tool"] == "submit_md_clean_engine"
    assert pe068["metrics"]["pr_density_g_cm3"]["after"] == pytest.approx(0.7704571496610338)
    assert pe068["metrics"]["msd_loglog_slope"]["after"] == pytest.approx(0.9999554748997184)
    assert pe068["outcome"]["strict_rule_pass"] is True
    assert pe068["outcome"]["re_gate_status"] == "resolved_after_missing_output_recovery"


def test_cohort_summary_matches_final_paper_release(snapshot):
    cohort = snapshot["overview"]["cohort"]
    assert cohort["audited_cases"] == 97
    assert cohort["model_calls"] == 291
    assert cohort["complete_three_model_panels"] == 97
    assert cohort["diagnosis_distribution"] == {
        "cross_model_disagreement": 4,
        "engine_or_protocol_sensitive_result": 46,
        "fit_window_or_method_issue": 41,
        "missing_md_outputs": 6,
    }
    assert cohort["selected_route_distribution"] == {
        "audit_review": 4,
        "md": 52,
        "postprocess": 41,
    }
    assert cohort["repair_domain_distribution"] == {
        "density_clean_regate": 77,
        "msd_strict_gate_or_review": 41,
        "missing_output_backfill": 6,
    }
    assert cohort["publication_pass"] == 92
    assert cohort["publication_review"] == 8
    assert cohort["ml_usable_before"] == 53
    assert cohort["ml_usable_after"] == 92
    assert cohort["ml_usable_is_strict_pass"] is False
    assert len(snapshot["audit_ledger"]) == 97
    assert sum(len(row["model_votes"]) for row in snapshot["audit_ledger"]) == 291
    assert len(snapshot["corpus"]) == 100
    assert sum(row["terminal_verdict"] == "PASS" for row in snapshot["corpus"]) == 92
    assert sum(row["terminal_verdict"] == "REVIEW" for row in snapshot["corpus"]) == 8


def test_snapshot_preserves_recorded_model_names(snapshot):
    clients = {
        vote["client"]
        for case in snapshot["cases"].values()
        for vote in case["audit"]["model_votes"]
    }
    assert clients == {"deepseek-v4-pro", "qwen3-max", "glm-4.6"}


def test_snapshot_is_sanitized_and_source_hashed(snapshot, tmp_path):
    text = json.dumps(snapshot, ensure_ascii=False)
    forbidden = [
        r"(?i)sk-[a-z0-9]{16,}",
        r"(?i)[a-f0-9]{32}\.[A-Za-z0-9]{12,}",
        r"(?i)[A-Z]:\\Users\\",
        r"(?i)D:\\lz project",
        r"/public/home/",
        r"(?i)\b(liuz|liuze)\b",
    ]
    for pattern in forbidden:
        assert re.search(pattern, text) is None, pattern
    assert snapshot["source_manifest"]
    for source in snapshot["source_manifest"]:
        assert source["sha256"].startswith("sha256:")
        assert "\\" not in source["public_name"]
        assert "/" not in source["public_name"]


def test_builder_writes_standalone_source_manifest(tmp_path):
    output = tmp_path / "demo_snapshot.json"
    snapshot = build_demo_snapshot(PROJECT_ROOT, output)
    manifest_path = output.with_name("source_manifest.json")

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "polymer_agent_demo.source_manifest.v1"
    assert manifest["sources"] == snapshot["source_manifest"]
