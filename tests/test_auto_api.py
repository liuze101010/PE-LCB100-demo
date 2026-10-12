import json
from pathlib import Path

from fastapi.testclient import TestClient

from polymer_agent_demo.app import create_app
from polymer_agent_demo.snapshot import build_demo_snapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (REPOSITORY_ROOT, *REPOSITORY_ROOT.parents)
        if (candidate / "generated_code" / "md_analysis").is_dir()
    ),
    REPOSITORY_ROOT,
)

GOAL = "计算受控支化聚乙烯体系的密度与动力学，识别可复用的结构—性质趋势。"
# A polymer goal with no warn/diagnosis keywords -> no corpus filter -> all 3 deep cases.
NO_FILTER_GOAL = "计算这个聚乙烯体系并给出可复用结论"


def make_client(tmp_path):
    snapshot_path = tmp_path / "demo_snapshot.json"
    build_demo_snapshot(PROJECT_ROOT, snapshot_path)
    app = create_app(
        snapshot_path=snapshot_path,
        project_root=PROJECT_ROOT,
        force_mode="replay",
    )
    return TestClient(app)


def test_auto_run_returns_intent_effect_and_ordered_timeline(tmp_path):
    client = make_client(tmp_path)
    res = client.post("/api/auto/run", json={"goal": NO_FILTER_GOAL})
    assert res.status_code == 200
    payload = res.json()

    assert payload["mode"] == "replay"

    intent = payload["intent"]
    assert intent["structured"]["polymer_family"] == "PE"
    assert intent["structured"]["architecture"] == "unspecified"
    assert intent["evidence_scope_match"] is False
    assert 0.0 < intent["confidence"] <= 0.98
    assert intent["engine"] == "offline"  # online model is opt-in only

    timeline = payload["timeline"]
    assert timeline[0]["step"] == "自然语言目标"
    assert timeline[-1]["kind"] == "report_ready"
    # step_index is non-decreasing across the stream
    indices = [event["step_index"] for event in timeline]
    assert indices == sorted(indices)
    # 3 representative cases expanded under steps 5/6/7/8
    assert sum(1 for event in timeline if event["kind"] == "case") == 12

    effect = payload["effect_table"]
    assert any("最终发布状态" in row["metric"] for row in effect)
    assert any(row["highlight"] for row in effect)

    evidence = payload["evidence"]
    assert len(evidence["sources"]) == 9
    assert {source["source_id"] for source in evidence["sources"]} >= {
        "tri_model_audit",
        "terminal_outcomes",
    }
    assert all(src["sha256"].startswith("sha256:") for src in evidence["sources"])
    assert evidence["cohort_scale"]["publication_pass"] == 92
    assert evidence["cohort_scale"]["publication_review"] == 8


def test_auto_report_markdown_has_demo_sections_and_provenance(tmp_path):
    client = make_client(tmp_path)
    res = client.get("/api/auto/report", params={"format": "md", "goal": GOAL})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/markdown")
    disposition = res.headers["content-disposition"]
    assert "attachment" in disposition and disposition.endswith('.md"')

    body = res.text
    for heading in (
        "一、任务目标与输入意图",
        "三、Demo 输出示例",
        "六、价值与验证结论",
        "七、下一步",
    ):
        assert heading in body
    assert "submit_md_clean_engine" in body
    assert "run_postprocess_repair" in body
    assert "sha256:" in body  # provenance hashes
    assert "92 PASS / 8 review" in body


def test_auto_report_json_is_structured(tmp_path):
    client = make_client(tmp_path)
    res = client.get("/api/auto/report", params={"format": "json", "goal": NO_FILTER_GOAL})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/json")

    data = json.loads(res.text)
    assert data["schema_version"] == "polymer_agent_demo.report.v1"
    assert len(data["cases"]) == 3
    assert data["corpus"]["total"] == 100
    assert data["effect_table"]
    assert data["evidence"]["sources"]
    assert data["boundaries"]


def test_auto_report_rejects_unknown_format(tmp_path):
    client = make_client(tmp_path)
    res = client.get("/api/auto/report", params={"format": "pdf", "goal": GOAL})
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "unsupported_format"


def test_auto_run_low_confidence_on_offtopic_input(tmp_path):
    client = make_client(tmp_path)
    res = client.post("/api/auto/run", json={"goal": "今天天气怎么样"})
    assert res.status_code == 200
    intent = res.json()["intent"]
    assert intent["confidence"] < 0.6
    assert intent["note"]
    # still maps to the only frozen cohort so the replay stays runnable
    assert intent["mapped_to_cohort"] == "pelcb100"


def test_auto_run_route_source_is_frozen_in_replay(tmp_path):
    # Upgrade 2: replay mode keeps frozen routes; connected mode swaps in live ones.
    client = make_client(tmp_path)
    payload = client.post("/api/auto/run", json={"goal": GOAL}).json()
    assert payload["route_source"] == "hashed frozen replay"
    route_events = [
        e for e in payload["timeline"] if e["kind"] == "case" and e["step_index"] == 6
    ]
    assert route_events
    assert all(e["payload"]["source"] == "hashed frozen replay" for e in route_events)


def test_build_timeline_uses_live_route_overrides(tmp_path):
    from polymer_agent_demo import auto_orchestrator

    snapshot_path = tmp_path / "demo_snapshot.json"
    snap = build_demo_snapshot(PROJECT_ROOT, snapshot_path)
    intent = auto_orchestrator.parse_intent(GOAL)

    first_cid = next(iter(snap["cases"].keys()))
    overrides = {
        first_cid: {
            "module": "LIVE_MODULE",
            "tool": "live_tool",
            "parameters": {"x": 1},
            "guard": {"verdict": "pass", "requires_for_execution": ["execute=true"]},
        }
    }
    timeline = auto_orchestrator.build_timeline(snap, intent, overrides)
    route_events = [
        e for e in timeline if e["kind"] == "case" and e["step_index"] == 6
    ]
    overridden = next(e for e in route_events if e["payload"]["case_id"] == first_cid)
    assert overridden["payload"]["tool"] == "live_tool"
    assert overridden["payload"]["source"] == "live read-only MCP plan"
    others = [e for e in route_events if e["payload"]["case_id"] != first_cid]
    assert others and all(e["payload"]["source"] == "hashed frozen replay" for e in others)


def test_resolve_intent_offline_by_default():
    # Upgrade 1: live LLM intent is opt-in; without the env flag it stays offline.
    from polymer_agent_demo import auto_orchestrator
    from polymer_agent_demo import live_intent

    assert live_intent.llm_enabled() is False
    intent = auto_orchestrator.resolve_intent(GOAL)
    assert intent["engine"] == "offline"
    assert intent["structured"]["polymer_family"] == "PE"
    assert intent["structured"]["architecture"] == "LCB"
    assert intent["evidence_scope_match"] is True

    expected_families = {
        "线性聚氧化乙烯": "PEO",
        "聚氯乙烯拓扑预检": "PVC",
        "聚四氟乙烯短测试": "PTFE",
        "线性聚苯乙烯": "PS",
        "尼龙6聚酰胺链": "PA6",
        "聚苯硫醚模拟": "PPS",
        "聚二甲基硅氧烷": "PDMS",
    }
    for goal, family in expected_families.items():
        parsed = auto_orchestrator.resolve_intent(goal)
        assert parsed["structured"]["polymer_family"] == family
        assert parsed["evidence_scope_match"] is False
        assert "不构成该请求材料的计算证据" in parsed["note"]

    linear_pe = auto_orchestrator.resolve_intent("线性聚乙烯密度")
    assert linear_pe["structured"]["polymer_family"] == "PE"
    assert linear_pe["structured"]["architecture"] == "linear"
    assert linear_pe["evidence_scope_match"] is False


def test_snapshot_has_full_filterable_corpus(tmp_path):
    # Upgrade 3: the snapshot carries the full 100-case cohort for intent filtering.
    snapshot_path = tmp_path / "demo_snapshot.json"
    snap = build_demo_snapshot(PROJECT_ROOT, snapshot_path)
    corpus = snap["corpus"]
    assert len(corpus) == snap["overview"]["cohort"]["total_cases"] == 100
    assert snap["overview"]["cohort"]["audited_cases"] == 97
    for key in ("case_id", "short_id", "branch_points", "warn_sources", "final_diagnosis"):
        assert key in corpus[0]
    assert sum(1 for c in corpus if c["is_representative"]) == 3


def test_build_corpus_view_filters_by_intent(tmp_path):
    from polymer_agent_demo import auto_orchestrator as ao

    snapshot_path = tmp_path / "demo_snapshot.json"
    snap = build_demo_snapshot(PROJECT_ROOT, snapshot_path)

    def matched(goal):
        return ao.build_corpus_view(snap, ao.parse_intent(goal))

    assert matched("只看密度 WARN")["matched_count"] == 77
    assert matched("只看 MSD 动力学")["matched_count"] == 41
    assert matched("MD 输出缺失")["matched_count"] == 6
    assert matched("拟合窗口问题")["matched_count"] == 41
    assert matched("引擎敏感")["matched_count"] == 46

    nofilter = matched("你好")
    assert nofilter["filtered"] is False
    assert nofilter["matched_count"] == 100


def test_auto_run_deep_cases_follow_intent_filter(tmp_path):
    client = make_client(tmp_path)

    def run(goal):
        r = client.post("/api/auto/run", json={"goal": goal}).json()
        deep = [
            e["payload"]["short_id"]
            for e in r["timeline"]
            if e["kind"] == "case" and e["step_index"] == 5
        ]
        return r, deep

    r_density, deep_density = run("只看密度 WARN")
    assert r_density["corpus_view"]["matched_count"] == 77
    assert set(deep_density) == {"PE006", "PE002"}

    r_msd, deep_msd = run("只看 MSD 动力学")
    assert deep_msd == ["PE002"]
    # effect-table per-case rows follow the filtered deep set
    pe_rows = [row for row in r_msd["effect_table"] if "PE0" in row["metric"]]
    assert len(pe_rows) == 1


def test_live_attribution_disabled_by_default(tmp_path):
    # Upgrade 4: live tri-model re-attribution is opt-in.
    client = make_client(tmp_path)
    health = client.get("/api/health").json()
    assert health["live_attribution"] is False
    assert health["live_intent"] is False
    res = client.post(
        "/api/auto/attribution",
        json={"case_id": "lcb_pe_006_bp01_bc008_pdend_c100"},
    )
    assert res.status_code == 200
    assert res.json()["enabled"] is False


def test_live_attribution_requires_full_three_model_panel(monkeypatch):
    from polymer_agent_demo import live_attribution as la

    for key in (
        "POLYMER_AGENT_LIVE_ATTRIBUTION",
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "ZHIPUAI_API_KEY",
        "ZHIPU_API_KEY",
        "GLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("POLYMER_AGENT_LIVE_ATTRIBUTION", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-test-key")
    assert la.attribution_enabled() is False

    monkeypatch.setenv("ZHIPUAI_API_KEY", "glm-test-key")
    assert la.attribution_enabled() is True


def test_snapshot_cases_carry_evidence_packet(tmp_path):
    snapshot_path = tmp_path / "demo_snapshot.json"
    snap = build_demo_snapshot(PROJECT_ROOT, snapshot_path)
    for case in snap["cases"].values():
        assert case.get("evidence_packet")  # real evidence for live re-adjudication
        text = json.dumps(case["evidence_packet"], ensure_ascii=False)
        assert "available_repair_tools" not in text
        assert "repair_tool" not in text
        assert "primary_audit_diagnosis" not in text


def test_aggregate_votes_majority_and_disagreement():
    from polymer_agent_demo import live_attribution as la

    agree = la.aggregate_votes(
        [
            {"diagnosis": "engine_or_protocol_sensitive_result", "confidence": 0.8},
            {"diagnosis": "engine_or_protocol_sensitive_result", "confidence": 0.9},
            {"diagnosis": "fit_window_or_method_issue", "confidence": 0.6},
        ]
    )
    assert agree["final_diagnosis"] == "engine_or_protocol_sensitive_result"
    assert agree["confidence"] == 0.85
    assert agree["cross_model_disagreement"] is True  # 2 of 3

    split = la.aggregate_votes(
        [
            {"diagnosis": "a", "confidence": 0.5},
            {"diagnosis": "b", "confidence": 0.6},
        ]
    )
    assert split["final_diagnosis"] == "cross_model_disagreement"


def test_live_attribution_prompt_removes_route_and_diagnosis_hints():
    from polymer_agent_demo import live_attribution as la

    messages = la._build_messages(
        {
            "case_id": "case_001",
            "rule": {"warn_sources": ["msd_warn"]},
            "design": {"polymer_family": "PE"},
            "evidence_packet": {
                "msd_warn": {
                    "numeric_fields": {"loglog_slope": 0.2},
                    "available_repair_tools": [
                        {
                            "diagnosis": "fit_window_or_method_issue",
                            "module": "postprocess",
                            "repair_tool": "run_postprocess_repair",
                        }
                    ],
                }
            },
        },
        la._diagnosis_options({}),
    )
    payload = json.loads(messages[1]["content"])
    text = json.dumps(payload["evidence"], ensure_ascii=False)
    assert payload["evidence"]["msd_warn"]["numeric_fields"]["loglog_slope"] == 0.2
    assert "available_repair_tools" not in text
    assert "repair_tool" not in text
    assert "diagnosis" not in text


def test_real_smoke_disabled_by_default(tmp_path):
    # Upgrade 5: real-compute smoke is opt-in; default keeps the demo self-contained.
    client = make_client(tmp_path)
    health = client.get("/api/health").json()
    assert health["real_smoke"] is False
    res = client.post(
        "/api/auto/smoke",
        json={"case_id": "lcb_pe_006_bp01_bc008_pdend_c100", "mode": "reproduce"},
    )
    assert res.status_code == 200
    assert res.json()["enabled"] is False


def test_real_smoke_remote_requires_confirm(tmp_path, monkeypatch):
    # Even when enabled, a remote submit must be explicitly confirmed.
    monkeypatch.setenv("POLYMER_AGENT_REAL_SMOKE", "1")
    client = make_client(tmp_path)
    res = client.post(
        "/api/auto/smoke",
        json={
            "case_id": "lcb_pe_006_bp01_bc008_pdend_c100",
            "mode": "smoke",
            "confirm": False,
        },
    )
    assert res.status_code == 403
    assert res.json()["detail"]["code"] == "confirm_required"


def test_run_real_smoke_classify_density_matches_frozen_rule(tmp_path):
    import importlib.util
    import zipfile

    runner = tmp_path / "run_real_smoke.py"
    bundle = Path(__file__).resolve().parents[1] / "deploy" / "science_core_bundle.zip"
    with zipfile.ZipFile(bundle) as archive:
        runner.write_bytes(
            archive.read("science_core/generated_code/lcb_md_pipeline/run_real_smoke.py")
        )
    spec = importlib.util.spec_from_file_location("run_real_smoke", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    th = {"pass_low": 0.74, "warn_low": 0.66}
    assert mod.classify_density(0.7804, th)["verdict"] == "PASS"  # sc02 reproduce value
    assert mod.classify_density(0.6868, th)["verdict"] == "WARN"  # PE006 frozen value
    assert mod.classify_density(0.60, th)["tier"] == "below_warn_low"
