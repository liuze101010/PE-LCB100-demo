from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_platform_exposes_automatic_builder_and_force_field_preview():
    html = (ROOT / "static" / "platform.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "platform.js").read_text(encoding="utf-8")

    assert 'id="preview-polymer-setup"' in html
    assert 'name="repeat_unit_smiles"' in html
    assert 'id="polymer-setup-result"' in html
    assert "/api/v1/polymer-setup/preview" in js
    assert "请求材料不在 PE-LCB100 冻结证据范围内" in (ROOT / "static" / "auto.js").read_text(encoding="utf-8")
    assert "自动选择构建方法和力场" in html

WORKFLOW_STEPS = (
    "读懂目标",
    "文献依据",
    "候选结构",
    "规则检查",
    "三模型判因",
    "定位修复",
    "修复",
    "复核通过",
)


def _read_static():
    html = (ROOT / "static" / "auto.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "auto.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "auto.css").read_text(encoding="utf-8")
    return html, js, css


def test_auto_page_has_nl_input_presets_steps_and_export():
    html, js, css = _read_static()
    combined = "\n".join((html, js, css))

    for label in WORKFLOW_STEPS:
        assert label in combined

    assert 'id="goalInput"' in html
    assert 'id="presetExamples"' in html
    assert 'id="autoRunButton"' in html
    assert 'id="exportMdButton"' in html
    assert 'id="exportJsonButton"' in html
    assert "修复效果" in html
    assert "数据来源" in html

    assert "92 通过 / 8 待复核" in combined
    assert "不提交计算" in combined
    assert "linear-gradient" not in css


def test_auto_page_links_shared_and_local_assets():
    html, _js, _css = _read_static()
    assert "/assets/styles.css" in html
    assert "/assets/auto.css" in html
    assert "/assets/auto.js" in html


def test_auto_stream_and_export_feedback_present_in_js():
    _html, js, _css = _read_static()
    assert "/api/auto/run" in js
    assert "/api/auto/report" in js
    assert "report_ready" in js
    assert "renderEffectTable" in js


def test_auto_low_confidence_warning_is_wired():
    html, js, css = _read_static()
    assert 'id="intentWarning"' in html
    assert "intent-low" in js
    assert ".intent-card.intent-low" in css
    assert "linear-gradient" not in css


def test_auto_corpus_panel_is_wired():
    html, js, _css = _read_static()
    assert 'id="corpusPanel"' in html
    assert 'id="corpusTable"' in html
    assert "匹配到的案例" in html
    assert "renderCorpus" in js
    assert "corpus_view" in js


def test_auto_live_attribution_button_is_wired():
    _html, js, css = _read_static()
    assert "live-attrib-btn" in js
    assert "/api/auto/attribution" in js
    assert "runAttribution" in js
    assert ".mini-btn" in css


def test_auto_real_smoke_button_is_wired():
    _html, js, _css = _read_static()
    assert "smoke-btn" in js
    assert "/api/auto/smoke" in js
    assert "runSmoke" in js
