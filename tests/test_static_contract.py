from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_single_screen_ui_contains_required_workflow_and_boundaries():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

    combined = "\n".join((html, js, css))
    for label in (
        "自然语言目标",
        "文献先验",
        "候选验证",
        "规则判定",
        "三模型归因",
        "MCP 路由",
        "重新判定",
        "92 PASS",
        "8 review",
    ):
        assert label in combined

    assert "execute=true" in combined
    assert "@media" in css
    assert "linear-gradient" not in css


def test_dry_run_has_visible_feedback_near_the_trigger():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

    assert 'id="dryRunBanner"' in html
    assert 'id="rejectExecuteButton"' in html
    assert 'id="safetyChecks"' in html
    assert "DRY-RUN 已完成" in js
    assert "正在生成工具计划" in js
    assert "HTTP 403" in js
    assert "execute: true" in js
    assert ".dry-run-banner.visible" in css
    assert ".route-section.route-highlight" in css
