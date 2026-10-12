"""Optional clean-browser QA for the static deployment and usage guide."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check(url: str, artifacts: Path) -> dict:
    from playwright.sync_api import expect, sync_playwright

    artifacts.mkdir(parents=True, exist_ok=True)
    requests, http_errors, console_errors, page_errors, failed_requests = [], [], [], [], []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100},
                                      permissions=["clipboard-read", "clipboard-write"])
        page = context.new_page()
        page.on("request", lambda request: requests.append({"url": request.url, "method": request.method}))
        page.on("response", lambda response: http_errors.append({"url": response.url, "status": response.status}) if response.status >= 400 else None)
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("requestfailed", lambda request: failed_requests.append(request.url))
        page.goto(url, wait_until="networkidle")
        expect(page.locator("h1")).to_have_text("面向聚合物分子动力学模拟的智能体框架")
        expect(page.locator(".workflow-step")).to_have_count(8)
        assert "论文" not in page.locator("body").inner_text()
        assert "PE006" not in page.locator("body").inner_text()
        expect(page.get_by_role("tab", name="Windows · PowerShell")).to_have_attribute("aria-selected", "true")
        page.get_by_role("tab", name="Linux", exact=True).click()
        assert "python3.12 -m venv" in page.locator("#installCode").inner_text()
        assert page.locator("#codexTemplateLink").get_attribute("href").endswith("/examples/codex/linux.toml")
        page.get_by_role("button", name="复制安装命令", exact=True).click()
        expect(page.locator("#copyFeedback")).to_contain_text("已复制")
        assert page.evaluate("navigator.clipboard.readText()").replace("\r\n", "\n") == page.locator("#installCode").inner_text()
        page.get_by_role("tab", name="Linux", exact=True).press("ArrowLeft")
        expect(page.get_by_role("tab", name="Windows · PowerShell")).to_have_attribute("aria-selected", "true")
        assert "py -3.12 -m venv" in page.locator("#installCode").inner_text()
        page.screenshot(path=str(artifacts / "desktop.png"), full_page=True)

        page.locator("[name=material]").fill("测试聚合物")
        page.locator("[name=structure]").fill("*CCO*")
        page.locator("[name=temperature]").fill("350")
        page.locator("[name=pressure]").fill("1")
        page.locator("[name=chains]").fill("10")
        page.locator("[name=repeat_units]").fill("20")
        page.locator("[name=resources]").fill("用户服务器与输出目录")
        page.locator("[name=stop_stage]").select_option("validation2")
        page.get_by_role("button", name="整理任务描述", exact=True).click()
        prompt = page.locator("#promptText").input_value()
        for value in ("测试聚合物", "*CCO*", "350 K", "1 atm", "10 条链", "20 个重复单元",
                      "密度", "Rg", "Ree", "链形状", "RDF", "MSD", "轨迹分析与校验2完成后停止"):
            assert value in prompt
        page.get_by_role("button", name="复制任务描述", exact=True).click()
        expect(page.locator("#copyFeedback")).to_contain_text("已复制")
        assert page.evaluate("navigator.clipboard.readText()").replace("\r\n", "\n") == prompt

        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(url, wait_until="networkidle")
        expect(page.locator(".workflow-step")).to_have_count(8)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.locator("h1").bounding_box()["x"] >= 0
        page.screenshot(path=str(artifacts / "mobile.png"), full_page=True)
        browser.close()
    assert not http_errors, http_errors
    assert not console_errors, console_errors
    assert not page_errors, page_errors
    assert not failed_requests, failed_requests
    assert all(request["method"] == "GET" and request["url"].startswith(url) for request in requests)
    report = {"status": "passed", "mode": "usage_guide", "workflow_steps": 8,
              "os_tabs": "passed", "code_copy": "passed", "prompt_metrics": 6,
              "stop_stage": "validation2", "requests": requests, "http_errors": http_errors,
              "console_errors": console_errors, "page_errors": page_errors,
              "failed_requests": failed_requests,
              "screenshots": [str(artifacts / "desktop.png"), str(artifacts / "mobile.png")]}
    (artifacts / "browser_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8876/site/")
    parser.add_argument("--artifacts", type=Path, default=Path(".runtime/pages_qa"))
    args = parser.parse_args()
    report = check(args.url, args.artifacts)
    print(json.dumps({key: report[key] for key in ("status", "mode", "workflow_steps", "os_tabs",
                                                  "code_copy", "prompt_metrics", "stop_stage", "screenshots")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
