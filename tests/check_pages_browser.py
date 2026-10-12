"""Optional real-browser QA for a locally served static replay.

Example: python tests/check_pages_browser.py --url http://127.0.0.1:8876/site/
Requires Playwright and a locally installed Playwright browser; no model keys.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
from pathlib import Path


def check(url: str, artifacts: Path) -> dict:
    from playwright.sync_api import expect, sync_playwright

    artifacts.mkdir(parents=True, exist_ok=True)
    requests = []
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.on("request", lambda request: requests.append({"url": request.url, "method": request.method}))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        expect(page.locator("#caseRows tr")).to_have_count(100)
        expect(page.locator(".case-tile.review")).to_have_count(8)
        expect(page.locator("#workflowDetails details")).to_have_count(8)
        page.screenshot(path=str(artifacts / "desktop.png"), full_page=True)

        page.get_by_role("button", name="8例待复核", exact=True).click()
        expect(page.locator("#caseRows tr")).to_have_count(8)
        expect(page.locator("#caseCount")).to_have_text("8例 · 0 PASS / 8 REVIEW")
        page.locator("#goalInput").fill("查看4个支化点、12碳支链的案例")
        page.get_by_role("button", name="查看记录", exact=True).click()
        matched = page.locator("#caseRows tr").count()
        assert matched > 0
        for cells in page.locator("#caseRows tr").all():
            assert cells.locator("td").nth(1).inner_text() == "4"
            assert cells.locator("td").nth(2).inner_text() == "12"

        page.get_by_role("button", name="显示全部", exact=True).click()
        expect(page.locator("#caseRows tr")).to_have_count(100)
        page.locator("[data-step='5']").click()
        assert page.locator("#step-5").evaluate("element => element.open")
        page.locator("#representatives [data-case]").first.click()
        expect(page.locator("#caseDetail")).to_be_visible()
        expect(page.locator("#detailTitle")).to_have_text("PE006 · PASS")
        with tempfile.TemporaryDirectory() as download_dir:
            with page.expect_download() as download_info:
                page.get_by_role("button", name="下载该案例JSON", exact=True).click()
            destination = Path(download_dir) / "case.json"
            download_info.value.save_as(str(destination))
            assert json.loads(destination.read_text(encoding="utf-8"))["short_id"] == "PE006"
            with page.expect_download() as csv_info:
                page.get_by_role("link", name="100例CSV", exact=True).click()
            csv_destination = Path(download_dir) / "cases.csv"
            csv_info.value.save_as(str(csv_destination))
            assert len(list(csv.DictReader(io.StringIO(csv_destination.read_text(encoding="utf-8-sig"))))) == 100

        page.get_by_role("button", name="收起详情", exact=True).click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(url, wait_until="networkidle")
        expect(page.locator("#caseRows tr")).to_have_count(100)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(artifacts / "mobile.png"), full_page=True)
        assert not errors, errors
        assert all(request["method"] == "GET" for request in requests)
        assert all(request["url"].startswith(url) or request["url"].startswith("blob:") for request in requests)
        browser.close()
    return {"status": "passed", "cases": 100, "review": 8,
            "coordinate_matches": matched, "requests": len(requests),
            "screenshots": [str(artifacts / "desktop.png"), str(artifacts / "mobile.png")]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8876/site/")
    parser.add_argument("--artifacts", type=Path, default=Path(".runtime/pages_qa"))
    args = parser.parse_args()
    print(json.dumps(check(args.url, args.artifacts), ensure_ascii=False))


if __name__ == "__main__":
    main()
