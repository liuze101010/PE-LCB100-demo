const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { filterCases, summarize } = require("../static/case_replay.js");
const snapshot = JSON.parse(fs.readFileSync(path.join(__dirname, "../data/demo_snapshot.json"), "utf8"));
const cases = snapshot.corpus.map((row) => ({ ...row,
  audit: snapshot.audit_ledger.find((record) => record.case_id === row.case_id) || null }));

test("the complete terminal cohort and audit ledger keep distinct scopes", () => {
  assert.deepEqual(summarize(filterCases(cases, "查看PE-LCB100全部案例及最终状态")), { total: 100, pass: 92, review: 8 });
  assert.equal(filterCases(cases, "查看审计记录").length, 97);
  assert.equal(filterCases(cases, "查看待复核案例").length, 8);
  assert.equal(filterCases(cases, "通过").length, 92);
});
test("free text selects actual IDs and combined molecular coordinates", () => {
  assert.equal(filterCases(cases, "查看PE006密度修复记录")[0].short_id, "PE006");
  assert.equal(filterCases(cases, "PE002 PE068").length, 2);
  assert.equal(filterCases(cases, "lcb_pe_002_bp01_bc008_pdmid_c100")[0].short_id, "PE002");
  const found = filterCases(cases, "查看4个支化点、12碳支链的案例");
  assert.ok(found.length > 0);
  assert.ok(found.every((row) => row.branch_points === 4 && row.branch_carbons === 12));
  assert.ok(filterCases(cases, "bp02 bc008 mid").every((row) => row.branch_points === 2 && row.branch_carbons === 8 && row.branch_position === "mid"));
});
test("an unsupported material returns an empty match and final-state filters combine", () => {
  assert.equal(filterCases(cases, "聚酰亚胺玻璃化温度预测").length, 0);
  assert.equal(filterCases(cases, "预测100种聚酰亚胺结构性质").length, 0);
  assert.equal(filterCases(cases, "PE-LCB介电性能").length, 0);
  assert.equal(filterCases(cases, "", "REVIEW").length, 8);
  assert.equal(filterCases(cases, "PE006", "REVIEW").length, 0);
});
