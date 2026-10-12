const test = require("node:test");
const assert = require("node:assert/strict");
const { METRICS, STOPS, normalizeOs, buildPrompt } = require("../static/pages.js");

test("the task template keeps six requested MD metrics and an explicit stop", () => {
  assert.deepEqual(METRICS, ["密度", "Rg", "Ree", "链形状", "RDF", "MSD"]);
  const prompt = buildPrompt({});
  for (const metric of METRICS) assert.ok(prompt.includes(metric));
  assert.ok(prompt.includes(STOPS.validation1));
  assert.ok(prompt.includes("[材料名称]"));
  assert.ok(prompt.includes("[服务器、CPU/GPU资源和输出目录]"));
});
test("user conditions and the selected stopping stage stay in the generated task", () => {
  const prompt = buildPrompt({ material: "测试聚合物", structure: "*CCO*", temperature: 350,
    pressure: 0, chains: 10, repeat_units: 20, resources: "用户服务器 /work/output", stop_stage: "validation2" });
  assert.ok(prompt.includes("测试聚合物"));
  assert.ok(prompt.includes("*CCO*"));
  assert.ok(prompt.includes("350 K"));
  assert.ok(prompt.includes("0 atm"));
  assert.ok(prompt.includes("10 条链"));
  assert.ok(prompt.includes("20 个重复单元"));
  assert.ok(prompt.includes(STOPS.validation2));
  assert.ok(!prompt.includes(STOPS.archive));
});
test("unknown operating systems and stopping stages use the local setup defaults", () => {
  assert.equal(normalizeOs("linux"), "linux");
  assert.equal(normalizeOs("windows"), "windows");
  assert.equal(normalizeOs("unknown"), "windows");
  assert.ok(buildPrompt({ stop_stage: "unregistered" }).includes(STOPS.validation1));
});
