/* Static paper-case replay: reads one relative JSON file, then filters locally. */
(function () {
  "use strict";

  const positions = { mid: "中部 · mid", end: "端部 · end", uni: "均匀 · uni" };
  const diagnoses = {
    missing_md_outputs: "计算输出恢复",
    fit_window_or_method_issue: "拟合窗口或后处理方法",
    engine_or_protocol_sensitive_result: "压力松弛与有效运行参数",
    cross_model_disagreement: "模型意见待复核",
  };
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
  const formatNumber = (value, digits = 4) => value == null ? "原记录缺失" : Number(value).toFixed(digits);

  function parseQuery(query) {
    const text = String(query || "").trim().toLowerCase();
    const ids = Array.from(text.matchAll(/(?:\bpe[\s_-]*|lcb_pe_)(\d{1,3})(?!\d)/g),
      (match) => "PE" + match[1].padStart(3, "0"));
    const point = text.match(/(?:bp|支化点(?:数)?[为是：:\s]*)(\d+)/) || text.match(/(\d+)\s*个?\s*支化点/);
    const carbon = text.match(/(?:bc|支链碳数[为是：:\s]*|支链\s*c)(\d+)/) || text.match(/(\d+)\s*碳\s*支链/);
    const position = text.match(/\b(mid|end|uni)\b/)?.[1] ||
      (text.includes("中部") ? "mid" : text.includes("端部") ? "end" : text.includes("均匀") ? "uni" : null);
    const status = /review|待复核|待审查/.test(text) ? "REVIEW" : /\bpass\b|通过/.test(text) ? "PASS" : null;
    const cause = /拟合|窗口/.test(text) ? "fit_window_or_method_issue" :
      /恢复|缺失|补齐/.test(text) ? "missing_md_outputs" : null;
    const audit = /审计|模型票|投票/.test(text);
    const general = /全部|所有|100|pe[-\s]?lcb|聚乙烯|支化|支链|密度|msd|动力学|结构|性质|论文|最终状态/.test(text);
    const outsideScope = /聚酰亚胺|polyimide|聚丙烯|polypropylene|聚氯乙烯|pvc|聚四氟乙烯|ptfe|聚醚|peo|聚苯乙烯|polystyrene|pdms|\bpp\b|\bpi\b|\bpa6\b/.test(text) ||
      /玻璃化|\btg\b|拉伸|弯曲|压缩强度|击穿|介电/.test(text);
    return { text, ids, points: point ? Number(point[1]) : null,
      carbons: carbon ? Number(carbon[1]) : null, position, status, cause, audit, general, outsideScope };
  }

  function filterCases(cases, query, status = "all") {
    const parsed = parseQuery(query);
    if (parsed.outsideScope) return [];
    return cases.filter((row) => {
      if (status !== "all" && row.terminal_verdict !== status) return false;
      if (parsed.status && row.terminal_verdict !== parsed.status) return false;
      if (parsed.ids.length && !parsed.ids.includes(row.short_id)) return false;
      if (parsed.points != null && row.branch_points !== parsed.points) return false;
      if (parsed.carbons != null && row.branch_carbons !== parsed.carbons) return false;
      if (parsed.position && row.branch_position !== parsed.position) return false;
      // An explicit case ID selects that case even when its goal names several checks.
      if (!parsed.ids.length && parsed.cause && row.final_diagnosis !== parsed.cause) return false;
      if (parsed.audit && !row.audit) return false;
      const structured = parsed.ids.length || parsed.points != null || parsed.carbons != null ||
        parsed.position || parsed.status || parsed.cause || parsed.audit || parsed.general;
      if (!parsed.text || structured) return true;
      const searchable = JSON.stringify(row).toLowerCase();
      const words = parsed.text.replace(/查看|查询|搜索|记录|案例|请|的/g, " ").split(/[\s,，;；]+/).filter(Boolean);
      return words.length > 0 && words.every((word) => searchable.includes(word));
    });
  }

  function summarize(cases) {
    return { total: cases.length, pass: cases.filter((row) => row.terminal_verdict === "PASS").length,
      review: cases.filter((row) => row.terminal_verdict === "REVIEW").length };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { parseQuery, filterCases, summarize, escapeHtml };
  }
  if (typeof document === "undefined") return;

  const $ = (selector) => document.querySelector(selector);
  let payload;
  let selectedId;

  const stepDescriptions = [
    "研究目标说明材料、结构与待分析性质。公开页把目标转为已有记录的筛选条件，并保留PE-LCB100的证据范围。",
    "候选清单保存100个受控支化聚乙烯设计。每例记录支化点数、支链碳数、位置与总碳数。代表案例还包含设计依据和输入哈希。",
    "原工具把候选设计转换为EMC构建输入和GROMACS坐标、拓扑。校验1检查结构与输入完整性；三个代表案例的冻结设计检查均为pass。完整实现见源码。",
    "原流程执行四阶段分子动力学，再分析密度、链尺寸、结构与动力学。页面读取已保存的模拟和分析结果；原阶段脚本及参数保留在开源科学工具中。",
    "可执行规则按已登记条件检查结果。WARN触发审计。最终记录分别保存密度、Rg、形状、RDF、接触、MSD、Fs(q,t)与链内动力学检查。",
    "冻结审计台账覆盖97例，共291张模型票。模型根据证据诊断原因，规划工具在允许的处理步骤中选择路线，再由工具检查参数。",
    "处理产物返回相同规则重新检查。PE006展示密度clean对照，PE002展示MSD拟合窗口处理，PE068展示输出恢复及规则复查。",
    "100例最终状态为92 PASS与8 REVIEW。每例保留终态、物性、检查项目与处理记录；论文数据与审计台账分别注明覆盖范围。",
  ];

  function renderWorkflow() {
    $("#workflowRail").innerHTML = payload.meta.steps.map((step, index) =>
      `<button class="progress-step" type="button" data-step="${index}"><span class="step-index">${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(step.label)}</strong></button>`).join("");
    $("#workflowDetails").innerHTML = payload.meta.steps.map((step, index) =>
      `<details id="step-${index}"><summary>${index + 1}. ${escapeHtml(step.label)}</summary><p>${stepDescriptions[index]}</p><a class="source-link" href="${escapeHtml(step.source)}" target="_blank" rel="noopener">查看该步骤源码 ↗</a></details>`).join("");
  }

  function renderRepresentatives() {
    const texts = {
      PE006: ["密度异常的同输入对照", "三模型审计指向压力松弛与有效运行参数敏感，原工具生成MD处理路线。", "density_g_cm3", "密度 · g/cm³"],
      PE002: ["MSD拟合窗口处理", "三模型审计选择后处理路线，允许的窗口扫描形成候选，并与最终重新判定结果分别保存。", "msd_loglog_slope", "MSD log–log斜率 · 无量纲"],
      PE068: ["计算输出恢复", "原输出缺失触发MD恢复路线，恢复产物随后完成后处理和规则复查。", "pr_density_g_cm3", "PR密度 · g/cm³"],
    };
    $("#representatives").innerHTML = payload.representative_ids.map((id) => {
      const row = payload.cases.find((item) => item.case_id === id);
      const [title, explanation, key, label] = texts[row.short_id];
      const metric = row.representative.metrics[key];
      return `<article class="representative-card"><span class="section-kicker">${row.short_id} · <span class="status-badge">PASS</span></span><h3>${title}</h3><p>${explanation}</p><div class="metric-pair"><span>${label}</span><strong>${formatNumber(metric.before)} → ${formatNumber(metric.after)}</strong>${metric.overlay_candidate != null ? `<span class="metric-unit">窗口候选：${formatNumber(metric.overlay_candidate)}；右侧为最终值</span>` : ""}</div><button class="text-button" type="button" data-case="${escapeHtml(id)}">查看依据与工具记录</button></article>`;
    }).join("");
  }

  function renderCases() {
    const rows = filterCases(payload.cases, $("#goalInput").value, $("#statusFilter").value);
    const summary = summarize(rows);
    $("#caseCount").textContent = `${summary.total}例 · ${summary.pass} PASS / ${summary.review} REVIEW`;
    $("#searchFeedback").textContent = rows.length ?
      `匹配${summary.total}个已有案例，其中${summary.pass} PASS、${summary.review} REVIEW。上方八步说明论文流程，下方展示匹配记录。` :
      "当前公开记录覆盖PE-LCB100。可按PE006、待复核、支化点数、支链碳数或原记录文字查看案例。";
    const visible = new Set(rows.map((row) => row.case_id));
    $("#statusGrid").innerHTML = payload.cases.map((row) =>
      `<button type="button" class="case-tile ${row.terminal_verdict === "REVIEW" ? "review" : ""} ${visible.has(row.case_id) ? "" : "filtered"} ${row.case_id === selectedId ? "selected" : ""}" data-case="${escapeHtml(row.case_id)}" aria-label="${row.short_id} ${row.terminal_verdict}" title="${row.short_id} · ${row.terminal_verdict}">${row.short_id.slice(2)}</button>`).join("");
    $("#caseRows").innerHTML = rows.length ? rows.map((row) =>
      `<tr><td><button class="case-link" type="button" data-case="${escapeHtml(row.case_id)}">${row.short_id}</button></td><td>${row.branch_points}</td><td>${row.branch_carbons}</td><td>${escapeHtml(positions[row.branch_position] || row.branch_position)}</td><td>${formatNumber(row.density_g_cm3)}</td><td>${formatNumber(row.msd_loglog_slope)}</td><td><span class="status-badge ${row.terminal_verdict === "REVIEW" ? "review" : ""}">${row.terminal_verdict}</span></td></tr>`).join("") :
      '<tr><td colspan="7">可选择上方示例目标继续查看。</td></tr>';
  }

  function detailBlock(title, value) {
    return `<details><summary>${escapeHtml(title)}</summary><pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre></details>`;
  }

  function showCase(caseId) {
    const row = payload.cases.find((item) => item.case_id === caseId);
    if (!row) return;
    selectedId = caseId;
    const representative = row.representative;
    const checks = Object.entries(row.checks || {}).map(([key, value]) => `${key}: ${value ? "通过" : "待复核"}`).join(" · ");
    $("#detailTitle").textContent = `${row.short_id} · ${row.terminal_verdict}`;
    $("#detailBody").innerHTML =
      `<div class="detail-facts"><div><strong>结构设计</strong>${row.branch_points}个支化点 · ${row.branch_carbons}碳支链<br>${escapeHtml(positions[row.branch_position])}</div><div><strong>最终密度</strong>${formatNumber(row.density_g_cm3, 6)} g/cm³</div><div><strong>最终MSD斜率</strong>${formatNumber(row.msd_loglog_slope, 6)}</div><div><strong>审计记录</strong>${row.audit ? "3张已保存模型票" : "终态记录单独归档"}</div></div>` +
      `<p><strong>终态规则：</strong>${escapeHtml(checks)}</p>` +
      `<p><strong>原记录复核原因：</strong>${escapeHtml(row.publication_state_reason)}</p>` +
      (row.audit ? `<p><strong>冻结审计归因：</strong>${escapeHtml(diagnoses[row.final_diagnosis] || row.final_diagnosis)} · <code>${escapeHtml(row.repair_tool)}</code></p>` : "") +
      `<ul class="record-list">${row.source_records.map((source) => `<li><a href="data/${source.file}" download>${source.file}</a> · ${escapeHtml(source.record)}</li>`).join("")}</ul>` +
      `<button class="text-button" id="downloadCase" type="button">下载该案例JSON</button>` +
      (representative ? detailBlock("代表案例：前后指标与来源", representative.metrics) +
        detailBlock("代表案例：设计依据与校验1", representative.design) +
        detailBlock("代表案例：工具路线与参数", representative.route) +
        detailBlock("代表案例：源记录与哈希", representative.evidence) : "") +
      (row.audit ? detailBlock("三模型原始投票与诊断依据", row.audit) : "") +
      detailBlock("候选manifest记录", row.design) + detailBlock("完整案例公开记录", row);
    $("#caseDetail").hidden = false;
    $("#downloadCase").addEventListener("click", () => {
      const objectUrl = URL.createObjectURL(new Blob([JSON.stringify(row, null, 2)], { type: "application/json" }));
      const anchor = document.createElement("a");
      anchor.href = objectUrl; anchor.download = `${row.short_id}.json`; anchor.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    });
    renderCases();
    $("#caseDetail").scrollIntoView({ block: "start" });
  }

  function renderSources() {
    $("#repositoryLink").href = payload.meta.repository;
    $("#dataVersion").textContent = `记录版本：${payload.meta.data_version} · 冻结记录生成时间：${payload.meta.recorded_at}`;
    $("#sourceFiles").innerHTML = payload.source_files.map((source) =>
      `<div class="source-row"><a href="data/${escapeHtml(source.file)}" download>${escapeHtml(source.file)}</a><code>${escapeHtml(source.sha256)}</code></div>`).join("");
    $("#sourceManifest").innerHTML = payload.source_manifest.map((source) =>
      `<div class="source-row">${escapeHtml(source.public_name)} · ${escapeHtml(source.role)}<code>${escapeHtml(source.sha256)}</code></div>`).join("");
  }

  async function bootstrap() {
    try {
      const response = await fetch("data/pages_data.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      payload = await response.json();
      renderWorkflow(); renderRepresentatives(); renderSources(); renderCases();
      $("#searchForm").addEventListener("submit", (event) => { event.preventDefault(); renderCases(); });
      $("#statusFilter").addEventListener("change", renderCases);
      $("#clearSearch").addEventListener("click", () => { $("#goalInput").value = ""; $("#statusFilter").value = "all"; renderCases(); });
      document.addEventListener("click", (event) => {
        const preset = event.target.closest("[data-goal]");
        if (preset) { $("#goalInput").value = preset.dataset.goal; $("#statusFilter").value = "all"; renderCases(); }
        const caseButton = event.target.closest("[data-case]");
        if (caseButton) showCase(caseButton.dataset.case);
        const step = event.target.closest("[data-step]");
        if (step) { const detail = $(`#step-${step.dataset.step}`); detail.open = true; detail.scrollIntoView({ block: "nearest" }); }
      });
      $("#closeDetail").addEventListener("click", () => { $("#caseDetail").hidden = true; selectedId = null; renderCases(); });
    } catch (error) {
      $("#searchFeedback").textContent = `公开记录载入失败（${error.message}）。可以使用下方CSV或JSON下载链接。`;
    }
  }
  bootstrap();
})();
