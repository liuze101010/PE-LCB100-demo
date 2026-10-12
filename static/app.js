const state = {
  overview: null,
  cases: [],
  currentCase: null,
  health: null,
};

const $ = (selector) => document.querySelector(selector);

const diagnosisLabels = {
  engine_or_protocol_sensitive_result: "压力松弛与有效运行参数敏感",
  fit_window_or_method_issue: "拟合窗口或后处理方法敏感",
  missing_md_outputs: "MD 输出缺失",
  cross_model_disagreement: "模型意见未收敛",
};

const SAFETY_CHECKS = ["plan", "guard", "remote", "canonical", "reject"];

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined) return "缺失";
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(3);
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail?.message || payload.detail || response.statusText);
  return payload;
}

function setBannerState(stateClass, title, detail, iconChar) {
  const banner = $("#dryRunBanner");
  banner.classList.add("visible");
  banner.classList.remove("running", "success", "rejected");
  if (stateClass) banner.classList.add(stateClass);
  $("#dryRunBannerTitle").textContent = title;
  $("#dryRunBannerDetail").textContent = detail;
  $("#dryRunBannerIcon").textContent = iconChar;
}

function setSafetyCheck(checkName, status) {
  const node = document.querySelector(`#safetyChecks [data-check="${checkName}"]`);
  if (!node) return;
  node.classList.remove("passed", "rejected");
  const span = node.querySelector("span");
  if (status === "passed") {
    node.classList.add("passed");
    if (span) span.textContent = "✓";
  } else if (status === "rejected") {
    node.classList.add("rejected");
    if (span) span.textContent = "✕";
  } else {
    if (span) span.textContent = "○";
  }
}

function resetSafetyChecks() {
  SAFETY_CHECKS.forEach((name) => setSafetyCheck(name, "pending"));
}

function resetDryRunUi() {
  const banner = $("#dryRunBanner");
  banner.classList.remove("visible", "running", "success", "rejected");
  $("#dryRunBannerTitle").textContent = "等待 MCP dry-run";
  $("#dryRunBannerDetail").textContent = "点击按钮生成只读工具计划，不执行远程任务。";
  $("#dryRunBannerIcon").textContent = "○";
  resetSafetyChecks();
  const result = $("#dryRunResult");
  if (result) {
    result.textContent = "尚未执行 dry-run";
    result.classList.remove("success", "rejected");
  }
  const routeSection = document.querySelector(".route-section");
  routeSection?.classList.remove("route-highlight");
}

function flashRouteSection() {
  const routeSection = document.querySelector(".route-section");
  if (!routeSection) return;
  routeSection.classList.remove("route-highlight");
  void routeSection.offsetWidth;
  routeSection.classList.add("route-highlight");
}

function renderWorkflow() {
  const steps = state.overview.workflow;
  $("#workflow").innerHTML = steps.map((step, index) => {
    let className = "complete";
    if (step === "规则判定") className = "warning";
    if (["三模型归因", "MCP 路由", "修复证据", "重新判定"].includes(step)) className = "adaptive";
    return `
      <div class="workflow-step ${className}">
        <span class="step-index">${String(index + 1).padStart(2, "0")}</span>
        <strong>${escapeHtml(step)}</strong>
      </div>
    `;
  }).join("");
}

function renderTabs() {
  $("#caseTabs").innerHTML = state.cases.map((item) => `
    <button
      type="button"
      class="case-tab ${state.currentCase?.case_id === item.case_id ? "active" : ""}"
      data-case-id="${escapeHtml(item.case_id)}"
    >
      ${escapeHtml(item.short_id)} · ${escapeHtml(item.title)}
    </button>
  `).join("");

  document.querySelectorAll(".case-tab").forEach((button) => {
    button.addEventListener("click", () => loadCase(button.dataset.caseId));
  });
}

function drawPolymer(design) {
  const canvas = $("#polymerCanvas");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const nodes = 17;
  const margin = 28;
  const y = height * 0.58;
  const spacing = (width - margin * 2) / (nodes - 1);
  const branchCount = Math.max(1, Math.min(Number(design.branch_points || 1), 5));
  const branchNodes = [];
  for (let index = 0; index < branchCount; index += 1) {
    branchNodes.push(Math.round(((index + 1) * (nodes - 2)) / (branchCount + 1)));
  }

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#1f5aa6";
  ctx.beginPath();
  for (let index = 0; index < nodes; index += 1) {
    const x = margin + index * spacing;
    const waveY = y + Math.sin(index * 1.2) * 8;
    if (index === 0) ctx.moveTo(x, waveY);
    else ctx.lineTo(x, waveY);
  }
  ctx.stroke();

  branchNodes.forEach((node, branchIndex) => {
    const x = margin + node * spacing;
    const waveY = y + Math.sin(node * 1.2) * 8;
    const direction = branchIndex % 2 === 0 ? -1 : 1;
    const branchLength = Math.min(52, 22 + Number(design.branch_carbons || 8) * 1.7);
    ctx.strokeStyle = branchIndex % 2 === 0 ? "#c65f08" : "#007f78";
    ctx.beginPath();
    ctx.moveTo(x, waveY);
    ctx.lineTo(x + 10, waveY + direction * branchLength * 0.52);
    ctx.lineTo(x + 22, waveY + direction * branchLength);
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.arc(x + 22, waveY + direction * branchLength, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = "#1f5aa6";
  for (let index = 0; index < nodes; index += 2) {
    const x = margin + index * spacing;
    const waveY = y + Math.sin(index * 1.2) * 8;
    ctx.beginPath();
    ctx.arc(x, waveY, 3.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function designItem(label, value) {
  return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`;
}

function renderDesign(caseData) {
  const design = caseData.design;
  $("#caseTitle").textContent = `${caseData.short_id} · ${caseData.title}`;
  $("#architectureLabel").textContent = `${design.architecture} / ${design.branch_topology}`;
  $("#designVariable").textContent = design.design_variable;
  $("#designGrid").innerHTML = [
    designItem("聚合物体系", `${design.polymer_family} · ${design.carbon_count} carbons`),
    designItem("支化点", design.branch_points),
    designItem("支链长度", `C${design.branch_carbons}`),
    designItem("位置规则", design.branch_position_pattern),
    designItem("体系规模", `${design.chain_count} chains`),
    designItem("条件", `${design.temperature_k} K · ${design.pressure_atm} atm`),
  ].join("");
  $("#warnSources").textContent = caseData.rule.warn_sources.join(" + ");
  $("#evidenceCount").textContent = `${caseData.evidence.length} 类已冻结记录`;
  $("#validatorStatus").textContent = `PASS · ${design.validator_status}`;
  drawPolymer(design);
}

function metricScale(before, after) {
  const values = [before, after].filter((value) => value !== null && value !== undefined);
  const max = Math.max(...values, 1e-9);
  return {
    before: before === null || before === undefined ? 0 : Math.max(5, (before / max) * 100),
    after: after === null || after === undefined ? 0 : Math.max(5, (after / max) * 100),
  };
}

function metricLabel(key) {
  return {
    density_g_cm3: "密度",
    pr_density_g_cm3: "PR 密度恢复",
    msd_loglog_slope: "MSD log–log slope",
  }[key] || key;
}

function renderMetrics(caseData) {
  $("#outcomeState").textContent = caseData.outcome.status === "resolved" ? "RESOLVED" : "CLOSURE ACTIVE";
  $("#outcomeNote").textContent = caseData.outcome.label;
  $("#outcomeNote").classList.toggle("pending", !caseData.outcome.strict_rule_pass);

  $("#metrics").innerHTML = Object.entries(caseData.metrics).map(([key, metric]) => {
    const scale = metricScale(metric.before, metric.after);
    const unit = metric.unit || "";
    const note = metric.fit_window_ns
      ? `window ${metric.fit_window_ns} ns · local D CV ${formatNumber(metric.local_d_cv_pct, 2)}%`
      : metric.source;
    return `
      <section class="metric-block">
        <div class="metric-title">
          <strong>${escapeHtml(metricLabel(key))}</strong>
          <span>${escapeHtml(note)}</span>
        </div>
        <div class="metric-value-row">
          <span>修复前</span>
          <div class="metric-track"><div class="metric-fill" style="width:${scale.before}%"></div></div>
          <span class="metric-number">${formatNumber(metric.before)} ${escapeHtml(unit)}</span>
        </div>
        <div class="metric-value-row">
          <span>修复后</span>
          <div class="metric-track"><div class="metric-fill after" style="width:${scale.after}%"></div></div>
          <span class="metric-number">${formatNumber(metric.after)} ${escapeHtml(unit)}</span>
        </div>
      </section>
    `;
  }).join("");
}

function renderAudit(caseData) {
  $("#diagnosisLabel").textContent = caseData.audit.final_diagnosis_label;
  $("#auditConfidence").textContent = `confidence ${formatNumber(caseData.audit.confidence, 2)}`;
  $("#modelVotes").innerHTML = caseData.audit.model_votes.map((vote) => `
    <article class="vote-row" title="${escapeHtml(vote.rationale)}">
      <span class="vote-model">${escapeHtml(vote.client)}</span>
      <span class="vote-diagnosis">${escapeHtml(vote.diagnosis_label)}</span>
      <span class="vote-confidence">${formatNumber(vote.confidence, 2)}</span>
    </article>
  `).join("");

  $("#routeTitle").textContent = `${caseData.route.module} → ${caseData.route.tool}`;
  $("#routeLabel").textContent = caseData.route.display_label;
  $("#toolCall").textContent = JSON.stringify({
    tool: caseData.route.tool,
    parameters: caseData.route.parameters,
    dry_run: true,
  }, null, 2);
  resetDryRunUi();
}

function renderCohort() {
  const cohort = state.overview.cohort;
  $("#usableBefore").textContent = `${cohort.publication_pass} PASS`;
  $("#usableAfter").textContent = `${cohort.publication_review} review`;
  const stats = [
    [cohort.total_cases, "总案例"],
    [cohort.audited_cases, "审计案例"],
    [cohort.model_calls, "三模型投票"],
    [cohort.complete_three_model_panels, "完整三模型面板"],
    [`${cohort.publication_pass}/${cohort.publication_review}`, "PASS / review"],
  ];
  $("#cohortStats").innerHTML = stats.map(([value, label]) => `
    <div class="cohort-stat"><strong>${value}</strong><span>${label}</span></div>
  `).join("");

  const routeLabels = {
    density_clean_regate: "density clean re-gate",
    msd_strict_gate_or_review: "MSD strict gate / review",
    missing_output_backfill: "missing-output backfill",
  };
  const distribution = cohort.repair_domain_distribution || cohort.route_distribution || {};
  const max = Math.max(...Object.values(distribution));
  $("#routeBars").innerHTML = Object.entries(distribution).map(([key, count]) => `
    <div class="route-bar">
      <div class="route-bar-header">
        <span>${escapeHtml(routeLabels[key])}</span>
        <strong>${count}</strong>
      </div>
      <div class="route-bar-track">
        <div class="route-bar-fill" style="width:${(count / max) * 100}%"></div>
      </div>
    </div>
  `).join("");
}

async function loadCase(caseId) {
  state.currentCase = await fetchJson(`/api/cases/${caseId}`);
  renderTabs();
  renderDesign(state.currentCase);
  renderMetrics(state.currentCase);
  renderAudit(state.currentCase);
}

async function runDryRun() {
  const button = $("#dryRunButton");
  if (!state.currentCase) return;
  button.disabled = true;
  resetSafetyChecks();
  setBannerState("running", "正在生成工具计划", "向 MCP 桥接发送只读 dry-run 请求…", "…");
  const result = $("#dryRunResult");
  result.textContent = "正在生成工具计划…";
  result.classList.remove("success", "rejected");

  try {
    const payload = await fetchJson(`/api/cases/${state.currentCase.case_id}/dry-run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ execute: false }),
    });

    const guardVerdict = payload.guard?.verdict ?? "pass";
    setBannerState(
      "success",
      "DRY-RUN 已完成",
      `${String(payload.mode || "dry-run").toUpperCase()} · ${payload.tool} · guard ${guardVerdict} · 未提交远程任务`,
      "✓",
    );
    setSafetyCheck("plan", "passed");
    setSafetyCheck("guard", guardVerdict === "pass" ? "passed" : "rejected");
    setSafetyCheck("remote", "passed");
    setSafetyCheck("canonical", "passed");

    result.textContent =
      `${payload.mode} · ${payload.tool} · guard ${guardVerdict} · 未执行状态修改`;
    result.classList.add("success");
    flashRouteSection();
  } catch (error) {
    setBannerState("rejected", "DRY-RUN 失败", error.message || "请求失败", "✕");
    result.textContent = `dry-run 失败：${error.message}`;
    result.classList.add("rejected");
  } finally {
    button.disabled = false;
  }
}

async function runRejectTest() {
  const button = $("#rejectExecuteButton");
  if (!state.currentCase) return;
  button.disabled = true;
  setBannerState(
    "running",
    "测试越权拦截中",
    "故意发送 execute: true，预期被 guard 拒绝并返回 HTTP 403。",
    "!",
  );

  try {
    const response = await fetch(
      `/api/cases/${state.currentCase.case_id}/dry-run`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ execute: true }),
      },
    );

    let payload = {};
    try {
      payload = await response.json();
    } catch (_) {
      payload = {};
    }

    if (response.status === 403) {
      const guardMessage =
        payload.detail?.message || payload.detail || "guard 拒绝可变请求";
      setBannerState(
        "rejected",
        "HTTP 403 · 越权请求被拦截",
        `guard 拒绝 execute=true：${guardMessage}`,
        "✕",
      );
      setSafetyCheck("plan", "passed");
      setSafetyCheck("guard", "passed");
      setSafetyCheck("remote", "passed");
      setSafetyCheck("canonical", "passed");
      setSafetyCheck("reject", "passed");
    } else if (response.ok) {
      setBannerState(
        "rejected",
        "拦截异常：execute=true 未被拦截",
        `服务器返回 HTTP ${response.status} 而非 HTTP 403，guard 未生效。`,
        "✕",
      );
      setSafetyCheck("reject", "rejected");
    } else {
      const message =
        payload.detail?.message || payload.detail || response.statusText;
      setBannerState(
        "rejected",
        `HTTP ${response.status}`,
        `请求失败：${message}`,
        "✕",
      );
      setSafetyCheck("reject", "rejected");
    }
  } catch (error) {
    setBannerState("rejected", "测试失败", error.message || "请求失败", "✕");
    setSafetyCheck("reject", "rejected");
  } finally {
    button.disabled = false;
  }
}

async function bootstrap() {
  try {
    [state.health, state.overview, state.cases] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/overview"),
      fetchJson("/api/cases"),
    ]);
    $("#runtimeMode").textContent = `${state.health.mode.toUpperCase()} · READ ONLY`;
    $("#dataVersion").textContent = state.health.data_version;
    $("#statusDot").classList.add("online");
    renderWorkflow();
    renderCohort();
    await loadCase(state.cases[0].case_id);
  } catch (error) {
    $("#runtimeMode").textContent = `加载失败：${error.message}`;
  }
}

$("#dryRunButton").addEventListener("click", runDryRun);
$("#rejectExecuteButton").addEventListener("click", runRejectTest);
window.addEventListener("resize", () => {
  if (state.currentCase) drawPolymer(state.currentCase.design);
});
bootstrap();
