// auto.js — drive the NL -> auto-orchestration page.
// The 8 closed-loop steps (must match the snapshot overview.workflow order).
const WORKFLOW_STEPS = [
  "读懂目标",
  "文献依据",
  "候选结构",
  "规则检查",
  "三模型判因",
  "定位修复",
  "修复",
  "复核通过",
];

const STAGE_DELAY_MS = 430;
const CASE_DELAY_MS = 320;

const state = {
  health: null,
  currentGoal: "",
  running: false,
};

const $ = (selector) => document.querySelector(selector);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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
  if (!response.ok) {
    throw new Error(payload.detail?.message || payload.detail || response.statusText);
  }
  return payload;
}

function renderProgressRail() {
  $("#autoProgress").innerHTML = WORKFLOW_STEPS.map(
    (label, index) => `
      <div class="progress-step" data-step="${index + 1}">
        <span class="step-index">${String(index + 1).padStart(2, "0")}</span>
        <strong>${escapeHtml(label)}</strong>
      </div>
    `,
  ).join("");
}

function setRail(stepIndex, { allDone = false } = {}) {
  document.querySelectorAll(".progress-step").forEach((node) => {
    const idx = Number(node.dataset.step);
    node.classList.remove("active", "done");
    if (allDone || idx < stepIndex) {
      node.classList.add("done");
    } else if (idx === stepIndex) {
      node.classList.add("active");
    }
  });
}

function renderIntent(intent) {
  const s = intent.structured;
  const engine = intent.engine || "offline";
  const confidence = Number(intent.confidence);
  const summary =
    `引擎 ${escapeHtml(engine)} · ${escapeHtml(s.polymer_family)}-${escapeHtml(s.architecture)} · ` +
    `观测量 ${escapeHtml(s.observables.join(", "))} · 任务 ${escapeHtml(s.task_type)}` +
    (s.condition_filters.length ? ` · 条件 ${escapeHtml(s.condition_filters.join(", "))}` : "") +
    (intent.note ? `（${escapeHtml(intent.note)}）` : "");
  $("#intentSummary").textContent = summary;

  const card = $("#intentCard");
  const badge = $("#intentConfidence");
  const warn = $("#intentWarning");
  card.classList.remove("intent-low", "intent-high");
  badge.classList.remove("low", "high");
  badge.hidden = false;
  badge.textContent = `confidence ${intent.confidence}`;

  if (confidence < 0.6 || intent.evidence_scope_match === false) {
    // Low match or cross-material replay -> make the evidence boundary obvious.
    card.classList.add("intent-low");
    badge.classList.add("low");
    warn.hidden = false;
    warn.textContent = intent.evidence_scope_match === false
      ? "⚠ 请求材料不在 PE-LCB100 冻结证据范围内；下方内容仅为对照回放。"
      : "⚠ 这句和聚合物计算目标关联较弱,已按默认 PE-LCB100 案例集演示。";
  } else {
    if (confidence >= 0.85) {
      card.classList.add("intent-high");
      badge.classList.add("high");
    }
    warn.hidden = true;
    warn.textContent = "";
  }
}

function renderCorpus(cv) {
  if (!cv) return;
  $("#corpusCount").textContent = `匹配 ${cv.matched_count} / ${cv.total}`;
  const diag = Object.entries(cv.by_diagnosis || {})
    .map(([k, v]) => `${escapeHtml(k)}=${v}`)
    .join(" · ");
  $("#corpusFilter").textContent =
    `筛选「${cv.filter_label}」` + (diag ? `,判定原因分布:${diag}` : "");
  const head =
    "<thead><tr><th>案例</th><th>支化(点/链长/位置)</th><th>判定原因</th><th>置信</th><th>警示项</th></tr></thead>";
  const rows = (cv.cases || [])
    .map(
      (c) => `
        <tr${c.is_representative ? ' class="highlight"' : ""}>
          <td>${escapeHtml(c.short_id)}</td>
          <td class="num">bp${escapeHtml(c.branch_points)} · C${escapeHtml(c.branch_carbons)} · ${escapeHtml(c.branch_position)}</td>
          <td>${escapeHtml(c.final_diagnosis_label)}</td>
          <td class="num">${escapeHtml(c.confidence)}</td>
          <td>${escapeHtml((c.warn_sources || []).join(", "))}</td>
        </tr>`,
    )
    .join("");
  $("#corpusTable").innerHTML = `${head}<tbody>${rows}</tbody>`;
  $("#corpusPanel").hidden = false;
}

function entryDetailHtml(event) {
  const p = event.payload || {};
  if (event.kind === "case" && event.step_index === 5) {
    const votes = (p.model_votes || [])
      .map((v) => `${escapeHtml(v.client)} ${escapeHtml(v.diagnosis_label)}(${v.confidence})`)
      .join("；");
    // Offline demo: never surface the live re-run / real-smoke buttons (they
    // would call the audit models / submit a real cluster job).
    const live = "";
    const smoke = "";
    return (
      `<div class="entry-detail"><span>最终归因 <b>${escapeHtml(p.final_diagnosis_label)}</b> ` +
      `(confidence ${p.confidence}, ${escapeHtml(p.selected_by)})</span><span>${votes}</span></div>` +
      live +
      smoke
    );
  }
  if (event.kind === "case" && event.step_index === 6) {
    return (
      `<div class="entry-detail"><span><code>${escapeHtml(p.module)} → ${escapeHtml(p.tool)}</code> · ` +
      `guard ${escapeHtml(p.guard)}${p.source ? " · " + escapeHtml(p.source) : ""}</span>` +
      `<span>参数 <code>${escapeHtml(JSON.stringify(p.parameters))}</code></span>` +
      `<span>执行需 ${escapeHtml((p.requires_for_execution || []).join(", "))}</span></div>`
    );
  }
  if (event.kind === "case" && event.step_index === 7) {
    return (
      `<div class="entry-detail entry-metric"><span>${escapeHtml(p.metric_label)}: ` +
      `<b>${escapeHtml(p.before)}</b> → <b>${escapeHtml(p.after)}</b></span>` +
      (p.source ? `<span>${escapeHtml(p.source)}</span>` : "") +
      `</div>`
    );
  }
  if (event.kind === "case" && event.step_index === 8) {
    return (
      `<div class="entry-detail"><span>re-gate: ${escapeHtml(p.re_gate_status)} · ` +
      `strict_rule_pass=${p.strict_rule_pass}</span><span>${escapeHtml(p.label)}</span></div>`
    );
  }
  return "";
}

function appendStreamEntry(event) {
  const stream = $("#autoStream");
  const kindClass =
    event.kind === "report_ready" ? "report" : event.kind === "case" ? "case" : "stage";
  const node = document.createElement("article");
  node.className = `stream-entry ${kindClass}`;
  node.innerHTML =
    `<div class="entry-head">` +
    `<span class="entry-title">${escapeHtml(event.title)}</span>` +
    `<span class="entry-step">${String(event.step_index).padStart(2, "0")} · ${escapeHtml(event.step)}</span>` +
    `</div>` +
    (event.narration ? `<p class="entry-narration">${escapeHtml(event.narration)}</p>` : "") +
    entryDetailHtml(event);
  stream.appendChild(node);
  stream.scrollTop = stream.scrollHeight;
}

function renderEffectTable(rows) {
  const head =
    `<thead><tr><th>指标</th><th>修复前</th><th>修复后</th><th>变化</th></tr></thead>`;
  const body = rows
    .map(
      (row) => `
        <tr class="${row.highlight ? "highlight" : ""}">
          <td>${escapeHtml(row.metric)}</td>
          <td class="num">${escapeHtml(row.before)}</td>
          <td class="num">${escapeHtml(row.after)}</td>
          <td class="delta">${escapeHtml(row.delta)}</td>
        </tr>`,
    )
    .join("");
  $("#effectTable").innerHTML = `${head}<tbody>${body}</tbody>`;
  $("#effectPanel").hidden = false;
}

function renderEvidence(evidence) {
  const scale = evidence.cohort_scale;
  const scaleCells = [
    [`${scale.total_cases}`, "案例数"],
    [`${scale.model_calls}`, "三模型投票"],
    [`${scale.complete_three_model_panels}`, "完整三模型面板"],
    [`${scale.publication_pass} / ${scale.publication_review}`, "通过 / 待复核"],
  ]
    .map(([value, label]) => `<div><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`)
    .join("");
  const sources = evidence.sources
    .map(
      (src) => `
        <div class="source-row">
          <span class="source-name">${escapeHtml(src.public_name)}</span>
          <span class="source-hash">${escapeHtml(src.role)} · ${escapeHtml(src.sha256)}</span>
        </div>`,
    )
    .join("");
  $("#evidenceBlock").innerHTML =
    `<div class="evidence-scale">${scaleCells}</div>` +
    `<p class="evidence-validation">${escapeHtml(evidence.validation)}</p>` +
    `<div class="source-list">${sources}</div>`;
  $("#evidencePanel").hidden = false;
}

function enableExports() {
  $("#exportBar").hidden = false;
  $("#exportMdButton").disabled = false;
  $("#exportJsonButton").disabled = false;
}

function resetOutputs() {
  $("#autoStream").innerHTML = "";
  $("#corpusPanel").hidden = true;
  $("#effectPanel").hidden = true;
  $("#evidencePanel").hidden = true;
  $("#exportBar").hidden = true;
  $("#exportMdButton").disabled = true;
  $("#exportJsonButton").disabled = true;
  setRail(0);
}

async function runAuto() {
  if (state.running) return;
  const goal = $("#goalInput").value.trim();
  state.running = true;
  state.currentGoal = goal;
  $("#autoRunButton").disabled = true;
  $("#streamState").textContent = "运行中";
  resetOutputs();

  try {
    const result = await fetchJson("/api/auto/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal }),
    });
    renderIntent(result.intent);
    renderCorpus(result.corpus_view);

    for (const event of result.timeline) {
      if (event.kind === "report_ready") {
        setRail(0, { allDone: true });
        appendStreamEntry(event);
        renderEffectTable(result.effect_table);
        renderEvidence(result.evidence);
        enableExports();
      } else {
        setRail(event.step_index);
        appendStreamEntry(event);
      }
      await sleep(event.kind === "case" ? CASE_DELAY_MS : STAGE_DELAY_MS);
    }
    $("#streamState").textContent = "已完成";
  } catch (error) {
    $("#streamState").textContent = "运行失败";
    appendStreamEntry({
      kind: "report_ready",
      step_index: 0,
      step: "错误",
      title: "运行失败",
      narration: error.message || "请求失败",
      payload: {},
    });
  } finally {
    state.running = false;
    $("#autoRunButton").disabled = false;
  }
}

async function exportReport(fmt) {
  const ext = fmt === "json" ? "json" : "md";
  const version = state.health?.data_version || "report";
  const url = `/api/auto/report?format=${encodeURIComponent(fmt)}&goal=${encodeURIComponent(state.currentGoal)}`;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = `polymer_agent_demo_report_${version}.${ext}`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(objectUrl);
  } catch (error) {
    $("#streamState").textContent = `导出失败：${error.message}`;
  }
}

async function runAttribution(caseId, button) {
  const result = document.querySelector(`.live-attrib-result[data-for="${caseId}"]`);
  if (!result) return;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "三模型运行中…";
  result.innerHTML = "";
  try {
    const payload = await fetchJson("/api/auto/attribution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case_id: caseId }),
    });
    if (payload.enabled === false) {
      result.innerHTML = `<span class="attrib-note">${escapeHtml(payload.reason)}</span>`;
      return;
    }
    const fresh = (payload.votes || [])
      .map(
        (v) =>
          `<div class="attrib-vote"><span>${escapeHtml(v.client)}</span>` +
          `<span>${escapeHtml(v.diagnosis_label)}</span><span class="num">${escapeHtml(v.confidence)}</span></div>`,
      )
      .join("");
    const frozenVotes = (payload.frozen?.votes || [])
      .map((v) => `${escapeHtml(v.client)} ${escapeHtml(v.diagnosis_label)}(${escapeHtml(v.confidence)})`)
      .join("；");
    result.innerHTML =
      `<div class="attrib-head">LIVE 重判：<b>${escapeHtml(payload.final_diagnosis_label)}</b> · ` +
      `${escapeHtml(payload.agreement)} · conf ${escapeHtml(payload.confidence)} · ` +
      `${escapeHtml(payload.latency_s)}s · ${escapeHtml(payload.engine)}</div>` +
      `<div class="attrib-votes">${fresh}</div>` +
      `<div class="attrib-frozen">冻结对照：${escapeHtml(payload.frozen?.final_diagnosis_label || "")}` +
      `(conf ${escapeHtml(payload.frozen?.confidence)}) — ${frozenVotes}</div>`;
  } catch (error) {
    result.innerHTML = `<span class="attrib-note">重跑失败：${escapeHtml(error.message)}</span>`;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function pollSmoke(jobId, resultEl) {
  for (let i = 0; i < 200; i += 1) {
    await sleep(3000);
    let s;
    try {
      s = await fetchJson(`/api/auto/smoke/${jobId}`);
    } catch (_) {
      continue;
    }
    if (s.status === "done") {
      const r = s.result || {};
      const scale = r.is_smoke ? "smoke 尺度·未收敛" : "reproduce·精确值";
      resultEl.innerHTML =
        `<div class="attrib-head">真实 smoke 完成 · 密度 <b>${escapeHtml(r.density_g_per_cc)}</b> g/cm³ · ` +
        `${escapeHtml(r.verdict)}（${scale}）</div>`;
      return;
    }
    if (s.status === "failed") {
      resultEl.innerHTML = `<span class="attrib-note">smoke 失败：${escapeHtml(s.error || ("exit " + s.returncode))}</span>`;
      return;
    }
    resultEl.innerHTML = `<div class="attrib-head">真实 smoke ${escapeHtml(s.status)}…(远程作业，约数分钟)</div>`;
  }
  resultEl.innerHTML = `<span class="attrib-note">smoke 轮询超时</span>`;
}

async function runSmoke(caseId, button) {
  const resultEl = document.querySelector(`.smoke-result[data-for="${caseId}"]`);
  if (!resultEl) return;
  if (!window.confirm("将向远程集群提交一个真实 smoke 作业（消耗共享算力，约数分钟，非收敛值）。确认继续？")) return;
  button.disabled = true;
  resultEl.innerHTML = `<div class="attrib-head">提交中…</div>`;
  try {
    const payload = await fetchJson("/api/auto/smoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case_id: caseId, mode: "smoke", confirm: true }),
    });
    if (payload.enabled === false) {
      resultEl.innerHTML = `<span class="attrib-note">${escapeHtml(payload.reason)}</span>`;
      return;
    }
    if (payload.error) {
      resultEl.innerHTML = `<span class="attrib-note">${escapeHtml(payload.error)}</span>`;
      return;
    }
    await pollSmoke(payload.job_id, resultEl);
  } catch (error) {
    resultEl.innerHTML = `<span class="attrib-note">提交失败：${escapeHtml(error.message)}</span>`;
  } finally {
    button.disabled = false;
  }
}

async function bootstrap() {
  renderProgressRail();
  try {
    state.health = await fetchJson("/api/health");
    $("#runtimeMode").textContent = `${state.health.mode.toUpperCase()} · READ ONLY`;
    $("#dataVersion").textContent = state.health.data_version;
    $("#statusDot").classList.add("online");
  } catch (error) {
    $("#runtimeMode").textContent = `加载失败：${error.message}`;
  }

  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#goalInput").value = chip.dataset.goal || "";
      $("#goalInput").focus();
    });
  });

  $("#autoRunButton").addEventListener("click", runAuto);
  $("#goalInput").addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      runAuto();
    }
  });
  $("#exportMdButton").addEventListener("click", () => exportReport("md"));
  $("#exportJsonButton").addEventListener("click", () => exportReport("json"));
  $("#autoStream").addEventListener("click", (event) => {
    const attrib = event.target.closest(".live-attrib-btn");
    if (attrib) {
      runAttribution(attrib.dataset.caseId, attrib);
      return;
    }
    const smoke = event.target.closest(".smoke-btn");
    if (smoke) runSmoke(smoke.dataset.caseId, smoke);
  });

  const firstPreset = document.querySelector(".preset-chip");
  if (firstPreset && !$("#goalInput").value) {
    $("#goalInput").value = firstPreset.dataset.goal || "";
  }
}

bootstrap();
