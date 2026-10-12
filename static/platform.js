const state = {
  csrf: sessionStorage.getItem("csrf") || "",
  user: null,
  health: null,
  clusters: [],
  projects: [],
  runs: [],
  reviews: [],
  releasesByProject: new Map(),
  designDraft: [],
  designPlanning: null,
  modelKeys: null,
  clusterSetupProgress: new Map(),
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatError(detail) {
  if (typeof detail === "string") {
    const map = {
      authentication_required: "请先登录。",
      session_expired: "登录已过期，请重新登录。",
      csrf_check_failed: "页面授权令牌失效，请刷新后重试。",
      executor_disabled: "执行器未启用：当前只能做本机验证，不会提交真实 Slurm 作业。",
      cluster_requires_admin_authorization: "集群连接需要管理员授权后才能预检。",
      cluster_setup_required: "服务器还没准备好，请先完成第 1 步。",
      model_keys_required: "请保存当前阶段所需的模型密钥。自然语言设计使用 OpenAI 密钥。",
      remote_provision_failed: "远端平台组件安装失败，请查看环境发现结果或联系集群管理员。",
      cluster_autoconfigure_failed: "无法自动识别服务器资源。请确认 SSH 能登录；如果仍失败，再打开“连接有问题时再打开”。",
      host_key_scan_failed: "无法自动读取主机指纹。请确认服务器地址、端口和网络可达，或手动粘贴 SHA256 指纹。",
      public_key_install_failed: "无法自动登录服务器。请填写服务器密码，或确认本机已经能免密 SSH 登录。",
      design_model_unavailable: "规划模型尚待配置，或本次设计输出需要重新校验。请查看平台配置和调用记录。",
      invalid_design_count: "设计数量必须写在 1–100 之间，例如“设计 50 个支化聚乙烯”。",
      admin_required: "该操作需要管理员权限。",
      not_found: "记录不存在，或你没有访问权限。",
    };
    return map[detail] || detail;
  }
  if (detail && typeof detail === "object") {
    if (detail.code) return formatError(detail.code);
    if (detail.message) return detail.message;
    const first = Array.isArray(detail) ? detail[0] : null;
    if (first?.msg) return first.msg;
  }
  return JSON.stringify(detail);
}

async function api(path, options = {}) {
  const { _csrfRetried = false, ...requestOptions } = options;
  const method = requestOptions.method || "GET";
  const headers = { ...(requestOptions.headers || {}) };
  if (!(requestOptions.body instanceof FormData)) headers["Content-Type"] = "application/json";
  if (method !== "GET") headers["X-CSRF-Token"] = state.csrf;
  const response = await fetch(path, { credentials: "same-origin", ...requestOptions, method, headers });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = response.statusText;
    }
    if (response.status === 403 && detail === "csrf_check_failed" && !_csrfRetried) {
      const sessionResponse = await fetch("/api/v1/auth/me", { credentials: "same-origin" });
      if (sessionResponse.ok) {
        const session = await sessionResponse.json();
        state.csrf = session.csrf_token || "";
        sessionStorage.setItem("csrf", state.csrf);
        return api(path, { ...requestOptions, _csrfRetried: true });
      }
    }
    throw new Error(formatError(detail));
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("json") ? response.json() : response.text();
}

function flash(message, bad = false) {
  const el = $("#flash");
  el.textContent = message;
  el.className = `flash${bad ? " bad" : ""}`;
  el.style.display = "block";
  clearTimeout(flash.timer);
  flash.timer = setTimeout(() => { el.style.display = "none"; }, 4800);
}

function badge(value) {
  const labels = {
    ready: "可用", locked: "已准备", validated: "已检查", pass: "完成",
    review: "需查看", review_required: "需查看", failed: "失败", cancelled: "已取消",
    awaiting_approval: "待提交", needs_admin: "需管理员", draft: "草稿", planned: "准备中",
    queued: "排队中", running: "计算中", collecting: "整理中", postprocessing: "整理中",
    evaluating: "检查中", repairing: "处理中", resolved: "已处理", open: "待处理",
    authorized: "已连接", installing: "准备中", unverified: "待连接",
  };
  const text = escapeHtml(labels[value] || value || "未知");
  const cls = value === "ready" || value === "locked" || value === "validated" || value === "pass" ? "ok"
    : value === "failed" || value === "cancelled" ? "danger"
    : value === "review" || value === "review_required" || value === "awaiting_approval" || value === "needs_admin" ? "warn"
    : "muted";
  return `<span class="badge ${cls}">${text}</span>`;
}

function short(value, n = 10) {
  return value ? `${String(value).slice(0, n)}...` : "-";
}

function parseSshCommand(value) {
  let text = String(value || "").trim();
  if (!text) return null;
  text = text.replace(/^ssh\s+/i, "").trim();
  let port = null;
  const portMatch = text.match(/(?:^|\s)-p\s+(\d{1,5})(?:\s|$)/);
  if (portMatch) {
    port = Number(portMatch[1]);
    text = text.replace(portMatch[0], " ").trim();
  }
  const target = text.split(/\s+/).find((part) => part.includes("@")) || text.split(/\s+/)[0];
  const match = target.match(/^([A-Za-z0-9_.-]+)@([A-Za-z0-9_.:-]+)$/);
  if (!match) throw new Error("SSH 登录命令格式应类似：ssh user@cluster.example.edu");
  let host = match[2];
  const bracket = host.match(/^\[([^\]]+)\](?::(\d+))?$/);
  if (bracket) {
    host = bracket[1];
    if (bracket[2]) port = Number(bracket[2]);
  } else {
    const hostPort = host.match(/^([^:]+):(\d{1,5})$/);
    if (hostPort) {
      host = hostPort[1];
      port = Number(hostPort[2]);
    }
  }
  return { username: match[1], host, port: port || 22 };
}

function applyQuickSshToForm() {
  const form = $("#cluster-form");
  const quick = form?.elements.quick_ssh?.value;
  if (!quick?.trim()) return null;
  const parsed = parseSshCommand(quick);
  form.elements.remote_username.value = parsed.username;
  form.elements.host.value = parsed.host;
  form.elements.port.value = String(parsed.port);
  if (!form.elements.remote_root.value || form.elements.remote_root.value === "/home/user/polymer-agent") {
    form.elements.remote_root.value = `/home/${parsed.username}/polymer-agent`;
  }
  if (form.elements.name.value === "lab-gpu") {
    form.elements.name.value = parsed.host.replace(/[^A-Za-z0-9_.-]+/g, "-").slice(0, 48) || "cluster";
  }
  return parsed;
}

function card(title, meta, actions = "") {
  return `<article class="item"><div><div class="item-title">${title}</div><div class="meta">${meta}</div></div><div class="actions">${actions}</div></article>`;
}

function empty(text) {
  return `<div class="empty">${escapeHtml(text)}</div>`;
}

const placementToCoarse = {
  "mid": "mid", "near-end": "end", "far-end": "end", "near-start": "end",
  "uniform": "uni", "two-ends": "term", "mid-pair": "db", "H-asymmetric": "asym",
  "clustered": "clus", "dumbbell-far": "db", "two-ends+mid": "term",
  "three-point": "tri", "end-cluster": "clus", "pompom-dense": "term",
  "comb-uniform-4": "uni", "pompom-4": "term", "cluster-4": "clus", "gradient": "grad",
  "comb-uniform-5": "uni", "pompom-5": "term", "cluster-5": "clus", "gradient-5": "grad",
};

const placementLabels = {
  "mid": "中心支化",
  "near-end": "靠近链端",
  "far-end": "远离链端",
  "near-start": "靠近起始端",
  "uniform": "沿主链均匀分布",
  "two-ends": "两端支化",
  "mid-pair": "中心成对支化",
  "H-asymmetric": "不对称 H 型",
  "clustered": "局部簇状支化",
  "dumbbell-far": "远端哑铃型",
  "two-ends+mid": "两端加中心支化",
  "three-point": "三点分布",
  "end-cluster": "端部簇状支化",
  "pompom-dense": "密集 pom-pom",
  "comb-uniform-4": "四支链梳状均匀",
  "pompom-4": "四支链 pom-pom",
  "cluster-4": "四支链簇状",
  "gradient": "梯度分布",
  "comb-uniform-5": "五支链梳状均匀",
  "pompom-5": "五支链 pom-pom",
  "cluster-5": "五支链簇状",
  "gradient-5": "五支链梯度",
};

function branchWord(count) {
  return `${count} 个支化点`;
}

function branchPositionText(candidate) {
  const positions = candidate.branch_positions || [];
  if (!positions.length) return "待验证";
  return positions.map((position) => `第 ${position} 个乙烯单元`).join("、");
}

function experimentalSummary(candidate) {
  const b = Number(candidate.branch_points || 0);
  const l = Number(candidate.branch_carbons || 0);
  const repeats = Number(candidate.backbone_repeat_units || 0);
  const placement = placementLabels[candidate.placement_code] || candidate.placement_code || candidate.branch_distribution_code || "注册放置方式";
  return `主链约 ${repeats} 个乙烯重复单元，带 ${branchWord(b)}，每条支链为 C${l} 烷基；放置方式为${placement}。`;
}

function structureSvg(candidate) {
  const repeats = Number(candidate.backbone_repeat_units || 1);
  const branchLength = Math.min(38, 12 + Number(candidate.branch_carbons || 0));
  const branches = (candidate.branch_positions || []).map((position, index) => {
    const x = 30 + (270 * Number(position)) / Math.max(repeats, 1);
    const up = index % 2 === 0;
    const y2 = up ? 70 - branchLength : 70 + branchLength;
    const labelY = up ? y2 - 8 : y2 + 16;
    return `<line x1="${x.toFixed(1)}" y1="70" x2="${x.toFixed(1)}" y2="${y2}" class="branch-line"/><circle cx="${x.toFixed(1)}" cy="70" r="4" class="branch-node"/><text x="${x.toFixed(1)}" y="${labelY}" text-anchor="middle">C${candidate.branch_carbons}</text>`;
  }).join("");
  return `<svg class="structure-svg" viewBox="0 0 330 140" role="img" aria-label="${escapeHtml(candidate.case_id)} 支化聚乙烯结构示意"><line x1="30" y1="70" x2="300" y2="70" class="backbone-line"/><circle cx="30" cy="70" r="3" class="end-node"/><circle cx="300" cy="70" r="3" class="end-node"/>${branches}<text x="30" y="124">PE 主链：${repeats} 个乙烯重复单元</text><text x="300" y="124" text-anchor="end">总碳 C${candidate.target_total_carbons}</text></svg>`;
}

function renderStructurePreview(candidates, strategy = "") {
  const target = $("#structure-preview");
  if (!target) return;
  if (!candidates?.length) {
    target.innerHTML = "";
    return;
  }
  target.innerHTML = `<div class="preview-head"><div><strong>候选结构预览（${candidates.length}）</strong><span>${escapeHtml(strategy)}</span></div><span class="tag">未锁定草案</span></div><div class="structure-grid">${candidates.map((candidate) => `
    <article class="structure-card">
      <div class="structure-title"><strong>${escapeHtml(candidate.case_id)}</strong><span>${escapeHtml(placementLabels[candidate.placement_code] || candidate.topology || candidate.placement_code)}</span></div>
      ${structureSvg(candidate)}
      <p class="structure-summary">${escapeHtml(experimentalSummary(candidate))}</p>
      <div class="structure-facts"><span>${branchWord(candidate.branch_points)}</span><span>支链 C${candidate.branch_carbons}</span><span>总碳 C${candidate.target_total_carbons}</span></div>
      <div class="branch-position-box"><strong>支化位点</strong><span>${escapeHtml(branchPositionText(candidate))}</span></div>

      <details><summary>查看具体结构参数</summary><div class="mono">位置编码：${escapeHtml(candidate.placement_code)}</div><div class="mono">支化位置：${escapeHtml((candidate.branch_positions || []).join(", "))}</div><div class="mono smiles">结构式：${escapeHtml(candidate.macro_smiles)}</div></details>
    </article>`).join("")}</div>`;
}

function renderPolymerSetup(result) {
  const target = $("#polymer-setup-result");
  if (!target) return;
  const setup = result.automatic_setup || {};
  const architecture = setup.architecture_selection?.architecture || "未识别";
  const builder = setup.builder_selection?.builder_backend || "未找到";
  const forceField = setup.force_field_selection?.force_field || "未找到";
  const probeRequired = Boolean(setup.force_field_selection?.parameterization_probe_required);
  const qualification = setup.force_field_selection?.scientific_qualification_status;
  const next = result.next_action === "automatic_remote_parameterization_probe"
    ? "服务器将自动测试分子构建和参数是否完整"
    : result.next_action === "submit_md"
      ? "可以进入计算"
      : "需要人工补充构建方法或力场";
  const qualificationText = qualification === "validated_pelcb100_scope"
    ? "已在 PE-LCB100 范围内验证"
    : qualification === "external_reference_qualification_required"
      ? "用于正式结论前还需核对外部参考"
      : "";
  target.hidden = false;
  target.innerHTML = `<strong>自动配置结果</strong>
    <div class="structure-facts">
      <span>结构类型：${escapeHtml(architecture)}</span>
      <span>构建方法：${escapeHtml(builder)}</span>
      <span>力场：${escapeHtml(forceField)}</span>
    </div>
    <p>${escapeHtml(next)}${probeRequired ? "。" : ""}${qualificationText ? ` ${escapeHtml(qualificationText)}。` : ""}</p>`;
}

async function boot() {
  try {
    state.health = await api("/api/v1/health");
    state.user = await api("/api/v1/auth/me");
    state.csrf = state.user.csrf_token || state.csrf;
    sessionStorage.setItem("csrf", state.csrf);
    showWorkspace();
  } catch {
    $("#login-card").hidden = false;
  }
}

function showWorkspace() {
  $("#login-card").hidden = true;
  $("#workspace").hidden = false;
  $("#logout").hidden = false;
  $("#identity").innerHTML = `${escapeHtml(state.user.username)} ${badge(state.user.role)}`;
  renderHealth();
  refreshAll();
}

function renderHealth() {
  const el = $("#health-pill");
  if (!state.health) {
    el.textContent = "控制端状态未知";
    el.className = "pill muted";
    return;
  }
  if (state.health.executor_enabled) {
    el.textContent = "真实执行器已启用";
    el.className = "pill ok";
  } else {
    el.textContent = "本机验证模式：不会提交 Slurm";
    el.className = "pill warn";
  }
}

function renderOverview() {}

function renderOnboarding() {
  const hasConnection = state.clusters.length > 0;
  const hasReady = state.clusters.some((x) => x.status === "ready");
  const hasClosedLoopKeys = Boolean(state.modelKeys?.ready_for_closed_loop);
  const canDesign = Boolean(state.modelKeys?.ready_for_design);
  const steps = [
    ["1", "连接服务器", hasReady, hasConnection ? "正在准备服务器" : "填写 SSH 登录命令"],
    ["2", "填写密钥", hasClosedLoopKeys, "设计使用规划密钥，审计使用三模型密钥"],
    ["3", "设计并提交", state.projects.length > 0, "输入一句话生成结构"],
    ["4", "等待结果", state.runs.length > 0, "提交后可关闭页面"],
  ];
  $("#onboarding-progress").innerHTML = `<div class="section-head"><div><h2>使用步骤</h2></div></div><div class="setup-steps">${steps.map(([number, title, done, note]) => `<article class="setup-step ${done ? "done" : ""}"><strong>${number}. ${title}</strong><span>${done ? "已完成" : note}</span></article>`).join("")}</div>`;
  const designButton = $(`[data-panel="projects-panel"]`);
  if (designButton) {
    designButton.disabled = false;
    designButton.title = "可导入候选并完成本地校验";
  }
  const form = $("#project-form");
  if (form) {
    Array.from(form.elements).forEach((element) => { element.disabled = false; });
    const draftButton = $("#draft-design");
    if (draftButton) {
      draftButton.disabled = !canDesign;
      draftButton.title = canDesign ? "调用已配置的规划模型" : "保存 OpenAI 规划模型密钥后可生成候选";
    }
    const submitButton = form.querySelector('button[type="submit"]');
    if (submitButton) submitButton.textContent = hasReady && state.health?.executor_enabled ? "创建项目并提交计算" : "创建项目并完成本地校验";
  }
  const readiness = $("#design-readiness");
  if (readiness) readiness.innerHTML = canDesign
    ? `<strong>可以开始设计和本地校验</strong><span>服务器和三模型密钥在相应的执行阶段检查。</span>`
    : `<strong>可以导入候选并完成本地校验</strong><span>保存 OpenAI 规划模型密钥后可生成候选结构。</span>`;
}

async function refreshAll() {
  await Promise.all([loadClusters(), loadProjects(), loadRuns(), loadReviews(), loadModelKeys()]);
  await Promise.all(state.projects.map((p) => loadReleases(p.id, false)));
  renderOverview();
  renderOnboarding();
  renderModelKeys();
  renderReleaseProjectList();
}

async function loadModelKeys() {
  state.modelKeys = await api("/api/v1/model-keys");
}

function renderModelKeys() {
  const target = $("#model-key-status");
  if (!target) return;
  const providers = state.modelKeys?.providers || {};
  const rows = [
    ["openai", state.modelKeys?.models?.design || "OpenAI 规划模型"],
    ["deepseek", "DeepSeek"],
    ["dashscope", "Qwen"],
    ["zhipuai", "GLM"],
  ];
  const configuredCount = rows.filter(([key]) => Boolean(providers[key]?.configured)).length;
  const pills = rows.map(([key, label]) => {
    const configured = Boolean(providers[key]?.configured);
    return `<span class="tag ${configured ? "ok" : "warn"}">${escapeHtml(label)} ${configured ? "已保存" : "未填"}</span>`;
  }).join("");
  const closed = state.modelKeys?.ready_for_closed_loop;
  const canDesign = state.modelKeys?.ready_for_design;
  target.innerHTML = `<div class="notice compact-notice"><strong>${closed ? "设计和审计密钥已配置" : `已填写 ${configuredCount}/4`}</strong><span>${canDesign ? "可以生成候选。三模型密钥用于 WARN 审计。" : "可先导入候选并校验，生成候选时需要 OpenAI 规划密钥。"}</span><div class="key-pills">${pills}</div></div>`;
}

async function loadClusters() {
  state.clusters = await api("/api/v1/clusters");
  renderClusters();
}

const clusterSetupStages = [
  ["connection", "连接", 10],
  ["discovery", "检查", 35],
  ["installation", "准备", 75],
  ["preflight", "完成", 100],
];

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForScienceInstall(id) {
  for (let attempt = 0; attempt < 1440; attempt += 1) {
    const result = await api(`/api/v1/clusters/${id}/setup-status`, { method: "POST", body: "{}" });
    const science = result.science_install || result.provisioning?.science_install || {};
    const rawPercent = Number(science.percent || 0);
    updateClusterSetupProgress(id, {
      percent: Math.min(88, 45 + Math.round(rawPercent * 0.43)),
      stage: "installation",
      message: science.message || "正在准备服务器",
      tone: ["failed", "needs_admin"].includes(science.status) ? "blocked" : "active",
      running: !["complete", "failed", "needs_admin"].includes(science.status),
    });
    if (["complete", "failed", "needs_admin"].includes(science.status)) return result;
    await sleep(5000);
  }
  throw new Error("服务器仍在准备，请稍后刷新");
}

const runWorkflowStages = [
  ["submission", "提交", 8],
  ["queue", "排队", 18],
  ["remote_loop", "计算", 78],
  ["collection", "整理", 92],
  ["release", "完成", 100],
];

function inferredClusterSetupProgress(cluster) {
  const provisioning = cluster.provisioning || {};
  if (cluster.status === "ready") {
    return { percent: 100, stage: "preflight", message: "服务器已准备就绪", tone: "done", running: false };
  }
  if (provisioning.status === "installed") {
    return { percent: 88, stage: "preflight", message: "服务器准备尚未完成", tone: "blocked", running: false };
  }
  if (provisioning.status === "installing" || cluster.status === "installing") {
    const science = provisioning.science_install || {};
    return { percent: Math.min(88, 45 + Math.round(Number(science.percent || 0) * 0.43)), stage: "installation", message: science.message || "正在准备服务器", tone: "active", running: true };
  }
  if (["discovered", "needs_admin"].includes(provisioning.status)) {
    return { percent: 35, stage: "discovery", message: "环境检查已完成，等待管理员补齐依赖", tone: "blocked", running: false };
  }
  return { percent: 0, stage: "connection", message: "等待自动准备", tone: "idle", running: false };
}

function clusterSetupProgress(cluster) {
  return state.clusterSetupProgress.get(cluster.id) || inferredClusterSetupProgress(cluster);
}

function updateClusterSetupProgress(id, update) {
  const previous = state.clusterSetupProgress.get(id) || {};
  state.clusterSetupProgress.set(id, { ...previous, ...update });
  renderClusters();
}

function renderClusterSetupProgress(cluster) {
  const progress = clusterSetupProgress(cluster);
  const currentIndex = clusterSetupStages.findIndex(([key]) => key === progress.stage);
  const stages = clusterSetupStages.map(([key, label, threshold], index) => {
    const completed = progress.percent >= threshold;
    const current = index === currentIndex && !completed;
    const statusClass = completed ? "done" : current ? progress.tone : "pending";
    return `<li class="${statusClass}"><span></span>${escapeHtml(label)}</li>`;
  }).join("");
  return `<section class="cluster-setup-progress ${escapeHtml(progress.tone)}" aria-live="polite">
    <div class="progress-caption"><strong>服务器准备进度</strong><span>${escapeHtml(progress.message)} · ${progress.percent}%</span></div>
    <div class="progress-track" role="progressbar" aria-label="服务器自动准备进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress.percent}">
      <span style="width:${progress.percent}%"></span>
    </div>
    <ol>${stages}</ol>
  </section>`;
}

function runWorkflowProgress(run) {
  const stateName = run.state;
  const mapping = {
    planned: { percent: 4, stage: "submission", message: "正在提交", tone: "active" },
    awaiting_approval: { percent: 8, stage: "submission", message: "等待提交", tone: "active" },
    queued: { percent: 18, stage: "queue", message: "已进入控制端队列，等待提交或调度", tone: "active" },
    running: { percent: 48, stage: "remote_loop", message: "正在计算，耗时取决于服务器队列", tone: "active" },
    postprocessing: { percent: 68, stage: "remote_loop", message: "正在整理结果", tone: "active" },
    evaluating: { percent: 76, stage: "remote_loop", message: "正在检查结果", tone: "active" },
    repairing: { percent: 78, stage: "remote_loop", message: "正在处理未通过的结果", tone: "active" },
    collecting: { percent: 92, stage: "collection", message: "正在生成结果文件", tone: "active" },
    pass: { percent: 100, stage: "release", message: "计算完成，可下载结果", tone: "done" },
    review: { percent: 100, stage: "release", message: "计算完成，部分结果需要查看", tone: "review" },
  };
  if (mapping[stateName]) return mapping[stateName];
  if (stateName === "failed") {
    const lateFailure = String(run.error_code || "").includes("collection");
    return {
      percent: lateFailure ? 92 : run.remote_job_id ? 48 : 18,
      stage: lateFailure ? "collection" : run.remote_job_id ? "remote_loop" : "queue",
      message: `流程失败${run.error_code ? `：${run.error_code}` : ""}`,
      tone: "failed",
    };
  }
  if (stateName === "cancelled") {
    return {
      percent: run.remote_job_id ? 48 : 18,
      stage: run.remote_job_id ? "remote_loop" : "queue",
      message: "运行已取消",
      tone: "cancelled",
    };
  }
  return { percent: 0, stage: "submission", message: `当前状态：${stateName}`, tone: "idle" };
}

function renderRunWorkflowProgress(run) {
  const progress = runWorkflowProgress(run);
  const currentIndex = runWorkflowStages.findIndex(([key]) => key === progress.stage);
  const stages = runWorkflowStages.map(([key, label, threshold], index) => {
    const completed = progress.percent >= threshold;
    const current = index === currentIndex && !completed;
    const statusClass = completed ? "done" : current ? progress.tone : "pending";
    return `<li class="${statusClass}"><span></span>${escapeHtml(label)}</li>`;
  }).join("");
  return `<section class="run-workflow-progress ${escapeHtml(progress.tone)}" aria-live="polite">
    <div class="progress-caption"><strong>计算进度</strong><span>${escapeHtml(progress.message)} · ${progress.percent}%</span></div>
    <div class="progress-track" role="progressbar" aria-label="计算进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress.percent}">
      <span style="width:${progress.percent}%"></span>
    </div>
    <ol>${stages}</ol>
  </section>`;
}

function renderClusters() {
  const isAdmin = state.user?.role === "admin";
  $("#cluster-list").innerHTML = state.clusters.map((cluster) => {
    const progress = clusterSetupProgress(cluster);
    const authButton = isAdmin && cluster.status === "unverified"
      ? `<button onclick="authorizeCluster('${cluster.id}')">授权并准备</button>` : "";
    const setupAllowed = ["authorized", "ready", "failed", "needs_admin", "installing"].includes(cluster.status);
    const setupButton = setupAllowed
      ? `<button onclick="${cluster.status === "installing" ? "resumeScienceInstall" : "setupCluster"}('${cluster.id}')" ${progress.running && cluster.status !== "installing" ? "disabled" : ""}>${cluster.status === "installing" ? "查看进度" : progress.running ? "准备中..." : "准备服务器"}</button>` : "";
    const copyButton = cluster.status === "ready" ? "" : `<button class="secondary" onclick="copyText('${cluster.id}', 'public-key')">复制公钥</button>`;
    const provisioning = cluster.provisioning || {};
    const discovery = provisioning.discovery || {};
    const adminActions = provisioning.admin_actions?.length ? provisioning.admin_actions : (discovery.admin_actions || []);
    const checklist = adminActions.length
      ? `<details><summary>管理员操作清单（${adminActions.length}）</summary><ul>${adminActions.map((item) => `<li><strong>${escapeHtml(item.component)}</strong>：${escapeHtml(item.action)}</li>`).join("")}</ul></details>`
      : "";
    const preflightText = cluster.status === "ready" ? "服务器已准备好" : cluster.last_preflight ? "准备尚未完成" : "等待自动准备";
    const portText = Number(cluster.port) === 22 ? "" : `:${cluster.port}`;
    const keyBox = cluster.status === "ready" ? "" : `<details class="key-details"><summary>连接失败时查看 SSH 公钥</summary><div id="public-key-${cluster.id}" class="copy-box">${escapeHtml(cluster.public_key)}</div></details>`;
    return card(
      `${escapeHtml(cluster.name)} ${badge(cluster.status)}`,
      `${escapeHtml(cluster.remote_username)}@${escapeHtml(cluster.host)}${portText}<br>${escapeHtml(preflightText)}${renderClusterSetupProgress(cluster)}${checklist}${keyBox}`,
      `${authButton}${setupButton}${copyButton}`,
    );
  }).join("") || empty("还没有连接服务器。");
}

window.authorizeCluster = async (id) => {
  try {
    await api(`/api/v1/clusters/${id}/authorize`, { method: "POST", body: "{}" });
    flash("集群连接已授权，正在自动准备服务器...");
    await loadClusters();
    renderOnboarding();
    await setupCluster(id);
  } catch (err) {
    flash(err.message, true);
  }
};

window.preflightCluster = async (id) => {
  try {
    const result = await api(`/api/v1/clusters/${id}/preflight`, { method: "POST", body: "{}" });
    flash(result.status === "ready" ? "服务器可以使用。" : "服务器尚未准备好。", result.status !== "ready");
    await loadClusters();
    renderOnboarding();
  } catch (err) {
    flash(err.message, true);
  }
};

window.discoverCluster = async (id) => {
  try {
    const result = await api(`/api/v1/clusters/${id}/discover`, { method: "POST", body: "{}" });
    const actions = result.provisioning?.discovery?.admin_actions || [];
    flash(actions.length ? `环境发现完成：有 ${actions.length} 项需要管理员处理。` : "环境发现完成，可以安装平台组件。", actions.length > 0);
    await loadClusters();
    renderOnboarding();
  } catch (err) {
    flash(err.message, true);
  }
};

window.provisionCluster = async (id) => {
  try {
    let result = await api(`/api/v1/clusters/${id}/provision`, {
      method: "POST",
      body: JSON.stringify({ confirmation: "install", overwrite: false }),
    });
    if (result.status === "installing") {
      flash("固定版本 EMC/GROMACS 正在远程后台安装；页面将持续读取真实安装阶段。");
      result = await waitForScienceInstall(id);
    }
    flash(result.status === "ready" ? "服务器可以使用。" : "仍有环境问题需要处理。", result.status !== "ready");
    await loadClusters();
    renderOnboarding();
  } catch (err) {
    flash(err.message, true);
  }
};

window.resumeScienceInstall = async (id) => {
  try {
    const result = await waitForScienceInstall(id);
    await loadClusters();
    renderOnboarding();
    flash(result.status === "ready" ? "服务器可以使用。" : "仍有环境问题需要处理。", result.status !== "ready");
  } catch (err) {
    flash(err.message, true);
  }
};

window.setupCluster = async (id) => {
  try {
    updateClusterSetupProgress(id, {
      percent: 8,
      stage: "connection",
      message: "正在建立 SSH 连接",
      tone: "active",
      running: true,
    });
    flash("正在连接服务器并自动选择计算资源...");
    await api(`/api/v1/clusters/${id}/autoconfigure`, { method: "POST", body: "{}" });
    const discoveryResult = await api(`/api/v1/clusters/${id}/discover`, { method: "POST", body: "{}" });
    const discovery = discoveryResult.provisioning?.discovery || {};
    const discoveryActions = discovery.admin_actions || [];
    updateClusterSetupProgress(id, {
      percent: 35,
      stage: "discovery",
      message: discoveryActions.length ? `环境检查完成，发现 ${discoveryActions.length} 项待处理依赖` : "Slurm、软件与目录检查完成",
      tone: discoveryActions.length ? "blocked" : "active",
      running: true,
    });
    if (!discovery.platform_installable) {
      updateClusterSetupProgress(id, {
        running: false,
        tone: "blocked",
        message: "当前账号无法自动安装，请按管理员清单补齐环境",
      });
      await loadClusters();
      renderOnboarding();
      flash(`环境检查已停止：有 ${discoveryActions.length || 1} 项需要管理员处理。`, true);
      return;
    }
    updateClusterSetupProgress(id, {
      percent: 45,
      stage: "installation",
      message: "正在准备服务器",
      tone: "active",
      running: true,
    });
    let result = await api(`/api/v1/clusters/${id}/provision`, {
      method: "POST",
      body: JSON.stringify({ confirmation: "install", overwrite: false }),
    });
    if (result.status === "installing") {
      result = await waitForScienceInstall(id);
    }
    updateClusterSetupProgress(id, {
      percent: 82,
      stage: "preflight",
      message: "正在完成最后检查",
      tone: "active",
      running: true,
    });
    const actions = result.provisioning?.admin_actions || result.provisioning?.discovery?.admin_actions || [];
    if (result.status === "ready") {
      updateClusterSetupProgress(id, {
        percent: 100,
        stage: "preflight",
        message: "服务器已准备就绪",
        tone: "done",
        running: false,
      });
      flash("服务器可以使用。");
    } else if (actions.length) {
      updateClusterSetupProgress(id, {
        percent: 88,
        stage: "preflight",
        message: `发现 ${actions.length} 项需要处理`,
        tone: "blocked",
        running: false,
      });
      flash(`平台已完成可自动安装部分；仍有 ${actions.length} 项需要管理员处理。`, true);
    } else {
      updateClusterSetupProgress(id, {
        percent: 88,
        stage: "preflight",
        message: "服务器准备尚未完成",
        tone: "blocked",
        running: false,
      });
      flash("仍有环境问题需要处理。", true);
    }
    await loadClusters();
    renderOnboarding();
  } catch (err) {
    updateClusterSetupProgress(id, {
      message: `准备失败：${err.message}`,
      tone: "failed",
      running: false,
    });
    flash(err.message, true);
  }
};

async function loadProjects() {
  state.projects = await api("/api/v1/projects");
  renderProjects();
}

function renderProjects() {
  $("#project-list").innerHTML = state.projects.map((project) => {
    const validation = project.validation || {};
    const actions = [
      project.status === "draft" ? `<button onclick="validateProject('${project.id}')">继续</button>` : "",
      project.status === "validated" ? `<button onclick="lockProject('${project.id}')">继续</button>` : "",
      project.status === "locked" ? `<button onclick="createRun('${project.id}')">提交计算</button>` : "",
      `<button class="secondary" onclick="selectReleaseProject('${project.id}')">结果</button>`,
    ].join("");
    const problems = (validation.errors || []).length ? `<br>错误：${escapeHtml(JSON.stringify(validation.errors))}` : "";
    const warnings = (validation.warnings || []).length ? `<br>警示：${escapeHtml(JSON.stringify(validation.warnings))}` : "";
    return card(
      `${escapeHtml(project.name)} ${badge(project.status)}`,
      `${project.manifest?.length || project.spec?.candidates?.length || 0} 个候选${problems}${warnings}`,
      actions,
    );
  }).join("") || empty("还没有项目。");
}

window.validateProject = async (id) => {
  try {
    const result = await api(`/api/v1/projects/${id}/validate`, { method: "POST", body: "{}" });
    flash(result.valid ? "设计检查通过。" : "设计检查未通过，请修改候选结构。", !result.valid);
    await Promise.all([loadProjects(), loadReviews()]);
    renderOverview();
  } catch (err) {
    flash(err.message, true);
  }
};

window.lockProject = async (id) => {
  try {
    await api(`/api/v1/projects/${id}/lock`, { method: "POST", body: "{}" });
    flash("项目已准备好。");
    await loadProjects();
  } catch (err) {
    flash(err.message, true);
  }
};

async function createRunPlan(projectId, { activate = true, quiet = false } = {}) {
  const ready = state.clusters.filter((x) => x.status === "ready");
  if (!ready.length) throw new Error("提交计算需要已通过预检的服务器连接。");
  const cluster = ready[0];
  const run = await api(`/api/v1/projects/${projectId}/runs`, {
    method: "POST",
    body: JSON.stringify({ cluster_id: cluster.id }),
  });
  if (!quiet) flash(`已选择服务器 ${cluster.name}，正在准备提交。`);
  await loadRuns();
  if (activate) activatePanel("runs-panel");
  return run;
}

function runApprovalPayload(run = null) {
  const cluster = run?.cluster_id ? state.clusters.find((item) => item.id === run.cluster_id) : null;
  const limits = cluster?.resource_limits || {};
  const maxCpus = Number(limits.max_cpus || 8);
  const maxGpus = Number(limits.max_gpus ?? 1);
  const maxWalltime = Number(limits.max_walltime_hours || 168);
  return {
    max_repair_attempts: 1,
    max_additional_md_ns: 400,
    max_cases: 100,
    cpus_per_task: Math.max(1, Math.min(8, maxCpus)),
    gpus: maxGpus > 0 ? 1 : 0,
    walltime_hours: Math.max(1, Math.min(168, maxWalltime)),
    confirmation: "approve",
  };
}

async function approveRunPlan(runOrId, { quiet = false } = {}) {
  const run = typeof runOrId === "string" ? state.runs.find((item) => item.id === runOrId) : runOrId;
  const runId = typeof runOrId === "string" ? runOrId : runOrId.id;
  await api(`/api/v1/runs/${runId}/approve`, {
    method: "POST",
    body: JSON.stringify(runApprovalPayload(run)),
  });
  if (!quiet) flash("任务已进入队列。");
  await loadRuns();
}

async function submitProjectToCompute(project) {
  flash("项目已创建，正在校验候选结构...");
  const validation = await api(`/api/v1/projects/${project.id}/validate`, { method: "POST", body: "{}" });
  await loadReviews();
  if (!validation.valid) {
    await loadProjects();
    renderOverview();
    flash("设计验证失败，已停在草案状态；请检查候选设计。", true);
    return;
  }
  await api(`/api/v1/projects/${project.id}/lock`, { method: "POST", body: "{}" });
  await loadProjects();
  renderOverview();
  if (!state.clusters.some((cluster) => cluster.status === "ready") || !state.health?.executor_enabled) {
    flash("本地校验通过，候选清单已锁定。服务器和执行器准备完成后可提交计算。");
    return;
  }
  try {
    const run = await createRunPlan(project.id, { activate: false, quiet: true });
    await approveRunPlan(run, { quiet: true });
    activatePanel("runs-panel");
    flash("计算已提交。");
  } catch (err) {
    await loadRuns();
    activatePanel("runs-panel");
    if (String(err.message || "").includes("executor_disabled")) {
      flash("项目已准备好，但当前只能进行本机测试。");
    } else {
      flash(`提交计算未完成：${err.message}`, true);
    }
  }
}

function parseCandidates(text) {
  const rows = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!rows.length) {
    throw new Error("请先生成或导入候选设计；常规 PE-LCB100 草案应包含 100 行。");
  }
  return rows.map((line, index) => {
    const cols = line.split(",").map((x) => x.trim());
    if (cols.length < 5) throw new Error(`第 ${index + 1} 行至少需要 5 列`);
    const placement = cols[3];
    const coarse = placementToCoarse[placement];
    if (!coarse) throw new Error(`第 ${index + 1} 行的精确放置码未注册：${placement}`);
    return {
      case_id: cols[0],
      branch_points: Number(cols[1]),
      branch_carbons: Number(cols[2]),
      branch_distribution_code: coarse,
      placement_code: placement,
      target_total_carbons: Number(cols[4]),
      design_rationale: cols.slice(5).join(",").slice(0, 500),
    };
  });
}

async function loadRuns() {
  state.runs = await api("/api/v1/runs");
  renderRuns();
}

let localAcceptanceTimer = null;

function formatElapsed(value) {
  const start = value ? Date.parse(value) : NaN;
  if (!Number.isFinite(start)) return "";
  const seconds = Math.max(0, Math.round((Date.now() - start) / 1000));
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} 分 ${seconds % 60} 秒`;
}

function localAcceptanceProgress(job) {
  const mapping = {
    queued: { percent: 8, label: "排队" },
    real_md: { percent: 45, label: "短程 MD" },
    release: { percent: 100, label: "发布校验" },
    failed: { percent: 100, label: "失败" },
  };
  if (job.status === "done") return { percent: 100, label: "完成" };
  if (job.status === "failed") return { percent: 100, label: "失败" };
  return mapping[job.stage] || { percent: job.status === "running" ? 35 : 8, label: job.stage || job.status };
}

function renderLocalAcceptance(job) {
  const target = $("#local-acceptance-status");
  if (!target) return;
  const terminal = job.status === "done" || job.status === "failed";
  const progress = localAcceptanceProgress(job);
  const elapsed = formatElapsed(job.started_at || job.created_at);
  const elapsedText = elapsed ? ` · 已运行 ${elapsed}` : "";
  const links = job.status === "done"
    ? `<div class="button-row"><a class="button-link" href="/api/v1/local-acceptance/${job.job_id}/result">下载测试结果</a></div>`
    : "";
  const result = job.result
    ? `<p>结果：<strong>${escapeHtml(job.result.publication_state)}</strong> · 密度 ${escapeHtml(job.result.density_g_cm3)} g/cm³</p>`
    : "";
  target.innerHTML = `<article class="acceptance-status ${escapeHtml(job.status)}">
    <div>
      <strong>${escapeHtml(job.message)}</strong>
      <span>${escapeHtml(progress.label)} · ${escapeHtml(job.status)}${elapsedText}</span>
      <div class="mini-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress.percent}"><span style="width:${progress.percent}%"></span></div>
      ${result}
    </div>
    ${links}
  </article>`;
  const button = $("#start-local-acceptance");
  if (button) {
    button.hidden = !terminal;
    button.disabled = false;
    button.textContent = terminal ? "重新验收" : "开始测试";
  }
}

async function pollLocalAcceptance(jobId) {
  try {
    const job = await api(`/api/v1/local-acceptance/${jobId}`);
    renderLocalAcceptance(job);
    if (["done", "failed"].includes(job.status)) {
      clearInterval(localAcceptanceTimer);
      localAcceptanceTimer = null;
    }
  } catch (err) {
    clearInterval(localAcceptanceTimer);
    localAcceptanceTimer = null;
    flash(err.message, true);
  }
}

async function startLocalAcceptance() {
  try {
    const button = $("#start-local-acceptance");
    if (button) {
      button.disabled = true;
      button.hidden = false;
      button.textContent = "启动中...";
    }
    flash("测试已开始。");
    const job = await api("/api/v1/local-acceptance", { method: "POST", body: "{}" });
    renderLocalAcceptance(job);
    if (localAcceptanceTimer) clearInterval(localAcceptanceTimer);
    localAcceptanceTimer = setInterval(() => pollLocalAcceptance(job.job_id), 1000);
    await pollLocalAcceptance(job.job_id);
  } catch (err) {
    const button = $("#start-local-acceptance");
    if (button) {
      button.disabled = false;
      button.hidden = false;
      button.textContent = "开始测试";
    }
    flash(err.message, true);
  }
}

function renderRuns() {
  $("#run-list").innerHTML = state.runs.map((run) => {
    const project = state.projects.find((p) => p.id === run.project_id);
    const actions = [
      `<button class="secondary" onclick="showRunEvents('${run.id}')">查看详情</button>`,
      run.state === "awaiting_approval" ? `<button onclick="approveRun('${run.id}')">提交计算</button>` : "",
      !["pass", "review", "failed", "cancelled"].includes(run.state) ? `<button class="danger" onclick="cancelRun('${run.id}')">取消</button>` : "",
    ].join("");
    return card(
      `${escapeHtml(project?.name || "unknown project")} ${badge(run.state)}`,
      `${renderRunWorkflowProgress(run)}`,
      actions,
    );
  }).join("") || empty("还没有提交计算。");
}

async function pollActiveRuns() {
  if (!state.user) return;
  const activeStates = new Set(["planned", "awaiting_approval", "queued", "running", "collecting", "postprocessing", "evaluating", "repairing"]);
  if (!state.runs.some((run) => activeStates.has(run.state))) return;
  try {
    await loadRuns();
    await Promise.all(state.projects.map((project) => loadReleases(project.id, false)));
    renderOverview();
    renderReleaseProjectList();
  } catch {
    // A transient refresh failure must not interrupt a remote calculation.
  }
}

window.createRun = async (projectId) => {
  try {
    await createRunPlan(projectId);
  } catch (err) {
    flash(err.message, true);
  }
};

window.approveRun = async (id) => {
  try {
    await approveRunPlan(id);
  } catch (err) {
    flash(err.message, true);
  }
};

window.cancelRun = async (id) => {
  if (!confirm("确认取消该运行？如果已有远程 Slurm 作业，平台会尝试 scancel。")) return;
  try {
    await api(`/api/v1/runs/${id}/cancel`, { method: "POST", body: "{}" });
    flash("运行已取消。");
    await loadRuns();
  } catch (err) {
    flash(err.message, true);
  }
};

window.showRunEvents = async (id) => {
  try {
    const text = await api(`/api/v1/runs/${id}/events`);
    const events = text.split("\n\n").filter(Boolean).map((line) => line.replace(/^data:\s*/, ""));
    $("#event-viewer").hidden = false;
    $("#event-viewer").innerHTML = events.length
      ? events.map((e) => `<pre>${escapeHtml(JSON.stringify(JSON.parse(e), null, 2))}</pre>`).join("")
      : "该运行还没有事件。";
    $("#event-viewer").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (err) {
    flash(err.message, true);
  }
};

async function loadReviews() {
  state.reviews = await api("/api/v1/reviews");
  renderReviews();
}

function renderReviews() {
  $("#review-list").innerHTML = state.reviews.map((review) => {
    const packetId = `packet-${review.id}`;
    const actions = review.status !== "resolved"
      ? `<button onclick="resolveReview('${review.id}')">标记复核完成</button>` : "";
    return card(
      `需要查看 ${badge(review.status)}`,
      `${escapeHtml(review.reason_code)}<details><summary>查看详情</summary><pre id="${packetId}" class="packet">${escapeHtml(JSON.stringify(review.packet, null, 2))}</pre></details>`,
      actions,
    );
  }).join("") || empty("当前没有复核项。");
}

window.resolveReview = async (id) => {
  const note = prompt("请输入复核说明（会写入版本化记录）：", "已人工复核，保留为 review。");
  if (!note) return;
  try {
    await api(`/api/v1/reviews/${id}/resolve`, {
      method: "POST",
      body: JSON.stringify({ decision: "accept_review", note }),
    });
    flash("复核项已标记完成。");
    await loadReviews();
  } catch (err) {
    flash(err.message, true);
  }
};

async function loadReleases(projectId, render = true) {
  const releases = await api(`/api/v1/projects/${projectId}/releases`);
  state.releasesByProject.set(projectId, releases);
  if (render) {
    renderReleaseList(projectId);
    renderOverview();
  }
  return releases;
}

function renderReleaseProjectList() {
  $("#release-project-list").innerHTML = state.projects.map((project) => {
    const releases = state.releasesByProject.get(project.id) || [];
    return card(
      `${escapeHtml(project.name)}`,
      `${releases.length} 个结果文件`,
      `<button onclick="createRelease('${project.id}')">生成结果</button><button class="secondary" onclick="selectReleaseProject('${project.id}')">查看</button>`,
    );
  }).join("") || empty("还没有可下载的结果。");
}

window.selectReleaseProject = async (projectId) => {
  activatePanel("reviews-panel");
  await loadReleases(projectId);
};

function renderReleaseList(projectId) {
  const project = state.projects.find((p) => p.id === projectId);
  const releases = state.releasesByProject.get(projectId) || [];
  $("#release-list").innerHTML = `<h3>${escapeHtml(project?.name || "项目")} 的结果</h3>` + (
    releases.map((release) => card(
      `结果 ${release.version}`,
      `${escapeHtml(release.created_at)}`,
      `<a class="button-link" href="/api/v1/releases/${release.id}/download-result">下载结果</a>`,
    )).join("") || empty("该项目还没有结果文件。")
  );
}

window.createRelease = async (projectId) => {
  const projectRuns = state.runs.filter((r) => r.project_id === projectId);
  const latestRun = projectRuns[0];
  const qs = latestRun ? `?run_id=${encodeURIComponent(latestRun.id)}` : "";
  try {
    await api(`/api/v1/projects/${projectId}/releases${qs}`, { method: "POST", body: "{}" });
    flash("结果文件已生成。");
    await loadReleases(projectId);
    renderReleaseProjectList();
  } catch (err) {
    flash(err.message, true);
  }
};

window.copyText = async (id, prefix) => {
  const text = $(`#${prefix}-${id}`)?.textContent || "";
  try {
    await navigator.clipboard.writeText(text);
    flash("已复制到剪贴板。");
  } catch {
    flash("浏览器阻止了复制，请手动选中复制。", true);
  }
};

function activatePanel(id) {
  const button = $(`[data-panel="${id}"]`);
  if (button?.disabled) {
    flash("请查看当前操作所需的配置。", true);
    return;
  }
  $$(".panel").forEach((panel) => { panel.hidden = panel.id !== id; });
  $$(".tabs button").forEach((button) => button.classList.toggle("active", button.dataset.panel === id));
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const payload = Object.fromEntries(new FormData(event.target));
    const result = await api("/api/v1/auth/login", { method: "POST", body: JSON.stringify(payload) });
    state.csrf = result.csrf_token;
    sessionStorage.setItem("csrf", state.csrf);
    state.user = result.user;
    state.health = await api("/api/v1/health");
    showWorkspace();
  } catch (err) {
    $("#login-error").textContent = err.message;
  }
});

$("#logout").addEventListener("click", async () => {
  try {
    await api("/api/v1/auth/logout", { method: "POST", body: "{}" });
  } finally {
    sessionStorage.clear();
    location.reload();
  }
});

$("#cluster-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    applyQuickSshToForm();
  } catch (err) {
    flash(err.message, true);
    return;
  }
  const form = Object.fromEntries(new FormData(event.target));
  const oneTimePassword = String(form.ssh_password_once || "").trim();
  delete form.quick_ssh;
  delete form.ssh_password_once;
  if (!String(form.host_key_sha256 || "").trim()) {
    try {
      const scanned = await scanHostKeyIntoForm({ quiet: true });
      form.host_key_sha256 = scanned.host_key_sha256;
    } catch (err) {
      flash(`主机指纹未填写，且自动读取失败：${err.message}`, true);
      return;
    }
  }
  const payload = {
    ...form,
    port: Number(form.port),
    account: form.account || null,
    max_cpus: Number(form.max_cpus),
    max_gpus: Number(form.max_gpus),
    max_walltime_hours: Number(form.max_walltime_hours),
    module_init: String(form.module_init || "").split(/\r?\n/).map((x) => x.trim()).filter(Boolean),
    allowed_nodes: [],
    environment_mode: form.environment_mode,
    container_runtime: form.environment_mode === "container" ? form.container_runtime : null,
    container_image: form.environment_mode === "container" ? (form.container_image || null) : null,
    pipeline_command: form.pipeline_command || "pelcb-pipeline-core",
    science_install_mode: form.science_install_mode || "existing",
    gromacs_acceleration: form.gromacs_acceleration || "auto",
    upstream_terms_accepted: form.upstream_terms_accepted === "on",
  };
  try {
    const cluster = await api("/api/v1/clusters", { method: "POST", body: JSON.stringify(payload) });
    flash("连接已创建，正在完成免密连接...");
    await api(`/api/v1/clusters/${cluster.id}/install-key`, {
      method: "POST",
      body: JSON.stringify({ password: oneTimePassword || null }),
    });
    event.target.reset();
    await loadClusters();
    renderOnboarding();
    const refreshedCluster = state.clusters.find((item) => item.id === cluster.id) || cluster;
    if (["authorized", "ready", "failed", "needs_admin"].includes(refreshedCluster.status)) {
      await setupCluster(refreshedCluster.id);
    } else {
      flash("服务器已连接，等待管理员授权后将自动准备。");
    }
  } catch (err) {
    flash(err.message, true);
  }
});

async function scanHostKeyIntoForm({ quiet = false } = {}) {
  try {
    applyQuickSshToForm();
  } catch (err) {
    if (!quiet) throw err;
  }
  const form = $("#cluster-form");
  const host = form.elements.host.value.trim();
  const port = Number(form.elements.port.value || 22);
  if (!host) {
    throw new Error("请先填写服务器主机名或 IP。");
  }
  const result = await api("/api/v1/clusters/host-key", {
    method: "POST",
    body: JSON.stringify({ host, port }),
  });
  form.elements.host_key_sha256.value = result.host_key_sha256;
  return result;
}

$("#scan-host-key").addEventListener("click", async () => {
  const button = $("#scan-host-key");
  try {
    button.disabled = true;
    button.textContent = "读取中...";
    await scanHostKeyIntoForm();
    flash("已读取主机指纹，请与管理员提供的指纹核对。");
  } catch (err) {
    flash(err.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "自动读取主机指纹";
  }
});

$("#project-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  try {
    const payload = {
      name: String(form.get("name")),
      spec: {
        natural_language_goal: String(form.get("goal") || ""),
        polymer_family: "PE",
        architecture: "pe_lcb",
        force_field: "opls/2024/opls-aa",
        temperature_k: Number(form.get("temperature_k") || 400),
        pressure_atm: Number(form.get("pressure_atm") || 1),
        chain_count: Number(form.get("chain_count") || 150),
        initial_density_g_cm3: 0.85,
        production_ns: Number(form.get("production_ns") || 400),
        rule_profile: "pelcb100_validated_v1",
        planning: state.designPlanning,
        candidates: state.designDraft.length ? state.designDraft : parseCandidates(form.get("candidates")),
      },
    };
    const project = await api("/api/v1/projects", { method: "POST", body: JSON.stringify(payload) });
    state.designDraft = [];
    state.designPlanning = null;
    renderStructurePreview([]);
    await loadProjects();
    renderOverview();
    await submitProjectToCompute(project);
  } catch (err) {
    flash(err.message, true);
  }
});

$("#draft-design").addEventListener("click", async () => {
  const form = $("#project-form");
  const goal = form.elements.goal.value || "PE-LCB C100";
  try {
    const draft = await api("/api/v1/design-drafts", { method: "POST", body: JSON.stringify({ goal }) });
    state.designDraft = draft.spec.candidates;
    state.designPlanning = draft.spec.planning;
    const lines = draft.spec.candidates.map((c) => `${c.case_id},${c.branch_points},${c.branch_carbons},${c.placement_code},${c.target_total_carbons},${c.design_rationale || ""}`);
    form.elements.candidates.value = lines.join("\n");
    const engine = draft.planner?.engine || "unknown_planner";
    const source = draft.planner?.candidate_source || "registered_grammar";
    renderStructurePreview(draft.spec.candidates, draft.planner?.strategy_summary || "");
    flash(`规划模型已提出 ${draft.spec.candidates.length} 个候选。规划器：${engine}；候选来源：${source}。请核对结构后再创建项目。`);
  } catch (err) {
    flash(err.message, true);
  }
});

$("#preview-polymer-setup").addEventListener("click", async () => {
  const form = $("#project-form");
  const repeatUnit = String(form.elements.repeat_unit_smiles.value || "").trim();
  if (!repeatUnit) {
    flash("请先填写重复单元结构，例如 *CCO*。", true);
    return;
  }
  const button = $("#preview-polymer-setup");
  try {
    button.disabled = true;
    button.textContent = "正在自动配置...";
    const result = await api("/api/v1/polymer-setup/preview", {
      method: "POST",
      body: JSON.stringify({ repeat_unit_smiles: repeatUnit }),
    });
    renderPolymerSetup(result);
    flash("已自动选择构建方法和力场。");
  } catch (err) {
    flash(err.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "自动选择构建方法和力场";
  }
});

$("#candidate-csv").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const data = lines[0]?.toLowerCase().includes("case_id") ? lines.slice(1) : lines;
  state.designDraft = [];
  state.designPlanning = null;
  renderStructurePreview([]);
  $("#project-form").elements.candidates.value = data.map((line) => line.split(",").slice(0, 6).map((x) => x.trim()).join(",")).join("\n");
  flash(`已载入 ${data.length} 条候选，请核对后创建项目。`);
});

$("#project-form").elements.candidates.addEventListener("input", () => {
  state.designDraft = [];
  state.designPlanning = null;
  renderStructurePreview([]);
});

$("#project-form").elements.goal.addEventListener("input", () => {
  state.designDraft = [];
  state.designPlanning = null;
  renderStructurePreview([]);
});

$("#environment-mode").addEventListener("change", (event) => {
  const containerMode = event.target.value === "container";
  $$(".container-option").forEach((element) => { element.hidden = !containerMode; });
  const image = $("#cluster-form").elements.container_image;
  if (image) image.required = containerMode;
  const acceptance = $("#cluster-form").elements.upstream_terms_accepted;
  const installMode = $("#science-install-mode");
  if (containerMode) installMode.value = "existing";
  installMode.disabled = containerMode;
  if (acceptance) acceptance.required = !containerMode && installMode.value === "managed";
  $$(".managed-science-option").forEach((element) => { element.hidden = containerMode || installMode.value !== "managed"; });
});

$("#science-install-mode").addEventListener("change", (event) => {
  const managed = event.target.value === "managed";
  $$(".managed-science-option").forEach((element) => { element.hidden = !managed; });
  const acceptance = $("#cluster-form").elements.upstream_terms_accepted;
  if (acceptance) acceptance.required = managed;
});

$("#quick-ssh").addEventListener("blur", () => {
  const value = $("#quick-ssh").value.trim();
  if (!value) return;
  try {
    applyQuickSshToForm();
  } catch (err) {
    flash(err.message, true);
  }
});

$("#refresh-clusters").addEventListener("click", loadClusters);
$("#refresh-model-keys").addEventListener("click", async () => {
  await loadModelKeys();
  renderModelKeys();
  renderOverview();
  renderOnboarding();
});
$("#refresh-projects").addEventListener("click", loadProjects);
$("#refresh-runs").addEventListener("click", loadRuns);
$("#start-local-acceptance").addEventListener("click", startLocalAcceptance);
$("#refresh-reviews").addEventListener("click", async () => {
  await loadReviews();
  renderReleaseProjectList();
});

$$(".tabs button").forEach((button) => {
  button.addEventListener("click", () => activatePanel(button.dataset.panel));
});

$("#model-key-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = {};
  ["openai_api_key", "deepseek_api_key", "dashscope_api_key", "zhipuai_api_key"].forEach((name) => {
    const value = form.elements[name].value.trim();
    if (value) payload[name] = value;
  });
  if (!Object.keys(payload).length) {
    flash("请输入至少一个模型 API key。", true);
    return;
  }
  try {
    state.modelKeys = await api("/api/v1/model-keys", { method: "POST", body: JSON.stringify(payload) });
    form.reset();
    renderModelKeys();
    renderOverview();
    renderOnboarding();
    flash("模型密钥已加密保存。");
  } catch (err) {
    flash(err.message, true);
  }
});

$("#clear-model-keys").addEventListener("click", async () => {
  if (!confirm("确认清空当前账号保存的模型密钥？清空后自然语言设计会重新锁定。")) return;
  try {
    state.modelKeys = await api("/api/v1/model-keys", { method: "DELETE", body: "{}" });
    renderModelKeys();
    renderOverview();
    renderOnboarding();
    flash("模型密钥已清空。");
  } catch (err) {
    flash(err.message, true);
  }
});

boot();
setInterval(pollActiveRuns, 15000);
