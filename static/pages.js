/* Local-only usage guide: switch instructions, copy text, and compose a task. */
(function () {
  "use strict";
  const METRICS = ["密度", "Rg", "Ree", "链形状", "RDF", "MSD"];
  const STOPS = {
    validation1: "构建与校验1完成后停止",
    md: "模拟完成后停止",
    validation2: "轨迹分析与校验2完成后停止",
    archive: "完成结果检查、异常处理、再次校验与归档后停止"
  };
  function normalizeOs(value) { return value === "linux" ? "linux" : "windows"; }
  function textOr(value, fallback) { return String(value ?? "").trim() || "[" + fallback + "]"; }
  function buildPrompt(fields) {
    const stop = STOPS[fields.stop_stage] || STOPS.validation1;
    return [
      "请使用本框架开展以下聚合物分子动力学计算。",
      "材料：" + textOr(fields.material, "材料名称") + "；结构：" + textOr(fields.structure, "重复单元SMILES或结构文件路径") + "。",
      "模拟条件：温度 " + textOr(fields.temperature, "温度") + " K，压力 " + textOr(fields.pressure, "压力") + " atm，" +
        textOr(fields.chains, "链数") + " 条链，每链 " + textOr(fields.repeat_units, "重复单元数") + " 个重复单元。",
      "计算资源与输出目录：" + textOr(fields.resources, "服务器、CPU/GPU资源和输出目录") + "。",
      "请先检查分子结构、力场参数、分析所需的原子选择、链编号和端点定义，以及计算环境，再完成分子构建和校验。",
      "分析目标：" + METRICS.join("、") + "。按本轮停止阶段安排模拟与分析，并保存各项结果的单位和方法。",
      "发现问题时，请分析原因、处理问题，并使用相同规则再次检查。",
      "本轮停止阶段：" + stop + "。停止时报告已完成步骤和继续条件。",
      "请按案例保存输入参数、工具调用、校验结果、处理记录与输出文件。"
    ].join("\n");
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { METRICS, STOPS, normalizeOs, buildPrompt };
  }
  if (typeof document === "undefined") return;
  const $ = (selector) => document.querySelector(selector);
  let guide;
  function selectOs(value, focus) {
    const os = normalizeOs(value);
    document.querySelectorAll("[data-os]").forEach((button) => {
      const selected = button.dataset.os === os;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      if (selected && focus) button.focus();
    });
    $("#installCode").textContent = guide.install_commands[os];
    $("#installCode").setAttribute("aria-labelledby", "tab-" + os);
    $("#pythonPath").textContent = os === "windows" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";
    $("#codexTemplateLink").href = guide.links[os + "_template"];
    $("#codexTemplateLink").textContent = (os === "windows" ? "Windows" : "Linux") + "配置模板 ↗";
  }
  async function copyText(targetId) {
    const target = document.getElementById(targetId);
    const text = target.value !== undefined ? target.value : target.textContent;
    $("#copyFeedback").textContent = "正在复制…";
    try {
      await navigator.clipboard.writeText(text);
      $("#copyFeedback").textContent = "已复制，可粘贴到本机终端或Codex。";
    } catch (_) {
      if (target.select) target.select();
      else {
        const range = document.createRange(); range.selectNodeContents(target);
        const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      }
      const copied = document.execCommand("copy");
      $("#copyFeedback").textContent = copied ? "已复制，可粘贴到本机终端或Codex。" : "内容已选中，请按Ctrl+C或Cmd+C复制。";
    }
  }
  function renderWorkflow() {
    const root = $("#workflow"); root.replaceChildren();
    guide.workflow.forEach((step, index) => {
      const article = document.createElement("article"); article.className = "workflow-step";
      const number = document.createElement("span"); number.textContent = String(index + 1).padStart(2, "0");
      const title = document.createElement("h3"); title.textContent = step.title;
      const description = document.createElement("p"); description.textContent = step.description;
      article.append(number, title, description); root.append(article);
    });
  }
  function updatePrompt() {
    $("#promptText").value = buildPrompt(Object.fromEntries(new FormData($("#promptForm"))));
    $("#copyFeedback").textContent = "任务描述已整理，可继续编辑并复制到本机Codex。";
  }
  $("#promptText").value = buildPrompt({ stop_stage: "validation1" });
  $("#buildPrompt").addEventListener("click", updatePrompt);
  $("#promptForm").addEventListener("submit", (event) => { event.preventDefault(); updatePrompt(); });
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", () => copyText(button.dataset.copy));
  });
  async function bootstrap() {
    try {
      const response = await fetch("data/guide.json");
      if (!response.ok) throw new Error("HTTP " + response.status);
      guide = await response.json();
      $("#repositoryLink").href = guide.repository;
      document.querySelectorAll("[data-doc]").forEach((link) => { link.href = guide.links[link.dataset.doc]; });
      selectOs("windows", false); renderWorkflow();
      document.querySelectorAll("[data-os]").forEach((button) => {
        button.addEventListener("click", () => selectOs(button.dataset.os, false));
        button.addEventListener("keydown", (event) => {
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            event.preventDefault(); selectOs(button.dataset.os === "linux" ? "windows" : "linux", true);
          }
        });
      });
    } catch (error) {
      $("#copyFeedback").textContent = "页面说明载入失败（" + error.message + "），可打开GitHub上的完整使用说明。";
    }
  }
  bootstrap();
})();
