from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from polymer_agent_demo.app import DEFAULT_SNAPSHOT, create_app
from polymer_agent_demo.control_plane.config import model_configuration, model_stage_readiness
from polymer_agent_demo.control_plane.production_audit import run_audit_panel
from polymer_agent_demo.scripts import doctor


ROOT = Path(__file__).resolve().parents[1]


def test_demo_blocks_live_flags_project_detection_and_execution(monkeypatch, tmp_path):
    from polymer_agent_demo import app as application

    for field in (
        "POLYMER_AGENT_LIVE_INTENT", "POLYMER_AGENT_LIVE_ATTRIBUTION",
        "POLYMER_AGENT_REAL_SMOKE", "POLYMER_AGENT_EXECUTOR_ENABLED",
    ):
        monkeypatch.setenv(field, "1")
    monkeypatch.setenv("POLYMER_AGENT_DEMO_MODE", "connected")
    monkeypatch.setenv("POLYMER_AGENT_ROOT", str(tmp_path / "unrelated_workspace"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-never-sent")

    def fail(*args, **kwargs):
        raise AssertionError("demo must stay within bundled offline inputs")

    monkeypatch.setattr(application, "_load_dotenv", fail)
    monkeypatch.setattr(application, "_detect_project_root", fail)
    client = TestClient(create_app(mode="demo"))
    health = client.get("/api/health").json()
    assert health["application_mode"] == "demo"
    assert health["mode"] == "replay"
    assert not any(health[name] for name in ("live_intent", "live_attribution", "real_smoke"))
    assert client.app.state.demo_service.project_root is None
    assert client.get("/api/v1/health").status_code == 404
    assert client.get("/").content == (ROOT / "static" / "auto.html").read_bytes()
    assert client.post("/api/auto/run", json={"goal": "研究支化聚乙烯密度"}).json()["intent"]["engine"] == "offline"
    assert client.get("/api/auto/report", params={"format": "json"}).status_code == 200
    case_id = next(iter(json.loads(DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))["cases"]))
    assert client.post("/api/auto/attribution", json={"case_id": case_id}).json()["enabled"] is False
    assert client.post("/api/auto/smoke", json={"case_id": case_id, "mode": "smoke", "confirm": True}).json()["enabled"] is False
    assert client.get("/api/auto/smoke/unknown").status_code == 404
    assert os.environ["POLYMER_AGENT_LIVE_INTENT"] == "1"
    assert os.environ["OPENAI_API_KEY"] == "test-only-never-sent"


def test_default_import_is_lightweight_and_ignores_local_env(tmp_path):
    package = tmp_path / "polymer_agent_demo"
    package.mkdir()
    for filename in ("__init__.py", "app.py", "auto_orchestrator.py", "mcp_bridge.py", "snapshot.py"):
        shutil.copyfile(ROOT / filename, package / filename)
    shutil.copytree(ROOT / "data", package / "data")
    shutil.copytree(ROOT / "static", package / "static")
    (package / ".env").write_text("POLYMER_AGENT_APP_MODE=platform\nPOLYMER_AGENT_DOTENV_SENTINEL=loaded\n", encoding="utf-8")
    code = r'''
import importlib.abc, os, sys
class BlockPlatform(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('polymer_agent_demo.control_plane', 'polymer_agent_demo.live_')) or fullname.split('.')[0] in {'sqlalchemy','redis','rq','paramiko','cryptography','openai','mcp','rdkit','numpy','pandas','yaml'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockPlatform())
from polymer_agent_demo.app import app
assert app.state.application_mode == 'demo'
assert app.state.demo_service.project_root is None
assert app.state.demo_service.health()['real_smoke'] is False
assert app.state.demo_service.auto_run('研究聚乙烯密度')['intent']['engine'] == 'offline'
assert 'POLYMER_AGENT_DOTENV_SENTINEL' not in os.environ
assert not (app.state.demo_service.snapshot_path.parent.parent / '.runtime').exists()
print('offline import isolated')
'''
    env = {key: value for key, value in os.environ.items() if not key.startswith("POLYMER_AGENT_")}
    env.update({"PYTHONPATH": str(tmp_path), "POLYMER_AGENT_ROOT": str(tmp_path / "unrelated_workspace"), "POLYMER_AGENT_REAL_SMOKE": "1", "POLYMER_AGENT_LIVE_INTENT": "1"})
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline import isolated"


def test_application_mode_is_validated():
    with pytest.raises(ValueError, match="demo or platform"):
        create_app(mode="invalid")
    with pytest.raises(ValueError, match="platform"):
        create_app(mode="demo", force_mode="connected")


def test_doctor_reports_stage_dependencies_and_redacts_config(monkeypatch):
    monkeypatch.setattr(doctor, "_available", lambda name: True)
    env = {field: "secret-never-print" for field in doctor.KEY_FIELDS.values()}
    env.update({
        "POLYMER_AGENT_LIVE_DESIGN": "1",
        "POLYMER_AGENT_ROOT": "/private/path/never-print",
        "POLYMER_AGENT_DATABASE_URL": "postgresql://user:secret-never-print@private-host/database",
        "POLYMER_AGENT_DESIGN_MODEL": "custom-planner-model",
        "POLYMER_AGENT_AUDIT_QWEN_MODEL": "custom-audit-model",
    })
    report = doctor.build_report("all", environment=env)
    text = json.dumps(report)
    assert "secret-never-print" not in text
    assert "private-host" not in text
    assert "/private/path/never-print" not in text
    assert report["models"]["design"] == "custom-planner-model"
    assert report["models"]["dashscope"] == "custom-audit-model"
    assert report["stages"]["demo"]["ready"] is True
    assert report["stages"]["design"]["ready"] is True
    assert report["stages"]["md"]["pending"] == ["remote_preflight"]
    assert report["network_checked"] is False


def test_doctor_demo_ready_without_keys_and_reports_missing_web_dep(monkeypatch):
    monkeypatch.setattr(doctor, "_available", lambda name: True)
    report = doctor.build_report(environment={})
    assert report["ready"] is True
    assert report["model_stage_readiness"]["ready_for_design"] is False
    assert doctor.build_report("design", environment={})["stages"]["design"]["missing"] == ["OPENAI_API_KEY", "POLYMER_AGENT_LIVE_DESIGN"]
    assert doctor.build_report("local-validation", environment={})["ready"] is True
    monkeypatch.setattr(doctor, "_available", lambda name: name != "uvicorn")
    assert doctor.build_report(environment={})["stages"]["demo"]["missing"] == ["uvicorn"]


def test_doctor_explicit_env_file_keeps_process_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "process-test-key")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-test-key\nDEEPSEEK_API_KEY=explicit-test-key\n", encoding="utf-8")
    env = doctor._explicit_environment(env_file)
    assert env["OPENAI_API_KEY"] == "process-test-key"
    assert env["DEEPSEEK_API_KEY"] == "explicit-test-key"
    assert "DEEPSEEK_API_KEY" not in os.environ


def test_model_defaults_and_stage_readiness():
    models = model_configuration({})
    assert models == {"design": "gpt-5.5", "upper_planner": "gpt-5.5", "deepseek": "deepseek-v4-pro", "dashscope": "qwen3.7-max", "zhipuai": "glm-4.6"}
    ready = model_stage_readiness({"openai": True})
    assert ready["ready_for_design"] is True
    assert ready["ready_for_audit_panel"] is False
    assert ready["ready_for_closed_loop"] is False


def test_audit_stage_requires_its_three_credentials_before_loading_core(tmp_path):
    with pytest.raises(RuntimeError, match="audit stage"):
        run_audit_panel({}, deepseek_key="", dashscope_key="", zhipuai_key="", cache_dir=tmp_path, approval={})


def test_production_audit_uses_configured_models_without_live_calls(monkeypatch, tmp_path):
    clients_module = ModuleType("agentic.tri_llm_judge")
    captured = []

    class Client:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            captured.append(kwargs)

        def generate(self, prompt):
            raise AssertionError("offline test must never call a model")

    clients_module.DeepSeekClient = clients_module.QwenClient = clients_module.GLMClient = Client
    audit_module = ModuleType("pelcb_warn_audit_llm")
    audit_module.audit_warn_case = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "agentic.tri_llm_judge", clients_module)
    monkeypatch.setitem(sys.modules, "pelcb_warn_audit_llm", audit_module)
    monkeypatch.setattr("polymer_agent_demo.control_plane.production_audit.analysis_path", lambda: tmp_path)
    for field, model in (("DEEPSEEK", "configured-deepseek"), ("QWEN", "configured-qwen"), ("GLM", "configured-glm")):
        monkeypatch.setenv(f"POLYMER_AGENT_AUDIT_{field}_MODEL", model)
    result = run_audit_panel({"case_id": "fixture", "evidence": {}}, deepseek_key="test-only", dashscope_key="test-only", zhipuai_key="test-only", cache_dir=tmp_path, approval={})
    assert [client["model"] for client in captured] == ["configured-deepseek", "configured-qwen", "configured-glm"]
    assert result["allowed_contracts"] == ["review_hold"]


def test_platform_design_ui_allows_import_and_validation_without_keys():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the platform UI behavior check")
    code = r'''
const fs = require('fs'), vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const formElement = {disabled: true}, draftButton = {disabled: true};
const submitButton = {textContent: ''}, tab = {disabled: true};
const nodes = {
  '#onboarding-progress': {}, '#design-readiness': {}, '#draft-design': draftButton,
  '#project-form': {elements: [formElement, draftButton], querySelector: () => submitButton},
  '[data-panel="projects-panel"]': tab,
};
const state = {clusters: [], projects: [], runs: [], modelKeys: {}, health: {executor_enabled: false}};
const context = {state, $: (selector) => nodes[selector]};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function renderOnboarding()'), source.indexOf('async function refreshAll()')), context);
context.renderOnboarding();
if (tab.disabled || formElement.disabled || !draftButton.disabled) throw Error('manual validation should be available without keys');
state.modelKeys = {ready_for_design: true, ready_for_closed_loop: false};
context.renderOnboarding();
if (draftButton.disabled || tab.disabled) throw Error('a planning key should enable candidate generation');
if (submitButton.textContent !== '创建项目并完成本地校验') throw Error('no server: local validation only');
state.clusters = [{status: 'ready'}]; state.health.executor_enabled = true;
context.renderOnboarding();
if (submitButton.textContent !== '创建项目并提交计算') throw Error('prepared executor should enable submission');
console.log('stage-aware UI ready');
'''
    result = subprocess.run([node, "-e", code, str(ROOT / "static" / "platform.js")], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stage-aware UI ready"


def test_repair_planner_uses_design_model_fallback_and_records_id(monkeypatch):
    from polymer_agent_demo.live_execution_plan import UpperRepairPlan, build_upper_repair_plan

    plan = UpperRepairPlan(
        plan_scope="post_audit_bounded_reentry",
        summary="Keep the registered review contract.",
        items=[{"case_ids": ["fixture"], "proposed_contract": "review_hold", "priority": 1, "rationale": "Requires review."}],
    )
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"output_parsed": plan, "id": "offline-test-response"})()

    module = ModuleType("openai")
    module.OpenAI = lambda **kwargs: type("Client", (), {"responses": Responses()})()
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("POLYMER_AGENT_UPPER_PLANNER", "1")
    monkeypatch.delenv("POLYMER_AGENT_UPPER_PLANNER_MODEL", raising=False)
    monkeypatch.setenv("POLYMER_AGENT_DESIGN_MODEL", "configured-planner")
    result = build_upper_repair_plan([{"case_id": "fixture", "allowed_contracts": ["review_hold"]}], {}, api_key="offline-test-key")
    assert captured["model"] == "configured-planner"
    assert result["model"] == "configured-planner"
    assert captured["store"] is False


def test_platform_port_guard_preserves_existing_listener():
    powershell = shutil.which("powershell")
    if os.name != "nt" or powershell is None:
        pytest.skip("Windows PowerShell launcher check")
    source = (ROOT / "run_platform.ps1").read_text(encoding="utf-8")
    assert "Stop-Process" not in source
    assert "Stop-PortListeners" not in source
    function = re.search(r"function Assert-PortAvailable\([^\n]+\) \{.*?\n\}", source, re.S).group(0)
    code = function + r'''
$ErrorActionPreference = "Stop"
$taskListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$taskListener.ExclusiveAddressUse = $true
$taskListener.Start()
$taskPort = $taskListener.LocalEndpoint.Port
$guardRejected = $false
try {
    try { Assert-PortAvailable $taskPort }
    catch {
        if ($_.Exception.Message -notlike '*is occupied*') { throw }
        $guardRejected = $true
    }
    if (-not $guardRejected) { throw 'occupied port was accepted' }
    $probeClient = [System.Net.Sockets.TcpClient]::new()
    $probeClient.Connect([System.Net.IPAddress]::Loopback, $taskPort)
    if (-not $taskListener.Pending()) { throw 'existing listener was changed' }
    $probeClient.Close()
    Write-Output 'existing listener preserved'
} finally { $taskListener.Stop() }
'''
    result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "existing listener preserved"
