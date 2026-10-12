from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_bundle_uses_repository_source_and_keeps_complete_audit_imports(monkeypatch, tmp_path):
    builder = load_file("portable_core_builder", ROOT / "scripts/build_science_core_bundle.py")
    monkeypatch.setenv("POLYMER_AGENT_ROOT", str(tmp_path / "unrelated_workspace"))
    assert builder.source_root() == ROOT / "science_core"
    with zipfile.ZipFile(ROOT / "deploy/science_core_bundle.zip") as archive:
        names = set(archive.namelist())
        for relative in ["agentic/schemas/judgment.py", "lcb_project_config.py",
                         "configs/pelcb100.json", "mcp/pelcb_mcp_server.py"]:
            assert "science_core/generated_code/md_analysis/" + relative in names
        manifest = json.loads(archive.read("science_core/manifest.json"))
        assert manifest["version"] == (ROOT / "science_core/VERSION").read_text().strip()
        for row in manifest["files"]:
            content = archive.read("science_core/" + row["path"])
            assert hashlib.sha256(content).hexdigest() == row["sha256"]
            assert content == (ROOT / "science_core" / row["path"]).read_bytes()


def test_core_imports_from_an_isolated_extraction_without_workspace(monkeypatch, tmp_path):
    runtime = load_file("portable_extraction_runtime", ROOT / "control_plane/science_core_runtime.py")
    with zipfile.ZipFile(ROOT / "deploy/science_core_bundle.zip") as archive:
        runtime.extract_bundle(archive, tmp_path)
    analysis = tmp_path / "science_core/generated_code/md_analysis"
    script = """
import importlib.util, pathlib, sys
p=pathlib.Path(sys.argv[1]); sys.path.insert(0,str(p))
from agentic.schemas.judgment import validate_judgment
import agentic.tri_llm_judge, pelcb_warn_audit_llm
spec=importlib.util.spec_from_file_location('original_mcp',p/'mcp/pelcb_mcp_server.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
result=module.validate_polymer_design_candidate_impl({'candidate_id':'valid','polymer_family':'polyether','polymer_smiles':'CCOCC(C)CCN','chain_count':4})
assert result['validator_status']=='pass'
print('isolated original MCP, schemas and audit import pass')
"""
    environment = os.environ.copy()
    for name in ["PELCB_PROJECT_CONFIG", "LCB_PROJECT_CONFIG", "PYTHONPATH"]:
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-I", "-c", script, str(analysis)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "isolated original MCP" in result.stdout


def test_original_md_template_keeps_stage_gpu_resume_merge_and_zero_warning_contract(monkeypatch, tmp_path):
    worker = load_file("portable_production_worker", ROOT / "deploy/production_case.py")
    monkeypatch.delenv("POLYMER_AGENT_SCIENCE_ROOT", raising=False)
    monkeypatch.setenv("POLYMER_AGENT_MDRUN_MODE", "gpu")
    monkeypatch.setenv("POLYMER_AGENT_GPU_ID", "0")
    md = tmp_path / "case_001/md"
    md.mkdir(parents=True)
    (md / "case_001.gro").write_text("input fixture")
    script, switches = worker.render_original_md_script(md, "gmx", 2)
    assert 'MDRUN_GPU_ARGS="-gpu_id 0 -nb gpu"' in script
    assert 'rlx)          STAGE_GPU_EXTRA="-update gpu -bonded gpu"' in script
    assert 'pr)           STAGE_GPU_EXTRA="-bonded gpu"' in script
    assert 'nvt_prod)     STAGE_GPU_EXTRA="-bonded gpu -notunepme"' in script
    assert '-cpi "${name}.cpt" -noappend' in script
    assert "archive_stage_outputs" in script and "merge_stage_parts" in script
    assert switches == {"RLX_MAXWARN": "0", "MERGE_PARTS": "1", "ENABLE_GPU_EXTRA": "1"}


def test_density_requires_complete_final_window_and_merges_resume_energy(monkeypatch, tmp_path):
    worker = load_file("portable_density_worker", ROOT / "deploy/production_case.py")
    case = tmp_path / "case"
    pr = case / "md/pr"
    pr.mkdir(parents=True)
    (pr / "pr.edr").write_bytes(b"first segment")
    (pr / "pr.part0002.edr").write_bytes(b"second segment")
    output = case / "evidence"
    output.mkdir()
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if "energy" in command:
            (output / "pr_density.xvg").write_text("40000 800\n50000 800\n")
    monkeypatch.setattr(worker, "run", fake_run)
    result = worker.extract_density(case, "gmx", output)
    assert "eneconv" in calls[0]
    assert result["density_g_cm3"] == 0.8
    assert result["density_time_first_ps"] == 40000
    assert result["density_time_last_ps"] == 50000


def test_wheel_fixture_fallback_rebuilds_identically_and_provisions(monkeypatch, tmp_path):
    builder = load_file("wheel_fixture_builder", ROOT / "scripts/build_science_core_bundle.py")
    temporary = (tmp_path / "wheel").resolve()
    if os.name == "nt":
        temporary = Path("\\\\?\\" + str(temporary))
    temporary.mkdir()
    def exclude_fixtures(directory, names):
        if Path(directory).name == "lcb01_star3_b1_l05_c101":
            return [name for name in names if name in {
                "lcb01_star3_b1_l05_c101.gro", "topol.top", "lcb01_star3_b1_l05_c101_polymer.itp"}]
        return [name for name in names if name in {"__pycache__", ".pytest_cache"}]
    shutil.copytree(ROOT / "science_core", temporary / "science_core", ignore=exclude_fixtures)
    deploy = temporary / "deploy"
    deploy.mkdir()
    for name in ["science_core_bundle.zip", "remote_runner.py", "site_pipeline_adapter.py",
                 "pelcb_pipeline_core.py", "production_case.py", "install_science_runtime.py"]:
        shutil.copy2(ROOT / "deploy" / name, deploy / name)
    monkeypatch.setattr(builder, "SOURCE", temporary / "science_core")
    monkeypatch.setattr(builder, "OUTPUT", deploy / "science_core_bundle.zip")
    original = hashlib.sha256(builder.OUTPUT.read_bytes()).hexdigest()
    built = builder.build()
    assert built["packaged_fixture_count"] == 3
    assert built["fixture_storage"] == "verified_same_version_zip"
    assert built["file_count"] == 153
    assert built["sha256"] == original
    from polymer_agent_demo.control_plane import provisioning
    monkeypatch.setattr(provisioning, "DEPLOY_ROOT", deploy)
    monkeypatch.setattr(provisioning, "build_science_core_bundle", builder.build)
    version, files = provisioning.deployment_bundle()
    assert version.startswith("v1-")
    for name in ["fixture/lcb01_star3_b1_l05_c101.gro", "fixture/topol.top",
                 "fixture/lcb01_star3_b1_l05_c101_polymer.itp"]:
        assert files[name]
    assert hashlib.sha256(builder.OUTPUT.read_bytes()).hexdigest() == original


def test_recommended_public_configs_parse_as_gromacs_2024_with_operator_resources(monkeypatch, tmp_path):
    analysis = ROOT / "science_core/generated_code/md_analysis"
    loader = load_file("portable_project_config", analysis / "lcb_project_config.py")
    for name, value in {"REMOTE_USER": "researcher", "REMOTE_HOST": "cluster.example.edu",
                        "PELCB100_REMOTE_HOME": "/srv/user", "PELCB100_REMOTE_ROOT": "/srv/project"}.items():
        monkeypatch.setenv(name, value)
    for name in ("pelcb100.json", "pelcb100.public.json"):
        path = analysis / "configs" / name
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        assert not any(value in raw for value in ("2021.2", "gromacs2021", "private-", "liuz@", "node-gpu"))
        config = loader.load_lcb_config(path)
        assert config.project_id == "pelcb100"
        assert config.case_roots and config.manifest_csv
        assert config.cluster_profiles
        for profile in config.cluster_profiles.values():
            assert profile.engine_version_expected == "GROMACS 2024.4"
            assert profile.module_preset == "none"
            assert profile.host == "researcher@cluster.example.edu"
            assert profile.allowed_nodes == () and profile.threads_by_node == {}
        assert payload["rho_pass_low_g_per_cc"] == 0.73
        assert payload["rho_pass_high_g_per_cc"] == 0.87
        assert payload["compaction_density_tiers"]["reference_count"] == 40
        # A receiver supplies their own executable, node whitelist and work root.
        chosen = payload["cluster_profiles"]["o7_clean"]
        chosen.update(gmx_bin="/opt/gromacs-2024.4/bin/gmx", allowed_nodes=["compute-01"],
                      remote_root="/srv/project/new_run")
        local = tmp_path / name
        local.write_text(json.dumps(payload), encoding="utf-8")
        operator = loader.load_lcb_config(local).cluster_profile("o7_clean")
        assert operator.gmx_bin == "/opt/gromacs-2024.4/bin/gmx"
        assert operator.allowed_nodes == ("compute-01",)
        assert operator.remote_root == "/srv/project/new_run"
    mcp = (analysis / "mcp/pelcb_mcp_server.py").read_text(encoding="utf-8")
    assert 'MD_ANALYSIS_DIR / "configs" / "pelcb100.json"' in mcp
