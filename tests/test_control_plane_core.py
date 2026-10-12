from __future__ import annotations

import json
import re
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest

from polymer_agent_demo.control_plane.audit import aggregate_panel_votes, sanitize_audit_payload
from polymer_agent_demo.control_plane.design_grammar import candidates_from_choices, design_catalog
from polymer_agent_demo.control_plane.provisioning import deployment_bundle
from polymer_agent_demo.control_plane.schemas import ClusterCreate, ProjectSpec
from polymer_agent_demo.control_plane.science import PELCBAdapter, draft_spec_from_goal, validate_remote_release
from polymer_agent_demo.control_plane.security import decrypt_secret, encrypt_secret, generate_ssh_keypair, hash_password, verify_password
from polymer_agent_demo.control_plane.slurm import ClusterRuntime, SSHSlurmBackend, safe_remote_path
from polymer_agent_demo.control_plane.state import require_transition
from polymer_agent_demo.deploy import install_science_runtime as science_runtime_installer
from polymer_agent_demo.live_execution_plan import ExecutionPlanItem, UpperExecutionPlan, build_upper_execution_plan
from polymer_agent_demo.live_design import ModelDesignChoice, ModelDesignPlan, _prompt, propose_designs, requested_count_from_goal
from polymer_agent_demo.deploy.install_science_runtime import (
    EMC_BINARIES,
    EMC_DISTRIBUTION,
    EMC_VERSION,
    GROMACS_SHA256,
    GROMACS_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cluster_site_defaults_choose_available_gpu_partition_and_remote_home(monkeypatch):
    backend = object.__new__(SSHSlurmBackend)
    monkeypatch.setattr(backend, "_connect", lambda: nullcontext(object()))
    monkeypatch.setattr(
        backend,
        "_run",
        lambda client, command: {
            "ok": True,
            "stdout": "HOME=/data/home/alice\nSCHEDULER=slurm\ncpu*|(null)|up\ngpu|gpu:a100:4|up\ngpu-old|gpu:v100:2|down\n",
            "stderr": "",
        },
    )

    detected = backend.detect_site_defaults()

    assert detected["partition"] == "gpu"
    assert detected["remote_root"] == "/data/home/alice"
    assert len(detected["available_partitions"]) == 3


def test_cluster_site_defaults_detect_manual_gpu_node_pool(monkeypatch):
    backend = object.__new__(SSHSlurmBackend)
    monkeypatch.setattr(backend, "_connect", lambda: nullcontext(object()))
    def fake_run(client, command, timeout=60):
        if command.startswith("for node in"):
            return {"ok": True, "stdout": "NODE=node-gpu1\nNODE=node-gpu2\n", "stderr": ""}
        return {
            "ok": True,
            "stdout": "HOME=/srv/users/alice\nSCHEDULER=manual\n10.0.0.2 node-gpu2\n10.0.0.1 node-gpu1\n",
            "stderr": "",
        }

    monkeypatch.setattr(backend, "_run", fake_run)

    detected = backend.detect_site_defaults()

    assert detected["scheduler_mode"] == "manual"
    assert detected["partition"] == "manual"
    assert detected["manual_nodes"] == ["node-gpu1", "node-gpu2"]
    assert detected["remote_root"] == "/srv/users/alice"


def test_manual_node_pool_submits_bounded_background_array(monkeypatch):
    runtime = ClusterRuntime(
        host="cluster.example.edu",
        port=22,
        username="alice",
        host_key_sha256="SHA256:abcdefghijklmnopqrstuvwxyz0123456789",
        private_key_pem="unused",
        partition="manual",
        account=None,
        remote_root="/home/alice/polymer-agent",
        gromacs_command="gmx",
        emc_command="emc",
        python_command="python3",
        runner_command="polymer-agent-remote",
        module_init=[],
        limits={"allowed_nodes": [], "max_cpus": 8, "max_gpus": 1, "max_walltime_hours": 24},
        environment={"scheduler_mode": "manual", "manual_nodes": ["node-gpu1", "node-gpu2"]},
    )
    backend = SSHSlurmBackend(runtime)
    written: dict[str, bytes] = {}
    commands: list[str] = []

    class Sink:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def write(self, content):
            written[self.path] = bytes(content)

        def __exit__(self, *args):
            return False

    class Sftp:
        def file(self, path, mode):
            return Sink(path)

        def close(self):
            return None

    class Client:
        def open_sftp(self):
            return Sftp()

    monkeypatch.setattr(backend, "_connect", lambda: nullcontext(Client()))

    def fake_run(client, command, timeout=60):
        commands.append(command)
        if command.startswith("cat "):
            return {"ok": True, "stdout": "COMPLETED\n", "stderr": "", "exit_code": 0}
        return {"ok": True, "stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(backend, "_run", fake_run)
    script = backend.render_array_script(
        run_id="run123456",
        stage="build",
        command=["polymer-agent-remote", "stage", "--stage", "build"],
        case_count=4,
        max_parallel=4,
        cpus=2,
        gpus=0,
        walltime_hours=1,
    )

    job_id, remote_dir = backend.submit_stage_bundle(
        run_id="run123456",
        stage="build",
        files={"manifest.json": b"[]"},
        script=script,
    )

    assert job_id.startswith("manual-")
    assert remote_dir.endswith("/polymer-agent/run123456")
    launcher = next(content.decode() for path, content in written.items() if path.endswith("stage-build.manual.sh"))
    assert "node-gpu1 node-gpu2" in launcher
    assert "SLURM_ARRAY_TASK_ID=$idx" in launcher
    assert any("nohup setsid" in command for command in commands)
    assert backend.status(job_id) == "COMPLETED"


def spec(**overrides):
    payload = {
        "natural_language_goal": "PE-LCB C100",
        "candidates": [
            {
                "case_id": "trial_001",
                "branch_points": 1,
                "branch_carbons": 5,
                "branch_distribution_code": "mid",
                "target_total_carbons": 100,
            }
        ],
    }
    payload.update(overrides)
    return ProjectSpec.model_validate(payload)


def test_validated_domain_and_out_of_domain_warn():
    adapter = PELCBAdapter()
    valid = adapter.validate(spec())
    assert valid.valid is True
    assert valid.domain_status == "validated_domain"
    assert len(valid.manifest_sha256) == 64

    outside = adapter.validate(spec(temperature_k=450))
    assert outside.valid is True
    assert outside.domain_status == "warn"
    assert {"field": "temperature_k", "code": "outside_validated_pelcb100_protocol"} in outside.warnings

    with pytest.raises(ValueError):
        spec(production_ns=800)


def test_duplicate_case_id_is_rejected_but_repeated_coarse_coordinate_is_warned():
    raw = spec().model_dump()
    raw["candidates"].append({**raw["candidates"][0], "case_id": "trial_002"})
    result = PELCBAdapter().validate(ProjectSpec.model_validate(raw))
    assert result.valid is True
    assert not result.errors
    assert {item["code"] for item in result.warnings} == {"repeated_coarse_design_coordinate"}

    raw["candidates"][1]["case_id"] = "trial_001"
    result = PELCBAdapter().validate(ProjectSpec.model_validate(raw))
    assert result.valid is False
    assert {item["code"] for item in result.errors} == {"duplicate_case_id"}


def test_standard_pelcb100_draft_validates_as_100_case_domain():
    draft = draft_spec_from_goal("研究 C100 支化聚乙烯的密度、结构和动力学")
    assert len(draft.candidates) == 100
    assert draft.candidates[0].case_id == "lcb_pe_001_bp01_bc005_pdmid_c101"
    assert draft.candidates[-1].case_id == "lcb_pe_100_bp05_bc008_pdgrad_c100"
    result = PELCBAdapter().validate(draft)
    assert result.valid is True
    assert result.domain_status == "validated_domain"
    assert len(result.manifest) == 100


def test_dynamic_design_grammar_builds_unique_concrete_structures():
    catalog = design_catalog()
    assert len(catalog) >= 100
    assert len({item["structure_sha256"] for item in catalog.values()}) == len(catalog)
    candidates = candidates_from_choices(
        [
            {"design_key": "b1-L5-mid", "rationale": "单支化基线"},
            {"design_key": "b3-L8-three-point", "rationale": "三点分布"},
            {"design_key": "b5-L5-gradient-5", "rationale": "高支化梯度"},
        ]
    )
    dynamic = spec(candidates=candidates)
    result = PELCBAdapter().validate(dynamic)
    assert result.valid is True
    assert result.domain_status == "validated_domain"
    assert all(row["emc_ready"] is True for row in result.manifest)
    assert all(row["builder_spec"]["monomer_smiles"] == row["macro_smiles"] for row in result.manifest)

    tampered = dynamic.model_dump()
    tampered["candidates"][0]["macro_smiles"] = "*CC*"
    invalid = PELCBAdapter().validate(ProjectSpec.model_validate(tampered))
    assert invalid.valid is False
    assert "deterministic_builder_spec_mismatch" in {item["code"] for item in invalid.errors}


def test_requested_design_count_is_parsed_and_bounded():
    assert requested_count_from_goal("设计 50 个支化聚乙烯") == 50
    assert requested_count_from_goal("设计五十个支化聚乙烯") == 50
    assert requested_count_from_goal("single smoke case") == 1
    assert requested_count_from_goal("研究 PE-LCB") == 100
    with pytest.raises(ValueError):
        requested_count_from_goal("设计 101 个支化聚乙烯")


def test_live_design_uses_structured_response_and_application_validation(monkeypatch):
    import openai

    class FakeResponse:
        id = "resp_structured"
        output_parsed = ModelDesignPlan(
            requested_count=2,
            strategy_summary="覆盖单支化与多支化",
            choices=[
                ModelDesignChoice(branch_points=1, branch_carbons=5, placement_code="mid", rationale="单支化基线"),
                ModelDesignChoice(branch_points=4, branch_carbons=8, placement_code="gradient", rationale="多支化梯度"),
            ],
        )

    class FakeResponses:
        def parse(self, **kwargs):
            assert kwargs["model"] == "gpt-5.5"
            assert kwargs["text_format"] is ModelDesignPlan
            assert kwargs["store"] is False
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("POLYMER_AGENT_LIVE_DESIGN", "1")
    monkeypatch.setenv("POLYMER_AGENT_DESIGN_MODEL", "gpt-5.5")
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: FakeClient())
    result = propose_designs("设计 2 个支化聚乙烯", 2)
    assert result["response_id"] == "resp_structured"
    assert len(result["candidates"]) == 2
    assert len({item["structure_sha256"] for item in result["candidates"]}) == 2


def test_live_design_prompt_exposes_rules_not_enumerated_answers():
    prompt = _prompt("设计 3 个支化聚乙烯", 3)
    assert "branch_points=1: mid, near-end" in prompt
    assert "b1-L5-mid" not in prompt
    assert "macro_smiles" not in prompt.lower()
    assert "Do not output a design_key" in prompt


def test_live_design_retries_when_proposal_breaks_registered_grammar(monkeypatch):
    import openai

    invalid = ModelDesignPlan(
        requested_count=1,
        strategy_summary="第一次故意给出不相容放置方式",
        choices=[ModelDesignChoice(branch_points=1, branch_carbons=5, placement_code="uniform", rationale="invalid")],
    )
    valid = ModelDesignPlan(
        requested_count=1,
        strategy_summary="修正为单支化中心位置",
        choices=[ModelDesignChoice(branch_points=1, branch_carbons=5, placement_code="mid", rationale="valid")],
    )

    class FakeResponse:
        def __init__(self, parsed, response_id):
            self.output_parsed = parsed
            self.id = response_id

    class FakeResponses:
        calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            return FakeResponse(invalid if self.calls == 1 else valid, f"resp_{self.calls}")

    responses = FakeResponses()

    class FakeClient:
        pass

    client = FakeClient()
    client.responses = responses
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("POLYMER_AGENT_LIVE_DESIGN", "1")
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    result = propose_designs("设计 1 个支化聚乙烯", 1)
    assert result["attempt"] == 2
    assert result["candidates"][0]["placement_code"] == "mid"


def test_upper_execution_plan_uses_openai_structured_response(monkeypatch):
    import openai

    class FakeResponse:
        id = "resp_upper_plan"
        output_parsed = UpperExecutionPlan(
            plan_scope="pre_run_bounded_closed_loop",
            summary="按注册 MD 管线执行，后续 WARN 只允许进入预注册修复契约。",
            items=[
                ExecutionPlanItem(
                    case_selector="all_locked_cases",
                    trigger_state="locked_manifest_ready",
                    proposed_contract="run_registered_md_pipeline",
                    priority=1,
                    rationale="项目已锁定，先执行注册模拟、后处理和规则判定。",
                )
            ],
        )

    class FakeResponses:
        def parse(self, **kwargs):
            assert kwargs["model"] == "gpt-5.5"
            assert kwargs["text_format"] is UpperExecutionPlan
            assert kwargs["store"] is False
            assert "registered_contracts" in kwargs["input"]
            assert "Final PASS is decided only by rule re-gating" in kwargs["input"]
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("POLYMER_AGENT_UPPER_PLANNER", "1")
    monkeypatch.setenv("POLYMER_AGENT_UPPER_PLANNER_MODEL", "gpt-5.5")
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: FakeClient())
    plan = build_upper_execution_plan(
        project={
            "id": "project_1",
            "name": "PE-LCB trial",
            "status": "locked",
            "domain_status": "validated_domain",
            "manifest_sha256": "a" * 64,
            "spec": {
                "polymer_family": "PE",
                "architecture": "pe_lcb",
                "force_field": "opls/2024/opls-aa",
                "temperature_k": 400,
                "pressure_atm": 1,
                "production_ns": 400,
                "rule_profile": "pelcb100_validated_v1",
            },
            "validation": {"valid": True, "domain_status": "validated_domain"},
            "manifest": [{"case_id": "case_001"}],
        },
        run={"id": "run_1", "state": "awaiting_approval", "cluster_id": "cluster_1"},
        approval={"max_repair_attempts": 1, "max_additional_md_ns": 400, "max_cases": 100},
    )
    assert plan["response_id"] == "resp_upper_plan"
    assert plan["guard_status"] == "metadata_only_requires_registered_contract_match"
    assert plan["plan"]["items"][0]["proposed_contract"] == "run_registered_md_pipeline"


def test_password_secret_and_ssh_key_material():
    encoded = hash_password("a sufficiently long password")
    assert verify_password("a sufficiently long password", encoded)
    assert not verify_password("wrong password", encoded)
    cipher = encrypt_secret("private", "deployment-key")
    assert "private" not in cipher
    assert decrypt_secret(cipher, "deployment-key") == "private"
    private, public = generate_ssh_keypair()
    assert "PRIVATE KEY" in private
    assert public.startswith("ssh-ed25519 ")


def test_run_state_does_not_skip_approval():
    require_transition("planned", "awaiting_approval")
    with pytest.raises(ValueError):
        require_transition("planned", "running")


def test_remote_path_and_slurm_template_are_bounded():
    assert safe_remote_path("/home/user/work", "polymer-agent", "abc") == "/home/user/work/polymer-agent/abc"
    with pytest.raises(ValueError):
        safe_remote_path("/home/user/work", "../escape")
    runtime = ClusterRuntime(
        host="cluster.example.edu",
        port=22,
        username="alice",
        host_key_sha256="SHA256:example",
        private_key_pem="unused",
        partition="gpu",
        account=None,
        remote_root="/home/alice/work",
        gromacs_command="gmx",
        emc_command="emc",
        python_command="python3",
        runner_command="polymer-agent-remote",
        module_init=["module load gromacs/2024"],
        limits={"max_cpus": 16, "max_gpus": 2, "max_walltime_hours": 200, "allowed_nodes": []},
        environment={"mode": "module", "pipeline_command": "pelcb-pipeline-core"},
    )
    script = SSHSlurmBackend(runtime).render_script(
        run_id="abcdef012345", command=["polymer-agent-remote", "run", "--manifest", "manifest.json"], cpus=8, gpus=1, walltime_hours=100
    )
    assert "#SBATCH --cpus-per-task=8" in script
    assert "polymer-agent-remote run --manifest manifest.json" in script
    with pytest.raises(ValueError):
        SSHSlurmBackend(runtime).render_script(run_id="x", command=["ok"], cpus=32, gpus=1, walltime_hours=1)


def test_cluster_environment_contract_and_deployment_bundle():
    module = ClusterCreate.model_validate(
        {
            "name": "lab", "host": "cluster.example.edu", "remote_username": "alice",
            "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu",
            "remote_root": "/home/alice/work", "environment_mode": "module",
            "module_init": ["module load gromacs/2024", "module load pelcb-pipeline/1"],
        }
    )
    assert module.environment_mode == "module"
    container = ClusterCreate.model_validate(
        {
            "name": "lab", "host": "cluster.example.edu", "remote_username": "alice",
            "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu",
            "remote_root": "/home/alice/work", "environment_mode": "container",
            "container_runtime": "apptainer", "container_image": "/shared/pelcb/pipeline.sif",
        }
    )
    assert container.container_image == "/shared/pelcb/pipeline.sif"
    managed = ClusterCreate.model_validate(
        {
            "name": "managed", "host": "cluster.example.edu", "remote_username": "alice",
            "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu",
            "remote_root": "/home/alice/work", "environment_mode": "module",
            "science_install_mode": "managed", "upstream_terms_accepted": True,
        }
    )
    assert managed.gromacs_acceleration == "auto"
    with pytest.raises(ValueError):
        ClusterCreate.model_validate(
            {
                "name": "managed", "host": "cluster.example.edu", "remote_username": "alice",
                "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu",
                "remote_root": "/home/alice/work", "science_install_mode": "managed",
            }
        )
    with pytest.raises(ValueError):
        ClusterCreate.model_validate(
            {
                "name": "lab", "host": "cluster.example.edu", "remote_username": "alice",
                "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu",
                "remote_root": "/home/alice/work", "environment_mode": "container",
            }
        )
    version, files = deployment_bundle()
    assert version.startswith("v1-")
    assert {
        "remote_runner.py",
        "site_pipeline_adapter.py",
        "pelcb_pipeline_core.py",
        "install_science_runtime.py",
        "run_real_smoke.py",
        "pelcb_audit_diagnosis.py",
        "diagnosis_candidate_registry.py",
        "mdp/steep.mdp",
        "fixture/lcb01_star3_b1_l05_c101.gro",
        "fixture/topol.top",
        "fixture/lcb01_star3_b1_l05_c101_polymer.itp",
    } <= set(files)
    assert all(files.values())
    assert "science_core_bundle.zip" in files
    assert GROMACS_VERSION == "2024.4"
    assert EMC_VERSION == "9.4.4"
    assert len(GROMACS_SHA256) == 64
    assert EMC_DISTRIBUTION["version"] == "9.4.4+20260721"
    assert len(EMC_DISTRIBUTION["sha256"]) == 64
    assert {"x86_64", "aarch64"} <= set(EMC_BINARIES)
    with zipfile.ZipFile(ROOT / "deploy" / "science_core_bundle.zip") as archive:
        names = set(archive.namelist())
        assert "science_core/generated_code/md_analysis/polymer_qc_rules.py" in names
        assert "science_core/generated_code/md_analysis/rule_profile_registry.py" in names
        assert "science_core/generated_code/md_analysis/pelcb_warn_audit_llm.py" in names
        assert "science_core/generated_code/md_analysis/polymer_design_validator.py" in names
        assert "science_core/generated_code/md_analysis/polymer_template_builder.py" in names
        forbidden = [
            re.compile(rb"\b(?:10\.|192\.168\.)\d{1,3}\.\d{1,3}\b"),
            re.compile(rb"/(?:home|public/home)/[A-Za-z0-9_.-]+/"),
            re.compile(rb"[A-Za-z]:\\Users\\[^\\]+\\", re.IGNORECASE),
        ]
        for name in names:
            if name.endswith("/"):
                continue
            content = archive.read(name)
            assert not any(pattern.search(content) for pattern in forbidden), name


def test_container_environment_wraps_science_commands_only():
    runtime = ClusterRuntime(
        host="cluster.example.edu", port=22, username="alice", host_key_sha256="SHA256:example",
        private_key_pem="unused", partition="gpu", account=None, remote_root="/home/alice/work",
        gromacs_command="gmx", emc_command="emc", python_command="python3",
        runner_command="/home/alice/work/.polymer-agent/bin/polymer-agent-remote", module_init=[],
        limits={"max_cpus": 16, "max_gpus": 2, "max_walltime_hours": 200, "allowed_nodes": []},
        environment={
            "mode": "container", "container_runtime": "apptainer",
            "container_image": "/shared/pelcb/pipeline.sif", "pipeline_command": "pelcb-pipeline-core",
        },
    )
    backend = SSHSlurmBackend(runtime)
    assert backend._environment_command(["gmx", "--version"]) == "apptainer exec /shared/pelcb/pipeline.sif gmx --version"


def test_science_runtime_download_rejects_truncated_http_response(monkeypatch, tmp_path):
    class TruncatedResponse:
        headers = {"Content-Length": "8"}

        def __init__(self):
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            if self.sent:
                return b""
            self.sent = True
            return b"abcd"

    monkeypatch.setattr(science_runtime_installer.urllib.request, "urlopen", lambda request, timeout: TruncatedResponse())
    monkeypatch.setattr(science_runtime_installer.shutil, "which", lambda name: None)
    monkeypatch.setattr(science_runtime_installer.time, "sleep", lambda seconds: None)

    expected = science_runtime_installer.hashlib.sha256(b"abcdefgh").hexdigest()
    with pytest.raises(RuntimeError, match="download failed after automatic retries"):
        science_runtime_installer.download("https://example.test/file.bin", tmp_path / "file.bin", expected)

    assert not (tmp_path / "file.bin").exists()


def test_science_runtime_download_falls_back_to_curl(monkeypatch, tmp_path):
    payload = b"abcdefgh"
    expected = science_runtime_installer.hashlib.sha256(payload).hexdigest()

    class TruncatedResponse:
        headers = {"Content-Length": "8"}

        def __init__(self):
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            if self.sent:
                return b""
            self.sent = True
            return b"abcd"

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, text, capture_output, timeout, check):
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_bytes(payload)
        return Completed()

    monkeypatch.setattr(science_runtime_installer.urllib.request, "urlopen", lambda request, timeout: TruncatedResponse())
    monkeypatch.setattr(science_runtime_installer.shutil, "which", lambda name: "/usr/bin/curl" if name == "curl" else None)
    monkeypatch.setattr(science_runtime_installer.subprocess, "run", fake_run)
    monkeypatch.setattr(science_runtime_installer.time, "sleep", lambda seconds: None)

    target = tmp_path / "file.bin"
    science_runtime_installer.download("https://example.test/file.bin", target, expected)

    assert target.read_bytes() == payload


def test_audit_payload_removes_deterministic_conclusions():
    payload = {
        "warn_sources": ["msd_warn"],
        "evidence": {"slope": 0.2, "available_repair_tools": [{"repair_tool": "extend"}]},
        "primary_audit_diagnosis": "fit_window_or_method_issue",
        "diagnosis_candidates": ["fit_window_or_method_issue"],
    }
    clean = sanitize_audit_payload(payload)
    text = json.dumps(clean)
    assert "primary_audit_diagnosis" not in text
    assert "available_repair_tools" not in text
    assert clean["evidence"]["slope"] == 0.2


def test_consensus_requires_cause_route_and_sufficient_information():
    votes = [
        {"cause": "protocol", "route": "clean", "information_sufficiency": "sufficient", "confidence": 0.9},
        {"cause": "protocol", "route": "clean", "information_sufficiency": "sufficient", "confidence": 0.5},
        {"cause": "protocol", "route": "review", "information_sufficiency": "sufficient", "confidence": 0.99},
    ]
    result = aggregate_panel_votes(votes)
    assert result["decision"] == "route"
    assert result["agreement"] == "2/3"
    assert result["model_self_confidence"] == [0.9, 0.5]

    votes[1]["information_sufficiency"] = "insufficient"
    assert aggregate_panel_votes(votes)["decision"] == "review"


def test_deployment_bundle_declares_required_services():
    import yaml

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert {"api", "worker", "reconciler", "postgres", "redis", "proxy"} <= set(compose["services"])
    assert (ROOT / "Dockerfile").exists()
    assert (ROOT / "deploy" / "remote_runner.py").exists()


def test_product_page_keeps_frozen_demo_separate():
    html = (ROOT / "static" / "platform.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "platform.js").read_text(encoding="utf-8")
    assert 'href="/demo"' in html
    assert 'href="/auto"' in html
    assert "冻结演示" in html
    assert "本机验证模式" in js
    assert "管理员授权" in js
    assert "执行器未启用" in js
    assert "/api/v1/projects" in js
    assert "/api/v1/clusters" in js
    assert "/api/v1/runs" in js
    assert "/releases" in js
    assert "调用 GPT-5.5 设计 n 个候选" in html
    assert 'id="structure-preview"' in html
    assert "macro_smiles" in js
    assert 'id="start-local-acceptance"' in html
    assert "/api/v1/local-acceptance" in js


def test_remote_release_rejects_false_800ns_pass():
    payload = {
        "schema_version": "polymer_agent.remote_release.v1",
        "publication_state": "pass",
        "config_sha256": "a" * 64,
        "provenance": [{"source": "canonical"}],
        "cases": [
            {
                "case_id": "case_001",
                "quality_status": "pass",
                "property_masks": {"density": True, "static": True, "diffusion_strict": True},
                "audit_votes": [],
                "repair_history": [],
                "msd_warn": True,
                "msd_total_ns": 800,
            }
        ],
    }
    with pytest.raises(ValueError, match="800 ns"):
        validate_remote_release(payload, project_domain_status="validated_domain")
    payload["cases"][0]["quality_status"] = "review"
    payload["publication_state"] = "review"
    validate_remote_release(payload, project_domain_status="validated_domain")


def test_out_of_domain_remote_release_cannot_auto_pass():
    payload = {
        "schema_version": "polymer_agent.remote_release.v1",
        "publication_state": "pass",
        "config_sha256": "b" * 64,
        "provenance": [{"source": "canonical"}],
        "cases": [
            {
                "case_id": "case_001",
                "quality_status": "pass",
                "property_masks": {"density": True, "static": True, "diffusion_strict": False},
                "audit_votes": [],
                "repair_history": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="out-of-domain"):
        validate_remote_release(payload, project_domain_status="review_required")


def test_remote_release_rejects_private_paths():
    payload = {
        "schema_version": "polymer_agent.remote_release.v1",
        "publication_state": "review",
        "config_sha256": "c" * 64,
        "provenance": [{"source": "/home/alice/private/result.csv"}],
        "cases": [
            {
                "case_id": "case_001",
                "quality_status": "review",
                "property_masks": {"density": False, "static": False, "diffusion_strict": False},
                "audit_votes": [],
                "repair_history": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="private path"):
        validate_remote_release(payload, project_domain_status="validated_domain")
