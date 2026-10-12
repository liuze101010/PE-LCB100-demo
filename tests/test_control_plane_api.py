from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from polymer_agent_demo.app import create_app
from polymer_agent_demo.control_plane.db import Base, get_db
from polymer_agent_demo.control_plane.models import ClusterConnection, LoginSession, Project, ReleaseSnapshot, ReviewRecord, RunEvent, User, WorkflowRun
from polymer_agent_demo.control_plane.security import hash_password
from polymer_agent_demo.control_plane.design_grammar import candidates_from_choices, catalog_sha256, design_catalog


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYMER_AGENT_EXECUTOR_ENABLED", "0")
    test_engine = create_engine(f"sqlite:///{(tmp_path / 'control-plane-test.db').as_posix()}", connect_args={"check_same_thread": False})
    TestingSession = sessionmaker(bind=test_engine, expire_on_commit=False)
    Base.metadata.create_all(test_engine)
    with TestingSession() as db:
        db.add_all(
            [
                User(username="admin", password_hash=hash_password("admin password 123"), role="admin"),
                User(username="alice", password_hash=hash_password("alice password 123"), role="researcher"),
                User(username="bob", password_hash=hash_password("bob password 12345"), role="researcher"),
            ]
        )
        db.commit()
    app = create_app(force_mode="replay", mode="platform")

    def override_db():
        with TestingSession() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def login(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["csrf_token"]


def test_auth_me_returns_current_csrf_for_browser_recovery(client):
    csrf = login(client, "alice", "alice password 123")

    response = client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["csrf_token"] == csrf


def test_polymer_setup_preview_selects_registered_builder_and_force_field(client):
    csrf = login(client, "alice", "alice password 123")

    denied = client.post(
        "/api/v1/polymer-setup/preview",
        json={"repeat_unit_smiles": "*CCO*"},
    )
    assert denied.status_code == 403

    response = client.post(
        "/api/v1/polymer-setup/preview",
        json={"repeat_unit_smiles": "*CCO*"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    payload = response.json()
    setup = payload["automatic_setup"]
    assert setup["architecture_selection"]["architecture"] == "linear_homopolymer"
    assert setup["builder_selection"]["builder_backend"] == "emc_repeat_unit_builder"
    assert setup["force_field_selection"]["force_field"] == "opls-aa"
    assert setup["force_field_selection"]["parameterization_probe_required"] is True
    assert payload["next_action"] == "automatic_remote_parameterization_probe"
    assert payload["preparation_rule"]["rule_verdict"] == "WARN"
    assert payload["preparation_rule"]["audit_required"] is True

    unsupported = client.post(
        "/api/v1/polymer-setup/preview",
        json={"repeat_unit_smiles": "*C[Fe]C*"},
        headers={"X-CSRF-Token": csrf},
    )
    assert unsupported.status_code == 200
    unsupported_setup = unsupported.json()["automatic_setup"]
    assert unsupported_setup["status"] == "review_required"
    assert unsupported_setup["force_field_selection"]["force_field"] is None
    assert unsupported.json()["preparation_rule"]["rule_verdict"] == "FAIL"
    assert unsupported.json()["preparation_rule"]["audit_required"] is False

    for incomplete_repeat in ("CCO", "*CCO"):
        incomplete = client.post(
            "/api/v1/polymer-setup/preview",
            json={"repeat_unit_smiles": incomplete_repeat},
            headers={"X-CSRF-Token": csrf},
        )
        assert incomplete.status_code == 200
        payload = incomplete.json()
        assert payload["valid"] is False
        assert payload["next_action"] == "review"
        assert any("exactly two '*' connection points" in item for item in payload["errors"])


def test_admin_cluster_autoconfigure_uses_detected_site_defaults(client, monkeypatch):
    csrf = login(client, "admin", "admin password 123")
    payload = {
        "name": "auto-site", "host": "cluster.example.edu", "port": 22,
        "remote_username": "admin", "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789",
        "partition": "auto", "account": None, "remote_root": "/home/admin/polymer-agent",
        "science_install_mode": "managed", "upstream_terms_accepted": True,
    }
    created = client.post("/api/v1/clusters", json=payload, headers={"X-CSRF-Token": csrf})
    cluster_id = created.json()["id"]
    monkeypatch.setattr(
        "polymer_agent_demo.control_plane.api.SSHSlurmBackend.detect_site_defaults",
        lambda self: {
            "scheduler_mode": "slurm",
            "partition": "gpu8",
            "remote_root": "/data/home/admin",
            "available_partitions": [{"name": "gpu8", "default": True, "has_gpu": True, "available": True}],
            "manual_nodes": [],
        },
    )

    configured = client.post(
        f"/api/v1/clusters/{cluster_id}/autoconfigure",
        json={},
        headers={"X-CSRF-Token": csrf},
    )

    assert configured.status_code == 200
    assert configured.json()["cluster"]["partition"] == "gpu8"
    assert configured.json()["detected"]["remote_root"] == "/data/home/admin"


def project_payload(name="trial", temperature=400):
    return {
        "name": name,
        "spec": {
            "natural_language_goal": "test PE-LCB",
            "polymer_family": "PE",
            "architecture": "pe_lcb",
            "force_field": "opls/2024/opls-aa",
            "temperature_k": temperature,
            "pressure_atm": 1,
            "chain_count": 150,
            "production_ns": 400,
            "rule_profile": "pelcb100_validated_v1",
            "candidates": [
                {"case_id": "case_001", "branch_points": 1, "branch_carbons": 5, "branch_distribution_code": "mid", "target_total_carbons": 100}
            ],
        },
    }


def prepare_ready_cluster(client: TestClient, monkeypatch) -> str:
    alice_csrf = login(client, "alice", "alice password 123")
    payload = {
        "name": "ready-lab", "host": "cluster.example.edu", "port": 22, "remote_username": "alice",
        "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu", "account": None,
        "remote_root": "/home/alice/polymer", "gromacs_command": "gmx", "emc_command": "emc",
        "python_command": "python3", "runner_command": "polymer-agent-remote", "module_init": [],
        "environment_mode": "module", "pipeline_command": "pelcb-pipeline-core",
    }
    cluster = client.post("/api/v1/clusters", json=payload, headers={"X-CSRF-Token": alice_csrf}).json()
    client.post("/api/v1/auth/logout", json={}, headers={"X-CSRF-Token": alice_csrf})
    admin_csrf = login(client, "admin", "admin password 123")
    client.post(f"/api/v1/clusters/{cluster['id']}/authorize", json={}, headers={"X-CSRF-Token": admin_csrf})
    client.post("/api/v1/auth/logout", json={}, headers={"X-CSRF-Token": admin_csrf})
    alice_csrf = login(client, "alice", "alice password 123")
    monkeypatch.setattr(
        "polymer_agent_demo.control_plane.api.SSHSlurmBackend.preflight",
        lambda self: {"ok": True, "checks": {}},
    )
    response = client.post(f"/api/v1/clusters/{cluster['id']}/preflight", json={}, headers={"X-CSRF-Token": alice_csrf})
    assert response.json()["status"] == "ready"
    return alice_csrf


def mock_gpt_design(monkeypatch, preferred_keys: list[str] | None = None) -> None:
    def fake(goal: str, count: int, api_key: str | None = None):
        keys = preferred_keys or list(design_catalog())[:count]
        keys = keys[:count]
        candidates = candidates_from_choices(
            [{"design_key": key, "rationale": f"覆盖注册坐标 {key}"} for key in keys]
        )
        return {
            "engine": "openai_responses:gpt-5.5", "model": "gpt-5.5", "response_id": "resp_test",
            "requested_count": count, "strategy_summary": "覆盖支化架构坐标", "attempt": 1,
            "catalog_size": len(design_catalog()), "catalog_sha256": catalog_sha256(), "candidates": candidates,
        }
    monkeypatch.setattr("polymer_agent_demo.control_plane.api.propose_designs", fake)


def save_model_keys(client: TestClient, csrf: str, openai_only: bool = False) -> None:
    payload = {"openai_api_key": "openai-user-test-secret"}
    if not openai_only:
        payload.update(
            {
                "deepseek_api_key": "deepseek-user-test-secret",
                "dashscope_api_key": "dashscope-user-test-secret",
                "zhipuai_api_key": "zhipu-user-test-secret",
            }
        )
    response = client.post("/api/v1/model-keys", json=payload, headers={"X-CSRF-Token": csrf})
    assert response.status_code == 200


def test_local_acceptance_is_authenticated_and_csrf_guarded(client, monkeypatch):
    csrf = login(client, "alice", "alice password 123")
    denied = client.post("/api/v1/local-acceptance", json={})
    assert denied.status_code == 403

    monkeypatch.setattr(
        "polymer_agent_demo.control_plane.api.start_local_acceptance",
        lambda owner_id, settings: {
            "job_id": "accept123",
            "status": "queued",
            "stage": "queued",
            "message": "Acceptance test queued",
        },
    )
    started = client.post("/api/v1/local-acceptance", json={}, headers={"X-CSRF-Token": csrf})
    assert started.status_code == 200
    assert started.json()["job_id"] == "accept123"


def test_model_keys_are_encrypted_and_never_returned(client):
    csrf = login(client, "alice", "alice password 123")
    payload = {
        "openai_api_key": "openai-test-secret",
        "deepseek_api_key": "deepseek-test-secret",
        "dashscope_api_key": "dashscope-test-secret",
        "zhipuai_api_key": "zhipu-test-secret",
    }
    saved = client.post("/api/v1/model-keys", json=payload, headers={"X-CSRF-Token": csrf})
    assert saved.status_code == 200
    body = saved.json()
    assert body["ready_for_closed_loop"] is True
    assert body["providers"]["openai"]["configured"] is True
    assert "openai-test-secret" not in saved.text

    listed = client.get("/api/v1/model-keys")
    assert listed.json()["openai_configured"] is True
    assert listed.json()["ready_for_design"] is True
    assert "deepseek-test-secret" not in listed.text

    cleared = client.delete("/api/v1/model-keys", headers={"X-CSRF-Token": csrf})
    assert cleared.status_code == 200
    assert cleared.json()["ready_for_design"] is False


def test_design_draft_uses_user_saved_openai_key(client, monkeypatch):
    csrf = prepare_ready_cluster(client, monkeypatch)
    client.post(
        "/api/v1/model-keys",
        json={
            "openai_api_key": "openai-user-specific-test-secret",
            "deepseek_api_key": "deepseek-user-specific-test-secret",
            "dashscope_api_key": "dashscope-user-specific-test-secret",
            "zhipuai_api_key": "zhipu-user-specific-test-secret",
        },
        headers={"X-CSRF-Token": csrf},
    )

    def fake(goal: str, count: int, api_key: str | None = None):
        assert api_key == "openai-user-specific-test-secret"
        candidates = candidates_from_choices(
            [{"design_key": key, "rationale": f"覆盖注册坐标 {key}"} for key in list(design_catalog())[:count]]
        )
        return {
            "engine": "openai_responses:gpt-5.5", "model": "gpt-5.5", "response_id": "resp_user_key",
            "requested_count": count, "strategy_summary": "使用用户保存的 GPT key 生成", "attempt": 1,
            "catalog_size": len(design_catalog()), "catalog_sha256": catalog_sha256(), "candidates": candidates,
        }

    monkeypatch.setattr("polymer_agent_demo.control_plane.api.propose_designs", fake)
    response = client.post(
        "/api/v1/design-drafts",
        json={"goal": "设计 3 个支化聚乙烯"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    assert response.json()["planner"]["response_id"] == "resp_user_key"


def test_design_draft_uses_only_planning_key_without_cluster(client, monkeypatch):
    csrf = login(client, "alice", "alice password 123")
    save_model_keys(client, csrf, openai_only=True)
    readiness = client.get("/api/v1/model-keys").json()
    assert readiness["ready_for_design"] is True
    assert readiness["ready_for_replay"] is True
    assert readiness["ready_for_local_validation"] is True
    assert readiness["ready_for_audit_panel"] is False
    assert readiness["ready_for_closed_loop"] is False
    mock_gpt_design(monkeypatch)
    response = client.post(
        "/api/v1/design-drafts",
        json={"goal": "设计 3 个支化聚乙烯"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    assert response.json()["executable"] is False


def test_auth_csrf_project_lock_and_user_isolation(client):
    csrf = login(client, "alice", "alice password 123")
    denied = client.post("/api/v1/projects", json=project_payload())
    assert denied.status_code == 403
    created = client.post("/api/v1/projects", json=project_payload(), headers={"X-CSRF-Token": csrf})
    assert created.status_code == 200
    project_id = created.json()["id"]
    validation = client.post(f"/api/v1/projects/{project_id}/validate", json={}, headers={"X-CSRF-Token": csrf})
    assert validation.json()["domain_status"] == "validated_domain"
    locked = client.post(f"/api/v1/projects/{project_id}/lock", json={}, headers={"X-CSRF-Token": csrf})
    assert locked.json()["status"] == "locked"

    client.post("/api/v1/auth/logout", json={}, headers={"X-CSRF-Token": csrf})
    login(client, "bob", "bob password 12345")
    assert client.get(f"/api/v1/projects/{project_id}").status_code == 404


def test_natural_language_only_creates_non_executable_draft(client, monkeypatch):
    csrf = prepare_ready_cluster(client, monkeypatch)
    save_model_keys(client, csrf)
    mock_gpt_design(monkeypatch)
    response = client.post(
        "/api/v1/design-drafts",
        json={"goal": "研究 C100 支化聚乙烯的密度、结构和动力学"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["executable"] is False
    assert payload["planner"]["candidate_source"] == "gpt_generated_within_registered_pe_lcb_grammar"
    assert len(payload["spec"]["candidates"]) == 100
    assert payload["spec"]["candidates"][0]["macro_smiles"].startswith("*")
    assert payload["spec"]["candidates"][0]["builder_spec"]["adapter"] == "pe_lcb"


def test_single_case_draft_requires_explicit_smoke_request(client, monkeypatch):
    csrf = prepare_ready_cluster(client, monkeypatch)
    save_model_keys(client, csrf)
    mock_gpt_design(monkeypatch, ["b3-L8-uniform"])
    response = client.post(
        "/api/v1/design-drafts",
        json={"goal": "创建单个小试 case：支化点 3、支链碳数 8、位置 uni 的 PE-LCB"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    candidate = response.json()["spec"]["candidates"][0]
    assert len(response.json()["spec"]["candidates"]) == 1
    assert candidate["branch_points"] == 3
    assert candidate["branch_carbons"] == 8
    assert candidate["branch_distribution_code"] == "uni"
    assert candidate["placement_code"] == "uniform"


def test_requested_count_produces_exactly_fifty_concrete_structures(client, monkeypatch):
    csrf = prepare_ready_cluster(client, monkeypatch)
    save_model_keys(client, csrf)
    mock_gpt_design(monkeypatch)
    response = client.post(
        "/api/v1/design-drafts",
        json={"goal": "设计 50 个约 C100 的支化聚乙烯，覆盖不同架构"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["planner"]["requested_count"] == 50
    assert len(payload["spec"]["candidates"]) == 50
    assert len({item["structure_sha256"] for item in payload["spec"]["candidates"]}) == 50
    assert all(item["branch_positions"] and item["macro_smiles"] for item in payload["spec"]["candidates"])
    created = client.post(
        "/api/v1/projects",
        json={"name": "gpt-designed-50", "spec": payload["spec"]},
        headers={"X-CSRF-Token": csrf},
    )
    assert created.status_code == 200
    validated = client.post(
        f"/api/v1/projects/{created.json()['id']}/validate",
        json={},
        headers={"X-CSRF-Token": csrf},
    )
    assert validated.json()["valid"] is True


def test_natural_language_design_requires_planning_key(client):
    csrf = login(client, "alice", "alice password 123")
    response = client.post(
        "/api/v1/design-drafts",
        json={"goal": "研究 C100 支化聚乙烯"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "model_keys_required"


def test_live_design_failure_never_falls_back_to_fixed_manifest(client, monkeypatch):
    from polymer_agent_demo.live_design import DesignPlanError

    csrf = prepare_ready_cluster(client, monkeypatch)
    save_model_keys(client, csrf)

    def fail_design(goal: str, count: int, api_key: str | None = None):
        raise DesignPlanError("model failed")

    monkeypatch.setattr("polymer_agent_demo.control_plane.api.propose_designs", fail_design)
    response = client.post(
        "/api/v1/design-drafts",
        json={"goal": "设计 50 个支化聚乙烯"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "design_model_unavailable"


def test_out_of_domain_project_enters_rule_layer_as_warn_without_early_review(client):
    csrf = login(client, "alice", "alice password 123")
    created = client.post("/api/v1/projects", json=project_payload(temperature=450), headers={"X-CSRF-Token": csrf}).json()
    result = client.post(f"/api/v1/projects/{created['id']}/validate", json={}, headers={"X-CSRF-Token": csrf}).json()
    assert result["domain_status"] == "warn"
    assert result["rule_layer_entry"] == "WARN"
    reviews = client.get("/api/v1/reviews").json()
    assert reviews == []


def test_cluster_response_never_exposes_private_key_and_requires_admin_authorization(client):
    csrf = login(client, "alice", "alice password 123")
    payload = {
        "name": "lab", "host": "cluster.example.edu", "port": 22, "remote_username": "alice",
        "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu", "account": None,
        "remote_root": "/home/alice/polymer", "gromacs_command": "gmx", "emc_command": "emc",
        "python_command": "python3", "runner_command": "polymer-agent-remote", "module_init": [],
    }
    response = client.post("/api/v1/clusters", json=payload, headers={"X-CSRF-Token": csrf})
    assert response.status_code == 200
    cluster = response.json()
    assert cluster["status"] == "unverified"
    assert "private" not in response.text.lower()
    assert cluster["public_key"].startswith("ssh-ed25519")
    preflight = client.post(f"/api/v1/clusters/{cluster['id']}/preflight", json={}, headers={"X-CSRF-Token": csrf})
    assert preflight.status_code == 409

    client.post("/api/v1/auth/logout", json={}, headers={"X-CSRF-Token": csrf})
    admin_csrf = login(client, "admin", "admin password 123")
    admin_cluster = client.post("/api/v1/clusters", json={**payload, "name": "admin-lab"}, headers={"X-CSRF-Token": admin_csrf})
    assert admin_cluster.status_code == 200
    assert admin_cluster.json()["status"] == "authorized"


def test_admin_can_authorize_cluster_but_researcher_cannot(client):
    alice_csrf = login(client, "alice", "alice password 123")
    payload = {
        "name": "lab", "host": "cluster.example.edu", "port": 22, "remote_username": "alice",
        "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu", "account": None,
        "remote_root": "/home/alice/polymer", "gromacs_command": "gmx", "emc_command": "emc",
        "python_command": "python3", "runner_command": "polymer-agent-remote", "module_init": [],
    }
    cluster = client.post("/api/v1/clusters", json=payload, headers={"X-CSRF-Token": alice_csrf}).json()
    denied = client.post(f"/api/v1/clusters/{cluster['id']}/authorize", json={}, headers={"X-CSRF-Token": alice_csrf})
    assert denied.status_code == 403

    client.post("/api/v1/auth/logout", json={}, headers={"X-CSRF-Token": alice_csrf})
    admin_csrf = login(client, "admin", "admin password 123")
    authorized = client.post(f"/api/v1/clusters/{cluster['id']}/authorize", json={}, headers={"X-CSRF-Token": admin_csrf})
    assert authorized.status_code == 200
    assert authorized.json()["status"] == "authorized"


def test_environment_discovery_and_platform_provision_unlock_ready_cluster(client, monkeypatch):
    alice_csrf = login(client, "alice", "alice password 123")
    payload = {
        "name": "provision-lab", "host": "cluster.example.edu", "port": 22, "remote_username": "alice",
        "host_key_sha256": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789", "partition": "gpu", "account": None,
        "remote_root": "/home/alice/polymer", "gromacs_command": "gmx", "emc_command": "emc",
        "python_command": "python3", "runner_command": "polymer-agent-remote", "module_init": [],
        "environment_mode": "module", "pipeline_command": "pelcb-pipeline-core",
    }
    cluster = client.post("/api/v1/clusters", json=payload, headers={"X-CSRF-Token": alice_csrf}).json()
    client.post("/api/v1/auth/logout", json={}, headers={"X-CSRF-Token": alice_csrf})
    admin_csrf = login(client, "admin", "admin password 123")
    client.post(f"/api/v1/clusters/{cluster['id']}/authorize", json={}, headers={"X-CSRF-Token": admin_csrf})
    client.post("/api/v1/auth/logout", json={}, headers={"X-CSRF-Token": admin_csrf})
    alice_csrf = login(client, "alice", "alice password 123")

    discovery = {
        "schema_version": "polymer_agent.environment_discovery.v1", "ok": False,
        "platform_installable": True, "ready_for_platform_install": True,
        "checks": {}, "admin_actions": [], "platform_actions": ["install_runner", "install_site_adapter"],
    }
    monkeypatch.setattr(
        "polymer_agent_demo.control_plane.api.SSHSlurmBackend.discover_environment",
        lambda self: discovery,
    )
    inspected = client.post(f"/api/v1/clusters/{cluster['id']}/discover", json={}, headers={"X-CSRF-Token": alice_csrf})
    assert inspected.status_code == 200
    assert inspected.json()["provisioning"]["status"] == "discovered"

    monkeypatch.setattr(
        "polymer_agent_demo.control_plane.api.provision_cluster",
        lambda backend, overwrite=False: {
            "ok": True, "installed": True, "discovery": discovery,
            "installation": {
                "installed": True, "version": "v1-test", "manifest_sha256": "a" * 64,
                "runner_command": "/home/alice/polymer/.polymer-agent/bin/polymer-agent-remote",
            },
        },
    )
    monkeypatch.setattr(
        "polymer_agent_demo.control_plane.api.SSHSlurmBackend.preflight",
        lambda self: {"ok": True, "checks": {}},
    )
    installed = client.post(
        f"/api/v1/clusters/{cluster['id']}/provision",
        json={"confirmation": "install", "overwrite": False},
        headers={"X-CSRF-Token": alice_csrf},
    )
    assert installed.status_code == 200
    assert installed.json()["status"] == "ready"
    assert installed.json()["provisioning"]["status"] == "installed"

    setup = client.post(f"/api/v1/clusters/{cluster['id']}/setup", json={}, headers={"X-CSRF-Token": alice_csrf})
    assert setup.status_code == 200
    assert setup.json()["status"] == "ready"
    assert setup.json()["provisioning"]["auto_setup"] is True


def test_real_execution_is_disabled_without_deployment_opt_in(client):
    health = client.get("/api/v1/health").json()
    assert health["executor_enabled"] is False


def test_project_release_list_and_download_are_owner_scoped(client):
    csrf = login(client, "alice", "alice password 123")
    created = client.post("/api/v1/projects", json=project_payload(), headers={"X-CSRF-Token": csrf}).json()
    project_id = created["id"]
    client.post(f"/api/v1/projects/{project_id}/validate", json={}, headers={"X-CSRF-Token": csrf})
    client.post(f"/api/v1/projects/{project_id}/lock", json={}, headers={"X-CSRF-Token": csrf})

    release = client.post(f"/api/v1/projects/{project_id}/releases", json={}, headers={"X-CSRF-Token": csrf})
    assert release.status_code == 200
    release_id = release.json()["id"]

    releases = client.get(f"/api/v1/projects/{project_id}/releases").json()
    assert len(releases) == 1
    assert releases[0]["id"] == release_id
    assert releases[0]["version"] == 1
    assert len(releases[0]["sha256"]) == 64
    assert "created_at" in releases[0]

    download = client.get(f"/api/v1/releases/{release_id}/download")
    assert download.status_code == 200
    assert download.headers["content-disposition"].endswith('pelcb-release-v1.json"')

    readable = client.get(f"/api/v1/releases/{release_id}/download-result")
    assert readable.status_code == 200
    assert "结果" in readable.text
    assert "物性可用性汇总" in readable.text
    assert readable.headers["content-disposition"].endswith('pelcb-release-v1-result.md"')

    client.post("/api/v1/auth/logout", json={}, headers={"X-CSRF-Token": csrf})
    login(client, "bob", "bob password 12345")
    assert client.get(f"/api/v1/projects/{project_id}/releases").status_code == 404
    assert client.get(f"/api/v1/releases/{release_id}/download").status_code == 404
    assert client.get(f"/api/v1/releases/{release_id}/download-result").status_code == 404
