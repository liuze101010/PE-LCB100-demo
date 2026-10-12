from pathlib import Path

from fastapi.testclient import TestClient

from polymer_agent_demo.app import create_app
from polymer_agent_demo.snapshot import build_demo_snapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (REPOSITORY_ROOT, *REPOSITORY_ROOT.parents)
        if (candidate / "generated_code" / "md_analysis").is_dir()
    ),
    REPOSITORY_ROOT,
)


def make_client(tmp_path):
    snapshot_path = tmp_path / "demo_snapshot.json"
    build_demo_snapshot(PROJECT_ROOT, snapshot_path)
    app = create_app(
        snapshot_path=snapshot_path,
        project_root=PROJECT_ROOT,
        force_mode="replay",
    )
    return TestClient(app)


def test_api_exposes_health_overview_and_cases(tmp_path):
    client = make_client(tmp_path)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["mode"] == "replay"
    assert health.json()["read_only"] is True

    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert overview.json()["cohort"]["audited_cases"] == 97
    assert overview.json()["cohort"]["model_calls"] == 291
    assert overview.json()["cohort"]["publication_pass"] == 92
    assert overview.json()["cohort"]["publication_review"] == 8

    cases = client.get("/api/cases")
    assert cases.status_code == 200
    assert len(cases.json()) == 3

    audit = client.get("/api/audit-ledger")
    assert audit.status_code == 200
    assert audit.json()["case_count"] == 97
    assert audit.json()["model_vote_count"] == 291


def test_case_detail_and_dry_run_are_read_only(tmp_path):
    client = make_client(tmp_path)
    case_id = "lcb_pe_002_bp01_bc008_pdmid_c100"

    detail = client.get(f"/api/cases/{case_id}")
    assert detail.status_code == 200
    assert detail.json()["route"]["module"] == "postprocess"

    options = client.get(f"/api/cases/{case_id}/route-options")
    assert options.status_code == 200
    assert options.json()["planning_selection_status"] == "awaiting_planning_model"
    assert "run_postprocess_repair" in {
        item["tool"] for item in options.json()["allowed_routes"]
    }

    selected = client.post(
        f"/api/cases/{case_id}/route-selection",
        json={
            "repair_tool": "run_postprocess_repair",
            "rationale": "The locked audit attributes this warning to the MSD fit window.",
        },
    )
    assert selected.status_code == 200
    assert selected.json()["planning_selection_status"] == "selected"
    assert selected.json()["module"] == "postprocess"

    dry_run = client.post(f"/api/cases/{case_id}/dry-run", json={})
    assert dry_run.status_code == 200
    payload = dry_run.json()
    assert payload["dry_run"] is True
    assert payload["tool"] == "run_postprocess_repair"
    assert payload["guard"]["verdict"] == "pass"

    rejected = client.post(f"/api/cases/{case_id}/dry-run", json={"execute": True})
    assert rejected.status_code == 403
    assert rejected.json()["detail"]["code"] == "mutating_request_rejected"


def test_unknown_case_is_not_exposed(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/api/cases/unknown").status_code == 404
    assert client.post("/api/cases/unknown/dry-run", json={}).status_code == 404
