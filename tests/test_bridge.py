from pathlib import Path

import pytest

from polymer_agent_demo.mcp_bridge import DemoService
from polymer_agent_demo.snapshot import SOURCE_PATHS, build_demo_snapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (REPOSITORY_ROOT, *REPOSITORY_ROOT.parents)
        if (candidate / "generated_code" / "md_analysis").is_dir()
    ),
    REPOSITORY_ROOT,
)


def test_connected_and_replay_routes_share_public_contract(tmp_path):
    snapshot_path = tmp_path / "demo_snapshot.json"
    build_demo_snapshot(PROJECT_ROOT, snapshot_path)

    replay = DemoService(snapshot_path, PROJECT_ROOT, force_mode="replay")
    connected_root = PROJECT_ROOT
    if not (connected_root / SOURCE_PATHS["mcp_server"]).exists():
        connected_root = tmp_path / "standalone_connected_fixture"
        mcp_path = connected_root / SOURCE_PATHS["mcp_server"]
        mcp_path.parent.mkdir(parents=True)
        mcp_path.write_text(
            """import json
def plan_repair_route_impl(case_id, planning_proposal_json=''):
    options = [
        {'diagnosis': 'engine_or_protocol_sensitive_result',
         'repair_tool': 'submit_md_clean_engine',
         'tool_parameters': {'engine_profile': 'gromacs_2024_4_clean', 'reentry_stage': 'md', 'rerun_scope': 'same_input_clean_engine_validation'},
         'repair_route': 'optional_clean_engine_validation',
         'recommended_reentry_module': 'md'}
    ]
    base = {'status': 'ok', 'dry_run': True, 'case_id': case_id, 'allowed_routes': options,
            'execution_guard': {'mutating_tools_require': ['execute=true', 'project_id', 'reason', 'allowlist_tag']}}
    if not planning_proposal_json:
        return {**base, 'planning_selection_status': 'awaiting_planning_model'}
    proposal = json.loads(planning_proposal_json)
    return {**base, 'planning_selection_status': 'selected', 'route_selected_by': 'planning_model',
            'planning_selection_rationale': proposal['rationale'], 'planning_guard': {'guard_verdict': 'pass'},
            'recommended_reentry_module': 'md', 'recommended_action': 'recompute',
            'repair_tool': 'submit_md_clean_engine', 'tool_parameters': options[0]['tool_parameters'],
            'repair_route': 'optional_clean_engine_validation'}
""",
            encoding="utf-8",
        )
    connected = DemoService(snapshot_path, connected_root, force_mode="connected")
    case_id = "lcb_pe_006_bp01_bc008_pdend_c100"

    replay_route = replay.dry_run(case_id)
    connected_route = connected.dry_run(case_id)

    for key in ("case_id", "dry_run", "module", "tool", "parameters"):
        assert connected_route[key] == replay_route[key]
    assert connected_route["mode"] == "connected"
    assert replay_route["mode"] == "replay"
    assert connected.route_options(case_id)["planning_selection_status"] == "awaiting_planning_model"
    assert connected_route["planning_selection_status"] == "selected"


def test_service_rejects_mutation(tmp_path):
    snapshot_path = tmp_path / "demo_snapshot.json"
    build_demo_snapshot(PROJECT_ROOT, snapshot_path)
    service = DemoService(snapshot_path, PROJECT_ROOT, force_mode="replay")

    with pytest.raises(PermissionError):
        service.dry_run(
            "lcb_pe_006_bp01_bc008_pdend_c100",
            execute=True,
        )
