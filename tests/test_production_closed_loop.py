from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from polymer_agent_demo.control_plane.design_grammar import catalog_sha256
from polymer_agent_demo.control_plane.production_evaluation import evaluate_case
from polymer_agent_demo.control_plane.schemas import DesignPlanningProvenance, ProjectSpec
from polymer_agent_demo.control_plane.science import PELCBAdapter, compare_release_with_frozen_oracle
from polymer_agent_demo.control_plane.slurm import ClusterRuntime, SSHSlurmBackend
from polymer_agent_demo.deploy.production_case import (
    emc_subprocess_env,
    missing_emc_artifacts,
    normalized_builder_spec,
)
from polymer_agent_demo.live_design import benchmark_candidates, benchmark_goal
from polymer_agent_demo.live_execution_plan import guard_upper_repair_plan


def passing_raw_evidence(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "density": {"status_density": "OK", "density_g_cm3": 0.8, "density_frame_count": 100},
        "rg_summary": {"status": "ok", "rg_mean_nm": 1.0, "rg2_mean_nm2": 1.0},
        "msd_summary": {
            "status": "ok", "D_mean_m2_s": 1e-11, "D_outlier_fraction": 0.1,
            "diffusive_regime_ok": True, "fit_quality": "ok", "fit_begin_ps": 100000,
            "fit_end_ps": 300000, "time_first_ps": 0, "time_last_ps": 400000,
            "time_final_ps": 400000, "loglog_slope": 1.0, "python_fit": {"R2": 0.99},
            "window_D_cv_pct": 2.0, "window_sensitive": False,
        },
        "rdf_summary": {"entries": [{"rdf_kind": kind, "first_peak_g": 1.2} for kind in ("all", "excl", "inter_chain_cc", "intra_chain_nonbonded_cc")]},
        "shape_summary": {"status": "ok", "rg2_carbon_mean_nm2": 1.0, "asphericity_mean_nm2": 0.2, "acylindricity_mean_nm2": 0.1, "relative_shape_anisotropy_mean": 0.2},
        "dynamics_summary": {"status": "ok", "tau_alpha_ps": 50.0, "tau_alpha_censored": False},
        "contact_summary": {"status": "ok", "P_contact_rc0p8_s10": 0.1, "P_contact_rc0p8_s20": 0.05, "P_contact_rc0p8_s50": 0.01},
        "intrachain_summary": {"g3_alpha_100_300": 1.0, "time_last_ps": 400000, "g1_monomer_over_g3_10ns": 2.0, "g3_error": ""},
        "runtime": {"engine_version": ["GROMACS version: 2024.4"], "gromacs_finished": True, "fatal_log_marker_count": 0},
    }


def test_locked_benchmark_mode_materializes_exact_100_case_manifest():
    candidates, digest = benchmark_candidates()
    assert benchmark_goal("\u590d\u73b0 PE-LCB100 \u8bba\u6587\u57fa\u51c6")
    assert len(candidates) == 100
    assert candidates[0]["case_id"] == "lcb_pe_001_bp01_bc005_pdmid_c101"
    assert candidates[-1]["case_id"] == "lcb_pe_100_bp05_bc008_pdgrad_c100"
    spec = ProjectSpec(
        natural_language_goal="\u590d\u73b0 PE-LCB100 \u8bba\u6587\u57fa\u51c6",
        planning=DesignPlanningProvenance(
            engine="test", model="test", response_id="test", requested_count=100,
            catalog_size=100, catalog_sha256=catalog_sha256(), strategy_summary="locked benchmark",
            attempt=1, mode="reproduce_pelcb100", benchmark_manifest_sha256=digest,
        ),
        candidates=candidates,
    )
    result = PELCBAdapter().validate(spec)
    assert result.valid and result.domain_status == "validated_domain"
    assert [row["case_id"] for row in result.manifest] == [row["case_id"] for row in candidates]


def test_frozen_oracle_reports_92_pass_8_review_without_driving_decisions():
    manifest = json.loads((ROOT / "data" / "pelcb100_design_manifest.json").read_text(encoding="utf-8"))["candidates"]
    oracle = json.loads((ROOT / "data" / "pelcb100_release_oracle.json").read_text(encoding="utf-8"))
    review = set(oracle["review_case_ids"])
    cases = [{"case_id": row["case_id"], "quality_status": "review" if row["case_id"] in review else "pass"} for row in manifest]
    report = compare_release_with_frozen_oracle(cases)
    assert report["exact_match"] is True
    assert report["actual_counts"] == {"pass": 92, "review": 8}
    cases[0]["quality_status"] = "pass"
    assert compare_release_with_frozen_oracle(cases)["exact_match"] is False
    production_sources = [
        ROOT / "deploy" / "production_case.py",
        ROOT / "control_plane" / "production_evaluation.py",
        ROOT / "control_plane" / "production_audit.py",
        ROOT / "control_plane" / "reconciler.py",
    ]
    for source in production_sources:
        text = source.read_text(encoding="utf-8")
        assert not any(case_id in text for case_id in review)


def test_rule_evaluation_uses_science_core_and_detects_msd_warn():
    case = {"case_id": "case_001", "force_field": "opls/2024/opls-aa", "temperature_k": 400, "pressure_atm": 1}
    with tempfile.TemporaryDirectory() as directory:
        result = evaluate_case(passing_raw_evidence(case["case_id"]), case, Path(directory))
        assert result["rule_verdict"] == "PASS"
        raw = passing_raw_evidence(case["case_id"])
        raw["msd_summary"]["loglog_slope"] = 0.2
        warned = evaluate_case(raw, case, Path(directory))
        assert warned["rule_verdict"] == "WARN"
        assert "msd_warn" in warned["warn_sources"]


def test_supported_force_field_without_external_qualification_enters_audit_as_warn():
    case = {
        "case_id": "oxygenated_polymer_probe",
        "force_field": "opls/2024/opls-aa",
        "temperature_k": 400,
        "pressure_atm": 1,
        "builder_spec": {
            "automatic_setup": {
                "force_field_selection": {
                    "scientific_qualification_status": "external_reference_qualification_required",
                }
            }
        },
    }

    with tempfile.TemporaryDirectory() as directory:
        result = evaluate_case(passing_raw_evidence(case["case_id"]), case, Path(directory))

    assert result["rule_verdict"] == "WARN"
    assert "force_field_qualification_warn" in result["warn_sources"]
    assert result["generic_qc"]["audit_required"] is True


def test_rule_layer_returns_fail_for_explicitly_unsupported_preparation():
    case = {
        "case_id": "unsupported_metal_polymer_probe",
        "temperature_k": 400,
        "pressure_atm": 1,
        "automatic_setup": {
            "status": "review_required",
            "builder_selection": {"builder_backend": "emc_alpha_olefin_builder"},
            "force_field_selection": {"force_field": None},
            "preparation_check": {
                "force_field_supported": False,
                "builder_backend_supported": True,
                "topology_supported": False,
                "source": "molecule_preparation_preflight",
            },
        },
    }

    with tempfile.TemporaryDirectory() as directory:
        result = evaluate_case(passing_raw_evidence(case["case_id"]), case, Path(directory))

    assert result["rule_verdict"] == "FAIL"
    assert result["generic_qc"]["audit_required"] is False
    assert result["generic_qc"]["terminal_reason"] == "unsupported_by_current_system"
    assert result["generic_qc"]["failure_sources"] == ["unsupported_configuration_fail"]
    assert result["evidence"]["support_assessment_source"] == "molecule_preparation_preflight"


def test_production_builder_preserves_automatically_selected_generic_polymer_setup():
    case = {
        "case_id": "oxygenated_polymer_probe",
        "builder_spec": {
            "case_id": "oxygenated_polymer_probe",
            "architecture": "linear_homopolymer",
            "adapter_id": "linear_homopolymer",
            "polymer_family": "custom_polymer",
            "builder_backend": "emc_alpha_olefin_builder",
            "force_field": "opls-aa",
            "repeat_unit_smiles": "*CCO*",
            "monomer_smiles": "*CCO*",
            "repeat_unit_total_carbons": 2,
            "repeat_units_per_chain": 50,
            "target_total_carbons_per_chain": 100,
            "actual_total_carbons_per_chain": 100,
            "parameterization_probe_required": True,
        },
    }

    normalized = normalized_builder_spec(case, {})

    assert normalized["architecture"] == "linear_homopolymer"
    assert normalized["adapter_id"] == "linear_homopolymer"
    assert normalized["repeat_unit_smiles"] == "*CCO*"
    assert normalized["force_field"] == "opls-aa"
    assert normalized["parameterization_probe_required"] is True
    assert emc_subprocess_env(Path("/opt/emc"))["EMC_ROOT"] == str(Path("/opt/emc"))
    with tempfile.TemporaryDirectory() as directory:
        assert missing_emc_artifacts(Path(directory), "oxygenated_polymer_probe") == [
            ".psf.gz",
            ".pdb.gz",
            ".params",
        ]


def test_slurm_array_and_upper_plan_guard_are_bounded():
    runtime = ClusterRuntime(
        host="cluster.example.edu", port=22, username="alice", host_key_sha256="SHA256:example",
        private_key_pem="unused", partition="gpu", account=None, remote_root="/home/alice/work",
        gromacs_command="gmx", emc_command="emc", python_command="python3",
        runner_command="polymer-agent-remote", module_init=[],
        limits={"max_cpus": 16, "max_gpus": 1, "max_walltime_hours": 200, "allowed_nodes": []},
        environment={"mode": "module"},
    )
    script = SSHSlurmBackend(runtime).render_array_script(
        run_id="abcdef012345", stage="md", command=["polymer-agent-remote", "stage", "--stage", "md"],
        case_count=100, max_parallel=8, cpus=8, gpus=1, walltime_hours=100,
    )
    assert "#SBATCH --array=0-99%8" in script
    assert "slurm-%A_%a.out" in script
    proposed = {"plan": {"items": [{"case_ids": ["a"], "proposed_contract": "clean_msd_bounded_extension"}, {"case_ids": ["b"], "proposed_contract": "clean_msd_bounded_extension"}]}}
    guarded = guard_upper_repair_plan(proposed, {"a": {"clean_msd_bounded_extension", "review_hold"}, "b": {"review_hold"}})
    assert guarded["accepted"]["a"] == "clean_msd_bounded_extension"
    assert guarded["accepted"]["b"] == "review_hold"
    assert any(item["reason"] == "contract_not_registered_for_case" for item in guarded["rejected"])
