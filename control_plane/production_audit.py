from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import sanitize_audit_payload
from .config import model_configuration
from .science_core_runtime import analysis_path


@dataclass
class RecordingClient:
    client: Any
    raw: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.client.name)

    def generate(self, prompt: str) -> str:
        response = self.client.generate(prompt)
        self.raw.append(response)
        return response


def allowed_contracts(evaluation: dict[str, Any], diagnosis: dict[str, Any], approval: dict[str, Any]) -> set[str]:
    independence = dict(diagnosis.get("initial_prompt_independence") or {})
    if (
        not diagnosis.get("model_panel_complete")
        or diagnosis.get("diagnosis_source") != "tri_model_llm"
        or not independence.get("prompt_independence_pass")
        or diagnosis.get("cross_model_disagreement")
    ):
        return {"review_hold"}
    sources = set(evaluation.get("warn_sources") or [])
    contracts: set[str] = {"review_hold"}
    if "msd_warn" in sources and int(approval.get("max_additional_md_ns", 0)) >= 400:
        contracts.add("clean_msd_bounded_extension")
    if "evidence_incomplete_warn" in sources:
        contracts.update({"missing_output_backfill", "postprocess_recompute"})
    if sources & {"rdf_warn", "shape_warn", "contact_warn", "fsqt_warn", "rg_warn"}:
        contracts.add("postprocess_recompute")
    if "density_warn" in sources:
        contracts.add("density_clean_profile_regate")
    tool = str(diagnosis.get("repair_tool") or "")
    tool_map = {
        "extend_clean_nvt_prod": "clean_msd_bounded_extension",
        "run_postprocess_repair": "postprocess_recompute",
        "validate_or_build_emc_input": "missing_output_backfill",
        "submit_md_clean_engine": "density_clean_profile_regate",
        "open_review_packet": "review_hold",
    }
    if tool in tool_map:
        contracts.add(tool_map[tool])
    return contracts


def run_audit_panel(
    evaluation: dict[str, Any],
    *,
    deepseek_key: str,
    dashscope_key: str,
    zhipuai_key: str,
    cache_dir: Path,
    approval: dict[str, Any],
) -> dict[str, Any]:
    if not all(str(key or "").strip() for key in (deepseek_key, dashscope_key, zhipuai_key)):
        raise RuntimeError("the audit stage requires DeepSeek, Qwen and GLM credentials")
    analysis_path()
    from agentic.tri_llm_judge import DeepSeekClient, GLMClient, QwenClient  # type: ignore
    from pelcb_warn_audit_llm import audit_warn_case  # type: ignore

    models = model_configuration()
    wrappers = [
        RecordingClient(DeepSeekClient(api_key=deepseek_key, model=models["deepseek"], name=models["deepseek"], timeout_s=240, max_tokens=6000)),
        RecordingClient(QwenClient(api_key=dashscope_key, model=models["dashscope"], name=models["dashscope"], timeout_s=180, max_tokens=3000)),
        RecordingClient(GLMClient(api_key=zhipuai_key, model=models["zhipuai"], name=models["zhipuai"], timeout_s=180, max_tokens=3000)),
    ]
    row = dict(evaluation.get("evidence") or {})
    row.update(
        {
            "rule_verdict": evaluation.get("rule_verdict"),
            "warn_sources": list(evaluation.get("warn_sources") or []),
            "generic_qc_verdict": (evaluation.get("generic_qc") or {}).get("rule_verdict"),
            "density_rule_verdict": (evaluation.get("density_rule") or {}).get("rule_verdict"),
        }
    )
    row = sanitize_audit_payload(row)
    diagnosis = audit_warn_case(
        row,
        clients=wrappers,
        cache_dir=cache_dir,
        audit_strategy="tri-model-majority",
        retries=1,
    )
    raw_records = []
    for wrapper in wrappers:
        for response in wrapper.raw:
            raw_records.append(
                {
                    "model": wrapper.name,
                    "response": response,
                    "sha256": hashlib.sha256(response.encode()).hexdigest(),
                }
            )
    return {
        "case_id": evaluation["case_id"],
        "diagnosis": diagnosis,
        "raw_responses": raw_records,
        "allowed_contracts": sorted(allowed_contracts(evaluation, diagnosis, approval)),
    }
