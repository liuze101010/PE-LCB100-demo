#!/usr/bin/env python3
"""Registered EMC builder adapters for polymer-MD sidecar workflows.

The registry converts validated `builder_spec` dictionaries into EMC case
artifacts.  It intentionally supports only registered architectures; unsupported
families must add an adapter instead of being guessed from arbitrary SMILES.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
MD_ANALYSIS_DIR = SCRIPT_PATH.parent
if str(MD_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(MD_ANALYSIS_DIR))

from plugin_manifest_loader import load_symbol_from_spec  # noqa: E402
from polymer_template_builder import render_template_esh  # noqa: E402


GENERATED_CODE_DIR = SCRIPT_PATH.parents[1]
EMC_BUILDER_DIR = GENERATED_CODE_DIR / "emc_alpha_olefin_builder"
if str(EMC_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(EMC_BUILDER_DIR))

from build_alpha_olefin_library import CaseMetadata, render_esh  # type: ignore  # noqa: E402


PLUGIN_DIR = SCRIPT_PATH.parent / "plugins"
BUILTIN_EMC_ADAPTER_MANIFEST = PLUGIN_DIR / "builtin_emc_builder_adapters.json"


class EmcBuilderAdapter:
    adapter_id: str = ""
    architectures: frozenset[str] = frozenset()

    def to_case_metadata(self, builder_spec: dict[str, Any]) -> CaseMetadata:
        raise NotImplementedError


_EMC_ADAPTERS_BY_ID: dict[str, EmcBuilderAdapter] = {}
_EMC_ADAPTERS_BY_ARCHITECTURE: dict[str, EmcBuilderAdapter] = {}


def register_emc_builder_adapter(adapter: EmcBuilderAdapter, *, replace: bool = False) -> EmcBuilderAdapter:
    adapter_id = str(adapter.adapter_id or "").strip()
    if not adapter_id:
        raise ValueError("adapter_id is required")
    if not adapter.architectures:
        raise ValueError(f"adapter {adapter_id} must declare at least one architecture")
    if adapter_id in _EMC_ADAPTERS_BY_ID and not replace:
        raise ValueError(f"EMC builder adapter already registered: {adapter_id}")
    for architecture in adapter.architectures:
        owner = _EMC_ADAPTERS_BY_ARCHITECTURE.get(str(architecture))
        if owner is not None and owner.adapter_id != adapter_id and not replace:
            raise ValueError(f"EMC architecture already registered: {architecture}")
    if replace:
        unregister_emc_builder_adapter(adapter_id)
        for architecture, owner in list(_EMC_ADAPTERS_BY_ARCHITECTURE.items()):
            if architecture in adapter.architectures or owner.adapter_id == adapter_id:
                _EMC_ADAPTERS_BY_ARCHITECTURE.pop(architecture, None)
    _EMC_ADAPTERS_BY_ID[adapter_id] = adapter
    for architecture in adapter.architectures:
        _EMC_ADAPTERS_BY_ARCHITECTURE[str(architecture)] = adapter
    return adapter


def unregister_emc_builder_adapter(adapter_id: str) -> None:
    adapter = _EMC_ADAPTERS_BY_ID.pop(str(adapter_id), None)
    if adapter is None:
        return
    for architecture, owner in list(_EMC_ADAPTERS_BY_ARCHITECTURE.items()):
        if owner.adapter_id == adapter.adapter_id:
            _EMC_ADAPTERS_BY_ARCHITECTURE.pop(architecture, None)


def get_emc_builder_adapter(architecture: str) -> EmcBuilderAdapter:
    adapter = _EMC_ADAPTERS_BY_ARCHITECTURE.get(str(architecture or ""))
    if adapter is None:
        raise ValueError(f"Unsupported EMC architecture: {architecture}")
    return adapter


def list_emc_builder_adapters() -> dict[str, dict[str, Any]]:
    return {
        adapter_id: {
            "adapter_id": adapter.adapter_id,
            "architectures": sorted(adapter.architectures),
        }
        for adapter_id, adapter in sorted(_EMC_ADAPTERS_BY_ID.items())
    }


def load_emc_builder_plugins(paths: list[str | Path] | tuple[str | Path, ...]) -> list[str]:
    loaded: list[str] = []
    for raw_path in paths:
        manifest_path = Path(raw_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for spec in manifest.get("emc_builder_adapters", []) or []:
            adapter_cls = load_symbol_from_spec(dict(spec), base_dir=manifest_path.parent)
            adapter = adapter_cls()
            register_emc_builder_adapter(adapter, replace=bool(spec.get("replace", True)))
            loaded.append(adapter.adapter_id)
    return loaded


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def emc_force_field(raw: str | None) -> str:
    text = str(raw or "").strip().lower()
    if text in {"opls-aa", "oplsaa", "opls"}:
        return "opls/2024/opls-aa"
    return str(raw or "opls/2024/opls-aa")


def _case_metadata_from_registered_architecture(builder_spec: dict[str, Any]) -> CaseMetadata:
    architecture = str(builder_spec.get("architecture") or "")

    target_total = int(builder_spec.get("target_total_carbons_per_chain") or 100)
    actual_total = int(builder_spec.get("actual_total_carbons_per_chain") or target_total)
    chain_count = int(builder_spec.get("chain_count") or 12)
    sidechain_carbons = int(builder_spec.get("sidechain_carbons") or builder_spec.get("branch_carbons") or 0)
    repeat_unit_total = int(builder_spec.get("repeat_unit_total_carbons") or max(1, sidechain_carbons + 2))
    repeats = int(
        builder_spec.get("repeat_units_per_chain")
        or (1 if architecture == "pe_lcb" else max(1, round(actual_total / repeat_unit_total)))
    )
    monomer_total = actual_total if architecture == "pe_lcb" else repeat_unit_total

    if architecture == "pe_lcb":
        branch_points = int(builder_spec.get("branch_points") or 0)
        branch_carbons = int(builder_spec.get("branch_carbons") or 0)
        monomer_name = f"MCP sidecar PE-LCB bp={branch_points} bc={branch_carbons}"
        tacticity = "achiral polyethylene backbone; registered PE-LCB adapter"
        case_architecture = "lcb"
    elif architecture == "linear_alpha_olefin":
        branch_points = None
        branch_carbons = None
        monomer_name = f"MCP sidecar linear alpha-olefin C{sidechain_carbons + 2}"
        tacticity = "isotactic (fixed [C@H] repeat stereocenter); registered alpha-olefin adapter"
        case_architecture = "linear"
    elif architecture == "explicit_polymer_chain":
        branch_points = None
        branch_carbons = None
        monomer_name = f"MCP explicit polymer chain {builder_spec.get('polymer_family') or 'polymer'}"
        tacticity = "as provided by the explicit full-chain molecular graph"
        case_architecture = "generic"
    elif architecture == "templated_polymer_chain":
        branch_points = None
        branch_carbons = None
        monomer_name = f"MCP template-built {builder_spec.get('polymer_family') or 'polymer'} chain"
        tacticity = "as specified by the registered repeat-unit templates"
        case_architecture = "generic"
    else:
        branch_points = None
        branch_carbons = None
        monomer_name = f"MCP sidecar linear homopolymer {builder_spec.get('polymer_family') or 'polymer'}"
        tacticity = "as provided by repeat_unit_smiles; registered linear-homopolymer adapter"
        case_architecture = "linear"

    return CaseMetadata(
        case_id=str(builder_spec["case_id"]),
        preset=str(builder_spec.get("preset") or "mcp_generic_emc_smoke"),
        sidechain_carbons=sidechain_carbons,
        repeat_unit_total_carbons=repeat_unit_total,
        monomer_total_carbons=monomer_total,
        monomer_name=monomer_name,
        chain_count=chain_count,
        target_total_carbons_per_chain=target_total,
        repeat_units_per_chain=repeats,
        actual_total_carbons_per_chain=actual_total,
        carbon_delta=actual_total - target_total,
        temperature_k=float(builder_spec.get("temperature_k") or 400.0),
        initial_density_g_per_cc=float(builder_spec.get("initial_density_g_per_cc") or 0.85),
        pressure_atm=float(builder_spec.get("pressure_atm") or 1.0),
        force_field=emc_force_field(str(builder_spec.get("force_field") or "")),
        monomer_smiles=str(builder_spec["monomer_smiles"]),
        tacticity=tacticity,
        literature_anchor={
            "primary": "MCP registered polymer-adapter EMC smoke test",
            "reason": "This sidecar case verifies that a registered validator plugin can materialize EMC build inputs.",
        },
        architecture=case_architecture,
        backbone_monomer=builder_spec.get("backbone_monomer"),
        backbone_repeat_units=builder_spec.get("backbone_repeat_units"),
        branch_points=branch_points,
        branch_carbons=branch_carbons,
        branch_topology=str(builder_spec.get("branch_position_pattern") or "sidecar") if architecture == "pe_lcb" else None,
        branch_position_pattern=builder_spec.get("branch_position_pattern"),
        branch_positions=builder_spec.get("branch_positions"),
        branch_smiles=builder_spec.get("branch_smiles"),
        regime_tag="mcp-smoke",
        ml_feature_coord=str(builder_spec.get("case_id") or ""),
        polymer_family=str(builder_spec.get("polymer_family") or ""),
        generic_case_name=str(builder_spec.get("case_id") or ""),
    )


class PeLcbEmcAdapter(EmcBuilderAdapter):
    adapter_id = "pe_lcb"
    architectures = frozenset({"pe_lcb"})

    def to_case_metadata(self, builder_spec: dict[str, Any]) -> CaseMetadata:
        return _case_metadata_from_registered_architecture(builder_spec)


class LinearAlphaOlefinEmcAdapter(EmcBuilderAdapter):
    adapter_id = "linear_alpha_olefin"
    architectures = frozenset({"linear_alpha_olefin"})

    def to_case_metadata(self, builder_spec: dict[str, Any]) -> CaseMetadata:
        return _case_metadata_from_registered_architecture(builder_spec)


class LinearHomopolymerEmcAdapter(EmcBuilderAdapter):
    adapter_id = "linear_homopolymer"
    architectures = frozenset({"linear_homopolymer"})

    def to_case_metadata(self, builder_spec: dict[str, Any]) -> CaseMetadata:
        return _case_metadata_from_registered_architecture(builder_spec)


class ExplicitPolymerChainEmcAdapter(EmcBuilderAdapter):
    adapter_id = "explicit_polymer_chain"
    architectures = frozenset({"explicit_polymer_chain"})

    def to_case_metadata(self, builder_spec: dict[str, Any]) -> CaseMetadata:
        return _case_metadata_from_registered_architecture(builder_spec)


class TemplatedPolymerChainEmcAdapter(EmcBuilderAdapter):
    adapter_id = "templated_polymer_chain"
    architectures = frozenset({"templated_polymer_chain"})

    def to_case_metadata(self, builder_spec: dict[str, Any]) -> CaseMetadata:
        return _case_metadata_from_registered_architecture(builder_spec)


def _register_builtin_emc_adapters() -> None:
    if BUILTIN_EMC_ADAPTER_MANIFEST.exists():
        load_emc_builder_plugins([BUILTIN_EMC_ADAPTER_MANIFEST])
        return
    for adapter in (
        PeLcbEmcAdapter(),
        LinearAlphaOlefinEmcAdapter(),
        LinearHomopolymerEmcAdapter(),
        ExplicitPolymerChainEmcAdapter(),
        TemplatedPolymerChainEmcAdapter(),
    ):
        register_emc_builder_adapter(adapter, replace=True)


_register_builtin_emc_adapters()


def builder_spec_to_case_metadata(builder_spec: dict[str, Any]) -> CaseMetadata:
    adapter = get_emc_builder_adapter(str(builder_spec.get("architecture") or ""))
    return adapter.to_case_metadata(builder_spec)


def write_emc_case(
    case_dir: Path,
    builder_spec: dict[str, Any],
    validation: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    architecture = str(builder_spec.get("architecture") or "")
    materialized_structure: dict[str, Any] = {}
    if architecture == "templated_polymer_chain":
        template_plan = builder_spec.get("template_plan")
        if not isinstance(template_plan, dict):
            raise ValueError("templated_polymer_chain builder_spec requires template_plan")
        rendered = render_template_esh(template_plan, builder_spec)
        materialized_structure = dict(rendered["materialized_structure"])
        case_id = str(builder_spec["case_id"])
        esh_path = case_dir / f"{case_id}.esh"
        case_json_path = case_dir / "case.json"
        esh_path.write_text(str(rendered["esh"]), encoding="ascii")
        payload = {
            "case_id": case_id,
            "preset": builder_spec.get("preset"),
            "architecture": architecture,
            "polymer_family": builder_spec.get("polymer_family"),
            "chain_count": builder_spec.get("chain_count"),
            "target_total_carbons_per_chain": builder_spec.get(
                "target_total_carbons_per_chain"
            ),
            "actual_total_carbons_per_chain": builder_spec.get(
                "actual_total_carbons_per_chain"
            ),
            "repeat_units_per_chain": builder_spec.get("repeat_units_per_chain"),
            "force_field": builder_spec.get("force_field"),
            "mcp_builder_spec": builder_spec,
            "mcp_materialized_structure": materialized_structure,
            "mcp_validation": validation or {},
            "mcp_plan_md_package": plan or {},
        }
        write_json(case_json_path, payload)
        return {
            "case_dir": str(case_dir),
            "case_json": str(case_json_path),
            "esh": str(esh_path),
        }

    materialized_spec = dict(builder_spec)
    meta = builder_spec_to_case_metadata(materialized_spec)
    esh_path = case_dir / f"{meta.case_id}.esh"
    case_json_path = case_dir / "case.json"
    esh_path.write_text(render_esh(meta), encoding="ascii")
    payload = asdict(meta)
    payload["mcp_builder_spec"] = builder_spec
    payload["mcp_materialized_structure"] = materialized_structure
    payload["mcp_validation"] = validation or {}
    payload["mcp_plan_md_package"] = plan or {}
    write_json(case_json_path, payload)
    return {
        "case_dir": str(case_dir),
        "case_json": str(case_json_path),
        "esh": str(esh_path),
    }
