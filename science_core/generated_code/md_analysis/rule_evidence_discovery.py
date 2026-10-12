#!/usr/bin/env python3
"""Read-only public database-record discovery for active rule profiles.

The discovery layer does not activate rules and does not decide thresholds at
runtime. It only turns a polymer/property context into structured reference
records that can be handed to :mod:`rule_profile_registry` for deterministic
unit normalization, conflict checks, and profile activation.

The implementation uses provider plugins.  RadonPy and HTPMD are read from
structured public CSV files. PoLyInfo is only read from official exports or
user-provided local caches; this module deliberately does not scrape PoLyInfo.
"""

from __future__ import annotations

import json
import math
import urllib.request
from csv import DictReader
from io import StringIO
from pathlib import Path
from typing import Any

from plugin_manifest_loader import load_symbol_from_spec
from radonpy_density_screening import (
    build_reference_pack as build_radonpy_density_reference_pack,
    match_context_to_density_references,
)
from rule_profile_registry import (
    PROPERTY_SPECS,
    canonical_engine_scope,
    canonical_force_field,
    canonical_property_name,
    is_qc_only_rule_property,
    rule_profile_automation_policy,
    rule_profile_missing_warn_source,
)
from structure_similarity_transfer import (
    DEFAULT_HIGH_SIMILARITY,
    DEFAULT_MIN_SIMILARITY,
    discover_similarity_transfer,
)

try:
    from rdkit import Chem

    _RDKIT_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - optional dependency
    Chem = None
    _RDKIT_AVAILABLE = False


DENSITY_TEMP_TOLERANCE_K = 200.0
DENSITY_TEMPERATURE_ADJUSTMENT_MAX_K = 150.0
SUPPORTED_PIPELINE_STATES = {"melt"}
COMPUTATIONAL_SOURCE_TYPES = {"radonpy", "htpmd", "simulation_database", "benchmark_simulation"}
SUPPORTED_PROPERTIES = set(PROPERTY_SPECS)
RADONPY_PI1070_URL = "https://raw.githubusercontent.com/RadonPy/RadonPy/develop${PELCB100_REMOTE_PATH}"
RADONPY_PI1070_STATE = "amorphous"
RADONPY_PI1070_FORCE_FIELD = "gaff2"
RADONPY_PI1070_ENGINE_SCOPE = "lammps"
# Reference pointer only (provenance/citation). PoLyInfo data enters exclusively
# through a user-provided local export (see _polyinfo_export_records); this URL is
# never fetched, and this module does not scrape PoLyInfo.
POLYINFO_DATAPOINT_URL = "https://polymer.nims.go.jp/"
HTPMD_INDEX_URL = "https://raw.githubusercontent.com/TRI-AMDD/htp_md/main/htpmd/ml_models/md_wo_structure_cleaned_0_1_4.csv"
PLUGIN_DIR = Path(__file__).resolve().parent / "plugins"
BUILTIN_RULE_PROVIDER_MANIFEST = PLUGIN_DIR / "builtin_rule_evidence_providers.json"

RADONPY_PROPERTY_COLUMNS = {
    "density": ("density", "g/cm3"),
    "self_diffusion": ("self-diffusion", "m2/s"),
    "rg": ("Rg", "nm"),
    "r2": ("r2", "nm2"),
    "cp": ("Cp", "J/kg/K"),
    "cv": ("Cv", "J/kg/K"),
    "compressibility": ("compressibility", "1/Pa"),
    "bulk_modulus": ("bulk_modulus", "Pa"),
    "thermal_expansion": ("volume_expansion", "1/K"),
    "dielectric_constant": ("static_dielectric_const", "dimensionless"),
    "refractive_index": ("refractive_index", "dimensionless"),
    "thermal_conductivity": ("thermal_conductivity", "W/m/K"),
    "thermal_diffusivity": ("thermal_diffusivity", "m2/s"),
}

HTPMD_PROPERTY_COLUMNS = {
    "conductivity": ("conductivity", "S/m"),
    "ion_diffusivity": ("li_diffusivity", "m2/s"),
    "li_diffusivity": ("li_diffusivity", "m2/s"),
    "tfsi_diffusivity": ("tfsi_diffusivity", "m2/s"),
    "poly_diffusivity": ("poly_diffusivity", "m2/s"),
    "transference_number": ("transference_number", "dimensionless"),
    "molarity": ("molarity", "mol/L"),
}


class RuleEvidenceProvider:
    provider_id: str = ""
    mode: str = ""
    supported_properties: frozenset[str] = frozenset()

    def supports(self, property_name: str) -> bool:
        prop = canonical_property_name(property_name)
        return "*" in self.supported_properties or prop in self.supported_properties

    def query_plan(self, context: dict[str, Any], property_name: str) -> dict[str, Any]:
        return {"provider": self.provider_id, "mode": self.mode, "property": canonical_property_name(property_name)}

    def discover(
        self,
        context: dict[str, Any],
        property_name: str,
        *,
        fetcher: Any,
        timeout_seconds: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError


class FunctionRuleEvidenceProvider(RuleEvidenceProvider):
    def __init__(
        self,
        *,
        provider_id: str,
        function: Any,
        supported_properties: list[str] | tuple[str, ...],
        mode: str,
    ) -> None:
        self.provider_id = str(provider_id)
        self.function = function
        self.supported_properties = frozenset(canonical_property_name(item) for item in supported_properties)
        self.mode = str(mode)

    def discover(
        self,
        context: dict[str, Any],
        property_name: str,
        *,
        fetcher: Any,
        timeout_seconds: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self.function(context, canonical_property_name(property_name), fetcher=fetcher, timeout_seconds=timeout_seconds)


_RULE_EVIDENCE_PROVIDERS: dict[str, RuleEvidenceProvider] = {}


def register_rule_evidence_provider(provider: RuleEvidenceProvider, *, replace: bool = False) -> RuleEvidenceProvider:
    provider_id = str(getattr(provider, "provider_id", "") or "").strip()
    if not provider_id:
        raise ValueError("provider_id is required")
    if not provider.supported_properties:
        raise ValueError(f"provider {provider_id} must declare supported_properties")
    if provider_id in _RULE_EVIDENCE_PROVIDERS and not replace:
        raise ValueError(f"rule evidence provider already registered: {provider_id}")
    _RULE_EVIDENCE_PROVIDERS[provider_id] = provider
    return provider


def unregister_rule_evidence_provider(provider_id: str) -> None:
    _RULE_EVIDENCE_PROVIDERS.pop(str(provider_id), None)


def list_rule_evidence_providers() -> dict[str, dict[str, Any]]:
    return {
        provider_id: {
            "provider_id": provider.provider_id,
            "mode": provider.mode,
            "supported_properties": sorted(provider.supported_properties),
        }
        for provider_id, provider in sorted(_RULE_EVIDENCE_PROVIDERS.items())
    }


def load_rule_provider_plugins(paths: list[str | Path] | tuple[str | Path, ...]) -> list[str]:
    loaded: list[str] = []
    for raw_path in paths:
        manifest_path = Path(raw_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for spec in manifest.get("rule_evidence_providers", []) or []:
            spec = dict(spec)
            if spec.get("class"):
                provider_cls = load_symbol_from_spec(spec, base_dir=manifest_path.parent)
                provider = provider_cls()
            else:
                function = load_symbol_from_spec(spec, base_dir=manifest_path.parent)
                provider = FunctionRuleEvidenceProvider(
                    provider_id=str(spec["provider_id"]),
                    function=function,
                    supported_properties=list(spec.get("supported_properties") or ["*"]),
                    mode=str(spec.get("mode") or "plugin"),
                )
            register_rule_evidence_provider(provider, replace=bool(spec.get("replace", True)))
            loaded.append(provider.provider_id)
    return loaded


def _slug(text: Any) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unspecified"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _parse_jsonish(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return dict(json.loads(value))
    return dict(value)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_fetcher(url: str, timeout_seconds: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Codex-PELCB-rule-discovery/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed public URLs only here.
        data = response.read()
    return data.decode("utf-8", errors="replace")


def normalize_discovery_context(context: str | dict[str, Any]) -> dict[str, Any]:
    """Normalize a melt-polymer context for layered database-record comparison."""

    ctx = _parse_jsonish(context)
    synonyms_raw = ctx.get("synonyms") or ctx.get("polymer_synonyms") or []
    if isinstance(synonyms_raw, str):
        synonyms = [synonyms_raw]
    else:
        synonyms = [str(item) for item in synonyms_raw]
    name_hints = {
        _slug(ctx.get("polymer_name")),
        _slug(ctx.get("polymer_family")),
        _slug(ctx.get("repeat_unit_name")),
        *{_slug(item) for item in synonyms},
    }
    name_hints.discard("unspecified")
    state = str(ctx.get("state") or "melt").strip().lower()
    return {
        "polymer_family": str(ctx.get("polymer_family") or ""),
        "repeat_unit_smiles": ctx.get("repeat_unit_smiles"),
        "monomer_smiles": ctx.get("monomer_smiles"),
        "synonyms": synonyms,
        # Names are retained only as review hints. They are not direct-match keys.
        "name_hints": sorted(name_hints),
        "aliases": sorted(name_hints),  # compatibility alias
        "state": state,
        "pipeline_state_supported": state in SUPPORTED_PIPELINE_STATES,
        "temperature_K": _as_float(ctx.get("temperature_K")),
        "pressure_bar": _as_float(ctx.get("pressure_bar")),
        "force_field": str(ctx.get("force_field") or ""),
        "engine_scope": str(ctx.get("engine_scope") or "any"),
        "polyinfo_export_path": (
            ctx.get("polyinfo_export_path")
            or (ctx.get("provider_paths") or {}).get("polyinfo_export")
            or ctx.get("polyinfo_export")
        ),
        "raw_context": ctx,
    }


def _density_seed_records() -> list[dict[str, Any]]:
    """Return registered structured reference-record seeds.

    Values are only used as structured, reproducible provider records
    for tests/offline smoke runs in the rule-profile activation pipeline. The
    `source_provider_kind` field prevents these records being mistaken for a
    live scrape during manuscript or ledger review.
    """

    return [
        {
            "source_id": "radonpy_seed_polyethylene_density_melt",
            "source": "RadonPy structured seed",
            "source_provider_kind": "registered_seed",
            "source_type": "radonpy",
            "url": "https://github.com/RadonPy/RadonPy",
            "independent_key": "radonpy_polyethylene_density_seed",
            "aliases": ["pe", "polyethylene", "linear_pe", "branched_polyethylene"],
            "repeat_unit_smiles": "*CC*",
            "property": "density",
            "value": 0.790,
            "units": "g/cm3",
            "temperature_K": 450.0,
            "pressure_bar": 1.0,
            "state": "melt",
            "force_field": "opls-aa",
            "engine_scope": "gromacs_2024.4",
            "method": "curated RadonPy-like polymer melt density evidence seed",
            "extraction_confidence": 0.92,
        },
        {
            "source_id": "htpmd_seed_polyethylene_density_melt",
            "source": "HTPMD structured seed",
            "source_provider_kind": "registered_seed",
            "source_type": "htpmd",
            "url": "https://github.com/learningmatter-mit/HTP",
            "independent_key": "htpmd_polyethylene_density_seed",
            "aliases": ["pe", "polyethylene", "linear_pe", "branched_polyethylene"],
            "repeat_unit_smiles": "*CC*",
            "property": "density",
            "value": 805.0,
            "units": "kg/m3",
            "temperature_K": 450.0,
            "pressure_bar": 1.0,
            "state": "melt",
            "force_field": "opls-aa",
            "engine_scope": "gromacs_2024.4",
            "method": "curated HTPMD-like polymer melt density evidence seed",
            "extraction_confidence": 0.90,
        },
        {
            "source_id": "polyinfo_seed_polyethylene_density_melt",
            "source": "PoLyInfo structured seed",
            "source_provider_kind": "registered_seed",
            "source_type": "polyinfo",
            "url": "https://polymer.nims.go.jp/",
            "independent_key": "polyinfo_polyethylene_density_seed",
            "aliases": ["pe", "polyethylene", "linear_pe", "branched_polyethylene"],
            "repeat_unit_smiles": "*CC*",
            "property": "density",
            "value": 0.800,
            "units": "g/cm3",
            "temperature_K": 450.0,
            "pressure_bar": 1.0,
            "state": "melt",
            "method": "curated PoLyInfo-like polymer density evidence seed",
            "extraction_confidence": 0.90,
        },
        {
            "source_id": "clean_control_seed_pe_lcb_density_melt",
            "source": "same-input clean/control simulation seed",
            "source_provider_kind": "control_seed",
            "source_type": "same_input_clean_control",
            "url": "local:pelcb_clean_engine_control",
            "independent_key": "pelcb_same_input_clean_control_density_seed",
            "aliases": ["pe", "polyethylene", "branched_polyethylene"],
            "repeat_unit_smiles": "*CC*",
            "property": "density",
            "value": 0.775,
            "units": "g/cm3",
            "temperature_K": 450.0,
            "pressure_bar": 1.0,
            "state": "melt",
            "method": "same-input clean-engine/control density evidence seed",
            "extraction_confidence": 0.88,
        },
        {
            "source_id": "radonpy_seed_polyethylene_self_diffusion_melt",
            "source": "RadonPy structured seed",
            "source_provider_kind": "registered_seed",
            "source_type": "radonpy",
            "url": "https://github.com/RadonPy/RadonPy",
            "independent_key": "radonpy_polyethylene_self_diffusion_seed",
            "aliases": ["pe", "polyethylene", "linear_pe", "branched_polyethylene"],
            "repeat_unit_smiles": "*CC*",
            "property": "self_diffusion",
            "value": 1.2e-10,
            "units": "m2/s",
            "temperature_K": 450.0,
            "pressure_bar": 1.0,
            "state": "melt",
            "force_field": "opls-aa",
            "engine_scope": "gromacs_2024.4",
            "method": "curated RadonPy-like polymer melt self-diffusion evidence seed",
            "extraction_confidence": 0.88,
        },
        {
            "source_id": "htpmd_seed_polyethylene_self_diffusion_melt",
            "source": "HTPMD structured seed",
            "source_provider_kind": "registered_seed",
            "source_type": "htpmd",
            "url": "https://github.com/learningmatter-mit/HTP",
            "independent_key": "htpmd_polyethylene_self_diffusion_seed",
            "aliases": ["pe", "polyethylene", "linear_pe", "branched_polyethylene"],
            "repeat_unit_smiles": "*CC*",
            "property": "self_diffusion",
            "value": 1.35e-10,
            "units": "m2/s",
            "temperature_K": 450.0,
            "pressure_bar": 1.0,
            "state": "melt",
            "force_field": "opls-aa",
            "engine_scope": "gromacs_2024.4",
            "method": "curated HTPMD-like polymer melt self-diffusion evidence seed",
            "extraction_confidence": 0.86,
        },
        {
            "source_id": "radonpy_seed_poly_alpha_olefin_density_melt",
            "source": "RadonPy structured seed",
            "source_provider_kind": "registered_seed",
            "source_type": "radonpy",
            "url": "https://github.com/RadonPy/RadonPy",
            "independent_key": "radonpy_poly_alpha_olefin_density_seed",
            "aliases": ["poly_alpha_olefin", "linear_alpha_olefin", "poly_1_octene", "pao"],
            "repeat_unit_smiles": "*C(C*)CCCCCC*",
            "degree_of_polymerization": 100,
            "property": "density",
            "value": 0.815,
            "units": "g/cm3",
            "temperature_K": 450.0,
            "pressure_bar": 1.0,
            "state": "melt",
            "force_field": "opls-aa",
            "engine_scope": "gromacs_2024.4",
            "method": "curated RadonPy-like alpha-olefin smoke density seed",
            "extraction_confidence": 0.88,
        },
        {
            "source_id": "htpmd_seed_poly_alpha_olefin_density_melt",
            "source": "HTPMD structured seed",
            "source_provider_kind": "registered_seed",
            "source_type": "htpmd",
            "url": "https://github.com/learningmatter-mit/HTP",
            "independent_key": "htpmd_poly_alpha_olefin_density_seed",
            "aliases": ["poly_alpha_olefin", "linear_alpha_olefin", "poly_1_octene", "pao"],
            "repeat_unit_smiles": "*C(C*)CCCCCC*",
            "degree_of_polymerization": 100,
            "property": "density",
            "value": 825.0,
            "units": "kg/m3",
            "temperature_K": 450.0,
            "pressure_bar": 1.0,
            "state": "melt",
            "force_field": "opls-aa",
            "engine_scope": "gromacs_2024.4",
            "method": "curated HTPMD-like alpha-olefin smoke density seed",
            "extraction_confidence": 0.87,
        },
    ]


def _row_aliases(row: dict[str, Any]) -> set[str]:
    aliases = set()
    for key in (
        "smiles",
        "poly_smiles",
        "mol_smiles",
        "monomer_smiles",
        "repeat_unit_smiles",
        "polymer_family",
        "polymer_name",
        "polymer",
        "name",
        "monomer_ID",
    ):
        value = row.get(key)
        if value not in (None, ""):
            aliases.add(_slug(value))
    return aliases


def _radonpy_live_records(context: dict[str, Any], property_name: str, csv_text: str) -> list[dict[str, Any]]:
    prop = canonical_property_name(property_name)
    if prop not in RADONPY_PROPERTY_COLUMNS:
        return []
    column, units = RADONPY_PROPERTY_COLUMNS[prop]
    rows = DictReader(StringIO(csv_text))
    records: list[dict[str, Any]] = []
    for row in rows:
        row_aliases = _row_aliases(row)
        value = _as_float(row.get(column))
        if value is None:
            continue
        temp = _as_float(row.get("temp"))
        press = _as_float(row.get("press"))
        monomer_id = str(row.get("monomer_ID") or "unknown")
        records.append(
            {
                "source_id": f"radonpy_live_{monomer_id}_{prop}",
                "source": "RadonPy PI1070.csv live fetch",
                "source_provider_kind": "live_public_dataset",
                "source_type": "radonpy",
                "url": RADONPY_PI1070_URL,
                "independent_key": f"radonpy_pi1070_{monomer_id}",
                "property": prop,
                "value": value,
                "units": units,
                "temperature_K": temp,
                "pressure_bar": press,
                # PI1070 contains amorphous homopolymers calculated with the
                # dataset-wide RadonPy GAFF2/LAMMPS protocol. These records are
                # analogues for a melt profile, not direct melt-state matches.
                "state": RADONPY_PI1070_STATE,
                "force_field": RADONPY_PI1070_FORCE_FIELD,
                "engine_scope": RADONPY_PI1070_ENGINE_SCOPE,
                "polymer_topology": "linear",
                "method": f"RadonPy PI1070.csv {column} column",
                "extraction_confidence": 0.95,
                "raw_fields": {
                    "monomer_ID": monomer_id,
                    "smiles": str(row.get("smiles") or "").strip(),
                    f"{column}_min": row.get(f"{column}_min"),
                    f"{column}_max": row.get(f"{column}_max"),
                    f"{column}_count": row.get(f"{column}_count"),
                    f"{column}_std": row.get(f"{column}_std"),
                    "volume_expansion": row.get("volume_expansion"),
                    "volume_expansion_std": row.get("volume_expansion_std"),
                    "DP": row.get("DP"),
                    "Mn": row.get("Mn"),
                    "dataset_state": RADONPY_PI1070_STATE,
                    "dataset_force_field": RADONPY_PI1070_FORCE_FIELD,
                    "dataset_engine": RADONPY_PI1070_ENGINE_SCOPE,
                },
                "DP": row.get("DP"),
                "Mn": row.get("Mn"),
                "value_min": row.get(f"{column}_min"),
                "value_max": row.get(f"{column}_max"),
                "value_std": row.get(f"{column}_std"),
                "replicate_count": row.get(f"{column}_count"),
                "volume_expansion_coefficient_K_inv": row.get("volume_expansion"),
                "volume_expansion_coefficient_std_K_inv": row.get("volume_expansion_std"),
                "repeat_unit_smiles": str(row.get("smiles") or "").strip(),
                "aliases": sorted(row_aliases),
                "extraction_confidence_origin": "provider_fixed",
            }
        )
    return records


def _temperature_adjust_public_density_record(
    record: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Convert a public density record to the requested temperature for screening.

    The conversion is deliberately narrow. It is available only for public
    RadonPy records that report a positive volumetric expansion coefficient,
    remain within a 150 K extrapolation window and contain repeated density
    estimates. The result is marked as a cross-method screening reference, not
    as a force-field-specific material truth value.
    """

    if canonical_property_name(record.get("property")) != "density":
        return record
    if str(record.get("source_type") or "").lower() != "radonpy":
        return record
    target_temperature = _as_float(context.get("temperature_K"))
    source_temperature = _as_float(record.get("temperature_K"))
    alpha = _as_float(record.get("volume_expansion_coefficient_K_inv"))
    if target_temperature is None or source_temperature is None or alpha is None or alpha <= 0:
        return record
    delta_temperature = target_temperature - source_temperature
    if delta_temperature <= 0 or delta_temperature > DENSITY_TEMPERATURE_ADJUSTMENT_MAX_K:
        return record

    source_value = _as_float(record.get("value"))
    if source_value is None:
        return record
    denominator = 1.0 + alpha * delta_temperature
    if denominator <= 0:
        return record

    adjusted = dict(record)
    adjusted["value"] = source_value / denominator
    adjusted["temperature_K"] = target_temperature
    adjusted["state"] = "melt"
    adjusted["screening_only"] = True
    adjusted["cross_method_screening"] = True
    adjusted["source_provider_kind"] = "live_public_dataset_temperature_adjusted"
    adjusted["original_value"] = source_value
    adjusted["original_temperature_K"] = source_temperature
    adjusted["temperature_adjustment"] = {
        "method": "rho_target = rho_source / [1 + alpha_V * (T_target - T_source)]",
        "source_temperature_K": source_temperature,
        "target_temperature_K": target_temperature,
        "delta_temperature_K": delta_temperature,
        "volume_expansion_coefficient_K_inv": alpha,
    }

    source_low = _as_float(record.get("value_min"))
    source_high = _as_float(record.get("value_max"))
    alpha_std = max(0.0, _as_float(record.get("volume_expansion_coefficient_std_K_inv")) or 0.0)
    if source_low is not None and source_high is not None:
        # Two standard deviations of alpha are carried into the transformed
        # bounds so the screening profile does not depend on rounded means.
        lower_denominator = 1.0 + (alpha + 2.0 * alpha_std) * delta_temperature
        upper_alpha = max(0.0, alpha - 2.0 * alpha_std)
        upper_denominator = 1.0 + upper_alpha * delta_temperature
        adjusted["value_min"] = source_low / lower_denominator
        adjusted["value_max"] = source_high / upper_denominator

    replicate_count = int(_as_float(record.get("replicate_count")) or 0)
    adjusted["single_source_screening_eligible"] = bool(
        replicate_count >= 3
        and _as_float(adjusted.get("value_min")) is not None
        and _as_float(adjusted.get("value_max")) is not None
    )
    return adjusted


def discover_radonpy_density_screening_pack(
    csv_content: str | bytes,
    *,
    downloaded_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the PE-LCB density source pack from a RadonPy PI1070 export.

    This density-specific path applies the saturated, acyclic hydrocarbon
    filter and derives the 400 K envelope. It deliberately bypasses the
    generic composite similarity-transfer path.
    """

    csv_bytes = csv_content if isinstance(csv_content, bytes) else csv_content.encode("utf-8")
    return build_radonpy_density_reference_pack(
        csv_bytes,
        downloaded_at_utc=downloaded_at_utc,
        source_url=RADONPY_PI1070_URL,
    )


def _htpmd_live_records(context: dict[str, Any], property_name: str, csv_text: str) -> list[dict[str, Any]]:
    prop = canonical_property_name(property_name)
    if prop not in HTPMD_PROPERTY_COLUMNS:
        return []
    column, units = HTPMD_PROPERTY_COLUMNS[prop]
    rows = DictReader(StringIO(csv_text))
    records: list[dict[str, Any]] = []
    for row in rows:
        row_aliases = _row_aliases(row)
        value = _as_float(row.get(column))
        if value is None:
            continue
        trajectory_id = str(row.get("trajectory_id") or row.get("id") or "unknown")
        records.append(
            {
                "source_id": f"htpmd_live_{trajectory_id}_{prop}",
                "source": "HTPMD public transport CSV live fetch",
                "source_provider_kind": "live_public_dataset",
                "source_type": "htpmd",
                "url": HTPMD_INDEX_URL,
                "independent_key": f"htpmd_transport_{trajectory_id}_{column}",
                "property": prop,
                "value": value,
                "units": units,
                "temperature_K": _as_float(row.get("temperature_K") or row.get("temperature")),
                "pressure_bar": _as_float(row.get("pressure_bar") or row.get("pressure")),
                "state": str(context.get("state") or "electrolyte"),
                "method": f"HTPMD public CSV {column} column",
                "extraction_confidence": 0.92,
                "raw_fields": {
                    "trajectory_id": trajectory_id,
                    "mol_smiles": row.get("mol_smiles"),
                    "poly_smiles": row.get("poly_smiles"),
                    "molarity": row.get("molarity"),
                },
                "poly_smiles": row.get("poly_smiles"),
                "mol_smiles": row.get("mol_smiles"),
                "aliases": sorted(row_aliases),
                "extraction_confidence_origin": "provider_fixed",
            }
        )
    return records


def _polyinfo_export_records(context: dict[str, Any], property_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prop = canonical_property_name(property_name)
    path_value = context.get("polyinfo_export_path")
    if not path_value:
        return [], {
            "provider": "polyinfo_export",
            "status": "blocked_provider_requires_export",
            "mode": "user_provided_export",
            "records_extracted": 0,
            "note": "PoLyInfo requires registered access; provide an official export/cache path to enable extraction.",
        }
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return [], {
            "provider": "polyinfo_export",
            "status": "error",
            "mode": "user_provided_export",
            "records_extracted": 0,
            "error": f"export path does not exist: {path}",
        }

    try:
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = raw if isinstance(raw, list) else raw.get("records", [])
        else:
            rows = list(DictReader(StringIO(path.read_text(encoding="utf-8-sig"))))
    except Exception as exc:  # noqa: BLE001 - provider status should capture parse failures.
        return [], {
            "provider": "polyinfo_export",
            "status": "error",
            "mode": "user_provided_export",
            "records_extracted": 0,
            "error": str(exc),
        }

    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_prop = canonical_property_name(row.get("property") or row.get("property_name") or "")
        if row_prop != prop:
            continue
        row_aliases = _row_aliases(row)
        value = _as_float(row.get("value") or row.get("property_value"))
        if value is None:
            continue
        source_id = str(row.get("source_id") or row.get("polyinfo_id") or row.get("id") or f"polyinfo_export_{len(records)+1}")
        records.append(
            {
                "source_id": source_id,
                "source": "PoLyInfo official export/local cache",
                "source_provider_kind": "user_provided_export",
                "source_type": "polyinfo",
                "url": row.get("url") or row.get("doi") or f"local:{path}",
                "independent_key": str(row.get("independent_key") or row.get("doi") or source_id),
                "property": prop,
                "value": value,
                "units": row.get("units") or row.get("unit"),
                "temperature_K": _as_float(row.get("temperature_K") or row.get("temp_K") or row.get("temperature")),
                "pressure_bar": _as_float(row.get("pressure_bar") or row.get("pressure")),
                "state": str(row.get("state") or context.get("state") or ""),
                "method": str(row.get("method") or "PoLyInfo official export/local cache"),
                "extraction_confidence": float(row.get("extraction_confidence") or row.get("confidence") or 0.9),
                "raw_fields": {"export_path": str(path), "polymer_name": row.get("polymer_name") or row.get("polymer_family")},
                "aliases": sorted(row_aliases or set(context.get("name_hints", []))),
                "repeat_unit_smiles": row.get("repeat_unit_smiles") or row.get("poly_smiles") or row.get("smiles"),
                "monomer_smiles": row.get("monomer_smiles") or row.get("mol_smiles"),
                "degree_of_polymerization": row.get("degree_of_polymerization") or row.get("DP") or row.get("dp"),
                "molecular_weight": row.get("molecular_weight") or row.get("Mn") or row.get("mn"),
                "extraction_confidence_origin": "user_export_field"
                if row.get("extraction_confidence") not in (None, "") or row.get("confidence") not in (None, "")
                else "provider_default",
            }
        )
    return records, {
        "provider": "polyinfo_export",
        "status": "ok",
        "mode": "user_provided_export",
        "records_extracted": len(records),
        "export_path": str(path),
    }


def _live_provider_records(
    context: dict[str, Any],
    property_name: str,
    *,
    fetcher: Any,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    prop = canonical_property_name(property_name)

    live_providers = []
    if prop in RADONPY_PROPERTY_COLUMNS:
        live_providers.append(("radonpy_pi1070", RADONPY_PI1070_URL, "radonpy_csv"))
    if prop in HTPMD_PROPERTY_COLUMNS:
        live_providers.append(("htpmd_public_index", HTPMD_INDEX_URL, "htpmd_csv"))
    for provider, url, parser in live_providers:
        try:
            text = fetcher(url, timeout_seconds)
            if parser == "radonpy_csv":
                provider_records = _radonpy_live_records(context, prop, text)
            else:
                provider_records = _htpmd_live_records(context, prop, text)
            records.extend(provider_records)
            statuses.append(
                {
                    "provider": provider,
                    "url": url,
                    "status": "ok",
                    "mode": "live_public_fetch",
                    "records_extracted": len(provider_records),
                }
            )
        except Exception as exc:  # noqa: BLE001 - provider errors must be ledgered, not fatal.
            statuses.append(
                {
                    "provider": provider,
                    "url": url,
                    "status": "error",
                    "mode": "live_public_fetch",
                    "records_extracted": 0,
                    "error": str(exc),
                }
            )
    return records, statuses


class RadonPyPi1070Provider(RuleEvidenceProvider):
    provider_id = "radonpy_pi1070"
    mode = "live_public_fetch"
    supported_properties = frozenset(RADONPY_PROPERTY_COLUMNS)

    def query_plan(self, context: dict[str, Any], property_name: str) -> dict[str, Any]:
        return {"provider": self.provider_id, "mode": self.mode, "url": RADONPY_PI1070_URL}

    def discover(
        self,
        context: dict[str, Any],
        property_name: str,
        *,
        fetcher: Any,
        timeout_seconds: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prop = canonical_property_name(property_name)
        text = fetcher(RADONPY_PI1070_URL, timeout_seconds)
        records = _radonpy_live_records(context, prop, text)
        return records, {
            "provider": self.provider_id,
            "url": RADONPY_PI1070_URL,
            "status": "ok",
            "mode": self.mode,
            "records_extracted": len(records),
        }


class HtpmdPublicIndexProvider(RuleEvidenceProvider):
    provider_id = "htpmd_public_index"
    mode = "live_public_fetch"
    supported_properties = frozenset(HTPMD_PROPERTY_COLUMNS)

    def query_plan(self, context: dict[str, Any], property_name: str) -> dict[str, Any]:
        return {"provider": self.provider_id, "mode": self.mode, "url": HTPMD_INDEX_URL}

    def discover(
        self,
        context: dict[str, Any],
        property_name: str,
        *,
        fetcher: Any,
        timeout_seconds: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prop = canonical_property_name(property_name)
        text = fetcher(HTPMD_INDEX_URL, timeout_seconds)
        records = _htpmd_live_records(context, prop, text)
        return records, {
            "provider": self.provider_id,
            "url": HTPMD_INDEX_URL,
            "status": "ok",
            "mode": self.mode,
            "records_extracted": len(records),
        }


class PolyInfoExportProvider(RuleEvidenceProvider):
    provider_id = "polyinfo_export"
    mode = "user_provided_export"
    supported_properties = frozenset({"*"})

    def query_plan(self, context: dict[str, Any], property_name: str) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "mode": self.mode,
            "path": context.get("polyinfo_export_path"),
        }

    def discover(
        self,
        context: dict[str, Any],
        property_name: str,
        *,
        fetcher: Any,
        timeout_seconds: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del fetcher, timeout_seconds
        return _polyinfo_export_records(context, property_name)


class StructuredSeedProvider(RuleEvidenceProvider):
    provider_id = "registered_structured_seed_provider"
    mode = "offline_structured_seed"
    supported_properties = frozenset({"*"})

    def query_plan(self, context: dict[str, Any], property_name: str) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "mode": self.mode,
            "property": canonical_property_name(property_name),
            "name_hints": context.get("name_hints", []),
        }

    def discover(
        self,
        context: dict[str, Any],
        property_name: str,
        *,
        fetcher: Any,
        timeout_seconds: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del context, property_name, fetcher, timeout_seconds
        records = [dict(record, extraction_confidence_origin="provider_fixed_seed") for record in _density_seed_records()]
        return records, {
            "provider": self.provider_id,
            "status": "ok",
            "mode": self.mode,
            "records_extracted": len(records),
        }


def _register_builtin_rule_evidence_providers() -> None:
    if BUILTIN_RULE_PROVIDER_MANIFEST.exists():
        load_rule_provider_plugins([BUILTIN_RULE_PROVIDER_MANIFEST])
        return
    for provider in (RadonPyPi1070Provider(), HtpmdPublicIndexProvider(), PolyInfoExportProvider(), StructuredSeedProvider()):
        register_rule_evidence_provider(provider, replace=True)


_SMILES_FIELDS = (
    "canonical_chain_smiles",
    "chain_smiles",
    "polymer_chain_smiles",
    "oligomer_smiles",
    "canonical_segment_smiles",
    "normalized_segment_smiles",
    "surrogate_dp_smiles",
    "repeat_unit_smiles",
    "canonical_repeat_unit_smiles",
    "monomer_smiles",
    "smiles",
    "poly_smiles",
    "mol_smiles",
)


def _canonical_smiles(value: Any) -> str | None:
    """Return canonical SMILES for structural identity, or None.

    None is returned when RDKit is unavailable or the SMILES is missing/invalid;
    callers then fall back to name/alias matching.
    """
    if not _RDKIT_AVAILABLE or Chem is None:
        return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover - defensive against RDKit edge cases
        return None


def _canonical_smiles_set(payload: dict[str, Any]) -> set[str]:
    """Canonical SMILES collected from a context or record (incl. ``raw_fields``)."""
    candidates: list[Any] = [payload.get(key) for key in _SMILES_FIELDS]
    raw_fields = payload.get("raw_fields")
    if isinstance(raw_fields, dict):
        candidates.extend(raw_fields.get(key) for key in _SMILES_FIELDS)
    raw_context = payload.get("raw_context")
    if isinstance(raw_context, dict):
        candidates.extend(raw_context.get(key) for key in _SMILES_FIELDS)
    out: set[str] = set()
    for candidate in candidates:
        canon = _canonical_smiles(candidate)
        if canon:
            out.add(canon)
    return out


def _normalized_structure_strings(payload: dict[str, Any]) -> set[str]:
    """Return exact normalized structure strings without using name aliases."""

    sources = [payload]
    for nested_key in ("raw_fields", "raw_context"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            sources.append(nested)
    out: set[str] = set()
    for source in sources:
        for key in _SMILES_FIELDS:
            value = source.get(key)
            if value not in (None, ""):
                out.add("".join(str(value).split()))
    return out


_STABLE_STRUCTURE_ID_FIELDS = (
    "structure_hash",
    "canonical_structure_hash",
    "inchikey",
    "inchi_key",
    "polymer_inchikey",
    "monomer_inchikey",
)
_CHAIN_SIZE_FIELDS = (
    "degree_of_polymerization",
    "polymerization_degree",
    "DP",
    "dp",
    "repeat_units_per_chain",
    "chain_length",
    "target_total_carbons_per_chain",
    "actual_total_carbons_per_chain",
    "molecular_weight",
    "Mn",
    "mn",
    "Mw",
    "mw",
)


def _stable_structure_ids(payload: dict[str, Any]) -> set[str]:
    raw_fields = payload.get("raw_fields") if isinstance(payload.get("raw_fields"), dict) else {}
    out: set[str] = set()
    for source in (payload, raw_fields):
        for key in _STABLE_STRUCTURE_ID_FIELDS:
            value = source.get(key)
            if value not in (None, ""):
                out.add(str(value).strip().upper())
    return out


def _chain_size_values(payload: dict[str, Any]) -> dict[str, float]:
    raw_fields = payload.get("raw_fields") if isinstance(payload.get("raw_fields"), dict) else {}
    out: dict[str, float] = {}
    for key in _CHAIN_SIZE_FIELDS:
        value = _as_float(payload.get(key))
        if value is None:
            value = _as_float(raw_fields.get(key))
        if value is not None:
            out[key] = value
    return out


def _chain_sizes_compatible(target: dict[str, float], reference: dict[str, float], relative_tolerance: float = 0.10) -> bool:
    if not target or not reference:
        return True
    for key in target.keys() & reference.keys():
        scale = max(abs(target[key]), abs(reference[key]), 1.0)
        if abs(target[key] - reference[key]) / scale <= relative_tolerance:
            return True
    return False


def _record_name_hints(record: dict[str, Any]) -> set[str]:
    raw_fields = record.get("raw_fields") if isinstance(record.get("raw_fields"), dict) else {}
    hints = {_slug(item) for item in record.get("aliases", []) if item not in (None, "")}
    for source in (record, raw_fields):
        for key in ("polymer_name", "polymer_family", "polymer", "name", "monomer_ID"):
            if source.get(key) not in (None, ""):
                hints.add(_slug(source[key]))
    hints.discard("unspecified")
    return hints


def _record_match_assessment(record: dict[str, Any], context: dict[str, Any], property_name: str) -> dict[str, Any]:
    """Compare one database record using structure, chain and condition tiers."""

    assessment: dict[str, Any] = {
        "matched": False,
        "reason": "not_checked",
        "match_basis": None,
        "name_hint_overlap": sorted(set(context.get("name_hints", [])) & _record_name_hints(record)),
        "review_hints_only": True,
    }
    if canonical_property_name(record.get("property")) != canonical_property_name(property_name):
        return assessment | {"reason": "property_mismatch"}

    requested_state = str(context.get("state") or "melt").lower()
    if requested_state not in SUPPORTED_PIPELINE_STATES:
        return assessment | {"reason": "unsupported_target_state"}
    record_state = str(record.get("state") or "").lower()
    if not record_state:
        return assessment | {"reason": "reference_state_missing"}
    if record_state not in SUPPORTED_PIPELINE_STATES:
        return assessment | {"reason": "reference_state_not_melt"}

    target_ids = _stable_structure_ids(context)
    record_ids = _stable_structure_ids(record)
    match_basis = None
    if target_ids and record_ids:
        if target_ids.isdisjoint(record_ids):
            return assessment | {"reason": "stable_structure_id_mismatch"}
        match_basis = "stable_structure_id"

    ctx_canon = _canonical_smiles_set(context)
    rec_canon = _canonical_smiles_set(record)
    if match_basis is None and ctx_canon and rec_canon:
        if ctx_canon.isdisjoint(rec_canon):
            return assessment | {"reason": "structure_mismatch"}
        match_basis = "canonical_smiles"
    if match_basis is None:
        target_structure_strings = _normalized_structure_strings(context)
        record_structure_strings = _normalized_structure_strings(record)
        if target_structure_strings and record_structure_strings:
            if target_structure_strings.isdisjoint(record_structure_strings):
                return assessment | {"reason": "structure_mismatch"}
            match_basis = "normalized_structure_string"
    if match_basis is None:
        return assessment | {"reason": "structure_identifier_missing"}

    target_chain = _chain_size_values(context.get("raw_context") or context)
    record_chain = _chain_size_values(record)
    if target_chain and not record_chain:
        return assessment | {"reason": "reference_chain_size_missing", "match_basis": match_basis}
    if target_chain and record_chain and not _chain_sizes_compatible(target_chain, record_chain):
        return assessment | {"reason": "chain_size_mismatch", "match_basis": match_basis}

    requested_temp = context.get("temperature_K")
    record_temp = _as_float(record.get("temperature_K"))
    if requested_temp is not None and record_temp is None:
        return assessment | {"reason": "reference_temperature_missing", "match_basis": match_basis}
    if requested_temp is not None and record_temp is not None and abs(float(requested_temp) - record_temp) > DENSITY_TEMP_TOLERANCE_K:
        return assessment | {"reason": "temperature_mismatch", "match_basis": match_basis}

    requested_pressure = _as_float(context.get("pressure_bar"))
    record_pressure = _as_float(record.get("pressure_bar"))
    if requested_pressure is not None and record_pressure is None:
        return assessment | {"reason": "reference_pressure_missing", "match_basis": match_basis}
    if requested_pressure is not None and record_pressure is not None and abs(requested_pressure - record_pressure) > 1.0:
        return assessment | {"reason": "pressure_mismatch", "match_basis": match_basis}

    source_type = _slug(record.get("source_type"))
    requested_force_field = canonical_force_field(context.get("force_field"))
    record_force_field = canonical_force_field(record.get("force_field"))
    requested_engine = canonical_engine_scope(context.get("engine_scope") or context.get("engine"))
    record_engine = canonical_engine_scope(record.get("engine_scope") or record.get("engine"))
    cross_method_screening = bool(record.get("cross_method_screening"))
    if source_type in COMPUTATIONAL_SOURCE_TYPES and not cross_method_screening:
        if not record_force_field:
            return assessment | {"reason": "reference_force_field_missing", "match_basis": match_basis}
        if record_engine.get("scope") == "missing":
            return assessment | {"reason": "reference_engine_missing", "match_basis": match_basis}
    if not cross_method_screening and requested_force_field and record_force_field and requested_force_field != record_force_field:
        return assessment | {"reason": "force_field_mismatch", "match_basis": match_basis}
    if not cross_method_screening and requested_engine.get("scope") not in {"missing", "any"} and record_engine.get("scope") not in {"missing", "any"}:
        if requested_engine.get("family") != record_engine.get("family"):
            return assessment | {"reason": "engine_mismatch", "match_basis": match_basis}
        if (
            requested_engine.get("version")
            and record_engine.get("version")
            and requested_engine.get("version") != record_engine.get("version")
        ):
            return assessment | {"reason": "engine_version_mismatch", "match_basis": match_basis}

    return assessment | {
        "matched": True,
        "reason": "matched",
        "match_basis": match_basis,
        "review_hints_only": False,
        "chain_fields_compared": sorted(target_chain.keys() & record_chain.keys()),
    }


def _record_matches_context(record: dict[str, Any], context: dict[str, Any], property_name: str) -> tuple[bool, str]:
    assessment = _record_match_assessment(record, context, property_name)
    if bool(assessment["matched"]):
        return True, "matched_by_structure"
    return False, str(assessment["reason"])


def discover_rule_evidence(
    context: str | dict[str, Any],
    property_name: str = "density",
    *,
    use_live: bool = True,
    include_seed: bool = True,
    fetcher: Any | None = None,
    timeout_seconds: float = 15.0,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    high_similarity: float = DEFAULT_HIGH_SIMILARITY,
) -> dict[str, Any]:
    """Compare registered database records and, if needed, similar references.

    Similarity thresholds apply to the deterministic composite structure and
    condition score. They do not define an allowed percentage difference in
    the physical-property values.
    """

    prop = canonical_property_name(property_name)
    normalized_context = normalize_discovery_context(context)
    if not normalized_context["pipeline_state_supported"]:
        readiness = "blocked_unsupported_state"
        return {
            "status": "unsupported_state",
            "property": prop,
            "context": _json_safe(normalized_context),
            "query_plan": [],
            "source_count": 0,
            "reference_records": [],
            "source_evidence": [],  # compatibility alias
            "excluded_source_count": 0,
            "excluded_sources": [],
            "provider_status": [],
            "activation_readiness": readiness,
            "freeze_readiness": readiness,  # compatibility alias
            "warn_source": rule_profile_missing_warn_source(prop),
            "notes": ["This pipeline supports melt-state records only; solid and glass records require a separate profile."],
        }
    if prop not in SUPPORTED_PROPERTIES:
        policy = rule_profile_automation_policy(prop)
        if is_qc_only_rule_property(prop):
            return {
                "status": "profile_not_activatable",
                "legacy_status": "profile_unfreezable",
                "property": prop,
                "context": _json_safe(normalized_context),
                "query_plan": [],
                "source_count": 0,
                "reference_records": [],
                "source_evidence": [],
                "excluded_source_count": 0,
                "excluded_sources": [],
                "provider_status": [],
                "activation_readiness": "qc_only_no_external_profile",
                "freeze_readiness": "profile_unfreezable_qc_only",
                "warn_source": rule_profile_missing_warn_source(prop),
                "automation_level": policy["automation_level"],
                "fallback_status": policy["fallback_status"],
                "notes": [
                    "This observable is handled by schema/quality-control rules, not by an external public-data pass threshold.",
                    "Return WARN and send the available references to the audit layer instead of activating a guessed rule_profile.",
                ],
            }
        return {
            "status": "unsupported_property",
            "property": prop,
            "context": _json_safe(normalized_context),
            "query_plan": [],
            "source_count": 0,
                "reference_records": [],
                "source_evidence": [],
            "excluded_source_count": 0,
            "excluded_sources": [],
            "provider_status": [],
            "activation_readiness": "blocked_unsupported_property",
            "freeze_readiness": "blocked_unsupported_property",
            "warn_source": rule_profile_missing_warn_source(prop),
            "automation_level": policy["automation_level"],
            "fallback_status": policy["fallback_status"],
            "notes": ["This property is not registered for public-data rule-profile discovery."],
        }

    active_providers: list[RuleEvidenceProvider] = []
    for provider in _RULE_EVIDENCE_PROVIDERS.values():
        if not provider.supports(prop):
            continue
        if provider.mode == "live_public_fetch" and not use_live:
            continue
        if provider.mode == "offline_structured_seed" and not include_seed:
            continue
        active_providers.append(provider)
    query_plan = [provider.query_plan(normalized_context, prop) for provider in active_providers]
    matches: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    provider_status: list[dict[str, Any]] = []

    candidate_records: list[dict[str, Any]] = []
    for provider in active_providers:
        try:
            records, status = provider.discover(
                normalized_context,
                prop,
                fetcher=fetcher or _default_fetcher,
                timeout_seconds=timeout_seconds,
            )
            provider_status.append(status)
            candidate_records.extend(records)
        except Exception as exc:  # noqa: BLE001 - provider failures are ledgered.
            provider_status.append(
                {
                    "provider": provider.provider_id,
                    "status": "error",
                    "mode": provider.mode,
                    "records_extracted": 0,
                    "error": str(exc),
                }
            )

    property_records: list[dict[str, Any]] = []
    for original_record in candidate_records:
        record = _temperature_adjust_public_density_record(original_record, normalized_context)
        assessment = _record_match_assessment(record, normalized_context, prop)
        matched = bool(assessment["matched"])
        reason = str(assessment["reason"])
        cleaned = {k: v for k, v in record.items() if k != "aliases"}
        if reason != "property_mismatch":
            property_records.append(record)
        if matched:
            cleaned["database_record_comparison"] = assessment
            matches.append(cleaned)
        elif reason != "property_mismatch":
            excluded.append(
                {
                    "source_id": record.get("source_id"),
                    "reason": reason,
                    "match_basis": assessment.get("match_basis"),
                    "name_hint_overlap": assessment.get("name_hint_overlap", []),
                }
            )

    automatic_density_profile = {
        "status": "not_run",
        "reason": "direct_profile_match" if matches else "property_is_not_density",
        "reference_records": [],
        "profile_activatable": False,
    }
    if not matches and prop == "density":
        automatic_density_profile = match_context_to_density_references(
            normalized_context,
            property_records,
            min_similarity=min_similarity,
            high_similarity=high_similarity,
        )
        if automatic_density_profile.get("profile_activatable"):
            for record in automatic_density_profile.get("reference_records", []):
                cleaned = {key: value for key, value in record.items() if key != "aliases"}
                cleaned["database_record_comparison"] = {
                    "matched": True,
                    "reason": "matched_by_structure_similarity",
                    "match_basis": "atom_pair_count_tanimoto",
                    "structure_similarity_score": record.get("structure_similarity_score"),
                    "structure_similarity_tier": record.get("structure_similarity_tier"),
                    "review_hints_only": False,
                }
                matches.append(cleaned)

    similarity_transfer = {
        "status": "skipped_direct_profile_match" if matches else "not_run",
        "property": prop,
        "min_similarity": min_similarity,
        "high_similarity": high_similarity,
        "similarity_reference_id": None,
        "similarity_score": None,
        "similarity_tier": None if matches else "NO_MATCH",
        "similarity_distance_reason": "direct_profile_match" if matches else "not_run",
        "audit_required": False if matches else True,
        # Compatibility field retained for archived callers.
        "review_flag": False if matches else True,
    }
    if matches:
        readiness = "ready_for_profile_proposal"
        if automatic_density_profile.get("profile_activatable"):
            notes = [
                "RadonPy records with high structure similarity and compatible melt conditions were selected automatically.",
                "The selected records can be passed to deterministic profile proposal before activation.",
            ]
        else:
            notes = ["Database comparison returned structurally matched reference records; pass them to deterministic profile proposal before activation."]
    else:
        computed_value = normalized_context.get("raw_context", {}).get("computed_value")
        if computed_value is None:
            computed_value = normalized_context.get("raw_context", {}).get("value")
        similarity_transfer = discover_similarity_transfer(
            normalized_context,
            prop,
            property_records,
            computed_value=computed_value,
            min_similarity=min_similarity,
            high_similarity=high_similarity,
        )
        state_rejections = [
            row
            for row in excluded
            if row.get("reason") in {"unsupported_target_state", "reference_state_missing", "reference_state_not_melt"}
        ]
        if similarity_transfer.get("status") == "similarity_transfer_candidate" and bool(
            similarity_transfer.get("machine_activation_eligible", similarity_transfer.get("machine_freeze_eligible"))
        ):
            readiness = "ready_for_similarity_transfer_profile_proposal"
            notes = [
                "No direct source matched the target polymer/property context.",
                "At least two independent high-similarity references passed the chemical-fingerprint, condition, phase and provenance activation gate.",
            ]
        elif similarity_transfer.get("status") == "similarity_transfer_candidate":
            readiness = "similarity_warn"
            notes = [
                "A deterministic similar-structure analogue is available, but it does not yet support automatic activation.",
                "Return rule_profile_missing as WARN and send the case to audit because the structure, condition, phase, force-field or provenance checks are incomplete.",
            ]
        elif state_rejections:
            readiness = "blocked_no_compatible_sources"
            notes = ["Registered sources matched the polymer but not the requested state/conditions."]
        elif similarity_transfer.get("status") == "condition_mismatch":
            readiness = "blocked_similarity_condition_mismatch"
            notes = [
                "A similar reference was found, but temperature, state, phase, force-field or engine compatibility failed.",
                "Do not activate a transferred rule_profile; return WARN and send the case to audit.",
            ]
        elif similarity_transfer.get("status") == "similarity_distance_too_large":
            readiness = "blocked_similarity_distance_too_large"
            notes = [
                "Registered sources did not directly match the polymer/property context.",
                "The best similar-structure reference failed the registered minimum-similarity threshold.",
            ]
        else:
            readiness = "blocked_no_sources"
            notes = ["No registered database records matched the polymer/property context; do not guess thresholds."]

    return {
        "status": "ok",
        "property": prop,
        "context": _json_safe(normalized_context),
        "query_plan": _json_safe(query_plan),
        "source_count": len(matches),
        "reference_records": _json_safe(matches),
        "source_evidence": _json_safe(matches),
        "excluded_source_count": len(excluded),
        "excluded_sources": _json_safe(excluded),
        "provider_status": _json_safe(provider_status),
        "automatic_density_profile": _json_safe(automatic_density_profile),
        "similarity_transfer": _json_safe(similarity_transfer),
        "activation_readiness": readiness,
        "freeze_readiness": "profile_unfreezable_similarity_warn" if readiness == "similarity_warn" else readiness,
        "warn_source": rule_profile_missing_warn_source(prop),
        "rule_verdict": None if readiness.startswith("ready_for_") else "WARN",
        "audit_required": not readiness.startswith("ready_for_"),
        "automation_level": rule_profile_automation_policy(prop)["automation_level"],
        "notes": notes,
    }


_register_builtin_rule_evidence_providers()


__all__ = [
    "discover_rule_evidence",
    "discover_radonpy_density_screening_pack",
    "normalize_discovery_context",
    "register_rule_evidence_provider",
    "unregister_rule_evidence_provider",
    "list_rule_evidence_providers",
    "load_rule_provider_plugins",
]
