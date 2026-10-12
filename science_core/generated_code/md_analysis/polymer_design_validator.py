#!/usr/bin/env python3
"""Pluggable polymer design validation for MCP-facing MD planning.

This module validates agent-proposed polymer candidates before they can be
translated into registered builder specifications. Built-in paths cover repeat-
unit polymers, compact template-based architectures and explicit short-chain
molecular graphs. The validator is deliberately conservative: unsupported
elements or builders are surfaced rather than guessed, and chemistry checks
remain limited to structure, topology, and builder compatibility.
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

SCRIPT_PATH = Path(__file__).resolve()
MD_ANALYSIS_DIR = SCRIPT_PATH.parent
if str(MD_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(MD_ANALYSIS_DIR))

from plugin_manifest_loader import load_symbol_from_spec  # noqa: E402
from polymer_template_builder import normalize_template_plan  # noqa: E402
from polymer_runtime_registry import (  # noqa: E402
    assess_preparation_capability,
    canonical_force_field,
    list_runtime_capabilities,
    preferred_builder_backend,
    preferred_force_field,
)


GENERATED_CODE_DIR = SCRIPT_PATH.parents[1]
EMC_BUILDER_DIR = GENERATED_CODE_DIR / "emc_alpha_olefin_builder"
if str(EMC_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(EMC_BUILDER_DIR))

try:  # pragma: no cover - exercised indirectly when the builder is present.
    from build_alpha_olefin_library import (  # type: ignore
        branch_positions_for_pattern as _builder_branch_positions_for_pattern,
        monomer_smiles as _builder_linear_monomer_smiles,
        pe_lcb_macro_smiles as _builder_pe_lcb_macro_smiles,
    )
except Exception:  # pragma: no cover - fallback keeps the validator importable.
    _builder_branch_positions_for_pattern = None
    _builder_linear_monomer_smiles = None
    _builder_pe_lcb_macro_smiles = None

try:  # Optional dependency.  The validator must degrade cleanly without RDKit.
    from rdkit import Chem  # type: ignore
except Exception:  # pragma: no cover - depends on local environment.
    Chem = None  # type: ignore


_RUNTIME_CAPABILITIES = list_runtime_capabilities()
ALLOWED_BACKENDS = set(_RUNTIME_CAPABILITIES["builders"])
ALLOWED_FORCE_FIELDS = set(_RUNTIME_CAPABILITIES["force_fields"])
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
COMMON_ORGANIC_ELEMENTS = frozenset({"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"})
ALLOWED_BRANCH_POINTS = {1, 2, 3, 4, 5}
ALLOWED_BRANCH_CARBONS = {3, 4, 5, 8, 12, 16, 20}
PLUGIN_DIR = SCRIPT_PATH.parent / "plugins"
BUILTIN_ADAPTER_MANIFEST = PLUGIN_DIR / "builtin_polymer_adapters.json"
FALLBACK_PATTERNS_BY_BP: dict[int, set[str]] = {
    1: {"mid", "near-end", "near-start", "far-end"},
    2: {"uniform", "two-ends", "H-asymmetric", "clustered", "dumbbell-far", "mid-pair"},
    3: {"uniform", "two-ends+mid", "pompom-dense", "clustered", "three-point", "end-cluster"},
    4: {"comb-uniform-4", "pompom-4", "cluster-4", "gradient"},
    5: {"comb-uniform-5", "pompom-5", "cluster-5", "gradient-5"},
}
SPECIAL_ARCHITECTURES_REQUIRING_REGISTERED_BUILDER = frozenset(
    {
        "cyclic_polymer",
        "star_polymer",
        "network_polymer",
        "hyperbranched_polymer",
        "dendrimer",
    }
)


@dataclass
class DesignCheckContext:
    """Mutable state shared by every declarative molecular-design check."""

    candidate: dict[str, Any]
    normalized: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    existing_features: Any = None
    trace: list[dict[str, str]] = field(default_factory=list)

    def record(self, rule_id: str, status: str) -> None:
        self.trace.append({"rule_id": rule_id, "status": status})


class DesignCheckRule:
    """One reusable check in an architecture profile."""

    rule_id: str

    def apply(self, context: DesignCheckContext) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class IntegerFieldRule(DesignCheckRule):
    rule_id: str
    field_name: str
    output_name: str | None = None
    minimum: int | None = None
    allowed_values: frozenset[int] = frozenset()
    fallback_fields: tuple[str, ...] = ()
    default: int | None = None

    def apply(self, context: DesignCheckContext) -> None:
        raw = context.candidate.get(self.field_name)
        if raw in (None, ""):
            for fallback in self.fallback_fields:
                raw = context.candidate.get(fallback, context.normalized.get(fallback))
                if raw not in (None, ""):
                    break
        if raw in (None, ""):
            raw = self.default
        if raw in (None, ""):
            context.errors.append(f"missing required field: {self.field_name}")
            context.record(self.rule_id, "error")
            return
        try:
            value = int(raw)
        except Exception:
            context.errors.append(f"{self.field_name} must be an integer")
            context.record(self.rule_id, "error")
            return
        output_name = self.output_name or self.field_name
        context.normalized[output_name] = value
        if self.minimum is not None and value < self.minimum:
            context.errors.append(f"{self.field_name} must be >= {self.minimum}")
            context.record(self.rule_id, "error")
            return
        if self.allowed_values and value not in self.allowed_values:
            context.errors.append(
                f"{self.field_name} must be one of {sorted(self.allowed_values)}"
            )
            context.record(self.rule_id, "error")
            return
        context.record(self.rule_id, "pass")


@dataclass(frozen=True)
class StringFieldRule(DesignCheckRule):
    rule_id: str
    field_name: str
    output_name: str | None = None

    def apply(self, context: DesignCheckContext) -> None:
        value = str(context.candidate.get(self.field_name, "") or "").strip()
        if not value:
            context.errors.append(f"missing required field: {self.field_name}")
            context.record(self.rule_id, "error")
            return
        context.normalized[self.output_name or self.field_name] = value
        context.record(self.rule_id, "pass")


@dataclass(frozen=True)
class DerivedFieldsRule(DesignCheckRule):
    rule_id: str
    dependencies: tuple[str, ...]
    calculator: Callable[[dict[str, Any]], dict[str, Any]]

    def apply(self, context: DesignCheckContext) -> None:
        if any(context.normalized.get(name) is None for name in self.dependencies):
            context.record(self.rule_id, "skipped")
            return
        try:
            context.normalized.update(self.calculator(context.normalized))
        except Exception as exc:
            context.errors.append(f"{self.rule_id} could not be evaluated: {exc}")
            context.record(self.rule_id, "error")
            return
        context.record(self.rule_id, "pass")


@dataclass(frozen=True)
class ConstraintRule(DesignCheckRule):
    rule_id: str
    dependencies: tuple[str, ...]
    predicate: Callable[[dict[str, Any]], bool]
    message: str
    severity: str = "error"

    def apply(self, context: DesignCheckContext) -> None:
        if any(context.normalized.get(name) is None for name in self.dependencies):
            context.record(self.rule_id, "skipped")
            return
        if self.predicate(context.normalized):
            context.record(self.rule_id, "pass")
            return
        target = context.warnings if self.severity == "warning" else context.errors
        target.append(self.message)
        context.record(self.rule_id, self.severity)


@dataclass(frozen=True)
class KeyedAllowedValueRule(DesignCheckRule):
    rule_id: str
    key_field: str
    value_field: str
    values_provider: Callable[[Any], dict[int, set[str]]]
    unknown_message: str
    incompatible_message: str

    def apply(self, context: DesignCheckContext) -> None:
        key = context.normalized.get(self.key_field)
        value = context.normalized.get(self.value_field)
        if key is None or value is None:
            context.record(self.rule_id, "skipped")
            return
        values_by_key = self.values_provider(context.existing_features)
        all_values = set().union(*values_by_key.values()) if values_by_key else set()
        if value not in all_values:
            context.errors.append(self.unknown_message)
            context.record(self.rule_id, "error")
            return
        allowed = values_by_key.get(int(key), set())
        if allowed and value not in allowed:
            context.errors.append(self.incompatible_message)
            context.record(self.rule_id, "error")
            return
        context.record(self.rule_id, "pass")


@dataclass(frozen=True)
class UniqueTupleRule(DesignCheckRule):
    rule_id: str
    fields: tuple[str, ...]
    existing_keys_provider: Callable[[Any], set[tuple[Any, ...]]]
    message: str

    def apply(self, context: DesignCheckContext) -> None:
        values = tuple(context.normalized.get(name) for name in self.fields)
        if any(value is None for value in values):
            context.record(self.rule_id, "skipped")
            return
        if values in self.existing_keys_provider(context.existing_features):
            context.errors.append(self.message)
            context.record(self.rule_id, "error")
            return
        context.record(self.rule_id, "pass")


@dataclass(frozen=True)
class DesignCheckProfile:
    profile_id: str
    rules: tuple[DesignCheckRule, ...]


def run_design_check_profile(
    profile: DesignCheckProfile,
    candidate: dict[str, Any],
    normalized: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    existing_features: Any = None,
) -> None:
    """Run one common validation pipeline using architecture data as input."""

    context = DesignCheckContext(
        candidate=candidate,
        normalized=normalized,
        errors=errors,
        warnings=warnings,
        existing_features=existing_features,
    )
    for rule in profile.rules:
        rule.apply(context)
    normalized["design_check_profile"] = profile.profile_id
    normalized["design_check_trace"] = context.trace


class PolymerAdapter:
    """Contract for a registered polymer-family construction adapter."""

    adapter_id: str = ""
    architectures: frozenset[str] = frozenset()
    polymer_families: frozenset[str] = frozenset()
    required_fields: frozenset[str] = frozenset()
    supported_force_fields: frozenset[str] = frozenset()
    supported_builder_backends: frozenset[str] = frozenset()
    design_check_profile: DesignCheckProfile | None = None

    def validate(
        self,
        candidate: dict[str, Any],
        normalized: dict[str, Any],
        errors: list[str],
        warnings: list[str],
        *,
        existing_features: Any = None,
    ) -> dict[str, Any] | None:
        if self.design_check_profile is None:
            raise NotImplementedError
        run_design_check_profile(
            self.design_check_profile,
            candidate,
            normalized,
            errors,
            warnings,
            existing_features=existing_features,
        )
        if errors:
            return None
        return self.build_spec(candidate, normalized, errors, warnings)

    def build_spec(
        self,
        candidate: dict[str, Any],
        normalized: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        raise NotImplementedError


_ADAPTERS_BY_ID: dict[str, PolymerAdapter] = {}
_ADAPTERS_BY_ARCHITECTURE: dict[str, PolymerAdapter] = {}


def _normalized_force_field(value: Any) -> str:
    return str(value or "").strip().lower()


def register_adapter(adapter: PolymerAdapter, *, replace: bool = False) -> PolymerAdapter:
    adapter_id = str(getattr(adapter, "adapter_id", "") or "").strip()
    if not adapter_id:
        raise ValueError("adapter_id is required")
    if not adapter.architectures:
        raise ValueError(f"adapter {adapter_id} must declare at least one architecture")
    if not adapter.supported_builder_backends:
        raise ValueError(f"adapter {adapter_id} must declare supported builder backends")
    if not adapter.supported_force_fields:
        raise ValueError(f"adapter {adapter_id} must declare supported force fields")

    if adapter_id in _ADAPTERS_BY_ID and not replace:
        raise ValueError(f"adapter_id already registered: {adapter_id}")
    for architecture in adapter.architectures:
        owner = _ADAPTERS_BY_ARCHITECTURE.get(str(architecture))
        if owner is not None and owner.adapter_id != adapter_id and not replace:
            raise ValueError(f"architecture already registered: {architecture}")

    if replace:
        unregister_adapter(adapter_id)
        for architecture, owner in list(_ADAPTERS_BY_ARCHITECTURE.items()):
            if architecture in adapter.architectures or owner.adapter_id == adapter_id:
                _ADAPTERS_BY_ARCHITECTURE.pop(architecture, None)

    _ADAPTERS_BY_ID[adapter_id] = adapter
    for architecture in adapter.architectures:
        _ADAPTERS_BY_ARCHITECTURE[str(architecture)] = adapter
    return adapter


def unregister_adapter(adapter_id: str) -> None:
    adapter = _ADAPTERS_BY_ID.pop(str(adapter_id), None)
    if adapter is None:
        return
    for architecture, owner in list(_ADAPTERS_BY_ARCHITECTURE.items()):
        if owner.adapter_id == adapter.adapter_id:
            _ADAPTERS_BY_ARCHITECTURE.pop(architecture, None)


def get_adapter(architecture: str, polymer_family: str | None = None) -> PolymerAdapter | None:
    adapter = _ADAPTERS_BY_ARCHITECTURE.get(str(architecture or ""))
    if adapter is None:
        return None
    families = {str(value) for value in adapter.polymer_families}
    if polymer_family and families and "*" not in families and str(polymer_family) not in families:
        return None
    return adapter


def list_registered_adapters() -> dict[str, dict[str, Any]]:
    return {
        adapter_id: {
            "adapter_id": adapter.adapter_id,
            "architectures": sorted(adapter.architectures),
            "polymer_families": sorted(adapter.polymer_families),
            "required_fields": sorted(adapter.required_fields),
            "supported_force_fields": sorted(adapter.supported_force_fields),
            "supported_builder_backends": sorted(adapter.supported_builder_backends),
        }
        for adapter_id, adapter in sorted(_ADAPTERS_BY_ID.items())
    }


def _read_plugin_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_adapter_plugins(paths: Iterable[str | Path]) -> list[str]:
    loaded: list[str] = []
    for raw_path in paths:
        manifest_path = Path(raw_path)
        manifest = _read_plugin_manifest(manifest_path)
        for spec in manifest.get("polymer_adapters", []) or []:
            adapter_cls = load_symbol_from_spec(dict(spec), base_dir=manifest_path.parent)
            adapter = adapter_cls()
            register_adapter(adapter, replace=bool(spec.get("replace", True)))
            loaded.append(adapter.adapter_id)
    return loaded


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _canonical_force_field(value: Any) -> str:
    return canonical_force_field(value)


def _structure_summary(smiles: str) -> dict[str, Any]:
    text = str(smiles or "").strip()
    if not text:
        return {
            "status": "missing",
            "elements": [],
            "carbon_count": 0,
            "atom_count": 0,
            "fragment_count": 0,
            "dummy_atom_count": 0,
            "canonical_smiles": "",
            "contains_metal": False,
        }
    if Chem is not None:
        try:
            molecule = Chem.MolFromSmiles(text)
        except Exception:
            molecule = None
        if molecule is not None:
            elements = sorted(
                {
                    atom.GetSymbol()
                    for atom in molecule.GetAtoms()
                    if int(atom.GetAtomicNum()) > 0
                }
            )
            return {
                "status": "parsed",
                "elements": elements,
                "carbon_count": sum(atom.GetSymbol() == "C" for atom in molecule.GetAtoms()),
                "atom_count": sum(int(atom.GetAtomicNum()) > 0 for atom in molecule.GetAtoms()),
                "fragment_count": len(Chem.GetMolFrags(molecule)),
                "dummy_atom_count": sum(
                    int(atom.GetAtomicNum()) == 0 for atom in molecule.GetAtoms()
                ),
                "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
                "contains_metal": any(
                    atom.GetAtomicNum() > 20
                    and atom.GetSymbol() not in {"Br", "I"}
                    for atom in molecule.GetAtoms()
                ),
            }

    tokens = re.findall(r"Cl|Br|[A-Z][a-z]?|[cnops]", text)
    aromatic = {"c": "C", "n": "N", "o": "O", "p": "P", "s": "S"}
    elements = [aromatic.get(token, token) for token in tokens]
    return {
        "status": "syntax_only",
        "elements": sorted(set(elements)),
        "carbon_count": sum(element == "C" for element in elements),
        "atom_count": len(elements),
        "fragment_count": text.count(".") + 1,
        "dummy_atom_count": text.count("*"),
        "canonical_smiles": text,
        "contains_metal": any(element not in COMMON_ORGANIC_ELEMENTS for element in elements),
    }


def _explicit_polymer_smiles(candidate: dict[str, Any]) -> str:
    for field_name in ("polymer_smiles", "chain_smiles", "molecule_smiles"):
        value = str(candidate.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _template_structure_summary(candidate: dict[str, Any]) -> dict[str, Any] | None:
    templates = list(candidate.get("repeat_unit_templates") or [])
    templates.extend(candidate.get("branch_templates") or [])
    if not templates:
        return None
    summaries = [
        _structure_summary(str(item.get("smiles") or ""))
        for item in templates
        if isinstance(item, dict)
    ]
    if not summaries:
        return None
    return {
        "status": "template_set",
        "elements": sorted({element for item in summaries for element in item.get("elements", [])}),
        "carbon_count": sum(int(item.get("carbon_count") or 0) for item in summaries),
        "atom_count": sum(int(item.get("atom_count") or 0) for item in summaries),
        "fragment_count": sum(int(item.get("fragment_count") or 0) for item in summaries),
        "dummy_atom_count": sum(int(item.get("dummy_atom_count") or 0) for item in summaries),
        "canonical_smiles": "",
        "contains_metal": any(bool(item.get("contains_metal")) for item in summaries),
        "template_count": len(summaries),
    }


def _automatic_candidate_id(candidate: dict[str, Any]) -> str:
    seed = {
        "polymer_family": candidate.get("polymer_family"),
        "architecture": candidate.get("architecture"),
        "polymer_smiles": _explicit_polymer_smiles(candidate),
        "repeat_unit_smiles": candidate.get("repeat_unit_smiles"),
        "monomer_smiles": candidate.get("monomer_smiles"),
        "repeat_unit_templates": candidate.get("repeat_unit_templates"),
        "sequence": candidate.get("sequence"),
        "branch_templates": candidate.get("branch_templates"),
        "branch_rules": candidate.get("branch_rules"),
        "branch_points": candidate.get("branch_points"),
        "branch_carbons": candidate.get("branch_carbons"),
        "sidechain_carbons": candidate.get("sidechain_carbons"),
    }
    digest = hashlib.sha256(
        json.dumps(seed, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()[:12]
    return f"auto_polymer_{digest}"


def _infer_architecture(candidate: dict[str, Any]) -> tuple[str, str]:
    explicit = str(candidate.get("architecture") or "").strip()
    if explicit:
        if explicit in SPECIAL_ARCHITECTURES_REQUIRING_REGISTERED_BUILDER:
            return explicit, "recognized_special_architecture"
        if candidate.get("repeat_unit_templates") and candidate.get("sequence"):
            return "templated_polymer_chain", "compact_template_description"
        if _explicit_polymer_smiles(candidate):
            return "explicit_polymer_chain", "explicit_full_chain_structure"
        if explicit in _ADAPTERS_BY_ARCHITECTURE:
            return explicit, "user"
        return explicit, "user"
    if candidate.get("repeat_unit_templates") and candidate.get("sequence"):
        return "templated_polymer_chain", "compact_template_description"
    if _explicit_polymer_smiles(candidate):
        return "explicit_polymer_chain", "explicit_full_chain_structure"
    if candidate.get("branch_points") not in (None, "") and candidate.get("branch_carbons") not in (None, ""):
        return "pe_lcb", "branch_fields"
    if candidate.get("sidechain_carbons") not in (None, ""):
        return "linear_alpha_olefin", "sidechain_field"
    if candidate.get("repeat_unit_smiles") or candidate.get("monomer_smiles"):
        return "linear_homopolymer", "repeat_unit_structure"
    return "", "insufficient_structure"


def _preferred_force_field(
    adapter: PolymerAdapter | None,
    builder_backend: str,
    elements: Iterable[str] | None = None,
) -> str:
    if adapter is None:
        return ""
    registered = preferred_force_field(builder_backend, elements)
    adapter_force_fields = {
        _canonical_force_field(value)
        for value in adapter.supported_force_fields
        if _canonical_force_field(value)
    }
    if registered and ("*" in adapter_force_fields or registered in adapter_force_fields):
        return registered
    canonical = sorted(
        {
            _canonical_force_field(value)
            for value in adapter.supported_force_fields
            if _canonical_force_field(value) and _canonical_force_field(value) != "*"
        }
    )
    if "opls-aa" in canonical:
        return "opls-aa"
    return canonical[0] if canonical else ""


def auto_configure_polymer_candidate(candidate: dict[str, Any] | str) -> dict[str, Any]:
    """Fill builder and force-field fields from registered deterministic rules.

    The selector only fills missing fields. Explicit unsupported user choices
    remain visible to the ordinary validator and are rejected there.
    """

    parsed, parse_errors = _parse_candidate(candidate)
    if not parsed:
        return {
            "status": "invalid_request",
            "candidate": {},
            "errors": parse_errors,
            "automatic_setup": {
                "status": "review_required",
                "reason": "candidate_structure_missing_or_invalid",
            },
        }

    configured = dict(parsed)
    filled_fields: dict[str, Any] = {}
    declared_architecture = str(configured.get("architecture") or "").strip()
    architecture, architecture_source = _infer_architecture(configured)
    if architecture and declared_architecture and declared_architecture != architecture:
        configured["declared_architecture"] = declared_architecture
        configured["architecture"] = architecture
        filled_fields["architecture"] = architecture
    elif architecture and not configured.get("architecture"):
        configured["architecture"] = architecture
        filled_fields["architecture"] = architecture

    structure_smiles = str(
        _explicit_polymer_smiles(configured)
        or configured.get("repeat_unit_smiles")
        or configured.get("monomer_smiles")
        or ""
    ).strip()
    structure = _template_structure_summary(configured) or _structure_summary(structure_smiles)

    if architecture == "explicit_polymer_chain" and structure_smiles and not configured.get("polymer_smiles"):
        configured["polymer_smiles"] = structure_smiles
        filled_fields["polymer_smiles"] = structure_smiles

    if not configured.get("candidate_id"):
        configured["candidate_id"] = _automatic_candidate_id(configured)
        filled_fields["candidate_id"] = configured["candidate_id"]

    if not configured.get("polymer_family"):
        if architecture == "pe_lcb":
            family = "PE"
        elif architecture == "linear_alpha_olefin":
            family = "poly_alpha_olefin"
        elif set(structure["elements"]) <= {"C", "H"}:
            family = "hydrocarbon_polymer"
        else:
            family = "custom_polymer"
        configured["polymer_family"] = family
        filled_fields["polymer_family"] = family

    if configured.get("chain_count") in (None, ""):
        configured["chain_count"] = 12
        filled_fields["chain_count"] = 12

    if configured.get("target_total_carbons_per_chain") in (None, ""):
        repeat_carbons = int(structure.get("carbon_count") or 0)
        if architecture == "templated_polymer_chain":
            try:
                target_total = int(
                    normalize_template_plan(configured)["actual_total_carbons_per_chain"]
                )
            except ValueError:
                target_total = 100
        elif architecture == "explicit_polymer_chain":
            target_total = max(1, repeat_carbons)
        else:
            target_total = max(repeat_carbons, repeat_carbons * max(1, round(100 / repeat_carbons))) if repeat_carbons else 100
        configured["target_total_carbons_per_chain"] = target_total
        filled_fields["target_total_carbons_per_chain"] = target_total

    if architecture == "linear_homopolymer" and structure_smiles:
        if not configured.get("input_mode"):
            configured["input_mode"] = "repeat_unit" if parsed.get("repeat_unit_smiles") else "monomer"
            filled_fields["input_mode"] = configured["input_mode"]
        if not configured.get("repeat_unit_smiles"):
            configured["repeat_unit_smiles"] = structure_smiles
            filled_fields["repeat_unit_smiles"] = structure_smiles
        repeat_carbons = int(structure.get("carbon_count") or 0)
        if repeat_carbons > 0 and not configured.get("repeat_unit_total_carbons"):
            configured["repeat_unit_total_carbons"] = repeat_carbons
            filled_fields["repeat_unit_total_carbons"] = repeat_carbons
        if repeat_carbons > 0 and not configured.get("repeat_units_per_chain"):
            repeats = max(
                1,
                round(int(configured["target_total_carbons_per_chain"]) / repeat_carbons),
            )
            configured["repeat_units_per_chain"] = repeats
            filled_fields["repeat_units_per_chain"] = repeats

    adapter = get_adapter(architecture, str(configured.get("polymer_family") or "")) if architecture else None
    if adapter is None and architecture:
        adapter = get_adapter(architecture)

    if adapter is not None and not configured.get("builder_backend"):
        registered_backend = preferred_builder_backend(architecture)
        backend = (
            registered_backend
            if "*" in adapter.supported_builder_backends
            or registered_backend in adapter.supported_builder_backends
            else sorted(adapter.supported_builder_backends)[0]
        )
        configured["builder_backend"] = backend
        filled_fields["builder_backend"] = backend

    force_field_status = "user_supplied"
    force_field_reason = "force_field_was_provided"
    parameterization_probe_required = False
    if not configured.get("force_field"):
        elements = set(structure.get("elements") or [])
        selected_force_field = _preferred_force_field(
            adapter,
            str(configured.get("builder_backend") or ""),
            elements,
        )
        if structure.get("contains_metal") or (elements and not elements <= COMMON_ORGANIC_ELEMENTS):
            force_field_status = "review_required"
            force_field_reason = "registered_force_fields_do_not_cover_detected_elements"
        elif selected_force_field:
            configured["force_field"] = selected_force_field
            filled_fields["force_field"] = selected_force_field
            force_field_status = "selected_for_automatic_probe"
            force_field_reason = "selected_from_registered_adapter_allowlist"
        else:
            force_field_status = "review_required"
            force_field_reason = "no_registered_force_field_for_adapter"

    if configured.get("force_field"):
        selected = _canonical_force_field(configured.get("force_field"))
    else:
        selected = ""
    elements = set(structure.get("elements") or [])
    capability = assess_preparation_capability(
        polymer_family=configured.get("polymer_family"),
        architecture=architecture,
        builder_backend=configured.get("builder_backend"),
        force_field=selected,
        elements=elements,
        topology_format="gromacs_atomistic",
    )
    parameterization_probe_required = bool(capability["parameterization_probe_required"])
    scientific_qualification_status = str(capability["scientific_qualification_status"])
    if selected and parameterization_probe_required and force_field_status == "user_supplied":
        force_field_status = "user_supplied_probe_required"
        force_field_reason = "parameter_coverage_and_external_validation_must_be_verified"
    if selected and not capability["technical_support"]:
        force_field_status = "review_required"
        force_field_reason = "registered_runtime_capability_does_not_cover_candidate"

    setup_status = "ready_for_validation"
    if not architecture or adapter is None or force_field_status == "review_required":
        setup_status = "review_required"
    elif parameterization_probe_required:
        setup_status = "ready_for_automatic_parameterization_probe"

    automatic_setup = {
        "schema_version": "polymer_automatic_setup.v1",
        "status": setup_status,
        "filled_fields": filled_fields,
        "architecture_selection": {
            "architecture": architecture,
            "source": architecture_source,
            "adapter_id": adapter.adapter_id if adapter is not None else None,
        },
        "builder_selection": {
            "builder_backend": configured.get("builder_backend"),
            "source": "registered_adapter" if "builder_backend" in filled_fields else "user",
        },
        "force_field_selection": {
            "force_field": selected or None,
            "status": force_field_status,
            "reason": force_field_reason,
            "parameterization_probe_required": parameterization_probe_required,
            "scientific_qualification_status": scientific_qualification_status,
        },
        "structure_summary": structure,
        "runtime_capability": capability,
        "production_policy": {
            "builder_and_force_field_are_selected_deterministically": True,
            "remote_build_and_topology_probe_required_before_md": parameterization_probe_required,
            "unsupported_elements_go_to_review": True,
            "publication_pass_requires_force_field_qualification": True,
        },
    }
    return {
        "status": "ok",
        "candidate": configured,
        "errors": parse_errors,
        "automatic_setup": automatic_setup,
    }


def _parse_candidate(candidate: dict[str, Any] | str) -> tuple[dict[str, Any], list[str]]:
    if isinstance(candidate, dict):
        return dict(candidate), []
    duplicate_keys: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                duplicate_keys.append(str(key))
            out[str(key)] = value
        return out

    try:
        parsed = json.loads(candidate, object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        return {}, [f"candidate_json is not valid JSON: {exc.msg}"]
    if not isinstance(parsed, dict):
        return {}, ["candidate_json must decode to an object"]
    return parsed, [f"duplicate candidate key: {key}" for key in sorted(set(duplicate_keys))]


def _as_int(candidate: dict[str, Any], field: str, errors: list[str], *, minimum: int | None = None) -> int | None:
    value = candidate.get(field)
    if value is None or value == "":
        errors.append(f"missing required field: {field}")
        return None
    try:
        number = int(value)
    except Exception:
        errors.append(f"{field} must be an integer")
        return None
    if minimum is not None and number < minimum:
        errors.append(f"{field} must be >= {minimum}")
    return number


def _required_string(candidate: dict[str, Any], field: str, errors: list[str]) -> str:
    value = str(candidate.get(field, "") or "").strip()
    if not value:
        errors.append(f"missing required field: {field}")
    return value


def _existing_design_keys(existing_features: Any) -> set[tuple[int, int, str]]:
    if existing_features is None:
        return set()
    if hasattr(existing_features, "iterrows"):
        iterator: Iterable[Any] = (row for _, row in existing_features.iterrows())
    else:
        iterator = existing_features
    keys: set[tuple[int, int, str]] = set()
    for row in iterator:
        try:
            getter = row.get if hasattr(row, "get") else row.__getitem__
            keys.add(
                (
                    int(getter("branch_points")),
                    int(getter("branch_carbons")),
                    str(getter("branch_position_pattern")),
                )
            )
        except Exception:
            continue
    return keys


def _existing_explicit_polymer_keys(existing_features: Any) -> set[tuple[str]]:
    if existing_features is None:
        return set()
    if hasattr(existing_features, "iterrows"):
        iterator: Iterable[Any] = (row for _, row in existing_features.iterrows())
    else:
        iterator = existing_features
    keys: set[tuple[str]] = set()
    for row in iterator:
        smiles = ""
        for field_name in ("polymer_smiles", "chain_smiles", "molecule_smiles"):
            try:
                value = row.get(field_name) if hasattr(row, "get") else row[field_name]
            except Exception:
                value = None
            if value not in (None, ""):
                smiles = str(value).strip()
                break
        if not smiles:
            continue
        canonical = str(_structure_summary(smiles).get("canonical_smiles") or smiles)
        keys.add((canonical,))
    return keys


def _allowed_patterns_by_bp(existing_features: Any) -> dict[int, set[str]]:
    patterns = {bp: set(values) for bp, values in FALLBACK_PATTERNS_BY_BP.items()}
    if existing_features is None:
        return patterns
    if hasattr(existing_features, "iterrows"):
        iterator: Iterable[Any] = (row for _, row in existing_features.iterrows())
    else:
        iterator = existing_features
    for row in iterator:
        try:
            getter = row.get if hasattr(row, "get") else row.__getitem__
            patterns.setdefault(int(getter("branch_points")), set()).add(str(getter("branch_position_pattern")))
        except Exception:
            continue
    return patterns


def _actual_pe_lcb_carbon_budget(branch_points: int, branch_carbons: int, carbon_budget: int) -> tuple[int, int]:
    branch_total = branch_points * branch_carbons
    actual_total = carbon_budget if (carbon_budget - branch_total) % 2 == 0 else carbon_budget + 1
    backbone_repeat_units = (actual_total - branch_total) // 2
    return actual_total, backbone_repeat_units


def _fallback_even_positions(backbone_repeat_units: int, branch_points: int) -> list[int]:
    if branch_points <= 0:
        return []
    if branch_points == 1:
        return [max(2, backbone_repeat_units // 2)]
    lo = 2
    hi = max(lo, backbone_repeat_units - 1)
    span = hi - lo
    return sorted({lo + round(index * span / (branch_points - 1)) for index in range(branch_points)})


def _branch_positions(backbone_repeat_units: int, branch_points: int, pattern: str) -> list[int]:
    if _builder_branch_positions_for_pattern is not None:
        return list(_builder_branch_positions_for_pattern(backbone_repeat_units, branch_points, pattern))
    return _fallback_even_positions(backbone_repeat_units, branch_points)


def _alkyl_smiles(carbons: int) -> str:
    return "C" * max(1, int(carbons))


def _pe_lcb_macro_smiles(backbone_repeat_units: int, branch_positions: list[int], branch_carbons: int) -> str:
    if _builder_pe_lcb_macro_smiles is not None:
        return str(_builder_pe_lcb_macro_smiles(backbone_repeat_units, branch_positions, branch_carbons))
    branch_lookup = set(branch_positions)
    branch = _alkyl_smiles(branch_carbons)
    fragments = [f"C({branch})C" if index in branch_lookup else "CC" for index in range(1, backbone_repeat_units + 1)]
    return f"*{''.join(fragments)}*"


def _linear_monomer_smiles(sidechain_carbons: int) -> str:
    if _builder_linear_monomer_smiles is not None:
        return str(_builder_linear_monomer_smiles(sidechain_carbons))
    return f"*[C@H]({_alkyl_smiles(sidechain_carbons)})C*"


def _validate_smiles(smiles: str, errors: list[str], warnings: list[str], field: str) -> None:
    if not smiles:
        return
    if Chem is None:
        if not re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$/*\\.]+", smiles):
            errors.append(f"{field} SMILES failed fallback syntax validation")
        else:
            warnings.append("RDKit is unavailable; SMILES was checked only by a conservative syntax fallback")
        return
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        errors.append(f"{field} SMILES is not RDKit-parseable")


def _validate_special_architecture_candidate(
    candidate: dict[str, Any],
    parse_errors: list[str],
) -> dict[str, Any]:
    """Validate recognized polymer architectures that need a dedicated builder."""

    errors = list(parse_errors)
    warnings: list[str] = []
    candidate_id = str(candidate.get("candidate_id") or _automatic_candidate_id(candidate))
    polymer_family = str(candidate.get("polymer_family") or "").strip()
    architecture = str(candidate.get("architecture") or "").strip()
    repeat_unit_smiles = str(candidate.get("repeat_unit_smiles") or "").strip()

    if not SAFE_ID_RE.fullmatch(candidate_id):
        errors.append("candidate_id must be a filesystem-safe identifier")
    if not polymer_family:
        errors.append("missing required field: polymer_family")
    if architecture not in SPECIAL_ARCHITECTURES_REQUIRING_REGISTERED_BUILDER:
        errors.append(
            "architecture is not a recognized special polymer architecture"
        )
    if not repeat_unit_smiles:
        errors.append("missing required field: repeat_unit_smiles")
    else:
        _validate_smiles(repeat_unit_smiles, errors, warnings, "repeat_unit_smiles")

    normalized: dict[str, Any] = {
        "candidate_id": candidate_id,
        "polymer_family": polymer_family,
        "architecture": architecture,
        "repeat_unit_smiles": repeat_unit_smiles,
        "design_check_profile": "recognized_special_polymer_architecture.v1",
    }

    def positive_integer(field: str, minimum: int = 1) -> int | None:
        value = _as_int(candidate, field, errors, minimum=minimum)
        normalized[field] = value
        return value

    def bounded_number(field: str, minimum: float, maximum: float) -> float | None:
        raw = candidate.get(field)
        try:
            value = float(raw)
        except Exception:
            errors.append(f"{field} must be numeric")
            normalized[field] = None
            return None
        if not minimum <= value <= maximum:
            errors.append(f"{field} must be within {minimum}..{maximum}")
        normalized[field] = value
        return value

    if architecture == "cyclic_polymer":
        positive_integer("repeat_units_per_cycle", minimum=3)
    elif architecture == "star_polymer":
        positive_integer("arm_count", minimum=3)
        positive_integer("arm_repeat_units", minimum=1)
        core_smiles = str(candidate.get("core_smiles") or "").strip()
        normalized["core_smiles"] = core_smiles
        if not core_smiles:
            errors.append("missing required field: core_smiles")
        else:
            _validate_smiles(core_smiles, errors, warnings, "core_smiles")
    elif architecture == "network_polymer":
        bounded_number("crosslink_fraction", 0.0, 1.0)
        crosslinker_smiles = str(candidate.get("crosslinker_smiles") or "").strip()
        normalized["crosslinker_smiles"] = crosslinker_smiles
        if not crosslinker_smiles:
            errors.append("missing required field: crosslinker_smiles")
        else:
            _validate_smiles(
                crosslinker_smiles, errors, warnings, "crosslinker_smiles"
            )
    elif architecture == "hyperbranched_polymer":
        positive_integer("repeat_units_per_chain", minimum=1)
        bounded_number("degree_of_branching", 0.0, 1.0)
    elif architecture == "dendrimer":
        positive_integer("generation", minimum=1)
        positive_integer("core_functionality", minimum=3)
        core_smiles = str(candidate.get("core_smiles") or "").strip()
        normalized["core_smiles"] = core_smiles
        if not core_smiles:
            errors.append("missing required field: core_smiles")
        else:
            _validate_smiles(core_smiles, errors, warnings, "core_smiles")

    if errors:
        validator_status = "reject"
        next_action = "review"
    else:
        validator_status = "review"
        next_action = "register_specialized_builder"
        warnings.append(
            f"recognized polymer architecture requires a registered builder: {architecture}"
        )

    automatic_setup = {
        "schema_version": "polymer_automatic_setup.v1",
        "status": "review_required",
        "architecture_selection": {
            "architecture": architecture,
            "source": "recognized_special_architecture",
            "adapter_id": None,
        },
        "builder_selection": {
            "builder_backend": None,
            "source": "no_registered_specialized_builder",
        },
        "force_field_selection": {
            "force_field": candidate.get("force_field"),
            "status": "review_required",
            "reason": "specialized_builder_and_parameterization_probe_required",
            "parameterization_probe_required": False,
            "scientific_qualification_status": "unsupported_configuration",
        },
    }
    return {
        "status": "ok",
        "validator_status": validator_status,
        "errors": errors,
        "warnings": warnings,
        "normalized_candidate": _json_safe(normalized),
        "builder_spec": {},
        "automatic_setup": _json_safe(automatic_setup),
        "builder_ready": False,
        "emc_ready": False,
        "production_ready": False,
        "next_action": next_action,
    }


def _base_checks(
    candidate: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    allow_missing_force_field_for_review: bool = False,
) -> dict[str, Any]:
    candidate_id = _required_string(candidate, "candidate_id", errors)
    polymer_family = _required_string(candidate, "polymer_family", errors)
    architecture = _required_string(candidate, "architecture", errors)
    builder_backend = _required_string(candidate, "builder_backend", errors)
    if allow_missing_force_field_for_review:
        force_field = str(candidate.get("force_field") or "").strip()
    else:
        force_field = _required_string(candidate, "force_field", errors)
    chain_count = _as_int(candidate, "chain_count", errors, minimum=1)
    target_total = _as_int(candidate, "target_total_carbons_per_chain", errors, minimum=1)

    if candidate_id and not SAFE_ID_RE.fullmatch(candidate_id):
        errors.append("candidate_id must be a filesystem-safe identifier")

    smiles = str(
        _explicit_polymer_smiles(candidate)
        or candidate.get("repeat_unit_smiles")
        or candidate.get("monomer_smiles")
        or ""
    ).strip()
    if smiles:
        _validate_smiles(smiles, errors, warnings, "candidate")

    return {
        "candidate_id": candidate_id,
        "polymer_family": polymer_family,
        "architecture": architecture,
        "builder_backend": builder_backend,
        "force_field": force_field or None,
        "chain_count": chain_count,
        "target_total_carbons_per_chain": target_total,
    }


def _build_pe_lcb_spec(
    candidate: dict[str, Any],
    normalized: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    branch_points = int(normalized["branch_points"])
    branch_carbons = int(normalized["branch_carbons"])
    pattern = str(normalized["branch_position_pattern"])
    actual_total = int(normalized["actual_total_carbons_per_chain"])
    backbone_repeat_units = int(normalized["backbone_repeat_units"])
    branch_positions = _branch_positions(backbone_repeat_units, branch_points, pattern)
    monomer_smiles = _pe_lcb_macro_smiles(backbone_repeat_units, branch_positions, branch_carbons)
    return {
        "case_id": normalized["candidate_id"],
        "preset": "screen_pelcb100_sidecar",
        "builder_backend": normalized["builder_backend"],
        "polymer_family": "PE",
        "architecture": "pe_lcb",
        "monomer_smiles": monomer_smiles,
        "chain_count": normalized["chain_count"],
        "actual_total_carbons_per_chain": actual_total,
        "target_total_carbons_per_chain": normalized["target_total_carbons_per_chain"],
        "force_field": normalized["force_field"],
        "backbone_monomer": "ethylene",
        "backbone_repeat_units": backbone_repeat_units,
        "branch_points": branch_points,
        "branch_carbons": branch_carbons,
        "branch_position_pattern": pattern,
        "branch_positions": branch_positions,
        "branch_smiles": _alkyl_smiles(branch_carbons),
    }


def _build_linear_alpha_olefin_spec(
    candidate: dict[str, Any],
    normalized: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    sidechain = int(normalized["sidechain_carbons"])
    repeat_unit_total = int(normalized["repeat_unit_total_carbons"])
    repeat_units = int(normalized["repeat_units_per_chain"])
    actual_total = int(normalized["actual_total_carbons_per_chain"])
    target_total = int(normalized["target_total_carbons_per_chain"])
    user_smiles = str(candidate.get("repeat_unit_smiles") or "").strip()
    return {
        "case_id": normalized["candidate_id"],
        "preset": "linear_alpha_olefin_sidecar",
        "builder_backend": normalized["builder_backend"],
        "polymer_family": normalized["polymer_family"],
        "architecture": "linear_alpha_olefin",
        "monomer_smiles": _linear_monomer_smiles(sidechain),
        "repeat_unit_smiles": user_smiles or None,
        "chain_count": normalized["chain_count"],
        "target_total_carbons_per_chain": target_total,
        "actual_total_carbons_per_chain": actual_total,
        "repeat_unit_total_carbons": repeat_unit_total,
        "repeat_units_per_chain": repeat_units,
        "sidechain_carbons": sidechain,
        "force_field": normalized["force_field"],
    }


def _build_linear_homopolymer_spec(
    candidate: dict[str, Any],
    normalized: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    repeat_smiles = str(normalized["repeat_unit_smiles"])
    repeat_unit_total = int(normalized["repeat_unit_total_carbons"])
    repeat_units = int(normalized["repeat_units_per_chain"])
    actual_total = int(normalized["actual_total_carbons_per_chain"])
    target_total = int(normalized.get("target_total_carbons_per_chain") or actual_total)
    return {
        "case_id": normalized["candidate_id"],
        "preset": "linear_homopolymer_sidecar",
        "builder_backend": normalized["builder_backend"],
        "polymer_family": normalized["polymer_family"],
        "architecture": "linear_homopolymer",
        "input_mode": normalized.get("input_mode", "repeat_unit"),
        "monomer_smiles": repeat_smiles,
        "repeat_unit_smiles": repeat_smiles,
        "chain_count": normalized["chain_count"],
        "target_total_carbons_per_chain": target_total,
        "actual_total_carbons_per_chain": actual_total,
        "repeat_unit_total_carbons": repeat_unit_total,
        "repeat_units_per_chain": repeat_units,
        "force_field": normalized["force_field"],
    }


def _build_explicit_polymer_chain_spec(
    candidate: dict[str, Any],
    normalized: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    polymer_smiles = str(normalized["polymer_smiles"])
    actual_total = int(normalized["actual_total_carbons_per_chain"])
    target_total = int(normalized.get("target_total_carbons_per_chain") or actual_total)
    return {
        "case_id": normalized["candidate_id"],
        "preset": "explicit_polymer_chain_sidecar",
        "builder_backend": normalized["builder_backend"],
        "polymer_family": normalized["polymer_family"],
        "architecture": "explicit_polymer_chain",
        "declared_architecture": candidate.get("declared_architecture")
        or candidate.get("architecture"),
        "monomer_smiles": polymer_smiles,
        "polymer_smiles": polymer_smiles,
        "chain_count": normalized["chain_count"],
        "target_total_carbons_per_chain": target_total,
        "actual_total_carbons_per_chain": actual_total,
        "repeat_unit_total_carbons": max(1, actual_total),
        "repeat_units_per_chain": 1,
        "force_field": normalized["force_field"],
        "structure_elements": normalized.get("structure_elements", []),
        "structure_atom_count": normalized.get("structure_atom_count"),
    }


def _build_templated_polymer_chain_spec(
    candidate: dict[str, Any],
    normalized: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    target_total = int(
        normalized.get("target_total_carbons_per_chain")
        or normalized["actual_total_carbons_per_chain"]
    )
    return {
        "case_id": normalized["candidate_id"],
        "preset": "templated_polymer_chain_sidecar",
        "builder_backend": normalized["builder_backend"],
        "polymer_family": normalized["polymer_family"],
        "architecture": "templated_polymer_chain",
        "declared_architecture": candidate.get("declared_architecture")
        or candidate.get("architecture"),
        "chain_count": normalized["chain_count"],
        "target_total_carbons_per_chain": target_total,
        "actual_total_carbons_per_chain": normalized["actual_total_carbons_per_chain"],
        "repeat_units_per_chain": normalized["repeat_units_per_chain"],
        "estimated_real_atoms_per_chain": normalized["estimated_real_atoms_per_chain"],
        "force_field": normalized["force_field"],
        "structure_elements": normalized["structure_elements"],
        "template_plan": {
            "schema_version": normalized["template_schema_version"],
            "repeat_unit_templates": normalized["repeat_unit_templates"],
            "sequence": normalized["sequence"],
            "branch_templates": normalized["branch_templates"],
            "branch_rules": normalized["branch_rules"],
            "repeat_units_per_chain": normalized["repeat_units_per_chain"],
            "actual_total_carbons_per_chain": normalized["actual_total_carbons_per_chain"],
            "estimated_real_atoms_per_chain": normalized["estimated_real_atoms_per_chain"],
            "structure_elements": normalized["structure_elements"],
            "template_design_hash": normalized["template_design_hash"],
        },
    }


def _derive_pe_lcb_carbon_fields(values: dict[str, Any]) -> dict[str, Any]:
    branch_points = int(values["branch_points"])
    branch_carbons = int(values["branch_carbons"])
    carbon_budget = int(values["requested_carbon_budget"])
    actual_total, backbone_repeat_units = _actual_pe_lcb_carbon_budget(
        branch_points,
        branch_carbons,
        carbon_budget,
    )
    return {
        "polymer_family": "PE",
        "architecture": "pe_lcb",
        "branch_total_carbons": branch_points * branch_carbons,
        "actual_total_carbons_per_chain": actual_total,
        "backbone_repeat_units": backbone_repeat_units,
    }


def _derive_alpha_olefin_chain_fields(values: dict[str, Any]) -> dict[str, Any]:
    sidechain = int(values["sidechain_carbons"])
    target_total = int(values["target_total_carbons_per_chain"])
    repeat_unit_total = sidechain + 2
    repeat_units = max(1, int((target_total / repeat_unit_total) + 0.5))
    return {
        "architecture": "linear_alpha_olefin",
        "repeat_unit_total_carbons": repeat_unit_total,
        "repeat_units_per_chain": repeat_units,
        "actual_total_carbons_per_chain": repeat_units * repeat_unit_total,
    }


def _derive_linear_chain_fields(values: dict[str, Any]) -> dict[str, Any]:
    repeat_unit_total = int(values["repeat_unit_total_carbons"])
    repeat_units = int(values["repeat_units_per_chain"])
    structure = _structure_summary(str(values["repeat_unit_smiles"]))
    return {
        "architecture": "linear_homopolymer",
        "actual_total_carbons_per_chain": repeat_unit_total * repeat_units,
        "input_mode": str(values.get("input_mode") or "repeat_unit"),
        "repeat_unit_dummy_atom_count": int(structure.get("dummy_atom_count") or 0),
        "repeat_unit_fragment_count": int(structure.get("fragment_count") or 0),
    }


def _derive_explicit_polymer_fields(values: dict[str, Any]) -> dict[str, Any]:
    summary = _structure_summary(str(values["polymer_smiles"]))
    return {
        "architecture": "explicit_polymer_chain",
        "canonical_polymer_smiles": summary.get("canonical_smiles") or values["polymer_smiles"],
        "actual_total_carbons_per_chain": int(summary.get("carbon_count") or 0),
        "structure_atom_count": int(summary.get("atom_count") or 0),
        "structure_fragment_count": int(summary.get("fragment_count") or 0),
        "structure_dummy_atom_count": int(summary.get("dummy_atom_count") or 0),
        "structure_elements": list(summary.get("elements") or []),
    }


PE_LCB_DESIGN_CHECKS = DesignCheckProfile(
    profile_id="branched_polymer_design.v1",
    rules=(
        IntegerFieldRule(
            "branch_count",
            "branch_points",
            minimum=1,
            allowed_values=frozenset(ALLOWED_BRANCH_POINTS),
        ),
        IntegerFieldRule(
            "branch_length",
            "branch_carbons",
            minimum=1,
            allowed_values=frozenset(ALLOWED_BRANCH_CARBONS),
        ),
        StringFieldRule("branch_placement", "branch_position_pattern"),
        IntegerFieldRule(
            "chain_carbon_budget",
            "carbon_budget",
            output_name="requested_carbon_budget",
            minimum=1,
            fallback_fields=("target_total_carbons_per_chain",),
            default=100,
        ),
        ConstraintRule(
            "study_design_range",
            ("requested_carbon_budget",),
            lambda values: int(values["requested_carbon_budget"]) in {100, 101},
            "carbon_budget is outside the C100/C101 PE-LCB study range",
            severity="warning",
        ),
        DerivedFieldsRule(
            "chain_composition",
            ("branch_points", "branch_carbons", "requested_carbon_budget"),
            _derive_pe_lcb_carbon_fields,
        ),
        ConstraintRule(
            "positive_backbone",
            ("backbone_repeat_units",),
            lambda values: int(values["backbone_repeat_units"]) > 0,
            "carbon budget leaves no valid backbone repeat units",
        ),
        ConstraintRule(
            "branch_spacing",
            ("backbone_repeat_units", "branch_points"),
            lambda values: int(values["backbone_repeat_units"])
            >= max(6, int(values["branch_points"]) * 2),
            "backbone is too short for the requested branch spacing",
        ),
        ConstraintRule(
            "branch_fraction_boundary",
            ("branch_total_carbons", "actual_total_carbons_per_chain"),
            lambda values: float(values["branch_total_carbons"])
            < 0.75 * float(values["actual_total_carbons_per_chain"]),
            "branch-carbon allocation is star-like/boundary and should be reviewed before submission",
            severity="warning",
        ),
        KeyedAllowedValueRule(
            "branch_placement_compatibility",
            "branch_points",
            "branch_position_pattern",
            _allowed_patterns_by_bp,
            "branch_position_pattern is not in the registered placement grammar",
            "branch_position_pattern is not supported for this branch_points value",
        ),
        UniqueTupleRule(
            "unique_design",
            ("branch_points", "branch_carbons", "branch_position_pattern"),
            _existing_design_keys,
            "duplicate existing PE-LCB100 design key",
        ),
    ),
)


LINEAR_ALPHA_OLEFIN_DESIGN_CHECKS = DesignCheckProfile(
    profile_id="substituted_linear_polymer_design.v1",
    rules=(
        IntegerFieldRule("sidechain_length", "sidechain_carbons", minimum=1),
        DerivedFieldsRule(
            "linear_chain_composition",
            ("sidechain_carbons", "target_total_carbons_per_chain"),
            _derive_alpha_olefin_chain_fields,
        ),
    ),
)


LINEAR_HOMOPOLYMER_DESIGN_CHECKS = DesignCheckProfile(
    profile_id="linear_repeat_unit_design.v1",
    rules=(
        StringFieldRule("repeat_unit", "repeat_unit_smiles"),
        StringFieldRule("linear_input_mode_field", "input_mode"),
        IntegerFieldRule("repeat_unit_carbon_count", "repeat_unit_total_carbons", minimum=1),
        IntegerFieldRule("repeat_unit_count", "repeat_units_per_chain", minimum=1),
        DerivedFieldsRule(
            "linear_chain_composition",
            ("repeat_unit_smiles", "repeat_unit_total_carbons", "repeat_units_per_chain", "input_mode"),
            _derive_linear_chain_fields,
        ),
        ConstraintRule(
            "repeat_unit_connection_points",
            ("input_mode", "repeat_unit_dummy_atom_count"),
            lambda values: values["input_mode"] == "monomer" or int(values["repeat_unit_dummy_atom_count"]) == 2,
            "repeat_unit_smiles must contain exactly two '*' connection points",
        ),
        ConstraintRule(
            "linear_input_mode",
            ("input_mode",),
            lambda values: values["input_mode"] in {"repeat_unit", "monomer"},
            "linear input_mode must be repeat_unit or monomer",
        ),
        ConstraintRule(
            "monomer_connection_points",
            ("input_mode", "repeat_unit_dummy_atom_count"),
            lambda values: values["input_mode"] != "monomer" or int(values["repeat_unit_dummy_atom_count"]) == 0,
            "monomer input must be a complete structure with zero '*' connection points",
        ),
        ConstraintRule(
            "single_repeat_unit_fragment",
            ("repeat_unit_fragment_count",),
            lambda values: int(values["repeat_unit_fragment_count"]) == 1,
            "repeat_unit_smiles must describe one connected repeat unit",
        ),
        ConstraintRule(
            "target_chain_size",
            ("actual_total_carbons_per_chain", "target_total_carbons_per_chain"),
            lambda values: abs(
                int(values["actual_total_carbons_per_chain"])
                - int(values["target_total_carbons_per_chain"])
            )
            <= max(2, int(0.1 * int(values["target_total_carbons_per_chain"]))),
            "actual_total_carbons_per_chain differs from target by both more than 2 carbons and more than 10%",
            severity="warning",
        ),
    ),
)


EXPLICIT_POLYMER_CHAIN_DESIGN_CHECKS = DesignCheckProfile(
    profile_id="explicit_polymer_graph.v1",
    rules=(
        StringFieldRule("polymer_structure", "polymer_smiles"),
        DerivedFieldsRule(
            "polymer_graph_summary",
            ("polymer_smiles",),
            _derive_explicit_polymer_fields,
        ),
        ConstraintRule(
            "single_connected_chain",
            ("structure_fragment_count",),
            lambda values: int(values["structure_fragment_count"]) == 1,
            "polymer_smiles must describe one connected polymer chain",
        ),
        ConstraintRule(
            "nonempty_polymer_graph",
            ("structure_atom_count",),
            lambda values: int(values["structure_atom_count"]) >= 2,
            "polymer_smiles must contain at least two atoms",
        ),
        ConstraintRule(
            "closed_polymer_chain",
            ("structure_dummy_atom_count",),
            lambda values: int(values["structure_dummy_atom_count"]) == 0,
            "explicit polymer chains cannot contain open connection markers; use repeat_unit_smiles for repeat units",
        ),
        ConstraintRule(
            "target_chain_size",
            ("actual_total_carbons_per_chain", "target_total_carbons_per_chain"),
            lambda values: abs(
                int(values["actual_total_carbons_per_chain"])
                - int(values["target_total_carbons_per_chain"])
            )
            <= max(2, int(0.1 * max(1, int(values["target_total_carbons_per_chain"])))),
            "explicit polymer carbon count differs from target by both more than 2 carbons and more than 10%",
            severity="warning",
        ),
        UniqueTupleRule(
            "unique_explicit_polymer",
            ("canonical_polymer_smiles",),
            _existing_explicit_polymer_keys,
            "duplicate explicit polymer-chain structure",
        ),
    ),
)


class PeLcbAdapter(PolymerAdapter):
    adapter_id = "pe_lcb"
    architectures = frozenset({"pe_lcb"})
    polymer_families = frozenset({"PE"})
    required_fields = frozenset({"branch_points", "branch_carbons", "branch_position_pattern"})
    supported_force_fields = frozenset(ALLOWED_FORCE_FIELDS)
    supported_builder_backends = frozenset({"emc_alpha_olefin_builder"})
    design_check_profile = PE_LCB_DESIGN_CHECKS

    def build_spec(
        self,
        candidate: dict[str, Any],
        normalized: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        return _build_pe_lcb_spec(candidate, normalized, errors, warnings)


class LinearAlphaOlefinAdapter(PolymerAdapter):
    adapter_id = "linear_alpha_olefin"
    architectures = frozenset({"linear_alpha_olefin"})
    polymer_families = frozenset({"poly_alpha_olefin"})
    required_fields = frozenset({"sidechain_carbons"})
    supported_force_fields = frozenset(ALLOWED_FORCE_FIELDS)
    supported_builder_backends = frozenset({"emc_alpha_olefin_builder"})
    design_check_profile = LINEAR_ALPHA_OLEFIN_DESIGN_CHECKS

    def build_spec(
        self,
        candidate: dict[str, Any],
        normalized: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        return _build_linear_alpha_olefin_spec(candidate, normalized, errors, warnings)


class LinearHomopolymerAdapter(PolymerAdapter):
    adapter_id = "linear_homopolymer"
    architectures = frozenset({"linear_homopolymer"})
    polymer_families = frozenset({"*"})
    required_fields = frozenset({"repeat_unit_smiles", "repeat_unit_total_carbons", "repeat_units_per_chain"})
    supported_force_fields = frozenset({"*"})
    supported_builder_backends = frozenset({"*"})
    design_check_profile = LINEAR_HOMOPOLYMER_DESIGN_CHECKS

    def build_spec(
        self,
        candidate: dict[str, Any],
        normalized: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        return _build_linear_homopolymer_spec(candidate, normalized, errors, warnings)


class ExplicitPolymerChainAdapter(PolymerAdapter):
    adapter_id = "explicit_polymer_chain"
    architectures = frozenset({"explicit_polymer_chain"})
    polymer_families = frozenset({"*"})
    required_fields = frozenset({"polymer_smiles"})
    supported_force_fields = frozenset({"*"})
    supported_builder_backends = frozenset({"emc_repeat_unit_builder"})
    design_check_profile = EXPLICIT_POLYMER_CHAIN_DESIGN_CHECKS

    def build_spec(
        self,
        candidate: dict[str, Any],
        normalized: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        return _build_explicit_polymer_chain_spec(candidate, normalized, errors, warnings)


class TemplatedPolymerChainAdapter(PolymerAdapter):
    adapter_id = "templated_polymer_chain"
    architectures = frozenset({"templated_polymer_chain"})
    polymer_families = frozenset({"*"})
    required_fields = frozenset({"repeat_unit_templates", "sequence"})
    supported_force_fields = frozenset({"*"})
    supported_builder_backends = frozenset({"emc_template_polymer_builder"})

    def validate(
        self,
        candidate: dict[str, Any],
        normalized: dict[str, Any],
        errors: list[str],
        warnings: list[str],
        *,
        existing_features: Any = None,
    ) -> dict[str, Any] | None:
        try:
            plan = normalize_template_plan(candidate)
        except ValueError as exc:
            errors.append(str(exc))
            normalized["design_check_profile"] = "compact_polymer_template.v1"
            normalized["design_check_trace"] = [
                {"rule_id": "compact_template_validation", "status": "error"}
            ]
            return None

        normalized.update(
            {
                "architecture": "templated_polymer_chain",
                "template_schema_version": plan["schema_version"],
                "repeat_unit_templates": plan["repeat_unit_templates"],
                "sequence": plan["sequence"],
                "branch_templates": plan["branch_templates"],
                "branch_rules": plan["branch_rules"],
                "repeat_units_per_chain": plan["repeat_units_per_chain"],
                "actual_total_carbons_per_chain": plan["actual_total_carbons_per_chain"],
                "estimated_real_atoms_per_chain": plan["estimated_real_atoms_per_chain"],
                "structure_elements": plan["structure_elements"],
                "template_design_hash": plan["template_design_hash"],
                "design_check_profile": "compact_polymer_template.v1",
                "design_check_trace": [
                    {"rule_id": "repeat_unit_templates", "status": "pass"},
                    {"rule_id": "sequence_definition", "status": "pass"},
                    {"rule_id": "branch_definition", "status": "pass"},
                ],
            }
        )
        target_total = int(normalized.get("target_total_carbons_per_chain") or 0)
        actual_total = int(plan["actual_total_carbons_per_chain"])
        if target_total and abs(actual_total - target_total) > max(2, int(0.1 * target_total)):
            warnings.append(
                "template-derived carbon count differs from target by both more than 2 carbons and more than 10%"
            )
        existing_hashes: set[str] = set()
        if existing_features is not None:
            iterator = (
                (row for _, row in existing_features.iterrows())
                if hasattr(existing_features, "iterrows")
                else existing_features
            )
            for row in iterator:
                try:
                    value = row.get("template_design_hash") if hasattr(row, "get") else row["template_design_hash"]
                except Exception:
                    value = None
                if value:
                    existing_hashes.add(str(value))
        if str(plan["template_design_hash"]) in existing_hashes:
            errors.append("duplicate compact polymer-template design")
            return None
        return _build_templated_polymer_chain_spec(candidate, normalized, errors, warnings)

    def build_spec(
        self,
        candidate: dict[str, Any],
        normalized: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        return _build_templated_polymer_chain_spec(candidate, normalized, errors, warnings)


def _register_builtin_adapters() -> None:
    if BUILTIN_ADAPTER_MANIFEST.exists():
        load_adapter_plugins([BUILTIN_ADAPTER_MANIFEST])
        return
    for adapter in (
        PeLcbAdapter(),
        LinearAlphaOlefinAdapter(),
        LinearHomopolymerAdapter(),
        ExplicitPolymerChainAdapter(),
        TemplatedPolymerChainAdapter(),
    ):
        register_adapter(adapter, replace=True)


def _validate_adapter_scope(adapter: PolymerAdapter | None, normalized: dict[str, Any], errors: list[str]) -> None:
    architecture = str(normalized.get("architecture") or "")
    polymer_family = str(normalized.get("polymer_family") or "")
    builder_backend = str(normalized.get("builder_backend") or "")
    force_field = _canonical_force_field(normalized.get("force_field"))

    if adapter is None:
        registered_architectures = sorted(_ADAPTERS_BY_ARCHITECTURE)
        errors.append(f"architecture must be one of {registered_architectures}")
        return

    families = {str(value) for value in adapter.polymer_families}
    if families and "*" not in families and polymer_family not in families:
        errors.append(
            f"polymer_family must be one of {sorted(families)} for architecture {architecture}"
        )
    runtime_capabilities = list_runtime_capabilities()
    registered_builders = set(runtime_capabilities["builders"])
    registered_force_fields = set(runtime_capabilities["force_fields"])
    if builder_backend and builder_backend not in registered_builders:
        errors.append(f"builder_backend is not registered: {builder_backend}")
    elif (
        builder_backend
        and "*" not in adapter.supported_builder_backends
        and builder_backend not in adapter.supported_builder_backends
    ):
        errors.append(f"builder_backend must be one of {sorted(adapter.supported_builder_backends)}")
    supported_force_fields = {
        _canonical_force_field(value) for value in adapter.supported_force_fields
    }
    if force_field and force_field not in registered_force_fields:
        errors.append(f"force_field is not registered: {force_field}")
    elif force_field and "*" not in supported_force_fields and force_field not in supported_force_fields:
        errors.append(f"force_field must be one of {sorted(supported_force_fields)}")


def _validate_required_adapter_fields(adapter: PolymerAdapter | None, candidate: dict[str, Any], errors: list[str]) -> None:
    if adapter is None:
        return
    if adapter.design_check_profile is not None:
        return
    for field in sorted(adapter.required_fields):
        if candidate.get(field) in (None, ""):
            errors.append(f"missing required field: {field}")


_register_builtin_adapters()


def validate_polymer_design_candidate(
    candidate: dict[str, Any] | str,
    *,
    existing_features: Any = None,
) -> dict[str, Any]:
    parsed_input, parse_errors = _parse_candidate(candidate)
    if (
        parsed_input
        and str(parsed_input.get("architecture") or "").strip()
        in SPECIAL_ARCHITECTURES_REQUIRING_REGISTERED_BUILDER
    ):
        return _validate_special_architecture_candidate(parsed_input, parse_errors)

    prepared = auto_configure_polymer_candidate(candidate)
    parsed = dict(prepared.get("candidate") or {})
    errors = list(prepared.get("errors") or [])
    warnings: list[str] = []
    automatic_setup = dict(prepared.get("automatic_setup") or {})
    force_field_selection = automatic_setup.get("force_field_selection")
    allow_missing_force_field_for_review = bool(
        automatic_setup.get("status") == "review_required"
        and isinstance(force_field_selection, dict)
        and force_field_selection.get("force_field") in (None, "")
    )
    normalized = (
        _base_checks(
            parsed,
            errors,
            warnings,
            allow_missing_force_field_for_review=allow_missing_force_field_for_review,
        )
        if parsed
        else {}
    )
    builder_spec: dict[str, Any] | None = None

    architecture = str(normalized.get("architecture") or parsed.get("architecture") or "")
    adapter = get_adapter(architecture)
    _validate_adapter_scope(adapter, normalized, errors)
    _validate_required_adapter_fields(adapter, parsed, errors)
    if adapter is not None:
        builder_spec = adapter.validate(parsed, normalized, errors, warnings, existing_features=existing_features)
    if builder_spec:
        builder_spec["adapter_id"] = adapter.adapter_id if adapter is not None else None
        builder_spec["automatic_setup"] = automatic_setup
        builder_spec["parameterization_probe_required"] = bool(
            automatic_setup.get("force_field_selection", {}).get("parameterization_probe_required")
        )

    validator_status = "pass" if not errors and builder_spec else "reject"
    automatic_review = automatic_setup.get("status") == "review_required"
    if automatic_review and validator_status == "pass":
        validator_status = "review"
    builder_ready = validator_status == "pass"
    production_ready = builder_ready and not bool(
        automatic_setup.get("force_field_selection", {}).get("parameterization_probe_required")
    )
    return {
        "status": "ok" if parsed else "invalid_request",
        "validator_status": validator_status,
        "errors": errors,
        "warnings": warnings,
        "normalized_candidate": _json_safe(normalized),
        "builder_spec": _json_safe(builder_spec or {}),
        "automatic_setup": _json_safe(automatic_setup),
        "builder_ready": builder_ready,
        "emc_ready": builder_ready,
        "production_ready": production_ready,
        "next_action": (
            "automatic_remote_parameterization_probe"
            if builder_ready and not production_ready
            else "submit_md"
            if production_ready
            else "review"
        ),
    }


def validate_pe_lcb_legacy_candidate(
    branch_points: int,
    branch_carbons: int,
    branch_position_pattern: str,
    carbon_budget: int = 100,
    *,
    existing_features: Any = None,
) -> dict[str, Any]:
    candidate = {
        "candidate_id": f"pe_lcb_bp{int(branch_points):02d}_bc{int(branch_carbons):03d}_{branch_position_pattern}",
        "polymer_family": "PE",
        "architecture": "pe_lcb",
        "builder_backend": "emc_alpha_olefin_builder",
        "force_field": "opls-aa",
        "chain_count": 150,
        "target_total_carbons_per_chain": int(carbon_budget),
        "carbon_budget": int(carbon_budget),
        "branch_points": int(branch_points),
        "branch_carbons": int(branch_carbons),
        "branch_position_pattern": str(branch_position_pattern),
    }
    result = validate_polymer_design_candidate(candidate, existing_features=existing_features)
    normalized = result.get("normalized_candidate", {})
    return {
        "status": "ok",
        "validator_status": result["validator_status"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "candidate": {
            "branch_points": int(branch_points),
            "branch_carbons": int(branch_carbons),
            "branch_position_pattern": str(branch_position_pattern),
            "requested_carbon_budget": normalized.get("requested_carbon_budget", int(carbon_budget)),
            "actual_total_carbons_per_chain": normalized.get("actual_total_carbons_per_chain"),
            "backbone_repeat_units": normalized.get("backbone_repeat_units"),
            "branch_total_carbons": normalized.get("branch_total_carbons"),
        },
    }
