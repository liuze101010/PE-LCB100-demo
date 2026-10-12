#!/usr/bin/env python3
"""Registered runtime capabilities for polymer molecular dynamics.

The registry is the shared source of truth for molecule preparation and the
rule layer.  A backend or force field is usable only when it is registered.
Scientific qualification is tracked separately from technical support: a
registered combination may be prepared and probed while still requiring
external reference evidence before publication use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_PATH = Path(__file__).resolve()
PLUGIN_DIR = SCRIPT_PATH.parent / "plugins"
BUILTIN_CAPABILITY_MANIFEST = PLUGIN_DIR / "builtin_polymer_runtime_capabilities.json"
RUNTIME_CAPABILITY_SCHEMA_VERSION = "polymer_md_runtime_capabilities.v1"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _slug(value: Any) -> str:
    return _text(value).lower().replace("_", "-")


def _string_set(values: Iterable[Any] | None) -> frozenset[str]:
    return frozenset(_text(value) for value in (values or []) if _text(value))


def _parse_elements(value: Any) -> frozenset[str]:
    if value is None or value == "":
        return frozenset()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text.replace(",", " ").split()
        else:
            value = text.replace(",", " ").split()
    if not isinstance(value, (list, tuple, set, frozenset)):
        value = [value]
    return _string_set(value)


@dataclass(frozen=True)
class ForceFieldCapability:
    force_field_id: str
    aliases: frozenset[str]
    supported_elements: frozenset[str]
    parameterization_probe_required_by_default: bool = True


@dataclass(frozen=True)
class BuilderCapability:
    backend_id: str
    architectures: frozenset[str]
    force_fields: frozenset[str]
    topology_formats: frozenset[str]
    supported_elements: frozenset[str]
    materializer: str


@dataclass(frozen=True)
class ValidatedScope:
    scope_id: str
    polymer_families: frozenset[str]
    architectures: frozenset[str]
    builder_backends: frozenset[str]
    force_fields: frozenset[str]
    topology_formats: frozenset[str]


_FORCE_FIELDS: dict[str, ForceFieldCapability] = {}
_FORCE_FIELD_ALIASES: dict[str, str] = {}
_BUILDERS: dict[str, BuilderCapability] = {}
_VALIDATED_SCOPES: dict[str, ValidatedScope] = {}


def register_force_field(capability: ForceFieldCapability, *, replace: bool = False) -> None:
    force_field_id = _slug(capability.force_field_id)
    if not force_field_id:
        raise ValueError("force_field_id is required")
    if force_field_id in _FORCE_FIELDS and not replace:
        raise ValueError(f"force field already registered: {force_field_id}")
    normalized = ForceFieldCapability(
        force_field_id=force_field_id,
        aliases=frozenset({_slug(value) for value in capability.aliases} | {force_field_id}),
        supported_elements=_string_set(capability.supported_elements),
        parameterization_probe_required_by_default=bool(
            capability.parameterization_probe_required_by_default
        ),
    )
    if replace and force_field_id in _FORCE_FIELDS:
        old = _FORCE_FIELDS[force_field_id]
        for alias in old.aliases:
            _FORCE_FIELD_ALIASES.pop(_slug(alias), None)
    _FORCE_FIELDS[force_field_id] = normalized
    for alias in normalized.aliases:
        owner = _FORCE_FIELD_ALIASES.get(alias)
        if owner and owner != force_field_id and not replace:
            raise ValueError(f"force-field alias already registered: {alias}")
        _FORCE_FIELD_ALIASES[alias] = force_field_id


def register_builder(capability: BuilderCapability, *, replace: bool = False) -> None:
    backend_id = _text(capability.backend_id)
    if not backend_id:
        raise ValueError("backend_id is required")
    if backend_id in _BUILDERS and not replace:
        raise ValueError(f"builder backend already registered: {backend_id}")
    normalized_force_fields = frozenset(canonical_force_field(value) for value in capability.force_fields)
    if not normalized_force_fields:
        raise ValueError(f"builder {backend_id} must declare at least one force field")
    _BUILDERS[backend_id] = BuilderCapability(
        backend_id=backend_id,
        architectures=_string_set(capability.architectures),
        force_fields=normalized_force_fields,
        topology_formats=_string_set(capability.topology_formats),
        supported_elements=_string_set(capability.supported_elements),
        materializer=_text(capability.materializer),
    )


def register_validated_scope(scope: ValidatedScope, *, replace: bool = False) -> None:
    scope_id = _text(scope.scope_id)
    if not scope_id:
        raise ValueError("scope_id is required")
    if scope_id in _VALIDATED_SCOPES and not replace:
        raise ValueError(f"validated scope already registered: {scope_id}")
    _VALIDATED_SCOPES[scope_id] = ValidatedScope(
        scope_id=scope_id,
        polymer_families=_string_set(scope.polymer_families),
        architectures=_string_set(scope.architectures),
        builder_backends=_string_set(scope.builder_backends),
        force_fields=frozenset(canonical_force_field(value) for value in scope.force_fields),
        topology_formats=_string_set(scope.topology_formats),
    )


def canonical_force_field(value: Any) -> str:
    slug = _slug(value)
    return _FORCE_FIELD_ALIASES.get(slug, slug)


def get_force_field_capability(value: Any) -> ForceFieldCapability | None:
    return _FORCE_FIELDS.get(canonical_force_field(value))


def get_builder_capability(backend_id: Any) -> BuilderCapability | None:
    return _BUILDERS.get(_text(backend_id))


def list_runtime_capabilities() -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_CAPABILITY_SCHEMA_VERSION,
        "builders": {
            key: {
                "backend_id": value.backend_id,
                "architectures": sorted(value.architectures),
                "force_fields": sorted(value.force_fields),
                "topology_formats": sorted(value.topology_formats),
                "supported_elements": sorted(value.supported_elements),
                "materializer": value.materializer,
            }
            for key, value in sorted(_BUILDERS.items())
        },
        "force_fields": {
            key: {
                "force_field_id": value.force_field_id,
                "aliases": sorted(value.aliases),
                "supported_elements": sorted(value.supported_elements),
                "parameterization_probe_required_by_default": (
                    value.parameterization_probe_required_by_default
                ),
            }
            for key, value in sorted(_FORCE_FIELDS.items())
        },
        "validated_scopes": {
            key: {
                "scope_id": value.scope_id,
                "polymer_families": sorted(value.polymer_families),
                "architectures": sorted(value.architectures),
                "builder_backends": sorted(value.builder_backends),
                "force_fields": sorted(value.force_fields),
                "topology_formats": sorted(value.topology_formats),
            }
            for key, value in sorted(_VALIDATED_SCOPES.items())
        },
    }


def _matches(value: str, allowed: frozenset[str]) -> bool:
    return not allowed or "*" in allowed or value in allowed


def matching_validated_scope(
    *,
    polymer_family: Any = None,
    architecture: Any = None,
    builder_backend: Any = None,
    force_field: Any = None,
    topology_format: Any = None,
) -> ValidatedScope | None:
    family = _text(polymer_family)
    arch = _text(architecture)
    builder = _text(builder_backend)
    force = canonical_force_field(force_field)
    topology = _text(topology_format)
    for scope in _VALIDATED_SCOPES.values():
        if not _matches(family, scope.polymer_families):
            continue
        if not _matches(arch, scope.architectures):
            continue
        if not _matches(builder, scope.builder_backends):
            continue
        if not _matches(force, scope.force_fields):
            continue
        if topology and not _matches(topology, scope.topology_formats):
            continue
        return scope
    return None


def assess_preparation_capability(
    *,
    polymer_family: Any,
    architecture: Any,
    builder_backend: Any,
    force_field: Any,
    elements: Any = None,
    topology_format: Any = None,
) -> dict[str, Any]:
    family = _text(polymer_family)
    arch = _text(architecture)
    builder_id = _text(builder_backend)
    force_id = canonical_force_field(force_field)
    topology = _text(topology_format)
    element_set = _parse_elements(elements)
    builder = get_builder_capability(builder_id)
    force = get_force_field_capability(force_id)

    builder_supported = builder is not None
    architecture_supported = bool(
        builder
        and (
            not arch
            or not builder.architectures
            or "*" in builder.architectures
            or arch in builder.architectures
        )
    )
    force_field_supported = bool(
        force and (builder is None or force_id in builder.force_fields)
    )
    element_supported = bool(
        force
        and (builder is None or not element_set or element_set <= builder.supported_elements)
        and (not element_set or element_set <= force.supported_elements)
    )
    topology_supported = bool(
        builder
        and (not topology or topology in builder.topology_formats)
    )
    technical_support = all(
        (
            builder_supported,
            architecture_supported,
            force_field_supported,
            element_supported,
            topology_supported,
        )
    )
    scope = matching_validated_scope(
        polymer_family=family,
        architecture=arch,
        builder_backend=builder_id,
        force_field=force_id,
        topology_format=topology,
    )
    qualification_status = (
        f"validated_scope:{scope.scope_id}"
        if scope is not None
        else "external_reference_qualification_required"
        if technical_support
        else "unsupported_configuration"
    )
    parameterization_probe_required = bool(
        technical_support
        and scope is None
        and force
        and force.parameterization_probe_required_by_default
    )
    return {
        "schema_version": RUNTIME_CAPABILITY_SCHEMA_VERSION,
        "technical_support": technical_support,
        "builder_backend_supported": builder_supported and architecture_supported,
        "force_field_supported": force_field_supported and element_supported,
        "topology_supported": topology_supported,
        "architecture_supported": architecture_supported,
        "element_supported": element_supported,
        "parameterization_probe_required": parameterization_probe_required,
        "scientific_qualification_status": qualification_status,
        "validated_scope_id": scope.scope_id if scope else None,
        "support_assessment_source": f"registered_capability:{RUNTIME_CAPABILITY_SCHEMA_VERSION}",
        "resolved": {
            "polymer_family": family or None,
            "architecture": arch or None,
            "builder_backend": builder_id or None,
            "force_field": force_id or None,
            "topology_format": topology or None,
            "elements": sorted(element_set),
        },
    }


def preferred_builder_backend(architecture: str) -> str:
    matches = [
        builder.backend_id
        for builder in _BUILDERS.values()
        if not builder.architectures
        or "*" in builder.architectures
        or architecture in builder.architectures
    ]
    if architecture == "linear_homopolymer" and "emc_repeat_unit_builder" in matches:
        return "emc_repeat_unit_builder"
    return sorted(matches)[0] if matches else ""


def preferred_force_field(builder_backend: str, elements: Any = None) -> str:
    builder = get_builder_capability(builder_backend)
    if builder is None:
        return ""
    element_set = _parse_elements(elements)
    matches = []
    for force_id in sorted(builder.force_fields):
        capability = get_force_field_capability(force_id)
        if capability is None:
            continue
        if element_set and not element_set <= capability.supported_elements:
            continue
        if element_set and not element_set <= builder.supported_elements:
            continue
        matches.append(force_id)
    return matches[0] if matches else ""


def load_capability_manifest(path: str | Path, *, replace: bool = True) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    counts = {"force_fields": 0, "builders": 0, "validated_scopes": 0}
    for item in payload.get("force_fields", []) or []:
        register_force_field(
            ForceFieldCapability(
                force_field_id=_text(item.get("force_field_id")),
                aliases=_string_set(item.get("aliases")),
                supported_elements=_string_set(item.get("supported_elements")),
                parameterization_probe_required_by_default=bool(
                    item.get("parameterization_probe_required_by_default", True)
                ),
            ),
            replace=replace,
        )
        counts["force_fields"] += 1
    for item in payload.get("builders", []) or []:
        register_builder(
            BuilderCapability(
                backend_id=_text(item.get("backend_id")),
                architectures=_string_set(item.get("architectures")),
                force_fields=_string_set(item.get("force_fields")),
                topology_formats=_string_set(item.get("topology_formats")),
                supported_elements=_string_set(item.get("supported_elements")),
                materializer=_text(item.get("materializer")),
            ),
            replace=replace,
        )
        counts["builders"] += 1
    for item in payload.get("validated_scopes", []) or []:
        register_validated_scope(
            ValidatedScope(
                scope_id=_text(item.get("scope_id")),
                polymer_families=_string_set(item.get("polymer_families")),
                architectures=_string_set(item.get("architectures")),
                builder_backends=_string_set(item.get("builder_backends")),
                force_fields=_string_set(item.get("force_fields")),
                topology_formats=_string_set(item.get("topology_formats")),
            ),
            replace=replace,
        )
        counts["validated_scopes"] += 1
    return counts


if BUILTIN_CAPABILITY_MANIFEST.exists():
    load_capability_manifest(BUILTIN_CAPABILITY_MANIFEST)


__all__ = [
    "BuilderCapability",
    "ForceFieldCapability",
    "RUNTIME_CAPABILITY_SCHEMA_VERSION",
    "ValidatedScope",
    "assess_preparation_capability",
    "canonical_force_field",
    "get_builder_capability",
    "get_force_field_capability",
    "list_runtime_capabilities",
    "load_capability_manifest",
    "matching_validated_scope",
    "preferred_builder_backend",
    "preferred_force_field",
    "register_builder",
    "register_force_field",
    "register_validated_scope",
]
