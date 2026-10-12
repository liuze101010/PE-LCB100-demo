#!/usr/bin/env python3
"""Dispatch validated polymer builder specifications to registered materializers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from emc_builder_registry import write_emc_case
from polymer_runtime_registry import get_builder_capability


Materializer = Callable[
    [Path, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None],
    dict[str, Any],
]


_MATERIALIZERS: dict[str, Materializer] = {
    "emc_builder_registry.write_emc_case": write_emc_case,
}


def register_builder_materializer(
    materializer_id: str,
    materializer: Materializer,
    *,
    replace: bool = False,
) -> None:
    key = str(materializer_id or "").strip()
    if not key:
        raise ValueError("materializer_id is required")
    if key in _MATERIALIZERS and not replace:
        raise ValueError(f"builder materializer already registered: {key}")
    if not callable(materializer):
        raise TypeError("materializer must be callable")
    _MATERIALIZERS[key] = materializer


def list_builder_materializers() -> list[str]:
    return sorted(_MATERIALIZERS)


def materialize_registered_polymer_input(
    case_dir: str | Path,
    builder_spec: dict[str, Any],
    validation: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend_id = str(builder_spec.get("builder_backend") or "").strip()
    capability = get_builder_capability(backend_id)
    if capability is None:
        raise ValueError(f"builder backend is not registered: {backend_id}")
    materializer = _MATERIALIZERS.get(capability.materializer)
    if materializer is None:
        raise ValueError(
            f"builder materializer is not registered: {capability.materializer}"
        )
    result = materializer(Path(case_dir), builder_spec, validation, plan)
    return {
        **result,
        "builder_backend": backend_id,
        "materializer": capability.materializer,
    }


__all__ = [
    "list_builder_materializers",
    "materialize_registered_polymer_input",
    "register_builder_materializer",
]
