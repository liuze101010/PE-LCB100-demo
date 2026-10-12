#!/usr/bin/env python3
"""Small helper for manifest-driven local plugins.

Plugin manifests may refer to importable modules or to local Python files.  The
loader returns the requested symbol only; each subsystem remains responsible for
validating the symbol against its own adapter/provider contract.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _module_name_from_path(path: Path) -> str:
    return "_md_analysis_plugin_" + path.stem + "_" + str(abs(hash(path.resolve())))


def load_module_from_spec(spec: dict[str, Any], *, base_dir: str | Path | None = None) -> Any:
    base = Path(base_dir or ".").resolve()
    if spec.get("module_path"):
        module_path = Path(str(spec["module_path"]))
        if not module_path.is_absolute():
            module_path = (base / module_path).resolve()
        name = str(spec.get("module_name") or _module_name_from_path(module_path))
        existing = sys.modules.get(name)
        if existing is not None:
            return existing
        module_spec = importlib.util.spec_from_file_location(name, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load plugin module from {module_path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        module_spec.loader.exec_module(module)
        return module
    if spec.get("module"):
        return importlib.import_module(str(spec["module"]))
    raise ValueError("plugin spec requires 'module' or 'module_path'")


def load_symbol_from_spec(spec: dict[str, Any], *, base_dir: str | Path | None = None) -> Any:
    module = load_module_from_spec(spec, base_dir=base_dir)
    symbol_name = spec.get("class") or spec.get("function") or spec.get("symbol")
    if not symbol_name:
        raise ValueError("plugin spec requires 'class', 'function', or 'symbol'")
    try:
        return getattr(module, str(symbol_name))
    except AttributeError as exc:
        raise AttributeError(f"plugin symbol {symbol_name!r} not found in {module!r}") from exc
