#!/usr/bin/env python3
"""Shared project configuration for LCB automation scripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "lcb300.json"


def _resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass(frozen=True)
class CaseRoot:
    label: str
    path: Path


@dataclass(frozen=True)
class ClusterProfile:
    label: str
    host_label: str
    host: str
    remote_root: str
    engine_module: str
    module_preset: str
    gmx_bin: str
    partition: str
    allowed_nodes: tuple[str, ...]
    disallowed_nodes: tuple[str, ...]
    default_threads: int
    threads_by_node: dict[str, int]
    launch_via_node: bool
    engine_label: str = ""
    engine_version_expected: str = ""
    node_cap_default: int = 0
    node_cap_by_node: dict[str, int] = field(default_factory=dict)
    per_gpu_target_default: int = 0
    per_gpu_target_by_node: dict[str, int] = field(default_factory=dict)
    module_preset_by_node: dict[str, str] = field(default_factory=dict)
    idle_skip_util_percent: int = 10
    idle_skip_memory_mib: int = 500
    project_memory_per_case_mib: int = 414
    external_memory_slack_mib: int = 250
    external_memory_per_case_mib: int = 400
    idle_external_memory_threshold_mib: int = 700
    active_extra_memory_skip_mib: int = 1000

    def threads_for_node(self, node: str) -> int:
        return int(self.threads_by_node.get(str(node), self.default_threads))

    def node_cap_for_node(self, node: str) -> int:
        return int(self.node_cap_by_node.get(str(node), self.node_cap_default))

    def per_gpu_target_for_node(self, node: str, *, node_cap: int, gpu_count: int) -> int:
        node = str(node)
        if node in self.per_gpu_target_by_node:
            return int(self.per_gpu_target_by_node[node])
        if self.per_gpu_target_default > 0:
            return int(self.per_gpu_target_default)
        if gpu_count <= 0:
            return max(0, int(node_cap))
        return int((max(0, int(node_cap)) + int(gpu_count) - 1) / int(gpu_count))

    def module_preset_for_node(self, node: str) -> str:
        return str(self.module_preset_by_node.get(str(node), self.module_preset))


@dataclass(frozen=True)
class LCBProjectConfig:
    project_id: str
    manifest_csv: Path
    case_roots: tuple[CaseRoot, ...]
    bridge_roots: tuple[Path, ...]
    integrated_root: Path
    runtime_output_dir: Path
    supervisor_output_dir: Path
    emc_output_dir: Path
    remote_md_roots: dict[str, list[str]]
    old_server_postprocess_root: str
    old_server_input_root: str
    old_server: str
    old_emc_root: str
    emc_remote_root: str
    new_server: str
    newnew_server: str
    new_gmx: str
    newnew_gmx: str
    disabled_new_nodes: tuple[str, ...]
    disabled_md_hosts: tuple[str, ...]
    a3_results_dir: Path
    repair_loop_dir: Path
    warn_audit_output_dir: Path
    packing_void_output_dir: Path
    overlay_dir: Path
    rule_profile_dir: Path
    o7_queue_dir: Path
    o7_closure_dir: Path
    cluster_profiles: dict[str, ClusterProfile]

    @property
    def output_root(self) -> Path:
        return SCRIPT_DIR / "output"

    @property
    def integrated_summary_csv(self) -> Path:
        return self.integrated_root / "aggregate" / f"{self.project_id}_summary_wide.csv"

    @property
    def integrated_output_dir(self) -> Path:
        return self.integrated_root / "aggregate"

    @property
    def default_emc_bridge_root(self) -> Path:
        return self.supervisor_output_dir / "bridge_ready"

    def cluster_profile(self, label: str) -> ClusterProfile:
        try:
            return self.cluster_profiles[label]
        except KeyError as exc:
            available = ", ".join(sorted(self.cluster_profiles)) or "<none>"
            raise KeyError(f"unknown cluster profile {label!r}; available: {available}") from exc


def _parse_cluster_profiles(payload: dict[str, Any]) -> dict[str, ClusterProfile]:
    profiles: dict[str, ClusterProfile] = {}
    for label, raw in payload.items():
        item = dict(raw or {})
        profiles[str(label)] = ClusterProfile(
            label=str(label),
            host_label=str(item.get("host_label", "")),
            host=str(item.get("host", "")),
            remote_root=str(item.get("remote_root", "")),
            engine_module=str(item.get("engine_module", "")),
            module_preset=str(item.get("module_preset", "")),
            gmx_bin=str(item.get("gmx_bin", "")),
            partition=str(item.get("partition", "")),
            allowed_nodes=tuple(str(node) for node in item.get("allowed_nodes", [])),
            disallowed_nodes=tuple(str(node) for node in item.get("disallowed_nodes", [])),
            default_threads=int(item.get("default_threads", 1)),
            threads_by_node={str(node): int(threads) for node, threads in dict(item.get("threads_by_node", {})).items()},
            launch_via_node=bool(item.get("launch_via_node", True)),
            engine_label=str(item.get("engine_label", "")),
            engine_version_expected=str(item.get("engine_version_expected", "")),
            node_cap_default=int(item.get("node_cap_default", 0)),
            node_cap_by_node={str(node): int(cap) for node, cap in dict(item.get("node_cap_by_node", {})).items()},
            per_gpu_target_default=int(item.get("per_gpu_target_default", 0)),
            per_gpu_target_by_node={
                str(node): int(target) for node, target in dict(item.get("per_gpu_target_by_node", {})).items()
            },
            module_preset_by_node={
                str(node): str(preset) for node, preset in dict(item.get("module_preset_by_node", {})).items()
            },
            idle_skip_util_percent=int(item.get("idle_skip_util_percent", 10)),
            idle_skip_memory_mib=int(item.get("idle_skip_memory_mib", 500)),
            project_memory_per_case_mib=int(item.get("project_memory_per_case_mib", 414)),
            external_memory_slack_mib=int(item.get("external_memory_slack_mib", 250)),
            external_memory_per_case_mib=int(item.get("external_memory_per_case_mib", 400)),
            idle_external_memory_threshold_mib=int(item.get("idle_external_memory_threshold_mib", 700)),
            active_extra_memory_skip_mib=int(item.get("active_extra_memory_skip_mib", 1000)),
        )
    return profiles


def _expand_environment_values(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment_values(item) for key, item in value.items()}
    return value


def load_lcb_config(path: str | Path | None = None) -> LCBProjectConfig:
    config_path = _resolve_path(path or DEFAULT_CONFIG)
    payload: dict[str, Any] = _expand_environment_values(json.loads(config_path.read_text(encoding="utf-8")))
    project_id = str(payload["project_id"])
    integrated_root = _resolve_path(payload["integrated_root"])
    supervisor_output_dir = _resolve_path(payload["supervisor_output_dir"])
    integrated_output_dir = integrated_root / "aggregate"
    output_root = SCRIPT_DIR / "output"
    case_roots = tuple(
        CaseRoot(label=str(item.get("label") or Path(str(item["path"])).name), path=_resolve_path(item["path"]))
        for item in payload.get("case_roots", [])
    )
    if not case_roots:
        raise ValueError(f"Config has no case_roots: {config_path}")
    bridge_roots = tuple(_resolve_path(path_item) for path_item in payload.get("bridge_roots", []))
    remote_md_roots = {
        str(key): [str(value) for value in values]
        for key, values in dict(payload.get("remote_md_roots", {})).items()
    }
    cluster_profiles = _parse_cluster_profiles(dict(payload.get("cluster_profiles", {})))
    return LCBProjectConfig(
        project_id=project_id,
        manifest_csv=_resolve_path(payload["manifest_csv"]),
        case_roots=case_roots,
        bridge_roots=bridge_roots,
        integrated_root=integrated_root,
        runtime_output_dir=_resolve_path(payload["runtime_output_dir"]),
        supervisor_output_dir=supervisor_output_dir,
        emc_output_dir=_resolve_path(payload["emc_output_dir"]),
        remote_md_roots=remote_md_roots,
        old_server_postprocess_root=str(payload.get("old_server_postprocess_root", "${PELCB100_REMOTE_ROOT}/lcb300_postprocess")),
        old_server_input_root=str(payload.get("old_server_input_root", "${PELCB100_REMOTE_ROOT}/lcb_rg_20260501/input")),
        old_server=str(payload.get("old_server", os.environ.get("PELCB_OLD_SERVER", ""))),
        old_emc_root=str(payload.get("old_emc_root", "${PELCB100_REMOTE_ROOT}/tools/emc")),
        emc_remote_root=str(payload.get("emc_remote_root", "${PELCB100_REMOTE_ROOT}/lz_project_runs/emc_alpha_olefin_builder")),
        new_server=str(payload.get("new_server", os.environ.get("PELCB_NEW_SERVER", ""))),
        newnew_server=str(payload.get("newnew_server", os.environ.get("PELCB_NEWNEW_SERVER", ""))),
        new_gmx=str(payload.get("new_gmx", "${PELCB100_REMOTE_PATH}")),
        newnew_gmx=str(payload.get("newnew_gmx", "${PELCB100_REMOTE_ROOT}/miniforge3/envs/md/bin/gmx")),
        disabled_new_nodes=tuple(str(item) for item in payload.get("disabled_new_nodes", [])),
        disabled_md_hosts=tuple(str(item) for item in payload.get("disabled_md_hosts", [])),
        a3_results_dir=_resolve_path(payload.get("a3_results_dir", SCRIPT_DIR / "agentic" / "results" / project_id)),
        repair_loop_dir=_resolve_path(payload.get("repair_loop_dir", output_root / f"{project_id}_repair_loop")),
        warn_audit_output_dir=_resolve_path(
            payload.get("warn_audit_output_dir", output_root / f"{project_id}_warn_audit_diagnosis")
        ),
        packing_void_output_dir=_resolve_path(payload.get("packing_void_output_dir", output_root / f"{project_id}_packing_void")),
        overlay_dir=_resolve_path(payload.get("overlay_dir", integrated_output_dir / "overlays")),
        rule_profile_dir=_resolve_path(payload.get("rule_profile_dir", output_root / "rule_profiles")),
        o7_queue_dir=_resolve_path(payload.get("o7_queue_dir", supervisor_output_dir / "o7_clean_cuda2024_supervisor_queue")),
        o7_closure_dir=_resolve_path(payload.get("o7_closure_dir", supervisor_output_dir / "o7_clean_cuda2024_closure")),
        cluster_profiles=cluster_profiles,
    )


def write_default_lcb300_config(path: Path = DEFAULT_CONFIG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_id": "lcb300",
        "manifest_csv": "generated_code/emc_alpha_olefin_builder/output/screen_lcb300/manifest.csv",
        "case_roots": [
            {"label": "screen_lcb100", "path": "generated_code/emc_alpha_olefin_builder/output/screen_lcb100"},
            {"label": "screen_lcb200", "path": "generated_code/emc_alpha_olefin_builder/output/screen_lcb200"},
            {"label": "screen_lcb300", "path": "generated_code/emc_alpha_olefin_builder/output/screen_lcb300"},
        ],
        "bridge_roots": [
            "generated_code/emc_alpha_olefin_builder/output/lcb100_bridge_fixed_20260424",
            "generated_code/emc_alpha_olefin_builder/output/lcb200_bridge_lcb101_lcb200_20260507",
            "generated_code/emc_alpha_olefin_builder/output/lcb300_bridge_lcb201_lcb300_20260507",
        ],
        "integrated_root": "generated_code/md_analysis/output/lcb300_integrated",
        "runtime_output_dir": "generated_code/md_analysis/output/lcb300_runtime_dashboard",
        "supervisor_output_dir": "generated_code/md_analysis/output/lcb300_supervisor",
        "emc_output_dir": "generated_code/md_analysis/output/lcb300_emc_orchestrator",
        "remote_md_roots": {
            "new": ["${PELCB100_REMOTE_HOME}/lcb_manual_runs"],
            "newnew": ["${PELCB100_REMOTE_ROOT}/lcb_manual_runs"],
        },
        "old_server_postprocess_root": "${PELCB100_REMOTE_ROOT}/lcb300_postprocess",
        "old_server_input_root": "${PELCB100_REMOTE_ROOT}/lcb_rg_20260501/input",
        "old_server": os.environ.get("PELCB_OLD_SERVER", ""),
        "old_emc_root": "${PELCB100_REMOTE_ROOT}/tools/emc",
        "emc_remote_root": "${PELCB100_REMOTE_ROOT}/lz_project_runs/emc_alpha_olefin_builder",
        "new_server": os.environ.get("PELCB_NEW_SERVER", ""),
        "newnew_server": os.environ.get("PELCB_NEWNEW_SERVER", ""),
        "new_gmx": "${PELCB100_REMOTE_PATH}",
        "newnew_gmx": "${PELCB100_REMOTE_ROOT}/miniforge3/envs/md/bin/gmx",
        "cluster_profiles": {
            "production_new": {
                "host_label": "new",
                "host": os.environ.get("PELCB_NEW_SERVER", ""),
                "remote_root": "${PELCB100_REMOTE_HOME}/lcb_manual_runs",
                "engine_module": "",
                "module_preset": "newserver-gromacs2021",
                "module_preset_by_node": {"node-d61b45bd": "newserver-gromacs2021-node8"},
                "gmx_bin": "${PELCB100_REMOTE_PATH}",
                "partition": "manual",
                "allowed_nodes": [],
                "disallowed_nodes": [],
                "default_threads": 2,
                "threads_by_node": {"node-d61b45bd": 6},
                "node_cap_default": 10,
                "node_cap_by_node": {"node-d61b45bd": 8},
                "per_gpu_target_default": 0,
                "per_gpu_target_by_node": {"node-d61b45bd": 2},
                "launch_via_node": True,
            },
            "production_newnew": {
                "host_label": "newnew",
                "host": os.environ.get("PELCB_NEWNEW_SERVER", ""),
                "remote_root": "${PELCB100_REMOTE_ROOT}/lcb_manual_runs",
                "engine_module": "",
                "module_preset": "none",
                "gmx_bin": "${PELCB100_REMOTE_ROOT}/miniforge3/envs/md/bin/gmx",
                "partition": "manual",
                "allowed_nodes": [],
                "disallowed_nodes": [],
                "default_threads": 4,
                "threads_by_node": {},
                "per_gpu_target_default": 2,
                "launch_via_node": False,
            },
        },
        "disabled_md_hosts": [],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_default_lcb300_config()
