#!/usr/bin/env python3
"""Sidecar packing/void postprocessing for PE-LCB release evidence.

This module computes conservative, grid-based structure QC metrics from a
local final-structure ``.gro`` file.  It writes sidecar summaries and overlay
tables only; it does not edit the PE-LCB100 canonical aggregate or ML tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
AGGREGATE_DIR = SCRIPT_DIR / "output" / "pelcb100_integrated" / "aggregate"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "pelcb100_packing_void"
DEFAULT_BRIDGE_ROOT = SCRIPT_DIR / "output" / "pelcb100_supervisor" / "bridge_ready"
SCHEMA_VERSION = "pelcb_packing_void.v1"
PACKING_COLUMNS = [
    "status_packing",
    "void_flag",
    "void_fraction",
    "low_density_voxel_fraction",
    "largest_connected_low_density_fraction",
    "largest_void_radius_nm",
    "min_nonbonded_distance_nm",
    "packing_uniformity_score",
    "box_occupancy_ratio",
    "packing_method",
    "packing_grid_spacing_nm",
    "packing_probe_radius_nm",
    "packing_low_density_radius_nm",
    "packing_source_path",
    "packing_source_kind",
    "packing_missing_reason",
]


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _float_from_fixed_width(line: str) -> tuple[float, float, float] | None:
    try:
        return float(line[20:28]), float(line[28:36]), float(line[36:44])
    except Exception:
        return None


def _float_tokens(line: str) -> list[float]:
    out: list[float] = []
    for token in line.split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def parse_gro(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"not enough lines for .gro file: {path}")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"invalid .gro atom count line in {path}") from exc
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f".gro atom count mismatch in {path}")
    coords: list[tuple[float, float, float]] = []
    for line in atom_lines:
        fixed = _float_from_fixed_width(line)
        if fixed is not None:
            coords.append(fixed)
            continue
        nums = _float_tokens(line)
        if len(nums) < 3:
            raise ValueError(f"cannot parse coordinate line in {path}: {line!r}")
        coords.append(tuple(nums[-3:]))
    box_tokens = _float_tokens(lines[2 + atom_count])
    if len(box_tokens) < 3:
        raise ValueError(f"invalid .gro box line in {path}")
    box = np.asarray(box_tokens[:3], dtype=float)
    if np.any(~np.isfinite(box)) or np.any(box <= 0):
        raise ValueError(f"non-positive .gro box dimensions in {path}: {box_tokens[:3]}")
    return np.asarray(coords, dtype=float), box


def _grid_dimensions(box_nm: np.ndarray, grid_spacing_nm: float, max_grid_cells: int) -> np.ndarray:
    dims = np.maximum(1, np.ceil(box_nm / float(grid_spacing_nm)).astype(int))
    total = int(np.prod(dims))
    if total <= max_grid_cells:
        return dims
    scale = (total / max_grid_cells) ** (1.0 / 3.0)
    return np.maximum(1, np.ceil(dims / scale).astype(int))


def _nearest_distances(points: np.ndarray, coords: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        safe_box = np.where(box_nm <= 0, 1.0, box_nm)
        tree = cKDTree(np.mod(coords, safe_box), boxsize=safe_box)
        distances, _ = tree.query(np.mod(points, safe_box), k=1, workers=-1)
        return np.asarray(distances, dtype=float)
    except Exception:
        distances: list[np.ndarray] = []
        for start in range(0, len(points), 1000):
            chunk = points[start : start + 1000]
            delta = np.abs(chunk[:, None, :] - coords[None, :, :])
            delta = np.minimum(delta, box_nm[None, None, :] - delta)
            distances.append(np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1))
        return np.concatenate(distances) if distances else np.asarray([], dtype=float)


def _min_pair_distance(coords: np.ndarray, box_nm: np.ndarray) -> float:
    if len(coords) < 2:
        return math.nan
    try:
        from scipy.spatial import cKDTree  # type: ignore

        safe_box = np.where(box_nm <= 0, 1.0, box_nm)
        tree = cKDTree(np.mod(coords, safe_box), boxsize=safe_box)
        distances, _ = tree.query(np.mod(coords, safe_box), k=2, workers=-1)
        return float(np.nanmin(distances[:, 1]))
    except Exception:
        if len(coords) > 3000:
            return math.nan
        best = math.inf
        for i in range(len(coords) - 1):
            delta = np.abs(coords[i + 1 :] - coords[i])
            delta = np.minimum(delta, box_nm - delta)
            distance = np.sqrt(np.sum(delta * delta, axis=1)).min()
            best = min(best, float(distance))
        return best if math.isfinite(best) else math.nan


def _largest_component_fraction(mask: np.ndarray) -> float:
    """Return largest 6-neighbor component fraction over all grid cells."""
    total = int(mask.size)
    if total == 0 or not bool(mask.any()):
        return 0.0
    visited = np.zeros(mask.shape, dtype=bool)
    largest = 0
    dims = mask.shape
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for start in zip(*np.where(mask & ~visited)):
        if visited[start]:
            continue
        count = 0
        queue: deque[tuple[int, int, int]] = deque([tuple(int(v) for v in start)])
        visited[start] = True
        while queue:
            i, j, k = queue.popleft()
            count += 1
            for di, dj, dk in neighbors:
                ni, nj, nk = i + di, j + dj, k + dk
                if ni < 0 or nj < 0 or nk < 0 or ni >= dims[0] or nj >= dims[1] or nk >= dims[2]:
                    continue
                if mask[ni, nj, nk] and not visited[ni, nj, nk]:
                    visited[ni, nj, nk] = True
                    queue.append((ni, nj, nk))
        largest = max(largest, count)
    return float(largest / total)


def _status_from_metrics(
    void_fraction: float,
    low_density_fraction: float,
    largest_void_radius_nm: float,
    min_distance_nm: float,
    uniformity_score: float,
    occupancy_ratio: float,
) -> tuple[str, bool, list[str]]:
    issues: list[str] = []
    if void_fraction > 0.10:
        issues.append("void_fraction > 0.10")
    if low_density_fraction > 0.15:
        issues.append("low_density_voxel_fraction > 0.15")
    if largest_void_radius_nm > 1.0:
        issues.append("largest_void_radius_nm > 1.0")
    if math.isfinite(min_distance_nm) and min_distance_nm < 0.08:
        issues.append("min_nonbonded_distance_nm < 0.08")
    if uniformity_score < 0.5:
        issues.append("packing_uniformity_score < 0.5")
    if occupancy_ratio < 0.5:
        issues.append("box_occupancy_ratio < 0.5")
    return ("WARN" if issues else "OK"), bool(issues), issues


def analyze_gro_file(
    gro_path: Path,
    grid_spacing_nm: float = 0.25,
    probe_radius_nm: float = 0.30,
    low_density_radius_nm: float = 0.35,
    max_grid_cells: int = 500_000,
) -> dict[str, Any]:
    coords, box = parse_gro(gro_path)
    coords = np.mod(coords, box)
    dims = _grid_dimensions(box, grid_spacing_nm, max_grid_cells)
    axes = [(np.arange(dim, dtype=float) + 0.5) * (box[axis] / dim) for axis, dim in enumerate(dims)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([axis.ravel() for axis in mesh])
    nearest = _nearest_distances(points, coords, box)
    void_mask_flat = nearest > float(probe_radius_nm)
    low_density_mask_flat = nearest > float(low_density_radius_nm)
    void_fraction = float(np.mean(void_mask_flat)) if nearest.size else math.nan
    low_density_fraction = float(np.mean(low_density_mask_flat)) if nearest.size else math.nan
    largest_component = _largest_component_fraction(low_density_mask_flat.reshape(tuple(int(v) for v in dims)))
    largest_void_radius = float(np.nanmax(nearest)) if nearest.size else math.nan
    min_distance = _min_pair_distance(coords, box)
    if nearest.size and float(np.nanmean(nearest)) > 0:
        uniformity_score = float(1.0 / (1.0 + np.nanstd(nearest) / np.nanmean(nearest)))
    else:
        uniformity_score = math.nan
    occupancy_ratio = float(1.0 - void_fraction) if math.isfinite(void_fraction) else math.nan
    status, void_flag, issues = _status_from_metrics(
        void_fraction,
        low_density_fraction,
        largest_void_radius,
        min_distance,
        uniformity_score,
        occupancy_ratio,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status_packing": status,
        "void_flag": void_flag,
        "void_fraction": void_fraction,
        "low_density_voxel_fraction": low_density_fraction,
        "largest_connected_low_density_fraction": largest_component,
        "largest_void_radius_nm": largest_void_radius,
        "min_nonbonded_distance_nm": min_distance,
        "packing_uniformity_score": uniformity_score,
        "box_occupancy_ratio": occupancy_ratio,
        "packing_method": "grid_occupancy_v1",
        "packing_grid_spacing_nm": float(grid_spacing_nm),
        "packing_probe_radius_nm": float(probe_radius_nm),
        "packing_low_density_radius_nm": float(low_density_radius_nm),
        "packing_source_path": str(gro_path),
        "packing_source_kind": "local_structure",
        "packing_missing_reason": "",
        "packing_issue_list": ";".join(issues),
    }


def _candidate_structure_paths(case_id: str, directory: Path) -> list[Path]:
    names = [
        "nvt_prod.gro",
        "confout.gro",
        "final.gro",
        f"{case_id}.gro",
    ]
    out: list[Path] = []
    for name in names:
        out.append(directory / name)
        out.append(directory / "nvt_prod" / name)
    out.extend(sorted(directory.glob("*.gro")))
    out.extend(sorted((directory / "nvt_prod").glob("*.gro")) if (directory / "nvt_prod").exists() else [])
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        resolved = path.resolve() if path.exists() else path
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def locate_structure_for_case(
    case_id: str,
    row: Mapping[str, Any],
    search_roots: list[Path] | None = None,
    allow_initial_structure_fallback: bool = False,
) -> Path | None:
    for key in ("packing_source_path", "final_gro_path", "structure_path"):
        path_text = cell_text(row.get(key, ""))
        if path_text:
            path = Path(path_text)
            if path.exists() and path.is_file():
                return path
    source_dir = cell_text(row.get("source_case_dir", ""))
    if source_dir:
        path = Path(source_dir)
        if path.exists() and path.is_dir():
            for candidate in _candidate_structure_paths(case_id, path):
                if candidate.exists() and candidate.is_file():
                    return candidate
    for root in search_roots or []:
        if not root.exists():
            continue
        direct = root / case_id
        if direct.exists():
            for candidate in _candidate_structure_paths(case_id, direct):
                if candidate.exists() and candidate.is_file():
                    return candidate
        for candidate in root.glob(f"**/*{case_id}*.gro"):
            if candidate.exists() and candidate.is_file():
                return candidate
    if allow_initial_structure_fallback:
        fallback = DEFAULT_BRIDGE_ROOT / case_id / f"{case_id}.gro"
        if fallback.exists():
            return fallback
    return None


def _missing_row(case_id: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "status_packing": "MISSING",
        "void_flag": "",
        "void_fraction": "",
        "low_density_voxel_fraction": "",
        "largest_connected_low_density_fraction": "",
        "largest_void_radius_nm": "",
        "min_nonbonded_distance_nm": "",
        "packing_uniformity_score": "",
        "box_occupancy_ratio": "",
        "packing_method": "grid_occupancy_v1",
        "packing_grid_spacing_nm": "",
        "packing_probe_radius_nm": "",
        "packing_low_density_radius_nm": "",
        "packing_source_path": "",
        "packing_source_kind": "",
        "packing_missing_reason": reason,
        "packing_issue_list": "",
    }


def build_packing_overlay(
    summary: pd.DataFrame,
    structure_map: dict[str, Path] | None = None,
    search_roots: list[Path] | None = None,
    grid_spacing_nm: float = 0.25,
    probe_radius_nm: float = 0.30,
    low_density_radius_nm: float = 0.35,
    allow_initial_structure_fallback: bool = False,
    case_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if summary.empty or "case_id" not in summary.columns:
        raise FileNotFoundError("summary table must contain case_id rows")
    selected = set(case_ids or [])
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        case_id = str(row["case_id"])
        if selected and case_id not in selected:
            continue
        row_dict = row.to_dict()
        case_status = cell_text(row_dict.get("case_status", ""))
        if "incomplete" in case_status.lower():
            rows.append(_missing_row(case_id, "incomplete_md"))
            continue
        structure = (structure_map or {}).get(case_id)
        if structure is None:
            structure = locate_structure_for_case(
                case_id,
                row_dict,
                search_roots=search_roots,
                allow_initial_structure_fallback=allow_initial_structure_fallback,
            )
        if structure is None:
            rows.append(_missing_row(case_id, "no_local_structure"))
            continue
        try:
            metrics = analyze_gro_file(
                Path(structure),
                grid_spacing_nm=grid_spacing_nm,
                probe_radius_nm=probe_radius_nm,
                low_density_radius_nm=low_density_radius_nm,
            )
        except Exception as exc:
            missing = _missing_row(case_id, f"analysis_error: {exc}")
            missing["packing_source_path"] = str(structure)
            rows.append(missing)
            continue
        rows.append({"case_id": case_id, **metrics})

    overlay = pd.DataFrame(summary.to_numpy(dtype=object), columns=summary.columns)
    for col in PACKING_COLUMNS:
        overlay[col] = pd.Series([""] * len(overlay), dtype=object)
    row_by_case = {row["case_id"]: row for row in rows}
    for idx, base_row in overlay.iterrows():
        case_id = str(base_row["case_id"])
        if selected and case_id not in selected:
            continue
        packing_row = row_by_case.get(case_id)
        if not packing_row:
            continue
        for col in PACKING_COLUMNS:
            overlay.at[idx, col] = packing_row.get(col, "")
    return overlay, rows


def _read_structure_map(path: Path | None) -> dict[str, Path]:
    if not path or not path.exists():
        return {}
    rows = read_csv(path)
    if rows.empty or "case_id" not in rows.columns or "structure_path" not in rows.columns:
        return {}
    return {
        str(row["case_id"]): Path(str(row["structure_path"]))
        for _, row in rows.iterrows()
        if cell_text(row.get("case_id", "")) and cell_text(row.get("structure_path", ""))
    }


def write_report(path: Path, rows: list[dict[str, Any]], overlay: pd.DataFrame) -> None:
    counts = Counter(str(row.get("status_packing", "")) for row in rows)
    lines = [
        "# PE-LCB100 packing/void sidecar overlay",
        "",
        f"- Generated: `{iso_now()}`",
        f"- Cases analyzed or marked: `{len(rows)}`",
        f"- Overlay rows: `{len(overlay)}`",
        "- Canonical PE-LCB100 aggregate and ML tables are unchanged.",
        "",
        "## Packing status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Policy", "", "- Missing local structures are marked `status_packing=MISSING`; they do not become physical void claims.", "- `WARN` indicates generic red-flag packing metrics and must enter audit diagnosis before any repair route.", "- Initial bridge structures may be used only when explicitly requested with `--allow-initial-structure-fallback`."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=AGGREGATE_DIR / "pelcb100_summary_wide.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="packing_void_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--structure-map", type=Path, default=None)
    parser.add_argument("--grid-spacing-nm", type=float, default=0.25)
    parser.add_argument("--probe-radius-nm", type=float, default=0.30)
    parser.add_argument("--low-density-radius-nm", type=float, default=0.35)
    parser.add_argument("--allow-initial-structure-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.output_dir / args.tag
    summary = read_csv(args.summary)
    overlay, rows = build_packing_overlay(
        summary,
        structure_map=_read_structure_map(args.structure_map),
        search_roots=list(args.search_root),
        grid_spacing_nm=float(args.grid_spacing_nm),
        probe_radius_nm=float(args.probe_radius_nm),
        low_density_radius_nm=float(args.low_density_radius_nm),
        allow_initial_structure_fallback=bool(args.allow_initial_structure_fallback),
        case_ids=[str(case_id) for case_id in args.case_id],
    )
    summary_fields = ["schema_version", "case_id", *PACKING_COLUMNS, "packing_issue_list"]
    summary_csv = run_dir / "packing_void_summary.csv"
    overlay_csv = run_dir / "pelcb100_summary_wide_with_packing_overlay.csv"
    report_md = run_dir / "packing_void_report.md"
    write_csv(summary_csv, rows, summary_fields)
    overlay.to_csv(overlay_csv, index=False, encoding="utf-8-sig")
    write_report(report_md, rows, overlay)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_now(),
        "mode": "sidecar_overlay",
        "canonical_release_modified": False,
        "md_rerun": False,
        "case_count": len(rows),
        "status_counts": dict(Counter(str(row.get("status_packing", "")) for row in rows)),
        "packing_summary_csv": str(summary_csv),
        "overlay_summary_csv": str(overlay_csv),
        "report_md": str(report_md),
        "source_summary_csv": str(args.summary),
    }
    write_json(run_dir / "packing_void_summary.json", payload)
    write_json(args.output_dir / "packing_void_latest.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
