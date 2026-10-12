#!/usr/bin/env python3
"""LCB300 postprocessing orchestrator.

Default mode is plan-only.  It reads the runtime dashboard outputs and writes a
concrete non-viscosity postprocessing plan plus old-server scripts.  Remote
staging and Slurm submission require ``--execute``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lcb_project_config import load_lcb_config


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ID = "lcb300"
DEFAULT_DASHBOARD = SCRIPT_DIR / "output" / "lcb300_runtime_dashboard"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "lcb300_orchestrator"
OLD_SERVER = "${REMOTE_USER}@private-3f2a50db.invalid"
OLD_WORK = "${PELCB100_REMOTE_ROOT}/lcb300_postprocess"
OLD_INPUT = "${PELCB100_REMOTE_ROOT}/lcb_rg_20260501/input"
OLD_CODE = f"{OLD_WORK}/md_analysis_current"
OLD_OUTPUT = f"{OLD_WORK}/results"
OLD_GMX = "${PELCB100_REMOTE_ROOT}/tools/gromacs/gromacs-2024.4_gcc11_avx2_nompi_staticstdcpp/bin/gmx"
OLD_PYTHON = "${PELCB100_REMOTE_PATH}"
SOURCE_HOST_FILTERS = {
    "new": {"${REMOTE_USER}@private-4e2f23ff.invalid"},
    "newnew": {"${REMOTE_USER}@private-169fa2c7.invalid"},
    "old": {"${REMOTE_USER}@private-3f2a50db.invalid"},
    "all": set(),
}
SOURCE_HOST_TO_REMOTE = {
    "new": "${REMOTE_USER}@private-4e2f23ff.invalid",
    "newnew": "${REMOTE_USER}@private-169fa2c7.invalid",
    "old": "${REMOTE_USER}@private-3f2a50db.invalid",
}

CURRENT_OBSERVABLES = ["Rg", "RDF", "MSD", "shape", "dynamics", "fsqt", "contact", "density"]
SUITE_BY_OBSERVABLE = {
    "density": "baseline_density",
    "Rg": "baseline_rg",
    "RDF": "deep_rdf",
    "MSD": "msd_monomer_chain",
    "shape": "deep_shape",
    "dynamics": "deep_dynamics",
    "fsqt": "deep_dynamics",
    "contact": "deep_contact",
}
STATUS_COLUMN = {
    "density": "status_density",
    "Rg": "status_Rg",
    "RDF": "status_RDF",
    "MSD": "status_MSD",
    "shape": "status_shape",
    "dynamics": "status_dynamics",
    "fsqt": "status_fsqt_tau_alpha",
    "contact": "status_contact",
}
SUITE_RESOURCES = {
    # QoS user_${REMOTE_USER} MaxJobs=10 has no corresponding CPU cap on the old server.
    # Per-case arrays still consume one QoS job slot per running array element,
    # so the small-CPU suites are packed below instead of competing with clean
    # MD and deep RDF for the ten available slots.
    "baseline_density": {"cpus": 2, "concurrency": None, "minutes": 60},
    "baseline_rg": {"cpus": 4, "concurrency": None, "minutes": 240},
    "deep_rdf": {"cpus": 32, "concurrency": 4, "minutes": 720},
    # Backward-compatible alias for in-flight or historical postprocess bundles.
    "baseline_rdf": {"cpus": 32, "concurrency": 4, "minutes": 720},
    "msd_monomer_chain": {"cpus": 8, "concurrency": 2, "minutes": 720},
    "deep_shape": {"cpus": 8, "concurrency": None, "minutes": 720},
    "deep_dynamics": {"cpus": 8, "concurrency": 3, "minutes": 720},
    "deep_contact": {"cpus": 8, "concurrency": 3, "minutes": 720},
}

FAST_BUNDLE_SUITES = (
    "baseline_density",
    "baseline_rg",
    "deep_contact",
    "deep_dynamics",
    "deep_shape",
    "msd_monomer_chain",
)
FAST_BUNDLE_CPUS = 48
FAST_BUNDLE_CONCURRENCY = 6
OLD_SERVER_QOS_MAX_RUNNING_JOBS = 10


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


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def read_manifest_metadata(manifest_csv: Path) -> dict[str, dict[str, str]]:
    """Load manifest rows keyed by case_id for staged metadata backfill."""

    if not manifest_csv.exists():
        return {}
    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        out[case_id] = {str(key): str(value) for key, value in row.items() if not is_blank(value)}
    return out


def metadata_backfill_for_staging(
    manifest_csv: Path,
    staging_rows: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Return clean-case metadata patches from the immutable source manifest.

    Clean-engine sidecars generated before the metadata-merge fix can contain a
    provenance-only ``case.json``.  The staged postprocess copy is safe to
    repair from the manifest because the manifest is the design source of truth
    and the clean ``case_id`` remains protected during patching.
    """

    manifest_by_case = read_manifest_metadata(manifest_csv)
    if not manifest_by_case:
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in staging_rows:
        clean_case_id = str(row.get("case_id", "")).strip()
        if not clean_case_id:
            continue
        source_case_id = str(row.get("source_case_id", "") or clean_case_id).strip()
        manifest_row = manifest_by_case.get(source_case_id) or manifest_by_case.get(clean_case_id)
        if not manifest_row:
            continue
        backfill = dict(manifest_row)
        backfill["metadata_backfill_source_case_id"] = source_case_id
        backfill["metadata_backfill_manifest_csv"] = str(manifest_csv)
        out[clean_case_id] = backfill
    return out


def patch_case_metadata_from_backfill(
    case_metadata: dict[str, Any],
    backfill: dict[str, Any],
) -> dict[str, Any]:
    """Fill missing staged case.json metadata without overwriting provenance."""

    protected = {
        "case_id",
        "source_case_id",
        "control_kind",
        "created_at",
        "pipeline_source",
        "run_tag",
        "source_dir",
        "source_host",
        "source_host_label",
        "local_run_dir",
        "remote_run_dir",
    }
    patched = dict(case_metadata or {})
    added: list[str] = []
    for key, value in dict(backfill or {}).items():
        if is_blank(value) or key in protected:
            continue
        if is_blank(patched.get(key)):
            patched[key] = value
            added.append(str(key))
    if added:
        patched["metadata_backfilled"] = True
        patched["metadata_backfill_fields"] = sorted(added)
    return patched


def is_missing(value: Any) -> bool:
    text = str(value if value is not None else "MISSING").strip().upper()
    return text.startswith("MISSING") or text in {
        "",
        "FAIL",
        "FAILED",
        "NAN",
    }


def split_semicolon(value: Any) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


SOURCE_PREFERENCE = ["new", "newnew"]

# Path-substring tokens that mark a deployed run as smoke/sandbox rather than
# production. When the dashboard reports multiple deployed_paths for the same
# case, prefer real production paths over these. Real example: case
# lcb_pe_005_bp01_bc020_pdmid_c100 has both an o7smoke run (1-frame trajectory)
# and a production o7_new_cuda2024_stage1_20260526 run; without filtering, the
# 'latest_by_remote' dedup would silently pick the smoke run because it appears
# later in the deployed_paths list.
SOURCE_PATH_ANTIPREFER_TOKENS = ("o7smoke", "/smoke/", "_smoke_", "/sandbox/")


def is_antiprefer_source_path(path: str) -> bool:
    """Return True if a deployed_path should be deprioritized as a smoke/sandbox run."""
    return any(tok in path for tok in SOURCE_PATH_ANTIPREFER_TOKENS)


def choose_source(row: pd.Series, allowed_remote_hosts: set[str] | None = None) -> tuple[str, str]:
    case_id = str(row.get("case_id", ""))
    hosts = split_semicolon(row.get("deployed_hosts", ""))
    paths = split_semicolon(row.get("deployed_paths", ""))
    if not hosts or not paths:
        return "", ""
    if len(hosts) != len(paths):
        if len(hosts) == 1 and len(paths) > 1:
            # Recovery can preserve multiple run directories for the same host while
            # the dashboard stores a deduplicated host label. Use the newest path
            # for postprocessing and keep older paths as provenance evidence.
            hosts = hosts * len(paths)
        else:
            raise RuntimeError(
                f"{case_id}: deployed_hosts/deployed_paths length mismatch; "
                f"hosts={hosts}, paths={paths}"
            )
    pairs = [(SOURCE_HOST_TO_REMOTE.get(h, h), p, h) for h, p in zip(hosts, paths)]
    if allowed_remote_hosts:
        pairs = [p for p in pairs if p[0] in allowed_remote_hosts]
    if not pairs:
        return "", ""
    # Deprioritize smoke/sandbox paths when production alternatives exist.
    production_pairs = [t for t in pairs if not is_antiprefer_source_path(t[1])]
    if production_pairs:
        pairs = production_pairs
    latest_by_remote: dict[str, tuple[str, str, str]] = {}
    for r, p, h in pairs:
        latest_by_remote[r] = (r, p, h)
    pairs = list(latest_by_remote.values())
    if len(pairs) == 1:
        return pairs[0][0], pairs[0][1]
    preferred_remotes = [SOURCE_HOST_TO_REMOTE.get(k) for k in SOURCE_PREFERENCE]
    for remote in preferred_remotes:
        for r, p, _ in pairs:
            if r == remote:
                return r, p
    return pairs[0][0], pairs[0][1]


def build_plan(
    runtime_cases: pd.DataFrame,
    max_cases: int | None = None,
    allowed_source_hosts: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if runtime_cases.empty:
        raise ValueError("runtime_cases.csv is empty")
    duplicate_rows = runtime_cases[runtime_cases.get("duplicate_reason", "").fillna("").astype(str) != ""]
    if len(duplicate_rows):
        raise RuntimeError("dashboard contains duplicate/conflict cases; resolve before postprocessing")

    case_rows = runtime_cases[runtime_cases["strict_done"].astype(str).str.lower().isin({"true", "1"})].copy()
    case_rows = case_rows.sort_values("case_seq") if "case_seq" in case_rows.columns else case_rows.sort_values("case_id")
    if max_cases:
        case_rows = case_rows.head(max_cases)

    jobs: list[dict[str, Any]] = []
    staging: dict[str, dict[str, Any]] = {}
    for _, row in case_rows.iterrows():
        case_id = str(row["case_id"])
        source_host, source_path = choose_source(row, allowed_source_hosts)
        if not source_host or not source_path:
            continue
        if allowed_source_hosts and source_host not in allowed_source_hosts:
            continue
        dest = f"{OLD_INPUT}/{Path(source_path).name}"
        source_case_id = str(row.get("source_case_id", "") or case_id)
        missing = [
            observable
            for observable in CURRENT_OBSERVABLES
            if is_missing(row.get(STATUS_COLUMN[observable], "MISSING"))
        ]
        if not missing:
            continue
        staging[case_id] = {
            "case_id": case_id,
            "source_case_id": source_case_id,
            "source_host": source_host,
            "source_path": source_path,
            "dest_path": dest,
            "missing_observables": ";".join(missing),
        }
        for observable in missing:
            if observable == "dynamics" and is_missing(row.get(STATUS_COLUMN["RDF"], "MISSING")):
                continue
            # Contact QC uses fixed cutoffs (rc0.6/0.8/1.0) as the
            # publication-blocking evidence; the RDF-derived cutoff is an
            # optional side channel.  Schedule contact in the same wave as RDF
            # so full clean releases are not held by an avoidable second
            # postprocessing pass.
            suite = SUITE_BY_OBSERVABLE[observable]
            jobs.append(
                {
                    "case_id": case_id,
                    "observable": observable,
                    "suite": suite,
                    "source_host": source_host,
                    "source_path": source_path,
                    "dest_path": dest,
                    "status_column": STATUS_COLUMN[observable],
                }
            )
    return jobs, list(staging.values())


def case_list_by_suite(jobs: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for job in jobs:
        suite = str(job["suite"])
        case_id = str(job["case_id"])
        key = (suite, case_id)
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(suite, []).append({"case_id": case_id, "case_dir": str(job["dest_path"])})
    return out


def write_case_lists(output_dir: Path, by_suite: dict[str, list[dict[str, str]]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    lists_dir = output_dir / "case_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    all_rows = sorted(
        {str((row["case_id"], row["case_dir"])): row for rows in by_suite.values() for row in rows}.values(),
        key=lambda row: row["case_id"],
    )
    all_path = lists_dir / "cases_all.tsv"
    write_tsv(all_path, all_rows, ["case_id", "case_dir"])
    paths["all"] = all_path
    for suite, rows in sorted(by_suite.items()):
        path = lists_dir / f"{suite}.tsv"
        write_tsv(path, rows, ["case_id", "case_dir"])
        paths[suite] = path
    return paths


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def stage_script() -> str:
    script = r"""#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-staging_manifest.tsv}"
DEST_ROOT="${DEST_ROOT:-__OLD_INPUT__}"
LOG_DIR="${LOG_DIR:-__OLD_WORK__/logs}"
OLD_SERVER="${OLD_SERVER:-__OLD_SERVER__}"
PYTHON_BIN="${PYTHON_BIN:-__OLD_PYTHON__}"
METADATA_BACKFILL="${METADATA_BACKFILL:-metadata_backfill.json}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20}"
STAGE_PARALLEL="${STAGE_PARALLEL:-4}"
RSYNC_COMPARE_ARGS="${RSYNC_COMPARE_ARGS:---size-only}"
mkdir -p "${DEST_ROOT}" "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_DIR}/stage___PROJECT_ID___${STAMP}.tsv"
OK_CASES="${LOG_DIR}/staged_ok_cases___PROJECT_ID___${STAMP}.tsv"
printf "case_id\tstatus\tdest\tmessage\n" > "${SUMMARY}"
printf "case_id\tdest_path\n" > "${OK_CASES}"

stage_one() {
  local line="$1"
  local case_id source_host source_path dest_path missing
  IFS=$'\t' read -r case_id source_host source_path dest_path missing <<< "${line}"
  [[ -z "${case_id}" || "${case_id}" == "case_id" ]] && return 0
  mkdir -p "${dest_path}"
  echo "STAGE ${case_id} from ${source_host}:${source_path}"
  local completion_check
  completion_check="(test -s '${source_path}/nvt_prod/nvt_prod.tpr' || test -s '${source_path}/nvt_prod.tpr') && (ls '${source_path}'/nvt_prod/nvt_prod*.xtc >/dev/null 2>&1 || ls '${source_path}'/nvt_prod*.xtc >/dev/null 2>&1) && (ls '${source_path}'/nvt_prod/nvt_prod*.gro >/dev/null 2>&1 || ls '${source_path}'/nvt_prod*.gro >/dev/null 2>&1 || test -s '${source_path}/nvt_prod/all.gro' || test -s '${source_path}/all.gro') && (grep -q 'Finished mdrun' '${source_path}'/nvt_prod/nvt_prod*.log 2>/dev/null || grep -q 'Finished mdrun' '${source_path}'/nvt_prod*.log 2>/dev/null)"
  if [[ "${source_host}" == "${OLD_SERVER}" ]]; then
    if ! bash -c "${completion_check}"; then
      printf "%s\tFAIL_COMPLETION_GATE\t%s\tstrict completion or required files missing\n" "${case_id}" "${dest_path}" >> "${SUMMARY}"
      return 0
    fi
  elif ! ssh ${SSH_OPTS} -n "${source_host}" "${completion_check}" < /dev/null; then
    printf "%s\tFAIL_COMPLETION_GATE\t%s\tstrict completion or required files missing\n" "${case_id}" "${dest_path}" >> "${SUMMARY}"
    return 0
  fi
  local -a rsync_source
  if [[ "${source_host}" == "${OLD_SERVER}" ]]; then
    rsync_source=("${source_path}/")
  else
    rsync_source=("${source_host}:${source_path}/")
  fi
  # Clean MD trajectories are large and immutable after the strict completion
  # gate above.  Size-based incremental staging avoids repeatedly checksumming
  # multi-GB trajectories while still re-copying truncated partial files.
  # shellcheck disable=SC2086
  if rsync -a --partial ${RSYNC_COMPARE_ARGS} -e "ssh ${SSH_OPTS}" \
      --include='/manual.out' \
      --include='/case.json' \
      --include='/all.gro' \
      --include='/nvt_prod.mdp' \
      --include='/mdout.mdp' \
      --include='/nvt_prod.tpr' \
      --include='/nvt_prod.cpt' \
      --include='/nvt_prod*.xtc' \
      --include='/nvt_prod*.edr' \
      --include='/nvt_prod*.gro' \
      --include='/nvt_prod*.log' \
      --include='/*.top' \
      --include='/*.itp' \
      --include='/nvt_prod/' \
      --include='/nvt_prod/nvt_prod.tpr' \
      --include='/nvt_prod/nvt_prod.mdp' \
      --include='/nvt_prod/mdout.mdp' \
      --include='/nvt_prod/nvt_prod*.xtc' \
      --include='/nvt_prod/nvt_prod*.edr' \
      --include='/nvt_prod/nvt_prod*.gro' \
      --include='/nvt_prod/all.gro' \
      --include='/nvt_prod/nvt_prod*.log' \
      --include='/nvt_prod/topol.top' \
      --include='/nvt_prod/*.itp' \
      --exclude='*' \
      "${rsync_source[@]}" "${dest_path}/" < /dev/null; then
    local alias_path="${DEST_ROOT}/${case_id}"
    if [[ "${alias_path}" != "${dest_path}" ]]; then
      rm -rf "${alias_path}"
      ln -s "${dest_path}" "${alias_path}"
    fi
    mkdir -p "${dest_path}/nvt_prod"
    if [[ -s "${dest_path}/nvt_prod.mdp" && ! -s "${dest_path}/nvt_prod/nvt_prod.mdp" ]]; then
      cp "${dest_path}/nvt_prod.mdp" "${dest_path}/nvt_prod/nvt_prod.mdp"
    fi
    if [[ -s "${dest_path}/mdout.mdp" && ! -s "${dest_path}/nvt_prod/mdout.mdp" ]]; then
      cp "${dest_path}/mdout.mdp" "${dest_path}/nvt_prod/mdout.mdp"
    fi
    if [[ -s "${dest_path}/all.gro" && ! -s "${dest_path}/nvt_prod/all.gro" && ! -s "${dest_path}/nvt_prod/nvt_prod.gro" ]]; then
      cp "${dest_path}/all.gro" "${dest_path}/nvt_prod/all.gro"
    fi
    shopt -s nullglob
    for staged_root_file in "${dest_path}"/nvt_prod.tpr "${dest_path}"/nvt_prod.cpt "${dest_path}"/nvt_prod*.xtc "${dest_path}"/nvt_prod*.edr "${dest_path}"/nvt_prod*.gro "${dest_path}"/nvt_prod*.log; do
      staged_base="$(basename "${staged_root_file}")"
      if [[ -f "${staged_root_file}" && ! -e "${dest_path}/nvt_prod/${staged_base}" ]]; then
        cp "${staged_root_file}" "${dest_path}/nvt_prod/${staged_base}"
      fi
    done
    shopt -u nullglob
    if [[ -s "${METADATA_BACKFILL}" && -s "${dest_path}/case.json" ]]; then
      CASE_ID="${case_id}" CASE_JSON="${dest_path}/case.json" METADATA_BACKFILL="${METADATA_BACKFILL}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys

def blank(value):
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}

protected = {
    "case_id",
    "source_case_id",
    "control_kind",
    "created_at",
    "pipeline_source",
    "run_tag",
    "source_dir",
    "source_host",
    "source_host_label",
    "local_run_dir",
    "remote_run_dir",
}
case_id = os.environ["CASE_ID"]
case_json = os.environ["CASE_JSON"]
backfill_path = os.environ["METADATA_BACKFILL"]
with open(backfill_path, "r", encoding="utf-8") as handle:
    backfill_payload = json.load(handle)
entry = backfill_payload.get(case_id, {})
if entry:
    with open(case_json, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    added = []
    for key, value in entry.items():
        if key in protected or blank(value):
            continue
        if blank(metadata.get(key)):
            metadata[key] = value
            added.append(key)
    if added:
        metadata["metadata_backfilled"] = True
        metadata["metadata_backfill_fields"] = sorted(added)
        with open(case_json, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print("metadata_backfilled", case_id, ",".join(sorted(added)))
PY
    fi
    printf "%s\tOK\t%s\tstaged\n" "${case_id}" "${dest_path}" >> "${SUMMARY}"
    printf "%s\t%s\n" "${case_id}" "${dest_path}" >> "${OK_CASES}"
  else
    printf "%s\tFAIL_RSYNC\t%s\trsync failed\n" "${case_id}" "${dest_path}" >> "${SUMMARY}"
  fi
}

export -f stage_one
export DEST_ROOT LOG_DIR OLD_SERVER PYTHON_BIN METADATA_BACKFILL SSH_OPTS SUMMARY OK_CASES
tail -n +2 "${MANIFEST}" | xargs -I{} -P "${STAGE_PARALLEL}" bash -c 'stage_one "$0"' {}
cp "${OK_CASES}" "${LOG_DIR}/staged_ok_cases_latest.tsv"
if grep -q $'\tFAIL' "${SUMMARY}"; then
  echo "Staging failures detected in ${SUMMARY}; continuing with staged OK cases only." >&2
fi
echo "Wrote ${SUMMARY}"
echo "Wrote ${OK_CASES}"
"""
    return (
        script.replace("__OLD_INPUT__", OLD_INPUT)
        .replace("__OLD_WORK__", OLD_WORK)
        .replace("__OLD_SERVER__", OLD_SERVER)
        .replace("__OLD_PYTHON__", OLD_PYTHON)
        .replace("__PROJECT_ID__", PROJECT_ID)
    )


def suite_command(suite: str) -> str:
    base = r'''if [[ -z "${case_dir}" || ! -d "${case_dir}" ]]; then
  echo "Missing exact staged case directory for ${case_id}: ${case_dir}" >&2
  exit 2
fi
'''
    if suite == "baseline_rg":
        return base + r'''
"${PYTHON_BIN}" "${CODE}/run_rg_polystat.py" --case-id "${case_id}" --case-dir "${case_dir}" --gmx-bin "${GMX}" --output-root "${OUTPUT}"
'''
    if suite == "baseline_density":
        return base + r'''
"${PYTHON_BIN}" "${CODE}/analyze_nvt_prod.py" --case-dir "${case_dir}" --gmx-bin "${GMX}" --output-dir "${OUTPUT}/${PROJECT_ID}_density_energy/${case_id}" --target-temperature 400 --scalars-only
'''
    if suite in {"deep_rdf", "baseline_rdf"}:
        return base + r'''
"${PYTHON_BIN}" "${CODE}/compute_full_rdf_suite.py" --case-dir "${case_dir}" --gmx-bin "${GMX}" --output-root "${OUTPUT}" --chain-count 150 --dt-ps 1000 --force
'''
    if suite == "msd_monomer_chain":
        return base + r'''
"${PYTHON_BIN}" "${CODE}/analyze_trajectory.py" --case-dir "${case_dir}" --gmx-bin "${GMX}" --output-root "${OUTPUT}" --analyses msd,ree --chain-count 150 --msd-beginfit-ps 100000 --msd-endfit-ps 300000 --msd-trestart-ps 1000 --trajectory-end-ps 800000 --force
"${PYTHON_BIN}" "${CODE}/compute_monomer_msd.py" --input-root "${INPUT}" --batch-dir "${OUTPUT}/${PROJECT_ID}_integrated/IntrachainDynamics" --gmx "${GMX}" --cases "${case_id}" --dt-ps 1000 --trajectory-end-ps 800000 --no-aggregate --force --continue-on-error
'''
    if suite == "deep_shape":
        return base + r'''
"${PYTHON_BIN}" "${CODE}/compute_shape_tensor.py" --case-id "${case_id}" --case-dir "${case_dir}" --out-dir "${OUTPUT}/${PROJECT_ID}_integrated/ShapeTensor/raw/cases/${case_id}/shape_tensor" --gmx "${GMX}" --n-chains 150 --dt-ps 1000 --force
'''
    if suite == "deep_dynamics":
        return base + r'''
"${PYTHON_BIN}" "${CODE}/analyze_dynamics.py" --input-root "${INPUT}" --output-root "${OUTPUT}" --cases "${case_id}" --gmx-bin "${GMX}" --rdf-summary "${OUTPUT}/${PROJECT_ID}_integrated/RDF/summary/rdf_summary.csv" --force
'''
    if suite == "deep_contact":
        return base + r'''
"${PYTHON_BIN}" "${CODE}/analyze_intrachain_dynamics.py" --input-root "${INPUT}" --batch-dir "${OUTPUT}/${PROJECT_ID}_integrated/IntrachainDynamics" --gmx "${GMX}" --cases "${case_id}" --dt-ps 1000 --rdf-summary "${OUTPUT}/${PROJECT_ID}_integrated/RDF/summary/rdf_summary.csv" --dynamics-summary "${OUTPUT}/${PROJECT_ID}_dynamics_20260504/aggregate/dynamics_summary.csv" --force --no-aggregate
"${PYTHON_BIN}" "${CODE}/compute_monomer_msd.py" --input-root "${INPUT}" --batch-dir "${OUTPUT}/${PROJECT_ID}_integrated/IntrachainDynamics" --gmx "${GMX}" --cases "${case_id}" --dt-ps 1000 --trajectory-end-ps 800000 --no-aggregate --force --continue-on-error
'''
    raise ValueError(f"unknown suite {suite}")


def sbatch_script(suite: str, case_list_remote: str, case_count: int) -> str:
    resource = SUITE_RESOURCES[suite]
    if case_count <= 0:
        raise ValueError(f"cannot build sbatch script for empty suite {suite}")
    array = f"0-{case_count - 1}"
    if resource.get("concurrency"):
        array = f"{array}%{resource['concurrency']}"
    return f"""#!/usr/bin/env bash
#SBATCH -J {PROJECT_ID}_{suite}
#SBATCH -p amd,intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c {resource['cpus']}
#SBATCH --array={array}
#SBATCH --time=30-00:00:00
#SBATCH --no-requeue
#SBATCH -o {OLD_WORK}/logs/{suite}_%A_%a.out
#SBATCH -e {OLD_WORK}/logs/{suite}_%A_%a.err

set -euo pipefail
PYTHON_BIN="${{PYTHON_BIN:-{OLD_PYTHON}}}"
GMX="${{GMX:-{OLD_GMX}}}"
CODE="${{CODE:-{OLD_CODE}}}"
INPUT="${{INPUT:-{OLD_INPUT}}}"
OUTPUT="${{OUTPUT:-{OLD_OUTPUT}}}"
PROJECT_ID="${{PROJECT_ID:-{PROJECT_ID}}}"
export LCB_PROJECT_ID="${{LCB_PROJECT_ID:-${{PROJECT_ID}}}}"
CASE_LIST="${{CASE_LIST:-{case_list_remote}}}"
mkdir -p "{OLD_WORK}/logs" "${{OUTPUT}}"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{resource['cpus']}}}"
"${{PYTHON_BIN}}" --version
echo "GMX=${{GMX}}"
"${{GMX}}" --version | head -20 || true
row="$(awk -F $'\\t' -v n="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==n {{print}}' "${{CASE_LIST}}")"
IFS=$'\\t' read -r case_id case_dir <<< "${{row}}"
if [[ -z "${{case_id}}" ]]; then
  echo "No case for task ${{SLURM_ARRAY_TASK_ID}}"
  exit 0
fi
if [[ -z "${{case_dir}}" || ! -d "${{case_dir}}" ]]; then
  echo "Missing exact staged case directory for ${{case_id}}: ${{case_dir}}" >&2
  exit 2
fi
echo "CASE ${{case_id}} SUITE {suite} START $(date)"
{suite_command(suite)}
echo "CASE ${{case_id}} SUITE {suite} DONE $(date)"
"""


def fast_bundle_script(fast_suites: list[str]) -> str:
    """Pack <=8-CPU suites into one Slurm allocation.

    The old-server QoS limits *running jobs*, while idle CPU capacity remains
    available on whole nodes.  A single allocation with six internal workers
    therefore uses the scheduler honestly and prevents independent arrays from
    starving one another behind QOSMaxJobsPerUserLimit.
    """

    suites = [suite for suite in fast_suites if suite in FAST_BUNDLE_SUITES]
    if not suites:
        raise ValueError("cannot build an empty fast postprocess bundle")
    branches: list[str] = []
    for suite in suites:
        cpus = int(SUITE_RESOURCES[suite]["cpus"])
        branches.append(
            f"""    {suite})
      export OMP_NUM_THREADS={cpus}
{suite_command(suite).rstrip()}
      ;;"""
        )
    branch_text = "\n".join(branches)
    return f"""#!/usr/bin/env bash
#SBATCH -J {PROJECT_ID}_fast_bundle
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c {FAST_BUNDLE_CPUS}
#SBATCH --time=30-00:00:00
#SBATCH --no-requeue
#SBATCH -o {OLD_WORK}/logs/fast_bundle_%j.out
#SBATCH -e {OLD_WORK}/logs/fast_bundle_%j.err

set -euo pipefail
PYTHON_BIN="${{PYTHON_BIN:-{OLD_PYTHON}}}"
GMX="${{GMX:-{OLD_GMX}}}"
CODE="${{CODE:-{OLD_CODE}}}"
INPUT="${{INPUT:-{OLD_INPUT}}}"
OUTPUT="${{OUTPUT:-{OLD_OUTPUT}}}"
PROJECT_ID="${{PROJECT_ID:-{PROJECT_ID}}}"
FAST_LIST="${{FAST_LIST:-{OLD_WORK}/case_lists/fast_bundle.tsv}}"
FAST_CONCURRENCY="${{FAST_CONCURRENCY:-{FAST_BUNDLE_CONCURRENCY}}}"
export PYTHON_BIN GMX CODE INPUT OUTPUT PROJECT_ID FAST_LIST
export LCB_PROJECT_ID="${{LCB_PROJECT_ID:-${{PROJECT_ID}}}}"
mkdir -p "{OLD_WORK}/logs" "${{OUTPUT}}"

run_one() {{
  local suite="$1"
  local case_id="$2"
  local case_dir="$3"
  if [[ -z "${{case_id}}" || -z "${{case_dir}}" || ! -d "${{case_dir}}" ]]; then
    echo "Missing exact staged case directory for ${{case_id}}: ${{case_dir}}" >&2
    return 2
  fi
  echo "CASE ${{case_id}} SUITE ${{suite}} START $(date)"
  case "${{suite}}" in
{branch_text}
    *)
      echo "Unknown fast-bundle suite: ${{suite}}" >&2
      return 2
      ;;
  esac
  echo "CASE ${{case_id}} SUITE ${{suite}} DONE $(date)"
}}
export -f run_one

# NUL-delimited records preserve exact paths while xargs supplies a bounded
# worker pool inside one scheduler allocation.
tail -n +2 "${{FAST_LIST}}" |
  while IFS=$'\t' read -r suite case_id case_dir; do
    printf '%s\\0%s\\0%s\\0' "${{suite}}" "${{case_id}}" "${{case_dir}}"
  done |
  xargs -0 -r -n 3 -P "${{FAST_CONCURRENCY}}" bash -c 'set -euo pipefail; run_one "$1" "$2" "$3"' _
"""


def write_oldserver_scripts(
    output_dir: Path,
    staging_rows: list[dict[str, Any]],
    by_suite: dict[str, list[dict[str, str]]],
    manifest_csv: Path | None = None,
) -> dict[str, str]:
    scripts = output_dir / "oldserver_scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    staging_path = scripts / "staging_manifest.tsv"
    write_tsv(
        staging_path,
        staging_rows,
        ["case_id", "source_host", "source_path", "dest_path", "missing_observables"],
    )
    metadata_backfill = metadata_backfill_for_staging(manifest_csv, staging_rows) if manifest_csv else {}
    write_json(scripts / "metadata_backfill.json", metadata_backfill)
    write_text_lf(scripts / "stage_from_manifest.sh", stage_script())
    generated = {
        "staging_manifest": str(staging_path),
        "stage_script": str(scripts / "stage_from_manifest.sh"),
        "metadata_backfill_json": str(scripts / "metadata_backfill.json"),
        "metadata_backfill_count": str(len(metadata_backfill)),
    }
    for suite, rows in sorted(by_suite.items()):
        if not rows:
            continue
        local_list = scripts / f"{suite}.tsv"
        write_tsv(local_list, rows, ["case_id", "case_dir"])
        remote_list = f"{OLD_WORK}/case_lists/{suite}.tsv"
        script_path = scripts / f"run_{suite}.sbatch"
        write_text_lf(script_path, sbatch_script(suite, remote_list, len(rows)))
        generated[f"{suite}_cases"] = str(local_list)
        generated[f"{suite}_sbatch"] = str(script_path)
    fast_suites = [suite for suite in FAST_BUNDLE_SUITES if by_suite.get(suite)]
    if fast_suites:
        fast_rows = [
            {"suite": suite, "case_id": row["case_id"], "case_dir": row["case_dir"]}
            for suite in fast_suites
            for row in by_suite[suite]
        ]
        fast_list = scripts / "fast_bundle.tsv"
        write_tsv(fast_list, fast_rows, ["suite", "case_id", "case_dir"])
        fast_script = scripts / "run_fast_bundle.sbatch"
        write_text_lf(fast_script, fast_bundle_script(fast_suites))
        generated["fast_bundle_cases"] = str(fast_list)
        generated["fast_bundle_sbatch"] = str(fast_script)
    submit_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"REMOTE_WORK={shell_quote(OLD_WORK)}",
        "mkdir -p \"${REMOTE_WORK}/case_lists\" \"${REMOTE_WORK}/logs\"",
        "bash stage_from_manifest.sh staging_manifest.tsv",
        "OK_CASES=\"${REMOTE_WORK}/logs/staged_ok_cases_latest.tsv\"",
        "if [[ ! -s \"${OK_CASES}\" || $(wc -l < \"${OK_CASES}\") -le 1 ]]; then echo \"No staged OK cases: ${OK_CASES}\" >&2; exit 2; fi",
    ]
    for suite in sorted(by_suite):
        if by_suite[suite]:
            submit_lines.append(
                f"awk -F $'\\t' 'NR==FNR {{if (FNR>1) ok[$1]=$2; next}} "
                f"FNR==1 {{print; next}} ($1 in ok) {{print $1 \"\\t\" ok[$1]}}' "
                f"\"${{OK_CASES}}\" {suite}.tsv > \"${{REMOTE_WORK}}/case_lists/{suite}.tsv\""
            )
            submit_lines.append(f"case_count=$(( $(wc -l < \"${{REMOTE_WORK}}/case_lists/{suite}.tsv\") - 1 ))")
    if fast_suites:
        submit_lines.extend(
            [
                f"printf 'suite\\tcase_id\\tcase_dir\\n' > \"${{REMOTE_WORK}}/case_lists/fast_bundle.tsv\"",
                *[
                    f"tail -n +2 \"${{REMOTE_WORK}}/case_lists/{suite}.tsv\" | "
                    f"awk -F $'\\t' -v suite={shell_quote(suite)} "
                    "'{print suite \"\\t\" $1 \"\\t\" $2}' >> "
                    '"${REMOTE_WORK}/case_lists/fast_bundle.tsv"'
                    for suite in fast_suites
                ],
                'fast_count=$(( $(wc -l < "${REMOTE_WORK}/case_lists/fast_bundle.tsv") - 1 ))',
                f"max_jobs=${{O7_MAX_RUNNING_JOBS:-{OLD_SERVER_QOS_MAX_RUNNING_JOBS}}}",
                "running_jobs=$(squeue -h -u \"${USER}\" -t RUNNING -o '%i' | wc -l)",
                "available_slots=$(( max_jobs - running_jobs ))",
                "if (( available_slots < 1 )); then available_slots=1; fi",
                "rdf_slots=$(( available_slots - (fast_count > 0 ? 1 : 0) ))",
                "if (( rdf_slots < 1 )); then rdf_slots=1; fi",
                f"if (( rdf_slots > {SUITE_RESOURCES['deep_rdf']['concurrency']} )); then "
                f"rdf_slots={SUITE_RESOURCES['deep_rdf']['concurrency']}; fi",
                'echo "QoS plan: running=${running_jobs} max=${max_jobs} fast_count=${fast_count} rdf_slots=${rdf_slots}"',
                "if (( fast_count > 0 )); then sbatch run_fast_bundle.sbatch; fi",
            ]
        )
    for suite in sorted(by_suite):
        if not by_suite[suite] or suite in fast_suites:
            continue
        if suite in {"deep_rdf", "baseline_rdf"}:
            submit_lines.append(
                f"case_count=$(( $(wc -l < \"${{REMOTE_WORK}}/case_lists/{suite}.tsv\") - 1 ))"
            )
            submit_lines.append(
                f"if (( case_count <= 0 )); then echo \"Skip {suite}: no successfully staged cases\"; "
                f"else sbatch --array=0-$((case_count - 1))%${{rdf_slots:-1}} run_{suite}.sbatch; fi"
            )
        else:
            submit_lines.append(
                f"case_count=$(( $(wc -l < \"${{REMOTE_WORK}}/case_lists/{suite}.tsv\") - 1 ))"
            )
            submit_lines.append(
                f"if (( case_count <= 0 )); then echo \"Skip {suite}: no successfully staged cases\"; "
                f"else sbatch run_{suite}.sbatch; fi"
            )
    submit_path = scripts / "submit_all_on_oldserver.sh"
    write_text_lf(submit_path, "\n".join(submit_lines) + "\n")
    generated["submit_script"] = str(submit_path)
    return generated


def execute_oldserver_bundle(scripts_dir: Path) -> None:
    remote_dir = f"{OLD_WORK}/orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    subprocess.run(["ssh", OLD_SERVER, "mkdir", "-p", remote_dir], check=True)
    with tempfile.TemporaryDirectory(prefix=f"{PROJECT_ID}_code_sync_") as tmpdir:
        archive = Path(tmpdir) / "md_analysis_current.tgz"
        excluded_dirs = {
            "output",
            "__pycache__",
            ".pytest_cache",
            "manuscripts",
            "paper_workspace",
            "reports",
            "tests",
        }
        allowed_suffixes = {
            ".py",
            ".sh",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".mdp",
            ".ndx",
            ".txt",
            ".csv",
            ".tsv",
        }
        with tarfile.open(archive, "w:gz") as tar:
            for root, dirs, files in os.walk(SCRIPT_DIR):
                dirs[:] = [name for name in dirs if name not in excluded_dirs]
                root_path = Path(root)
                for filename in files:
                    item = root_path / filename
                    if item.suffix.lower() not in allowed_suffixes:
                        continue
                    rel = item.relative_to(SCRIPT_DIR)
                    tar.add(item, arcname=str(rel).replace("\\", "/"))
        remote_archive = f"{remote_dir}/md_analysis_current.tgz"
        subprocess.run(["scp", str(archive), f"{OLD_SERVER}:{remote_archive}"], check=True)
        subprocess.run(
            [
                "ssh",
                OLD_SERVER,
                (
                    f"rm -rf {shell_quote(OLD_CODE)} && "
                    f"mkdir -p {shell_quote(OLD_CODE)} && "
                    f"mkdir -p {shell_quote(OLD_OUTPUT)} && "
                    f"tar -xzf {shell_quote(remote_archive)} -C {shell_quote(OLD_CODE)}"
                ),
            ],
            check=True,
        )
    subprocess.run(["scp", "-r", str(scripts_dir) + "/.", f"{OLD_SERVER}:{remote_dir}/"], check=True)
    subprocess.run(["ssh", OLD_SERVER, f"cd {shell_quote(remote_dir)} && bash submit_all_on_oldserver.sh"], check=True)


def safe_extract_tar(archive: Path, target_dir: Path) -> None:
    root = target_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            destination = (target_dir / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Refusing to extract path outside target directory: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                tar.extract(member, target_dir, filter="data")
            except TypeError:
                tar.extract(member, target_dir)


def collection_staging_base(output_dir: Path, *, platform_name: str | None = None) -> Path:
    platform_name = platform_name or os.name
    resolved = output_dir.resolve()
    # Deep per-case postprocessing paths can add >140 characters under the
    # collection root.  Keep the Windows collection base short enough that
    # nested IntrachainDynamics/RDF files stay below the legacy MAX_PATH edge.
    if platform_name == "nt" and len(str(resolved)) > 80:
        return SCRIPT_DIR / "output" / "_remote_collect_staging" / PROJECT_ID
    return output_dir


def collect_remote_outputs(output_dir: Path) -> Path:
    """Copy current old-server postprocessing outputs into a local staging dir.

    This intentionally does not merge into the canonical LCB300 integrated tree.
    It gives us an auditable boundary between remote Slurm products and the
    local formal aggregate rebuild.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    collect_base = collection_staging_base(output_dir)
    collect_base.mkdir(parents=True, exist_ok=True)
    collect_dir = collect_base / datetime.now().strftime("remote_collect_%Y%m%d_%H%M%S")
    collect_dir.mkdir(parents=True, exist_ok=True)
    remote_archive = f"{OLD_WORK}/logs/{PROJECT_ID}_postprocess_collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tgz"
    remote_list = f"{OLD_WORK}/logs/{PROJECT_ID}_postprocess_collect_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    integrated_dir = f"{PROJECT_ID}_integrated"
    remote_cmd = (
        f"cd {shell_quote(OLD_OUTPUT)} && "
        "{ "
        f"find {PROJECT_ID}_density_energy {PROJECT_ID}_rg_* {PROJECT_ID}_rdf_* "
        f"{PROJECT_ID}_msd_* {PROJECT_ID}_ree_* {PROJECT_ID}_trajectory_* "
        f"{PROJECT_ID}_dynamics_* {integrated_dir}/IntrachainDynamics "
        f"{integrated_dir}/ShapeTensor {integrated_dir}/RDF -type f 2>/dev/null || true; "
        "} | grep -Ev '(^|/)(#.*#|nvt_prod.*\\.(xtc|trr|edr|tpr|gro|cpt)$|.*_coords_dt[0-9]+\\.xvg$|.*coords\\.xvg$|nvt_prod_energy_terms\\.xvg$)' "
        f"> {shell_quote(remote_list)} && "
        # Coordinate scratch files can disappear while postprocessing cleanup runs;
        # formal CSV/JSON/XVG products are still collected from the manifest list.
        f"tar --ignore-failed-read -czf {shell_quote(remote_archive)} -T {shell_quote(remote_list)}"
    )
    subprocess.run(["ssh", OLD_SERVER, remote_cmd], check=True)
    local_archive = collect_dir / "remote_outputs.tgz"
    subprocess.run(["scp", f"{OLD_SERVER}:{remote_archive}", str(local_archive)], check=True)
    safe_extract_tar(local_archive, collect_dir)
    manifest = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "old_server": OLD_SERVER,
        "remote_code_dir": OLD_CODE,
        "remote_output_dir": OLD_OUTPUT,
        "remote_archive": remote_archive,
        "local_archive": str(local_archive),
        "project_id": PROJECT_ID,
        "note": f"Staging copy only; canonical {PROJECT_ID}_integrated was not modified.",
    }
    write_json(collect_dir / "collect_manifest.json", manifest)
    write_json(
        output_dir / "remote_collect_latest.json",
        {
            "collect_dir": str(collect_dir),
            "requested_output_dir": str(output_dir),
            "collect_base": str(collect_base),
        },
    )
    return collect_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="LCB project config JSON. Defaults to configs/lcb300.json.")
    parser.add_argument("--dashboard-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--source-host",
        choices=sorted(SOURCE_HOST_FILTERS),
        default="all",
        help="Limit planned/staged cases by data source. Use new when the old server cannot SSH into newnew.",
    )
    parser.add_argument("--execute", action="store_true", help="Upload scripts to old server and submit sbatch jobs.")
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect old-server postprocessing outputs into a local staging directory without merging.",
    )
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> None:
    global PROJECT_ID, OLD_SERVER, OLD_WORK, OLD_INPUT, OLD_CODE, OLD_OUTPUT
    global SOURCE_HOST_FILTERS, SOURCE_HOST_TO_REMOTE

    config = load_lcb_config(args.config)
    PROJECT_ID = config.project_id
    OLD_SERVER = config.old_server
    OLD_WORK = config.old_server_postprocess_root.rstrip("/")
    OLD_INPUT = config.old_server_input_root.rstrip("/")
    OLD_CODE = f"{OLD_WORK}/md_analysis_current"
    OLD_OUTPUT = f"{OLD_WORK}/results"
    SOURCE_HOST_TO_REMOTE = {
        "new": config.new_server,
        "newnew": config.newnew_server,
        "old": config.old_server,
    }
    SOURCE_HOST_FILTERS = {
        "new": {config.new_server},
        "newnew": {config.newnew_server},
        "old": {config.old_server},
        "all": set(),
    }
    if args.dashboard_dir is None:
        args.dashboard_dir = config.runtime_output_dir
    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "output" / f"{PROJECT_ID}_orchestrator"
    args.manifest_csv = config.manifest_csv


def main() -> None:
    args = parse_args()
    apply_config(args)
    if args.collect and not args.execute:
        collect_dir = collect_remote_outputs(args.output_dir)
        print(f"Collected remote outputs into {collect_dir}")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    runtime = read_csv(args.dashboard_dir / "runtime_cases.csv")
    allowed = SOURCE_HOST_FILTERS[args.source_host]
    jobs, staging_rows = build_plan(runtime, max_cases=args.max_cases, allowed_source_hosts=allowed)
    by_suite = case_list_by_suite(jobs)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        run_dir / "full_postprocess_plan.csv",
        jobs,
        ["case_id", "observable", "suite", "source_host", "source_path", "dest_path", "status_column"],
    )
    case_lists = write_case_lists(run_dir, by_suite)
    generated = write_oldserver_scripts(run_dir, staging_rows, by_suite, manifest_csv=args.manifest_csv)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_id": PROJECT_ID,
        "mode": "execute" if args.execute else "dry-run",
        "job_count": len(jobs),
        "case_count": len(staging_rows),
        "suite_counts": {suite: len(cases) for suite, cases in sorted(by_suite.items())},
        "source_host_filter": args.source_host,
        "viscosity_policy": "historical-only; no new viscosity jobs are generated",
        "case_lists": {key: str(value) for key, value in case_lists.items()},
        "oldserver_scripts": generated,
    }
    write_json(run_dir / "postprocess_run_manifest.json", manifest)
    if not args.json_only:
        lines = [
            f"# {PROJECT_ID.upper()} Postprocess Plan",
            "",
            f"- Generated: `{manifest['generated_at']}`",
            f"- Mode: `{manifest['mode']}`",
            f"- Cases staged/planned: `{manifest['case_count']}`",
            f"- Jobs: `{manifest['job_count']}`",
            "- Viscosity: historical only; no new jobs",
            "",
            "## Suite Counts",
            "",
        ]
        for suite, count in manifest["suite_counts"].items():
            lines.append(f"- `{suite}`: {count}")
        (run_dir / "postprocess_run_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.execute:
        execute_oldserver_bundle(run_dir / "oldserver_scripts")
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()

