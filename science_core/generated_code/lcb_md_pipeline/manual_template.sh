#!/usr/bin/env bash

# Manual GROMACS run template for servers without Slurm.
#
# Resume-aware 4-stage pipeline (steep -> rlx -> pr -> nvt_prod).
# Re-running this script resumes from the latest checkpoint instead of
# restarting from steep. State detection is per-stage:
#   - DONE: <stage>/<stage>.gro (or .part*.gro) + 'Finished mdrun' in log -> skip
#   - RESUME: <stage>.tpr + <stage>.cpt present -> mdrun -cpi -noappend
#   - DIRTY_NO_CPT: <stage>.tpr, no .cpt, but stale .log/.xtc/.edr -> archive + fresh mdrun
#   - FRESH_MDRUN: <stage>.tpr only -> fresh mdrun
#   - FRESH_ALL: no .tpr -> copy + (editconf if steep) + grompp + mdrun
#
# IMPORTANT: run with bash, not sh. Uses shopt/arrays/local (bash-only).
#   bash run_manual.sh
#   nohup bash run_manual.sh > manual.out 2> manual.err &
#
# Environment switches:
#   FORCE_FRESH=1   Archive existing stage dirs to <stage>.bak.<ts>/ and rerun
#                   from scratch (accumulates .bak.* dirs; smoke use only).
#   MERGE_PARTS=1   After nvt_prod completes, best-effort merge part*.xtc/.edr
#                   into nvt_prod.full.xtc/.full.edr via trjcat/eneconv. Default
#                   off; merge failures are non-fatal (WARN only).

set -eo pipefail

{module_lines}

CASE_ID="{case_id}"
GRO="{gro_file}"
GMX_BIN="{gmx_bin}"
MAXWARN="{maxwarn}"
# GROMACS 2026+ emits a known Berendsen-barostat warning for the control-derived
# rlx stage. Allow exactly that stage to pass one grompp warning; keep all other
# stages on the stricter package-level MAXWARN.
RLX_MAXWARN="${{RLX_MAXWARN:-1}}"
THREADS="{cpus}"
MDRUN_GPU_ARGS="{mdrun_gpu_args}"
MDRUN_PARALLEL="{mdrun_parallel}"
ENABLE_STEEP_GPU="{enable_steep_gpu}"

echo "================================"
echo "Starting manual MD pipeline for $CASE_ID"
echo "Host: $(hostname)"
echo "Workdir: $PWD"
echo "Threads: $THREADS"
echo "Shell: bash ${{BASH_VERSION:-unknown}} (DO NOT invoke with sh)"
echo "FORCE_FRESH=${{FORCE_FRESH:-0}}  MERGE_PARTS=${{MERGE_PARTS:-0}}"
echo "Started: $(date)"
echo "================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- nvidia-smi snapshot ---"
  nvidia-smi || true
fi

if command -v "$GMX_BIN" >/dev/null 2>&1; then
  "$GMX_BIN" --version | head -n 1 || true
else
  echo "ERROR: GMX_BIN not found: $GMX_BIN" >&2
  exit 2
fi

# --- Resume-aware helpers ------------------------------------------------

# Find the newest confout .gro for <name>. Prefers <name>.partNNNN.gro (highest N)
# because a resume produces a newer part file while the original <name>.gro may
# be stale from the first aborted run. Echoes the path; returns 0 on success.
latest_stage_gro() {{
  local name="$1"
  local latest
  latest=$(ls -1 "${{name}}/${{name}}".part*.gro 2>/dev/null | sort -V | tail -n 1)
  if [ -n "$latest" ] && [ -s "$latest" ]; then
    echo "$latest"
    return 0
  fi
  if [ -s "${{name}}/${{name}}.gro" ]; then
    echo "${{name}}/${{name}}.gro"
    return 0
  fi
  return 1
}}

stage_is_complete() {{
  local name="$1"
  latest_stage_gro "$name" >/dev/null 2>&1 || return 1
  if grep -q "Finished mdrun" "${{name}}/${{name}}.log" 2>/dev/null; then
    return 0
  fi
  local part
  for part in "${{name}}/${{name}}".part*.log; do
    [ -e "$part" ] || continue
    if grep -q "Finished mdrun" "$part"; then return 0; fi
  done
  return 1
}}

# tpr present, no cpt, but stale output artifacts exist (mdrun ran but crashed
# before writing first checkpoint). Covers .log/.xtc/.trr/.edr/.gro and part*.
stage_is_dirty_no_cpt() {{
  local name="$1"
  [ -s "${{name}}/${{name}}.tpr" ] || return 1
  [ -e "${{name}}/${{name}}.cpt" ] && return 1
  local f
  for f in "${{name}}/${{name}}.log" "${{name}}/${{name}}.xtc" "${{name}}/${{name}}.trr" \
           "${{name}}/${{name}}.edr" "${{name}}/${{name}}.gro" \
           "${{name}}/${{name}}".part*.log "${{name}}/${{name}}".part*.xtc \
           "${{name}}/${{name}}".part*.trr "${{name}}/${{name}}".part*.edr \
           "${{name}}/${{name}}".part*.gro; do
    [ -e "$f" ] && return 0
  done
  return 1
}}

archive_stage_outputs() {{
  local name="$1"
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  local backup="${{name}}/_incomplete_backup_${{ts}}"
  mkdir -p "$backup"
  shopt -s nullglob
  local stale=( "${{name}}/${{name}}".log "${{name}}/${{name}}".xtc "${{name}}/${{name}}".trr \
                "${{name}}/${{name}}".edr "${{name}}/${{name}}".gro \
                "${{name}}/${{name}}".part* )
  shopt -u nullglob
  local f
  for f in "${{stale[@]}}"; do
    [ -e "$f" ] && mv "$f" "$backup/" || true
  done
  echo "[resume] Archived stale outputs for $name -> $backup"
}}

grompp_maxwarn() {{
  local name="$1"
  if [ "$name" = "rlx" ]; then
    echo "$RLX_MAXWARN"
  else
    echo "$MAXWARN"
  fi
}}

prepare_stage_inputs() {{
  local name="$1"
  local prev_gro_rel="$2"
  mkdir -p "$name"
  if [ -s "${{name}}/${{name}}.tpr" ]; then
    return 0
  fi
  # nullglob array to avoid literal ./*.itp when no itp files exist
  shopt -s nullglob
  local itps=( ./*.itp )
  shopt -u nullglob
  cp "${{name}}.mdp" topol.top "${{itps[@]}}" "${{name}}/"
  if [ -z "$prev_gro_rel" ]; then
    cp "$GRO" "${{name}}/"
    ( cd "$name" && "$GMX_BIN" editconf -f "$GRO" -o all.gro -c )
  else
    cp "$prev_gro_rel" "${{name}}/all.gro"
  fi
  local stage_maxwarn
  stage_maxwarn=$(grompp_maxwarn "$name")
  echo "[grompp] stage=$name maxwarn=$stage_maxwarn"
  ( cd "$name" && "$GMX_BIN" grompp -f "${{name}}.mdp" -c all.gro -p topol.top \
      -maxwarn "$stage_maxwarn" -o "${{name}}.tpr" )
}}

# Best-effort merge of part segments into .full.xtc/.full.edr for nvt_prod.
# OFF by default; non-fatal on any failure.
merge_stage_parts() {{
  local name="$1"
  [ "${{MERGE_PARTS:-0}}" = "1" ] || return 0
  [ "$name" = "nvt_prod" ] || return 0
  (
    cd "$name" || return 0
    shopt -s nullglob
    local parts_xtc=( "${{name}}".part*.xtc )
    local parts_edr=( "${{name}}".part*.edr )
    shopt -u nullglob
    local xtc_files=()
    [ -s "${{name}}.xtc" ] && xtc_files+=( "${{name}}.xtc" )
    xtc_files+=( "${{parts_xtc[@]}}" )
    local edr_files=()
    [ -s "${{name}}.edr" ] && edr_files+=( "${{name}}.edr" )
    edr_files+=( "${{parts_edr[@]}}" )
    if [ "${{#xtc_files[@]}}" -gt 1 ]; then
      echo "[merge] trjcat ${{#xtc_files[@]}} xtc -> ${{name}}.full.xtc"
      if ! "$GMX_BIN" trjcat -f "${{xtc_files[@]}}" -o "${{name}}.full.xtc" -cat < /dev/null; then
        echo "[merge] WARN: trjcat failed for $name (non-fatal)" >&2
      fi
    elif [ "${{#xtc_files[@]}}" -eq 1 ]; then
      cp "${{xtc_files[0]}}" "${{name}}.full.xtc"
    fi
    if [ "${{#edr_files[@]}}" -gt 1 ]; then
      echo "[merge] eneconv ${{#edr_files[@]}} edr -> ${{name}}.full.edr"
      if ! "$GMX_BIN" eneconv -f "${{edr_files[@]}}" -o "${{name}}.full.edr" < /dev/null; then
        echo "[merge] WARN: eneconv failed for $name (non-fatal)" >&2
      fi
    elif [ "${{#edr_files[@]}}" -eq 1 ]; then
      cp "${{edr_files[0]}}" "${{name}}.full.edr"
    fi
  )
  return 0
}}

run_stage() {{
  local name="$1"
  local prev_gro_rel="$2"

  # FORCE_FRESH=1 archives each existing stage dir (doesn't delete). Each
  # invocation accumulates <stage>.bak.<ts>/ -- smoke/debug use only.
  if [ "${{FORCE_FRESH:-0}}" = "1" ] && [ -d "$name" ]; then
    local ts; ts=$(date +%Y%m%d_%H%M%S)
    mv "$name" "${{name}}.bak.${{ts}}"
    echo "[fresh] Archived existing stage dir to ${{name}}.bak.${{ts}}"
  fi

  if stage_is_complete "$name"; then
    echo "[resume] Stage $name already completed; skipping at $(date)"
    return 0
  fi

  if stage_is_dirty_no_cpt "$name"; then
    echo "[resume] Stage $name has tpr but no cpt with stale outputs; archiving"
    archive_stage_outputs "$name"
  fi

  prepare_stage_inputs "$name" "$prev_gro_rel"

  # Per-stage GPU offload extras (GROMACS 2021.2 constraints, smoke-tested).
  # This is opt-in: set ENABLE_GPU_EXTRA=1 and use a GPU mdrun package.
  # Keeping it opt-in prevents CPU-only packages from receiving GPU-only flags.
  # rlx (control-derived V-rescale + Berendsen, no constraints): full offload OK
  # after smoke validation; override GPU_EXTRA_RLX if a GROMACS build rejects
  # Berendsen with -update gpu.
  # pr (Nose-Hoover + Parrinello-Rahman): only -bonded gpu;
  #   -update gpu rejects both Nose-Hoover and Parrinello-Rahman in 2021.2.
  # nvt_prod (Nose-Hoover + NVT): only -bonded gpu, and disable PME
  #   load tuning to avoid rare FFT allocation failures during startup;
  #   -update gpu rejects Nose-Hoover in 2021.2 ("Nose-Hoover ... not supported" mdrun error).
# steep: bind to the assigned GPU for supervised multi-GPU scheduling.  This
# avoids GROMACS auto-selecting GPU0 when several manual jobs start together.
# Set ENABLE_STEEP_GPU=0 to force CPU nonbonded work for smoke/debug runs.
  # Override per stage via env: GPU_EXTRA_RLX, GPU_EXTRA_PR, GPU_EXTRA_NVT_PROD, GPU_EXTRA_STEEP
  local STAGE_MDRUN_GPU_ARGS="$MDRUN_GPU_ARGS"
  local STAGE_GPU_EXTRA=""
  if [ "$name" = "steep" ] && [ "${{ENABLE_STEEP_GPU:-0}}" != "1" ]; then
    STAGE_MDRUN_GPU_ARGS="-nb cpu"
  fi
  if [ "${{ENABLE_GPU_EXTRA:-0}}" = "1" ] && [ -n "${{MDRUN_GPU_ARGS:-}}" ]; then
    case "$name" in
      rlx)          STAGE_GPU_EXTRA="-update gpu -bonded gpu" ;;
      pr)           STAGE_GPU_EXTRA="-bonded gpu" ;;
      nvt_prod)     STAGE_GPU_EXTRA="-bonded gpu -notunepme" ;;
      steep)        STAGE_GPU_EXTRA="" ;;
    esac
  fi
  local override_var="GPU_EXTRA_$(echo "$name" | tr '[:lower:]' '[:upper:]')"
  if [ -n "${{!override_var+x}}" ]; then
    STAGE_GPU_EXTRA="${{!override_var}}"
    echo "[gpu] Stage $name extras overridden via $override_var: '$STAGE_GPU_EXTRA'"
  fi

  (
    cd "$name"
    if [ -s "${{name}}.cpt" ]; then
      echo "[resume] Resuming mdrun for $name from ${{name}}.cpt at $(date)"
      "$GMX_BIN" mdrun $MDRUN_PARALLEL -pin off $STAGE_MDRUN_GPU_ARGS $STAGE_GPU_EXTRA \
        -deffnm "$name" -cpi "${{name}}.cpt" -noappend
    else
      echo "[resume] Starting fresh mdrun for $name at $(date)"
      "$GMX_BIN" mdrun $MDRUN_PARALLEL -pin off $STAGE_MDRUN_GPU_ARGS $STAGE_GPU_EXTRA \
        -deffnm "$name"
    fi
  )

  if ! stage_is_complete "$name"; then
    echo "ERROR: stage $name did not complete (no final gro or 'Finished mdrun')" >&2
    exit 3
  fi

  merge_stage_parts "$name"
  echo "Stage $name completed at $(date)"
}}

# -------------------------------------------------------------------------

run_stage steep ""
RLX_PREV=$(latest_stage_gro steep) || {{ echo "ERROR: steep has no final gro" >&2; exit 4; }}
run_stage rlx "$RLX_PREV"
PR_PREV=$(latest_stage_gro rlx) || {{ echo "ERROR: rlx has no final gro" >&2; exit 4; }}
run_stage pr "$PR_PREV"
NVT_PREV=$(latest_stage_gro pr) || {{ echo "ERROR: pr has no final gro" >&2; exit 4; }}
run_stage nvt_prod "$NVT_PREV"

echo "================================"
echo "All stages completed for $CASE_ID"
echo "Finished: $(date)"
echo "================================"
