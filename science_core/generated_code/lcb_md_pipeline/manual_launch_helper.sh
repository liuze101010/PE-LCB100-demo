#!/bin/bash

# Print-only helper for manual GPU servers.
# It never scans remote nodes, never chooses a GPU automatically, never edits
# run_manual.sh, and never launches a job. It only checks the current node and
# prints the final commands for the user to run manually.

set -eo pipefail

line() {
  echo "------------------------------------------------------------"
}

usage() {
  cat <<'EOF'
Usage:
  bash manual_launch_helper.sh <gpu_id>

Purpose:
  Check this exact compute node and this exact run directory.
  Then print the manual launch commands. It does not launch anything.

How to read the result:
  OK       = this part looks usable
  BLOCKED  = do not launch from this directory/GPU now
  NEXT     = the next command or action to do by hand

This helper does NOT:
  - scan other nodes
  - choose a GPU for you
  - edit run_manual.sh
  - run nohup
  - create manual.pid
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

GPU_ID="${1:-}"
if [ -z "$GPU_ID" ]; then
  usage >&2
  exit 2
fi

case "$GPU_ID" in
  ''|*[!0-9]*)
    echo "BLOCKED: gpu_id must be a non-negative integer, got: $GPU_ID" >&2
    exit 2
    ;;
esac

if [ ! -s run_manual.sh ]; then
  echo "BLOCKED: run_manual.sh was not found in this directory."
  echo "CURRENT_DIR: $PWD"
  echo "NEXT: cd into one unpacked case directory, then run this helper again."
  exit 2
fi

CASE_ID="$(grep '^CASE_ID=' run_manual.sh | head -1 | sed 's/^CASE_ID=//' | tr -d '"')"
if [ -z "$CASE_ID" ]; then
  CASE_ID="$(basename "$PWD")"
fi

line
echo "MANUAL GPU LAUNCH CHECK"
line
echo "CASE:        $CASE_ID"
echo "HOST:        $(hostname)"
echo "WORKDIR:     $PWD"
echo "GPU_TO_CHECK:$GPU_ID"
echo "TIME:        $(date)"
echo
echo "IMPORTANT: this helper only checks and prints. It has not started a run."
line

echo "1) CPU / node snapshot"
if command -v lscpu >/dev/null 2>&1; then
  lscpu | grep -E '^(CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|Model name)' || true
else
  echo "INFO: lscpu not found."
fi
if command -v top >/dev/null 2>&1; then
  top -bn1 | head -5 || true
fi
line

echo "2) GPU snapshot"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "BLOCKED: nvidia-smi was not found on this node."
  echo "NEXT: do not launch GPU production here; switch to a GPU compute node."
  exit 3
fi

echo "All GPUs:"
nvidia-smi
echo
echo "Requested GPU only: $GPU_ID"
GPU_SNAPSHOT="$(nvidia-smi -i "$GPU_ID" 2>&1 || true)"
printf '%s\n' "$GPU_SNAPSHOT"

if printf '%s\n' "$GPU_SNAPSHOT" | grep -q "No devices were found\\|Invalid GPU"; then
  echo
  echo "BLOCKED: GPU $GPU_ID is not visible on this node."
  echo "NEXT: run nvidia-smi and choose an existing GPU id."
  exit 3
fi

GPU_BUSY=0
if ! printf '%s\n' "$GPU_SNAPSHOT" | grep -q "No running processes found"; then
  GPU_BUSY=1
fi

GPU_QUICK="$(nvidia-smi -i "$GPU_ID" --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true)"
if [ -n "$GPU_QUICK" ]; then
  echo
  echo "Requested GPU quick view:"
  echo "index,name,gpu_util_percent,mem_used_MiB,mem_total_MiB,temp_C,power_W"
  echo "$GPU_QUICK"
fi

if [ "$GPU_BUSY" = "1" ]; then
  echo
  echo "GPU RESULT: BLOCKED"
  echo "Reason: GPU $GPU_ID has at least one visible running process."
  echo "NEXT: choose another idle GPU id or wait."
else
  echo
  echo "GPU RESULT: OK"
  echo "Meaning: nvidia-smi shows no visible process on GPU $GPU_ID."
fi
line

echo "3) Current directory safety"
DIR_BLOCKED=0

if [ -s manual.pid ]; then
  PID="$(cat manual.pid)"
  if ps -p "$PID" >/dev/null 2>&1; then
    DIR_BLOCKED=1
    echo "DIRECTORY PID: BLOCKED"
    echo "Reason: manual.pid points to a running process: $PID"
    ps -p "$PID" -o pid,stat,etime,cmd || true
  else
    echo "DIRECTORY PID: OK_WITH_NOTE"
    echo "manual.pid exists but process $PID is not running."
    echo "If this is an old failed run, use a fresh directory or inspect logs before reusing it."
  fi
else
  echo "DIRECTORY PID: OK"
  echo "No manual.pid file found."
fi

if [ -d steep ] || [ -d rlx ] || [ -d pr ] || [ -d nvt_prod ]; then
  DIR_BLOCKED=1
  echo "DIRECTORY STAGES: BLOCKED"
  echo "Reason: one or more stage directories already exist: steep/ rlx/ pr/ nvt_prod/"
  echo "This usually means this directory has already been used."
else
  echo "DIRECTORY STAGES: OK"
  echo "No stage directory found."
fi
line

echo "4) Decision"
if [ "$GPU_BUSY" = "1" ] || [ "$DIR_BLOCKED" = "1" ]; then
  echo "FINAL RESULT: BLOCKED"
  echo "中文结论：现在不要启动。"
  if [ "$GPU_BUSY" = "1" ]; then
    echo "- GPU $GPU_ID is not free."
    echo "  中文原因：GPU $GPU_ID 上已经有程序在跑，不能再占这个 GPU。"
  fi
  if [ "$DIR_BLOCKED" = "1" ]; then
    echo "- This run directory is not clean for a new launch."
    echo "  中文原因：当前目录已经有正在跑的任务或旧的阶段目录，不能在同一个目录再开一次。"
  fi
  echo
  echo "如果你只是想看当前目录里正在跑的任务，复制下面这些命令："
  echo "  ps -p \$(cat manual.pid) -o pid,stat,etime,cmd"
  echo "  tail -40 manual.out"
  echo "  tail -40 manual.err"
  echo "  nvidia-smi"
  echo
  echo "如果你想开一个新任务："
  echo "  1. 换到一个全新的 case 目录，不要复用当前目录。"
  echo "  2. 先用 nvidia-smi 找一个没有进程的 GPU。"
  echo "  3. 再运行：bash manual_launch_helper.sh <空闲GPU编号>"
  echo
  echo "本 helper 没有启动任何命令。"
  exit 0
fi

echo "FINAL RESULT: READY_TO_LAUNCH_MANUALLY"
echo "GPU $GPU_ID looks idle, and this directory looks unused."
echo
echo "中文结论：可以手动启动。"
echo "中文解释：GPU $GPU_ID 当前没有可见进程，并且这个目录看起来还没跑过。"
echo
echo "下一步：如果你确认要启动，请逐行复制并执行下面命令："
cat <<EOF
cp run_manual.sh run_manual_gpu${GPU_ID}.sh
sed -i 's|^MDRUN_GPU_ARGS=.*|MDRUN_GPU_ARGS="-gpu_id ${GPU_ID} -nb gpu"|' run_manual_gpu${GPU_ID}.sh
nohup bash run_manual_gpu${GPU_ID}.sh > manual.out 2> manual.err &
echo \$! > manual.pid
ps -p \$(cat manual.pid) -o pid,stat,etime,cmd
tail -f manual.out manual.err
EOF

echo
echo "执行后怎么看："
echo "  - ps 那行能看到 bash run_manual_gpu${GPU_ID}.sh，说明后台脚本还在。"
echo "  - nvidia-smi 里 GPU $GPU_ID 的 Processes 出现 gmx_mpi，说明 GROMACS 跑上 GPU 了。"
echo "  - tail -f 会持续看日志；按 Ctrl+C 只是退出看日志，不会停止后台任务。"
echo
echo "本 helper 没有启动任何命令；是否启动由你手动执行上面的 nohup 命令决定。"
