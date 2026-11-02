#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/qumode_sep
cd "$ROOT"
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH="$ROOT"
OUT=Nov/runs/snap_easy
PIDFILE="$OUT/PID"
STOP=Nov/runs/STOP_SNAP_EASY
LOG="$OUT/watchdog.log"
echo "$(date +%H:%M:%S) watch_snap_easy restart" >> "$LOG"
while true; do
  if [[ -f "$STOP" ]]; then
    echo "$(date +%H:%M:%S) STOP; exit" >> "$LOG"; exit 0
  fi
  if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
      sleep 45; continue
    fi
  fi
  if [[ ! -f "$OUT/HYBRID_STARTED" ]]; then
    echo "$(date +%H:%M:%S) launching hybrid wave" >> "$LOG"
    touch "$OUT/HYBRID_STARTED"
    nohup python -m Nov.scripts.snap_easy_explore --L 32 --seeds 2,3 --steps 600 --lr 5e-3 --scale_B 1000 \
      --modes gauss,gauss_plus1,kerr,poly2_n,few_m,poly3_pc \
      >> "$OUT/run_hybrid.log" 2>&1 &
    echo $! > "$PIDFILE"
    echo "$(date +%H:%M:%S) hybrid pid=$!" >> "$LOG"
    sleep 45; continue
  fi
  echo "$(date +%H:%M:%S) waves done; exit" >> "$LOG"
  exit 0
done
