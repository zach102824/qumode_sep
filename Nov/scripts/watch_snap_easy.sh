#!/usr/bin/env bash
# Keep SNAP-easiness explore alive across waves. Honor STOP_SNAP_EASY.
# Does NOT touch angle_explore polish PIDs.
set -euo pipefail
ROOT=/workspace/qumode_sep
cd "$ROOT"
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH="$ROOT"
OUT=Nov/runs/snap_easy
PIDFILE="$OUT/PID"
STOP=Nov/runs/STOP_SNAP_EASY
LOG="$OUT/watchdog.log"
WAVEFILE="$OUT/WAVE"
mkdir -p "$OUT"
echo "$(date +%H:%M:%S) watch_snap_easy restart" >> "$LOG"
if [[ ! -f "$WAVEFILE" ]]; then echo 0 > "$WAVEFILE"; fi

launch_wave() {
  local wave=$1
  local logf modes seeds steps
  case "$wave" in
    0)
      # should already be running L32 seeds0,1; fallback relaunch
      logf="$OUT/run_L32c.log"
      modes="gauss,poly2_n,few_m,gauss_plus1,poly3_pc,poly3_n,shell"
      seeds="0,1"; steps=400
      ;;
    1)
      logf="$OUT/run_hybrid.log"
      modes="gauss,gauss_plus1,kerr,poly2_n,few_m,poly3_pc"
      seeds="2,3"; steps=600
      touch "$OUT/HYBRID_STARTED"
      ;;
    2)
      logf="$OUT/run_wave2.log"
      # tighter restrictions + more seeds; include kerr full-path + few_m M via few_m
      modes="kerr,gauss,few_m,poly2_n,gauss_plus1,poly3_n,const"
      seeds="4,5,6"; steps=500
      ;;
    3)
      logf="$OUT/run_wave3.log"
      modes="kerr,gauss,few_m,poly2_n,shell,free"
      seeds="7,8,10,22"; steps=500
      ;;
    *)
      # keep exploring forever: rotate seeds
      local base=$(( 11 + (wave - 4) * 3 ))
      logf="$OUT/run_wave${wave}.log"
      modes="kerr,gauss,few_m,poly2_n,gauss_plus1,poly3_pc"
      seeds="${base},$((base+1)),$((base+2))"
      steps=450
      ;;
  esac
  echo "$(date +%H:%M:%S) launching wave=$wave modes=$modes seeds=$seeds" >> "$LOG"
  nohup python -m Nov.scripts.snap_easy_explore --L 32 --seeds "$seeds" --steps "$steps" --lr 5e-3 --scale_B 1000 \
    --modes "$modes" >> "$logf" 2>&1 &
  echo $! > "$PIDFILE"
  echo "$(date +%H:%M:%S) wave=$wave pid=$!" >> "$LOG"
}

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
  wave=$(cat "$WAVEFILE")
  # if wave0 pid died, advance
  next=$((wave + 1))
  # On first death of current wave0 (714841), WAVE may still be 0 → launch wave 1
  if [[ ! -f "$OUT/L32_STARTED" ]]; then
    touch "$OUT/L32_STARTED"
  fi
  launch_wave "$next"
  echo "$next" > "$WAVEFILE"
  sleep 45
done
