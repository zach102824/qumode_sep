#!/usr/bin/env bash
# Keep SNAP-easiness explore alive across waves. Honor STOP_SNAP_EASY.
# Does NOT touch angle_explore polish PIDs.
# Also keeps a side gauss B90-gap polish job alive.
set -euo pipefail
ROOT=/workspace/qumode_sep
cd "$ROOT"
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH="$ROOT"
OUT=Nov/runs/snap_easy
PIDFILE="$OUT/PID"
GPIDFILE="$OUT/GAUSS_POLISH_PID"
KPIDFILE="$OUT/KERR_POLISH_PID"
FPIDFILE="$OUT/FEWM_N12_POLISH_PID"
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
      modes="kerr,gauss,few_m,poly2_n,gauss_plus1,poly3_n,const"
      seeds="4,5,6"; steps=500
      ;;
    3)
      logf="$OUT/run_wave3.log"
      modes="kerr,gauss,few_m,poly2_n,shell,free"
      seeds="7,8,10,22"; steps=500
      ;;
    *)
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


launch_kerr_polish() {
  local logf="$OUT/run_kerr_polish.log"
  echo "$(date +%H:%M:%S) launching kerr/few_m polish (n12 CVaR + skip-done)" >> "$LOG"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    nohup python -m Nov.scripts.snap_easy_kerr_polish \
      --L 32 --seeds 0,2,10,22,33,44,55 --scale_B 1000 --skip-baseline-scale \
      >> "$logf" 2>&1 &
  echo $! > "$KPIDFILE"
  echo "$(date +%H:%M:%S) kerr_polish pid=$!" >> "$LOG"
}


launch_fewm_n12() {
  local logf="$OUT/run_kerr_polish_fewm_n12.log"
  echo "$(date +%H:%M:%S) launching few_m n12/n15 polish-chain (≤2p)" >> "$LOG"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    nohup python -m Nov.scripts.snap_easy_kerr_polish \
      --L 32 --seeds 0,33,44,55,66 --scale_B 1000 --skip-baseline-scale \
      --arms fewm2_cvar095_n12_from_polish,fewm2_cvar09_n12_from_polish,fewm2_adam1200_n12_from_polish,fewm2_cvar095_n15_from_polish,fewm2_adam900_n15_from_polish,fewm2_cvar09_n15_from_polish,fewm2_cvar095_n12_from_best,fewm2_cvar09_n12_from_best,fewm2_adam900_n12_from_best \
      >> "$logf" 2>&1 &
  echo $! > "$FPIDFILE"
  echo "$(date +%H:%M:%S) fewm_n12_polish pid=$!" >> "$LOG"
}

launch_gauss_polish() {
  local logf="$OUT/run_gauss_polish.log"
  echo "$(date +%H:%M:%S) launching gauss polish" >> "$LOG"
  # Be polite to the main ladder: limited BLAS threads
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    nohup python -m Nov.scripts.snap_easy_gauss_polish \
      --L 32 --seeds 0,10,22 --scale_B 1000 \
      >> "$logf" 2>&1 &
  echo $! > "$GPIDFILE"
  echo "$(date +%H:%M:%S) gauss_polish pid=$!" >> "$LOG"
}

while true; do
  if [[ -f "$STOP" ]]; then
    echo "$(date +%H:%M:%S) STOP; exit" >> "$LOG"; exit 0
  fi
  # main ladder
  if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE")
    if ! kill -0 "$pid" 2>/dev/null; then
      wave=$(cat "$WAVEFILE")
      next=$((wave + 1))
      if [[ ! -f "$OUT/L32_STARTED" ]]; then
        touch "$OUT/L32_STARTED"
      fi
      launch_wave "$next"
      echo "$next" > "$WAVEFILE"
    fi
  else
    wave=$(cat "$WAVEFILE")
    launch_wave "$wave"
  fi
  # side gauss polish (restart if dead and no BEAT yet)
  if [[ ! -f "$OUT/BEAT_SEALED.json" ]]; then
    need_gp=1
    if [[ -f "$GPIDFILE" ]]; then
      gpid=$(cat "$GPIDFILE")
      if kill -0 "$gpid" 2>/dev/null; then
        need_gp=0
      fi
    fi
    if [[ "$need_gp" -eq 1 ]]; then
      launch_gauss_polish
    fi
  fi
  # side kerr/few_m ≤2-param polish until 2PARAM beat
  if [[ ! -f "$OUT/BEAT_SEALED_2PARAM.json" ]]; then
    need_kp=1
    if [[ -f "$KPIDFILE" ]]; then
      kpid=$(cat "$KPIDFILE")
      if kill -0 "$kpid" 2>/dev/null; then
        need_kp=0
      fi
    fi
    if [[ "$need_kp" -eq 1 ]]; then
      launch_kerr_polish
    fi
  fi
  # few_m n12/n15 chain polish until 2PARAM beat
  if [[ ! -f "$OUT/BEAT_SEALED_2PARAM.json" ]]; then
    need_fp=1
    if [[ -f "$FPIDFILE" ]]; then
      fpid=$(cat "$FPIDFILE")
      if kill -0 "$fpid" 2>/dev/null; then
        need_fp=0
      fi
    fi
    if [[ "$need_fp" -eq 1 ]]; then
      launch_fewm_n12
    fi
  fi
  # refresh RESULTS/PARETO periodically
  python -m Nov.scripts.snap_easy_refresh_docs >> "$OUT/refresh_docs.log" 2>&1 || true
  sleep 45
done
