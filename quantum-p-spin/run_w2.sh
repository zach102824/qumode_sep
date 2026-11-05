#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qumode_sep/quantum-p-spin
export PYTHONPATH=/workspace/qumode_sep
PY=/workspace/qtimes_venv/bin/python
OUT=runs/uc_explore_w2
mkdir -p "$OUT"
LOG="$OUT/train_w2.log"
exec > >(tee -a "$LOG") 2>&1
echo "==== Wave2 start $(date) ===="

# Phase A: cheap structured channels (fast)
$PY train_w2.py --recipes two_scale_snap poly3_snap mag_bucket_snap vert_bucket_snap \
  --L 8 --rounds 30 --seeds 0 1 2 --opt lbfgs --out "$OUT"

# Phase B: layer schedule + longer adam→lbfgs
$PY train_w2.py --recipes sched_aw_snap sched_aeps_snap \
  --L 8 --rounds 60 --seeds 0 1 2 --opt adam_then_lbfgs --adam-rounds 20 --out "$OUT"

# Phase C: sign/soft per-triple mag (warm)
$PY train_w2.py --recipes signj_mag_snap soft_tanh_snap soft_norm_snap \
  --L 8 --rounds 40 --seeds 0 1 2 --opt lbfgs --out "$OUT"

# Phase D: bigger data on Wave1 champ recipe via train_perzz
$PY train_perzz.py --recipes full_xj_snap_warm \
  --L 8 --rounds 40 --seeds 0 1 2 --B-train 256 --B-hold 64 \
  --out "$OUT/bigdata"

# Phase E: deeper factored (L=12) — skip L16 if too slow
$PY train_perzz.py --recipes factorth_xj_snap \
  --L 12 --rounds 30 --seeds 0 1 2 \
  --out "$OUT/L12_factorth"

# Phase F: L=12 two_scale + poly3 (cheap deeper)
$PY train_w2.py --recipes two_scale_snap poly3_snap \
  --L 12 --rounds 30 --seeds 0 1 2 --opt lbfgs --out "$OUT"

echo "==== Wave2 finished $(date) ===="
