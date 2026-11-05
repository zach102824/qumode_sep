#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qumode_sep/quantum-p-spin
export PYTHONPATH=/workspace/qumode_sep:/workspace/qumode_sep/quantum-p-spin
exec /workspace/qtimes_venv/bin/python -u train_nondiag.py \
  --recipes dbl_hc_snap dbl_shift_snap dbl_mag_snap givens_inner_snap givens_after_snap xx_match_snap xx_path_snap \
  --L 8 --Gamma 0.5 --seeds 0 1 2 --B-train 64 --B-hold 32 --rounds 30 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 10
