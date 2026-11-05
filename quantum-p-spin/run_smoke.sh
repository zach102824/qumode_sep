#!/usr/bin/env bash
# Quantum p-spin fidelity smoke — all under quantum-p-spin/
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="/workspace/qumode_sep:${PYTHONPATH:-}"
PY="${PY:-/workspace/qtimes_venv/bin/python}"
OUT="${OUT_DIR:-runs/smoke/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "[smoke] out=$OUT"
"$PY" -u train_fidelity.py \
  --Gammas 0.5 1.0 1.5 \
  --Ls 1 2 4 8 \
  --seeds 0 1 2 \
  --B-train 64 --B-hold 32 \
  --rounds 20 --batch 32 \
  --modes joint n6 n9 \
  --out "$OUT" | tee "$OUT/train.log"
"$PY" -u freeze_eval.py \
  --train-dir "$OUT" \
  --eval-ns 12 15 \
  --B-eval 32 \
  --Gammas 0.5 1.0 1.5 \
  --Ls 1 2 4 8 | tee "$OUT/eval.log"
echo "[smoke] DONE → $OUT/RESULTS.md"
