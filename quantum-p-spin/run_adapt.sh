#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qumode_sep/quantum-p-spin
export PYTHONPATH=/workspace/qumode_sep:/workspace/qumode_sep/quantum-p-spin
exec /workspace/qtimes_venv/bin/python -u train_adapt.py \
  --L 8 --Gamma 0.5 --seeds 0 \
  --B-train 64 --B-hold 32 \
  --rounds-base 25 --rounds-op 20 --batch 32 \
  --max-adapt 4 --min-grad 1e-4 --min-gain 0.002
