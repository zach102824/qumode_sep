#!/usr/bin/env bash
# Controls for RELAYOUT_LOWBUDGET_SUMMARY.md: (i) same rounds without relabel (target none),
# (ii) plain tuned growth L1->4 with steps/stage S matched to the setting's total evals
# (4*(2S+1) ~= 84 + 4*(2*rs+1) for round 0 = L4s10).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1
PY=${PY:-/workspace/venv-qumode/bin/python}
OUT=${OUT:-noiseless/results/relayout_runs/lowbudget}
CTRL_RS=${CTRL_RS:-"100 200"}
for rs in $CTRL_RS; do
  TARGET=none R0S=L4s10 RL_STEPS=$rs noiseless/run_relayout_lowbudget.sh
  S=$(( ( (84 + 4*(2*rs+1)) / 4 - 1) / 2 ))
  $PY -m noiseless.run_u_sweep --u-names jp --layers 4 --trials 25 --workers 8 --optimizer spsa_adam \
    --grow --grow-steps-per-stage "$S" --grow-lr-schedule 0.5,0.2,0.05,0.02 --seed-layers 4 \
    --outdir "$OUT" --tag "lbgrow_L4s10_r${rs}" | grep -v '^  \['
done
echo CTRL_DONE
