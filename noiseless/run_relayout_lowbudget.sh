#!/usr/bin/env bash
# Low-budget multi-round relayout sweep (RELAYOUT_LOWBUDGET_SUMMARY.md).
# Round-0 cost {L1-2@10, L1-4@5, L1-4@10, L1-4@25 steps/layer} x relabel-round steps {25,50,100}
# (+ optional extra columns via RL_STEPS), 4 relabel rounds at L=4, small-beta init, lr 0.05,
# best-so-far guess, radius-1 polish, XOR->vacuum, no fixed-point stop, best-by-cost return.
# 20 four_sat n=8 H x 25 trials, seeds paired (seed formula uses L=4 for every run).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1
PY=${PY:-/workspace/venv-qumode/bin/python}
OUT=${OUT:-noiseless/results/relayout_runs/lowbudget}
WORKERS=${WORKERS:-8}
TARGET=${TARGET:-xor_vacuum}   # none = no-relabel control
R0S=${R0S:-"L2s10 L4s5 L4s10 L4s25"}
RL_STEPS=${RL_STEPS:-"25 50 100"}
mkdir -p "$OUT"
for r0 in $R0S; do
  L=${r0:1:1}; S=${r0#*s}
  if [ "$L" = 2 ]; then LR=0.5,0.2; else LR=0.5,0.2,0.05,0.02; fi
  for rs in $RL_STEPS; do
    tag="lb_${TARGET}_${r0}_r${rs}"
    $PY -m noiseless.run_u_sweep --u-names jp --layers "$L" --trials 25 --workers "$WORKERS" \
      --optimizer spsa_adam --grow --grow-steps-per-stage "$S" --grow-lr-schedule "$LR" \
      --relayout --relayout-rounds 4 --relayout-steps "$rs" --relayout-lr 0.05 \
      --relayout-init small --relayout-return best --relayout-layers 4 --relayout-guess best \
      --no-relayout-fixed-stop --relayout-target "$TARGET" --seed-layers 4 \
      --outdir "$OUT" --tag "$tag" | grep -v '^  \['
  done
done
echo DONE
