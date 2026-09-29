#!/usr/bin/env bash
# Full fleets (20 four_sat × 25 trials, seed 20260917, final L=4, 800 SPSA steps = 1604 evals/trial)
# for the chosen grow+Adam settings. Tags grow_tune_F_<name>.
set -euo pipefail
cd "$(dirname "$0")/.."
export PREFIX=grow_tune_F TRIALS=25 MAXH=20
S=noiseless/run_grow_adam_screen.sh
LR=(--grow-lr-schedule 0.5,0.2,0.05,0.02)
$S combo "${LR[@]}" --grow-eta-scale 0.5,0.7,1.5,2 --spsa-c 0.1
$S lr_sched "${LR[@]}"
$S lr_sched_split "${LR[@]}" --grow-steps-schedule 100,150,250,300
