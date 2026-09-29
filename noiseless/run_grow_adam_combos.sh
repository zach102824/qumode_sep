#!/usr/bin/env bash
# Stage 2 of the grow+Adam tuning: a few follow-up single-knob points (tag grow_tune_S_*) and
# combinations of the per-knob winners (tag grow_tune_X_*) on the same screen subset
# (four_sat_000..004 × 10 trials, seed 20260917, final L=4, 800 SPSA steps).
set -euo pipefail
cd "$(dirname "$0")/.."
S=noiseless/run_grow_adam_screen.sh
# follow-up single-knob points (lr schedule and kick were the strongest knobs)
$S lrs_decay_lo --grow-lr-schedule 0.1,0.05,0.03,0.02
$S lrs_decay_01 --grow-lr-schedule 0.2,0.1,0.05,0.01
$S lrs_decay_hi --grow-lr-schedule 0.3,0.15,0.05,0.02
$S kick1p0 --grow-kick-sigma 1.0
LR=(--grow-lr-schedule 0.2,0.1,0.05,0.02)
export PREFIX=grow_tune_X
$S lr_kick "${LR[@]}" --grow-kick-sigma 0.5
$S lr_kick_eta "${LR[@]}" --grow-kick-sigma 0.5 --grow-eta-scale 0.5,0.7,1.5,2
$S lr_kick_split "${LR[@]}" --grow-kick-sigma 0.5 --grow-steps-schedule 100,150,250,300
$S lr_kick_c "${LR[@]}" --grow-kick-sigma 0.5 --spsa-c 0.1
$S lr_eta "${LR[@]}" --grow-eta-scale 0.5,0.7,1.5,2
$S lr_split "${LR[@]}" --grow-steps-schedule 100,150,250,300
$S lr_c "${LR[@]}" --spsa-c 0.1
$S all5 "${LR[@]}" --grow-kick-sigma 0.5 --grow-eta-scale 0.5,0.7,1.5,2 --grow-steps-schedule 100,150,250,300 --spsa-c 0.1
