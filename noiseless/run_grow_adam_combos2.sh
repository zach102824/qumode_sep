#!/usr/bin/env bash
# Stage 3: combos around the follow-up winners (lr 0.3,0.15,0.05,0.02; kick σ 1.0), same subset.
set -euo pipefail
cd "$(dirname "$0")/.."
S=noiseless/run_grow_adam_screen.sh
$S lrs_decay_vhi --grow-lr-schedule 0.5,0.2,0.05,0.02
$S kick2p0 --grow-kick-sigma 2.0
LR=(--grow-lr-schedule 0.3,0.15,0.05,0.02)
export PREFIX=grow_tune_X
$S lrhi_kick1 "${LR[@]}" --grow-kick-sigma 1.0
$S lrhi_kick0p5 "${LR[@]}" --grow-kick-sigma 0.5
$S lrhi_split "${LR[@]}" --grow-steps-schedule 100,150,250,300
$S lrhi_kick1_split "${LR[@]}" --grow-kick-sigma 1.0 --grow-steps-schedule 100,150,250,300
$S lrhi_eta "${LR[@]}" --grow-eta-scale 0.5,0.7,1.5,2
$S lrhi_c "${LR[@]}" --spsa-c 0.1
$S lrhi_kick1_eta "${LR[@]}" --grow-kick-sigma 1.0 --grow-eta-scale 0.5,0.7,1.5,2
