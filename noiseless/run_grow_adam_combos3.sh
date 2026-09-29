#!/usr/bin/env bash
# Stage 4: refine the lr schedule around 0.5,0.2,0.05,0.02 and combine with eta / split / kick / c.
set -euo pipefail
cd "$(dirname "$0")/.."
S=noiseless/run_grow_adam_screen.sh
$S lrs_1p0 --grow-lr-schedule 1.0,0.3,0.05,0.02
$S lrs_vhi_s3hi --grow-lr-schedule 0.5,0.2,0.1,0.02
$S lrs_vhi_last01 --grow-lr-schedule 0.5,0.2,0.05,0.01
$S lrs_vhi_last03 --grow-lr-schedule 0.5,0.2,0.05,0.03
LR=(--grow-lr-schedule 0.5,0.2,0.05,0.02)
export PREFIX=grow_tune_X
$S lrvhi_eta "${LR[@]}" --grow-eta-scale 0.5,0.7,1.5,2
$S lrvhi_split "${LR[@]}" --grow-steps-schedule 100,150,250,300
$S lrvhi_kick0p5 "${LR[@]}" --grow-kick-sigma 0.5
$S lrvhi_c "${LR[@]}" --spsa-c 0.1
$S lrvhi_eta_split "${LR[@]}" --grow-eta-scale 0.5,0.7,1.5,2 --grow-steps-schedule 100,150,250,300
$S lrvhi_eta_c "${LR[@]}" --grow-eta-scale 0.5,0.7,1.5,2 --spsa-c 0.1
