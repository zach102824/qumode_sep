#!/usr/bin/env bash
# Grow+Adam hyper-parameter screen: jp, binary, λ=0 Gibbs, final L=4, 800 total SPSA steps
# (1604 evals/trial for start 1; 1603 for start 2), four_sat_000..004 (--max-h 5) × 10 trials,
# seed 20260917. One knob at a time around the C_adam baseline. Tags grow_tune_S_<name>.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/venv-qumode/bin/python}
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
COMMON=(--u-names jp --layers 4 --trials ${TRIALS:-10} --steps 800 --max-h ${MAXH:-5} --workers 8
        --seed 20260917 --lambda1 0 --encoding binary --optimizer spsa_adam --grow)
run() { local name=$1; shift; echo "== $name $*"; local t0=$SECONDS
        "$PY" noiseless/run_u_sweep.py "${COMMON[@]}" --tag "${PREFIX:-grow_tune_S}_$name" "$@" | tail -1
        echo "   wall $((SECONDS-t0)) s"; }
if [[ $# -gt 0 ]]; then run "$@"; exit; fi
run base
run lr0p02 --adam-lr 0.02
run lr0p1 --adam-lr 0.1
run lr0p2 --adam-lr 0.2
run lrs_hi_lo --grow-lr-schedule 0.1,0.1,0.1,0.03
run lrs_decay --grow-lr-schedule 0.2,0.1,0.05,0.02
run split_100_150_250_300 --grow-steps-schedule 100,150,250,300
run split_50_100_250_400 --grow-steps-schedule 50,100,250,400
run kick0p2 --grow-kick-sigma 0.2
run kick0p5 --grow-kick-sigma 0.5
run eta_hot_early --grow-eta-scale 0.5,0.7,1,1
run eta_cold_late --grow-eta-scale 1,1,1.5,2
run eta_hot_cold --grow-eta-scale 0.5,0.7,1.5,2
run c0p1 --spsa-c 0.1
run c0p05 --spsa-c 0.05
run cs_last0p05 --grow-c-schedule 0.15,0.15,0.15,0.05
run cs_taper --grow-c-schedule 0.15,0.15,0.1,0.05
run start2 --grow-start 2
