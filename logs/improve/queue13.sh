#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
# 30-trial screens (10 instances x 3 starts). Overrides after $B win.
# R1 + rawk24 at lower E
run n14_E5_sab98_rawk24_R1 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --R 1
run n14_E6_sab98_rawk24_R1 --n 14 --explore 6 --sab 98 --rawk 24 --polt 2 --trials 3 $B --R 1
run n14_E7_sab98_rawk24_R1 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 3 $B --R 1
run n14_E4_sab98_rawk24_R1 --n 14 --explore 4 --sab 98 --rawk 24 --polt 2 --trials 3 $B --R 1
# xL3 + rawk24 (and with R1)
run n14_E7_sab98_rawk24_xL3 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xL 3
run n14_E5_sab98_rawk24_xL3 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xL 3
run n14_E7_sab98_rawk24_xL3_R1 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xL 3 --R 1
run n14_E5_sab98_rawk24_xL3_R1 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xL 3 --R 1
# xs35 + rawk24 (and with R1)
run n14_E7_sab98_rawk24_xs35 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xs 35
run n14_E5_sab98_rawk24_xs35 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xs 35
run n14_E7_sab98_rawk24_xs35_R1 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xs 35 --R 1
run n14_E5_sab98_rawk24_xs35_R1 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xs 35 --R 1
# n12 stacks
run n12_E2_sab72_rawk24_R1 --n 12 --explore 2 --sab 72 --rawk 24 --polt 2 --trials 3 $B --R 1
run n12_E2_sab72_rawk24_nz40 --n 12 --explore 2 --sab 72 --rawk 24 --polt 2 --trials 3 $B --sanoise 0.4
run n12_E1_sab72_rawk24_nz40 --n 12 --explore 1 --sab 72 --rawk 24 --polt 2 --trials 3 $B --sanoise 0.4
# n14 layout diversity with rawk24
run n14_E5_sab98_rawk24_nz40 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --sanoise 0.4
run n14_E5_sab98_rawk24_nz40_R1 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --sanoise 0.4 --R 1
echo DONE > logs/improve/queue13.done
