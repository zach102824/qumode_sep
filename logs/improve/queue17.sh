#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue16.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
# n=12 (fast): cheaper explore stacks, adaptive E (xagree), budgets. All 50 trials, R1, rawk24 polt2
run n12_E2_sab72_rawk24_xs35_R1_c50 --n 12 --explore 2 --sab 72 $B --xs 35
run n12_E2_sab72_rawk24_xL3_xs35_R1_c50 --n 12 --explore 2 --sab 72 $B --xL 3 --xs 35
run n12_E3_sab72_rawk24_xL3_xs35_R1_c50 --n 12 --explore 3 --sab 72 $B --xL 3 --xs 35
run n12_E3_sab72_rawk24_xL3_R1_c50 --n 12 --explore 3 --sab 72 $B --xL 3
run n12_E4_sab72_rawk24_xL3_R1_xag2_c50 --n 12 --explore 4 --sab 72 $B --xL 3 --xagree 2
run n12_E5_sab72_rawk24_xL3_xs35_R1_xag2_c50 --n 12 --explore 5 --sab 72 $B --xL 3 --xs 35 --xagree 2
run n12_E3_sab72_rawk24_R1_xag2_c50 --n 12 --explore 3 --sab 72 $B --xagree 2
run n12_E2_sab72_rawk24_xL3_R1_nz40_c50 --n 12 --explore 2 --sab 72 $B --xL 3 --sanoise 0.4
run n12_E2_sab36_rawk24_xL3_R1_c50 --n 12 --explore 2 --sab 36 $B --xL 3
run n12_E2_sab72_rawk24_xL2_R1_c50 --n 12 --explore 2 --sab 72 $B --xL 2
run n12_E3_sab72_rawk24_xL2_R1_c50 --n 12 --explore 3 --sab 72 $B --xL 2
run n12_E2_sab72_rawk24_R1_r100_c50 --n 12 --explore 2 --sab 72 $B --r 100
# n=14 priority: replicate best cheap 0.9+ point with 50 NEW trials (trial offset 5), then cheaper explore stacks
run n14_E6_sab98_rawk24_xL3_R1_rep50 --n 14 --explore 6 --sab 98 $B --xL 3 --trial-offset 5
run n14_E6_sab98_rawk24_xL3_xs35_R1_c50 --n 14 --explore 6 --sab 98 $B --xL 3 --xs 35
run n14_E7_sab98_rawk24_xL3_xs35_R1_c50 --n 14 --explore 7 --sab 98 $B --xL 3 --xs 35
run n14_E6_sab98_rawk24_xs35_R1_c50 --n 14 --explore 6 --sab 98 $B --xs 35
echo DONE > logs/improve/queue17.done
