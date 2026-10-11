#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue14.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
# More aggressive low-E + diversity (50 trials)
run n14_E4_sab98_rawk24_R1_c50 --n 14 --explore 4 --sab 98 --rawk 24 --polt 2 --trials 5 $B --R 1
run n14_E5_sab98_rawk24_nz40_R1_c50 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 5 $B --sanoise 0.4 --R 1
run n14_E7_sab98_rawk24_xL3_R1_c50 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 5 $B --xL 3 --R 1
run n14_E7_sab98_rawk24_xs35_R1_c50 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 5 $B --xs 35 --R 1
run n14_E9_sab98_rawk24_xL3_R1_c50 --n 14 --explore 9 --sab 98 --rawk 24 --polt 2 --trials 5 $B --xL 3 --R 1
run n14_E9_sab98_rawk24_xs35_R1_c50 --n 14 --explore 9 --sab 98 --rawk 24 --polt 2 --trials 5 $B --xs 35 --R 1
# n12 cheaper
run n12_E1_sab72_rawk24_nz40_c50 --n 12 --explore 1 --sab 72 --rawk 24 --polt 2 --trials 5 $B --sanoise 0.4
run n12_E2_sab72_rawk24_xL3_R1_c50 --n 12 --explore 2 --sab 72 --rawk 24 --polt 2 --trials 5 $B --xL 3 --R 1
run n12_E3_sab72_rawk24_R1_c50 --n 12 --explore 3 --sab 72 --rawk 24 --polt 2 --trials 5 $B --R 1
# rawk32 at low E
run n14_E5_sab98_rawk32_R1_c50 --n 14 --explore 5 --sab 98 --rawk 32 --polt 2 --trials 5 $B --R 1
run n14_E5_sab98_rawk32_xL3_R1_c50 --n 14 --explore 5 --sab 98 --rawk 32 --polt 2 --trials 5 $B --xL 3 --R 1
echo DONE > logs/improve/queue15.done
