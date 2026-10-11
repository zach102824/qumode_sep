#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue15.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
# 30-trial screens: cold layout mutation of best earlier explore layout
run n14_E5_sab98_rawk24_R1_lm2 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --R 1 --lmut 2
run n14_E5_sab98_rawk24_R1_lm4 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --R 1 --lmut 4
run n14_E7_sab98_rawk24_lm2 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 3 $B --lmut 2
run n14_E7_sab98_rawk16_lm2 --n 14 --explore 7 --sab 98 --rawk 16 --polt 2 --trials 3 $B --lmut 2
run n14_E5_sab98_rawk24_xL3_R1_lm2 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 3 $B --xL 3 --R 1 --lmut 2
run n12_E2_sab72_rawk24_R1_lm2 --n 12 --explore 2 --sab 72 --rawk 24 --polt 2 --trials 3 $B --R 1 --lmut 2
run n12_E2_sab72_rawk16_lm2 --n 12 --explore 2 --sab 72 --rawk 16 --polt 2 --trials 3 $B --lmut 2
# 50-trial confirms of the same (reuse screen trials)
run n14_E5_sab98_rawk24_R1_lm2_c50 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 5 $B --R 1 --lmut 2
run n14_E5_sab98_rawk24_R1_lm4_c50 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 5 $B --R 1 --lmut 4
run n14_E7_sab98_rawk24_lm2_c50 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 5 $B --lmut 2
run n14_E5_sab98_rawk24_xL3_R1_lm2_c50 --n 14 --explore 5 --sab 98 --rawk 24 --polt 2 --trials 5 $B --xL 3 --R 1 --lmut 2
run n12_E2_sab72_rawk24_R1_lm2_c50 --n 12 --explore 2 --sab 72 --rawk 24 --polt 2 --trials 5 $B --R 1 --lmut 2
echo DONE > logs/improve/queue16.done
