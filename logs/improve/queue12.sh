#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue11.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; }
# cheaper relabel stage (fewer rounds / smaller rounds) with rawk16 polt2; overrides come after $B so they win
run n14_E7_sab98_rawk16_R1 --n 14 --explore 7 --sab 98 --rawk 16 --polt 2 --trials 5 $B --R 1
run n12_E3_sab72_rawk16_R1 --n 12 --explore 3 --sab 72 --rawk 16 --polt 2 --trials 5 $B --R 1
run n14_E7_sab98_rawk16_r100 --n 14 --explore 7 --sab 98 --rawk 16 --polt 2 --trials 5 $B --r 100
run n12_E3_sab72_rawk16_r100 --n 12 --explore 3 --sab 72 --rawk 16 --polt 2 --trials 5 $B --r 100
run n14_E6_sab98_rawk24 --n 14 --explore 6 --sab 98 --rawk 24 --polt 2 --trials 5 $B
run n12_E2_sab72_rawk24 --n 12 --explore 2 --sab 72 --rawk 24 --polt 2 --trials 5 $B
run n14_E7_sab98_rawk16_polt1 --n 14 --explore 7 --sab 98 --rawk 16 --polt 1 --trials 5 $B
run n14_E7_sab49_rawk32 --n 14 --explore 7 --sab 49 --rawk 32 --polt 2 --trials 5 $B
echo DONE > logs/improve/queue12.done
