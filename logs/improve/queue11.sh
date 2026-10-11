#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue10.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
# cheaper explore runs (fewer steps per stage) and warm-started explore runs with rawk16, 50 trials
run n14_E9_sab98_rawk16_xs35 --n 14 --explore 9 --sab 98 --xs 35 --rawk 16 --polt 2 --trials 5
run n12_E3_sab72_rawk16_xs35 --n 12 --explore 3 --sab 72 --xs 35 --rawk 16 --polt 2 --trials 5
run n14_E9_sab98_rawk16_xL3 --n 14 --explore 9 --sab 98 --xL 3 --rawk 16 --polt 2 --trials 5
run n12_E3_sab72_rawk16_xL3 --n 12 --explore 3 --sab 72 --xL 3 --rawk 16 --polt 2 --trials 5
run n14_E9_sab98_xw50m2_rawk16 --n 14 --explore 9 --sab 98 --xwarm 50 --xmut 2 --rawk 16 --polt 2 --trials 5
run n12_E4_sab72_xw50m2_rawk16 --n 12 --explore 4 --sab 72 --xwarm 50 --xmut 2 --rawk 16 --polt 2 --trials 5
run n14_E7_sab98_rawk16_polt3 --n 14 --explore 7 --sab 98 --rawk 16 --polt 3 --trials 5
run n14_E7_sab98_rawk16_rawk8 --n 14 --explore 7 --sab 98 --rawk 8 --polt 2 --trials 5
echo DONE > logs/improve/queue11.done
