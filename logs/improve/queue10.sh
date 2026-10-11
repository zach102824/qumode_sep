#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue9.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
# push rawk16 polt2 to fewer explore runs, 50 trials: lower E, quarter SA, more diverse explore layouts (sanoise), bigger raw pool, top-K
run n14_E5_sab98_rawk16 --n 14 --explore 5 --sab 98 --rawk 16 --polt 2 --trials 5
run n14_E6_sab98_rawk16 --n 14 --explore 6 --sab 98 --rawk 16 --polt 2 --trials 5
run n12_E1_sab72_rawk16 --n 12 --explore 1 --sab 72 --rawk 16 --polt 2 --trials 5
run n14_E7_sab49_rawk16 --n 14 --explore 7 --sab 49 --rawk 16 --polt 2 --trials 5
run n12_E2_sab36_rawk16 --n 12 --explore 2 --sab 36 --rawk 16 --polt 2 --trials 5
run n14_E7_sab98_rawk16_nz40 --n 14 --explore 7 --sab 98 --sanoise 0.4 --rawk 16 --polt 2 --trials 5
run n12_E2_sab72_rawk16_nz40 --n 12 --explore 2 --sab 72 --sanoise 0.4 --rawk 16 --polt 2 --trials 5
run n14_E7_sab98_rawk24 --n 14 --explore 7 --sab 98 --rawk 24 --polt 2 --trials 5
run n14_E7_sab98_rawk16_topk3 --n 14 --explore 7 --sab 98 --rawk 16 --polt 2 --topk 3 --trials 5
echo DONE > logs/improve/queue10.done
