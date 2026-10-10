#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue7.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
# rawk16 polt2 gave 1.00/0.993 at 30 trials (sab=n^2 SA); confirm at 50 trials with n^2/2 SA and fewer explore runs
run n14_E11_sab98_rawk16 --n 14 --explore 11 --sab 98 --rawk 16 --polt 2 --trials 5
run n12_E4_sab72_rawk16 --n 12 --explore 4 --sab 72 --rawk 16 --polt 2 --trials 5
run n14_E9_sab98_rawk16 --n 14 --explore 9 --sab 98 --rawk 16 --polt 2 --trials 5
run n12_E3_sab72_rawk16 --n 12 --explore 3 --sab 72 --rawk 16 --polt 2 --trials 5
run n14_E7_sab98_rawk16 --n 14 --explore 7 --sab 98 --rawk 16 --polt 2 --trials 5
run n12_E2_sab72_rawk16 --n 12 --explore 2 --sab 72 --rawk 16 --polt 2 --trials 5
run n14_E9_sab98_rawk8 --n 14 --explore 9 --sab 98 --rawk 8 --polt 2 --trials 5
run n12_E3_sab72_rawk8 --n 12 --explore 3 --sab 72 --rawk 8 --polt 2 --trials 5
run n14_E9_sab49_rawk16 --n 14 --explore 9 --sab 49 --rawk 16 --polt 2 --trials 5
run n12_E3_sab36_rawk16 --n 12 --explore 3 --sab 36 --rawk 16 --polt 2 --trials 5
run n14_E9_sab98_rawk16_polt3 --n 14 --explore 9 --sab 98 --rawk 16 --polt 3 --trials 5
run n14_E9_sab98_rawk16_thr05 --n 14 --explore 9 --sab 98 --rawk 16 --polt 2 --thr 0.05 --trials 5
echo DONE > logs/improve/queue8.done
