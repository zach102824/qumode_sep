#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue6.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
run n14_fl_sab49_E13 --n 14 --explore 13 --sab 49 --trials 5
run n12_fl_sab36_E4 --n 12 --explore 4 --sab 36 --trials 5
run n14_E13_sab98_thr05 --n 14 --explore 13 --sab 98 --thr 0.05 --trials 5
run n14_E15_sab98_xstop1 --n 14 --explore 15 --sab 98 --xstop 1 --trials 5
run n12_E4_sab72_thr05 --n 12 --explore 4 --sab 72 --thr 0.05 --trials 5
echo DONE > logs/improve/queue7.done
