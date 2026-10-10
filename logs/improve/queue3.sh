#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
run n14_fl_E9_50 --n 14 --explore 9 --sab 196 --samode fixed_low --trials 5
run n14_fl_E11 --n 14 --explore 11 --sab 196 --samode fixed_low --trials 5
run n14_fl_nz06_E9 --n 14 --explore 9 --sab 196 --samode fixed_low --sanoise 0.6 --trials 5
run n14_fl_sab98_E9 --n 14 --explore 9 --sab 98 --samode fixed_low --trials 5
run n12_fl_E4 --n 12 --explore 4 --sab 144 --samode fixed_low --trials 5
run n12_fl_nz06_E4 --n 12 --explore 4 --sab 144 --samode fixed_low --sanoise 0.6 --trials 5
run n14_fl_E7 --n 14 --explore 7 --sab 196 --samode fixed_low --trials 5
echo DONE > logs/improve/queue3.done
