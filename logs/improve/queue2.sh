#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
run n14_nz06 --n 14 --explore 9 --sab 196 --sanoise 0.6 --trials 3
run n14_nz03 --n 14 --explore 9 --sab 196 --sanoise 0.3 --trials 3
run n14_fl --n 14 --explore 9 --sab 196 --samode fixed_low --trials 3
run n14_sab98 --n 14 --explore 9 --sab 98 --sanoise 0.3 --trials 3
run n12_E3_fl --n 12 --explore 3 --sab 144 --samode fixed_low --trials 5
run n12_E3_unc --n 12 --explore 3 --sab 144 --sanoise 0.3 --trials 5
echo DONE > logs/improve/queue2.done
