#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue3.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
run n14_fla_E9 --n 14 --explore 9 --sab 196 --samode fixed_low_adapt --sanoise 0.3 --trials 5
run n14_fla_nz06_E9 --n 14 --explore 9 --sab 196 --samode fixed_low_adapt --sanoise 0.6 --trials 5
run n12_fla_E4 --n 12 --explore 4 --sab 144 --samode fixed_low_adapt --sanoise 0.3 --trials 5
run n12_fla_E3 --n 12 --explore 3 --sab 144 --samode fixed_low_adapt --sanoise 0.3 --trials 5
echo DONE > logs/improve/queue4.done
