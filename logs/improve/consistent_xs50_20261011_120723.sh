#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5 --xs 50 --polt 3 --xL 2"
run(){ name=$1; shift; echo "START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "DONE $name"; }
run consxs50_n8_E1_rawk24 --n 8 --explore 1 --sab 36 --rawk 24 $B
run consxs50_n10_E1_rawk30 --n 10 --explore 1 --sab 50 --rawk 30 $B
run consxs50_n12_E1_rawk36 --n 12 --explore 1 --sab 72 --rawk 36 $B
run consxs50_n14_E4_rawk42 --n 14 --explore 4 --sab 98 --rawk 42 $B
run consxs50_n12_E1_rawk48 --n 12 --explore 1 --sab 72 --rawk 48 $B
run consxs50_n14_E4_rawk48 --n 14 --explore 4 --sab 98 --rawk 48 $B
echo ALLDONE
