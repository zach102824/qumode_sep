#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue5.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
run n14_E11_thr05 --n 14 --explore 11 --sab 196 --thr 0.05 --trials 3
run n14_E11_xstop1 --n 14 --explore 11 --sab 196 --xstop 1 --trials 3
run n14_E11_rawk16 --n 14 --explore 11 --sab 196 --rawk 16 --polt 2 --trials 3
run n12_E4_thr05 --n 12 --explore 4 --sab 144 --thr 0.05 --trials 3
run n12_E4_xstop1 --n 12 --explore 4 --sab 144 --xstop 1 --trials 3
run n12_E4_rawk16 --n 12 --explore 4 --sab 144 --rawk 16 --polt 2 --trials 3
run n14_E11_thr0 --n 14 --explore 11 --sab 196 --thr 0 --trials 3
run n14_E15_xstop1 --n 14 --explore 15 --sab 196 --xstop 1 --trials 3
echo DONE > logs/improve/queue6.done
