#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue8.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"
run(){ name=$1; shift; python noiseless/run_relayout_kbit.py "$@" $B > logs/improve/$name.json 2>&1; }
# warm-started + mutated-layout explore runs (cheap: one 50-step stage each), 30-trial screens, with and without rawk
run n14_E13_xw50m2 --n 14 --explore 13 --sab 98 --xwarm 50 --xmut 2 --trials 3
run n14_E13_xw50m2_rawk16 --n 14 --explore 13 --sab 98 --xwarm 50 --xmut 2 --rawk 16 --polt 2 --trials 3
run n14_E13_xw100m4_rawk16 --n 14 --explore 13 --sab 98 --xwarm 100 --xmut 4 --rawk 16 --polt 2 --trials 3
run n14_E9_xw50m1_rawk16 --n 14 --explore 9 --sab 98 --xwarm 50 --xmut 1 --rawk 16 --polt 2 --trials 3
run n12_E5_xw50m2_rawk16 --n 12 --explore 5 --sab 72 --xwarm 50 --xmut 2 --rawk 16 --polt 2 --trials 3
run n12_E5_xw50m2 --n 12 --explore 5 --sab 72 --xwarm 50 --xmut 2 --trials 3
echo DONE > logs/improve/queue9.done
