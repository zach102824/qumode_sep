#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue17.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
# n=14 adaptive E (agreement stop) and budget variants
run n14_E9_sab98_rawk24_xL3_xs35_R1_xag2_c50 --n 14 --explore 9 --sab 98 $B --xL 3 --xs 35 --xagree 2
run n14_E9_sab98_rawk24_xL3_R1_xag2_c50 --n 14 --explore 9 --sab 98 $B --xL 3 --xagree 2
run n14_E9_sab98_rawk24_xL3_xs35_R1_xag3_c50 --n 14 --explore 9 --sab 98 $B --xL 3 --xs 35 --xagree 3
run n14_E7_sab98_rawk24_xs35_R1_xag2_c50 --n 14 --explore 7 --sab 98 $B --xs 35 --xagree 2
run n14_E6_sab98_rawk24_xL3_R1_sab49_c50 --n 14 --explore 6 --sab 49 $B --xL 3
run n14_E6_sab98_rawk24_xL3_R1_nz40_c50 --n 14 --explore 6 --sab 98 $B --xL 3 --sanoise 0.4
run n14_E6_sab98_rawk24_xL3_R1_polt3_c50 --n 14 --explore 6 --sab 98 $B --xL 3 --polt 3
run n14_E6_sab98_rawk24_xL3_R1_r100_c50 --n 14 --explore 6 --sab 98 $B --xL 3 --r 100
run n14_E7_sab98_rawk24_xL2_R1_c50 --n 14 --explore 7 --sab 98 $B --xL 2
run n14_E9_sab98_rawk24_xL2_xs35_R1_c50 --n 14 --explore 9 --sab 98 $B --xL 2 --xs 35
echo DONE > logs/improve/queue18.done
