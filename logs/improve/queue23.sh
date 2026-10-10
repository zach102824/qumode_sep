#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue22.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
B32="${B/--xeta 16/--xeta 32}"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
# n=8 ladder point (not yet measured on this stack)
run n8_E1_sab32_rawk24_xL2_R1_c50 --n 8 --explore 1 --sab 32 $B --xL 2
run n8_E2_sab32_rawk24_xL2_R1_c50 --n 8 --explore 2 --sab 32 $B --xL 2
run n8_E1_sab32_rawk24_xL3_R1_c50 --n 8 --explore 1 --sab 32 $B --xL 3
run n8_E1_sab32_rawk16_xL2_R1_c50 --n 8 --explore 1 --sab 32 $B --xL 2 --rawk 16
run n8_E1_sab32_rawk24_xL2_R1_xeta32_c50 --n 8 --explore 1 --sab 32 $B32 --xL 2
run n8_E3_sab32_rawk24_xL2_R1_c50 --n 8 --explore 3 --sab 32 $B --xL 2
# n=10 fill
run n10_E1_sab50_rawk24_xL2_R1_c50 --n 10 --explore 1 --sab 50 $B --xL 2
run n10_E2_sab50_rawk24_xL2_R1_c50 --n 10 --explore 2 --sab 50 $B --xL 2
run n10_E3_sab50_rawk24_xL2_R1_c50 --n 10 --explore 3 --sab 50 $B --xL 2
run n10_E2_sab50_rawk24_xL3_R1_c50 --n 10 --explore 2 --sab 50 $B --xL 3
# n=14 more under-2000 tries
run n14_E4_sab98_rawk32_xL2_R1_xeta32_c50 --n 14 --explore 4 --sab 98 $B32 --xL 2 --rawk 32
run n14_E5_sab98_rawk32_xL2_R1_xeta32_c50 --n 14 --explore 5 --sab 98 $B32 --xL 2 --rawk 32
run n14_E5_sab98_rawk32_xs35_R1_xeta32_c50 --n 14 --explore 5 --sab 98 $B32 --xs 35 --rawk 32
touch logs/improve/queue23.done
