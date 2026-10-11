#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue20.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
B32="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 32 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }

# --- n=14 push under ~2000 with xeta32 and cheaper stacks ---
run n14_E5_sab98_rawk24_xs35_R1_xeta32_c50 --n 14 --explore 5 --sab 98 $B32 --xs 35
run n14_E5_sab98_rawk24_xL3_R1_xeta32_c50 --n 14 --explore 5 --sab 98 $B32 --xL 3
run n14_E6_sab98_rawk24_xs35_R1_xeta32_c50 --n 14 --explore 6 --sab 98 $B32 --xs 35
run n14_E6_sab98_rawk24_xL2_R1_c50 --n 14 --explore 6 --sab 98 $B --xL 2
run n14_E6_sab98_rawk24_xL2_xs35_R1_c50 --n 14 --explore 6 --sab 98 $B --xL 2 --xs 35
run n14_E5_sab98_rawk24_xL2_R1_c50 --n 14 --explore 5 --sab 98 $B --xL 2
run n14_E5_sab98_rawk24_xL2_xs35_R1_c50 --n 14 --explore 5 --sab 98 $B --xL 2 --xs 35
run n14_E4_sab98_rawk24_xs35_R1_xeta32_c50 --n 14 --explore 4 --sab 98 $B32 --xs 35
run n14_E7_sab98_rawk24_xL3_xs35_R1_xeta32_c50 --n 14 --explore 7 --sab 98 $B32 --xL 3 --xs 35
run n14_E6_sab98_rawk32_xs35_R1_c50 --n 14 --explore 6 --sab 98 $B --xs 35 --rawk 32

# --- n=16 (sab=n^2/2=128); screen E then confirm winners ---
run n16_E8_sab128_rawk24_xs35_R1_c50 --n 16 --explore 8 --sab 128 $B --xs 35
run n16_E10_sab128_rawk24_xs35_R1_c50 --n 16 --explore 10 --sab 128 $B --xs 35
run n16_E12_sab128_rawk24_xs35_R1_c50 --n 16 --explore 12 --sab 128 $B --xs 35
run n16_E14_sab128_rawk24_xs35_R1_c50 --n 16 --explore 14 --sab 128 $B --xs 35
run n16_E8_sab128_rawk24_xL3_R1_c50 --n 16 --explore 8 --sab 128 $B --xL 3
run n16_E10_sab128_rawk24_xL3_R1_c50 --n 16 --explore 10 --sab 128 $B --xL 3
run n16_E12_sab128_rawk24_xL3_R1_c50 --n 16 --explore 12 --sab 128 $B --xL 3
run n16_E10_sab128_rawk24_xL3_xs35_R1_c50 --n 16 --explore 10 --sab 128 $B --xL 3 --xs 35
run n16_E12_sab128_rawk24_xs35_R1_xeta32_c50 --n 16 --explore 12 --sab 128 $B32 --xs 35
run n16_E10_sab128_rawk32_xs35_R1_c50 --n 16 --explore 10 --sab 128 $B --xs 35 --rawk 32
run n16_E12_sab128_rawk24_R1_c50 --n 16 --explore 12 --sab 128 $B
run n16_E16_sab128_rawk24_xs35_R1_c50 --n 16 --explore 16 --sab 128 $B --xs 35

# --- n=12 mirrors for scaling table at matched quality ---
run n12_E2_sab72_rawk24_xL2_R1_xeta32_c50 --n 12 --explore 2 --sab 72 $B32 --xL 2
run n12_E1_sab72_rawk24_xL2_R1_c50 --n 12 --explore 1 --sab 72 $B --xL 2
run n12_E1_sab72_rawk24_xL2_xs35_R1_c50 --n 12 --explore 1 --sab 72 $B --xL 2 --xs 35
run n12_E2_sab72_rawk24_xL2_xs35_R1_c50 --n 12 --explore 2 --sab 72 $B --xL 2 --xs 35

echo DONE > logs/improve/queue21.done
