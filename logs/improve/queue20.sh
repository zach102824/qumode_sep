#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
B2="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
while [ ! -f logs/improve/queue19.done ]; do sleep 20; done
run n12_E1_sab72_rawk24_xL3_R1_xeta32_c50 --n 12 --explore 1 --sab 72 $B2 --xeta 32 --xL 3
run n12_E2_sab72_rawk24_xL3_R1_xeta32_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 32 --xL 3
run n12_E3_sab72_rawk24_xL3_R1_xeta32_c50 --n 12 --explore 3 --sab 72 $B2 --xeta 32 --xL 3
run n12_E1_sab72_rawk24_xL3_R1_xeta64_c50 --n 12 --explore 1 --sab 72 $B2 --xeta 64 --xL 3
run n12_E2_sab72_rawk24_xL3_R1_xeta64_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 64 --xL 3
run n12_E3_sab72_rawk24_xL3_R1_xeta64_c50 --n 12 --explore 3 --sab 72 $B2 --xeta 64 --xL 3
run n12_E1_sab72_rawk24_xL3_R1_xeta128_c50 --n 12 --explore 1 --sab 72 $B2 --xeta 128 --xL 3
run n12_E2_sab72_rawk24_xL3_R1_xeta128_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 128 --xL 3
run n12_E3_sab72_rawk24_xL3_R1_xeta128_c50 --n 12 --explore 3 --sab 72 $B2 --xeta 128 --xL 3
run n14_E3_sab98_rawk24_xL3_R1_xeta32_c50 --n 14 --explore 3 --sab 98 $B2 --xeta 32 --xL 3
run n14_E4_sab98_rawk24_xL3_R1_xeta32_c50 --n 14 --explore 4 --sab 98 $B2 --xeta 32 --xL 3
run n14_E5_sab98_rawk24_xL3_R1_xeta32_c50 --n 14 --explore 5 --sab 98 $B2 --xeta 32 --xL 3
run n14_E6_sab98_rawk24_xL3_R1_xeta32_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 32 --xL 3
run n14_E3_sab98_rawk24_xL3_R1_xeta64_c50 --n 14 --explore 3 --sab 98 $B2 --xeta 64 --xL 3
run n14_E4_sab98_rawk24_xL3_R1_xeta64_c50 --n 14 --explore 4 --sab 98 $B2 --xeta 64 --xL 3
run n14_E5_sab98_rawk24_xL3_R1_xeta64_c50 --n 14 --explore 5 --sab 98 $B2 --xeta 64 --xL 3
run n14_E6_sab98_rawk24_xL3_R1_xeta64_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 64 --xL 3
run n14_E3_sab98_rawk24_xL3_R1_xeta128_c50 --n 14 --explore 3 --sab 98 $B2 --xeta 128 --xL 3
run n14_E4_sab98_rawk24_xL3_R1_xeta128_c50 --n 14 --explore 4 --sab 98 $B2 --xeta 128 --xL 3
run n14_E5_sab98_rawk24_xL3_R1_xeta128_c50 --n 14 --explore 5 --sab 98 $B2 --xeta 128 --xL 3
run n14_E6_sab98_rawk24_xL3_R1_xeta128_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 128 --xL 3
run n14_E4_sab98_rawk24_xL2_R1_xeta32_c50 --n 14 --explore 4 --sab 98 $B2 --xeta 32 --xL 2
run n14_E5_sab98_rawk24_xL2_R1_xeta32_c50 --n 14 --explore 5 --sab 98 $B2 --xeta 32 --xL 2
run n14_E6_sab98_rawk24_xL2_R1_xeta32_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 32 --xL 2
run n12_E1_sab72_rawk24_xL2_R1_xeta32_c50 --n 12 --explore 1 --sab 72 $B2 --xeta 32 --xL 2
run n12_E2_sab72_rawk24_xL2_R1_xeta32_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 32 --xL 2
run n14_E4_sab98_rawk24_xL2_R1_xeta64_c50 --n 14 --explore 4 --sab 98 $B2 --xeta 64 --xL 2
run n14_E5_sab98_rawk24_xL2_R1_xeta64_c50 --n 14 --explore 5 --sab 98 $B2 --xeta 64 --xL 2
run n14_E6_sab98_rawk24_xL2_R1_xeta64_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 64 --xL 2
run n12_E1_sab72_rawk24_xL2_R1_xeta64_c50 --n 12 --explore 1 --sab 72 $B2 --xeta 64 --xL 2
run n12_E2_sab72_rawk24_xL2_R1_xeta64_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 64 --xL 2
run n14_E5_sab98_rawk24_xL3_xs35_R1_xeta32_c50 --n 14 --explore 5 --sab 98 $B2 --xeta 32 --xL 3 --xs 35
run n14_E6_sab98_rawk24_xL3_xs35_R1_xeta32_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 32 --xL 3 --xs 35
run n12_E2_sab72_rawk24_xL3_xs35_R1_xeta32_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 32 --xL 3 --xs 35
run n12_E3_sab72_rawk24_xL3_xs35_R1_xeta32_c50 --n 12 --explore 3 --sab 72 $B2 --xeta 32 --xL 3 --xs 35
run n14_E5_sab98_rawk24_xL3_xs35_R1_xeta64_c50 --n 14 --explore 5 --sab 98 $B2 --xeta 64 --xL 3 --xs 35
run n14_E6_sab98_rawk24_xL3_xs35_R1_xeta64_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 64 --xL 3 --xs 35
run n12_E2_sab72_rawk24_xL3_xs35_R1_xeta64_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 64 --xL 3 --xs 35
run n12_E3_sab72_rawk24_xL3_xs35_R1_xeta64_c50 --n 12 --explore 3 --sab 72 $B2 --xeta 64 --xL 3 --xs 35
run n14_E6_sab98_rawk24_xL3_R1_xeta32_rep50 --n 14 --explore 6 --sab 98 $B2 --xeta 32 --xL 3 --trial-offset 5
run n12_E2_sab72_rawk24_xL3_R1_xeta32_rep50 --n 12 --explore 2 --sab 72 $B2 --xeta 32 --xL 3 --trial-offset 5
echo DONE > logs/improve/queue20.done
