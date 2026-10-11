#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue21.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
B32="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 32 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }

# denser n16 E scan around the first winners (will still be useful even if earlier E failed)
run n16_E6_sab128_rawk24_xs35_R1_c50 --n 16 --explore 6 --sab 128 $B --xs 35
run n16_E7_sab128_rawk24_xs35_R1_c50 --n 16 --explore 7 --sab 128 $B --xs 35
run n16_E9_sab128_rawk24_xs35_R1_c50 --n 16 --explore 9 --sab 128 $B --xs 35
run n16_E11_sab128_rawk24_xs35_R1_c50 --n 16 --explore 11 --sab 128 $B --xs 35
run n16_E13_sab128_rawk24_xs35_R1_c50 --n 16 --explore 13 --sab 128 $B --xs 35
run n16_E15_sab128_rawk24_xs35_R1_c50 --n 16 --explore 15 --sab 128 $B --xs 35
run n16_E18_sab128_rawk24_xs35_R1_c50 --n 16 --explore 18 --sab 128 $B --xs 35
run n16_E20_sab128_rawk24_xs35_R1_c50 --n 16 --explore 20 --sab 128 $B --xs 35
run n16_E12_sab128_rawk24_xL3_xs35_R1_c50 --n 16 --explore 12 --sab 128 $B --xL 3 --xs 35
run n16_E14_sab128_rawk24_xL3_R1_c50 --n 16 --explore 14 --sab 128 $B --xL 3
run n16_E14_sab128_rawk32_xs35_R1_c50 --n 16 --explore 14 --sab 128 $B --xs 35 --rawk 32
run n16_E12_sab64_rawk24_xs35_R1_c50 --n 16 --explore 12 --sab 64 $B --xs 35
run n16_E12_sab128_rawk24_xs35_R1_nz40_c50 --n 16 --explore 12 --sab 128 $B --xs 35 --sanoise 0.4

# n=8 / n=10 quality: E bumps and xeta32
run n08_E1_sab32_rawk24_xL2_R1_xeta32_c50 --n 8 --explore 1 --sab 32 $B32 --xL 2
run n08_E2_sab32_rawk24_xL2_R1_xeta32_c50 --n 8 --explore 2 --sab 32 $B32 --xL 2
run n08_E3_sab32_rawk24_xL2_R1_c50 --n 8 --explore 3 --sab 32 $B --xL 2
run n10_E2_sab50_rawk24_xL2_R1_xeta32_c50 --n 10 --explore 2 --sab 50 $B32 --xL 2
run n10_E3_sab50_rawk24_xL2_R1_xeta32_c50 --n 10 --explore 3 --sab 50 $B32 --xL 2
run n10_E4_sab50_rawk24_xL2_R1_c50 --n 10 --explore 4 --sab 50 $B --xL 2
run n10_E2_sab50_rawk32_xL2_R1_c50 --n 10 --explore 2 --sab 50 $B --xL 2 --rawk 32

# n14 agreement early-stop on cheap stack (xagree) as optional eval saver
run n14_E8_sab98_rawk24_xs35_R1_xag2_c50 --n 14 --explore 8 --sab 98 $B --xs 35 --xagree 2
run n14_E8_sab98_rawk24_xs35_R1_c50 --n 14 --explore 8 --sab 98 $B --xs 35
run n14_E4_sab98_rawk24_xL2_xs35_R1_xeta32_c50 --n 14 --explore 4 --sab 98 $B32 --xL 2 --xs 35

echo DONE > logs/improve/queue22.done
