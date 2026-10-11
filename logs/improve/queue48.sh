#!/bin/bash
# after queue45 and queue47: n16 xeta256 mean>=0.9 points between E13 and E16, n8/n10 independent-seed confirms (trial offset 50), n14 E3/E4 xL3 xs50 combos. Gray only.
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue45.done ] || [ ! -f logs/improve/queue47_xconsec_saind.done ]; do sleep 30; done
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BX 128); X256=$(BX 256)
run n16_E15_sab128_rawk48_polt3_xL3_xs35_R1_xeta256_c50 --n 16 --explore 15 --sab 128 $X256 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E14_sab128_rawk64_polt4_xL3_xs35_R1_xeta256_c50 --n 16 --explore 14 --sab 128 $X256 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E13_sab128_rawk48_polt3_xL3_xs35_R1_xeta256_c50 --n 16 --explore 13 --sab 128 $X256 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E16_sab128_rawk48_polt3_xL3_xs35_R1_xeta512_c50 --n 16 --explore 16 --sab 128 $(BX 512) --xL 3 --xs 35 --rawk 48 --polt 3
# independent 50-trial confirms of the cheap rows (trial offset 50 = fresh seeds)
run n08_E1_sab32_rawk24_polt3_xL2_xs35_R1_xeta128_off50 --n 8 --explore 1 --sab 32 $X128 --xL 2 --xs 35 --rawk 24 --polt 3 --trial-offset 50
run n10_E1_sab50_rawk30_polt3_xL2_xs50_R1_xeta128_off50 --n 10 --explore 1 --sab 50 $X128 --xL 2 --xs 50 --rawk 30 --polt 3 --trial-offset 50
run n12_E1_sab72_rawk36_polt3_xL2_xs50_R1_xeta128_off50 --n 12 --explore 1 --sab 72 $X128 --xL 2 --xs 50 --rawk 36 --polt 3 --trial-offset 50
run n14_E4_sab98_rawk42_polt3_xL2_xs50_R1_xeta128_off50 --n 14 --explore 4 --sab 98 $X128 --xL 2 --xs 50 --rawk 42 --polt 3 --trial-offset 50
touch logs/improve/queue48.done
