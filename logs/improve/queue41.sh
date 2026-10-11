#!/bin/bash
# cheap ladder confirms at matched rawk48 polt3, 1 worker so as not to oversubscribe; n8/10/12 and n14 under ~1100 evals
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
BW(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 1 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BW 128); X256=$(BW 256)
run n08_E1_sab32_rawk48_polt3_xL2_xs35_R1_xeta128_c50 --n 8 --explore 1 --sab 32 $X128 --xL 2 --xs 35 --rawk 48 --polt 3
run n10_E1_sab50_rawk48_polt3_xL2_xs35_R1_xeta128_c50 --n 10 --explore 1 --sab 50 $X128 --xL 2 --xs 35 --rawk 48 --polt 3
run n12_E1_sab72_rawk48_polt3_xL2_xs35_R1_xeta128_c50 --n 12 --explore 1 --sab 72 $X128 --xL 2 --xs 35 --rawk 48 --polt 3
run n08_E1_sab32_rawk48_polt3_xL2_R1_xeta128_c50 --n 8 --explore 1 --sab 32 $X128 --xL 2 --rawk 48 --polt 3
run n10_E1_sab50_rawk48_polt3_xL2_R1_xeta128_c50 --n 10 --explore 1 --sab 50 $X128 --xL 2 --rawk 48 --polt 3
run n14_E4_sab98_rawk48_polt3_xL2_xs35_R1_xeta128_c50 --n 14 --explore 4 --sab 98 $X128 --xL 2 --xs 35 --rawk 48 --polt 3
run n14_E3_sab98_rawk48_polt3_xL2_xs35_R1_xeta128_c50 --n 14 --explore 3 --sab 98 $X128 --xL 2 --xs 35 --rawk 48 --polt 3
run n14_E3_sab98_rawk48_polt3_xL2_R1_xeta128_c50 --n 14 --explore 3 --sab 98 $X128 --xL 2 --rawk 48 --polt 3
run n14_E3_sab98_rawk48_polt3_xL2_R1_xeta256_c50 --n 14 --explore 3 --sab 98 $X256 --xL 2 --rawk 48 --polt 3
run n14_E2_sab98_rawk48_polt3_xL3_R1_xeta128_c50 --n 14 --explore 2 --sab 98 $X128 --xL 3 --rawk 48 --polt 3
touch logs/improve/queue41.done
