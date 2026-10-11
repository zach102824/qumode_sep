#!/bin/bash
# priority (starts now, 4 workers): confirm n16 E12 R2 rawk64 polt4 (0.94/0.933 at 3158), push R2 to cheaper E, early-stop combo, n14 sub-1100 and n8/n10 cheaper. Gray only.
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 4 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X1=$(BX 128 1); X2=$(BX 128 2)
run q50_n16_E12_R2_rawk64_xL3_xs35_off50 --n 16 --explore 12 --sab 128 $X2 --xL 3 --xs 35 --rawk 64 --polt 4 --trial-offset 50
run q50_n16_E10_R2_rawk64_xL3_xs35 --n 16 --explore 10 --sab 128 $X2 --xL 3 --xs 35 --rawk 64 --polt 4
run q50_n16_E11_R2_rawk64_xL3_xs35 --n 16 --explore 11 --sab 128 $X2 --xL 3 --xs 35 --rawk 64 --polt 4
run q50_n14_E3_R2_rawk48_xL2 --n 14 --explore 3 --sab 98 $X2 --xL 2 --rawk 48 --polt 3
run q50_n16_E12_R2_rawk64_xL3_xs35_xc3_si64 --n 16 --explore 12 --sab 128 $X2 --xL 3 --xs 35 --rawk 64 --polt 4 --xconsec 3 --saind 64
run q50_n08_E1_xL2_xs25_rawk48 --n 8 --explore 1 --sab 32 $X1 --xL 2 --xs 25 --rawk 48 --polt 3
run q50_n10_E1_xL2_xs25_rawk48 --n 10 --explore 1 --sab 50 $X1 --xL 2 --xs 25 --rawk 48 --polt 3
run q50_n16_E9_R2_rawk64_xL3_xs35 --n 16 --explore 9 --sab 128 $X2 --xL 3 --xs 35 --rawk 64 --polt 4
run q50_n14_E2_R2_rawk48_xL3_xeta256 --n 14 --explore 2 --sab 98 $(BX 256 2) --xL 3 --rawk 48 --polt 3
run q50_n16_E12_R2_rawk64_xL3_xs35_xeta256 --n 16 --explore 12 --sab 128 $(BX 256 2) --xL 3 --xs 35 --rawk 64 --polt 4
touch logs/improve/queue50.done
