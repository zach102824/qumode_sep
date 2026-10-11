#!/bin/bash
# PRIORITY (Zach): n16 at about 1700-2000 evals. Fewer, deeper explore runs + diverse SA centers (_div). Runs now, alongside queue29.
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode $2 --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
D=$(BX 128 fixed_low_div); F=$(BX 128 fixed_low)
run n16_E8_div_rawk48_polt3_xL3_xs35_R1_c50 --n 16 --explore 8 --sab 128 $D --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E6_div_rawk48_polt3_xL3_xs50_R1_c50 --n 16 --explore 6 --sab 128 $D --xL 3 --xs 50 --rawk 48 --polt 3
run n16_E6_rawk48_polt3_xL3_xs50_R1_c50 --n 16 --explore 6 --sab 128 $F --xL 3 --xs 50 --rawk 48 --polt 3
run n16_E4_rawk48_polt3_xL4_xs50_R1_c50 --n 16 --explore 4 --sab 128 $F --xL 4 --xs 50 --rawk 48 --polt 3
run n16_E4_rawk48_polt3_xL3_xs80_R2_c50 --n 16 --explore 4 --sab 128 ${F/--R 1/--R 2} --xL 3 --xs 80 --rawk 48 --polt 3
run n16_E5_div_rawk48_polt3_xL4_xs50_R1_c50 --n 16 --explore 5 --sab 128 $D --xL 4 --xs 50 --rawk 48 --polt 3
run n16_E8_rawk48_polt3_xL4_xs35_R1_c50 --n 16 --explore 8 --sab 128 $F --xL 4 --xs 35 --rawk 48 --polt 3
touch logs/improve/queue46.done
