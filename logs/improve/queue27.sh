#!/bin/bash
# follow-up on rawk48 polt3 win at n16 E12 (0.90/0.878 @2857); waits for queue26
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue26.done ]; do sleep 20; done
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X=$(BX 128)
run n16_E10_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 10 --sab 128 $X --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E12_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 12 --sab 128 $X --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E8_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 8 --sab 128 $X --xL 3 --xs 35 --rawk 48 --polt 3
run n14_E3_sab98_rawk48_polt3_xL3_R1_xeta256_c50 --n 14 --explore 3 --sab 98 $(BX 256) --xL 3 --rawk 48 --polt 3
run n14_E4_sab98_rawk48_polt3_xL2_R1_xeta128_c50 --n 14 --explore 4 --sab 98 $X --xL 2 --rawk 48 --polt 3
run n12_E1_sab72_rawk48_polt3_xL2_R1_xeta128_c50 --n 12 --explore 1 --sab 72 $X --xL 2 --rawk 48 --polt 3
run n16_E14_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 14 --sab 128 $X --xL 3 --xs 35 --rawk 48 --polt 3
touch logs/improve/queue27.done
