#!/bin/bash
# Zach-approved: (3) adaptive early stop --xconsec, (2) independent SA per explore run --saind, and both. n16 wmaxsat gray.
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
B="--n 16 --sab 128 --L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5 --xL 3 --rawk 48 --polt 3"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py $B "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
run q47_n16_E16_xc2_xs35 --explore 16 --xs 35 --xconsec 2
run q47_n16_E16_xc3_xs35 --explore 16 --xs 35 --xconsec 3
run q47_n16_E8_si128_xs35 --explore 8 --xs 35 --saind 128
run q47_n16_E16_xc2_si128_xs35 --explore 16 --xs 35 --xconsec 2 --saind 128
run q47_n16_E16_xc3_si128_xs35 --explore 16 --xs 35 --xconsec 3 --saind 128
run q47_n16_E16_xc2_xs50 --explore 16 --xs 50 --xconsec 2
run q47_n16_E16_xc3_xs50 --explore 16 --xs 50 --xconsec 3
run q47_n16_E8_si128_xs50 --explore 8 --xs 50 --saind 128
run q47_n16_E16_xc2_si128_xs50 --explore 16 --xs 50 --xconsec 2 --saind 128
run q47_n16_E16_xc3_si128_xs50 --explore 16 --xs 50 --xconsec 3 --saind 128
run q47_n16_E12_si64_xs35 --explore 12 --xs 35 --saind 64
touch logs/improve/queue47_xconsec_saind.done
