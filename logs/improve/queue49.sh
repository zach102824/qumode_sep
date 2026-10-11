#!/bin/bash
# after queue48: cheaper per-run SA with early stop xc3 at n16 (si32/si64), E12 cap with si128. Gray only.
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue48.done ]; do sleep 30; done
B="--n 16 --sab 128 --L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5 --xL 3 --rawk 48 --polt 3"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py $B "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
run q49_n16_E16_xc3_si64_xs35 --explore 16 --xs 35 --xconsec 3 --saind 64
run q49_n16_E16_xc3_si32_xs35 --explore 16 --xs 35 --xconsec 3 --saind 32
run q49_n16_E12_xc3_si128_xs35 --explore 12 --xs 35 --xconsec 3 --saind 128
run q49_n16_E16_xc3_si128_xs35_off50 --explore 16 --xs 35 --xconsec 3 --saind 128 --trial-offset 50
touch logs/improve/queue49.done
