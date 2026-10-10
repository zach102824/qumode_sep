#!/bin/bash
# after queue28: more n16 mean-p(GS) levers if still <0.9; waits for queue28
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue28.done ]; do sleep 20; done
PAT="run_relayout_kbit.py .*--xeta (16|32) "
cleanup(){ touch logs/improve/queue29.stopwatch_off; sleep 6; pkill -CONT -f "$PAT"; }
trap cleanup EXIT INT TERM
rm -f logs/improve/queue29.stopwatch_off
( while [ ! -f logs/improve/queue29.stopwatch_off ]; do pkill -STOP -f "$PAT"; sleep 4; done ) &
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BX 128); X256=$(BX 256); X512=$(BX 512)
run n16_E16_sab128_rawk48_polt3_xL3_xs35_R1_xeta256_c50 --n 16 --explore 16 --sab 128 $X256 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E12_sab128_rawk64_polt4_xL3_xs35_R1_xeta256_c50 --n 16 --explore 12 --sab 128 $X256 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E14_sab128_rawk48_polt3_xL3_xs35_R2_xeta128_c50 --n 16 --explore 14 --sab 128 ${X128/--R 1/--R 2} --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E15_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 15 --sab 128 $X128 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E12_sab128_rawk48_polt3_xL3_xs35_R1_xeta512_c50 --n 16 --explore 12 --sab 128 $X512 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E18_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 18 --sab 128 $X128 --xL 3 --xs 35 --rawk 48 --polt 3
run n14_E3_sab98_rawk48_polt3_xL2_R1_xeta256_c50 --n 14 --explore 3 --sab 98 $X256 --xL 2 --rawk 48 --polt 3
run n14_E4_sab98_rawk48_polt3_xL3_R1_xeta256_c50 --n 14 --explore 4 --sab 98 $X256 --xL 3 --rawk 48 --polt 3
touch logs/improve/queue29.done
