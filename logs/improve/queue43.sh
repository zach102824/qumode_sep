#!/bin/bash
# after queue42: deepen n16 mean p(GS) under ~3000 and n14 under ~1100; waits for queue42
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue42.done ]; do sleep 20; done
PAT="run_relayout_kbit.py .*--xeta (16|32) "
cleanup(){ touch logs/improve/queue43.stopwatch_off; sleep 6; pkill -CONT -f "$PAT"; }
trap cleanup EXIT INT TERM
rm -f logs/improve/queue43.stopwatch_off
( while [ ! -f logs/improve/queue43.stopwatch_off ]; do pkill -STOP -f "$PAT"; sleep 4; done ) &
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BX 128); X256=$(BX 256)
# n16: denser E with rawk64; xs50 (full explore SPSA) at E12; rawk96 at E14; R2 rawk96
run n16_E13_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 13 --sab 128 $X128 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E15_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 15 --sab 128 $X128 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E12_sab128_rawk64_polt4_xL3_xs50_R1_xeta128_c50 --n 16 --explore 12 --sab 128 $X128 --xL 3 --xs 50 --rawk 64 --polt 4
run n16_E14_sab128_rawk96_polt5_xL3_xs35_R1_xeta128_c50 --n 16 --explore 14 --sab 128 $X128 --xL 3 --xs 35 --rawk 96 --polt 5
run n16_E12_sab128_rawk96_polt5_xL3_xs35_R2_xeta128_c50 --n 16 --explore 12 --sab 128 ${X128/--R 1/--R 2} --xL 3 --xs 35 --rawk 96 --polt 5
# n14 under ~1100 mean-p pushes: E3 xs50 (no xs35), E3 rawk64, E4 rawk64
run n14_E3_sab98_rawk48_polt3_xL2_xs50_R1_xeta128_c50 --n 14 --explore 3 --sab 98 $X128 --xL 2 --xs 50 --rawk 48 --polt 3
run n14_E3_sab98_rawk64_polt4_xL2_R1_xeta256_c50 --n 14 --explore 3 --sab 98 $X256 --xL 2 --rawk 64 --polt 4
run n14_E4_sab98_rawk64_polt4_xL2_R1_xeta128_c50 --n 14 --explore 4 --sab 98 $X128 --xL 2 --rawk 64 --polt 4
touch logs/improve/queue43.done
