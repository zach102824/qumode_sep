#!/bin/bash
# after queue43: more n16 under-3000 / E17-18 and n12/n8 confirms; waits for queue43
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue43.done ]; do sleep 20; done
PAT="run_relayout_kbit.py .*--xeta (16|32) "
cleanup(){ touch logs/improve/queue44.stopwatch_off; sleep 6; pkill -CONT -f "$PAT"; }
trap cleanup EXIT INT TERM
rm -f logs/improve/queue44.stopwatch_off
( while [ ! -f logs/improve/queue44.stopwatch_off ]; do pkill -STOP -f "$PAT"; sleep 4; done ) &
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BX 128); X256=$(BX 256)
# n16: E17/E18 rawk64; E12 rawk128; E14 R2 rawk64; E16 rawk96
run n16_E17_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 17 --sab 128 $X128 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E18_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 18 --sab 128 $X128 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E12_sab128_rawk128_polt5_xL3_xs35_R1_xeta128_c50 --n 16 --explore 12 --sab 128 $X128 --xL 3 --xs 35 --rawk 128 --polt 5
run n16_E14_sab128_rawk64_polt4_xL3_xs35_R2_xeta128_c50 --n 16 --explore 14 --sab 128 ${X128/--R 1/--R 2} --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E16_sab128_rawk96_polt5_xL3_xs35_R1_xeta128_c50 --n 16 --explore 16 --sab 128 $X128 --xL 3 --xs 35 --rawk 96 --polt 5
# cheap ladder confirms / n12 xL3 mirror
run n12_E1_sab72_rawk48_polt3_xL3_xs50_R1_xeta128_c50 --n 12 --explore 1 --sab 72 $X128 --xL 3 --xs 50 --rawk 48 --polt 3
run n14_E3_sab98_rawk48_polt3_xL3_xs50_R1_xeta128_c50 --n 14 --explore 3 --sab 98 $X128 --xL 3 --xs 50 --rawk 48 --polt 3
run n08_E1_sab32_rawk24_polt3_xL2_xs35_R1_xeta128_c50 --n 8 --explore 1 --sab 32 $X128 --xL 2 --xs 35 --rawk 24 --polt 3
touch logs/improve/queue44.done
