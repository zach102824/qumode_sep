#!/bin/bash
# priority n=16 knob screen at E12 (xL3 xs35 xeta128 base 0.84/0.818 @2857), waits for queue25
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue25.done ]; do sleep 20; done
PAT="run_relayout_kbit.py .*--xeta (16|32) "
cleanup(){ touch logs/improve/queue26.stopwatch_off; sleep 6; pkill -CONT -f "$PAT"; }
trap cleanup EXIT INT TERM
rm -f logs/improve/queue26.stopwatch_off
( while [ ! -f logs/improve/queue26.stopwatch_off ]; do pkill -STOP -f "$PAT"; sleep 4; done ) &
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BX 128); X256=$(BX 256); X512=$(BX 512)
run n16_E12_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 12 --sab 128 $X128 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E12_sab128_rawk24_xL3_xs35_R1_xeta256_c50 --n 16 --explore 12 --sab 128 $X256 --xL 3 --xs 35
run n16_E12_sab128_rawk24_xL3_R1_xeta256_c50 --n 16 --explore 12 --sab 128 $X256 --xL 3
run n16_E12_sab128_rawk24_xL3_xs35_R1_xeta512_c50 --n 16 --explore 12 --sab 128 $X512 --xL 3 --xs 35
run n16_E12_sab192_rawk24_xL3_xs35_R1_xeta128_c50 --n 16 --explore 12 --sab 192 $X128 --xL 3 --xs 35
run n16_E12_sab128_rawk24_xL3_xs35_R1_xeta128_nz40_c50 --n 16 --explore 12 --sab 128 $X128 --xL 3 --xs 35 --sanoise 0.4
run n16_E12_sab128_rawk24_xL3_xs35_R2_xeta128_c50 --n 16 --explore 12 --sab 128 ${X128/--R 1/--R 2} --xL 3 --xs 35
run n16_E16_sab128_rawk24_xL3_xs35_R1_xeta128_c50 --n 16 --explore 16 --sab 128 $X128 --xL 3 --xs 35
run n16_E16_sab128_rawk32_xL3_xs35_R1_xeta256_c50 --n 16 --explore 16 --sab 128 $X256 --xL 3 --xs 35 --rawk 32
# n14 and n12 pushes with the same new knobs (cheap)
run n14_E3_sab98_rawk32_xL3_R1_xeta256_c50 --n 14 --explore 3 --sab 98 $X256 --xL 3 --rawk 32
run n14_E3_sab98_rawk24_xL3_R1_xeta512_c50 --n 14 --explore 3 --sab 98 $X512 --xL 3
run n14_E4_sab98_rawk24_xL2_R1_xeta256_c50 --n 14 --explore 4 --sab 98 $X256 --xL 2
run n14_E4_sab98_rawk32_xL2_R1_xeta128_c50 --n 14 --explore 4 --sab 98 $X128 --xL 2 --rawk 32
run n14_E3_sab98_rawk32_xL2_R1_xeta256_c50 --n 14 --explore 3 --sab 98 $X256 --xL 2 --rawk 32
run n12_E1_sab72_rawk32_xL2_R1_xeta128_c50 --n 12 --explore 1 --sab 72 $X128 --xL 2 --rawk 32
run n12_E1_sab72_rawk24_xL2_R1_xeta512_c50 --n 12 --explore 1 --sab 72 $X512 --xL 2
touch logs/improve/queue26.done
