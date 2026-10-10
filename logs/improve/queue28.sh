#!/bin/bash
# after queue27: push n16 mean p(GS) toward >=0.9 and denser E scan with rawk48; waits for queue27
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue27.done ]; do sleep 20; done
PAT="run_relayout_kbit.py .*--xeta (16|32) "
cleanup(){ touch logs/improve/queue28.stopwatch_off; sleep 6; pkill -CONT -f "$PAT"; }
trap cleanup EXIT INT TERM
rm -f logs/improve/queue28.stopwatch_off
( while [ ! -f logs/improve/queue28.stopwatch_off ]; do pkill -STOP -f "$PAT"; sleep 4; done ) &
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BX 128); X256=$(BX 256)
# Push n16 mean p(GS): more E / deeper readout / second relabel / larger xeta with rawk48
run n16_E16_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 16 --sab 128 $X128 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E12_sab128_rawk48_polt3_xL3_xs35_R1_xeta256_c50 --n 16 --explore 12 --sab 128 $X256 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E12_sab128_rawk48_polt3_xL3_xs35_R2_xeta128_c50 --n 16 --explore 12 --sab 128 ${X128/--R 1/--R 2} --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E14_sab128_rawk48_polt3_xL3_xs35_R1_xeta256_c50 --n 16 --explore 14 --sab 128 $X256 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E12_sab128_rawk48_polt3_xL2_xs35_R1_xeta128_c50 --n 16 --explore 12 --sab 128 $X128 --xL 2 --xs 35 --rawk 48 --polt 3
# Denser cheaper E scan if E8/E10 in q27 miss (still useful either way)
run n16_E11_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 11 --sab 128 $X128 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E13_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 13 --sab 128 $X128 --xL 3 --xs 35 --rawk 48 --polt 3
run n16_E9_sab128_rawk48_polt3_xL3_xs35_R1_xeta128_c50 --n 16 --explore 9 --sab 128 $X128 --xL 3 --xs 35 --rawk 48 --polt 3
# Apply rawk48 back to n14 under ~2000 / n12 if q27 points are thin
run n14_E4_sab98_rawk48_polt3_xL3_R1_xeta128_c50 --n 14 --explore 4 --sab 98 $X128 --xL 3 --rawk 48 --polt 3
run n14_E5_sab98_rawk48_polt3_xL2_R1_xeta128_c50 --n 14 --explore 5 --sab 98 $X128 --xL 2 --rawk 48 --polt 3
run n12_E1_sab72_rawk48_polt3_xL3_R1_xeta128_c50 --n 12 --explore 1 --sab 72 $X128 --xL 3 --rawk 48 --polt 3
touch logs/improve/queue28.done
