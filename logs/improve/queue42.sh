#!/bin/bash
# after queue29: rawk64 polt4 push at n16 (mean p(GS) 0.898 at E12), cheaper E10/E11, R2, deeper readout; waits for queue29
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue29.done ]; do sleep 20; done
PAT="run_relayout_kbit.py .*--xeta (16|32) "
cleanup(){ touch logs/improve/queue42.stopwatch_off; sleep 6; pkill -CONT -f "$PAT"; }
trap cleanup EXIT INT TERM
rm -f logs/improve/queue42.stopwatch_off
( while [ ! -f logs/improve/queue42.stopwatch_off ]; do pkill -STOP -f "$PAT"; sleep 4; done ) &
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BX 128); X256=$(BX 256)
run n16_E14_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 14 --sab 128 $X128 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E10_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 10 --sab 128 $X128 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E12_sab128_rawk64_polt4_xL3_xs35_R2_xeta128_c50 --n 16 --explore 12 --sab 128 ${X128/--R 1/--R 2} --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E12_sab128_rawk96_polt5_xL3_xs35_R1_xeta128_c50 --n 16 --explore 12 --sab 128 $X128 --xL 3 --xs 35 --rawk 96 --polt 5
run n16_E16_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 16 --sab 128 $X128 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E11_sab128_rawk64_polt4_xL3_xs35_R1_xeta128_c50 --n 16 --explore 11 --sab 128 $X128 --xL 3 --xs 35 --rawk 64 --polt 4
run n16_E12_sab128_rawk64_polt4_xL3_xs35_R1_xeta256_c50 --n 16 --explore 12 --sab 128 $X256 --xL 3 --xs 35 --rawk 64 --polt 4
touch logs/improve/queue42.done
