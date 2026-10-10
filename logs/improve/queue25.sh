#!/bin/bash
# priority xeta>=64 ladder: while this runs, any queue21-style job (xeta 16/32) is SIGSTOPped (cores stay at 8), then SIGCONT at end
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
PAT="run_relayout_kbit.py .*--xeta (16|32) "
cleanup(){ touch logs/improve/queue25.stopwatch_off; sleep 6; pkill -CONT -f "$PAT"; }
trap cleanup EXIT INT TERM
rm -f logs/improve/queue25.stopwatch_off
( while [ ! -f logs/improve/queue25.stopwatch_off ]; do pkill -STOP -f "$PAT"; sleep 4; done ) &
BX(){ echo "--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128=$(BX 128); X256=$(BX 256); X64=$(BX 64)
# n=8 / 10 / 12 cheap xeta128
for n in 8 10 12; do sab=$((n*n/2)); nn=$(printf "%d" $n)
  for E in 1 2; do
    run n${n}_E${E}_sab${sab}_rawk24_xL2_xs35_R1_xeta128_c50 --n $n --explore $E --sab $sab $X128 --xL 2 --xs 35
    run n${n}_E${E}_sab${sab}_rawk24_xL2_R1_xeta128_c50 --n $n --explore $E --sab $sab $X128 --xL 2
    run n${n}_E${E}_sab${sab}_rawk24_xL3_xs35_R1_xeta128_c50 --n $n --explore $E --sab $sab $X128 --xL 3 --xs 35
  done
done
run n12_E1_sab72_rawk24_xL2_R1_xeta256_c50 --n 12 --explore 1 --sab 72 $X256 --xL 2
run n12_E1_sab72_rawk24_xL2_xs35_R1_xeta256_c50 --n 12 --explore 1 --sab 72 $X256 --xL 2 --xs 35
run n12_E1_sab72_rawk24_xL3_xs35_R1_xeta128_nz40_c50 --n 12 --explore 1 --sab 72 $X128 --xL 3 --xs 35 --sanoise 0.4
# n=14 under 2000 with xeta128
for E in 3 4 5; do
  run n14_E${E}_sab98_rawk24_xL3_xs35_R1_xeta128_c50 --n 14 --explore $E --sab 98 $X128 --xL 3 --xs 35
  run n14_E${E}_sab98_rawk24_xL3_R1_xeta256_c50 --n 14 --explore $E --sab 98 $X256 --xL 3
  run n14_E${E}_sab98_rawk24_xL2_R1_xeta128_c50 --n 14 --explore $E --sab 98 $X128 --xL 2
  run n14_E${E}_sab98_rawk24_xL2_xs35_R1_xeta128_c50 --n 14 --explore $E --sab 98 $X128 --xL 2 --xs 35
done
run n14_E4_sab98_rawk24_xL3_R1_xeta128_nz40_c50 --n 14 --explore 4 --sab 98 $X128 --xL 3 --sanoise 0.4
run n14_E4_sab98_rawk32_xL3_R1_xeta128_c50 --n 14 --explore 4 --sab 98 $X128 --xL 3 --rawk 32
# n=16 xeta128 with cheap explore
run n16_E12_sab128_rawk24_xL3_xs35_R1_xeta128_c50 --n 16 --explore 12 --sab 128 $X128 --xL 3 --xs 35
run n16_E8_sab128_rawk24_xL3_xs35_R1_xeta128_c50 --n 16 --explore 8 --sab 128 $X128 --xL 3 --xs 35
run n16_E6_sab128_rawk24_xL3_xs35_R1_xeta128_c50 --n 16 --explore 6 --sab 128 $X128 --xL 3 --xs 35
run n16_E10_sab128_rawk24_xL3_R1_xeta128_c50 --n 16 --explore 10 --sab 128 $X128 --xL 3
run n16_E8_sab128_rawk24_xL3_R1_xeta256_c50 --n 16 --explore 8 --sab 128 $X256 --xL 3
run n16_E8_sab128_rawk24_xL2_xs35_R1_xeta128_c50 --n 16 --explore 8 --sab 128 $X128 --xL 2 --xs 35
run n16_E8_sab128_rawk32_xL3_xs35_R1_xeta128_c50 --n 16 --explore 8 --sab 128 $X128 --xL 3 --xs 35 --rawk 32
run n16_E8_sab192_rawk24_xL3_xs35_R1_xeta128_c50 --n 16 --explore 8 --sab 192 $X128 --xL 3 --xs 35
run n16_E8_sab128_rawk24_xL3_xs35_R1_xeta128_nz40_c50 --n 16 --explore 8 --sab 128 $X128 --xL 3 --xs 35 --sanoise 0.4
touch logs/improve/queue25.done
