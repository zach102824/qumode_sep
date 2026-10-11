#!/bin/bash
# after queue49: n14 sub-1100 near-miss pushes (complement to queue51_n16cheap_n18n20 which covers n16/n18/n20 after queue50), plus cheap n8/n10/n12 confirms and a few non-overlapping n16 confirms. Gray only. Avoid xwarm/lmut/thr/xstop/xagree/xc3-small-saind.
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue49.done ]; do sleep 30; done
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
X128R1=$(BX 128 1); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1)
# n14: sub-1100 / sub-1000 near-miss pushes (907-eval E3 xL2 xeta256 rawk64 was 0.92/0.895)
run q51_n14_E3_R1_xL2_rawk96_xeta256 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5
run q51_n14_E3_R1_xL2_rawk64_xeta256_off50 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 50
run q51_n14_E3_R1_xL2_rawk80_xeta256 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5
run q51_n14_E3_R1_xL2_rawk64_xeta512 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 64 --polt 4
run q51_n14_E2_R1_xL3_rawk64_xeta256 --n 14 --explore 2 --sab 98 $X256R1 --xL 3 --rawk 64 --polt 4
run q51_n14_E2_R1_xL3_rawk48_xeta256 --n 14 --explore 2 --sab 98 $X256R1 --xL 3 --rawk 48 --polt 3
run q51_n14_E2_R1_xL2_rawk64_xeta256 --n 14 --explore 2 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4
run q51_n14_E4_R1_xL2_xs35_rawk48_xeta128 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --xs 35 --rawk 48 --polt 3
# n16: only complements (off50 confirm of live E12 R2 xeta256; E14 R2 xeta256 quality check)
run q51_n16_E12_R2_rawk64_xL3_xs35_xeta256_off50 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 64 --polt 4 --trial-offset 50
run q51_n16_E14_R2_rawk64_xL3_xs35_xeta256 --n 16 --explore 14 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 64 --polt 4
# cheap ladder confirms (independent seeds)
run q51_n08_E1_xL2_xs25_rawk48_off50 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 50
run q51_n10_E1_xL2_xs25_rawk48_off50 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 50
run q51_n12_E1_xL2_rawk48_xeta128_off50 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 50
touch logs/improve/queue51.done
