#!/bin/bash
# After both queue51 waiters: more n16 under-3000 mean>=0.9 tries (R2 + wide rawk/polt, xs50, xeta256; no tiny-saind xc3),
# n14 sub-1100 near-miss pushes (R2 on 907-eval recipe, denser rawk, third seed), cheap ladder confirms.
# Gray + wmaxsat only. Avoid xwarm/lmut/thr/xstop/xagree/xc3-tiny-saind.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue51.done ] || [ ! -f logs/improve/queue51_n16cheap_n18n20.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1)

echo "[$(date +%H:%M:%S)] === queue52 n16 under-3000 mean>=0.9 (R2 + wide readout / xs50 / xeta256) ==="
# denser rawk on the E12 R2 winner family; xs50+R2+xeta256; E11/E10/E13 fills not already in queue51*
run q52_n16_E12_R2_rawk80_polt5_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q52_n16_E12_R2_rawk112_polt5_xs35_xeta128 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 112 --polt 5
run q52_n16_E12_R2_rawk64_xs50_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 50 --rawk 64 --polt 4
run q52_n16_E12_R2_rawk80_xs50_xeta128 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 50 --rawk 80 --polt 5
run q52_n16_E11_R2_rawk80_polt5_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q52_n16_E10_R2_rawk96_polt5_xs50_xeta256 --n 16 --explore 10 --sab 128 $X256R2 --xL 3 --xs 50 --rawk 96 --polt 5
run q52_n16_E13_R2_rawk80_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q52_n16_E14_R2_rawk80_polt5_xs35_xeta128 --n 16 --explore 14 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q52_n16_E12_R2_rawk64_xs35_xeta512 --n 16 --explore 12 --sab 128 $(BX 512 2) --xL 3 --xs 35 --rawk 64 --polt 4
# careful early-stop only with large saind (>=128)
run q52_n16_E12_R2_xc3_si128_rawk64_xs35 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 64 --polt 4 --xconsec 3 --saind 128
run q52_n16_E12_R1_xc3_si192_rawk64_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 64 --polt 4 --xconsec 3 --saind 192

echo "[$(date +%H:%M:%S)] === queue52 n14 sub-1100 near-miss pushes ==="
# 907-eval near-miss was E3 xL2 xeta256 rawk64 0.92/0.895; try R2, denser rawk, third seed, xeta512
run q52_n14_E3_R2_xL2_rawk64_xeta256 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 64 --polt 4
run q52_n14_E3_R2_xL2_rawk80_xeta256 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 80 --polt 5
run q52_n14_E3_R1_xL2_rawk112_xeta256 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 112 --polt 5
run q52_n14_E3_R1_xL2_rawk64_xeta256_off100 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 100
run q52_n14_E3_R1_xL2_rawk64_xeta512_off50 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 64 --polt 4 --trial-offset 50
run q52_n14_E4_R1_xL2_rawk48_xeta256 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 48 --polt 3
run q52_n14_E3_R1_xL3_rawk80_xeta256 --n 14 --explore 3 --sab 98 $X256R1 --xL 3 --rawk 80 --polt 5

echo "[$(date +%H:%M:%S)] === queue52 cheap n8/n10/n12 confirms ==="
run q52_n08_E1_xL2_xs25_rawk36_off100 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 36 --polt 3 --trial-offset 100
run q52_n10_E1_xL2_xs25_rawk36_off100 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 36 --polt 3 --trial-offset 100
run q52_n12_E1_xL2_xs25_rawk48 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3
run q52_n12_E1_xL2_rawk48_xeta256 --n 12 --explore 1 --sab 72 $X256R1 --xL 2 --rawk 48 --polt 3

echo "[$(date +%H:%M:%S)] === queue52 n18/n20 denser confirms (hams already exist) ==="
run q52_n18_E16_R1_rawk72_xeta256_t2 --n 18 --explore 16 --sab 162 $X256R1 --xL 3 --xs 50 --rawk 72 --polt 4 --trials 2
run q52_n18_E20_R2_rawk72_xeta128_t2 --n 18 --explore 20 --sab 162 $X128R2 --xL 3 --xs 50 --rawk 72 --polt 4 --trials 2
run q52_n20_E20_R1_rawk80_xeta256_t2 --n 20 --explore 20 --sab 200 $X256R1 --xL 3 --xs 50 --rawk 80 --polt 4 --trials 2
run q52_n20_E24_R2_rawk80_xeta128_t2 --n 20 --explore 24 --sab 200 $X128R2 --xL 3 --xs 50 --rawk 80 --polt 4 --trials 2

touch logs/improve/queue52.done
echo "[$(date +%H:%M:%S)] queue52 ALL DONE"
