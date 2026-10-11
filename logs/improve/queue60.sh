#!/bin/bash
# After queue59: denser n16 under-3158 mean>=0.9 (E13 R1 rawk64/80/96 xeta128/256; E12 R2 rawk64 xeta128 offs;
# E11/E12 rawk80–96 midpoints; large-saind>=192 xc3 only), n14 sub-1100 near-miss family (E3 xL2 xeta256 rawk80/96 offs),
# cheap ladder off* confirms, light n18/n20 screens not already in queue51_n16cheap / queue59.
# Gray + wmaxsat only. Avoid xwarm/lmut/thr/xstop/xagree/xc3-tiny-saind.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue59.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1); X512R2=$(BX 512 2)

echo "[$(date +%H:%M:%S)] === queue60 n16 under-3158 mean>=0.9 ==="
# E13 R1 near-miss family (3070 evals was 0.90/0.891): widen rawk + dual xeta
run q60_n16_E13_R1_rawk64_polt4_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 64 --polt 4
run q60_n16_E13_R1_rawk80_polt5_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 80 --polt 5
run q60_n16_E13_R1_rawk96_polt5_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 96 --polt 5
run q60_n16_E13_R1_rawk64_polt4_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 64 --polt 4
run q60_n16_E13_R1_rawk80_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 80 --polt 5
run q60_n16_E13_R1_rawk96_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 96 --polt 5
# E12 R2 rawk64 xeta128 confirms (not the xeta256 path already queued)
run q60_n16_E12_R2_rawk64_polt4_xs35_xeta128 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 64 --polt 4
run q60_n16_E12_R2_rawk64_polt4_xs35_xeta128_off100 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 64 --polt 4 --trial-offset 100
run q60_n16_E12_R2_rawk80_polt5_xs35_xeta128_off50 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 80 --polt 5 --trial-offset 50
# E11/E12 rawk80–96 midpoints with xeta128/256 R1 (queue59 used R2 + other rawks)
run q60_n16_E11_R1_rawk80_polt5_xs35_xeta128 --n 16 --explore 11 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 80 --polt 5
run q60_n16_E11_R1_rawk96_polt5_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 96 --polt 5
run q60_n16_E12_R1_rawk80_polt5_xs35_xeta128 --n 16 --explore 12 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 80 --polt 5
run q60_n16_E12_R1_rawk96_polt5_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 96 --polt 5
run q60_n16_E11_R1_rawk80_polt5_xs50_xeta256 --n 16 --explore 11 --sab 128 $X256R1 --xL 3 --xs 50 --rawk 80 --polt 5
run q60_n16_E12_R1_rawk80_polt5_xs50_xeta128 --n 16 --explore 12 --sab 128 $X128R1 --xL 3 --xs 50 --rawk 80 --polt 5
# careful early-stop only with large saind (>=192)
run q60_n16_E13_R1_xc3_si192_rawk80_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 80 --polt 5 --xconsec 3 --saind 192
run q60_n16_E12_R2_xc3_si256_rawk96_xs35_xeta128 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 96 --polt 5 --xconsec 3 --saind 256
run q60_n16_E11_R2_xc3_si192_rawk80_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 80 --polt 5 --xconsec 3 --saind 192

echo "[$(date +%H:%M:%S)] === queue60 n14 sub-1100 near-miss family ==="
# 907-eval E3 xL2 xeta256 rawk64/96 near-miss 0.92/0.895; fresh offs + rawk80 mid
run q60_n14_E3_R1_xL2_rawk80_xeta256_off400 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 400
run q60_n14_E3_R1_xL2_rawk64_xeta256_off400 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 400
run q60_n14_E3_R1_xL2_rawk96_xeta256_off400 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5 --trial-offset 400
run q60_n14_E3_R1_xL2_rawk80_xeta512_off300 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 80 --polt 5 --trial-offset 300
run q60_n14_E3_R2_xL2_rawk80_xeta256_off400 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 80 --polt 5 --trial-offset 400
run q60_n14_E2_R1_xL2_rawk80_xeta256_off300 --n 14 --explore 2 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 300
run q60_n14_E4_R1_xL2_rawk80_xeta256_off300 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 300
run q60_n14_E3_R1_xL2_rawk80_xeta128_off350 --n 14 --explore 3 --sab 98 $X128R1 --xL 2 --rawk 80 --polt 5 --trial-offset 350

echo "[$(date +%H:%M:%S)] === queue60 cheap ladder confirms ==="
run q60_n08_E1_xL2_xs25_rawk48_off500 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 500
run q60_n10_E1_xL2_xs25_rawk48_off500 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 500
run q60_n12_E1_xL2_xs25_rawk48_off400 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 400
run q60_n14_E4_xL2_rawk48_xeta128_off450 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 450
run q60_n08_E1_xL2_xs35_rawk36_off350 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 35 --rawk 36 --polt 3 --trial-offset 350
run q60_n10_E1_xL2_xs35_rawk36_off350 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 35 --rawk 36 --polt 3 --trial-offset 350
run q60_n12_E1_xL2_rawk64_xeta256_off300 --n 12 --explore 1 --sab 72 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 300

echo "[$(date +%H:%M:%S)] === queue60 n18/n20 light screens (E not covered by queue51_n16cheap / queue59) ==="
run q60_n18_E32_R1_rawk96_xeta256_t2 --n 18 --explore 32 --sab 162 $X256R1 --xL 3 --xs 50 --rawk 96 --polt 5 --trials 2
run q60_n18_E40_R2_rawk80_xeta512_t2 --n 18 --explore 40 --sab 162 $X512R2 --xL 3 --xs 50 --rawk 80 --polt 5 --trials 2
run q60_n20_E40_R1_rawk96_xeta256_t2 --n 20 --explore 40 --sab 200 $X256R1 --xL 3 --xs 50 --rawk 96 --polt 5 --trials 2
run q60_n20_E48_R2_rawk80_xeta512_t2 --n 20 --explore 48 --sab 200 $X512R2 --xL 3 --xs 50 --rawk 80 --polt 5 --trials 2

touch logs/improve/queue60.done
echo "[$(date +%H:%M:%S)] queue60 ALL DONE"
