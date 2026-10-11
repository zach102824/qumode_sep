#!/bin/bash
# After queue55: keep hunting n16 mean>=0.9 under 3158 (R2, rawk64-128, xeta256/512, large-saind>=128 xc3 only),
# n14 sub-1100 near-miss family (E3 xL2 xeta256 rawk64/96/off*), cheap ladder confirms with fresh offsets,
# optional n18/n20 light screens. Gray + wmaxsat only.
# Avoid xwarm/lmut/thr/xstop/xagree/xc3-tiny-saind.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue55.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1); X512R2=$(BX 512 2)

echo "[$(date +%H:%M:%S)] === queue56 n16 under-3158 mean>=0.9 ==="
# complements to queue52-55: E10-E14 denser, colder xeta, wider rawk, R2 + large-saind xc3 only
run q56_n16_E10_R2_rawk96_polt5_xs35_xeta256 --n 16 --explore 10 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 96 --polt 5
run q56_n16_E11_R2_rawk64_polt4_xs35_xeta512 --n 16 --explore 11 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 64 --polt 4
run q56_n16_E12_R2_rawk128_polt5_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 128 --polt 5
run q56_n16_E13_R1_rawk128_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 128 --polt 5
run q56_n16_E14_R1_rawk64_polt4_xs35_xeta512 --n 16 --explore 14 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 64 --polt 4
run q56_n16_E12_R1_rawk80_polt5_xs35_xeta512 --n 16 --explore 12 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 80 --polt 5
run q56_n16_E11_R2_rawk112_polt5_xs35_xeta512 --n 16 --explore 11 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 112 --polt 5
run q56_n16_E10_R2_rawk112_polt5_xs35_xeta512 --n 16 --explore 10 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 112 --polt 5
run q56_n16_E13_R2_rawk64_polt4_xs35_xeta256_off50 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 64 --polt 4 --trial-offset 50
run q56_n16_E12_R2_rawk96_polt5_xs35_xeta512 --n 16 --explore 12 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 96 --polt 5
run q56_n16_E14_R2_rawk48_polt3_xs35_xeta128 --n 16 --explore 14 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 48 --polt 3
# careful early-stop only with large saind (>=128)
run q56_n16_E12_R2_xc3_si192_rawk128_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 128 --polt 5 --xconsec 3 --saind 192
run q56_n16_E11_R2_xc3_si256_rawk96_xs35_xeta512 --n 16 --explore 11 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 96 --polt 5 --xconsec 3 --saind 256
run q56_n16_E13_R1_xc3_si256_rawk80_xs35_xeta512 --n 16 --explore 13 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 80 --polt 5 --xconsec 3 --saind 256

echo "[$(date +%H:%M:%S)] === queue56 n14 sub-1100 near-miss family ==="
# 907-eval E3 xL2 xeta256 rawk64 was 0.92/0.895; push rawk64/96/off* + colder + R2
run q56_n14_E3_R1_xL2_rawk64_xeta256_off100 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 100
run q56_n14_E3_R1_xL2_rawk96_xeta256_off150 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5 --trial-offset 150
run q56_n14_E3_R2_xL2_rawk64_xeta256_off100 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 64 --polt 4 --trial-offset 100
run q56_n14_E3_R1_xL2_rawk128_xeta256 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 128 --polt 5
run q56_n14_E3_R1_xL2_rawk80_xeta512_off50 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 80 --polt 5 --trial-offset 50
run q56_n14_E3_R2_xL2_rawk96_xeta512 --n 14 --explore 3 --sab 98 $X512R2 --xL 2 --rawk 96 --polt 5
run q56_n14_E4_R1_xL2_rawk80_xeta256_off50 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 50
run q56_n14_E3_R1_xL2_rawk48_xeta256_off50 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 48 --polt 3 --trial-offset 50

echo "[$(date +%H:%M:%S)] === queue56 cheap ladder confirms ==="
run q56_n08_E1_xL2_xs25_rawk48_off250 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 250
run q56_n10_E1_xL2_xs25_rawk48_off250 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 250
run q56_n12_E1_xL2_xs25_rawk48_off150 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 150
run q56_n14_E4_xL2_rawk48_xeta128_off200 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 200
run q56_n12_E1_xL2_rawk48_xeta256_off100 --n 12 --explore 1 --sab 72 $X256R1 --xL 2 --rawk 48 --polt 3 --trial-offset 100
run q56_n08_E1_xL2_xs35_rawk48_off100 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 35 --rawk 48 --polt 3 --trial-offset 100

echo "[$(date +%H:%M:%S)] === queue56 n18/n20 light screens ==="
run q56_n18_E20_R1_rawk96_xeta512_t2 --n 18 --explore 20 --sab 162 $X512R1 --xL 3 --xs 50 --rawk 96 --polt 5 --trials 2
run q56_n18_E24_R2_rawk80_xeta256_t2 --n 18 --explore 24 --sab 162 $X256R2 --xL 3 --xs 50 --rawk 80 --polt 4 --trials 2
run q56_n20_E24_R1_rawk96_xeta512_t2 --n 20 --explore 24 --sab 200 $X512R1 --xL 3 --xs 50 --rawk 96 --polt 5 --trials 2
run q56_n20_E40_R1_rawk80_xeta256_t2 --n 20 --explore 40 --sab 200 $X256R1 --xL 3 --xs 50 --rawk 80 --polt 4 --trials 2

touch logs/improve/queue56.done
echo "[$(date +%H:%M:%S)] queue56 ALL DONE"
