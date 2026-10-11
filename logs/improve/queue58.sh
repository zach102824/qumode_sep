#!/bin/bash
# After queue57: denser n16 under-3158 mean>=0.9 (R2 E10-E13, rawk64-128, xeta256/512, large-saind>=128 xc3 only),
# n14 sub-1100 near-miss family (E3 xL2 xeta256 rawk64/96/off* seeds), cheap ladder off* confirms,
# light n18/n20 screens. Gray + wmaxsat only.
# Avoid xwarm/lmut/thr/xstop/xagree/xc3-tiny-saind.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue57.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1); X512R2=$(BX 512 2)

echo "[$(date +%H:%M:%S)] === queue58 n16 under-3158 mean>=0.9 ==="
# complements to queue52-57: R2 E10-E13 denser rawk/xeta/off*, large-saind xc3 only
run q58_n16_E10_R2_rawk80_polt5_xs35_xeta256 --n 16 --explore 10 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q58_n16_E10_R2_rawk112_polt5_xs35_xeta256 --n 16 --explore 10 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 112 --polt 5
run q58_n16_E11_R2_rawk96_polt5_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 96 --polt 5
run q58_n16_E11_R2_rawk80_polt5_xs35_xeta512 --n 16 --explore 11 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q58_n16_E12_R2_rawk96_polt5_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 96 --polt 5
run q58_n16_E12_R2_rawk128_polt5_xs35_xeta512 --n 16 --explore 12 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 128 --polt 5
run q58_n16_E13_R2_rawk80_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q58_n16_E13_R2_rawk128_polt5_xs35_xeta512 --n 16 --explore 13 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 128 --polt 5
run q58_n16_E10_R2_rawk64_polt4_xs35_xeta512_off100 --n 16 --explore 10 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 64 --polt 4 --trial-offset 100
run q58_n16_E11_R2_rawk112_polt5_xs35_xeta256_off50 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 112 --polt 5 --trial-offset 50
run q58_n16_E12_R2_rawk80_polt5_xs35_xeta512_off100 --n 16 --explore 12 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 80 --polt 5 --trial-offset 100
run q58_n16_E13_R1_rawk112_polt5_xs35_xeta512 --n 16 --explore 13 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 112 --polt 5
run q58_n16_E14_R2_rawk96_polt5_xs35_xeta256 --n 16 --explore 14 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 96 --polt 5
# careful early-stop only with large saind (>=128)
run q58_n16_E12_R2_xc3_si192_rawk80_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 80 --polt 5 --xconsec 3 --saind 192
run q58_n16_E11_R2_xc3_si256_rawk128_xs35_xeta512 --n 16 --explore 11 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 128 --polt 5 --xconsec 3 --saind 256
run q58_n16_E13_R2_xc3_si128_rawk96_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 96 --polt 5 --xconsec 3 --saind 128
run q58_n16_E10_R2_xc3_si192_rawk112_xs35_xeta512 --n 16 --explore 10 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 112 --polt 5 --xconsec 3 --saind 192

echo "[$(date +%H:%M:%S)] === queue58 n14 sub-1100 near-miss family ==="
# 907-eval E3 xL2 xeta256 rawk64 was 0.92/0.895; push rawk64/96/off* seeds + colder + R2
run q58_n14_E3_R1_xL2_rawk64_xeta256_off300 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 300
run q58_n14_E3_R1_xL2_rawk96_xeta256_off300 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5 --trial-offset 300
run q58_n14_E3_R1_xL2_rawk80_xeta256_off200 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 200
run q58_n14_E3_R2_xL2_rawk64_xeta256_off200 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 64 --polt 4 --trial-offset 200
run q58_n14_E3_R1_xL2_rawk112_xeta256_off100 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 112 --polt 5 --trial-offset 100
run q58_n14_E3_R1_xL2_rawk128_xeta256_off50 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 128 --polt 5 --trial-offset 50
run q58_n14_E3_R2_xL2_rawk80_xeta512_off150 --n 14 --explore 3 --sab 98 $X512R2 --xL 2 --rawk 80 --polt 5 --trial-offset 150
run q58_n14_E3_R1_xL2_rawk96_xeta512_off150 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 96 --polt 5 --trial-offset 150
run q58_n14_E4_R1_xL2_rawk64_xeta256_off150 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 150
run q58_n14_E3_R1_xL2_rawk48_xeta256_off200 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 48 --polt 3 --trial-offset 200

echo "[$(date +%H:%M:%S)] === queue58 cheap ladder confirms ==="
run q58_n08_E1_xL2_xs25_rawk48_off350 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 350
run q58_n10_E1_xL2_xs25_rawk48_off350 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 350
run q58_n12_E1_xL2_xs25_rawk48_off250 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 250
run q58_n14_E4_xL2_rawk48_xeta128_off300 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 300
run q58_n12_E1_xL2_rawk48_xeta256_off200 --n 12 --explore 1 --sab 72 $X256R1 --xL 2 --rawk 48 --polt 3 --trial-offset 200
run q58_n08_E1_xL2_xs35_rawk48_off200 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 35 --rawk 48 --polt 3 --trial-offset 200
run q58_n10_E1_xL2_xs35_rawk48_off200 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 35 --rawk 48 --polt 3 --trial-offset 200

echo "[$(date +%H:%M:%S)] === queue58 n18/n20 light screens ==="
run q58_n18_E24_R1_rawk112_xeta512_t2 --n 18 --explore 24 --sab 162 $X512R1 --xL 3 --xs 50 --rawk 112 --polt 5 --trials 2
run q58_n18_E32_R2_rawk96_xeta256_t2 --n 18 --explore 32 --sab 162 $X256R2 --xL 3 --xs 50 --rawk 96 --polt 5 --trials 2
run q58_n20_E32_R1_rawk112_xeta512_t2 --n 20 --explore 32 --sab 200 $X512R1 --xL 3 --xs 50 --rawk 112 --polt 5 --trials 2
run q58_n20_E40_R2_rawk96_xeta256_t2 --n 20 --explore 40 --sab 200 $X256R2 --xL 3 --xs 50 --rawk 96 --polt 5 --trials 2

touch logs/improve/queue58.done
echo "[$(date +%H:%M:%S)] queue58 ALL DONE"
