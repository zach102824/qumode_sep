#!/bin/bash
# After queue62: more n16 under-3158 mean>=0.9 (E13/E12/E11 R1–R2, rawk60/70/90/110 midpoints,
# xeta128/256/512, large-saind>=192 xc3 only), n14 sub-1100 offs on 907 family, cheap ladder,
# light n18/n20. Gray + wmaxsat only. Avoid xwarm/lmut/thr/xstop/xagree/xc3-tiny-saind.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue62.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1); X512R2=$(BX 512 2)

echo "[$(date +%H:%M:%S)] === queue63 n16 under-3158 mean>=0.9 ==="
run q63_n16_E13_R1_rawk60_polt4_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 60 --polt 4
run q63_n16_E13_R1_rawk70_polt4_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 70 --polt 4
run q63_n16_E13_R1_rawk90_polt5_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 90 --polt 5
run q63_n16_E13_R1_rawk110_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 110 --polt 5
run q63_n16_E13_R1_rawk70_polt4_xs35_xeta512 --n 16 --explore 13 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 70 --polt 4
run q63_n16_E13_R1_rawk90_polt5_xs35_xeta512 --n 16 --explore 13 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 90 --polt 5
run q63_n16_E12_R2_rawk60_polt4_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 60 --polt 4
run q63_n16_E12_R2_rawk70_polt4_xs35_xeta128_off150 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 70 --polt 4 --trial-offset 150
run q63_n16_E12_R2_rawk90_polt5_xs35_xeta256_off250 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 90 --polt 5 --trial-offset 250
run q63_n16_E12_R1_rawk70_polt4_xs50_xeta256 --n 16 --explore 12 --sab 128 $X256R1 --xL 3 --xs 50 --rawk 70 --polt 4
run q63_n16_E12_R1_rawk110_polt5_xs35_xeta128 --n 16 --explore 12 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 110 --polt 5
run q63_n16_E11_R1_rawk60_polt4_xs35_xeta128 --n 16 --explore 11 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 60 --polt 4
run q63_n16_E11_R2_rawk70_polt4_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 70 --polt 4
run q63_n16_E11_R2_rawk90_polt5_xs35_xeta128_off200 --n 16 --explore 11 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 90 --polt 5 --trial-offset 200
run q63_n16_E11_R1_rawk110_polt5_xs50_xeta512 --n 16 --explore 11 --sab 128 $X512R1 --xL 3 --xs 50 --rawk 110 --polt 5
run q63_n16_E14_R1_rawk70_polt4_xs35_xeta256 --n 16 --explore 14 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 70 --polt 4
run q63_n16_E14_R1_rawk90_polt5_xs35_xeta128 --n 16 --explore 14 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 90 --polt 5
run q63_n16_E13_R1_xc3_si256_rawk70_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 70 --polt 4 --xconsec 3 --saind 256
run q63_n16_E12_R2_xc3_si192_rawk90_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 90 --polt 5 --xconsec 3 --saind 192
run q63_n16_E11_R2_xc3_si256_rawk60_xs35_xeta128 --n 16 --explore 11 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 60 --polt 4 --xconsec 3 --saind 256
run q63_n16_E14_R1_xc3_si192_rawk110_xs35_xeta256 --n 16 --explore 14 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 110 --polt 5 --xconsec 3 --saind 192

echo "[$(date +%H:%M:%S)] === queue63 n14 sub-1100 907-family offs ==="
run q63_n14_E3_R1_xL2_rawk64_xeta256_off650 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 650
run q63_n14_E3_R1_xL2_rawk80_xeta256_off650 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 650
run q63_n14_E3_R1_xL2_rawk96_xeta256_off650 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5 --trial-offset 650
run q63_n14_E3_R1_xL2_rawk60_xeta256_off700 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 60 --polt 4 --trial-offset 700
run q63_n14_E3_R1_xL2_rawk70_xeta256_off700 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 70 --polt 4 --trial-offset 700
run q63_n14_E3_R1_xL2_rawk90_xeta256_off700 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 90 --polt 5 --trial-offset 700
run q63_n14_E3_R1_xL2_rawk80_xeta512_off600 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 80 --polt 5 --trial-offset 600
run q63_n14_E3_R2_xL2_rawk96_xeta256_off600 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 96 --polt 5 --trial-offset 600
run q63_n14_E2_R1_xL2_rawk96_xeta256_off600 --n 14 --explore 2 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5 --trial-offset 600
run q63_n14_E4_R1_xL2_rawk64_xeta256_off600 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 600

echo "[$(date +%H:%M:%S)] === queue63 cheap ladder confirms ==="
run q63_n08_E1_xL2_xs25_rawk48_off800 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 800
run q63_n10_E1_xL2_xs25_rawk48_off800 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 800
run q63_n12_E1_xL2_xs25_rawk48_off700 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 700
run q63_n14_E4_xL2_rawk48_xeta128_off750 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 750

echo "[$(date +%H:%M:%S)] === queue63 n18/n20 light screens ==="
run q63_n18_E26_R1_rawk90_xeta512_t2 --n 18 --explore 26 --sab 162 $X512R1 --xL 3 --xs 50 --rawk 90 --polt 5 --trials 2
run q63_n18_E42_R2_rawk70_xeta256_t2 --n 18 --explore 42 --sab 162 $X256R2 --xL 3 --xs 50 --rawk 70 --polt 5 --trials 2
run q63_n20_E34_R1_rawk90_xeta512_t2 --n 20 --explore 34 --sab 200 $X512R1 --xL 3 --xs 50 --rawk 90 --polt 5 --trials 2
run q63_n20_E50_R2_rawk70_xeta256_t2 --n 20 --explore 50 --sab 200 $X256R2 --xL 3 --xs 50 --rawk 70 --polt 5 --trials 2

touch logs/improve/queue63.done
echo "[$(date +%H:%M:%S)] queue63 ALL DONE"
