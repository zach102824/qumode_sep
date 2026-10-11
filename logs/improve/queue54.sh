#!/bin/bash
# After queue53: keep hunting n16 mean>=0.9 under 3158 (R2 E10-E13, xeta256/512, rawk80-128, large-saind xc3 only),
# n14 sub-1100 near-miss pushes (907-eval E3 xL2 xeta256 rawk64 was 0.92/0.895), cheap ladder off50/off100 confirms.
# Gray + wmaxsat only. Avoid xwarm/lmut/thr/xstop/xagree/xc3-tiny-saind.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue53.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1); X512R2=$(BX 512 2)

echo "[$(date +%H:%M:%S)] === queue54 n16 under-3158 mean>=0.9 ==="
# complements to queue52/53: denser rawk on E13/E10/E11, colder xeta512, xs50+R2 mixes, large-saind only
run q54_n16_E13_R2_rawk96_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 96 --polt 5
run q54_n16_E13_R2_rawk112_polt5_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 112 --polt 5
run q54_n16_E10_R2_rawk80_polt5_xs35_xeta512 --n 16 --explore 10 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q54_n16_E11_R2_rawk128_polt5_xs35_xeta128 --n 16 --explore 11 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 128 --polt 5
run q54_n16_E12_R2_rawk96_polt5_xs50_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 50 --rawk 96 --polt 5
run q54_n16_E12_R1_rawk112_polt5_xs35_xeta512 --n 16 --explore 12 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 112 --polt 5
run q54_n16_E10_R2_rawk128_polt5_xs35_xeta256 --n 16 --explore 10 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 128 --polt 5
run q54_n16_E13_R2_rawk80_polt5_xs50_xeta256 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 50 --rawk 80 --polt 5
run q54_n16_E11_R2_rawk80_polt5_xs35_xeta512 --n 16 --explore 11 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q54_n16_E12_R2_rawk64_polt4_xs35_xeta256_off100 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 64 --polt 4 --trial-offset 100
# careful early-stop only with large saind (>=128)
run q54_n16_E12_R2_xc3_si256_rawk80_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 80 --polt 5 --xconsec 3 --saind 256
run q54_n16_E13_R1_xc3_si192_rawk96_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 96 --polt 5 --xconsec 3 --saind 192

echo "[$(date +%H:%M:%S)] === queue54 n14 sub-1100 near-miss pushes ==="
# 907-eval E3 xL2 xeta256 rawk64 was 0.92/0.895; push denser rawk / colder / R2 / more seeds
run q54_n14_E3_R2_xL2_rawk96_xeta256 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 96 --polt 5
run q54_n14_E3_R1_xL2_rawk96_xeta512 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 96 --polt 5
run q54_n14_E3_R2_xL2_rawk112_xeta256 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 112 --polt 5
run q54_n14_E3_R1_xL2_rawk80_xeta256_off100 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 100
run q54_n14_E4_R1_xL2_rawk64_xeta256 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4
run q54_n14_E3_R2_xL2_rawk64_xeta256_off50 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 64 --polt 4 --trial-offset 50
run q54_n14_E3_R1_xL3_rawk96_xeta256 --n 14 --explore 3 --sab 98 $X256R1 --xL 3 --rawk 96 --polt 5

echo "[$(date +%H:%M:%S)] === queue54 cheap ladder confirms ==="
run q54_n08_E1_xL2_xs25_rawk48_off150 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 150
run q54_n10_E1_xL2_xs25_rawk48_off150 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 150
run q54_n12_E1_xL2_xs25_rawk48_off50 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 50
run q54_n14_E4_xL2_rawk48_xeta128_off100 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 100
run q54_n12_E1_xL2_rawk48_xeta256_off50 --n 12 --explore 1 --sab 72 $X256R1 --xL 2 --rawk 48 --polt 3 --trial-offset 50

echo "[$(date +%H:%M:%S)] === queue54 n18/n20 follow-ups ==="
run q54_n18_E16_R2_rawk72_xeta256_t2 --n 18 --explore 16 --sab 162 $X256R2 --xL 3 --xs 50 --rawk 72 --polt 4 --trials 2
run q54_n18_E24_R1_rawk80_xeta256_t2 --n 18 --explore 24 --sab 162 $X256R1 --xL 3 --xs 50 --rawk 80 --polt 4 --trials 2
run q54_n20_E20_R2_rawk80_xeta256_t2 --n 20 --explore 20 --sab 200 $X256R2 --xL 3 --xs 50 --rawk 80 --polt 4 --trials 2
run q54_n20_E32_R1_rawk96_xeta256_t2 --n 20 --explore 32 --sab 200 $X256R1 --xL 3 --xs 50 --rawk 96 --polt 5 --trials 2

touch logs/improve/queue54.done
echo "[$(date +%H:%M:%S)] queue54 ALL DONE"
