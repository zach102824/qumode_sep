#!/bin/bash
# After queue58: denser n16 under-3158 mean>=0.9 (R2 E9–E14, rawk72–120 midpoints, xeta256/512, xs50 mixes,
# large-saind>=128 xc3 only), n14 sub-1100 near-miss family (E2/E3/E4 xL2 xeta256/512 rawk* off*),
# cheap ladder off* confirms, light n18/n20 screens. Gray + wmaxsat only.
# Avoid xwarm/lmut/thr/xstop/xagree/xc3-tiny-saind.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue58.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1); X512R2=$(BX 512 2)

echo "[$(date +%H:%M:%S)] === queue59 n16 under-3158 mean>=0.9 ==="
# complements to queue52-58: midpoint rawk 72/88/104, xs50 mixes, E9/E14 fills, colder+R2, large-saind xc3 only
run q59_n16_E9_R2_rawk96_polt5_xs35_xeta256 --n 16 --explore 9 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 96 --polt 5
run q59_n16_E9_R2_rawk112_polt5_xs35_xeta512 --n 16 --explore 9 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 112 --polt 5
run q59_n16_E10_R2_rawk72_polt4_xs35_xeta256 --n 16 --explore 10 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 72 --polt 4
run q59_n16_E10_R2_rawk88_polt5_xs50_xeta256 --n 16 --explore 10 --sab 128 $X256R2 --xL 3 --xs 50 --rawk 88 --polt 5
run q59_n16_E11_R2_rawk72_polt4_xs50_xeta256 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 50 --rawk 72 --polt 4
run q59_n16_E11_R2_rawk104_polt5_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 104 --polt 5
run q59_n16_E12_R2_rawk72_polt4_xs35_xeta512 --n 16 --explore 12 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 72 --polt 4
run q59_n16_E12_R2_rawk88_polt5_xs50_xeta512 --n 16 --explore 12 --sab 128 $X512R2 --xL 3 --xs 50 --rawk 88 --polt 5
run q59_n16_E13_R2_rawk72_polt4_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 72 --polt 4
run q59_n16_E13_R2_rawk104_polt5_xs50_xeta256 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 50 --rawk 104 --polt 5
run q59_n16_E14_R2_rawk72_polt4_xs35_xeta256 --n 16 --explore 14 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 72 --polt 4
run q59_n16_E14_R1_rawk104_polt5_xs35_xeta512 --n 16 --explore 14 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 104 --polt 5
run q59_n16_E10_R2_rawk104_polt5_xs35_xeta512_off150 --n 16 --explore 10 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 104 --polt 5 --trial-offset 150
run q59_n16_E11_R2_rawk88_polt5_xs35_xeta256_off150 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 88 --polt 5 --trial-offset 150
run q59_n16_E12_R2_rawk104_polt5_xs35_xeta256_off150 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 104 --polt 5 --trial-offset 150
run q59_n16_E13_R1_rawk88_polt5_xs35_xeta512_off100 --n 16 --explore 13 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 88 --polt 5 --trial-offset 100
# careful early-stop only with large saind (>=128)
run q59_n16_E12_R2_xc3_si192_rawk88_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 88 --polt 5 --xconsec 3 --saind 192
run q59_n16_E11_R2_xc3_si256_rawk104_xs50_xeta512 --n 16 --explore 11 --sab 128 $X512R2 --xL 3 --xs 50 --rawk 104 --polt 5 --xconsec 3 --saind 256
run q59_n16_E13_R2_xc3_si192_rawk72_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 72 --polt 4 --xconsec 3 --saind 192
run q59_n16_E10_R2_xc3_si256_rawk120_xs35_xeta512 --n 16 --explore 10 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 120 --polt 5 --xconsec 3 --saind 256

echo "[$(date +%H:%M:%S)] === queue59 n14 sub-1100 near-miss family ==="
# 907-eval E3 xL2 xeta256 rawk64 was 0.92/0.895; push E2/E3/E4 + mid rawk + colder + fresh offs
run q59_n14_E2_R1_xL2_rawk64_xeta256_off250 --n 14 --explore 2 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 250
run q59_n14_E2_R1_xL2_rawk96_xeta256_off250 --n 14 --explore 2 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5 --trial-offset 250
run q59_n14_E3_R1_xL2_rawk72_xeta256_off350 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 72 --polt 4 --trial-offset 350
run q59_n14_E3_R1_xL2_rawk88_xeta256_off350 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 88 --polt 5 --trial-offset 350
run q59_n14_E3_R1_xL2_rawk104_xeta256_off250 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 104 --polt 5 --trial-offset 250
run q59_n14_E3_R2_xL2_rawk72_xeta256_off250 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 72 --polt 4 --trial-offset 250
run q59_n14_E3_R1_xL2_rawk72_xeta512_off200 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 72 --polt 4 --trial-offset 200
run q59_n14_E3_R2_xL2_rawk96_xeta256_off350 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 96 --polt 5 --trial-offset 350
run q59_n14_E4_R1_xL2_rawk72_xeta256_off250 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 72 --polt 4 --trial-offset 250
run q59_n14_E4_R1_xL2_rawk88_xeta256_off200 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 88 --polt 5 --trial-offset 200

echo "[$(date +%H:%M:%S)] === queue59 cheap ladder confirms ==="
run q59_n08_E1_xL2_xs25_rawk48_off400 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 400
run q59_n10_E1_xL2_xs25_rawk48_off400 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 400
run q59_n12_E1_xL2_xs25_rawk48_off300 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 300
run q59_n14_E4_xL2_rawk48_xeta128_off350 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 350
run q59_n12_E1_xL2_rawk48_xeta256_off250 --n 12 --explore 1 --sab 72 $X256R1 --xL 2 --rawk 48 --polt 3 --trial-offset 250
run q59_n08_E1_xL2_xs35_rawk48_off250 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 35 --rawk 48 --polt 3 --trial-offset 250
run q59_n10_E1_xL2_xs35_rawk48_off250 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 35 --rawk 48 --polt 3 --trial-offset 250

echo "[$(date +%H:%M:%S)] === queue59 n18/n20 light screens ==="
run q59_n18_E28_R1_rawk104_xeta512_t2 --n 18 --explore 28 --sab 162 $X512R1 --xL 3 --xs 50 --rawk 104 --polt 5 --trials 2
run q59_n18_E36_R2_rawk88_xeta256_t2 --n 18 --explore 36 --sab 162 $X256R2 --xL 3 --xs 50 --rawk 88 --polt 5 --trials 2
run q59_n20_E36_R1_rawk104_xeta512_t2 --n 20 --explore 36 --sab 200 $X512R1 --xL 3 --xs 50 --rawk 104 --polt 5 --trials 2
run q59_n20_E44_R2_rawk88_xeta256_t2 --n 20 --explore 44 --sab 200 $X256R2 --xL 3 --xs 50 --rawk 88 --polt 5 --trials 2

touch logs/improve/queue59.done
echo "[$(date +%H:%M:%S)] queue59 ALL DONE"
