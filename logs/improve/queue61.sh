#!/bin/bash
# After queue60: more n16 under-3158 mean>=0.9 (E13/E12/E14 R1–R2 rawk56/84/100, xeta128/256/512 mixes,
# large-saind>=192 xc3), n14 sub-1100 offs, cheap ladder, light n18/n20. Gray + wmaxsat only.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue60.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1); X512R2=$(BX 512 2)

echo "[$(date +%H:%M:%S)] === queue61 n16 under-3158 mean>=0.9 ==="
run q61_n16_E13_R1_rawk56_polt4_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 56 --polt 4
run q61_n16_E13_R1_rawk84_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 84 --polt 5
run q61_n16_E13_R1_rawk100_polt5_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 100 --polt 5
run q61_n16_E12_R2_rawk56_polt4_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 56 --polt 4
run q61_n16_E12_R2_rawk84_polt5_xs35_xeta128_off150 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 84 --polt 5 --trial-offset 150
run q61_n16_E14_R1_rawk80_polt5_xs35_xeta128 --n 16 --explore 14 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 80 --polt 5
run q61_n16_E14_R1_rawk96_polt5_xs35_xeta256 --n 16 --explore 14 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 96 --polt 5
run q61_n16_E11_R2_rawk80_polt5_xs35_xeta128 --n 16 --explore 11 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q61_n16_E10_R2_rawk96_polt5_xs35_xeta256 --n 16 --explore 10 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 96 --polt 5
run q61_n16_E13_R2_rawk80_polt5_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 80 --polt 5
run q61_n16_E12_R1_rawk84_polt5_xs50_xeta256 --n 16 --explore 12 --sab 128 $X256R1 --xL 3 --xs 50 --rawk 84 --polt 5
run q61_n16_E13_R1_xc3_si256_rawk84_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 84 --polt 5 --xconsec 3 --saind 256
run q61_n16_E12_R2_xc3_si192_rawk100_xs35_xeta128 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 100 --polt 5 --xconsec 3 --saind 192
run q61_n16_E14_R1_xc3_si192_rawk80_xs35_xeta128 --n 16 --explore 14 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 80 --polt 5 --xconsec 3 --saind 192

echo "[$(date +%H:%M:%S)] === queue61 n14 sub-1100 ==="
run q61_n14_E3_R1_xL2_rawk84_xeta256_off500 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 84 --polt 5 --trial-offset 500
run q61_n14_E3_R1_xL2_rawk100_xeta256_off400 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 100 --polt 5 --trial-offset 400
run q61_n14_E3_R2_xL2_rawk64_xeta256_off450 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 64 --polt 4 --trial-offset 450
run q61_n14_E2_R1_xL2_rawk96_xeta256_off400 --n 14 --explore 2 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5 --trial-offset 400
run q61_n14_E4_R1_xL2_rawk64_xeta256_off400 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 400
run q61_n14_E3_R1_xL2_rawk84_xeta512_off350 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 84 --polt 5 --trial-offset 350

echo "[$(date +%H:%M:%S)] === queue61 cheap ladder ==="
run q61_n08_E1_xL2_xs25_rawk48_off600 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 600
run q61_n10_E1_xL2_xs25_rawk48_off600 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 600
run q61_n12_E1_xL2_xs25_rawk48_off500 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 500
run q61_n14_E4_xL2_rawk48_xeta128_off550 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 550

echo "[$(date +%H:%M:%S)] === queue61 n18/n20 light ==="
run q61_n18_E34_R1_rawk88_xeta256_t2 --n 18 --explore 34 --sab 162 $X256R1 --xL 3 --xs 50 --rawk 88 --polt 5 --trials 2
run q61_n20_E42_R1_rawk88_xeta256_t2 --n 20 --explore 42 --sab 200 $X256R1 --xL 3 --xs 50 --rawk 88 --polt 5 --trials 2

touch logs/improve/queue61.done
echo "[$(date +%H:%M:%S)] queue61 ALL DONE"
