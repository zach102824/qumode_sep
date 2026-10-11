#!/bin/bash
# After queue61: denser n16 under-3158 mean>=0.9 (E13/E12/E11 R1–R2, rawk68/76/92/108 midpoints,
# xeta128/256/512, large-saind>=192 xc3 only), n14 sub-1100 offs on 907 family (E3 xL2 xeta256 rawk64/80/96),
# cheap ladder off* confirms, light n18/n20 screens. Gray + wmaxsat only.
# Avoid xwarm/lmut/thr/xstop/xagree/xc3-tiny-saind.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue61.done ]; do sleep 45; done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --trials 5"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2); X512R1=$(BX 512 1); X512R2=$(BX 512 2)

echo "[$(date +%H:%M:%S)] === queue62 n16 under-3158 mean>=0.9 ==="
# E13 R1 midpoints not in q60/61 (68/76/92/108)
run q62_n16_E13_R1_rawk68_polt4_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 68 --polt 4
run q62_n16_E13_R1_rawk76_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 76 --polt 5
run q62_n16_E13_R1_rawk92_polt5_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 92 --polt 5
run q62_n16_E13_R1_rawk108_polt5_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 108 --polt 5
run q62_n16_E13_R1_rawk68_polt4_xs35_xeta512 --n 16 --explore 13 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 68 --polt 4
run q62_n16_E13_R1_rawk92_polt5_xs35_xeta512 --n 16 --explore 13 --sab 128 $X512R1 --xL 3 --xs 35 --rawk 92 --polt 5
# E12 R2 midpoints + offs
run q62_n16_E12_R2_rawk68_polt4_xs35_xeta128 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 68 --polt 4
run q62_n16_E12_R2_rawk76_polt5_xs35_xeta256_off100 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 76 --polt 5 --trial-offset 100
run q62_n16_E12_R2_rawk92_polt5_xs35_xeta128_off200 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 92 --polt 5 --trial-offset 200
run q62_n16_E12_R1_rawk76_polt5_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 76 --polt 5
run q62_n16_E12_R1_rawk108_polt5_xs50_xeta128 --n 16 --explore 12 --sab 128 $X128R1 --xL 3 --xs 50 --rawk 108 --polt 5
# E11 R1–R2 midpoints
run q62_n16_E11_R1_rawk68_polt4_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 68 --polt 4
run q62_n16_E11_R2_rawk76_polt5_xs35_xeta128 --n 16 --explore 11 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 76 --polt 5
run q62_n16_E11_R2_rawk92_polt5_xs35_xeta512_off100 --n 16 --explore 11 --sab 128 $X512R2 --xL 3 --xs 35 --rawk 92 --polt 5 --trial-offset 100
run q62_n16_E11_R1_rawk108_polt5_xs50_xeta256 --n 16 --explore 11 --sab 128 $X256R1 --xL 3 --xs 50 --rawk 108 --polt 5
# E14 R1 under-cost probes
run q62_n16_E14_R1_rawk68_polt4_xs35_xeta128 --n 16 --explore 14 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 68 --polt 4
run q62_n16_E14_R1_rawk92_polt5_xs35_xeta256 --n 16 --explore 14 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 92 --polt 5
# large-saind>=192 xc3 only
run q62_n16_E13_R1_xc3_si192_rawk76_xs35_xeta256 --n 16 --explore 13 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 76 --polt 5 --xconsec 3 --saind 192
run q62_n16_E12_R2_xc3_si256_rawk92_xs35_xeta128 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 92 --polt 5 --xconsec 3 --saind 256
run q62_n16_E11_R2_xc3_si192_rawk108_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 108 --polt 5 --xconsec 3 --saind 192
run q62_n16_E13_R2_xc3_si256_rawk68_xs35_xeta128 --n 16 --explore 13 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 68 --polt 4 --xconsec 3 --saind 256

echo "[$(date +%H:%M:%S)] === queue62 n14 sub-1100 907-family offs ==="
run q62_n14_E3_R1_xL2_rawk64_xeta256_off550 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 64 --polt 4 --trial-offset 550
run q62_n14_E3_R1_xL2_rawk80_xeta256_off550 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 550
run q62_n14_E3_R1_xL2_rawk96_xeta256_off550 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 96 --polt 5 --trial-offset 550
run q62_n14_E3_R1_xL2_rawk68_xeta256_off600 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 68 --polt 4 --trial-offset 600
run q62_n14_E3_R1_xL2_rawk76_xeta256_off600 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 76 --polt 5 --trial-offset 600
run q62_n14_E3_R1_xL2_rawk92_xeta256_off600 --n 14 --explore 3 --sab 98 $X256R1 --xL 2 --rawk 92 --polt 5 --trial-offset 600
run q62_n14_E3_R1_xL2_rawk80_xeta512_off500 --n 14 --explore 3 --sab 98 $X512R1 --xL 2 --rawk 80 --polt 5 --trial-offset 500
run q62_n14_E3_R2_xL2_rawk80_xeta256_off550 --n 14 --explore 3 --sab 98 $X256R2 --xL 2 --rawk 80 --polt 5 --trial-offset 550
run q62_n14_E2_R1_xL2_rawk80_xeta256_off550 --n 14 --explore 2 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 550
run q62_n14_E4_R1_xL2_rawk80_xeta256_off550 --n 14 --explore 4 --sab 98 $X256R1 --xL 2 --rawk 80 --polt 5 --trial-offset 550

echo "[$(date +%H:%M:%S)] === queue62 cheap ladder confirms ==="
run q62_n08_E1_xL2_xs25_rawk48_off700 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 700
run q62_n10_E1_xL2_xs25_rawk48_off700 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 700
run q62_n12_E1_xL2_xs25_rawk48_off600 --n 12 --explore 1 --sab 72 $X128R1 --xL 2 --xs 25 --rawk 48 --polt 3 --trial-offset 600
run q62_n14_E4_xL2_rawk48_xeta128_off650 --n 14 --explore 4 --sab 98 $X128R1 --xL 2 --rawk 48 --polt 3 --trial-offset 650
run q62_n08_E1_xL2_xs35_rawk36_off450 --n 8 --explore 1 --sab 32 $X128R1 --xL 2 --xs 35 --rawk 36 --polt 3 --trial-offset 450
run q62_n10_E1_xL2_xs35_rawk36_off450 --n 10 --explore 1 --sab 50 $X128R1 --xL 2 --xs 35 --rawk 36 --polt 3 --trial-offset 450

echo "[$(date +%H:%M:%S)] === queue62 n18/n20 light screens ==="
run q62_n18_E30_R1_rawk92_xeta256_t2 --n 18 --explore 30 --sab 162 $X256R1 --xL 3 --xs 50 --rawk 92 --polt 5 --trials 2
run q62_n18_E38_R2_rawk76_xeta512_t2 --n 18 --explore 38 --sab 162 $X512R2 --xL 3 --xs 50 --rawk 76 --polt 5 --trials 2
run q62_n20_E38_R1_rawk92_xeta256_t2 --n 20 --explore 38 --sab 200 $X256R1 --xL 3 --xs 50 --rawk 92 --polt 5 --trials 2
run q62_n20_E46_R2_rawk76_xeta512_t2 --n 20 --explore 46 --sab 200 $X512R2 --xL 3 --xs 50 --rawk 76 --polt 5 --trials 2

touch logs/improve/queue62.done
echo "[$(date +%H:%M:%S)] queue62 ALL DONE"
