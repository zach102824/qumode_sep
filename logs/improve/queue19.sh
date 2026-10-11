#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue18.done ]; do sleep 30; done
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
B2="${B/--xeta 16/}"   # base without xeta (set per run)
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
# replicate cheap candidates with 50 NEW trials (offset 5) -> 100-trial totals
run n14_E7_sab98_rawk24_xs35_R1_rep50 --n 14 --explore 7 --sab 98 $B --xs 35 --trial-offset 5
run n14_E6_sab98_rawk24_xs35_R1_rep50 --n 14 --explore 6 --sab 98 $B --xs 35 --trial-offset 5
run n12_E3_sab72_rawk24_xL2_R1_rep50 --n 12 --explore 3 --sab 72 $B --xL 2 --trial-offset 5
run n12_E2_sab72_rawk24_xL2_R1_rep50 --n 12 --explore 2 --sab 72 $B --xL 2 --trial-offset 5
run n12_E2_sab72_rawk24_xL3_xs35_R1_rep50 --n 12 --explore 2 --sab 72 $B --xL 3 --xs 35 --trial-offset 5
# explore-quality knobs at n14 (E6 xL3 R1 base: 0.94 / 0.928 over 100 trials at 2119)
run n14_E6_sab98_rawk24_xL3_R1_xeta8_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 8 --xL 3
run n14_E6_sab98_rawk24_xL3_R1_xeta32_c50 --n 14 --explore 6 --sab 98 $B2 --xeta 32 --xL 3
run n14_E6_sab98_rawk24_xL3_R1_xb15_c50 --n 14 --explore 6 --sab 98 $B --xL 3 --xbeta 1.5
run n14_E6_sab98_rawk24_xL3_R1_xb5_c50 --n 14 --explore 6 --sab 98 $B --xL 3 --xbeta 5
run n14_E6_sab98_rawk24_xL3_R1_xk10_c50 --n 14 --explore 6 --sab 98 $B --xL 3 --xkick 0.1
run n14_E6_sab98_rawk24_xL3_R1_xk02_c50 --n 14 --explore 6 --sab 98 $B --xL 3 --xkick 0.02
run n14_E6_sab98_rawk24_xL3_R1_reta4_c50 --n 14 --explore 6 --sab 98 ${B/--reta 2/--reta 4} --xL 3
# fewer, longer explore runs
run n14_E3_sab98_rawk24_xL3_xs100_R1_c50 --n 14 --explore 3 --sab 98 $B --xL 3 --xs 100
run n14_E4_sab98_rawk24_xL3_xs75_R1_c50 --n 14 --explore 4 --sab 98 $B --xL 3 --xs 75
# n12 mirrors
run n12_E2_sab72_rawk24_xL2_xs35_R1_c50 --n 12 --explore 2 --sab 72 $B --xL 2 --xs 35
run n12_E3_sab72_rawk24_xL2_xs35_R1_c50 --n 12 --explore 3 --sab 72 $B --xL 2 --xs 35
run n12_E2_sab72_rawk24_xL2_R1_nz40_c50 --n 12 --explore 2 --sab 72 $B --xL 2 --sanoise 0.4
run n12_E2_sab72_rawk24_xL3_R1_xeta8_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 8 --xL 3
run n12_E2_sab72_rawk24_xL3_R1_xeta32_c50 --n 12 --explore 2 --sab 72 $B2 --xeta 32 --xL 3
run n12_E2_sab72_rawk24_xL3_R1_xb5_c50 --n 12 --explore 2 --sab 72 $B --xL 3 --xbeta 5
run n12_E2_sab72_rawk24_xL3_R1_xk10_c50 --n 12 --explore 2 --sab 72 $B --xL 3 --xkick 0.1
echo DONE > logs/improve/queue19.done
