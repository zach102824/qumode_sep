#!/bin/bash
# priority n=8/n=10 fill: briefly SIGSTOP the running n=16 job (keeps 8 cores, no oversubscription), then SIGCONT
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
PAT="python noiseless/run_relayout_kbit.py --n 16"
cleanup(){ pkill -CONT -f "$PAT"; }
trap cleanup EXIT INT TERM
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
B32="${B/--xeta 16/--xeta 32}"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
pkill -STOP -f "$PAT"
run n8_E1_sab32_rawk24_xL2_R1_c50 --n 8 --explore 1 --sab 32 $B --xL 2
run n8_E2_sab32_rawk24_xL2_R1_c50 --n 8 --explore 2 --sab 32 $B --xL 2
run n8_E1_sab32_rawk24_xL2_xs35_R1_c50 --n 8 --explore 1 --sab 32 $B --xL 2 --xs 35
run n8_E1_sab32_rawk24_xL3_R1_c50 --n 8 --explore 1 --sab 32 $B --xL 3
run n8_E1_sab32_rawk24_xL2_R1_xeta32_c50 --n 8 --explore 1 --sab 32 $B32 --xL 2
run n8_E3_sab32_rawk24_xL2_R1_c50 --n 8 --explore 3 --sab 32 $B --xL 2
run n10_E1_sab50_rawk24_xL2_R1_c50 --n 10 --explore 1 --sab 50 $B --xL 2
run n10_E2_sab50_rawk24_xL2_R1_c50 --n 10 --explore 2 --sab 50 $B --xL 2
run n10_E2_sab50_rawk24_xL2_R1_xeta32_c50 --n 10 --explore 2 --sab 50 $B32 --xL 2
run n10_E3_sab50_rawk24_xL2_R1_c50 --n 10 --explore 3 --sab 50 $B --xL 2
run n10_E2_sab50_rawk24_xL2_xs35_R1_c50 --n 10 --explore 2 --sab 50 $B --xL 2 --xs 35
run n10_E4_sab50_rawk24_xL2_R1_c50 --n 10 --explore 4 --sab 50 $B --xL 2
cleanup
touch logs/improve/queue24.done
