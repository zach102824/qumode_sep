#!/bin/bash
# After queue50: hunt n16 >=0.9 under ~2500 evals (ideas 2+3, R2, readout), then screen n18 and n20.
# Gray + wmaxsat. Detached via setsid/nohup — never stopped by 50min chat limit.
# NOTE: --trials T means T trials PER instance (10 instances => 10T total jobs).
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
while [ ! -f logs/improve/queue50.done ]; do sleep 30; done
while pgrep -f 'gen_n18_n20.py' >/dev/null 2>&1; do sleep 20; done
WAIT=0
while pgrep -af 'logs/improve/queue45\.sh' >/dev/null 2>&1 && [ $WAIT -lt 120 ]; do sleep 60; WAIT=$((WAIT+1)); done

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"; }
X128R1=$(BX 128 1); X128R2=$(BX 128 2); X256R1=$(BX 256 1); X256R2=$(BX 256 2)

echo "[$(date +%H:%M:%S)] === n16 under-2500 hunt (50 trials = --trials 5) ==="
run q51_n16_E12_xc3_si32_xs35_rawk48 --n 16 --explore 12 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 48 --polt 3 --trials 5 --xconsec 3 --saind 32
run q51_n16_E14_xc3_si32_xs35_rawk48 --n 16 --explore 14 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 48 --polt 3 --trials 5 --xconsec 3 --saind 32
run q51_n16_E12_xc3_si48_xs35_rawk64 --n 16 --explore 12 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 64 --polt 4 --trials 5 --xconsec 3 --saind 48
run q51_n16_E10_xc2_si64_xs35_rawk64 --n 16 --explore 10 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 64 --polt 4 --trials 5 --xconsec 2 --saind 64
run q51_n16_E12_R2_rawk48_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 48 --polt 3 --trials 5
run q51_n16_E11_R2_rawk64_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R2 --xL 3 --xs 35 --rawk 64 --polt 4 --trials 5
run q51_n16_E13_R2_rawk64_xs35 --n 16 --explore 13 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 64 --polt 4 --trials 5
run q51_n16_E12_R2_rawk96_xs35 --n 16 --explore 12 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 96 --polt 5 --trials 5
run q51_n16_E12_R1_rawk64_xs35_xeta256 --n 16 --explore 12 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 64 --polt 4 --trials 5
run q51_n16_E11_R1_rawk96_xs35_xeta256 --n 16 --explore 11 --sab 128 $X256R1 --xL 3 --xs 35 --rawk 96 --polt 5 --trials 5
run q51_n16_E10_R1_rawk96_xs50_xeta256 --n 16 --explore 10 --sab 128 $X256R1 --xL 3 --xs 50 --rawk 96 --polt 5 --trials 5
run q51_n16_E8_si128_xs35_rawk64_R2 --n 16 --explore 8 --sab 128 $X128R2 --xL 3 --xs 35 --rawk 64 --polt 4 --trials 5 --saind 128
run q51_n16_E10_si64_xs35_rawk64_R1 --n 16 --explore 10 --sab 128 $X128R1 --xL 3 --xs 35 --rawk 64 --polt 4 --trials 5 --saind 64

echo "[$(date +%H:%M:%S)] === n18 screens (10 then 20 jobs = --trials 1 then 2) ==="
n18=$(ls Hamiltonians/wmaxsat/default_n18_*.json 2>/dev/null | wc -l)
echo "n18 hams: $n18"
N18="--n 18 --sab 162 --L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --xL 3 --xs 50 --rawk 54 --polt 3"
for E in 4 8 12 16 20 24; do
  run q51_n18_E${E}_t1 $N18 --explore $E --trials 1
done
# 20-trial confirms on mid/high E
for E in 12 16 20 24; do
  run q51_n18_E${E}_t2 $N18 --explore $E --trials 2
done
run q51_n18_E16_R2_t1 --n 18 --explore 16 --sab 162 $(BX 128 2) --xL 3 --xs 50 --rawk 54 --polt 3 --trials 1
run q51_n18_E20_R2_t1 --n 18 --explore 20 --sab 162 $(BX 128 2) --xL 3 --xs 50 --rawk 54 --polt 3 --trials 1
run q51_n18_E16_R2_t2 --n 18 --explore 16 --sab 162 $(BX 128 2) --xL 3 --xs 50 --rawk 54 --polt 3 --trials 2
run q51_n18_E20_R2_t2 --n 18 --explore 20 --sab 162 $(BX 128 2) --xL 3 --xs 50 --rawk 54 --polt 3 --trials 2

echo "[$(date +%H:%M:%S)] === n20 screens ==="
WAIT=0
while [ $(ls Hamiltonians/wmaxsat/default_n20_*.json 2>/dev/null | wc -l) -lt 10 ] && [ $WAIT -lt 40 ]; do
  echo "waiting n20 hams ($(ls Hamiltonians/wmaxsat/default_n20_*.json 2>/dev/null | wc -l)/10)"; sleep 60; WAIT=$((WAIT+1))
done
echo "n20 hams: $(ls Hamiltonians/wmaxsat/default_n20_*.json 2>/dev/null | wc -l)"
N20="--n 20 --sab 200 --L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --xL 3 --xs 50 --rawk 60 --polt 3"
for E in 8 12 16 20 24 32; do
  run q51_n20_E${E}_t1 $N20 --explore $E --trials 1
done
for E in 16 20 24 32; do
  run q51_n20_E${E}_t2 $N20 --explore $E --trials 2
done

touch logs/improve/queue51_n16cheap_n18n20.done
echo "[$(date +%H:%M:%S)] queue51 ALL DONE"
