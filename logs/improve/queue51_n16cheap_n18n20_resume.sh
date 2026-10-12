#!/bin/bash
# Resume after interruption: finish n18 E20+ and remaining arms, then n20, then done flag.
# Gray + wmaxsat. Detached via setsid/nohup.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1

run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
BX(){ echo "--L 4 --s 50 --r 150 --R $2 --lr 0.05 --xeta $1 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low"; }

echo "[$(date +%H:%M:%S)] === RESUME n18 from E20 (jsonl skips finished inst) ==="
n18=$(ls Hamiltonians/wmaxsat/default_n18_*.json 2>/dev/null | wc -l)
echo "n18 hams: $n18"
N18="--n 18 --sab 162 --L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --xL 3 --xs 50 --rawk 54 --polt 3"
for E in 20 24; do
  run q51_n18_E${E}_t1 $N18 --explore $E --trials 1
done
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
echo "[$(date +%H:%M:%S)] queue51_n16cheap RESUME ALL DONE"
