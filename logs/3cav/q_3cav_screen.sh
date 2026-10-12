#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
STOPF=logs/3cav/q_3cav_screen.STOP
C="--device 3cav --ks 5,4,4 --n 16 --L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 3 --explore-mask sa --samode fixed_low --rawk 48 --polt 3 --xL 3 --xs 35 --sab 128 --trials 1"
for EE in 4 8 12; do
  [ -f $STOPF ] && { echo stopped; exit 0; }
  echo "[$(date +%T)] START E$EE"
  python noiseless/run_relayout_kbit.py $C --explore $EE > logs/3cav/screen3cav_E$EE.json 2>&1
  echo "[$(date +%T)] DONE E$EE"
done
touch logs/3cav/q_3cav_screen.done
