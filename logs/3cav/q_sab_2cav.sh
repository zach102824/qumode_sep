#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
STOPF=logs/3cav/q_sab_2cav.STOP
C="--n 16 --L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 4 --explore-mask sa --samode fixed_low --rawk 48 --polt 3 --trials 5 --explore 12 --xL 3 --xs 35"
for S in 128 256 512; do
  [ -f $STOPF ] && { echo stopped; exit 0; }
  echo "[$(date +%T)] START sab$S"
  python noiseless/run_relayout_kbit.py $C --sab $S > logs/3cav/sab2cav_$S.json 2>&1
  echo "[$(date +%T)] DONE sab$S"
done
touch logs/3cav/q_sab_2cav.done
