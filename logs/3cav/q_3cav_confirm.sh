#!/bin/bash
# 50 trials (5 per instance) at the cheapest screened E clearing 0.9/0.9 (E4 at screen time), after the screen finishes.
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
STOPF=logs/3cav/q_3cav_confirm.STOP
while [ ! -f logs/3cav/q_3cav_screen.done ]; do [ -f $STOPF ] && exit 0; sleep 30; done
W=3; [ -f logs/3cav/q_sab_2cav.done ] && W=7
[ -f $STOPF ] && exit 0
echo "[$(date +%T)] START E4 50 trials workers=$W"
python noiseless/run_relayout_kbit.py --device 3cav --ks 5,4,4 --n 16 --L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers $W --explore-mask sa --samode fixed_low --rawk 48 --polt 3 --xL 3 --xs 35 --sab 128 --trials 5 --explore 4 > logs/3cav/confirm3cav_E4.json 2>&1
echo "[$(date +%T)] DONE"; touch logs/3cav/q_3cav_confirm.done
