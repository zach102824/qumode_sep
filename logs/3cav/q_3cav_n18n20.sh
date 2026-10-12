#!/bin/bash
# 3-cavity n=18 (5,5,5) then n=20 (6,6,5): screen E=4,6,8 (10 trials), then 50 trials at cheapest E with both >= 0.9.
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
STOPF=logs/3cav/q_3cav_n18n20.STOP
while [ ! -f logs/3cav/q_3cav_confirm.done ]; do [ -f $STOPF ] && exit 0; sleep 30; done
B="--device 3cav --L 4 --s 50 --r 150 --R 2 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 7 --explore-mask sa --samode fixed_low --rawk 48 --polt 3 --xL 3 --xs 35"
for spec in "18 5,5,5 162" "20 6,6,5 200"; do
  set -- $spec; N=$1; KS=$2; SAB=$3
  BEST=""
  for EE in 4 6 8; do
    [ -f $STOPF ] && exit 0
    echo "[$(date +%T)] START n$N E$EE screen"
    python noiseless/run_relayout_kbit.py $B --n $N --ks $KS --sab $SAB --trials 1 --explore $EE > logs/3cav/screen3cav_n${N}_E$EE.json 2>&1
    echo "[$(date +%T)] DONE n$N E$EE"
    if [ -z "$BEST" ] && python - logs/3cav/screen3cav_n${N}_E$EE.json <<'PY'
import json,sys
t=open(sys.argv[1]).read(); d=json.loads(t[t.index("{"):])
sys.exit(0 if d["success"]>=0.9 and d["mean_p_gs"]>=0.9 else 1)
PY
    then BEST=$EE; fi
  done
  if [ -n "$BEST" ]; then
    [ -f $STOPF ] && exit 0
    echo "[$(date +%T)] START n$N E$BEST confirm 50"
    python noiseless/run_relayout_kbit.py $B --n $N --ks $KS --sab $SAB --trials 5 --explore $BEST > logs/3cav/confirm3cav_n${N}_E$BEST.json 2>&1
    echo "[$(date +%T)] DONE n$N confirm"
  else echo "n$N: no screened E reached 0.9/0.9; no confirm"; fi
done
touch logs/3cav/q_3cav_n18n20.done
