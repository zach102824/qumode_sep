#!/bin/bash
# sl3 recipe (run_variant.sh sl3 3, QM_CLS_ALGO=satlike) + --shots 1000, 6 workers, new outdir
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 QM_SA_COUNT=raw QM_FIXED_COST=1 QM_CLS_ALGO=satlike
OUT=noiseless/results/relayout_runs/shots1000; D=logs/shots1000; mkdir -p $OUT
CM="--L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 6 --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --sanoise 0.3 --outdir $OUT --shots 1000"
C2="$CM --r 150 --xL 2"; C3="$CM --device 3cav --r 150 --xL 3 --xs 35 --ecd-fast"
declare -A A
A[8]="$C2 --n 8 --xs 25 --explore 2"; A[10]="$C2 --n 10 --xs 35 --explore 2"; A[12]="$C2 --n 12 --xs 50 --explore 2"
A[14]="$C3 --n 14 --ks 4,4,3 --explore 2"; A[16]="$C3 --n 16 --ks 5,4,4 --explore 4"
A[18]="$C3 --n 18 --ks 5,5,5 --explore 6"; A[20]="$C3 --n 20 --ks 6,6,5 --explore 8"
for n in "$@"; do
  echo "[$(date '+%F %T') CST] START n$n"
  python noiseless/run_relayout_kbit.py ${A[$n]} --sab $((3*n)) --rawk $((2*n)) > $D/n$n.json 2>&1
  echo "[$(date '+%F %T') CST] DONE n$n"
done
