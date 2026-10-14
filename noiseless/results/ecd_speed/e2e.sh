#!/bin/bash
# one full n=20 sl3 trial (instance 0, seed default), old vs --ecd-fast, run side by side, 1 thread each
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QM_SA_COUNT=raw QM_FIXED_COST=1 QM_CLS_ALGO=satlike
O=noiseless/results/ecd_speed
ARGS="--L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 1 --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --sanoise 0.3 --device 3cav --r 150 --xL 3 --xs 35 --n 20 --ks 6,6,5 --explore 8 --sab 60 --rawk 40 --max-h 1"
( s=$(date +%s); python noiseless/run_relayout_kbit.py $ARGS --outdir $O/e2e_old > $O/e2e_old.out 2>&1; echo "WALL $(( $(date +%s)-s ))" >> $O/e2e_old.out ) &
( s=$(date +%s); python noiseless/run_relayout_kbit.py $ARGS --ecd-fast --outdir $O/e2e_new > $O/e2e_new.out 2>&1; echo "WALL $(( $(date +%s)-s ))" >> $O/e2e_new.out ) &
wait
