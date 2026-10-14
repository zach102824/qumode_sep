#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 QM_SA_COUNT=raw QM_FIXED_COST=1 QM_CLS_ALGO=satlike
O=noiseless/results/ecd_speed
py-spy record --subprocesses -d 240 -r 50 -f raw -o $O/prof_raw.txt -- python noiseless/run_relayout_kbit.py --L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 1 --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --sanoise 0.3 --device 3cav --r 20 --xL 3 --xs 5 --n 20 --ks 6,6,5 --explore 1 --sab 60 --rawk 40 --outdir $O/prof_run > $O/prof_run.out 2>&1
echo done >> $O/prof_run.out
