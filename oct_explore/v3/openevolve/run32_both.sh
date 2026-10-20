#!/bin/bash
cd /workspace/qumode_sep/oct_explore/v3/openevolve
for k in checkpoint_10 checkpoint_20; do
  MEMMB=3500 NEED=6700 ./run_paused.sh runs/net32o/check/${k}_chunked.json OE_TRAIN_N=10,15 OE_VAL_N=15 OE_CHECK_N=20 OE_NVAL=30 OE_BUDGET=150 taskset -c 0-7 python3.13 net/chunked_check.py runs/net32o/checkpoints/$k/best_program.py
done
