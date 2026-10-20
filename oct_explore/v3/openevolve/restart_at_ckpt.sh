#!/bin/bash
# restart net16o at checkpoint_50 and net32o at checkpoint_10 with Opus-only configs (keeps Fable programs in DB)
cd /workspace/qumode_sep/oct_explore/v3/openevolve
common='export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"; cd /workspace/qumode_sep/oct_explore/v3/openevolve'
d16=0; d32=0
while [ $d16 = 0 ] || [ $d32 = 0 ]; do
  if [ $d16 = 0 ] && [ -f runs/net16o/checkpoints/checkpoint_50/metadata.json ]; then sleep 20
    pkill -f "openevolve.cli runs/net16o"; sleep 5
    setsid nohup bash -c "$common; export OE_TRAIN_N=8,12 OE_VAL_N=16 OE_CHECK_N=16 OE_NVAL=60 OE_BUDGET=180 OE_MEM_MB=3000; taskset -c 4-7 python3.13 -m openevolve.cli runs/net16o/seed_program.py net/evaluator_net.py --config config_net16c.yaml --iterations 200 --output runs/net16o --checkpoint runs/net16o/checkpoints/checkpoint_50" > logs/net16o_opus2.log 2>&1 &
    echo "$(date +%T) net16o restarted from ckpt_50 (Opus-only)"; d16=1; fi
  if [ $d32 = 0 ] && [ -f runs/net32o/checkpoints/checkpoint_10/metadata.json ]; then sleep 20
    pkill -f "openevolve.cli runs/net32o"; sleep 5
    setsid nohup bash -c "$common; export OE_TRAIN_N=10,15 OE_VAL_N=15 OE_CHECK_N=20 OE_NVAL=30 OE_BUDGET=150 OE_MEM_MB=3500; taskset -c 0-3 python3.13 -m openevolve.cli runs/net32o/seed_program.py net/evaluator_net.py --config config_net32.yaml --iterations 200 --output runs/net32o --checkpoint runs/net32o/checkpoints/checkpoint_10" > logs/net32o_opus2.log 2>&1 &
    echo "$(date +%T) net32o restarted from ckpt_10 (Opus-only)"; d32=1; fi
  sleep 30
done
