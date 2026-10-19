#!/bin/bash
cd /workspace/qumode_sep/oct_explore
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for r in F16_gray_chain F16_qaoa; do for k in 16 20 12; do
  taskset -c 0-6 /usr/bin/python3.13 v3/native/evalnative.py results/v3/native/fixed/$r.json --k 4:$k --tests 8 12 16 > logs/v3/native/evn_${r}_K4k$k.log 2>&1
done; done
