#!/bin/bash
cd /workspace/qumode_sep/oct_explore
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
taskset -c 0-6 /usr/bin/python3.13 v3/native/mixcal.py gray 5 $1 --NF 80 --nb 9 --starts 2 --sweeps 1 --maxiter 3000 --out results/v3/native/mixcal_nf80 > logs/v3/native/mixcal80_gray_K5_k$1.log 2>&1
