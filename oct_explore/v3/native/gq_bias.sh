#!/bin/bash
cd /workspace/qumode_sep/oct_explore; tag=$1; shift; mkdir -p logs/v3/native/gnnw
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec taskset -c ${CORES:-0-6} /usr/bin/python3.13 v3/native/gnn_bias.py --tag $tag "$@" > logs/v3/native/gnnw/$tag.log 2>&1
