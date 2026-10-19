#!/bin/bash
# usage: sdj.sh <threads> <tag> [args...]  -> detached-friendly single sd_n6 job, log logs/v3/sd_n6/<tag>.log
cd /workspace/qumode_sep/oct_explore; th=$1; tag=$2; shift 2
exec taskset -c 0-7 env XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=$th XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$th" \
  /usr/bin/python3.13 v3/sd_n6.py --tag $tag "$@" > logs/v3/sd_n6/$tag.log 2>&1
