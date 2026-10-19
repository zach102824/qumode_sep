#!/bin/bash
# usage: jq.sh <tag> <args...>  -> one detached single-thread fixed.py job
cd /workspace/qumode_sep/oct_explore; tag=$1; shift
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec taskset -c 0-6 /usr/bin/python3.13 v3/native/fixed.py --tag $tag "$@" > logs/v3/native/fixed/$tag.log 2>&1
