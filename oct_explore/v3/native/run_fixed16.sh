#!/bin/bash
cd /workspace/qumode_sep/oct_explore
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for spec in "F16_gray_chain --mixer gray --proj chain" "F16_qaoa --mixer gray --proj full" "F16_tri_chain --mixer tri --proj chain" "F16_gray_local --mixer gray --proj local"; do
  set -- $spec; tag=$1; shift
  taskset -c 0-6 /usr/bin/python3.13 v3/native/fixed.py --tag $tag --train 8 12 --test 8 12 16 "$@" > logs/v3/native/fixed/$tag.log 2>&1
done
