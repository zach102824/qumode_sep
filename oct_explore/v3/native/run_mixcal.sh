#!/bin/bash
# Milestone-1 calibration queue: 6 single-thread workers (cores 0-6; PID 2888494 owns one core).
cd /workspace/qumode_sep/oct_explore
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
jobs=()
for t in gray tri; do
  for k in 40 32 24; do jobs+=("$t 5 $k --nb 9 --starts 3 --sweeps 1 --maxiter 3000"); done
  for k in 20 16 12; do jobs+=("$t 4 $k --nb 9 --starts 4 --sweeps 1"); done
  for k in 16 8 6 4 3 2; do jobs+=("$t 5 $k --nb 9 --starts 4 --sweeps 1"); done
  for k in 8 6 4 3 2; do jobs+=("$t 4 $k --nb 17 --starts 6 --sweeps 2"); done
  for k in 12 10 8 6 4 3 2; do jobs+=("$t 3 $k --nb 17 --starts 8 --sweeps 2"); done
done
printf '%s\n' "${jobs[@]}" | xargs -P 6 -I{} bash -c 'set -- {}; taskset -c 0-6 /usr/bin/python3.13 v3/native/mixcal.py {} > logs/v3/native/mixcal_$1_K$2_k$3.log 2>&1'
echo ALLDONE > logs/v3/native/mixcal.done
