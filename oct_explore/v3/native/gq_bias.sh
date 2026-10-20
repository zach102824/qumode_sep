#!/bin/bash
# usage: CORES=0-1 MEMMB=3000 gq_bias.sh TAG args...   (RSS watchdog kills only this job if it exceeds MEMMB)
cd /workspace/qumode_sep/oct_explore; tag=$1; shift; mkdir -p logs/v3/native/gnnw
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MEMMB=${MEMMB:-3000}
taskset -c ${CORES:-0-6} /usr/bin/python3.13 v3/native/gnn_bias.py --tag $tag "$@" > logs/v3/native/gnnw/$tag.log 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  r=$(awk '/VmRSS/{print int($2/1024)}' /proc/$P/status 2>/dev/null); r=${r:-0}
  if [ "$r" -gt "$MEMMB" ]; then echo "$(date +%T) WATCHDOG: RSS ${r}MB > ${MEMMB}MB, killed" >> logs/v3/native/gnnw/$tag.log; kill -9 $P; fi
  sleep 2
done
