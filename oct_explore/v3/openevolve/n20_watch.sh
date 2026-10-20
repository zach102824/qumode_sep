#!/bin/bash
# every new checkpoint (10 iterations): evaluate its best program at held-out n=20 (fair rr + single). cores 5-6.
cd "$(dirname "$0")"; RUN=${1:-runs/net_long}; export OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OE_N20=1 OE_BUDGET=${OE_BUDGET:-300} OE_MEM_MB=6000
mkdir -p $RUN/n20
while pgrep -f "openevolve.cli.*$RUN" >/dev/null || [ ! -d $RUN ]; do
  for c in $RUN/checkpoints/checkpoint_*; do
    [ -f $c/best_program.py ] || continue; k=$(basename $c); [ -f $RUN/n20/$k.json ] && continue
    taskset -c 5-6 python3.13 net/evaluator_net.py $c/best_program.py > $RUN/n20/$k.json 2>$RUN/n20/$k.err
  done; sleep 120
done
