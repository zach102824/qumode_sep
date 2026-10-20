#!/bin/bash
# lightweight official-TEST checker: every new checkpoint's best program, 1 core, RSS cap. usage: check_watch.sh runs/net16 16 CORE
cd "$(dirname "$0")"; RUN=$1; N=$2; CORE=${3:-5}
export OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OE_N20=1 OE_BUDGET=180 OE_MEM_MB=${MEM:-3500}
if [ "$N" = 16 ]; then export OE_TRAIN_N=8,12 OE_VAL_N=16 OE_CHECK_N=16 OE_NVAL=60; else export OE_TRAIN_N=10,15 OE_VAL_N=20 OE_CHECK_N=20 OE_NVAL=20; fi
mkdir -p $RUN/check
while pgrep -f "openevolve.cli.*$RUN" >/dev/null; do
  for c in $RUN/checkpoints/checkpoint_*; do
    [ -f $c/best_program.py ] || continue; k=$(basename $c); [ -f $RUN/check/$k.json ] && continue
    avail=$(free -m | awk '/Mem:/{print $7}'); [ "$avail" -lt ${NEED:-6800} ] && continue
    taskset -c $CORE python3.13 net/evaluator_net.py $c/best_program.py > $RUN/check/$k.json 2> $RUN/check/$k.err
  done; sleep 120
done
