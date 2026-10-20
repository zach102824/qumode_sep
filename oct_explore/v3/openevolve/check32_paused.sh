#!/bin/bash
# For each new net32o checkpoint: wait for net16o's LLM-only phase (no evaluator child), SIGSTOP net16o, run official n=20 TEST, SIGCONT.
cd /workspace/qumode_sep/oct_explore/v3/openevolve; RUN=runs/net32o; mkdir -p $RUN/check
export OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OE_N20=1 OE_BUDGET=150 OE_MEM_MB=3800
export OE_TRAIN_N=10,15 OE_VAL_N=15 OE_CHECK_N=20 OE_NVAL=30
desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
while pgrep -f "openevolve.cli $RUN" >/dev/null; do
  for c in $RUN/checkpoints/checkpoint_*; do
    [ -f $c/best_program.py ] || continue; k=$(basename $c); [ -f $RUN/check/$k.json ] && continue
    P16=$(pgrep -f "openevolve.cli runs/net16o" | tr '\n' ' ')
    busy=1; for p in $P16; do for d in $(desc $p); do tr '\0' ' ' </proc/$d/cmdline 2>/dev/null | grep -q "evaluator_net.py --inner" && busy=0; done; done
    # busy=0 means net16o is evaluating -> wait; also wait for net32o's own evaluator to finish (LLM phase)
    [ $busy = 0 ] && continue
    for p in $(pgrep -f "openevolve.cli $RUN"); do for d in $(desc $p); do tr '\0' ' ' </proc/$d/cmdline 2>/dev/null | grep -q "evaluator_net.py --inner" && busy=0; done; done
    [ $busy = 0 ] && continue
    [ "$(free -m | awk '/Mem:/{print $7}')" -lt 7000 ] && continue
    P32=$(pgrep -f "openevolve.cli $RUN" | tr "\n" " "); ALL16="$P16 $P32 $(for p in $P16 $P32; do desc $p; done)"
    kill -STOP $ALL16 2>/dev/null; echo "$(date +%T) STOP net16o+net32o OE for $k"
    avail=$(free -m | awk '/Mem:/{print $7}')
    if [ "$avail" -ge 7000 ]; then
      taskset -c 4-7 python3.13 net/evaluator_net.py $c/best_program.py > $RUN/check/$k.json 2> $RUN/check/$k.err
      echo "$(date +%T) checked $k: $(head -c 300 $RUN/check/$k.json | tr '\n' ' ')"
    else echo "$(date +%T) avail $avail < 7000 even with net16o stopped; retry later"; fi
    kill -CONT $ALL16 2>/dev/null; echo "$(date +%T) CONT both"
  done; sleep 30
done
