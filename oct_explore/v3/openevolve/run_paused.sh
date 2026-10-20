#!/bin/bash
# usage: MEMMB=.. NEED=.. run_paused.sh OUT.json ENV... -- prog.py [args]   (waits for both OE runs to be in LLM phase, pauses them, runs chunked_check with RSS watchdog, resumes)
cd /workspace/qumode_sep/oct_explore/v3/openevolve; OUT=$1; shift
desc() { local p; for p in $(pgrep -P $1); do echo $p; desc $p; done; }
evaluating() { for p in $(pgrep -f "openevolve.cli runs/net"); do for d in $(desc $p); do tr '\0' ' ' </proc/$d/cmdline 2>/dev/null | grep -q "evaluator_net.py --inner" && return 0; done; done; return 1; }
while true; do
  if ! evaluating && [ "$(free -m | awk '/Mem:/{print $7}')" -ge ${NEED:-6800} ]; then
    P=$(pgrep -f "openevolve.cli runs/net" | tr '\n' ' '); ALL="$P $(for p in $P; do desc $p; done)"
    kill -STOP $ALL 2>/dev/null; echo "$(date +%T) STOP OE runs"
    env OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" "$@" > $OUT 2> $OUT.err &
    C=$!; peak=0
    while kill -0 $C 2>/dev/null; do
      r=$(awk '/VmRSS/{print int($2/1024)}' /proc/$C/status 2>/dev/null); r=${r:-0}; [ $r -gt $peak ] && peak=$r
      if [ $r -gt ${MEMMB:-3500} ]; then kill -9 $C; echo "{\"error\": \"watchdog RSS ${r}MB > ${MEMMB}MB\"}" > $OUT; fi
      sleep 1; done
    kill -CONT $ALL 2>/dev/null; echo "$(date +%T) CONT OE runs; peak ${peak}MB"; break
  fi; sleep 20
done
