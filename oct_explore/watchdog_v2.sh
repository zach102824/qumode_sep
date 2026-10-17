#!/bin/bash
# v2 queue runner: up to 2 jobs x 4 threads (8 cores). Deadline Mon 09:00 CST
set -u
cd /workspace/qumode_sep/oct_explore
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog_v2.log; QF=queue_v2.txt; PY=/usr/bin/python3.13
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_run() { pgrep -f "$PY -u costgx.py --tag R_" | wc -l; }
mem_mb() { awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
running() { pgrep -f -- "--tag ${1}( |$)" >/dev/null; }
done_() { [[ -f "results/${1}.json" ]] && grep -q ALLDONE "logs/${1}.log" 2>/dev/null; }
finalize() {
  local t="$1"; done_ "$t" || return 0; [[ -f "logs/${t}.pushed" ]] && return 0
  say "DONE $t"
  $PY - "$t" >>LOG.md 2>>"$LOGW" <<'PY' || true
import json,sys,time
t=sys.argv[1]; te=json.load(open(f'results/{t}.json')).get('test',{})
ns=sorted(te,key=int); Ls=sorted({int(L) for n in te for L in te[n]})
print(f"\n## {t} finished ({time.strftime('%Y-%m-%d %H:%M')} CST), median/B90 of 1/p\n")
print('| L | '+' | '.join(f'n={n}' for n in ns)+' |'); print('|---|'+'|'.join(['---']*len(ns))+'|')
for L in Ls: print('| '+' | '.join([str(L)]+[(lambda s: f"{s['median_inv_p']:.1f}/{s['B90_inv_p']:.1f}" if s else '-')(te[n].get(str(L))) for n in ns])+' |')
PY
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${t}.json oct_explore/logs/${t}.log oct_explore/queue_v2.txt oct_explore/watchdog_v2.sh 2>/dev/null
  git -C /workspace/qumode_sep commit -qm "oct_explore v2: ${t} results" 2>/dev/null
  git -C /workspace/qumode_sep pull -q --rebase --autostash origin main 2>/dev/null; git -C /workspace/qumode_sep push -q origin main 2>/dev/null || say "push failed $t"
  touch "logs/${t}.pushed"
}
say "watchdog_v2 start"
while [[ $(date +%s) -lt $DEADLINE ]]; do
  while IFS='|' read -r tag args; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    finalize "$tag"
    done_ "$tag" && continue; running "$tag" && continue
    [[ $(n_run) -ge 2 ]] && continue
    m=$(mem_mb); [[ $m -lt 4000 ]] && continue
    say "LAUNCH $tag (mem=${m}MB)"
    setsid nohup env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4" \
      taskset -c 0-7 $PY -u $args >>"logs/${tag}.log" 2>&1 </dev/null &
    sleep 20
  done < "$QF"
  sleep 60
done
say "watchdog_v2 exit"
