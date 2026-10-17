#!/bin/bash
# v3 runner (approved plan 15:18 CST 2026-10-17): up to 4 jobs x 2 threads = 8 cores, until Mon 2026-10-19 09:00 CST.
# queue line: tag|after|script args...   (after = tag whose results/v3/<after>.json must contain "keep" or "winner")
set -u
cd /workspace/qumode_sep/oct_explore
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/v3/runner.log; QF=v3/queue_v3.txt; PY=/usr/bin/python3.13; MAXJ=4; MINMEM=3500; MAXGNN=2
say() { echo "$(date '+%F %T')" "$*" >> "$LOGW"; }
n_run() { pgrep -f "$PY -u v3/" | wc -l; }
mem_mb() { awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
running() { pgrep -f -- "--tag ${1}( |$)" >/dev/null; }
done_() { grep -q ALLDONE "logs/v3/${1}.log" 2>/dev/null; }
ready() { [[ -z "$1" ]] && return 0; grep -qE '"(keep|winner)"' "results/v3/${1}.json" 2>/dev/null; }
finalize() {
  local t="$1"; done_ "$t" || return 0; [[ -f "logs/v3/${t}.pushed" ]] && return 0
  say "DONE $t"; $PY v3/report_v3.py "$t" >> LOG.md 2>>"$LOGW" || true
  cd /workspace/qumode_sep && git add oct_explore/LOG.md oct_explore/results/v3/${t}.json oct_explore/logs/v3/${t}.log oct_explore/v3 2>/dev/null
  git commit -qm "oct_explore v3: ${t} results" 2>/dev/null
  git pull -q --rebase --autostash origin main 2>>"/workspace/qumode_sep/oct_explore/$LOGW"; git push -q origin main 2>>"/workspace/qumode_sep/oct_explore/$LOGW" || say "push failed $t"
  cd /workspace/qumode_sep/oct_explore; touch "logs/v3/${t}.pushed"
}
say "run_v3 start pid $$"
while [[ $(date +%s) -lt $DEADLINE ]]; do
  while IFS='|' read -r tag after args; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    finalize "$tag"; done_ "$tag" && continue; running "$tag" && continue
    ready "$after" || continue
    [[ $(n_run) -ge $MAXJ ]] && continue
    [[ "$args" == *gnn_v3.py* && $(pgrep -f "$PY -u v3/gnn_v3.py" | wc -l) -ge $MAXGNN ]] && continue
    m=$(mem_mb); [[ $m -lt $MINMEM ]] && { say "hold $tag mem=${m}MB"; continue; }
    say "LAUNCH $tag (mem=${m}MB)"
    setsid nohup env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" \
      $PY -u $args >>"logs/v3/${tag}.log" 2>&1 </dev/null &
    sleep 30
  done < "$QF"
  sleep 60
done
say "run_v3 exit (deadline)"
