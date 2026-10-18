#!/bin/bash
# v3 runner (approved plan 15:18 CST 2026-10-17): up to 4 jobs x 2 threads = 8 cores, until Mon 2026-10-19 09:00 CST.
# queue line: tag|after|script args...   (after = tag whose results/v3/<after>.json must contain top-level "keep")
set -u
cd /workspace/qumode_sep/oct_explore
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/v3/runner.log; QF=v3/queue_v3.txt; PY=/usr/bin/python3.13; MAXJ=4; MINMEM=3500; MAXGNN=2
say() { echo "$(date '+%F %T')" "$*" >> "$LOGW"; }
n_run() { pgrep -f "$PY -u v3/" | wc -l; }
mem_mb() { awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
running() { pgrep -f -- "--tag ${1}( |$)" >/dev/null; }
# Done if ALLDONE in log, OR result JSON has keep+winner and test.winner covers n=20 (survives deleted log fds).
done_() {
  grep -q ALLDONE "logs/v3/${1}.log" 2>/dev/null && return 0
  $PY -c "import json,sys
try:
 d=json.load(open('results/v3/$1.json'))
 te=d.get('test',{}).get('winner',{})
 sys.exit(0 if 'keep' in d and 'winner' in d and '20' in te else 1)
except Exception:
 sys.exit(1)" 2>/dev/null
}
ready() { [[ -z "$1" ]] && return 0; $PY -c "import json,sys; d=json.load(open(\"results/v3/$1.json\")); sys.exit(0 if \"keep\" in d else 1)" 2>/dev/null; }
# If a job's stdout still points at a deleted inode (log path recreated), refresh the visible log from the live fd.
sync_deleted_logs() {
  local pid tgt path
  for pid in $(pgrep -f "$PY -u v3/" 2>/dev/null); do
    tgt=$(readlink /proc/$pid/fd/1 2>/dev/null) || continue
    [[ "$tgt" == *"(deleted)" ]] || continue
    path="${tgt% (deleted)}"
    cat /proc/$pid/fd/1 > "$path" 2>/dev/null || true
  done
}
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
  sync_deleted_logs
  while IFS='|' read -r tag after args; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    finalize "$tag"; done_ "$tag" && continue; running "$tag" && continue
    ready "$after" || continue
    [[ $(n_run) -ge $MAXJ ]] && continue
    # Count gnn_v3 and gnn_snap toward MAXGNN so SNAP front-of-queue cannot stack 4 neural jobs.
    if [[ "$args" == *gnn_v3.py* || "$args" == *gnn_snap.py* ]]; then
      [[ $(pgrep -f "$PY -u v3/gnn_" | wc -l) -ge $MAXGNN ]] && continue
    fi
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
