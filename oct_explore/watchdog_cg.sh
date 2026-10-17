#!/bin/bash
# watchdog v8: ECD + cost·GrayX improvement queue (costgx.py). Resumable; max 3 jobs x 2 threads.
set -u
cd /workspace/qumode_sep/oct_explore
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog.log
QF=queue_cg.txt   # lines: TAG|args   (edit to add jobs; read every loop)
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_jobs() { pgrep -f 'python3 -u (costgx|train|theory_u|native_u|qaoa_ref).py' | wc -l; }
job_running() { pgrep -f -- "--tag ${1}( |$)" >/dev/null; }
job_done() { [[ -f "results/${1}.json" ]] && grep -q ALLDONE "logs/${1}.log" 2>/dev/null; }
finalize() {
  local tag="$1"; job_done "$tag" || return 0; [[ -f "logs/${tag}.pushed" ]] && return 0
  say "DONE $tag"
  python3 cg_report.py "$tag" >> LOG.md 2>>"$LOGW" || true
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${tag}.json oct_explore/logs/${tag}.log oct_explore/costgx.py oct_explore/cg_report.py oct_explore/watchdog_cg.sh oct_explore/queue_cg.txt 2>/dev/null
  git -C /workspace/qumode_sep commit -qm "oct_explore: ${tag} results" 2>/dev/null
  git -C /workspace/qumode_sep push -q origin main 2>/dev/null || say "push failed $tag"
  touch "logs/${tag}.pushed"
}
say "watchdog v8 costgx start"
while [[ $(date +%s) -lt $DEADLINE ]]; do
  while IFS='|' read -r tag args; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    finalize "$tag"
    if job_done "$tag" || job_running "$tag"; then continue; fi
    [[ $(n_jobs) -ge 3 ]] && continue
    say "LAUNCH $tag: $args"
    # shellcheck disable=SC2086
    setsid nohup env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" \
      taskset -c 0-7 python3 -u costgx.py --tag "$tag" $args >>"logs/${tag}.log" 2>&1 < /dev/null &
    echo $! > "logs/${tag}.pid"; sleep 3
  done < "$QF"
  sleep 60
done
say "watchdog v8 exit"
