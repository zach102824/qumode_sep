#!/bin/bash
# NN curriculum watchdog: ≤2 jobs × 4 threads, deadline Mon 09:00 CST
set -u
cd /workspace/qumode_sep/oct_explore
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog_nn.log
QF=queue_nn.txt
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_jobs() { pgrep -f 'python3 -u nn_costgx.py' | wc -l; }
job_running() { pgrep -f -- "--tag ${1}( |$)" >/dev/null; }
job_done() { [[ -f "results/${1}.json" ]] && grep -q ALLDONE "logs/${1}.log" 2>/dev/null; }
finalize() {
  local tag="$1"; job_done "$tag" || return 0; [[ -f "logs/${tag}.pushed" ]] && return 0
  say "DONE $tag"
  python3 nn_report.py "$tag" >> LOG.md 2>>"$LOGW" || true
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${tag}.json oct_explore/logs/${tag}.log \
    oct_explore/nn_costgx.py oct_explore/nn_report.py oct_explore/watchdog_nn.sh oct_explore/queue_nn.txt 2>/dev/null
  git -C /workspace/qumode_sep commit -qm "oct_explore: ${tag} NN results" 2>/dev/null
  git -C /workspace/qumode_sep push -q origin main 2>/dev/null || say "push failed $tag"
  touch "logs/${tag}.pushed"
}
say "watchdog_nn start"
while [[ $(date +%s) -lt $DEADLINE ]]; do
  while IFS='|' read -r tag args; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    finalize "$tag"
    if job_done "$tag" || job_running "$tag"; then continue; fi
    [[ $(n_jobs) -ge 2 ]] && continue
    say "LAUNCH $tag: $args"
    # shellcheck disable=SC2086
    setsid nohup env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4" \
      taskset -c 0-7 python3 -u nn_costgx.py --tag "$tag" $args >>"logs/${tag}.log" 2>&1 < /dev/null &
    echo $! > "logs/${tag}.pid"; sleep 3
  done < "$QF"
  sleep 60
done
say "watchdog_nn exit"
