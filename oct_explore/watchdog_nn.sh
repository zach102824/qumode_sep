#!/bin/bash
# NN curriculum watchdog: ≤2 jobs, but ≤1 when SNAP running or RAM tight. Deadline Mon 09:00 CST
set -u
cd /workspace/qumode_sep/oct_explore
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog_nn.log
QF=queue_nn.txt
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_jobs() { pgrep -f 'python3 -u nn_costgx.py' | wc -l; }
n_snap() { pgrep -f 'python3 -u snap_costgx.py' | wc -l; }
mem_avail_mb() { awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
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
max_nn() {
  local snap mem; snap=$(n_snap); mem=$(mem_avail_mb)
  if [[ $snap -ge 1 ]]; then echo 1; return; fi
  if [[ $mem -lt 2800 ]]; then echo 1; return; fi
  echo 2
}
say "watchdog_nn start (v3: ≤1 NN while SNAP runs / RAM<2.8G)"
while [[ $(date +%s) -lt $DEADLINE ]]; do
  while IFS='|' read -r tag args; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    finalize "$tag"
    if job_done "$tag" || job_running "$tag"; then continue; fi
    mx=$(max_nn); nj=$(n_jobs)
    [[ $nj -ge $mx ]] && continue
    mem=$(mem_avail_mb)
    [[ $mem -lt 1800 ]] && { say "SKIP $tag mem=${mem}MB"; continue; }
    say "LAUNCH $tag (nn=$nj max=$mx mem=${mem}MB): $args"
    # shellcheck disable=SC2086
    setsid nohup env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4" \
      taskset -c 0-7 python3 -u nn_costgx.py --tag "$tag" $args >>"logs/${tag}.log" 2>&1 < /dev/null &
    echo $! > "logs/${tag}.pid"; sleep 8
  done < "$QF"
  sleep 60
done
say "watchdog_nn exit"
