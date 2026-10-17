#!/bin/bash
# SNAP-instead-of-ECD watchdog. Max 1 job. Waits until NN jobs are done (RAM).
set -u
cd /workspace/qumode_sep/oct_explore
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog_snap.log
QF=queue_snap.txt
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_nn() { pgrep -f 'python3 -u nn_costgx.py' | wc -l; }
n_snap() { pgrep -f 'python3 -u snap_costgx.py' | wc -l; }
mem_avail_mb() { awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
job_running() { pgrep -f -- "--tag ${1}( |$)" >/dev/null; }
job_done() { [[ -f "results/${1}.json" ]] && grep -q ALLDONE "logs/${1}.log" 2>/dev/null; }
finalize() {
  local tag="$1"; job_done "$tag" || return 0; [[ -f "logs/${tag}.pushed" ]] && return 0
  say "DONE $tag"
  python3 snap_report.py "$tag" >>"$LOGW" 2>&1 || true
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${tag}.json oct_explore/logs/${tag}.log \
    oct_explore/snap_costgx.py oct_explore/snap_report.py oct_explore/watchdog_snap.sh oct_explore/queue_snap.txt 2>/dev/null
  git -C /workspace/qumode_sep commit -qm "oct_explore: ${tag} SNAP-instead-of-ECD results" 2>/dev/null
  git -C /workspace/qumode_sep push -q origin main 2>/dev/null || say "push failed $tag"
  touch "logs/${tag}.pushed"
}
say "watchdog_snap start (waits for NN clear, MemAvailable>=3500MB)"
while [[ $(date +%s) -lt $DEADLINE ]]; do
  while IFS='|' read -r tag args; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    finalize "$tag"
    if job_done "$tag" || job_running "$tag"; then continue; fi
    [[ $(n_snap) -ge 1 ]] && continue
    # Prefer empty NN; allow launch if only 1 NN and plenty of RAM
    nn=$(n_nn); mem=$(mem_avail_mb)
    if [[ $nn -ge 2 ]]; then continue; fi
    if [[ $nn -eq 1 && $mem -lt 4500 ]]; then continue; fi
    if [[ $nn -eq 0 && $mem -lt 3000 ]]; then continue; fi
    say "LAUNCH $tag (nn=$nn mem=${mem}MB): $args"
    # shellcheck disable=SC2086
    setsid nohup env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4" \
      taskset -c 0-7 python3 -u snap_costgx.py --tag "$tag" $args >>"logs/${tag}.log" 2>&1 < /dev/null &
    echo $! > "logs/${tag}.pid"; sleep 5
  done < "$QF"
  sleep 60
done
say "watchdog_snap exit"
