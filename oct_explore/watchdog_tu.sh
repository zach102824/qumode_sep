#!/bin/bash
# theory_u / costgx (non-QAOA) — only when NN and SNAP are idle. Deadline Mon 09:00 CST
set -u
cd /workspace/qumode_sep/oct_explore
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog_tu.log
QF=queue_tu.txt
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_nn() { pgrep -f 'python3 -u nn_costgx.py' | wc -l; }
n_snap() { pgrep -f 'python3 -u snap_costgx.py' | wc -l; }
n_tu() { pgrep -f 'python3 -u theory_u.py|python3 -u costgx.py' | wc -l; }
mem_avail_mb() { awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }
job_running() { pgrep -f -- "--tag ${1}( |$)" >/dev/null; }
job_done() { [[ -f "results/${1}.json" ]] && grep -q ALLDONE "logs/${1}.log" 2>/dev/null; }
finalize() {
  local tag="$1"; job_done "$tag" || return 0; [[ -f "logs/${tag}.pushed" ]] && return 0
  say "DONE $tag"
  # append short table via existing reporters if present
  if [[ -f cg_report.py ]] && [[ "$tag" == C_* ]]; then python3 cg_report.py "$tag" >> LOG.md 2>>"$LOGW" || true; fi
  if [[ -f report.py ]] && [[ "$tag" == S_* ]]; then
    python3 - "$tag" <<'PY' >>LOG.md 2>>"$LOGW" || true
import json,sys,time
tag=sys.argv[1]
d=json.load(open(f'results/{tag}.json'))
te=d.get('test',{})
ns=sorted(te,key=lambda x:int(x))
Ls=sorted({int(L) for n in te for L in te[n]},key=int)
print(f"\n## {tag} finished ({time.strftime('%Y-%m-%d %H:%M CST')})\n")
print('| L | '+' | '.join(f'n={n}' for n in ns)+' |')
print('|---|' + '|'.join(['---']*len(ns)) + '|')
for L in Ls:
  row=[str(L)]
  for n in ns:
    s=te[str(n)].get(str(L))
    row.append(f"{s['median_inv_p']:.1f}/{s['B90_inv_p']:.1f}" if s else '-')
  print('| '+' | '.join(row)+' |')
PY
  fi
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${tag}.json oct_explore/logs/${tag}.log \
    oct_explore/queue_tu.txt oct_explore/watchdog_tu.sh 2>/dev/null
  git -C /workspace/qumode_sep commit -qm "oct_explore: ${tag} binary/joint results" 2>/dev/null
  git -C /workspace/qumode_sep push -q origin main 2>/dev/null || say "push failed $tag"
  touch "logs/${tag}.pushed"
}
say "watchdog_tu start (idle-only: no NN, no SNAP, mem>=3.5G)"
while [[ $(date +%s) -lt $DEADLINE ]]; do
  while IFS='|' read -r tag args; do
    [[ -z "$tag" || "$tag" == \#* ]] && continue
    finalize "$tag"
    if job_done "$tag" || job_running "$tag"; then continue; fi
    [[ $(n_tu) -ge 1 ]] && continue
    [[ $(n_nn) -ge 1 ]] && continue
    [[ $(n_snap) -ge 1 ]] && continue
    mem=$(mem_avail_mb)
    [[ $mem -lt 3500 ]] && continue
    say "LAUNCH $tag (mem=${mem}MB): $args"
    # shellcheck disable=SC2086
    setsid nohup env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4" \
      taskset -c 0-7 python3 -u $args >>"logs/${tag}.log" 2>&1 < /dev/null &
    echo $! > "logs/${tag}.pid"; sleep 8
  done < "$QF"
  sleep 60
done
say "watchdog_tu exit"
