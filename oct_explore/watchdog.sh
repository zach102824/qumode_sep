#!/bin/bash
set -u
cd /workspace/qumode_sep/oct_explore
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog.log
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_oct_jobs() { pgrep -af 'python3 -u train.py|python3 -u theory_u.py|python3 -u bitflip_mix.py|python3 -u qaoa_ref.py' 2>/dev/null | grep -v grep | wc -l; }
job_running() { pgrep -af -- "--tag ${1}" 2>/dev/null | grep -v grep >/dev/null; }
job_done() { [[ -f "results/${1}.json" ]] && grep -q ALLDONE "logs/${1}.log" 2>/dev/null; }

run_job() {
  local tag="$1"; shift
  if job_done "$tag"; then say "skip $tag (done)"; return 0; fi
  if job_running "$tag"; then say "skip $tag (running)"; return 0; fi
  while [[ $(n_oct_jobs) -ge 3 ]]; do
    say "wait slots (jobs=$(n_oct_jobs)) for $tag"
    sleep 120
    now=$(date +%s); [[ $now -ge $DEADLINE ]] && return 1
  done
  say "LAUNCH $tag: $*"
  setsid nohup env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u "$@" >"logs/${tag}.log" 2>&1 < /dev/null &
  echo $! >"logs/${tag}.pid"
  sleep 5
}

finalize_and_push() {
  local tag="$1"
  if ! job_done "$tag"; then return 0; fi
  if [[ -f "logs/${tag}.pushed" ]]; then return 0; fi
  say "DONE $tag — updating LOG + push"
  python3 report.py BF_gray QAOA_raw T_raw U_ecd_raw U_poly2 U_soft U_step U_poly2_810 U_cav U_L4deep R_raw 2>/dev/null || true
  python3 - "$tag" <<'PY'
import json, sys, time
from pathlib import Path
tag = sys.argv[1]
d = json.load(open(f"results/{tag}.json"))
te = d.get("test", {})
if not te:
    Path(f"logs/{tag}.pushed").write_text("no test\n"); raise SystemExit
ns = sorted(te, key=lambda x: int(x))
Ls = sorted({int(L) for n in te for L in te[n]}, key=int)
lines = [f"\n## {tag} finished ({time.strftime('%H:%M CST')})\n",
         "| L | " + " | ".join(f"n={n}" for n in ns) + " |",
         "|---|" + "|".join(["---"] * len(ns)) + "|"]
for L in Ls:
    row = [str(L)]
    for n in ns:
        block = te[str(n)] if str(n) in te else te[n]
        s = block.get(str(L), block.get(L))
        row.append(f"{s['median_inv_p']:.1f}/{s['B90_inv_p']:.1f}" if s else "-")
    lines.append("| " + " | ".join(row) + " |")
Path("LOG.md").write_text(Path("LOG.md").read_text().rstrip() + "\n" + "\n".join(lines) + "\n")
print("appended", tag)
PY
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${tag}.json oct_explore/logs/${tag}.log oct_explore/results/table.md oct_explore/results/scaling.png 2>/dev/null || true
  git -C /workspace/qumode_sep commit -m "oct_explore: ${tag} results" 2>/dev/null || true
  git -C /workspace/qumode_sep push origin main 2>/dev/null || say "push failed $tag"
  touch "logs/${tag}.pushed"
}

say "watchdog v3 start jobs=$(n_oct_jobs)"

# Theory settled: Gray-X + raw cost. Nonlinear overfits. Next: cav-only, L4 depth, R_raw resume.
QUEUE=(
  "U_soft|theory_u.py --tag U_soft --ecd 0 --diag soft --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_step|theory_u.py --tag U_step --ecd 0 --diag step --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_poly2_810|theory_u.py --tag U_poly2_810 --ecd 0 --diag poly2 --Ls 2 4 6 8 --nt2 10 --restarts 4 --maxiter 250 --ntrain 80"
  "U_cav|theory_u.py --tag U_cav --layout C --ecd 0 --diag raw --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_L4deep|theory_u.py --tag U_L4deep --ecd 0 --diag raw --Ls 1 2 3 4 --restarts 8 --maxiter 400 --ntrain 100"
  "R_raw|train.py --tag R_raw --layout R --cost raw --kind logp --Ls 1 2 4 6 8 --restarts 3 --maxiter 120 --test 10 12 14 20 --ntrain 60"
)

while true; do
  now=$(date +%s)
  [[ $now -ge $DEADLINE ]] && { say "deadline reached"; break; }
  for ent in "${QUEUE[@]}"; do finalize_and_push "${ent%%|*}"; done
  finalize_and_push U_ecd_raw
  finalize_and_push U_poly2
  for ent in "${QUEUE[@]}"; do
    tag="${ent%%|*}"; cmd="${ent#*|}"
    if job_done "$tag" || job_running "$tag"; then continue; fi
    # skip R_raw resume if partial exists without ALLDONE — restart clean only if dead and incomplete
    if [[ "$tag" == "R_raw" ]] && [[ -f results/R_raw.json ]] && ! job_done R_raw; then
      # only relaunch if not running
      if ! job_running R_raw; then
        say "R_raw incomplete and dead — restarting clean"
        rm -f results/R_raw.json
        # shellcheck disable=SC2086
        run_job "$tag" $cmd || break 2
        break
      fi
      continue
    fi
    # shellcheck disable=SC2086
    run_job "$tag" $cmd || break 2
    break
  done
  sleep 180
done
say "watchdog exit"
