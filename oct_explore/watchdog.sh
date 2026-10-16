#!/bin/bash
set -u
cd /workspace/qumode_sep/oct_explore
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog.log
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_oct_jobs() { pgrep -af 'python3 -u train.py|python3 -u theory_u.py|python3 -u bitflip_mix.py|python3 -u qaoa_ref.py' 2>/dev/null | grep -v grep | grep -v smoke_ | wc -l; }
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
  python3 report.py BF_gray QAOA_raw T_raw U_ecd_raw U_poly2 U_soft U_step U_poly2_810 U_cav U_L4deep \
    S_gxcost S_costgx S_sexp S_gxonly S_costonly S_bin S_joint S_R 2>/dev/null || true
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
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${tag}.json oct_explore/logs/${tag}.log oct_explore/results/table.md oct_explore/results/scaling.png oct_explore/theory_u.py oct_explore/watchdog.sh 2>/dev/null || true
  git -C /workspace/qumode_sep commit -m "oct_explore: ${tag} results" 2>/dev/null || true
  git -C /workspace/qumode_sep push origin main 2>/dev/null || say "push failed $tag"
  touch "logs/${tag}.pushed"
}

say "watchdog v4 broad-scan start jobs=$(n_oct_jobs)"

# Finish leftover theory jobs, then BROAD SCAN (ECD on every layer; one joint U).
# Short screens L=2,4,8; n=8 train -> n=14/20 test. Do not lock onto one yet.
QUEUE=(
  "U_poly2_810|theory_u.py --tag U_poly2_810 --ecd 0 --diag poly2 --Ls 2 4 6 8 --nt2 10 --restarts 4 --maxiter 250 --ntrain 80"
  "U_L4deep|theory_u.py --tag U_L4deep --ecd 0 --diag raw --Ls 1 2 3 4 --restarts 8 --maxiter 400 --ntrain 100"
  "S_gxcost|theory_u.py --tag S_gxcost --ecd 1 --diag raw --u_form product --u_order gx_cost --code gray --mixer local --Ls 2 4 8 --restarts 4 --maxiter 250 --ntrain 80"
  "S_costgx|theory_u.py --tag S_costgx --ecd 1 --diag raw --u_form product --u_order cost_gx --code gray --mixer local --Ls 2 4 8 --restarts 4 --maxiter 250 --ntrain 80"
  "S_sexp|theory_u.py --tag S_sexp --ecd 1 --diag raw --u_form single_exp --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60 --test 10 12"
  "S_gxonly|theory_u.py --tag S_gxonly --ecd 1 --diag raw --u_form gx_only --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
  "S_costonly|theory_u.py --tag S_costonly --ecd 1 --diag raw --u_form cost_only --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
  "S_bin|theory_u.py --tag S_bin --ecd 1 --diag raw --u_form product --u_order gx_cost --code binary --mixer local --Ls 2 4 8 --restarts 4 --maxiter 250 --ntrain 80"
  "S_joint|theory_u.py --tag S_joint --ecd 1 --diag raw --u_form product --u_order gx_cost --code gray --mixer joint --Ls 2 4 8 --restarts 4 --maxiter 250 --ntrain 80"
  "S_R|theory_u.py --tag S_R --layout R --ecd 1 --diag raw --u_form product --u_order gx_cost --code gray --mixer local --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
  "U_cav|theory_u.py --tag U_cav --layout C --ecd 0 --diag raw --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100 --test 10 12 14"
  "R_raw|train.py --tag R_raw --layout R --cost raw --kind logp --Ls 1 2 4 6 8 --restarts 3 --maxiter 120 --test 10 12 14 20 --ntrain 60"
)

while true; do
  now=$(date +%s)
  [[ $now -ge $DEADLINE ]] && { say "deadline reached"; break; }
  for ent in "${QUEUE[@]}"; do finalize_and_push "${ent%%|*}"; done
  for ent in U_ecd_raw U_poly2 U_soft U_step; do finalize_and_push "$ent"; done
  launched=0
  for ent in "${QUEUE[@]}"; do
    tag="${ent%%|*}"; cmd="${ent#*|}"
    if job_done "$tag" || job_running "$tag"; then continue; fi
    # shellcheck disable=SC2086
    run_job "$tag" $cmd || break 2
    launched=1
    break
  done
  if [[ $launched -eq 0 ]]; then
    # all queued done or running — idle
    remaining=0
    for ent in "${QUEUE[@]}"; do
      tag="${ent%%|*}"
      job_done "$tag" || remaining=1
    done
    [[ $remaining -eq 0 ]] && { say "all queue done"; break; }
  fi
  sleep 120
done
say "watchdog exit"
