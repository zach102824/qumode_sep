#!/bin/bash
set -u
cd /workspace/qumode_sep/oct_explore
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog.log
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }
n_oct_jobs() { pgrep -af 'python3 -u train.py|python3 -u theory_u.py|python3 -u bitflip_mix.py|python3 -u native_u.py|python3 -u qaoa_ref.py' 2>/dev/null | grep -v grep | grep -v smoke_ | wc -l; }
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
  python3 report.py S_costgx BF_gray QAOA_raw T_raw N_dsnap N_snapd N_dsnapd N_snapcost N_gensum N_bssnap N_dsnap_R 2>/dev/null || true
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
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${tag}.json oct_explore/logs/${tag}.log oct_explore/native_u.py oct_explore/watchdog.sh oct_explore/results/table.md 2>/dev/null || true
  git -C /workspace/qumode_sep commit -m "oct_explore: ${tag} results" 2>/dev/null || true
  git -C /workspace/qumode_sep push origin main 2>/dev/null || say "push failed $tag"
  touch "logs/${tag}.pushed"
}

say "watchdog v7 NATIVE (ECD+SNAP+disp; Gray-X reference only) start jobs=$(n_oct_jobs)"

# Native qumode joint U — no Pauli/Gray-X. Goal: transfer ≈ SATLike.
QUEUE=(
  "N_dsnap|native_u.py --tag N_dsnap --form d_snap --snap 2 --Ls 2 4 8 --restarts 4 --maxiter 250 --ntrain 80"
  "N_snapd|native_u.py --tag N_snapd --form snap_d --snap 2 --Ls 2 4 8 --restarts 4 --maxiter 250 --ntrain 80"
  "N_dsnapd|native_u.py --tag N_dsnapd --form d_snap_d --snap 2 --Ls 2 4 8 --restarts 4 --maxiter 250 --ntrain 80"
  "N_snapcost|native_u.py --tag N_snapcost --form snap_cost --snap 2 --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
  "N_gensum|native_u.py --tag N_gensum --form gen_sum --snap 2 --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
  "N_bssnap|native_u.py --tag N_bssnap --form bs_snap --snap 2 --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
  "N_dsnap4|native_u.py --tag N_dsnap4 --form d_snap --snap 4 --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
  "N_dsnap_R|native_u.py --tag N_dsnap_R --layout R --form d_snap --snap 2 --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
  "N_dsnapd4|native_u.py --tag N_dsnapd4 --form d_snap_d --snap 4 --Ls 2 4 8 --restarts 3 --maxiter 200 --ntrain 60"
)

while true; do
  now=$(date +%s)
  [[ $now -ge $DEADLINE ]] && { say "deadline reached"; break; }
  for ent in "${QUEUE[@]}"; do finalize_and_push "${ent%%|*}"; done
  # also finalize any leftover Gray-X refs if they finish
  for ent in S_costgx S_snap2 S_bin S_costonly; do finalize_and_push "$ent"; done
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
    remaining=0
    for ent in "${QUEUE[@]}"; do
      tag="${ent%%|*}"
      job_done "$tag" || remaining=1
    done
    [[ $remaining -eq 0 ]] && { say "all native queue done"; break; }
  fi
  sleep 120
done
say "watchdog exit"
