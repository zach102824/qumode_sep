#!/bin/bash
# Keep oct_explore going until Monday 2026-10-19 09:00 Asia/Shanghai. Never touch *.STOP.
set -u
cd /workspace/qumode_sep/oct_explore
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=false
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s 2>/dev/null || date -j -f '%Y-%m-%d %H:%M:%S' '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog.log
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }

n_oct_jobs() {
  pgrep -af 'oct_explore/(train|theory_u|bitflip|qaoa|fit_ecd)' 2>/dev/null | grep -v 'watchdog\|pgrep\|grep' | wc -l
}

run_job() {
  local tag="$1"; shift
  if [[ -f "results/${tag}.json" ]] && grep -q ALLDONE "logs/${tag}.log" 2>/dev/null; then
    say "skip $tag (done)"; return 0
  fi
  if pgrep -af "theory_u.py --tag $tag|train.py --tag $tag" | grep -v grep >/dev/null; then
    say "skip $tag (running)"; return 0
  fi
  # wait until <2 oct jobs (R_raw + one theory)
  while [[ $(n_oct_jobs) -ge 2 ]]; do
    say "wait slots (jobs=$(n_oct_jobs)) for $tag"; sleep 120
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
  if grep -q ALLDONE "logs/${tag}.log" 2>/dev/null; then
    if [[ ! -f "logs/${tag}.pushed" ]]; then
      say "DONE $tag — updating LOG + push"
      python3 report.py BF_gray QAOA_raw T_raw U_ecd_raw U_poly2 U_soft U_step R_raw 2>/dev/null || true
      python3 - << PY
import json, time
from pathlib import Path
tag = "$tag"
p = Path("LOG.md")
try:
    d = json.load(open(f"results/{tag}.json"))
except Exception as e:
    print(e); raise SystemExit
lines = [f"\\n## {tag} finished ({time.strftime('%H:%M CST')})\\n"]
te = d.get("test", {})
hdr = "| L | " + " | ".join(f"n={n}" for n in sorted(te, key=int)) + " |"
sep = "|---|" + "|".join(["---"]*len(te)) + "|"
lines += [hdr, sep]
Ls = sorted({int(L) for n in te for L in te[n]}, key=int)
for L in Ls:
    row = [str(L)]
    for n in sorted(te, key=int):
        s = te[str(n) if str(n) in te else n].get(str(L)) or te[str(n) if str(n) in te else n].get(L)
        if s: row.append(f"{s['median_inv_p']:.1f}/{s['B90_inv_p']:.1f}")
        else: row.append("-")
    lines.append("| " + " | ".join(row) + " |")
Path("LOG.md").write_text(p.read_text().rstrip() + "\\n" + "\\n".join(lines) + "\\n")
print("appended", tag)
PY
      git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results oct_explore/theory_u.py oct_explore/report.py oct_explore/logs/${tag}.log 2>/dev/null || true
      git -C /workspace/qumode_sep add -u oct_explore/ 2>/dev/null || true
      git -C /workspace/qumode_sep commit -m "oct_explore: ${tag} theory unitary results" 2>/dev/null || true
      git -C /workspace/qumode_sep push origin main 2>/dev/null || say "push failed for $tag"
      touch "logs/${tag}.pushed"
    fi
  fi
}

say "watchdog start deadline=$(date -d @$DEADLINE '+%F %T' 2>/dev/null || echo Monday-09:00)"

# Queue (theory first)
QUEUE=(
  "U_ecd_raw|theory_u.py --tag U_ecd_raw --ecd 1 --diag raw --Ls 1 2 4 6 8 --restarts 4 --maxiter 250 --ntrain 80"
  "U_poly2|theory_u.py --tag U_poly2 --ecd 0 --diag poly2 --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_soft|theory_u.py --tag U_soft --ecd 0 --diag soft --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_step|theory_u.py --tag U_step --ecd 0 --diag step --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_poly2_810|theory_u.py --tag U_poly2_810 --ecd 0 --diag poly2 --Ls 2 4 6 8 --nt2 10 --restarts 4 --maxiter 250 --ntrain 80"
  "U_ecd_poly2|theory_u.py --tag U_ecd_poly2 --ecd 1 --diag poly2 --Ls 1 2 4 6 8 --restarts 4 --maxiter 250 --ntrain 80"
)

idx=0
while true; do
  now=$(date +%s)
  if [[ $now -ge $DEADLINE ]]; then
    say "deadline reached — stopping launches (running jobs keep going)"
    break
  fi
  # finalize finished
  for ent in "${QUEUE[@]}"; do
    tag="${ent%%|*}"; finalize_and_push "$tag"
  done
  finalize_and_push R_raw

  # launch next pending
  for ent in "${QUEUE[@]}"; do
    tag="${ent%%|*}"; cmd="${ent#*|}"
    if [[ -f "results/${tag}.json" ]] && grep -q ALLDONE "logs/${tag}.log" 2>/dev/null; then
      continue
    fi
    if pgrep -af -- "--tag ${tag}" | grep -v grep >/dev/null; then
      continue
    fi
    # shellcheck disable=SC2086
    run_job "$tag" $cmd || break 2
    break  # one launch per loop
  done

  # if R_raw died early without ALLDONE, leave it (don't restart unless clearly dead with no result)
  if ! pgrep -af 'train.py --tag R_raw' | grep -v grep >/dev/null; then
    if [[ -f results/R_raw.json ]] && ! grep -q ALLDONE logs/R_raw.log 2>/dev/null; then
      # resume only if train has some L saved and test incomplete — skip auto-resume to avoid clobber
      say "R_raw not running; log has no ALLDONE (leave for manual)"
    fi
  fi

  sleep 180
done
say "watchdog exit"
