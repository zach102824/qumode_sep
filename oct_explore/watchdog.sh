#!/bin/bash
# Keep oct_explore going until Monday 2026-10-19 09:00 Asia/Shanghai. Never touch *.STOP.
set -u
cd /workspace/qumode_sep/oct_explore
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
DEADLINE=$(date -d '2026-10-19 09:00:00' +%s)
LOGW=logs/watchdog.log
say() { echo "$(date '+%F %T')" "$*" | tee -a "$LOGW"; }

n_oct_jobs() {
  pgrep -af 'python3 -u train.py|python3 -u theory_u.py|python3 -u bitflip_mix.py|python3 -u qaoa_ref.py' 2>/dev/null | grep -v grep | wc -l
}

job_running() {
  local tag="$1"
  pgrep -af -- "--tag ${tag}" 2>/dev/null | grep -v grep >/dev/null
}

job_done() {
  local tag="$1"
  [[ -f "results/${tag}.json" ]] && grep -q ALLDONE "logs/${tag}.log" 2>/dev/null
}

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
  python3 report.py BF_gray QAOA_raw T_raw U_ecd_raw U_poly2 U_soft U_step U_poly2_810 U_ecd_poly2 R_raw 2>/dev/null || true
  python3 - "$tag" <<'PY'
import json, sys, time
from pathlib import Path
tag = sys.argv[1]
p = Path("LOG.md")
d = json.load(open(f"results/{tag}.json"))
te = d.get("test", {})
if not te:
    Path(f"logs/{tag}.pushed").write_text("no test\n"); raise SystemExit
lines = [f"\n## {tag} finished ({time.strftime('%H:%M CST')})\n"]
ns = sorted(te, key=lambda x: int(x))
hdr = "| L | " + " | ".join(f"n={n}" for n in ns) + " |"
sep = "|---|" + "|".join(["---"] * len(ns)) + "|"
lines += [hdr, sep]
Ls = sorted({int(L) for n in te for L in te[n]}, key=int)
for L in Ls:
    row = [str(L)]
    for n in ns:
        block = te[str(n)] if str(n) in te else te[n]
        s = block.get(str(L), block.get(L))
        row.append(f"{s['median_inv_p']:.1f}/{s['B90_inv_p']:.1f}" if s else "-")
    lines.append("| " + " | ".join(row) + " |")
# note vs SATLike / BF_gray at n=20 L=4 and L=8 if present
note = []
for L in (4, 8):
    for n in (14, 20):
        if str(n) in te or n in te:
            block = te[str(n)] if str(n) in te else te[n]
            s = block.get(str(L), block.get(L))
            if s:
                note.append(f"n={n} L={L}: {s['median_inv_p']:.1f}/{s['B90_inv_p']:.1f}")
if note:
    lines.append("\nKey: " + "; ".join(note))
p.write_text(p.read_text().rstrip() + "\n" + "\n".join(lines) + "\n")
print("appended", tag)
PY
  git -C /workspace/qumode_sep add oct_explore/LOG.md oct_explore/results/${tag}.json oct_explore/logs/${tag}.log oct_explore/results/table.md oct_explore/results/scaling.png 2>/dev/null || true
  git -C /workspace/qumode_sep commit -m "oct_explore: ${tag} theory unitary results" 2>/dev/null || true
  git -C /workspace/qumode_sep push origin main 2>/dev/null || say "push failed for $tag"
  touch "logs/${tag}.pushed"
}

say "watchdog (re)start deadline=2026-10-19 09:00:00 jobs=$(n_oct_jobs)"

QUEUE=(
  "U_ecd_raw|theory_u.py --tag U_ecd_raw --ecd 1 --diag raw --Ls 1 2 4 6 8 --restarts 4 --maxiter 250 --ntrain 80"
  "U_poly2|theory_u.py --tag U_poly2 --ecd 0 --diag poly2 --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_soft|theory_u.py --tag U_soft --ecd 0 --diag soft --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_step|theory_u.py --tag U_step --ecd 0 --diag step --Ls 1 2 4 6 8 --restarts 5 --maxiter 300 --ntrain 100"
  "U_poly2_810|theory_u.py --tag U_poly2_810 --ecd 0 --diag poly2 --Ls 2 4 6 8 --nt2 10 --restarts 4 --maxiter 250 --ntrain 80"
  "U_ecd_poly2|theory_u.py --tag U_ecd_poly2 --ecd 1 --diag poly2 --Ls 1 2 4 6 8 --restarts 4 --maxiter 250 --ntrain 80"
)

while true; do
  now=$(date +%s)
  if [[ $now -ge $DEADLINE ]]; then
    say "deadline reached — no new launches"
    break
  fi
  for ent in "${QUEUE[@]}"; do
    finalize_and_push "${ent%%|*}"
  done
  finalize_and_push R_raw

  launched=0
  for ent in "${QUEUE[@]}"; do
    tag="${ent%%|*}"; cmd="${ent#*|}"
    if job_done "$tag" || job_running "$tag"; then continue; fi
    # shellcheck disable=SC2086
    if run_job "$tag" $cmd; then launched=1; break; else break 2; fi
  done

  sleep 180
done
say "watchdog exit"
