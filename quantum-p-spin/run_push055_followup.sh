#!/usr/bin/env bash
# Follow-up after main PUSH055: QAOA-SNAP deeper / longer / triple
set -euo pipefail
cd /workspace/qumode_sep/quantum-p-spin
export PYTHONPATH=/workspace/qumode_sep:/workspace/qumode_sep/quantum-p-spin
PY=/workspace/qtimes_venv/bin/python
OUT=runs/nondiag_uc_explore
LOG=$OUT/push055_followup.log

# Wait for main push to finish
while pgrep -f 'run_push055.sh' >/dev/null 2>&1; do
  echo "$(date) waiting for run_push055.sh ..." | tee -a "$LOG"
  sleep 30
done

exec > >(tee -a "$LOG") 2>&1
echo "===== PUSH055 followup start $(date) ====="

run() {
  local tag="$1"; shift
  echo "----- $tag $(date) -----"
  "$PY" -u train_nondiag.py --out "$OUT" "$@"
  echo "----- done $tag $(date) -----"
}

run qaoa_B256_r60 \
  --recipes dbl_hc_qaoa_snap --L 8 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 256 --B-hold 32 --rounds 60 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 25

run qaoa_L12 \
  --recipes dbl_hc_qaoa_snap --L 12 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 64 --B-hold 32 --rounds 50 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 20

run qaoa_L16 \
  --recipes dbl_hc_qaoa_snap --L 16 --Gamma 0.5 --seeds 0 1 \
  --B-train 64 --B-hold 32 --rounds 40 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 15

run tri_qaoa \
  --recipes tri_hc_qaoa_snap --L 8 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 64 --B-hold 32 --rounds 50 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 20

run tri_qaoa_L12 \
  --recipes tri_hc_qaoa_snap --L 12 --Gamma 0.5 --seeds 0 1 \
  --B-train 64 --B-hold 32 --rounds 40 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 15

echo "===== PUSH055 followup finished $(date) ====="
"$PY" -u - <<'PY'
from pathlib import Path
rows=[]
for line in Path("runs/nondiag_uc_explore/LEADERBOARD.md").read_text().splitlines():
    if not line.startswith("| "): continue
    parts=[p.strip() for p in line.strip("|").split("|")]
    if parts[0] in ("recipe","") or parts[0].startswith("-"): continue
    try: hold=float(parts[2])
    except: continue
    rows.append((hold, parts[0], parts[1], parts[4]))
rows.sort(reverse=True)
best=rows[0][0]
txt=["# PUSH_055 final result\n", f"Best hold **{best:.4f}**\n", f"Hit ≥0.55: **{'YES' if best>=0.55 else 'NO'}**\n\n", "## Top\n"]
for h,n,L,p in rows[:15]:
    txt.append(f"- {h:.4f} `{n}` L={L} p={p}\n")
Path("runs/nondiag_uc_explore/PUSH_055_RESULT.md").write_text("".join(txt))
print("BEST", best, "HIT055", best>=0.55)
for r in rows[:10]:
    print(f"  {r[0]:.4f} {r[1]} L={r[2]} p={r[3]}")
PY
