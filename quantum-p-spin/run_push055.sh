#!/usr/bin/env bash
# Nonstop push toward hold ≥0.55 from dbl_hc_snap variants + ablations.
set -euo pipefail
cd /workspace/qumode_sep/quantum-p-spin
export PYTHONPATH=/workspace/qumode_sep:/workspace/qumode_sep/quantum-p-spin
PY=/workspace/qtimes_venv/bin/python
OUT=runs/nondiag_uc_explore
LOG=$OUT/push055.log
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "===== PUSH055 start $(date) ====="

run() {
  local tag="$1"; shift
  echo "----- $tag $(date) -----"
  "$PY" -u train_nondiag.py --out "$OUT" "$@"
  echo "----- done $tag $(date) -----"
}

# --- Ablations (seed 0–2, standard train) ---
run ablate_nomid \
  --recipes dbl_hc_nomid --L 8 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 64 --B-hold 32 --rounds 30 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 10

run ablate_delta0 \
  --recipes dbl_hc_delta0 --L 8 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 64 --B-hold 32 --rounds 30 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 10

run ablate_onesnap \
  --recipes dbl_hc_onesnap --L 8 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 64 --B-hold 32 --rounds 30 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 10

# --- Free vs QAOA SNAP ---
run qaoa_snap \
  --recipes dbl_hc_qaoa_snap --L 8 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 64 --B-hold 32 --rounds 40 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 15

# --- Champ push: more data + longer Adam ---
run dbl_B256_r50 \
  --recipes dbl_hc_snap --L 8 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 256 --B-hold 32 --rounds 50 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 20

# --- Triple sandwich ---
run tri_hc \
  --recipes tri_hc_snap --L 8 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 64 --B-hold 32 --rounds 40 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 15

# --- Deeper L ---
run dbl_L12 \
  --recipes dbl_hc_snap --L 12 --Gamma 0.5 --seeds 0 1 2 \
  --B-train 64 --B-hold 32 --rounds 40 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 15

run dbl_L16 \
  --recipes dbl_hc_snap --L 16 --Gamma 0.5 --seeds 0 1 \
  --B-train 64 --B-hold 32 --rounds 35 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 12

# If still short of 0.55: tri + B256
run tri_B256 \
  --recipes tri_hc_snap --L 8 --Gamma 0.5 --seeds 0 1 \
  --B-train 256 --B-hold 32 --rounds 50 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 20

run tri_L12 \
  --recipes tri_hc_snap --L 12 --Gamma 0.5 --seeds 0 1 \
  --B-train 64 --B-hold 32 --rounds 35 --batch 32 \
  --opt adam_then_lbfgs --adam-rounds 12

echo "===== PUSH055 finished $(date) ====="
"$PY" -u - <<'PY'
from pathlib import Path
import re
lb = Path("runs/nondiag_uc_explore/LEADERBOARD.md")
rows = []
for line in lb.read_text().splitlines():
    if not line.startswith("| "):
        continue
    parts = [p.strip() for p in line.strip("|").split("|")]
    if len(parts) < 4 or parts[0] == "recipe" or parts[0].startswith("-"):
        continue
    try:
        hold = float(parts[2])
    except ValueError:
        continue
    rows.append((hold, parts[0], parts[1], parts[3], parts[4] if len(parts)>4 else "?"))
rows.sort(reverse=True)
print("TOP holds:")
for r in rows[:15]:
    print(f"  {r[0]:.4f}  {r[1]} L={r[2]} Δ={r[3]} p={r[4]}")
best = rows[0][0] if rows else 0.0
print(f"BEST={best:.4f}  HIT055={'YES' if best >= 0.55 else 'NO'}")
# stall count: distinct recipes with best<=0.54 among push variants
push_names = {"dbl_hc_snap","tri_hc_snap","dbl_hc_qaoa_snap","dbl_hc_nomid","dbl_hc_delta0","dbl_hc_onesnap"}
by = {}
for hold, name, L, d, p in rows:
    if name in push_names:
        by.setdefault((name, L), hold)
        if hold > by[(name, L)]:
            by[(name, L)] = hold
stall = sum(1 for v in by.values() if v <= 0.54)
print(f"variant_cfgs={len(by)} stall_le_054={stall}")
Path("runs/nondiag_uc_explore/PUSH_055_RESULT.md").write_text(
    f"# PUSH_055 result\n\nBest hold **{best:.4f}**\n\nHit ≥0.55: **{'YES' if best >= 0.55 else 'NO'}**\n\n"
    + "\n".join(f"- {h:.4f} `{n}` L={L}" for h,n,L,_,_ in rows[:10])
    + "\n"
)
PY
