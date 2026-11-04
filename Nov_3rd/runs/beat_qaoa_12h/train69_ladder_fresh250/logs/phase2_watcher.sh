#!/usr/bin/env bash
set -euo pipefail
RUN=/workspace/qumode_sep/Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250
LOG=$RUN/logs/phase2_watcher.log
cd /workspace/qumode_sep
exec >>"$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] watcher start"

while [[ ! -f $RUN/PHASE2_n18_COMPLETE ]]; do
  if ! pgrep -f 'run_train69_ladder_fresh250_phase2.py --pass n18' >/dev/null; then
    echo "n18 orch exited; checking rows"
    /workspace/venv-explore/bin/python - <<'PY'
import json
from pathlib import Path
RUN = Path("Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250")
rows = json.loads((RUN / "results_rows.json").read_text())
Ls = [1, 4, 8, 12, 16, 20, 24, 28, 32]
ok = all("18" in rows["rows"].get(str(L), {}) for L in Ls)
print("all_n18", ok)
if ok:
    (RUN / "PHASE2_n18_COMPLETE").write_text("synth\n")
PY
    break
  fi
  sleep 30
done

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] n18 complete — summarizing + commit"
/workspace/venv-explore/bin/python - <<'PY'
import json
from pathlib import Path
RUN = Path("Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250")
rows = json.loads((RUN / "results_rows.json").read_text())
lines = [
    "# n18 Phase2 summary",
    "",
    "| L | SNAP Med | Q Med | Med | SNAP B90 | Q B90 | B90 |",
    "|---|----------|-------|-----|----------|-------|-----|",
]
for L in [1, 4, 8, 12, 16, 20, 24, 28, 32]:
    r = rows["rows"][str(L)]["18"]
    lines.append(
        f"| {L} | {r['SNAP_Med']:.4f} | {r['QAOA_Med']:.4f} | {r['Med_mark']} | "
        f"{r['SNAP_B90']:.4f} | {r['QAOA_B90']:.4f} | {r['B90_mark']} |"
    )
lines.append("")
(RUN / "N18_SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

git add Nov_3rd/scripts/run_train69_ladder_fresh250_phase2.py \
  Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/ \
  Nov_3rd/DATA_MANIFEST.md || true
git reset -- Nov_3rd/data/scale1000/data_n21.npz 2>/dev/null || true
# avoid staging unrelated dirty tree files
git status -sb | head -50
git commit -m "$(cat <<'EOF'
train69_ladder_fresh250: Phase2 n18 Med+B90 for all L

Eval existing QAOA + power_p seed0 on scale1000 n18 B=1000; no retrain.
EOF
)" || echo "commit may be empty / nothing to commit"
git push origin HEAD
SHA=$(git rev-parse --short HEAD)
echo "PUSHED_N18 $SHA" | tee "$RUN/PHASE2_N18_PUSHED"

# wait for n21 pack
for i in $(seq 1 240); do
  if [[ -f Nov_3rd/data/scale1000/data_n21.npz ]]; then
    echo "n21 pack ready"
    break
  fi
  if ! pgrep -f 'regen_scale1000.py 21' >/dev/null; then
    echo "regen exited without pack"
    break
  fi
  sleep 30
done

if [[ ! -f Nov_3rd/data/scale1000/data_n21.npz ]]; then
  echo "BLOCKER n21 pack missing" | tee "$RUN/PHASE2_N21_BLOCKER"
  TS=$(date '+%Y-%m-%d %H:%M:%S %z')
  echo "" >> "$RUN/STATUS.md"
  echo "- [$TS Asia/Singapore] Phase2 n21 BLOCKER: scale1000 data_n21.npz could not be built. See logs/regen_scale1000_n21.log" >> "$RUN/STATUS.md"
  git add "$RUN/STATUS.md" "$RUN/PHASE2_N21_BLOCKER" 2>/dev/null || true
  git commit -m "train69_ladder_fresh250: Phase2 n21 pack blocker" || true
  git push origin HEAD || true
  exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] starting n21 pass"
env PYTHON=/workspace/venv-explore/bin/python PYTHONPATH=/workspace/qumode_sep \
  /workspace/venv-explore/bin/python Nov_3rd/scripts/run_train69_ladder_fresh250_phase2.py --pass n21 \
  > "$RUN/logs/phase2_n21_orchestrator.log" 2>&1

git add Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/ Nov_3rd/DATA_MANIFEST.md || true
git reset -- Nov_3rd/data/scale1000/data_n21.npz 2>/dev/null || true
git commit -m "$(cat <<'EOF'
train69_ladder_fresh250: Phase2 n21 Med+B90 for all L

Eval existing angles on scale1000 n21 B=1000; Phase2 complete.
EOF
)" || true
git push origin HEAD
echo "PUSHED_FINAL $(git rev-parse --short HEAD)" | tee "$RUN/PHASE2_FINAL_PUSHED"
