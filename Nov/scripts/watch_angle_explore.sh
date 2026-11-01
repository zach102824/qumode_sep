#!/usr/bin/env bash
# Watchdog: when current angle_explore exits, launch next wave if no sealed beat.
set -euo pipefail
cd /workspace/qumode_sep
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
OUT=Nov/runs/angle_explore
LOG=$OUT/run.log
STOP=Nov/runs/STOP_ANGLE_EXPLORE
mkdir -p "$OUT"

wait_pid() {
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
}

launch_polish() {
  echo "$(date '+%H:%M:%S') [watchdog] launch polish" | tee -a "$LOG"
  nohup python -m Nov.scripts.angle_explore --phase polish --max_train 200 --scale_B 1000 >>"$LOG" 2>&1 &
  echo $!
}

# If a polish/all is already running, wait on it
PID=$(pgrep -f 'python -m Nov.scripts.angle_explore' | head -1 || true)
if [[ -n "${PID:-}" ]]; then
  echo "$(date '+%H:%M:%S') [watchdog] waiting on existing PID $PID" | tee -a "$LOG"
  wait_pid "$PID"
fi

# Loop: if no beat and no stop, keep launching polish (idempotent warm-starts)
for round in 1 2 3; do
  if [[ -f "$STOP" ]]; then
    echo "$(date '+%H:%M:%S') [watchdog] STOP present — exit" | tee -a "$LOG"
    exit 0
  fi
  if [[ -f "$OUT/BEAT_SEALED.json" ]]; then
    echo "$(date '+%H:%M:%S') [watchdog] BEAT_SEALED present — exit" | tee -a "$LOG"
    exit 0
  fi
  # After first polish completes, re-run polish once more is wasteful unless we change seeds.
  # Wave-3: extra seeds polish via env override — call python inline for adam1200 on best pt only.
  echo "$(date '+%H:%M:%S') [watchdog] wave3 round=$round heavy polish on best pts" | tee -a "$LOG"
  nohup python - <<'PY' >>"$LOG" 2>&1 &
import json, time, sys
from pathlib import Path
import torch
sys.path.insert(0, "/workspace/qumode_sep")
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH
from Nov.scripts.angle_explore import (
    OUT, log, stopped, train_one, scale_record, beats_sealed, append_leaderboard, lb_row, SEALED_REF,
)

device = torch.device("cpu")
W = build_W_BinH(nf=8)
polish = OUT / "polish"
polish.mkdir(exist_ok=True)
# pick best completed scale by sum Med_x
best_pt = None
best_sx = 9e9
best_tag = None
for f in sorted(polish.glob("scale1000_*.json")):
    d = json.loads(f.read_text())
    rows = {r["n"]: r for r in d.get("rows") or []}
    if 9 not in rows: continue
    sx = sum(rows[n]["Med_x"] for n in (9,12,15,18) if n in rows)
    if sx < best_sx:
        best_sx = sx
        # meta sibling
        meta = polish / f.name.replace("scale1000_", "meta_").replace(".json",".json")
        # find matching params under polish/shell
        # fallback: champ
        best_tag = f.stem
for cand in [
    OUT / "polish" / "shell",
    OUT / "progressive",
]:
    pass
# load adam800 params if present
pts = list((OUT / "polish" / "shell").glob("params_polish_shell_pL_adam800*.pt"))
pts += list((OUT / "progressive").glob("params_prog_snap_shell_perL_L32_seed*.pt"))
if not pts:
    log("watchdog wave3: no pts"); raise SystemExit(0)
# unique by resolve
seen=set(); up=[]
for p in pts:
    r=str(p.resolve())
    if r in seen: continue
    seen.add(r); up.append(p)
log(f"watchdog wave3 pts={len(up)} best_scale_sumMedx={best_sx}")
recipes = [
    ("w3_adam1200_lbfgs", 1200, 2e-3, "log_inv", 0.1, 6, 16, 100),
    ("w3_adam800_cvar", 800, 1.5e-3, "cvar", 0.08, 5, 12, 110),
    ("w3_adam800_lr8e4", 800, 8e-4, "log_inv", 0.1, 5, 12, 120),
]
results=[]
beat_any=False
for pt in up[:4]:
    if stopped(): break
    st = torch.load(pt, map_location="cpu", weights_only=False)
    if "snap" not in st and st.get("phi") is not None:
        st["snap"]=st["phi"]
    mode=st.get("snap_mode","shell"); shared=bool(st.get("snap_shared",False)); L=int(st.get("L",32))
    for rname, steps, lr, loss, leakpen, lb_r, lb_i, soff in recipes:
        if stopped(): break
        seed=int(st.get("seed",0))+soff
        arm=f"polish_{mode}_{'sh' if shared else 'pL'}_{rname}"
        # skip if already have scale for this arm+seed
        out_scale = polish / f"scale1000_{arm}_seed{seed}.json"
        if out_scale.exists():
            log(f"skip existing {out_scale.name}"); continue
        log(f"=== wave3 {pt.name} → {arm} ===")
        best = train_one(
            W, L, 8, device, max_train=200, max_val=40, seed=seed,
            lr=lr, steps=steps, gamma0=st["gamma"], snap0=st["snap"],
            snap_mode=mode, snap_shared=shared, loss_kind=loss, leakpen=leakpen,
            lbfgs_mid=True, lbfgs_final=True, lbfgs_rounds=lb_r, lbfgs_iter=lb_i,
            arm_tag=arm, out_dir=polish / mode,
        )
        append_leaderboard(OUT/"leaderboard.csv", lb_row(arm,L,seed,lr,steps,loss,leakpen,best,mode,shared))
        sc, rows = scale_record(W, best["state"], 8, device, [9,12,15,18], 1000, polish, f"{arm}_seed{seed}")
        beat, detail = beats_sealed(sc)
        entry={"arm":arm,"start":str(pt),"beat_sealed":beat,"detail":detail,"seed":seed}
        (polish/f"meta_{arm}_seed{seed}.json").write_text(json.dumps(entry, indent=2, default=str))
        results.append(entry)
        if beat:
            beat_any=True
            (OUT/"BEAT_SEALED.json").write_text(json.dumps(entry, indent=2, default=str))
            log("*** BEAT SEALED ***", arm)
        else:
            n9=detail.get(9,{})
            log(f"no-beat {arm} n9× {n9.get('Med_x')}/{n9.get('B90_x')}")
log("watchdog wave3 DONE beat=", beat_any)
PY
  WPID=$!
  echo "$(date '+%H:%M:%S') [watchdog] wave3 PID=$WPID" | tee -a "$LOG"
  wait_pid "$WPID"
done
echo "$(date '+%H:%M:%S') [watchdog] finished rounds" | tee -a "$LOG"
