#!/usr/bin/env bash
# Head-to-head: Fourier (φ_k=-βk, 2L) vs Nov_3rd SNAP (5L) at Γ=0.5
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="/workspace/qumode_sep:${PYTHONPATH:-}"
PY="${PY:-/workspace/qtimes_venv/bin/python}"

OUT="${OUT_DIR:-runs/compare/$(date +%Y%m%d_%H%M%S)_G0.5}"
mkdir -p "$OUT"
FOURIER_OUT="$OUT/fourier"
SNAP_OUT="$OUT/snap"
mkdir -p "$FOURIER_OUT" "$SNAP_OUT"

LS_CSV="${LS_CSV:-1,2,4,8}"
if [[ "${INCLUDE_L16:-0}" == "1" ]]; then
  LS_CSV="${LS_CSV},16"
fi
IFS=',' read -r -a LS_ARGS <<< "$LS_CSV"

echo "[compare] out=$OUT  Ls=${LS_ARGS[*]}"

if [[ "${SKIP_FOURIER_TRAIN:-0}" != "1" ]]; then
  echo "[compare] train FOURIER (2L params)…"
  "$PY" -u train_compare.py \
    --method fourier \
    --Gammas 0.5 \
    --Ls "${LS_ARGS[@]}" \
    --seeds 0 1 2 \
    --B-train 64 --B-hold 32 \
    --rounds 20 --batch 32 \
    --modes joint n6 n9 \
    --out "$FOURIER_OUT" | tee "$FOURIER_OUT/train.log"
fi

REUSE_SNAP="${REUSE_SNAP:-runs/smoke/20261105_smoke1}"
if [[ -f "$REUSE_SNAP/train_results.json" ]] && [[ "${FORCE_SNAP_TRAIN:-0}" != "1" ]]; then
  echo "[compare] reuse SNAP ckpts from $REUSE_SNAP"
  LS_JSON=$(printf '%s\n' "${LS_ARGS[@]}" | "$PY" -c 'import sys,json; print(json.dumps([int(x) for x in sys.stdin if x.strip()]))')
  REUSE_SNAP="$REUSE_SNAP" SNAP_OUT="$SNAP_OUT" LS_JSON="$LS_JSON" "$PY" - << 'PY'
import json, os
from pathlib import Path
src = Path(os.environ["REUSE_SNAP"])
dst = Path(os.environ["SNAP_OUT"])
ls = set(json.loads(os.environ["LS_JSON"]))
rows = json.loads((src / "train_results.json").read_text())
out = []
for r in rows:
    if float(r["Gamma"]) != 0.5 or int(r["L"]) not in ls:
        continue
    r = dict(r)
    r["method"] = "snap"
    r["n_params"] = 5 * int(r["L"])
    ck = Path(r["ckpt"])
    if not ck.is_absolute():
        cand = src / ck.name
        ck = cand if cand.exists() else (src / ck)
    r["ckpt"] = str(ck.resolve())
    out.append(r)
(dst / "train_results.json").write_text(json.dumps(out, indent=2))
print(f"wrote {len(out)} SNAP rows → {dst}/train_results.json")
PY
else
  echo "[compare] train SNAP (5L params)…"
  "$PY" -u train_compare.py \
    --method snap \
    --Gammas 0.5 \
    --Ls "${LS_ARGS[@]}" \
    --seeds 0 1 2 \
    --B-train 64 --B-hold 32 \
    --rounds 20 --batch 32 \
    --modes joint n6 n9 \
    --out "$SNAP_OUT" | tee "$SNAP_OUT/train.log"
fi

# keep hold-only companion
if [[ -f write_hold_results.py ]]; then
  "$PY" -u write_hold_results.py "$OUT" "$FOURIER_OUT" "$SNAP_OUT" > "$OUT/RESULTS_hold_only.md" || true
fi

echo "[compare] freeze+interp eval…"
"$PY" -u freeze_eval_compare.py \
  --train-dirs "$FOURIER_OUT" "$SNAP_OUT" \
  --out-dir "$OUT" \
  --eval-ns 12 15 \
  --B-eval 32 \
  --Gammas 0.5 \
  --Ls "${LS_ARGS[@]}" \
  --methods fourier snap | tee "$OUT/eval.log"

echo "[compare] DONE → $OUT/RESULTS.md"
