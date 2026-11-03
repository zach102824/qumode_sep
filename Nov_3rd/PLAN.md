# PLAN — Nov_3rd (final data)

_Asia/Shanghai 2026-11-03 — Zach t2442–t2448_

**Everything lives under `Nov_3rd/`** — plan, scripts, locked settings, train/eval code, sealed + scale1000 packs, QAOA warm angles, and run outputs. No runtime reads from `Nov/` or `explore_OCT29/data` (only shared physics libs for the simulator).

## Layout
```
Nov_3rd/
  PLAN.md README.md settings.py
  ansatz.py data_io.py train_bfgs.py eval_scale.py run_final.py
  data/sealed/data_n{6,9,12,15,18}.npz
  data/scale1000/data_n{6,9,12,15,18}.npz
  qaoa_angles/L{L}/angles.json (+ results.json when present)
  runs/   # RESULTS, params, logs
```

## Motif
\[
|0\rangle \xrightarrow{[U_c^{X}\to\mathrm{bare\,SNAP}]^{L}} W \to \text{Fock},
\quad U_c^{X}=W\,D\,W,\quad D=\mathrm{e}^{-i\gamma E},\quad W=H^{\otimes 3}.
\]
**3 bits/cavity** (\(b=3\), \(C=n/3\)). Binary only.

## Locked (`settings.py`)
| knob | value |
|------|-------|
| bits/cavity | **3** |
| W | **BinH** |
| SNAP | **shell**, shared across cavities, per layer |
| Init | **QAOA-warm** (from `qaoa_angles/L{L}/`) |
| Train | **n=6 × 1000**, **L-BFGS only** |
| L | **\{1,4,8,12,16,20,24,28,32\}** |
| Eval n | **6→18** (sim: 6,9,12,15,18) |
| Eval | scale1000 **B=1000**; Med/B90 vs QAOA same L |

Expect **tie or beat** same-L QAOA.

## Protocol
1. Ensure `qaoa_angles/L{L}/angles.json` for every L (have 1/2/4/8/16; need 12/20/24/28/32).
2. Each L: warm → L-BFGS on sealed n6×1000 (seeds 0,1,2).
3. Eval `data/scale1000` B=1000 on n∈{6,9,12,15,18}.
4. Write `runs/RESULTS_L{L}.md` + `runs/RESULTS.md`.

## Run
```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov_3rd.run_final
```
