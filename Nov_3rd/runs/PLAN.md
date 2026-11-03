# PLAN — Nov_3rd (final data)

_Asia/Shanghai 2026-11-03 — Zach t2442–t2444_

Fresh package `Nov_3rd/` — knobs only in `settings.py` (no `Nov/` defaults).

## Motif
\[
|0\rangle \xrightarrow{[U_c^{X}\to\mathrm{bare\,SNAP}]^{L}} W \to \text{Fock},
\quad U_c^{X}=W\,D\,W,\quad D=\mathrm{e}^{-i\gamma E},\quad W=H^{\otimes 3}.
\]
**Each cavity holds exactly 3 bits** (\(b=3\), \(C=n/3\)). Binary frame only.

## Locked
| knob | value |
|------|-------|
| bits / cavity | **3** |
| W | **BinH** \(H^{\otimes 3}\) |
| SNAP | **shell**, shared across cavities, per layer |
| Init | **QAOA-warm** |
| Train | **n=6 × 1000**, **L-BFGS only** |
| **L** | **\{1, 4, 8, 12, 16, 20, 24, 28, 32\}** (always) |
| **Eval n** | **9 → 18** inclusive; with \(b=3\) sim on \(n\in\{9,12,15,18\}\) (packs on disk); skip \(n\nmid 3\) |
| Eval | scale1000 **B=1000**; Med/B90 (+ R vs QAOA same \(L\)) |

## Protocol
1. **QAOA warm angles** for every \(L\) in the set (train via `explore_OCT29.scripts.eval_qaoa_L_scale1000 --L L` if `angles.json` missing). On disk now: L∈{1,4,8,16}; need {12,20,24,28,32}.
2. For each \(L\): QAOA-warm → L-BFGS on n6×1000 (seeds 0,1,2) → best ckpt.
3. Eval scale1000 B=1000 on n=9…18 (sim: 9,12,15,18).
4. Write `RESULTS_L{L}.md` + summary `RESULTS.md` under `Nov_3rd/runs/`.

## Run
```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
# optional: fill missing QAOA L first
# python -m explore_OCT29.scripts.eval_qaoa_L_scale1000 --L 12
python -m Nov_3rd.run_final
```
