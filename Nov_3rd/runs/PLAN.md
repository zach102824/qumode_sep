# PLAN — Nov_3rd (final data)

_Asia/Shanghai 2026-11-03 — Zach t2442–t2449_

**Everything under `Nov_3rd/`** — plan, scripts, settings, tests, data, QAOA angles, runs.

## Motif
\[
|0\rangle \xrightarrow{[U_c^{X}\to\mathrm{bare\,SNAP}]^{L}} W \to \text{Fock},
\quad U_c^{X}=W\,D\,W,\quad D=\mathrm{e}^{-i\gamma E},\quad W=H^{\otimes 3}.
\]
**3 bits/cavity**. Binary. QAOA-warm shared shell. Train n=6×1000, **L-BFGS only**.

## Locked
| knob | value |
|------|-------|
| bits/cavity | **3** |
| W / SNAP | **BinH** / **shell** (shared cavities, per layer) |
| Init | **QAOA-warm** |
| Train | **n=6 × 1000**, **L-BFGS only** |
| L | **\{1,4,8,12,16,20,24,28,32\}** small→large |
| Eval n | **6→18** (sim: 6,9,12,15,18), scale1000 **B=1000** |

## Gate 0 — tests (required before any train)
```bash
python -m Nov_3rd.tests.run_tests
```
Suite T1–T12: locks, local data, packing/W=H⊗3, shell warm map, motif sealed overlap, warm≈QAOA ≤2%, blockmix fidelity, no Adam, train N=1000, eval ns, vacuum+final W, L_LIST.  
**Must all PASS.** Results → `runs/TESTS.md`.

## Ladder protocol
1. Ensure `qaoa_angles/L{L}/angles.json` (+ results) for that L (train/copy if missing).
2. L-BFGS train n6×1000, seeds 0,1,2 → best ckpt.
3. Eval scale1000 B=1000 on {6,9,12,15,18}.
4. **Compare vs QAOA same L** on each n with a QAOA Med:
   - \(R=\mathrm{Med}/\mathrm{Med}_{QAOA}\)
   - **BEAT** if \(R\le0.98\); **TIE** if \(0.98<R\le1.02\); **UNDER** if \(R>1.02\)
5. If **any n is UNDER** → **STOP** the ladder; treat as bug/missing setting; do not continue to larger L.
6. Else write `runs/RESULTS_L{L}.md` + append summary; proceed to next L.

## Run
```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov_3rd.tests.run_tests && python -m Nov_3rd.run_final
```
