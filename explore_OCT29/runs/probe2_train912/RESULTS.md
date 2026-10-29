# Opus probe 2 — projector train {n9,n12}, zero-shot n15/n18

_Generated 2026-10-29 17:44:41 +0800 (Asia/Shanghai)_

Fixed W from bake-off. No GNN. L=4.

## Table (val Med / B90)

| model | n9 | n12 | n15 | n18 | b_Med | b_B90 | Med15/12 | Med18/15 |
|-------|----|-----|-----|-----|-------|-------|----------|----------|
| n9-only (bake-off) | 9.27/28.3 | 36.8/65.3 | 76.5/152 | 373/996 | 1.483 | 1.469 | 2.08 | 4.88 |
| train {9,12} (probe2) | 6.03/19.9 | 17.5/40.4 | 30.9/75.3 | 149/581 | 1.405 | 1.431 | 1.76 | 4.84 |

## Verdict

n9-only b_Med=1.483; train{9,12} b_Med=1.405. Med15: n9-only=76.46405410766602 → train912=30.86266803741455. Ratios train912 Med15/12=1.7637801880182307, Med18/15=4.839697158256049. Transfer looks healthy (b≪2, no Med15 cliff) → next: probe 3 Fourier schedule (cheap depth-at-deploy) OR probe 5 seq vocab with W atoms.
