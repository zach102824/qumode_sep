# Opus probe 7 — Fourier + α multi-depth distill

_Generated 2026-10-30 00:02:37 +0800 (Asia/Shanghai)_

Gate = **val2** (full val packs: n15 B=40, n18 B=16). Test sealed except milestone.
Retired yardstick: probe5 val b≈1.08 (selection bias).

## Phase 1 — LS distill (no retrain) at L=16

| q | mse | val2 n15 Med/B90 | val2 n18 Med/B90 | L×Med15 |
|---|-----|------------------|------------------|---------|
| 3 | 0.03429 | nan/nan | nan/nan | nan |
| 4 | 0.01095 | nan/nan | nan/nan | nan |
| 6 | 0.003601 | nan/nan | nan/nan | nan |
| 8 | 0.002217 | nan/nan | nan/nan | nan |

## Phase 2 — multi-L train q + α

- q=6 α*=0.008859816007316113 best_step=100 Scomb=3.364084324191556

### val2 deploy sweep

| n \ L | 16 | 24 | 32 |
|-------|------|------|------|
| 9 | 11.7/34 | 9.86/28.9 | 12.5/34.9 |
| 12 | 28.8/102 | 26.1/111 | 32.2/153 |
| 15 | 80.5/225 | 77/244 | 114/425 |
| 18 | 247/871 | 229/1.32e+03 | 385/2.46e+03 |

### b_Med over n∈{12,15,18} by L (val2)

| L | b_Med | b_B90 |
|---|-------|-------|
| 16 | 1.431 | 1.430 |
| 24 | 1.436 | 1.510 |
| 32 | 1.512 | 1.590 |

## Test milestone (sealed; look once)

- test n15 L24: Med=157.658 B90=1075.358 L×Med=3783.8 frac≤2=0.00
- test n18 L24: Med=214.960 B90=1746.157 L×Med=5159.0 frac≤2=0.00

