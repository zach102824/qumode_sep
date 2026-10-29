# Opus probe 3 — Fourier schedule on projector

_Generated 2026-10-29 17:47:30 +0800 (Asia/Shanghai)_

Train Fourier modes at L=4 on {n9,n12}; deploy same modes at larger L, no retrain.

## Med / B90 vs deploy L

| n \ L | 4 | 6 | 8 | 12 |
|-------|------|------|------|------|
| 9 | 6.58/24.7 | 4.07/14.5 | 3.13/9.91 | 2.62/8.07 |
| 12 | 18.9/30.1 | 8.48/19.7 | 5.39/15.4 | 3.76/10.4 |
| 15 | 50.2/102 | 25.1/37.2 | 16/21.5 | 9.67/14.6 |
| 18 | 194/674 | 88.6/361 | 77.5/688 | 75.4/481 |

## Fits b_Med over n at each L

| L | b_Med | b_B90 |
|---|-------|-------|
| 4 | 1.449 | 1.449 |
| 6 | 1.411 | 1.408 |
| 8 | 1.429 | 1.545 |
| 12 | 1.444 | 1.522 |

## Verdict

Fourier q=2: b_Med@L4=1.449, b_Med@L12=1.444. n15 Med vs L: [50.23627853393555, 25.07676601409912, 15.960568904876709, 9.668326377868652]. monotonic_nonincreasing≈True. Depth-at-deploy works → keep Fourier; optional probe 5 for native novelty.
