# Joint-SNAP pulse budget + paper table

_Generated 2026-10-31 00:05:20 +0800 (Asia/Shanghai)_

## Honesty

Library schedule **renames** densify: still `Uc=exp(-iγ Hc)`. Two-cav needs **joint SNAP** (local SNAP⊗SNAP Fid≪1).

## Pulse account (representatives, γ=1)

| split | joint SNAP Fid | best local Fid | nonzero θ | Kerr T (μs) |
|-------|----------------|----------------|-----------|-------------|
| (1,3) | 1.000000 | 0.9732 | 4 | 1.44 |
| (2,2) | 1.000000 | 0.9657 | 4 | 1.44 |
| (1,3)_g0.50 | 1.000000 | 0.9932 | 4 | 0.72 |
| (2,2)_g0.50 | 1.000000 | 0.9913 | 4 | 0.72 |
| (1,3)_g3.14 | 1.000000 | 0.7789 | 4 | 4.54 |
| (2,2)_g3.14 | 1.000000 | 0.7656 | 4 | 4.54 |

K_AB≈9 kHz. Abstract BinH sandwich: `W†⊗W† · jointSNAP · W⊗W` (5 blocks).

## Clause-span census (scale1000)

| n | frac span1 | span2 | span3 | span4 | L32 abstract pulses/inst |
|---|------------|-------|-------|-------|--------------------------|
| 9 | 0.000 | 0.358 | 0.642 | 0.000 | 2848.0 |
| 12 | 0.000 | 0.185 | 0.653 | 0.162 | 3808.0 |
| 15 | 0.000 | 0.110 | 0.592 | 0.298 | 4768.0 |
| 18 | 0.000 | 0.073 | 0.523 | 0.404 | 5728.0 |
