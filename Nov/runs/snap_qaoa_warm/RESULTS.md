# RESULTS — snap_qaoa_warm (QAOA-warm SNAP explore)

_2026-11-03 13:14 Asia/Shanghai_

## One-liner (LIVE Opus 5.5 high)
Every packing starts as the identical QAOA circuit, so we screen bits-per-cavity b∈{1,2,3,4,6,12} at n=12 with QAOA-warm gauss/shell/free SNAP at L≤8, then carry the winner to n18 transfer and L=16/32. Bin vs Gray is an exact identity in ideal simulation; free θ ≈ shells by symmetry.

## Gate-0
**PASS** — dual-QAOA θ=-β·pc matches ideal QAOA ~1e-6 rel (`gate0/RESULTS.md`).

## QAOA baselines (scale1000 Med / B90)
| L | n9 | n12 | n15 | source |
|---|----|-----|-----|--------|
| 1 | 29.28 / 63.4 | 90.03 / 216.6 | 289.9 / 750.5 | existing |
| 2 | 12.74 / 30.0 | 30.08 / 78.0 | 76.42 / 223.9 | existing |
| 4 | **5.324 / 13.69** | **9.958 / 31.32** | **20.31 / 65.37** | grown |
| 8 | **2.396 / 7.17** | **3.649 / 14.38** | **5.966 / 27.26** | grown |
| 16 | 2.24 / 5.53 | 3.26 / 10.9 | 5.10 / 18.6 | existing |

## PACKING VERDICT — **LOCKED: b=6 (6bits×2cav)**
Opus item-1 pass (≥2/3 seeds R≤0.95× b=3 same head at L=8):
- **shell:** 2/3 WIN (s1 0.926, s2 0.933; s0 0.952 TIE)
- **free:** 3/3 WIN (0.949 / 0.938 / 0.940)
- Absolute champ: **b=6 shell** Med=3.323 / **R=0.9106×** QAOA at L=8

Caveat: b=6 only divides n∈{6,12,18,…} — n15 stays on b=3.

## PACK-12 tables (R = Med / Med_QAOA, best of 3 seeds)

### L=2
| b | gauss | shell | free |
|---|-------|-------|------|
| 3 | 0.960 | 0.953 | 0.960 |
| 4 | 0.956 | 0.956 | 0.952 |
| **6** | **0.935** | **0.928** | **0.940** |

### L=4 joint-γ
| b | gauss | shell | free |
|---|-------|-------|------|
| 3 | 0.962 | 0.958 | 0.970 |
| 4 | 0.955 | 0.945 | 0.962 |
| **6** | **0.927** | **0.921** | **0.933** |

### L=8 joint-γ
| b | gauss | shell | free |
|---|-------|-------|------|
| 3 | 0.959 | 0.956 | 0.977 |
| 4 | 0.944 | 0.939 | 0.952 |
| **6** | **0.932** | **0.911** | **0.928** |

### L=8 freeze-γ seed0
b=6 shell 0.918× vs b=3 shell 0.955× (0.962× relative).

## Honesty
- Always QAOA-warm θ=-β·pc, γ=γ_QAOA.
- Bit-frame ideal W.
- SNAP-easy ≤2p **stopped**.
