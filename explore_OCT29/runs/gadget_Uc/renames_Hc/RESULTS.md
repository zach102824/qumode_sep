# renames_Hc wave — SNAP φ noise + soft-n18 (+ retrain seed)

_Generated 2026-10-31 02:19:42 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. Blind cost banned.

Opus-style pick: (1) SNAP φ noise (2) WalkSAT fair table (3) soft-n18 (4) L36/40 defer.
Top-2 under renames_Hc: **SNAP φ noise** + **soft-n18**. WalkSAT filled in parallel.

## Locked prior champ (LBFGS seed0) — reference

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.092 | 1.776 |
| 12 | 1.197 | 2.843 |
| 15 | 1.418 | 5.505 |
| 18 | 1.752 | 11.353 |

## Retrain L32 seed (params were missing from git)

| n | Med | B90 |
|---|-----|-----|
| 9 | 2.600 | 6.102 |
| 12 | 4.309 | 12.488 |
| 15 | 7.257 | 23.292 |
| 18 | 12.739 | 49.196 |

b_Med≈**1.1928**, b_B90≈**1.2580**

Far from locked LBFGS (short CPU retrain; no multi-seed polish lineage).

## Soft-n18 (keep=1)

| n | Med | B90 |
|---|-----|-----|
| 9 | 2.241 | 5.200 |
| 12 | 3.555 | 10.412 |
| 15 | 5.760 | 18.361 |
| 18 | 9.802 | 38.644 |

b_Med≈**1.1778**, b_B90≈**1.2454**

- n18 B90 38.644 vs thr 48.212 (ok)
- n9/12/15 Med ≤1.05× retrain (ok)

Verdict: soft-n18 **improved** retrain seed (n18 Med/B90 down) but **no promote to locked champ** (still ≫ 1.752/11.353).

## SNAP φ phase noise (freeze angles; φ ← φ+σξ)

| σ | n9 Med/B90 | n12 Med/B90 | vs σ=0 n9 Med |
|---|------------|-------------|----------------|
| 0.00 | 2.241/5.200 | 3.555/10.412 | ×1.000 |
| 0.01 | 2.246/5.261 | 3.538/10.601 | ×1.002 |
| 0.02 | 2.332/5.436 | 3.674/11.121 | ×1.041 |
| 0.05 | 2.765/6.381 | 4.555/13.295 | ×1.234 |
| 0.10 | 3.294/7.647 | 5.959/16.885 | ×1.470 |

Verdict: σ≤2% mild; σ=5% ~+20% n9 Med; σ=10% ~+40%. Similar story to γ-jitter (OK to ~2%).

## WalkSAT fair table (scale1000, R=20, max_inst=200)

| n | flips Med/B90 | evals Med/B90 | locked champ 1/p Med/B90 |
|---|---------------|---------------|--------------------------|
| 9 | 12.5/40.1 | 51.0/161.4 | 1.092/1.776 |
| 12 | 26.5/65.9 | 107.0/264.6 | 1.197/2.843 |
| 15 | 43.8/128.1 | 176.0/513.6 | 1.418/5.505 |
| 18 | 71.5/203.6 | 287.0/815.4 | 1.752/11.353 |

Champ 1/p ≪ WalkSAT flips/evals (lookup units differ; paper must state shot vs flip).

## Notes

- Retrain seed ≠ locked LBFGS seed0 (`unsat_params.pt` never committed).
- Soft-n18 keep gate vs **retrain** seed; did not beat locked.
- Still renames Hc; no blind stencil. L=36/40 deferred.

