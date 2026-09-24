# Adaptive soft-cap λ — B0 / B1 / B2 (200 steps)

Updated Asia/Shanghai: 2026-09-24 11:22 (CST) / UTC 03:22.

Setup (all arms): `--u-names ck_pi4 --layers 4 --trials 25 --steps 200`, 20 unique-GS `four_sat` H, 500 trials/arm, workers=7.

| arm | tag | flags |
|---|---|---|
| B0 Gibbs | `fleet_adapt_B0_gibbs` | no λ / no adapt |
| B1 fixed | `fleet_adapt_B1_fixed_l3` | `--lambda 3 --beta-max 2.1` |
| B2 adaptive | `fleet_adapt_B2_adaptive` | `--adapt-lambda --beta-max 2.1` (default knobs) |

Pass-bar context (do **not** stop early): success≥0.90 and ≥15% mean|β| cut vs ~1.84 was the old target; overnight fixed best was at **800** steps — at **200** expect weaker.

Baseline reference mean|β| = **1.84** (historical A0 / this B0 ≈ 1.844).

## Comparison

| arm | success | mean p(GS) | mean\|β\| | median\|β\| | mean trial-max\|β\| | \|β\| cut vs 1.84 | mean final λ | median final λ | mean λ post-warmup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Gibbs | 0.940 | 0.1620 | 1.844 | 1.862 | 3.378 | -0.2% | 0.000 | 0.000 | 0.000 |
| B1 fixed λ=3 | 0.810 | 0.1315 | 1.551 | 1.570 | 2.123 | 15.7% | 3.000 | 3.000 | 3.000 |
| B2 adaptive | 0.598 | 0.1032 | 1.593 | 1.609 | 2.379 | 13.4% | 3.148 | 3.797 | 0.967 |

## B2 final-λ distribution (500 trials)

- frac(final λ > 0) = 0.888
- p10 / p50 / p90 = 0.000 / 3.797 / 3.797
- max final λ = 3.797
- buckets: `{'0': 56, '(0,0.5]': 7, '(0.5,1.5]': 13, '(1.5,3]': 34, '(3,5]': 390}`

## Verdict (200 steps)

**Adaptive does not work well vs B0/B1 at 200 steps.**

- **B0** recovers the Gibbs baseline: success **0.940**, mean|β|≈**1.844** (matches prior A0).
- **B1** fixed λ=3 cuts |β| (15.7% vs 1.84) but drops success to **0.810** — expected to be weaker than the overnight 800-step fixed soft-cap winners (success≥0.90 there).
- **B2** adaptive cuts |β| 13.4% (slightly less than B1’s 15.7%) but **collapses success to 0.598**, far below B0 and B1. Warm-up (λ=0 for 50/200 steps) then aggressive up-ramping appears to over-penalize before the optimizer recovers Gibbs progress; median final λ≈3.80 with ~78% of trials in the (3,5] bucket.

At 200 steps the fair ranking is **B0 ≫ B1 > B2** on success; adaptive is not a free lunch here. Re-test at 400–800 steps (where fixed soft-cap previously cleared the pass bar) before judging the controller.

## Artifacts

- `fleet_adapt_B0_gibbs_20260924T032122Z.json` / `_summary.json`
- `fleet_adapt_B1_fixed_l3_20260924T032202Z.json` / `_summary.json`
- `fleet_adapt_B2_adaptive_20260924T032242Z.json` / `_summary.json`
