# Adaptive soft-cap λ — B0 / B1 / B2

Updated Asia/Shanghai: 2026-09-24 11:30 (CST) / UTC 03:30.

Baseline reference mean|β| = **1.84** (historical A0 / B0 ≈ 1.84–1.85).

Pass-bar context (do **not** stop early): success≥0.90 and ≥15% mean|β| cut vs ~1.84 was the old target; overnight fixed best was at **800** steps.

---

## 200-step fleets (prior)

Setup (all arms): `--u-names ck_pi4 --layers 4 --trials 25 --steps 200`, 20 unique-GS `four_sat` H, 500 trials/arm, workers=7.

| arm | tag | flags |
|---|---|---|
| B0 Gibbs | `fleet_adapt_B0_gibbs` | no λ / no adapt |
| B1 fixed | `fleet_adapt_B1_fixed_l3` | `--lambda 3 --beta-max 2.1` |
| B2 adaptive | `fleet_adapt_B2_adaptive` | `--adapt-lambda --beta-max 2.1` (default knobs) |

Warm-up: first 50 steps with λ=0 (`warmup_frac=0.25`); adapt every 25.

### Comparison (200 steps)

| arm | success | mean p(GS) | mean\|β\| | median\|β\| | mean trial-max\|β\| | \|β\| cut vs 1.84 | mean final λ | median final λ | mean λ post-warmup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Gibbs | 0.940 | 0.1620 | 1.844 | 1.862 | 3.378 | -0.2% | 0.000 | 0.000 | 0.000 |
| B1 fixed λ=3 | 0.810 | 0.1315 | 1.551 | 1.570 | 2.123 | 15.7% | 3.000 | 3.000 | 3.000 |
| B2 adaptive | 0.598 | 0.1032 | 1.593 | 1.609 | 2.379 | 13.4% | 3.148 | 3.797 | 0.967 |

### B2 final-λ distribution (200 steps, 500 trials)

- frac(final λ > 0) = 0.888
- p10 / p50 / p90 = 0.000 / 3.797 / 3.797
- max final λ = 3.797
- buckets: `{'0': 56, '(0,0.5]': 7, '(0.5,1.5]': 13, '(1.5,3]': 34, '(3,5]': 390}`

### Verdict (200 steps)

**Adaptive does not work well vs B0/B1 at 200 steps.** Ranking **B0 ≫ B1 > B2** on success.

Artifacts: `fleet_adapt_B0_gibbs_20260924T032122Z_*`, `fleet_adapt_B1_fixed_l3_20260924T032202Z_*`, `fleet_adapt_B2_adaptive_20260924T032242Z_*`.

---

## 400-step fleets (this run)

Setup (all arms): `--u-names ck_pi4 --layers 4 --trials 25 --steps 400`, 20 unique-GS `four_sat` H, 500 trials/arm, workers=7.

| arm | tag | flags |
|---|---|---|
| B0 Gibbs | `fleet_adapt_B0_gibbs_steps400` | no λ / no adapt |
| B1 fixed | `fleet_adapt_B1_fixed_l3_steps400` | `--lambda 3 --beta-max 2.1` |
| B2 adaptive | `fleet_adapt_B2_adaptive_steps400` | `--adapt-lambda --beta-max 2.1` (default knobs) |

Warm-up at 400 steps = first **100** steps with λ=0 (`warmup_frac=0.25`); adapt every 25 — existing code.

### Comparison (400 steps)

| arm | success | mean p(GS) | mean\|β\| | median\|β\| | mean trial-max\|β\| | \|β\| cut vs 1.84 | mean final λ | median final λ | mean λ post-warmup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Gibbs | 0.948 | 0.1944 | 1.849 | 1.844 | 3.417 | -0.5% | 0.000 | 0.000 | 0.000 |
| B1 fixed λ=3 | 0.870 | 0.1599 | 1.545 | 1.545 | 2.113 | 16.0% | 3.000 | 3.000 | 3.000 |
| B2 adaptive | 0.676 | 0.1276 | 1.559 | 1.579 | 2.237 | 15.3% | 3.991 | 5.000 | 2.448 |

### B2 final-λ distribution (400 steps, 500 trials)

- frac(final λ > 0) = 0.880
- p10 / p50 / p90 = 0.000 / 5.000 / 5.000
- max final λ = 5.000 (`adapt_lam_max`)
- buckets: `{'0': 60, '(0,0.5]': 10, '(0.5,1.5]': 15, '(1.5,3]': 26, '(3,5]': 389}`

### Compare to 200-step

| arm | success @200 | success @400 | Δ success | mean\|β\| @200 | mean\|β\| @400 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.940 | 0.948 | +0.008 | 1.844 | 1.849 |
| B1 | 0.810 | 0.870 | +0.060 | 1.551 | 1.545 |
| B2 | 0.598 | 0.676 | +0.078 | 1.593 | 1.559 |

### Verdict (400 steps)

**Adaptive still does not work better than B0/B1 at 400 steps.**

- **B0** stays the success leader (**0.948**), mean|β|≈**1.85** (no cut).
- **B1** fixed λ=3 improves vs 200 (0.810→**0.870**) with a solid **16.0%** |β| cut — still short of the ≥0.90 pass bar but clearly stronger with more steps.
- **B2** adaptive also improves (0.598→**0.676**) and nearly matches B1’s |β| cut (15.3% vs 16.0%), but success remains **far below** B0 and B1. With longer post-warmup (300 vs 150 steps) the controller drives λ harder: median final λ saturates at **`adapt_lam_max=5`**, mean λ post-warmup rises 0.97→**2.45**, and ~78% of trials end in the (3,5] bucket.

Fair ranking at 400: **B0 > B1 ≫ B2** on success. Extra steps help fixed soft-cap more usefully than the adaptive controller; adaptive is still not a free lunch vs fixed λ=3. Re-test at 800 (where fixed soft-cap previously cleared success≥0.90) before any controller changes.

### Artifacts (400)

- `fleet_adapt_B0_gibbs_steps400_20260924T032645Z.json` / `_summary.json`
- `fleet_adapt_B1_fixed_l3_steps400_20260924T032805Z.json` / `_summary.json`
- `fleet_adapt_B2_adaptive_steps400_20260924T032922Z.json` / `_summary.json`
