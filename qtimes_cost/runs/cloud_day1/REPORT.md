# cloud_day1 CPU explore

Target is the box **NEW BEST**: L=10, leakpen=0.1, lr=5e-5, learned per-cavity α0. Small val n6 Med≈10.2 / n9≈18. **reval40 ≈ 14.5 / 19.5** is the bar. A result is a win only when a val-40 reval beats both of those medians and the n9/n6 ratio stays in [0.75, 2.0]. Search-val numbers are not a win.

The question for this farm is whether **L∈{12,14}** at a similar lr and leakpen can beat that bar. An L=10 run at the same lr, leakpen, and step count is a control, not a new box config. Learned α0 only. Earlier r014 (L=10, lr=1e-4, small-val 18/23) is an easier, superseded reference.

## Protocol

- Workers: 4 processes × 1 torch thread (host CPUs: 4).
- Search: 1000 Adam steps, batch 4 per n, train pool 128, val 16 (early-stop on S_comb, patience 5 evals).
- S_comb = 0.35·S6 + 0.65·S9, S = (log Med + log B90) / 2. Lower is better.
- Promote only when S_comb improves **and** n9 Med does not regress. A healthier n6→n9 ratio with a worse n9 Med is rejected (`ratio_only`).
- Reval: val n=40 and test n=100 (test is reported, not used to pick the winner). nf=24 (library default).
- Grad clip 5.0, same as `short_train`. nf=24. No inter-trial kill: deeper L is slower early and keeps its full step budget.

## Box reference

| | L | leakpen | lr | α0 | n6 Med | n9 Med | what it is |
|---|---:|---:|---:|---|---:|---:|---|
| box NEW BEST, small val | 10 | 0.1 | 5e-5 | learned | 10.2 | 18 | optimistic |
| **box NEW BEST, reval40** | 10 | 0.1 | 5e-5 | learned | **14.5** | **19.5** | win bar |
| old r014, small val | 10 | 0.1 | 1e-4 | learned | 18 | 23 | superseded |

## Search trials

Med/B90 are 1/p on the **search val** (16 formulas). Lower is better. keep=1 means that trial promoted when it finished.

| id | tag | L | lp | lr | H | r | steps | status | S_comb | n6 Med/B90 | n9 Med/B90 | ratio | leak9 | keep |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|
| 16 | grid | 12 | 0.2 | 0.0001 | 128 | 4 | 1000 | ok | 3.145 | 11.6/25.4 | 15.7/47.5 | 1.36 | 0.060 | 0 |
| 4 | core | 14 | 0.2 | 5e-05 | 128 | 4 | 1000 | ok | 3.246 | 8.5/22.5 | 24.6/52.4 | 2.90 | 0.070 | 0 |
| 0 | core | 12 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.247 | 12.6/23.5 | 11.1/91.6 | 0.88 | 0.032 | 1 |
| 14 | seed | 14 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.411 | 21.2/123.5 | 11.6/45.2 | 0.55 | 0.033 | 0 |
| 15 | l10_control | 10 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.473 | 12.4/28.6 | 27.2/68.3 | 2.20 | 0.061 | 0 |
| 3 | core | 12 | 0.2 | 5e-05 | 128 | 4 | 801 | early_stop | 3.509 | 12.5/25.4 | 21.7/101.5 | 1.73 | 0.050 | 1 |
| 11 | grid | 12 | 0.3 | 5e-05 | 128 | 4 | 1000 | ok | 3.563 | 9.8/16.7 | 25.4/145.7 | 2.58 | 0.052 | 0 |
| 2 | l10_control | 10 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.631 | 14.9/28.4 | 22.0/124.6 | 1.48 | 0.047 | 0 |
| 1 | core | 14 | 0.1 | 5e-05 | 128 | 4 | 1000 | ok | 3.676 | 15.9/100.4 | 18.8/81.8 | 1.18 | 0.037 | 0 |
| 5 | core | 12 | 0.05 | 5e-05 | 128 | 4 | 841 | early_stop | 3.685 | 8.5/40.9 | 23.7/151.7 | 2.78 | 0.034 | 0 |
| 10 | grid | 14 | 0.1 | 3e-05 | 128 | 4 | 841 | early_stop | 3.699 | 22.6/190.6 | 14.1/68.4 | 0.62 | 0.040 | 0 |
| 13 | seed | 12 | 0.1 | 5e-05 | 128 | 4 | 761 | early_stop | 3.746 | 7.7/39.6 | 38.4/121.4 | 5.02 | 0.083 | 0 |
| 17 | grid | 14 | 0.2 | 0.0001 | 128 | 4 | 1000 | ok | 3.749 | 43.9/121.6 | 16.0/63.0 | 0.36 | 0.068 | 0 |
| 20 | grid | 12 | 0.2 | 3e-05 | 128 | 4 | 1000 | ok | 3.777 | 11.6/35.5 | 25.9/168.3 | 2.22 | 0.039 | 0 |
| 21 | grid | 14 | 0.2 | 3e-05 | 128 | 4 | 921 | early_stop | 3.854 | 26.7/47.9 | 34.7/86.4 | 1.30 | 0.042 | 0 |
| 6 | core | 14 | 0.05 | 5e-05 | 128 | 4 | 1000 | ok | 3.937 | 12.2/55.1 | 36.8/148.6 | 3.00 | 0.098 | 0 |
| 18 | gnn_H192 | 12 | 0.1 | 5e-05 | 192 | 4 | 761 | early_stop | 4.241 | 48.4/167.9 | 30.5/119.6 | 0.63 | 0.049 | 0 |
| 8 | grid | 14 | 0.1 | 0.0001 | 128 | 4 | 681 | early_stop | 4.298 | 34.0/314.0 | 35.9/104.4 | 1.05 | 0.063 | 0 |
| 7 | grid | 12 | 0.1 | 0.0001 | 128 | 4 | 921 | early_stop | 4.523 | 42.4/107.5 | 46.6/254.3 | 1.10 | 0.077 | 0 |
| 19 | gnn_rounds6 | 14 | 0.1 | 5e-05 | 128 | 6 | 801 | early_stop | 4.599 | 38.7/94.8 | 86.9/193.5 | 2.25 | 0.082 | 0 |
| 9 | grid | 12 | 0.1 | 3e-05 | 128 | 4 | 761 | early_stop | 5.415 | 12.3/55.5 | 320.4/1600.1 | 26.07 | 0.215 | 0 |
| 12 | grid | 14 | 0.3 | 5e-05 | 128 | 4 | 481 | early_stop | 5.679 | 54.3/270.3 | 207.9/1064.7 | 3.83 | 0.090 | 0 |

Trials logged: **22** (finite scores: 22).

## Winner

Search champion was trial **0** (S_comb=3.247 on val 16). Final BEST is trial **0** (`core`, L=12, leakpen=0.1, lr=5e-05, H=128, rounds=4, seed=0), chosen on the val-40 reval (same trial).

| split | n | B | Med | B90 | S | mean leak |
|---|---:|---:|---:|---:|---:|---:|
| val | 6 | 40.0 | 12.01 | 29.71 | 2.938 | 0.0901 |
| val | 9 | 40.0 | 16.98 | 85.26 | 3.639 | 0.0353 |
| test | 6 | 100.0 | 15.12 | 36.26 | 3.153 | 0.0868 |
| test | 9 | 100.0 | 16.79 | 120.84 | 3.807 | 0.0383 |

### Versus box reval40 (14.5 / 19.5)

**Val-40 full win: trial 0**, L=12, leakpen=0.1, lr=5e-5, learned α0. Medians **12.01 / 16.98** versus the box bar **14.5 / 19.5**. Ratio n9/n6 = 1.41, inside [0.75, 2.0]. This is the saved BEST. It was chosen on val-40 S_comb among candidates that did not regress n9; trial 16 has a slightly lower val-40 S (3.366 vs 3.394) but n9=21.3, which misses 19.5, so it was not promoted.

The same weights still clear the bar on larger val slices (not used to pick the trial; checked after the farm):

| val formulas | n6 Med | n9 Med | ratio | under 14.5 / 19.5 |
|---:|---:|---:|---:|---|
| 40 | 12.01 | 16.98 | 1.41 | yes |
| 80 | 11.50 | 18.64 | 1.62 | yes |
| 120 | 12.59 | 18.66 | 1.48 | yes |
| 200 (all val) | 13.66 | 18.07 | 1.32 | yes |

Held-out test (100 formulas, not used to select): n6 Med 15.12, n9 Med 16.79, ratio 1.11. n9 stays under 19.5. n6 is 15.12, which is above 14.5. The n6 edge is real on val and does not repeat on this test split.

Same-budget L=10 control (trial 15, val-40): n6=10.56, n9=27.75, ratio=2.63. That control beats the box on n6 and loses on n9. Trial 0's extra depth is what brings n9 from 27.8 down to 17.0, with n6 still under 14.5 on val. Our L=10 control is not a reproduction of the box checkpoint; the win claim is against the published 14.5 / 19.5 numbers.

Trial 16 (L=12, leakpen=0.2, lr=1e-4) is the best deep run by val-40 S alone (n6=12.52, n9=21.32) and is not a full win, because n9 misses 19.5.

### Other revals (val 40 / test)

| id | tag | L | lp | lr | H | r | val n6 | val n9 | val ratio | val S | test n6 | test n9 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | core | 12 | 0.1 | 5e-05 | 128 | 4 | 12.0 | 17.0 | 1.41 | 3.394 | 15.1 | 16.8 |
| 16 | grid | 12 | 0.2 | 0.0001 | 128 | 4 | 12.5 | 21.3 | 1.70 | 3.366 | 15.5 | 19.4 |
| 4 | core | 14 | 0.2 | 5e-05 | 128 | 4 | 10.0 | 28.9 | 2.89 | 3.538 | 13.6 | 22.6 |
| 15 | l10_control | 10 | 0.1 | 5e-05 | 128 | 4 | 10.6 | 27.8 | 2.63 | 3.563 | 15.0 | 22.4 |

## Forks

- `gnn_H192` trial 18: L=12 lp=0.1 lr=5e-05 H=192 rounds=4 → search S=4.241 n6=48.4 n9=30.5 ratio=0.63 keep=0 (ratio_only).
- `gnn_rounds6` trial 19: L=14 lp=0.1 lr=5e-05 H=128 rounds=6 → search S=4.599 n6=38.7 n9=86.9 ratio=2.25 keep=0 (s_comb).

A stronger H or extra rounds is a thin fork on the same ansatz (learned α0, exact cost phase). It is not an OCT20 calibrated SNAP–D–SNAP mixer; that stack is a different layer primitive and was not ported.

## Files

- `trials.csv` — every trial, including early-stops.
- `BEST/` — final weights (`weights_n6.pt`, `weights_n9.pt`), `best.json`, `arch.json`. Mid-saved on each search promote, then refreshed if the val-40 reval picks a different trial.
- `BEST_reval.json` — val-40 and test metrics.
- `BEST_search/` — present only if the reval winner differs from the search champion.
