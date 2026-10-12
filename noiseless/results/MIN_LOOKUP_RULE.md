# A systematic rule for the minimum classical lookups (R1 recipe, Gray, fixed_low)

Goal: the fewest distinct energy-table lookups per n that still give success >= 0.9 and mean p(GS) >= 0.9.
Brute-force screens (MIN_LOOKUP.md) turn one knob at a time. This file explains *why* each knob can go down, and sets each knob from measured statistics.

Lookups per trial = SA part + top-K readouts per explore run + Hamming-1 fix-ups (n+1 strings around each of the polt lowest pool strings), counted as distinct strings.
Upper bound: L(n) <= P_SA + E(n) * (K + polt * (n + 1)), where E(n) = explore runs (1 at n=8-12, 2 at n=14, 4 at n=16-18).
Today about 85 lookups per explore run come from top-48 plus fix-ups, and SA adds 50-162. At n=18 the total is about 500.

## 1. SA budget: measured offline, and above 64 it does not help at n >= 12

`noiseless/analysis/min_lookup_rule.py` resamples `sa_sample` (classical only) on every instance with 10 seeds each and scores the fixed_low layout by the fraction of its n/2 "surest" bits that match the GS (0.5 = chance). Full table: `min_lookup_sa_quality.txt`.

| n | quality at 64 | at the current sab | at 400 | SA best = GS at the current sab |
|---|---|---|---|---|
| 8 | 0.69 | 0.69 (sab 32 -> 64 probes) | n/a | 0.29 |
| 10 | 0.65 | 0.65 (sab 50 -> 64) | 0.93 | 0.03 |
| 12 | 0.62 | 0.62 (sab 72) | 0.77 | 0.00 |
| 14 | 0.62 | 0.62 (64) | 0.69 | 0.00 |
| 16 | 0.60 | 0.58 (128) | 0.70 | 0.01 |
| 18 | 0.61 | 0.59 (162) | 0.62 | 0.02 |

From 64 up to 200 the quality is flat at 0.58-0.62, only a little above chance, for n >= 12. The SA layout that works today is therefore mostly an arbitrary layout plus one low centre. The 98-162 extra SA lookups buy nothing that can be measured.
**SA rule:** use the fewest probes that only set the energy scale and a starting centre. The floor in `sa_sample` was hard-coded at 64. It is now set by the env var `QM_SA_PROBES` (default 64, so behaviour is unchanged). The validation runs try 16 probes at n >= 12 and fall back to 64.
To get a real gain from SA you would need a budget of order 2^n / 3 at n=10 and far more for larger n, so it does not scale.

## 2. Top-K readouts: K(n) from the GS rank in the explore readouts

New diagnostic (algorithm unchanged): each explore run now logs `run_gs_diag` = {rank_gs, rank_h1, pool_erank_h1, p_gs_run}. rank_h1 is the rank, in readout probability order, of the first string within Hamming 1 of the GS. pool_erank_h1 is its energy rank in the raw pool.
Per trial, K_need = 1 + the minimum over runs of rank_h1.
**K rule:** K(n) = the smallest K whose coverage, the fraction of trials with K_need <= K, is within 0.02 of the coverage at K = 48. Equivalently, K(n) is about the 95th percentile of K_need among reachable trials. Fit script: `noiseless/analysis/min_lookup_rule_fit.py`, output `relayout_runs/kbit_rule/rule_fit.json`.
Measured so far without ranks (`run_guess_ground`, top-48 within H1 of the GS): 1.00 / 0.92 / 0.97 per run at n=8/10/12, 0.66 at n=14, about 0.5 at n=18 and 0.28 at n=20. A single run does not cover the GS at n >= 14, so E carries the coverage at large n, and K should be set where coverage per run saturates, not pushed toward 48.

## 3. Fix-ups: polt from pool_erank_h1

polt_need = 1 + the energy rank, among the rawk-48 raw pool, of the H1-neighbour of the GS at the covering run. This is an upper bound for smaller K. **polt rule:** the smallest polt in {1, 2, 3} whose coverage is within 0.02 of the coverage at polt = 3. Every fix-up centre costs n + 1 lookups (19 at n=18), so polt 3 -> 1 alone saves up to 2 * E * (n + 1), which is 152 at n=18.

## 4. Adaptive stopping: measured offline, rejected

When the explore stage stops once the guess energy has not changed for m consecutive runs (old records with E >= 3, rawk 48):

| n | guess = GS, full E | m=1: guess = GS, share of runs kept | m=2: guess = GS, share of runs kept |
|---|---|---|---|
| 14 | 0.90 | 0.73, 0.70 | 0.87, 0.94 |
| 16 | 0.82 | 0.38, 0.27 | 0.51, 0.42 |
| 18 | 0.93 | 0.79, 0.56 | 0.85, 0.82 |
| 20 | 0.85 | 0.50, 0.56 | 0.80, 0.84 |

Accuracy drops by 0.05 to 0.3 for 6-44% fewer explore runs, so this rule cannot hold the 0.9 bar. The guess often stays the same for a while and then gets better later. A stop based on probability mass is not usable either, because the readout mass near the GS per run is small (see p_gs_run once logged). Keep a fixed E.

## Combined procedure

1. SA: QM_SA_PROBES = 16 at n >= 12 (fallback 64). At n <= 10 keep 64 probes.
2. Run E(n) explore runs. Look up the top K(n) readouts, where K(n) is the 95%-coverage rank from the diag pass.
3. Fix up the polt(n) lowest pool strings.
Predicted lookups <= P + E * (K + polt * (n + 1)). For example, with K = 24 and polt = 1 at n=18 this gives 16 + 4 * (24 + 19) = 188, against about 500 now. At n=14 it gives 16 + 2 * (24 + 15) = 94, against about 300.

| n | current R1 lookups (approx.) | rule bound (K=24, polt=1 placeholder; fit replaces it) |
|---|---|---|
| 8 | 126 | 64 + 1*(24+9) = 97 |
| 10 | 165 | 64 + 1*(24+11) = 99 |
| 12 | 191 | 16 + 1*(24+13) = 53 |
| 14 | 311 | 16 + 2*(24+15) = 94 |
| 16 | ~430 | 16 + 4*(24+17) = 180 |
| 18 | 504 | 16 + 4*(24+19) = 188 |
| 20 | 590-640 (E=5-6) | 16 + 6*(24+21) = 286 (prediction only, base not settled) |

The current values are averages of saved records. The bound column is a prediction until the queue below finishes. The fitted K(n) and polt(n) are appended automatically.

## Status: validated vs predicted

- Measured offline (done): SA layout quality vs budget (section 1) and the rejection of adaptive stopping (section 4).
- Predicted, not yet validated: K(n), polt(n), SA probes of 16, and the lookup numbers above.
- Queued: `logs/3cav/q_3cav_rule.sh` (detached, 8 workers, gated on q_3cav_minlookup.done, own outdir `relayout_runs/kbit_rule`, stop with `logs/3cav/q_3cav_rule.STOP`). It runs:
  - A. A 10-trial diag pass at the base settings for n=18, 16, 14, 12, 10, 8, logging GS ranks.
  - B. The fit of K(n) and polt(n).
  - C. A 10-trial screen at the rule setting. If it fails: K x2 (max 48) with 64 probes.
  - D. A 50-trial confirm of each passing setting.
  - Rows are appended below. The script commits and pushes at the end.
- Comparison with MIN_LOOKUP.md: the brute-force descent never tries SA below 64, K = 12 together with polt = 1, or K values that are not 24 or 12. The rule screens fill those gaps. Compare the "cheapest passing" lines there with the confirm rows here once both are in.

## Validation runs (auto-appended by logs/3cav/q_3cav_rule.sh, started 2026-10-12 23:14 CST)

Fit (from diag pass + earlier rawk-48 records with ranks):
```
8 {'trials': 10, 'cov48': 1.0, 'K': 9, 'covK': 1.0, 'polt': 1, 'cov_polt3': 1.0, 'cov_poltP': 1.0, 'K_quantiles': {0.5: 4.5, 0.9: 7.199999999999999, 0.95: 8.099999999999998}, 'per_run_rank_h1_median': 3.5}
10 {'trials': 20, 'cov48': 1.0, 'K': 37, 'covK': 1.0, 'polt': 1, 'cov_polt3': 1.0, 'cov_poltP': 1.0, 'K_quantiles': {0.5: 9.5, 0.9: 36.1, 0.95: 37.0}, 'per_run_rank_h1_median': 8.5}
12 {'trials': 20, 'cov48': 1.0, 'K': 23, 'covK': 1.0, 'polt': 1, 'cov_polt3': 1.0, 'cov_poltP': 1.0, 'K_quantiles': {0.5: 7.5, 0.9: 15.300000000000004, 0.95: 18.250000000000004}, 'per_run_rank_h1_median': 6.5}
14 {'trials': 30, 'cov48': 1.0, 'K': 2, 'covK': 1.0, 'polt': 1, 'cov_polt3': 1.0, 'cov_poltP': 1.0, 'K_quantiles': {0.5: 1.0, 0.9: 2.0, 0.95: 2.0}, 'per_run_rank_h1_median': 0.0}
16 {'trials': 30, 'cov48': 1.0, 'K': 5, 'covK': 1.0, 'polt': 1, 'cov_polt3': 1.0, 'cov_poltP': 1.0, 'K_quantiles': {0.5: 1.0, 0.9: 4.0, 0.95: 4.549999999999997}, 'per_run_rank_h1_median': 4.0}
18 {'trials': 10, 'cov48': 0.9, 'K': 10, 'covK': 0.9, 'polt': 1, 'cov_polt3': 0.9, 'cov_poltP': 0.9, 'K_quantiles': {0.5: 4.0, 0.9: 19.799999999999965, 0.95: 63.89999999999989}, 'per_run_rank_h1_median': 46.0}
```

| run | sab | SA probes | rawk K | polt | trials | success | mean p(GS) | total lookups | SA lookups | result |
|---|---|---|---|---|---|---|---|---|---|---|
| n18_rule_sab16_k10_p1 | 16 | 16 | 10 | 1 | 10 | 0.90 | 0.838 | 99.1 | 16.0 | fail |
| n18_rule2_sab64_k20_p1 | 64 | 64 | 20 | 1 | 10 | 1.00 | 0.854 | 187.5 | 64.0 | fail |
| n16_rule_sab16_k5_p1 | 16 | 16 | 5 | 1 | 10 | 0.90 | 0.819 | 64.0 | 16.0 | fail |
| n16_rule2_sab64_k10_p1 | 64 | 64 | 10 | 1 | 10 | 1.00 | 0.914 | 135.9 | 64.0 | PASS |
| n16_confirm_sab64_k10_p1 | 64 | 64 | 10 | 1 | 50 | 1.00 | 0.909 | 129.4 | 64.0 | PASS |
| n14_rule_sab16_k2_p1 | 16 | 16 | 2 | 1 | 10 | 1.00 | 0.891 | 36.5 | 16.0 | fail |
| n14_rule2_sab64_k4_p1 | 64 | 64 | 4 | 1 | 10 | 1.00 | 0.891 | 88.3 | 64.0 | fail |
| n12_rule_sab16_k23_p1 | 16 | 16 | 23 | 1 | 10 | 1.00 | 0.986 | 66.9 | 16.0 | PASS |
| n12_confirm_sab16_k23_p1 | 16 | 16 | 23 | 1 | 50 | 0.92 | 0.909 | 67.2 | 16.0 | PASS |
| n10_rule_sab50_k37_p1 | 50 | 64 | 37 | 1 | 10 | 1.00 | 0.991 | 128.3 | 62.6 | PASS |
| n10_confirm_sab50_k37_p1 | 50 | 64 | 37 | 1 | 50 | 1.00 | 0.990 | 129.2 | 62.2 | PASS |
| n8_rule_sab32_k9_p1 | 32 | 64 | 9 | 1 | 10 | 1.00 | 0.990 | 74.2 | 56.7 | PASS |
| n8_confirm_sab32_k9_p1 | 32 | 64 | 9 | 1 | 50 | 0.98 | 0.967 | 73.5 | 56.8 | PASS |
